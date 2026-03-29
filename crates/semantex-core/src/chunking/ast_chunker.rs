use crate::chunking::Chunker;
use crate::chunking::call_graph;
use crate::chunking::structured_meta::{
    ImplRelation, StructuredChunkMeta, TypeRef, TypeRefContext,
};
use crate::chunking::text_chunker::TextChunker;
use crate::file::detector::FileType;
use crate::types::{AstNodeKind, Chunk, ChunkType};
use anyhow::{Result, anyhow};
use std::path::Path;
use tree_sitter::{Language, Node, Parser};

/// Approximate characters per token
const CHARS_PER_TOKEN: usize = 4;

pub struct AstChunker {
    chunk_size: usize,
    chunk_overlap: usize,
    fallback: TextChunker,
}

impl AstChunker {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size,
            chunk_overlap,
            fallback: TextChunker::new(chunk_size, chunk_overlap),
        }
    }
}

impl Chunker for AstChunker {
    #[allow(clippy::too_many_lines)]
    fn chunk(&self, path: &Path, content: &str) -> Result<Vec<Chunk>> {
        let file_type = FileType::detect(path);

        let Some(lang_fn) = get_language(path, file_type) else {
            return self.fallback.chunk(path, content);
        };

        let Some(definition_kinds) = definition_node_kinds(file_type) else {
            return self.fallback.chunk(path, content);
        };

        let mut parser = Parser::new();
        parser
            .set_language(&lang_fn)
            .map_err(|e| anyhow!("Failed to set language: {e}"))?;

        let tree = parser
            .parse(content, None)
            .ok_or_else(|| anyhow!("tree-sitter parse failed for {}", path.display()))?;

        let source = content.as_bytes();

        // Extract file-level imports (for attaching to chunks)
        let language_name_str = file_type.language_name();
        let file_imports = crate::chunking::import_resolver::extract_imports(
            &tree.root_node(),
            source,
            language_name_str,
        );
        // Keep up to 8 most relevant imports per chunk
        let truncated_imports: Vec<String> = file_imports.into_iter().take(8).collect();

        let mut ast_spans: Vec<AstSpan> = Vec::new();
        collect_definitions(tree.root_node(), source, definition_kinds, &mut ast_spans);

        // Strip type_refs from generated code (Dart codegen) — too noisy for cross-file resolution
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if file_name.ends_with(".freezed.dart")
            || file_name.ends_with(".g.dart")
            || file_name.ends_with(".pb.dart")
        {
            for span in &mut ast_spans {
                span.meta.type_refs.clear();
            }
        }

        // Sort by start byte to process gaps in order
        ast_spans.sort_by_key(|s| s.start_byte);

        // Remove overlapping spans (keep the first / outermost one)
        let mut deduped: Vec<AstSpan> = Vec::new();
        for span in ast_spans {
            if let Some(last) = deduped.last()
                && span.start_byte < last.end_byte
            {
                continue; // nested inside previous span, skip
            }
            deduped.push(span);
        }

        // Attach file-level imports to each span's metadata
        for span in &mut deduped {
            span.meta.resolved_imports.clone_from(&truncated_imports);
        }

        let language_name = file_type.language_name().to_string();
        let max_node_chars = self.chunk_size * CHARS_PER_TOKEN * 4;
        let mut chunks: Vec<Chunk> = Vec::new();
        // Track which chunk indices carry structured_meta (for call graph post-pass)
        let mut meta_indices: Vec<(usize, String, StructuredChunkMeta)> = Vec::new();
        let mut window_index = 0u32;
        let mut cursor = 0usize; // byte offset tracking position in content

        for span in &deduped {
            // Emit gap chunk(s) for content before this AST node
            if span.start_byte > cursor {
                let gap = &content[cursor..span.start_byte];
                if !gap.trim().is_empty() {
                    let gap_start_line = byte_offset_to_line(content, cursor);
                    let gap_end_line =
                        byte_offset_to_line(content, span.start_byte.saturating_sub(1));
                    chunks.push(Chunk {
                        id: 0,
                        file_path: path.to_path_buf(),
                        start_line: gap_start_line,
                        end_line: gap_end_line.max(gap_start_line),
                        content: gap.to_string(),
                        chunk_type: ChunkType::TextWindow { window_index },
                    });
                    window_index += 1;
                }
            }

            let node_text_str = &content[span.start_byte..span.end_byte];
            // tree-sitter rows are 0-based, Chunk lines are 1-based
            let start_line = span.start_row + 1;
            let end_line = span.end_row + 1;

            if node_text_str.len() > max_node_chars {
                // Split oversized AST nodes with sliding window.
                // Attach metadata only to the first sub-chunk.
                let sub_chunks = split_large_node(
                    path,
                    node_text_str,
                    start_line,
                    &span.name,
                    &span.kind,
                    &language_name,
                    self.chunk_size,
                    self.chunk_overlap,
                    &span.meta,
                );
                chunks.extend(sub_chunks);
            } else {
                let chunk_idx = chunks.len();
                chunks.push(Chunk {
                    id: 0,
                    file_path: path.to_path_buf(),
                    start_line: start_line as u32,
                    end_line: end_line as u32,
                    content: node_text_str.to_string(),
                    chunk_type: ChunkType::AstNode {
                        name: span.name.clone(),
                        kind: span.kind.clone(),
                        language: language_name.clone(),
                        structured_meta: None, // filled after call graph post-pass
                    },
                });
                meta_indices.push((chunk_idx, span.name.clone(), span.meta.clone()));
            }

            cursor = span.end_byte;
        }

        // Emit trailing gap
        if cursor < content.len() {
            let gap = &content[cursor..];
            if !gap.trim().is_empty() {
                let gap_start_line = byte_offset_to_line(content, cursor);
                let gap_end_line = byte_offset_to_line(content, content.len().saturating_sub(1));
                chunks.push(Chunk {
                    id: 0,
                    file_path: path.to_path_buf(),
                    start_line: gap_start_line,
                    end_line: gap_end_line.max(gap_start_line),
                    content: gap.to_string(),
                    chunk_type: ChunkType::TextWindow { window_index },
                });
            }
        }

        // If we found no AST nodes at all, fall back to text chunking
        if chunks.is_empty() {
            return self.fallback.chunk(path, content);
        }

        // Call graph post-pass: build bidirectional caller/callee relationships
        if !meta_indices.is_empty() {
            let mut chunks_with_meta: Vec<(String, StructuredChunkMeta)> = meta_indices
                .iter()
                .map(|(_, name, meta)| (name.clone(), meta.clone()))
                .collect();

            call_graph::build_call_graph(&mut chunks_with_meta);

            // Generate NL summaries (after called_by is populated)
            for (_, meta) in &mut chunks_with_meta {
                meta.generate_nl_summary();
            }

            // Store metadata back into chunks
            for (i, (chunk_idx, _, _)) in meta_indices.iter().enumerate() {
                if let ChunkType::AstNode {
                    ref mut structured_meta,
                    ..
                } = chunks[*chunk_idx].chunk_type
                {
                    *structured_meta = Some(Box::new(chunks_with_meta[i].1.clone()));
                }
            }
        }

        Ok(chunks)
    }
}

/// A span extracted from the AST, with pre-extracted structured metadata.
struct AstSpan {
    start_byte: usize,
    end_byte: usize,
    start_row: usize,
    end_row: usize,
    name: String,
    kind: AstNodeKind,
    meta: StructuredChunkMeta,
}

// ---------------------------------------------------------------------------
// Tree-sitter helpers
// ---------------------------------------------------------------------------

/// Get UTF-8 text for a tree-sitter node.
fn ts_node_text<'a>(node: Node<'a>, source: &'a [u8]) -> &'a str {
    node.utf8_text(source).unwrap_or("")
}

/// Find the first direct child of `node` whose `kind()` equals `kind`.
fn find_child_by_type<'a>(node: Node<'a>, kind: &str) -> Option<Node<'a>> {
    let mut cursor = node.walk();
    node.children(&mut cursor)
        .find(|child| child.kind() == kind)
}

// ---------------------------------------------------------------------------
// 5-layer metadata extraction
// ---------------------------------------------------------------------------

/// Extract structured metadata (layers 1-4) from a tree-sitter AST node.
///
/// Layer 1 (AST): name, signature, params, return type, docstring
/// Layer 2 (Call Graph): outgoing calls (called_by filled in post-pass)
/// Layer 3 (Control Flow): complexity, branches, loops, error handling
/// Layer 4 (Data Flow): local variables, state mutations
fn extract_structured_meta(node: Node, source: &[u8]) -> StructuredChunkMeta {
    let mut meta = StructuredChunkMeta::default();

    // Layer 1: AST identity
    if let Some(name_node) = node.child_by_field_name("name") {
        meta.name = Some(ts_node_text(name_node, source).to_string());
    } else if node.kind() == "export_statement" {
        // export function foo() / export class Foo — name is in nested declaration
        let mut inner_cursor = node.walk();
        for child in node.children(&mut inner_cursor) {
            if let Some(inner_name) = child.child_by_field_name("name") {
                meta.name = Some(ts_node_text(inner_name, source).to_string());
                break;
            }
        }
    }

    // Signature: text up to first '{' or ':'
    let full_text = ts_node_text(node, source);
    if let Some(brace_pos) = full_text.find('{').or_else(|| full_text.find(':')) {
        let sig = full_text[..brace_pos].trim();
        if !sig.is_empty() {
            meta.signature = Some(sig.to_string());
        }
    }

    // Parameters
    let params_node = find_child_by_type(node, "formal_parameters")
        .or_else(|| find_child_by_type(node, "parameters"))
        .or_else(|| find_child_by_type(node, "parameter_list"));
    if let Some(pn) = params_node {
        let mut cursor = pn.walk();
        for child in pn.children(&mut cursor) {
            if child.kind().contains("parameter") || child.kind() == "identifier" {
                let param_text = ts_node_text(child, source).to_string();
                if !param_text.is_empty() && param_text.len() < 100 {
                    meta.params.push(param_text);
                }
            }
        }
    }

    // Return type
    if let Some(ret_node) = node
        .child_by_field_name("return_type")
        .or_else(|| find_child_by_type(node, "type_annotation"))
    {
        meta.return_type = Some(ts_node_text(ret_node, source).to_string());
    }

    // Layer 2: Outgoing calls
    walk_for_calls(node, source, &mut meta.calls);

    // Layer 1 (cont): Preceding docstring/comment
    #[allow(clippy::collapsible_if)]
    if let Some(prev) = node.prev_sibling() {
        if prev.kind() == "comment" || prev.kind().contains("doc") {
            let doc_text = ts_node_text(prev, source).to_string();
            if !doc_text.is_empty() {
                meta.docstring = Some(doc_text);
            }
        }
    }

    // Layer 3: Control flow
    extract_control_flow(node, source, &mut meta);

    // Layer 4: Data flow
    extract_data_flow(node, source, &mut meta);

    // Layer 5 enhanced: Type references
    meta.type_refs = extract_type_refs(node, source);

    // Layer 5 enhanced: Implementation relationships
    meta.implements = extract_implementations(node, source);

    // Layer 1 enhanced: Structured docstring tags
    if let Some(ref docstring) = meta.docstring {
        let lang = infer_language_from_node(node);
        meta.doc_tags = crate::chunking::doc_parser::parse_doc_tags(docstring, lang);
    }

    meta
}

/// Recursively collect function/method call targets from a tree-sitter subtree.
fn walk_for_calls(node: Node, source: &[u8], calls: &mut Vec<String>) {
    let mut cursor = node.walk();
    #[allow(clippy::collapsible_if)]
    for child in node.children(&mut cursor) {
        if child.kind() == "call_expression"
            || child.kind() == "method_invocation"
            || child.kind() == "call"
        {
            if let Some(func_node) = child
                .child_by_field_name("function")
                .or_else(|| child.child_by_field_name("name"))
                .or_else(|| child.child(0))
            {
                let call_text = ts_node_text(func_node, source).to_string();
                if !call_text.is_empty() && call_text.len() < 100 {
                    calls.push(call_text);
                }
            }
        }
        walk_for_calls(child, source, calls);
    }
}

/// Extract control flow information (Layer 3).
fn extract_control_flow(node: Node, source: &[u8], meta: &mut StructuredChunkMeta) {
    let mut complexity: u32 = 1; // base complexity
    walk_for_control_flow(node, source, &mut complexity, meta);
    meta.complexity = complexity;
}

fn walk_for_control_flow(
    node: Node,
    source: &[u8],
    complexity: &mut u32,
    meta: &mut StructuredChunkMeta,
) {
    match node.kind() {
        "if_statement"
        | "if_expression"
        | "conditional_expression"
        | "match_expression"
        | "switch_statement"
        | "case_clause" => {
            *complexity += 1;
            meta.has_branches = true;
        }
        "for_statement" | "for_expression" | "while_statement" | "loop_expression"
        | "for_in_statement" | "for_of_statement" => {
            *complexity += 1;
            meta.has_loops = true;
        }
        "try_statement" | "catch_clause" | "rescue" => {
            meta.has_error_handling = true;
        }
        "match_arm" => {
            let arm_text = ts_node_text(node, source);
            if arm_text.contains("Err(") || arm_text.contains("None") {
                meta.has_error_handling = true;
            }
        }
        _ => {}
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        walk_for_control_flow(child, source, complexity, meta);
    }
}

/// Extract data flow information (Layer 4).
fn extract_data_flow(node: Node, source: &[u8], meta: &mut StructuredChunkMeta) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "let_declaration"
            | "variable_declaration"
            | "variable_declarator"
            | "assignment_statement"
            | "let_statement" => {
                if let Some(name_node) = child
                    .child_by_field_name("name")
                    .or_else(|| child.child_by_field_name("pattern"))
                    .or_else(|| child.child(0))
                {
                    let var_name = ts_node_text(name_node, source).to_string();
                    if !var_name.is_empty() && var_name.len() < 50 {
                        meta.local_vars.push(var_name);
                    }
                }
            }
            "assignment_expression" => {
                if let Some(left) = child.child_by_field_name("left").or_else(|| child.child(0)) {
                    let lhs = ts_node_text(left, source);
                    if lhs.contains("self.") || lhs.contains("this.") || lhs.contains("->") {
                        meta.state_mutations.push(lhs.to_string());
                    }
                }
            }
            _ => {}
        }
        extract_data_flow(child, source, meta);
    }
}

// ---------------------------------------------------------------------------
// Layer 5 enhanced: Type reference extraction
// ---------------------------------------------------------------------------

/// Primitive type names to exclude from type references.
const PRIMITIVE_TYPES: &[&str] = &[
    "str",
    "string",
    "String",
    "int",
    "i8",
    "i16",
    "i32",
    "i64",
    "i128",
    "u8",
    "u16",
    "u32",
    "u64",
    "u128",
    "f32",
    "f64",
    "float",
    "double",
    "bool",
    "boolean",
    "void",
    "None",
    "null",
    "undefined",
    "number",
    "usize",
    "isize",
    "char",
    "byte",
    "self",
    "Self",
];

/// Check if a type name is a primitive that should be filtered out.
fn is_primitive_type(name: &str) -> bool {
    PRIMITIVE_TYPES.contains(&name)
}

/// Generic container types with no cross-file resolution value.
const GENERIC_CONTAINER_TYPES: &[&str] = &[
    "List",
    "Map",
    "Vec",
    "Array",
    "Set",
    "Option",
    "Result",
    "Optional",
    "Promise",
    "Future",
    "Stream",
    "Observable",
    "HashMap",
    "HashSet",
    "BTreeMap",
    "BTreeSet",
    "Dictionary",
];

/// Check if a type name looks user-defined (starts with uppercase or contains `::` / `.`).
fn is_user_type(name: &str) -> bool {
    if name.is_empty() || is_primitive_type(name) {
        return false;
    }
    if GENERIC_CONTAINER_TYPES.contains(&name) {
        return false;
    }
    // Must start with uppercase letter, or contain a path separator
    name.starts_with(|c: char| c.is_uppercase()) || name.contains("::") || name.contains('.')
}

/// Extract type references from a tree-sitter AST node.
///
/// Looks for user-defined types in parameter annotations, return types, and field types.
fn extract_type_refs(node: Node, source: &[u8]) -> Vec<TypeRef> {
    let mut refs = Vec::new();

    // Extract from parameter type annotations
    let params_node = find_child_by_type(node, "formal_parameters")
        .or_else(|| find_child_by_type(node, "parameters"))
        .or_else(|| find_child_by_type(node, "parameter_list"));
    if let Some(pn) = params_node {
        collect_type_identifiers(pn, source, TypeRefContext::Param, &mut refs);
    }

    // Extract from return type
    if let Some(ret_node) = node
        .child_by_field_name("return_type")
        .or_else(|| find_child_by_type(node, "type_annotation"))
    {
        collect_type_identifiers(ret_node, source, TypeRefContext::Return, &mut refs);
    }

    // Extract from struct/class field types
    let node_kind = node.kind();
    if node_kind == "struct_item"
        || node_kind == "struct_specifier"
        || node_kind == "class_declaration"
        || node_kind == "class_definition"
        || node_kind == "class_specifier"
        || node_kind == "struct_declaration"
    {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            let ck = child.kind();
            if ck == "field_declaration"
                || ck == "field_definition"
                || ck == "field_declaration_list"
                || ck == "class_body"
            {
                collect_type_identifiers(child, source, TypeRefContext::Field, &mut refs);
            }
        }
    }

    refs
}

/// Walk a subtree and collect type identifier nodes as `TypeRef`s.
fn collect_type_identifiers(
    node: Node,
    source: &[u8],
    context: TypeRefContext,
    refs: &mut Vec<TypeRef>,
) {
    let kind = node.kind();

    if kind == "type_identifier" || kind == "scoped_type_identifier" {
        let name = ts_node_text(node, source);
        if is_user_type(name) && name.len() < 100 {
            refs.push(TypeRef {
                type_name: name.to_string(),
                context,
            });
        }
        return; // Don't recurse into children of type identifiers
    }

    if kind == "generic_type" {
        // Extract the base type from generic_type (e.g. `Vec<T>` -> look for type_identifier child)
        if let Some(base) = node.child(0) {
            let base_kind = base.kind();
            if base_kind == "type_identifier" || base_kind == "scoped_type_identifier" {
                let name = ts_node_text(base, source);
                if is_user_type(name) && name.len() < 100 {
                    refs.push(TypeRef {
                        type_name: name.to_string(),
                        context,
                    });
                }
            }
        }
        // Also check type arguments for user types
        if let Some(type_args) = find_child_by_type(node, "type_arguments") {
            collect_type_identifiers(type_args, source, context, refs);
        }
        return;
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_type_identifiers(child, source, context, refs);
    }
}

// ---------------------------------------------------------------------------
// Layer 5 enhanced: Trait/impl extraction
// ---------------------------------------------------------------------------

/// Extract implementation relationships from a tree-sitter AST node.
///
/// Handles Rust `impl Trait for Type`, TS/Java `class X implements Y`,
/// and Python class inheritance.
fn extract_implementations(node: Node, source: &[u8]) -> Vec<ImplRelation> {
    let mut relations = Vec::new();
    let node_kind = node.kind();

    match node_kind {
        // Rust: `impl Trait for Type { ... }`
        "impl_item" => {
            let trait_node = node.child_by_field_name("trait");
            let type_node = node.child_by_field_name("type");
            if let (Some(tn), Some(ty)) = (trait_node, type_node) {
                let trait_name = ts_node_text(tn, source).to_string();
                let implementor = ts_node_text(ty, source).to_string();
                if !trait_name.is_empty() && !implementor.is_empty() {
                    relations.push(ImplRelation {
                        implementor,
                        trait_name,
                    });
                }
            }
        }
        // Java/TS/Dart/C# class declarations with implements/interfaces clause
        "class_declaration" | "class_definition" => {
            let implementor_name = node
                .child_by_field_name("name")
                .map(|n| ts_node_text(n, source).to_string())
                .unwrap_or_default();

            if implementor_name.is_empty() {
                return relations;
            }

            // Look for implements/interfaces clause
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                let ck = child.kind();
                if ck == "implements_clause"
                    || ck == "interfaces"
                    || ck == "super_interfaces"
                    || ck == "implements"
                {
                    let mut inner_cursor = child.walk();
                    for iface in child.children(&mut inner_cursor) {
                        if iface.kind() == "type_identifier"
                            || iface.kind() == "scoped_type_identifier"
                        {
                            let trait_name = ts_node_text(iface, source).to_string();
                            if !trait_name.is_empty() {
                                relations.push(ImplRelation {
                                    implementor: implementor_name.clone(),
                                    trait_name,
                                });
                            }
                        }
                    }
                }

                // Python: class Foo(Base1, Base2) — argument_list holds base classes
                if ck == "argument_list" {
                    let mut inner_cursor = child.walk();
                    for arg in child.children(&mut inner_cursor) {
                        if arg.kind() == "identifier" || arg.kind() == "attribute" {
                            let base_name = ts_node_text(arg, source).to_string();
                            if !base_name.is_empty() && base_name != "object" {
                                relations.push(ImplRelation {
                                    implementor: implementor_name.clone(),
                                    trait_name: base_name,
                                });
                            }
                        }
                    }
                }

                // extends clause (TS/Java)
                if ck == "extends_clause" || ck == "superclass" {
                    let mut inner_cursor = child.walk();
                    for base in child.children(&mut inner_cursor) {
                        if base.kind() == "type_identifier"
                            || base.kind() == "identifier"
                            || base.kind() == "scoped_type_identifier"
                        {
                            let base_name = ts_node_text(base, source).to_string();
                            if !base_name.is_empty() {
                                relations.push(ImplRelation {
                                    implementor: implementor_name.clone(),
                                    trait_name: base_name,
                                });
                            }
                        }
                    }
                }
            }
        }
        _ => {}
    }

    relations
}

// ---------------------------------------------------------------------------
// Language inference helper
// ---------------------------------------------------------------------------

/// Infer a language name from a tree-sitter node by walking up to the root
/// and checking the grammar name. Returns a best-effort language string.
fn infer_language_from_node(node: Node) -> &'static str {
    // Walk to root and check tree language via node kind patterns
    let kind = node.kind();
    if kind == "function_item"
        || kind == "impl_item"
        || kind == "struct_item"
        || kind == "enum_item"
        || kind == "trait_item"
        || kind == "mod_item"
    {
        return "rust";
    }
    if kind == "function_definition" || kind == "class_definition" {
        // Could be Python — check for `def` or `class` keywords
        // Python function_definition and class_definition are distinct from other languages
        // that use the same names (e.g., C). We use a heuristic: presence of `parameters` child.
        if find_child_by_type(node, "parameters").is_some() {
            return "python";
        }
    }
    if kind.contains("method_declaration") || kind.contains("interface_declaration") {
        return "java";
    }
    // Default fallback — generic tag parsing still works
    ""
}

// ---------------------------------------------------------------------------
// Language / grammar support
// ---------------------------------------------------------------------------

/// Get the tree-sitter language for a file, based on its type and extension
fn get_language(path: &Path, file_type: FileType) -> Option<Language> {
    match file_type {
        FileType::Rust => Some(tree_sitter_rust::LANGUAGE.into()),
        FileType::Python => Some(tree_sitter_python::LANGUAGE.into()),
        FileType::JavaScript => Some(tree_sitter_javascript::LANGUAGE.into()),
        FileType::TypeScript => {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if ext == "tsx" {
                Some(tree_sitter_typescript::LANGUAGE_TSX.into())
            } else {
                Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into())
            }
        }
        FileType::Go => Some(tree_sitter_go::LANGUAGE.into()),
        FileType::Java => Some(tree_sitter_java::LANGUAGE.into()),
        FileType::C => Some(tree_sitter_c::LANGUAGE.into()),
        FileType::Cpp => Some(tree_sitter_cpp::LANGUAGE.into()),
        FileType::Ruby => Some(tree_sitter_ruby::LANGUAGE.into()),
        FileType::Dart => Some(tree_sitter_dart_orchard::LANGUAGE.into()),
        FileType::CSharp => Some(tree_sitter_c_sharp::LANGUAGE.into()),
        FileType::Scala => Some(tree_sitter_scala::LANGUAGE.into()),
        FileType::Php => Some(tree_sitter_php::LANGUAGE_PHP.into()),
        FileType::Lua => Some(tree_sitter_lua::LANGUAGE.into()),
        FileType::Haskell => Some(tree_sitter_haskell::LANGUAGE.into()),
        FileType::OCaml => {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if ext == "mli" {
                Some(tree_sitter_ocaml::LANGUAGE_OCAML_INTERFACE.into())
            } else {
                Some(tree_sitter_ocaml::LANGUAGE_OCAML.into())
            }
        }
        FileType::Zig => Some(tree_sitter_zig::LANGUAGE.into()),
        FileType::R => Some(tree_sitter_r::LANGUAGE.into()),
        FileType::Html => Some(tree_sitter_html::LANGUAGE.into()),
        FileType::Swift => Some(tree_sitter_swift::LANGUAGE.into()),
        FileType::Elixir => Some(tree_sitter_elixir::LANGUAGE.into()),
        FileType::Svelte => Some(tree_sitter_svelte_next::LANGUAGE.into()),
        FileType::Kotlin => Some(tree_sitter_kotlin_ng::LANGUAGE.into()),
        _ => None,
    }
}

/// Get the set of AST node kind strings that represent definitions for a language
fn definition_node_kinds(file_type: FileType) -> Option<&'static [&'static str]> {
    match file_type {
        FileType::Rust => Some(&[
            "function_item",
            "impl_item",
            "struct_item",
            "enum_item",
            "mod_item",
            "trait_item",
        ]),
        FileType::Python => Some(&["function_definition", "class_definition"]),
        FileType::JavaScript | FileType::TypeScript => Some(&[
            "function_declaration",
            "class_declaration",
            "method_definition",
            "lexical_declaration",
            "export_statement",
        ]),
        FileType::Go => Some(&[
            "function_declaration",
            "method_declaration",
            "type_declaration",
        ]),
        FileType::Java => Some(&[
            "class_declaration",
            "method_declaration",
            "interface_declaration",
            "enum_declaration",
        ]),
        FileType::C => Some(&["function_definition", "struct_specifier"]),
        FileType::Cpp => Some(&["function_definition", "class_specifier", "struct_specifier"]),
        FileType::Ruby => Some(&["method", "class", "module"]),
        FileType::Dart => Some(&[
            "class_definition",
            "enum_declaration",
            "mixin_declaration",
            "extension_declaration",
            "extension_type_declaration",
            "function_signature",
            "getter_signature",
            "setter_signature",
            "constructor_signature",
            "type_alias",
        ]),
        FileType::CSharp => Some(&[
            "class_declaration",
            "method_declaration",
            "interface_declaration",
            "enum_declaration",
            "struct_declaration",
            "namespace_declaration",
        ]),
        FileType::Scala => Some(&[
            "class_definition",
            "object_definition",
            "trait_definition",
            "function_definition",
            "val_definition",
        ]),
        FileType::Php => Some(&[
            "class_declaration",
            "method_declaration",
            "function_definition",
            "interface_declaration",
            "trait_declaration",
        ]),
        FileType::Lua => Some(&["function_declaration", "local_function_declaration"]),
        FileType::Haskell => Some(&[
            "function",
            "type_alias",
            "newtype",
            "adt",
            "class",
            "instance",
        ]),
        FileType::OCaml => Some(&[
            "value_definition",
            "type_definition",
            "module_definition",
            "class_definition",
        ]),
        FileType::Zig => Some(&["function_declaration", "container_declaration"]),
        FileType::R => Some(&["function_definition", "left_assignment"]),
        FileType::Html | FileType::Svelte => Some(&["element", "script_element", "style_element"]),
        FileType::Swift => Some(&[
            "class_declaration",
            "function_declaration",
            "protocol_declaration",
            "struct_declaration",
            "enum_declaration",
        ]),
        FileType::Elixir => Some(&["call"]),
        FileType::Kotlin => Some(&[
            "class_declaration",
            "function_declaration",
            "object_declaration",
            "interface_declaration",
        ]),
        _ => None,
    }
}

/// For Dart: `function_signature`, `getter_signature`, and `setter_signature` nodes
/// only cover the declaration line — the body is a separate sibling `function_body`
/// node. Walk forward through siblings to find and include it so that the full
/// function implementation is captured in the chunk.
fn extend_to_function_body(node: &Node) -> Option<(usize, usize)> {
    let mut cursor = node.next_sibling();
    while let Some(sib) = cursor {
        match sib.kind() {
            "function_body" => {
                return Some((sib.end_byte(), sib.end_position().row));
            }
            // Skip anonymous/punctuation siblings (whitespace tokens, "native" keyword, etc.)
            k if !sib.is_named() || k == "native" => {
                cursor = sib.next_sibling();
            }
            // Hit a different named node — stop searching
            _ => break,
        }
    }
    None
}

/// Node kinds whose span must be extended to include a following `function_body`
/// sibling. Applies only to Dart, where signature and body are separate grammar
/// nodes with no named parent wrapper.
const DART_SIGNATURE_KINDS: &[&str] =
    &["function_signature", "getter_signature", "setter_signature"];

/// Recursively walk the AST and collect definition nodes with structured metadata.
fn collect_definitions(node: Node, source: &[u8], kinds: &[&str], out: &mut Vec<AstSpan>) {
    let node_kind = node.kind();

    if kinds.contains(&node_kind) {
        let name = extract_name(&node, source).unwrap_or_else(|| "<anonymous>".to_string());
        let kind = classify_node_kind(node_kind);
        let node_text = ts_node_text(node, source);
        let mut meta = extract_structured_meta(node, source);
        meta.kind = Some(kind.label().to_string());

        // Layer 6: Semantic role (after all other metadata is available)
        meta.semantic_role =
            crate::chunking::semantic_role::classify_semantic_role(&meta, node_text);

        // Dart: signature nodes don't include the function body — extend span.
        let (end_byte, end_row) = if DART_SIGNATURE_KINDS.contains(&node_kind) {
            extend_to_function_body(&node).unwrap_or((node.end_byte(), node.end_position().row))
        } else {
            (node.end_byte(), node.end_position().row)
        };

        out.push(AstSpan {
            start_byte: node.start_byte(),
            end_byte,
            start_row: node.start_position().row,
            end_row,
            name,
            kind,
            meta,
        });
        // Continue recursing — nested definitions (e.g. methods in classes,
        // declarations in export_statement) are collected separately and
        // deduplicated later by the overlap-removal pass.
    }

    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            collect_definitions(child, source, kinds, out);
        }
    }
}

/// Try to extract a name from a definition node
fn extract_name(node: &Node, source: &[u8]) -> Option<String> {
    // Try common field names first
    for field in &["name", "identifier"] {
        if let Some(child) = node.child_by_field_name(field) {
            let text = &source[child.start_byte()..child.end_byte()];
            return Some(String::from_utf8_lossy(text).to_string());
        }
    }

    // Walk immediate children looking for identifier-like nodes
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            match child.kind() {
                "identifier" | "name" | "property_identifier" | "type_identifier" => {
                    let text = &source[child.start_byte()..child.end_byte()];
                    return Some(String::from_utf8_lossy(text).to_string());
                }
                _ => {}
            }
        }
    }

    None
}

/// Map tree-sitter node kind strings to AstNodeKind
fn classify_node_kind(kind: &str) -> AstNodeKind {
    match kind {
        "function_item"
        | "function_definition"
        | "function_declaration"
        | "function_signature"
        | "local_function_declaration"
        | "function"
        | "getter_signature"
        | "setter_signature" => AstNodeKind::Function,
        "method_definition"
        | "method_declaration"
        | "method"
        | "method_signature"
        | "constructor_signature" => AstNodeKind::Method,
        "class_definition" | "class_declaration" | "class_specifier" | "class" => {
            AstNodeKind::Class
        }
        "struct_item" | "struct_specifier" | "struct_declaration" => AstNodeKind::Struct,
        "enum_item" | "enum_declaration" => AstNodeKind::Enum,
        "interface_declaration"
        | "trait_item"
        | "protocol_declaration"
        | "trait_definition"
        | "trait_declaration" => AstNodeKind::Interface,
        "mod_item"
        | "module"
        | "mixin_declaration"
        | "extension_declaration"
        | "namespace_declaration"
        | "module_definition"
        | "object_definition" => AstNodeKind::Module,
        "type_alias" => AstNodeKind::Other("type_alias".to_string()),
        "impl_item" => AstNodeKind::Other("impl".to_string()),
        "type_declaration" => AstNodeKind::Other("type".to_string()),
        other => AstNodeKind::Other(other.to_string()),
    }
}

/// Convert a byte offset in content to a 1-based line number
fn byte_offset_to_line(content: &str, byte_offset: usize) -> u32 {
    let capped = byte_offset.min(content.len());
    content[..capped].bytes().filter(|&b| b == b'\n').count() as u32 + 1
}

/// Split a large AST node into smaller chunks using a sliding window.
/// Attaches structured metadata to the first sub-chunk only.
#[allow(clippy::too_many_arguments)]
fn split_large_node(
    path: &Path,
    text: &str,
    base_line: usize,
    name: &str,
    kind: &AstNodeKind,
    language: &str,
    chunk_size: usize,
    chunk_overlap: usize,
    meta: &StructuredChunkMeta,
) -> Vec<Chunk> {
    let chunk_chars = chunk_size * CHARS_PER_TOKEN;
    let overlap_chars = chunk_overlap * CHARS_PER_TOKEN;
    let step = if chunk_chars > overlap_chars {
        chunk_chars - overlap_chars
    } else {
        chunk_chars
    };

    let mut chunks = Vec::new();
    let mut offset = 0usize;
    let mut part = 0u32;
    let bytes = text.as_bytes();

    while offset < text.len() {
        let end = text.floor_char_boundary((offset + chunk_chars).min(text.len()));

        // Try to break at a line boundary
        let split_at = if end < text.len() {
            let search_start = if end > overlap_chars {
                text.floor_char_boundary(end - overlap_chars)
            } else {
                offset
            };
            let mut best = end;
            for i in (search_start..end).rev() {
                if bytes[i] == b'\n' {
                    best = i + 1;
                    break;
                }
            }
            best
        } else {
            end
        };

        let chunk_text = &text[offset..split_at];
        if !chunk_text.trim().is_empty() {
            let lines_before = text[..offset].bytes().filter(|&b| b == b'\n').count();
            let lines_in = chunk_text.bytes().filter(|&b| b == b'\n').count();
            let start_line = (base_line + lines_before) as u32;
            let end_line = (base_line + lines_before + lines_in) as u32;

            let part_name = if part == 0 {
                name.to_string()
            } else {
                format!("{name}[part {part}]")
            };

            // First sub-chunk gets full metadata; continuations get lightweight identity copy
            let chunk_meta = Some(Box::new(if part == 0 {
                meta.clone()
            } else {
                StructuredChunkMeta {
                    name: meta.name.clone(),
                    kind: meta.kind.clone(),
                    semantic_role: meta.semantic_role,
                    ..Default::default()
                }
            }));

            chunks.push(Chunk {
                id: 0,
                file_path: path.to_path_buf(),
                start_line,
                end_line: end_line.max(start_line),
                content: chunk_text.to_string(),
                chunk_type: ChunkType::AstNode {
                    name: part_name,
                    kind: kind.clone(),
                    language: language.to_string(),
                    structured_meta: chunk_meta,
                },
            });
            part += 1;
        }

        let new_offset = text.floor_char_boundary(offset + step.max(1));
        offset = if split_at > new_offset {
            if split_at > overlap_chars {
                text.floor_char_boundary(split_at - overlap_chars)
            } else {
                split_at
            }
        } else {
            new_offset
        };
    }

    chunks
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_function_chunking() {
        let chunker = AstChunker::new(256, 64);
        let content = r#"
fn hello() {
    println!("hello");
}

fn world() {
    println!("world");
}
"#;
        let chunks = chunker.chunk(Path::new("test.rs"), content).unwrap();
        // Should find at least the two functions
        let ast_chunks: Vec<_> = chunks
            .iter()
            .filter(|c| matches!(c.chunk_type, ChunkType::AstNode { .. }))
            .collect();
        assert_eq!(ast_chunks.len(), 2);
    }

    #[test]
    fn test_python_class_chunking() {
        let chunker = AstChunker::new(256, 64);
        let content = r#"
class Greeter:
    def greet(self):
        print("hello")

def standalone():
    pass
"#;
        let chunks = chunker.chunk(Path::new("test.py"), content).unwrap();
        let ast_chunks: Vec<_> = chunks
            .iter()
            .filter(|c| matches!(c.chunk_type, ChunkType::AstNode { .. }))
            .collect();
        // class_definition contains the method, and standalone function
        assert!(ast_chunks.len() >= 2);
    }

    #[test]
    fn test_unsupported_extension_falls_back() {
        let chunker = AstChunker::new(256, 64);
        let content = "some yaml content\nkey: value\n";
        let chunks = chunker.chunk(Path::new("config.yaml"), content).unwrap();
        assert!(!chunks.is_empty());
        assert!(matches!(chunks[0].chunk_type, ChunkType::TextWindow { .. }));
    }

    #[test]
    fn test_ast_node_name_extraction() {
        let chunker = AstChunker::new(256, 64);
        let content = "fn my_function() { }\n";
        let chunks = chunker.chunk(Path::new("test.rs"), content).unwrap();
        let ast_chunk = chunks
            .iter()
            .find(|c| matches!(c.chunk_type, ChunkType::AstNode { .. }))
            .unwrap();
        match &ast_chunk.chunk_type {
            ChunkType::AstNode { name, .. } => assert_eq!(name, "my_function"),
            _ => panic!("Expected AstNode"),
        }
    }

    #[test]
    fn test_dart_ast_chunking() {
        let chunker = AstChunker::new(256, 64);
        let content = r"
import 'package:flutter/material.dart';

class MyWidget extends StatelessWidget {
  final String title;

  @override
  Widget build(BuildContext context) {
    return Container();
  }

  void _helper() {
    print('hello');
  }
}

enum Color { red, green, blue }

void topLevelFunction() {
  print('top level');
}
";
        let chunks = chunker.chunk(Path::new("lib/main.dart"), content).unwrap();
        let ast_chunks: Vec<_> = chunks
            .iter()
            .filter(|c| matches!(c.chunk_type, ChunkType::AstNode { .. }))
            .collect();
        // Should find: class_definition, enum_declaration, function_signature
        assert!(
            ast_chunks.len() >= 3,
            "Expected at least 3 AST chunks, got {}: {:?}",
            ast_chunks.len(),
            ast_chunks.iter().map(|c| &c.chunk_type).collect::<Vec<_>>()
        );
        // Verify it produces AstNode chunks (not just TextWindow)
        let has_class = ast_chunks.iter().any(|c| match &c.chunk_type {
            ChunkType::AstNode { kind, .. } => matches!(kind, AstNodeKind::Class),
            _ => false,
        });
        assert!(has_class, "Should find a class definition");
    }

    /// Verify that top-level Dart function bodies are included in the chunk text,
    /// not just the signature line. Regression test for the `function_signature`
    /// body-capture fix (extend_to_function_body).
    #[test]
    fn test_dart_function_body_captured() {
        let chunker = AstChunker::new(512, 64);
        let content = "void refreshToken(String userId) {\n  final token = fetchFromVault(userId);\n  return token;\n}\n\nString getSecret() {\n  return 'hunter2';\n}\n";
        let chunks = chunker.chunk(Path::new("lib/auth.dart"), content).unwrap();

        // Find the function chunk for refreshToken
        let fn_chunk = chunks.iter().find(|c| match &c.chunk_type {
            ChunkType::AstNode { name, kind, .. } => {
                name == "refreshToken" && matches!(kind, AstNodeKind::Function)
            }
            _ => false,
        });
        assert!(
            fn_chunk.is_some(),
            "Should find refreshToken as an AstNode chunk"
        );
        // The chunk content must include the body, not just the signature
        assert!(
            fn_chunk.unwrap().content.contains("fetchFromVault"),
            "refreshToken chunk must contain body content, got: {}",
            fn_chunk.unwrap().content
        );

        // Same check for getSecret
        let secret_chunk = chunks.iter().find(|c| match &c.chunk_type {
            ChunkType::AstNode { name, .. } => name == "getSecret",
            _ => false,
        });
        assert!(
            secret_chunk.is_some(),
            "Should find getSecret as an AstNode chunk"
        );
        assert!(
            secret_chunk.unwrap().content.contains("hunter2"),
            "getSecret chunk must contain body, got: {}",
            secret_chunk.unwrap().content
        );
    }

    /// Verify setter_signature is captured with its body.
    #[test]
    fn test_dart_setter_captured() {
        let chunker = AstChunker::new(512, 64);
        let content = "class Cache {\n  int _size = 0;\n  set size(int val) {\n    _size = val.clamp(0, 1024);\n  }\n}\n";
        let chunks = chunker.chunk(Path::new("lib/cache.dart"), content).unwrap();
        // The class chunk should contain the setter body
        let class_chunk = chunks.iter().find(|c| match &c.chunk_type {
            ChunkType::AstNode { kind, .. } => matches!(kind, AstNodeKind::Class),
            _ => false,
        });
        assert!(class_chunk.is_some(), "Should find Cache class");
        assert!(
            class_chunk.unwrap().content.contains("clamp"),
            "Class chunk should contain setter body"
        );
    }

    #[test]
    fn test_gap_content_becomes_text_window() {
        let chunker = AstChunker::new(256, 64);
        let content = r#"// This is a file header comment
// with some info

use std::io;

fn hello() {
    println!("hello");
}
"#;
        let chunks = chunker.chunk(Path::new("test.rs"), content).unwrap();
        let has_text_window = chunks
            .iter()
            .any(|c| matches!(c.chunk_type, ChunkType::TextWindow { .. }));
        let has_ast_node = chunks
            .iter()
            .any(|c| matches!(c.chunk_type, ChunkType::AstNode { .. }));
        assert!(has_text_window, "Should have TextWindow chunks for gaps");
        assert!(has_ast_node, "Should have AstNode chunks for functions");
    }

    #[test]
    fn test_structured_meta_extracted_for_rust_function() {
        let chunker = AstChunker::new(256, 64);
        let content = r"
/// Adds two numbers.
fn add(a: i32, b: i32) -> i32 {
    a + b
}
";
        let chunks = chunker.chunk(Path::new("test.rs"), content).unwrap();
        let ast_chunk = chunks
            .iter()
            .find(|c| matches!(c.chunk_type, ChunkType::AstNode { .. }))
            .unwrap();
        match &ast_chunk.chunk_type {
            ChunkType::AstNode {
                structured_meta: Some(meta),
                ..
            } => {
                assert_eq!(meta.name.as_deref(), Some("add"));
                assert!(meta.signature.is_some(), "Should have a signature");
                assert!(!meta.params.is_empty(), "Should have parameters");
                assert!(!meta.nl_summary.is_empty(), "Should have NL summary");
            }
            ChunkType::AstNode {
                structured_meta: None,
                ..
            } => panic!("Expected structured_meta to be Some"),
            _ => panic!("Expected AstNode"),
        }
    }

    #[test]
    fn test_structured_meta_call_graph() {
        let chunker = AstChunker::new(256, 64);
        let content = r#"
fn caller() {
    callee();
}

fn callee() {
    println!("done");
}
"#;
        let chunks = chunker.chunk(Path::new("test.rs"), content).unwrap();
        let callee_chunk = chunks.iter().find(|c| match &c.chunk_type {
            ChunkType::AstNode { name, .. } => name == "callee",
            _ => false,
        });
        if let Some(chunk) = callee_chunk {
            match &chunk.chunk_type {
                ChunkType::AstNode {
                    structured_meta: Some(meta),
                    ..
                } => {
                    assert!(
                        meta.called_by.contains(&"caller".to_string()),
                        "callee should be called_by caller, got: {:?}",
                        meta.called_by
                    );
                }
                _ => panic!("Expected AstNode with structured_meta"),
            }
        }
    }

    #[test]
    fn test_structured_meta_control_flow() {
        let chunker = AstChunker::new(256, 64);
        let content = r#"
fn complex(x: i32) -> i32 {
    if x > 0 {
        for i in 0..x {
            println!("{}", i);
        }
        x
    } else {
        0
    }
}
"#;
        let chunks = chunker.chunk(Path::new("test.rs"), content).unwrap();
        let ast_chunk = chunks
            .iter()
            .find(|c| matches!(c.chunk_type, ChunkType::AstNode { .. }))
            .unwrap();
        match &ast_chunk.chunk_type {
            ChunkType::AstNode {
                structured_meta: Some(meta),
                ..
            } => {
                assert!(meta.has_branches, "Should detect branches");
                assert!(meta.has_loops, "Should detect loops");
                assert!(meta.complexity >= 3, "Complexity should be >= 3");
            }
            _ => panic!("Expected AstNode with structured_meta"),
        }
    }
}
