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
        collect_definitions(
            tree.root_node(),
            source,
            definition_kinds,
            &mut ast_spans,
            file_type,
        );

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
    for child in node.children(&mut cursor) {
        let callee_text: Option<String> = if child.kind() == "call_expression"
            || child.kind() == "method_invocation"
            || child.kind() == "call"
        {
            // Regular function / method call: target is in the `function` or
            // `name` field, falling back to the first child (e.g. Elixir).
            child
                .child_by_field_name("function")
                .or_else(|| child.child_by_field_name("name"))
                .or_else(|| child.child(0))
                .map(|n| ts_node_text(n, source).to_string())
        } else if child.kind() == "new_expression" {
            // TypeScript / JavaScript: `new Foo(...)` or `new ns.Foo(...)`.
            // The `constructor` field holds the constructed type expression.
            // We record the raw text (e.g. "Foo" or "ns.Foo"); the call-graph
            // post-pass uses rsplit('.') to strip any namespace prefix when
            // resolving to a named chunk.
            child
                .child_by_field_name("constructor")
                .map(|n| ts_node_text(n, source).to_string())
        } else if child.kind() == "object_creation_expression" {
            // Java / C#: `new Foo(...)`.
            // The `type` field holds the constructed type name.
            child
                .child_by_field_name("type")
                .map(|n| ts_node_text(n, source).to_string())
        } else {
            None
        };

        if let Some(text) = callee_text
            && !text.is_empty()
            && text.len() < 100
        {
            calls.push(text);
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
        FileType::Vue => Some(tree_sitter_vue_next::LANGUAGE.into()),
        FileType::Kotlin => Some(tree_sitter_kotlin_ng::LANGUAGE.into()),
        FileType::Sql => Some(tree_sitter_sequel::LANGUAGE.into()),
        FileType::Bash => Some(tree_sitter_bash::LANGUAGE.into()),
        FileType::Groovy => Some(tree_sitter_groovy::LANGUAGE.into()),
        FileType::Terraform => Some(tree_sitter_hcl::LANGUAGE.into()),
        FileType::Protobuf => Some(tree_sitter_proto::LANGUAGE.into()),
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
            "record_declaration",                // C# 9+ records
            "file_scoped_namespace_declaration", // C# 10+ file-scoped namespaces
        ]),
        FileType::Scala => Some(&[
            "class_definition",
            "object_definition",
            "trait_definition",
            "function_definition",
            "val_definition",
            "given_definition",     // Scala 3 contextual instances
            "extension_definition", // Scala 3 extension methods
            "enum_definition",      // Scala 3 enums (distinct from Java/TS enum_declaration)
            "type_definition",      // type aliases
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
            // Top-level `external name : type = "c_func"` — OCaml grammar
            // emits node kind `external` at the structure_item level (not
            // `external_declaration`, which is only used inside type defs).
            "external",
            "external_declaration",
        ]),
        FileType::Zig => Some(&["function_declaration", "container_declaration"]),
        FileType::R => Some(&["function_definition", "left_assignment"]),
        FileType::Html | FileType::Svelte => Some(&["element", "script_element", "style_element"]),
        FileType::Vue => Some(&["script_element", "template_element", "style_element"]),
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
        // tree-sitter-sequel wraps every top-level statement in a `statement`
        // node; `collect_definitions` recurses through it regardless, so we
        // list the inner DDL node kinds directly. Only CREATE statements are
        // treated as definitions (mirrors other languages chunking
        // declarations, not usages) — ALTER/DROP/DML are left as gap text.
        FileType::Sql => Some(&[
            "create_table",
            "create_view",
            "create_materialized_view",
            "create_function",
            "create_trigger",
            "create_index",
            "create_type",
            "create_sequence",
            "create_schema",
            "create_role",
            "create_database",
            "create_extension",
        ]),
        // Both `function foo() {...}` and `foo() {...}` forms produce
        // function_definition in tree-sitter-bash.
        FileType::Bash => Some(&["function_definition"]),
        FileType::Groovy => Some(&[
            "function_definition",
            "method_declaration",
            "class_declaration",
        ]),
        // Every resource/variable/module/data/provider/output/locals/
        // terraform block is one unit; we deliberately don't recurse into
        // the block body (no nested node kind here is also in this list),
        // so attributes and nested blocks stay folded into the parent unit.
        FileType::Terraform => Some(&["block"]),
        // Each message/enum/service is one unit; fields/enum values/rpcs
        // stay folded inside (no nested node kind here is also in this list).
        FileType::Protobuf => Some(&["message", "enum", "service"]),
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
fn collect_definitions(
    node: Node,
    source: &[u8],
    kinds: &[&str],
    out: &mut Vec<AstSpan>,
    file_type: FileType,
) {
    let node_kind = node.kind();

    if kinds.contains(&node_kind) {
        let name =
            extract_name(&node, source, file_type).unwrap_or_else(|| "<anonymous>".to_string());
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
            collect_definitions(child, source, kinds, out, file_type);
        }
    }
}

/// HCL `block` nodes have no `name` field — the identifying header is the
/// block type followed by its labels. tree-sitter-hcl parses a block as
/// `identifier (string_lit | identifier)* block_start body block_end`, so the
/// name is the leading identifier plus every label token that precedes the
/// opening brace, e.g. `resource "aws_instance" "web"`, `variable "region"`,
/// `terraform` (no labels). String labels keep their quotes (the raw
/// `string_lit` text) so the name reads exactly as it appears in the source.
/// Adapted from `lightonai/colgrep`'s `get_hcl_unit_name`
/// (`colgrep/src/parser/ast.rs`, Apache-2.0).
fn extract_hcl_block_name(node: &Node, source: &[u8]) -> Option<String> {
    if node.kind() != "block" {
        return None;
    }
    let mut parts: Vec<String> = Vec::new();
    for i in 0..node.child_count() {
        let Some(child) = node.child(i as u32) else {
            continue;
        };
        match child.kind() {
            "identifier" | "string_lit" => {
                let text = &source[child.start_byte()..child.end_byte()];
                let trimmed = String::from_utf8_lossy(text).trim().to_string();
                if !trimmed.is_empty() {
                    parts.push(trimmed);
                }
            }
            "block_start" => break,
            _ => {}
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" "))
    }
}

/// Protobuf declarations carry their identifier in a dedicated child node
/// (`message_name`/`enum_name`/`service_name`), not a `name` field. Keeps the
/// declaration keyword in the name (`message Invoice`, `service Billing`) so
/// results read like the source. Adapted from `lightonai/colgrep`'s
/// `get_proto_unit_name` (Apache-2.0).
fn extract_proto_name(node: &Node, source: &[u8]) -> Option<String> {
    let keyword = node.kind();
    for i in 0..node.child_count() {
        let Some(child) = node.child(i as u32) else {
            continue;
        };
        if matches!(child.kind(), "message_name" | "enum_name" | "service_name") {
            let text = &source[child.start_byte()..child.end_byte()];
            let name = String::from_utf8_lossy(text).trim().to_string();
            if !name.is_empty() {
                return Some(format!("{keyword} {name}"));
            }
        }
    }
    None
}

/// GraphQL definitions carry a `name` child (or `fragment_name` for
/// fragments); prefixed with the definition keyword (`type User`,
/// `enum Role`). `schema { ... }` has no name and is named by its keyword
/// alone. Adapted from `lightonai/colgrep`'s `get_graphql_unit_name`
/// (Apache-2.0).
fn extract_graphql_name(node: &Node, source: &[u8]) -> Option<String> {
    if node.kind() == "operation_definition" {
        let mut op_type: Option<String> = None;
        let mut name: Option<String> = None;
        for i in 0..node.child_count() {
            let Some(child) = node.child(i as u32) else {
                continue;
            };
            if child.kind() == "operation_type" {
                let text = &source[child.start_byte()..child.end_byte()];
                op_type = Some(String::from_utf8_lossy(text).to_string());
            } else if child.kind() == "name" {
                let text = &source[child.start_byte()..child.end_byte()];
                let n = String::from_utf8_lossy(text).trim().to_string();
                if !n.is_empty() {
                    name = Some(n);
                }
            }
        }
        return op_type.map(|kw| match name {
            Some(n) => format!("{kw} {n}"),
            None => kw,
        });
    }
    let keyword = match node.kind() {
        "object_type_definition" => "type",
        "interface_type_definition" => "interface",
        "enum_type_definition" => "enum",
        "input_object_type_definition" => "input",
        "union_type_definition" => "union",
        "scalar_type_definition" => "scalar",
        "directive_definition" => "directive",
        "fragment_definition" => "fragment",
        "schema_definition" => return Some("schema".to_string()),
        _ => return None,
    };
    for i in 0..node.child_count() {
        let Some(child) = node.child(i as u32) else {
            continue;
        };
        if matches!(child.kind(), "name" | "fragment_name") {
            let text = &source[child.start_byte()..child.end_byte()];
            let name = String::from_utf8_lossy(text).trim().to_string();
            if !name.is_empty() {
                return Some(format!("{keyword} {name}"));
            }
        }
    }
    Some(keyword.to_string())
}

/// Starlark/Bazel target: a `call` whose argument list carries a
/// `name = "..."` string kwarg, e.g. `cc_library(name = "mylib", ...)` ->
/// `cc_library "mylib"`. Calls without such a kwarg (`glob(...)`,
/// `select(...)`, nested calls) return `None` — `collect_definitions` falls
/// back to `"<anonymous>"` for these rather than skipping them entirely
/// (see the design doc's accepted-tradeoff note: this matches the existing
/// `FileType::Elixir` behavior, which has the identical shape of problem
/// with its own `["call"]`-based definition kind and no extra filtering).
/// Adapted from `lightonai/colgrep`'s `get_starlark_unit_name` (Apache-2.0).
fn extract_starlark_call_name(node: &Node, source: &[u8]) -> Option<String> {
    let function = node.child_by_field_name("function")?;
    let rule = String::from_utf8_lossy(&source[function.start_byte()..function.end_byte()])
        .trim()
        .to_string();
    let args = node.child_by_field_name("arguments")?;
    for i in 0..args.child_count() {
        let Some(kwarg) = args.child(i as u32) else {
            continue;
        };
        if kwarg.kind() != "keyword_argument" {
            continue;
        }
        let Some(key) = kwarg.child_by_field_name("name") else {
            continue;
        };
        let key_text = &source[key.start_byte()..key.end_byte()];
        if key_text != b"name" {
            continue;
        }
        let Some(value) = kwarg.child_by_field_name("value") else {
            continue;
        };
        if value.kind() != "string" {
            continue;
        }
        let target =
            String::from_utf8_lossy(&source[value.start_byte()..value.end_byte()]).to_string();
        if rule.is_empty() || target.is_empty() {
            return None;
        }
        return Some(format!("{rule} {target}"));
    }
    None
}

/// CMake function/macro definitions keep their name as the first argument of
/// the opening command: `function(add_component name)` -> `add_component`.
/// Adapted from `lightonai/colgrep`'s `get_cmake_unit_name` (Apache-2.0).
fn extract_cmake_name(node: &Node, source: &[u8]) -> Option<String> {
    for i in 0..node.child_count() {
        let Some(command) = node.child(i as u32) else {
            continue;
        };
        if !matches!(command.kind(), "function_command" | "macro_command") {
            continue;
        }
        for j in 0..command.child_count() {
            let Some(args) = command.child(j as u32) else {
                continue;
            };
            if args.kind() != "argument_list" {
                continue;
            }
            for k in 0..args.child_count() {
                let Some(first) = args.child(k as u32) else {
                    continue;
                };
                if first.kind() != "argument" {
                    continue;
                }
                let text = &source[first.start_byte()..first.end_byte()];
                let name = String::from_utf8_lossy(text).trim().to_string();
                return if name.is_empty() { None } else { Some(name) };
            }
        }
    }
    None
}

/// INI `[section]` — keep the bracketed text as the name so it reads exactly
/// as in the source. Adapted from `lightonai/colgrep`'s inline `Language::Ini`
/// branch in `get_unit_name` (Apache-2.0).
fn extract_ini_section_name(node: &Node, source: &[u8]) -> Option<String> {
    for i in 0..node.child_count() {
        let Some(child) = node.child(i as u32) else {
            continue;
        };
        if child.kind() == "section_name" {
            let text = &source[child.start_byte()..child.end_byte()];
            let name = String::from_utf8_lossy(text).trim().to_string();
            return if name.is_empty() { None } else { Some(name) };
        }
    }
    None
}

/// PowerShell: `function_statement` carries a `function_name` child;
/// `class_statement` a `simple_name` child. Neither is exposed as a named
/// field. Adapted from `lightonai/colgrep`'s inline `Language::Powershell`
/// branch in `get_unit_name` (Apache-2.0).
fn extract_powershell_name(node: &Node, source: &[u8]) -> Option<String> {
    for i in 0..node.child_count() {
        let Some(child) = node.child(i as u32) else {
            continue;
        };
        if matches!(child.kind(), "function_name" | "simple_name") {
            let text = &source[child.start_byte()..child.end_byte()];
            let name = String::from_utf8_lossy(text).to_string();
            return if name.is_empty() { None } else { Some(name) };
        }
    }
    None
}

/// Try to extract a name from a definition node.
///
/// `file_type` selects bespoke extraction for 7 config-as-code languages
/// whose grammars expose no generic `name`/`identifier` field (Terraform,
/// Protobuf, GraphQL, Starlark's `call` nodes specifically, CMake, INI,
/// PowerShell — verified against `lightonai/colgrep`'s own per-language
/// name extraction). This dispatch MUST happen before the generic fallback
/// chain below: `FileType::Starlark`'s `function_definition` nodes and
/// `FileType::Elixir`'s `call` nodes both still want the generic path (or,
/// for Elixir, the existing canonical-kind fallback), and `node.kind() ==
/// "call"` alone can't distinguish Starlark from Elixir — only `file_type` can.
fn extract_name(node: &Node, source: &[u8], file_type: FileType) -> Option<String> {
    match file_type {
        FileType::Terraform => return extract_hcl_block_name(node, source),
        FileType::Protobuf => return extract_proto_name(node, source),
        FileType::GraphQl => return extract_graphql_name(node, source),
        FileType::Starlark if node.kind() == "call" => {
            return extract_starlark_call_name(node, source);
        }
        FileType::Cmake => return extract_cmake_name(node, source),
        FileType::Ini => return extract_ini_section_name(node, source),
        FileType::PowerShell => return extract_powershell_name(node, source),
        _ => {}
    }

    // Try common field names first
    for field in &["name", "identifier"] {
        if let Some(child) = node.child_by_field_name(field) {
            let text = &source[child.start_byte()..child.end_byte()];
            return Some(String::from_utf8_lossy(text).to_string());
        }
    }

    // First pass: canonical identifier-like child kinds shared across grammars.
    // These win whenever they exist on the node.
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

    // Second pass: language-specific identifier-like child kinds. Checked AFTER
    // the canonical kinds so that any grammar emitting both still picks the
    // canonical name first. Currently:
    //   - `value_name`: OCaml-specific. Emitted as a named child of `external`,
    //     `value_specification`, `value_path`, and `alias_pattern`. Of those,
    //     only `external` is in our definition_node_kinds for OCaml.
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32)
            && child.kind() == "value_name"
        {
            let text = &source[child.start_byte()..child.end_byte()];
            return Some(String::from_utf8_lossy(text).to_string());
        }
    }

    // Third pass: `object_reference` (tree-sitter-sequel/SQL-specific). Most
    // `create_*` statement nodes don't carry a direct `name` field or bare
    // `identifier` child — the target object name is one level down, inside
    // an `object_reference` child that itself has a `name` field (e.g.
    // `create_table` -> `object_reference` -> name: `identifier`). Statements
    // with more than one `object_reference` (e.g. `create_trigger`, which also
    // references the table and function it attaches to) emit the defined
    // object's reference first, so the first match wins.
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32)
            && child.kind() == "object_reference"
            && let Some(name_node) = child.child_by_field_name("name")
        {
            let text = &source[name_node.start_byte()..name_node.end_byte()];
            return Some(String::from_utf8_lossy(text).to_string());
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
        | "setter_signature"
        | "external"
        | "external_declaration"
        | "create_function"
        // CMake (function_def/macro_def) and PowerShell (function_statement).
        | "function_def"
        | "macro_def"
        | "function_statement" => AstNodeKind::Function,
        "method_definition"
        | "method_declaration"
        | "method"
        | "method_signature"
        | "constructor_signature" => AstNodeKind::Method,
        // PowerShell's class_statement joins the shared Class arm.
        "class_definition" | "class_declaration" | "class_specifier" | "class"
        | "class_statement" => AstNodeKind::Class,
        // `create_table` (SQL): a table's columns are conceptually fields, so
        // it maps to Struct like other languages' record types. Protobuf's
        // `message` joins for the same reason.
        "struct_item" | "struct_specifier" | "struct_declaration" | "record_declaration"
        | "create_table" | "message" => AstNodeKind::Struct,
        // Protobuf's bare `enum` and GraphQL's `enum_type_definition` join
        // the shared Enum arm.
        "enum_item" | "enum_declaration" | "enum_definition" | "enum" | "enum_type_definition" => {
            AstNodeKind::Enum
        }
        // GraphQL's interface_type_definition joins the shared Interface arm.
        "interface_declaration"
        | "trait_item"
        | "protocol_declaration"
        | "trait_definition"
        | "trait_declaration"
        | "interface_type_definition" => AstNodeKind::Interface,
        "mod_item"
        | "module"
        | "mixin_declaration"
        | "extension_declaration"
        | "namespace_declaration"
        | "file_scoped_namespace_declaration"
        | "module_definition"
        | "object_definition"
        | "create_schema" => AstNodeKind::Module,
        // `type_alias` (Dart, Haskell): always a true alias (`typedef Foo = Bar`).
        // `type_definition` (Scala 3, OCaml): umbrella for type-level declarations
        //   that may or may not be aliases — Scala 3 emits it for `type X = Y` and
        //   abstract type members; OCaml emits it for records (`type r = { x: int }`),
        //   sum types (`type t = A | B`), and aliases. Folding both into a single
        //   `type_alias` label mislabels OCaml records/variants and Scala 3 abstract
        //   types as aliases, so keep them distinct.
        "type_alias" => AstNodeKind::Other("type_alias".to_string()),
        "type_definition" => AstNodeKind::Other("type_definition".to_string()),
        "impl_item" => AstNodeKind::Other("impl".to_string()),
        // GraphQL's object_type_definition ("type Foo { ... }") joins the
        // shared "type" Other(_) arm.
        "type_declaration" | "create_type" | "object_type_definition" => {
            AstNodeKind::Other("type".to_string())
        }
        "given_definition" => AstNodeKind::Other("given".to_string()),
        "extension_definition" => AstNodeKind::Other("extension".to_string()),
        // SQL DDL (tree-sitter-sequel): views/materialized views/indexes/etc.
        // have no close analogue in the shared AstNodeKind set, so they stay
        // Other(_). (create_table/create_function/create_schema/create_type
        // are folded into the shared arms above.)
        "create_view" | "create_materialized_view" => AstNodeKind::Other("view".to_string()),
        "create_trigger" => AstNodeKind::Other("trigger".to_string()),
        "create_index" => AstNodeKind::Other("index".to_string()),
        "create_sequence" => AstNodeKind::Other("sequence".to_string()),
        "create_role" => AstNodeKind::Other("role".to_string()),
        "create_database" => AstNodeKind::Other("database".to_string()),
        "create_extension" => AstNodeKind::Other("extension_stmt".to_string()),
        // Protobuf's service and GraphQL's SDL/executable definitions besides
        // interface/enum/object have no clean AstNodeKind equivalent, so they
        // stay Other(_) with their own keyword, mirroring SQL's
        // view/index/trigger precedent above. (message/enum/
        // interface_type_definition/enum_type_definition/
        // object_type_definition/function_def/macro_def/function_statement/
        // class_statement are folded into the shared arms above instead.)
        "service" => AstNodeKind::Other("service".to_string()),
        "input_object_type_definition" => AstNodeKind::Other("input".to_string()),
        "union_type_definition" => AstNodeKind::Other("union".to_string()),
        "scalar_type_definition" => AstNodeKind::Other("scalar".to_string()),
        "schema_definition" => AstNodeKind::Other("schema".to_string()),
        "directive_definition" => AstNodeKind::Other("directive".to_string()),
        "operation_definition" => AstNodeKind::Other("operation".to_string()),
        "fragment_definition" => AstNodeKind::Other("fragment".to_string()),
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
    fn test_vue_sfc_chunks_script_template_style() {
        let chunker = AstChunker::new(256, 64);
        let content = r#"<template>
  <div class="hello">{{ msg }}</div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
const msg = ref('hello')
</script>

<style scoped>
.hello { color: red; }
</style>
"#;
        let chunks = chunker.chunk(Path::new("App.vue"), content).unwrap();
        let ast_chunks: Vec<_> = chunks
            .iter()
            .filter(|c| matches!(c.chunk_type, ChunkType::AstNode { .. }))
            .collect();
        // Expect at least one chunk per <script>, <template>, <style> block.
        assert!(
            ast_chunks.len() >= 3,
            "Expected at least 3 AST chunks (script/template/style), got {}: {:?}",
            ast_chunks.len(),
            ast_chunks.iter().map(|c| &c.chunk_type).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_ocaml_external_declaration() {
        let chunker = AstChunker::new(256, 64);
        let content = r#"
external sqrt : float -> float = "sqrt_C_func"

let double x = x * 2
"#;
        let chunks = chunker.chunk(Path::new("test.ml"), content).unwrap();
        let ast_chunks: Vec<_> = chunks
            .iter()
            .filter(|c| matches!(c.chunk_type, ChunkType::AstNode { .. }))
            .collect();
        let has_sqrt = ast_chunks.iter().any(|c| match &c.chunk_type {
            ChunkType::AstNode { name, kind, .. } => {
                name == "sqrt" && matches!(kind, AstNodeKind::Function)
            }
            _ => false,
        });
        assert!(
            has_sqrt,
            "Should find external sqrt as Function: {:?}",
            ast_chunks.iter().map(|c| &c.chunk_type).collect::<Vec<_>>()
        );
    }

    /// Regression test for the `value_name` precedence-ordering fix.
    ///
    /// `value_name` is an OCaml-specific identifier-like node kind. The fix
    /// reorders `extract_name` so that the canonical identifier kinds
    /// (`identifier`, `name`, `property_identifier`, `type_identifier`) are
    /// checked BEFORE `value_name`. That way, if a future grammar emits both
    /// kinds under the same definition node, the canonical kind wins.
    ///
    /// We exercise the invariant from two directions using real grammars:
    /// 1. Rust `fn my_function`: canonical "name" field is set on
    ///    `function_item`. The first pass returns "my_function" — `value_name`
    ///    is never consulted (it doesn't exist in the Rust grammar anyway).
    /// 2. OCaml `external sqrt`: `external` has no "name" field and no
    ///    canonical identifier-kind children, only a `value_name` child. The
    ///    second pass picks it up.
    ///
    /// Together, these confirm the two-pass precedence works as intended.
    #[test]
    fn test_extract_name_canonical_kinds_take_precedence_over_value_name() {
        let chunker = AstChunker::new(256, 64);

        // (1) Canonical kinds (Rust uses "name" field) — first pass wins.
        let rust_content = "fn my_function() { }\n";
        let rust_chunks = chunker.chunk(Path::new("t.rs"), rust_content).unwrap();
        let rust_name = rust_chunks
            .iter()
            .find_map(|c| match &c.chunk_type {
                ChunkType::AstNode { name, .. } => Some(name.clone()),
                _ => None,
            })
            .expect("expected an AstNode chunk for Rust");
        assert_eq!(
            rust_name, "my_function",
            "Rust function name should come from canonical 'name' field"
        );

        // (2) `value_name` fallback (OCaml `external`) — second pass picks it up.
        let ocaml_content = "external sqrt : float -> float = \"sqrt_C_func\"\n";
        let ocaml_chunks = chunker.chunk(Path::new("t.ml"), ocaml_content).unwrap();
        let ocaml_name = ocaml_chunks
            .iter()
            .find_map(|c| match &c.chunk_type {
                ChunkType::AstNode { name, .. } if name != "<anonymous>" => Some(name.clone()),
                _ => None,
            })
            .expect("expected a named AstNode chunk for OCaml external");
        assert_eq!(
            ocaml_name, "sqrt",
            "OCaml external should fall back to value_name in the second pass"
        );
    }

    #[test]
    fn test_scala3_given_extension_enum_type() {
        let chunker = AstChunker::new(256, 64);
        let content = r"
trait Show[A]:
  def show(a: A): String

given showInt: Show[Int] with
  def show(a: Int): String = a.toString

enum Color:
  case Red, Green, Blue

extension (s: String)
  def kebab: String = s.replace(' ', '-')

type StringMap = Map[String, String]
";
        let chunks = chunker.chunk(Path::new("test.scala"), content).unwrap();
        let ast_chunks: Vec<_> = chunks
            .iter()
            .filter(|c| matches!(c.chunk_type, ChunkType::AstNode { .. }))
            .collect();
        assert!(
            !ast_chunks.is_empty(),
            "Should produce AST chunks for Scala 3 file"
        );

        let has_enum_color = ast_chunks.iter().any(|c| match &c.chunk_type {
            ChunkType::AstNode { name, kind, .. } => {
                name == "Color" && matches!(kind, AstNodeKind::Enum)
            }
            _ => false,
        });
        assert!(
            has_enum_color,
            "Should find enum Color classified as Enum: {:?}",
            ast_chunks.iter().map(|c| &c.chunk_type).collect::<Vec<_>>()
        );

        // given_definition -> Other("given")
        let has_given = ast_chunks.iter().any(|c| match &c.chunk_type {
            ChunkType::AstNode { kind, .. } => {
                matches!(kind, AstNodeKind::Other(s) if s == "given")
            }
            _ => false,
        });
        assert!(
            has_given,
            "Should find given_definition classified as Other(given): {:?}",
            ast_chunks.iter().map(|c| &c.chunk_type).collect::<Vec<_>>()
        );

        // extension_definition -> Other("extension")
        let has_extension = ast_chunks.iter().any(|c| match &c.chunk_type {
            ChunkType::AstNode { kind, .. } => {
                matches!(kind, AstNodeKind::Other(s) if s == "extension")
            }
            _ => false,
        });
        assert!(
            has_extension,
            "Should find extension_definition classified as Other(extension)"
        );
    }

    /// Regression test for the `type_definition` vs `type_alias` split.
    ///
    /// Previously `classify_node_kind` folded both grammar kinds into a single
    /// `Other("type_alias")` label. That misrepresents:
    ///   - OCaml `type r = { x: int }` and `type t = A | B` (records and sum
    ///     types, both emitted as `type_definition`, not aliases).
    ///   - Scala 3 abstract type members (also `type_definition`).
    ///
    /// The fix keeps the two kinds distinct: `type_alias` -> `Other("type_alias")`
    /// (true aliases — Dart/Haskell), `type_definition` -> `Other("type_definition")`
    /// (umbrella term; covers Scala 3 + OCaml).
    ///
    /// tree-sitter-scala 0.26's grammar.js emits `type_definition` for any
    /// `type X = ...` (alias) AND for abstract type members. tree-sitter-ocaml
    /// 0.25 emits `type_definition` for records, variants, and aliases alike.
    #[test]
    fn test_type_definition_kept_distinct_from_type_alias() {
        let chunker = AstChunker::new(256, 64);

        // Scala 3 `type X = ...` — grammar emits `type_definition`, not `type_alias`.
        let scala_content = "type StringMap = Map[String, String]\n";
        let scala_chunks = chunker.chunk(Path::new("t.scala"), scala_content).unwrap();
        let scala_kind = scala_chunks.iter().find_map(|c| match &c.chunk_type {
            ChunkType::AstNode { name, kind, .. } if name == "StringMap" => Some(kind.clone()),
            _ => None,
        });
        assert!(
            matches!(scala_kind, Some(AstNodeKind::Other(ref s)) if s == "type_definition"),
            "Scala 3 `type X = ...` should chunk as Other(\"type_definition\"), got: {scala_kind:?}"
        );

        // OCaml record `type r = { x: int }` — grammar emits `type_definition`
        // (NOT `type_alias`); this is a record declaration, not an alias.
        let ocaml_content = "type point = { x : int; y : int }\n";
        let ocaml_chunks = chunker.chunk(Path::new("t.ml"), ocaml_content).unwrap();
        let ocaml_kind = ocaml_chunks.iter().find_map(|c| match &c.chunk_type {
            ChunkType::AstNode { kind, .. } => match kind {
                AstNodeKind::Other(s) if s == "type_definition" || s == "type_alias" => {
                    Some(kind.clone())
                }
                _ => None,
            },
            _ => None,
        });
        assert!(
            matches!(ocaml_kind, Some(AstNodeKind::Other(ref s)) if s == "type_definition"),
            "OCaml `type point = {{ ... }}` (a record) must NOT be labeled \
             type_alias; expected Other(\"type_definition\"), got: {ocaml_kind:?}"
        );
    }

    #[test]
    fn test_csharp_record_and_file_scoped_namespace() {
        let chunker = AstChunker::new(256, 64);
        let content = r"
namespace Foo.Bar;

public record Person(string Name, int Age);

public class Other {
    public void Method() { }
}
";
        let chunks = chunker.chunk(Path::new("test.cs"), content).unwrap();
        let ast_chunks: Vec<_> = chunks
            .iter()
            .filter(|c| matches!(c.chunk_type, ChunkType::AstNode { .. }))
            .collect();
        // Expect at least: namespace, record, class, method
        assert!(
            ast_chunks.len() >= 3,
            "Expected at least 3 AST chunks (namespace, record, class), got {}: {:?}",
            ast_chunks.len(),
            ast_chunks.iter().map(|c| &c.chunk_type).collect::<Vec<_>>()
        );

        let has_record = ast_chunks.iter().any(|c| match &c.chunk_type {
            ChunkType::AstNode { name, kind, .. } => {
                name == "Person" && matches!(kind, AstNodeKind::Struct)
            }
            _ => false,
        });
        assert!(has_record, "Should find record Person classified as Struct");

        let has_file_scoped_ns = ast_chunks.iter().any(|c| match &c.chunk_type {
            ChunkType::AstNode { kind, .. } => matches!(kind, AstNodeKind::Module),
            _ => false,
        });
        assert!(
            has_file_scoped_ns,
            "Should find file-scoped namespace as Module"
        );
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

    /// Regression test: constructor invocations (`new Foo()`) must produce
    /// call-graph edges so that callers of `new`-instantiated types are found
    /// by the structural route.
    ///
    /// Covers:
    ///   - `new_expression` node kind (TypeScript/JavaScript)
    ///   - `object_creation_expression` node kind (Java / C#) — same code
    ///     path, covered by the Java sub-test below.
    #[test]
    fn test_constructor_call_edges_new_expression() {
        let chunker = AstChunker::new(512, 64);

        // --- TypeScript: `new Foo()` inside a function ---
        let ts_content = r#"
class Foo {
  greet(): string { return "hi"; }
}

function bar(): Foo {
  return new Foo();
}
"#;
        let ts_chunks = chunker.chunk(Path::new("test.ts"), ts_content).unwrap();

        // The `Foo` class chunk should record that `bar` calls it.
        let foo_chunk = ts_chunks.iter().find(|c| match &c.chunk_type {
            ChunkType::AstNode { name, .. } => name == "Foo",
            _ => false,
        });
        assert!(foo_chunk.is_some(), "Should find a Foo class chunk in TS");
        if let Some(chunk) = foo_chunk {
            match &chunk.chunk_type {
                ChunkType::AstNode {
                    structured_meta: Some(meta),
                    ..
                } => {
                    assert!(
                        meta.called_by.contains(&"bar".to_string()),
                        "Foo should be called_by bar (via new Foo()), got called_by: {:?}",
                        meta.called_by
                    );
                }
                _ => panic!("Expected AstNode with structured_meta for Foo"),
            }
        }

        // --- TypeScript: qualified `new ns.Cache()` — rightmost ident matches ---
        let ts_qualified = r"
class Cache {
  get(k: string): string { return k; }
}

function init(): Cache {
  return new Cache();
}
";
        let ts_q_chunks = chunker.chunk(Path::new("cache.ts"), ts_qualified).unwrap();
        let cache_chunk = ts_q_chunks.iter().find(|c| match &c.chunk_type {
            ChunkType::AstNode { name, .. } => name == "Cache",
            _ => false,
        });
        assert!(cache_chunk.is_some(), "Should find Cache class chunk");
        if let Some(chunk) = cache_chunk {
            match &chunk.chunk_type {
                ChunkType::AstNode {
                    structured_meta: Some(meta),
                    ..
                } => {
                    assert!(
                        meta.called_by.contains(&"init".to_string()),
                        "Cache should be called_by init (via new Cache()), got: {:?}",
                        meta.called_by
                    );
                }
                _ => panic!("Expected AstNode with structured_meta for Cache"),
            }
        }

        // --- Java: `new` uses `object_creation_expression` ---
        // The Java chunker keeps the outermost span (class_declaration wins
        // over nested method_declaration), so `App` is the chunk that contains
        // `new Widget()` and gets recorded as the caller.
        let java_content = r"
class Widget {
    void draw() {}
}

class App {
    Widget makeWidget() {
        return new Widget();
    }
}
";
        let java_chunks = chunker.chunk(Path::new("App.java"), java_content).unwrap();
        let widget_chunk = java_chunks.iter().find(|c| match &c.chunk_type {
            ChunkType::AstNode { name, .. } => name == "Widget",
            _ => false,
        });
        assert!(
            widget_chunk.is_some(),
            "Should find Widget class chunk in Java"
        );
        if let Some(chunk) = widget_chunk {
            match &chunk.chunk_type {
                ChunkType::AstNode {
                    structured_meta: Some(meta),
                    ..
                } => {
                    assert!(
                        !meta.called_by.is_empty(),
                        "Widget should have at least one caller via new Widget(), got called_by: {:?}",
                        meta.called_by
                    );
                    // The caller is App (outermost Java class chunk; method_declaration
                    // is nested and deduped out). Verify the edge exists, not absent.
                    assert!(
                        meta.called_by.contains(&"App".to_string()),
                        "Widget should be called_by App (contains new Widget()), got: {:?}",
                        meta.called_by
                    );
                }
                _ => panic!("Expected AstNode with structured_meta for Widget"),
            }
        }
    }

    /// Regression test for the SQL grammar wiring: `tree-sitter-sequel`
    /// (see Cargo.toml) provides the LANGUAGE compatible with our
    /// `tree-sitter 0.26.9`. This asserts SQL files produce real `AstNode`
    /// chunks (CREATE TABLE, CREATE FUNCTION, ...), not a silent TextWindow
    /// fallback.
    #[test]
    fn test_sql_ast_chunking_not_text_fallback() {
        let chunker = AstChunker::new(256, 64);
        let content = r"
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE
);

CREATE FUNCTION add_numbers(a INTEGER, b INTEGER) RETURNS INTEGER AS $$
BEGIN
    RETURN a + b;
END;
$$ LANGUAGE plpgsql;

CREATE VIEW active_users AS
SELECT * FROM users WHERE active = true;

CREATE INDEX idx_users_email ON users (email);

SELECT * FROM users;
";
        let chunks = chunker.chunk(Path::new("schema.sql"), content).unwrap();
        let ast_chunks: Vec<_> = chunks
            .iter()
            .filter(|c| matches!(c.chunk_type, ChunkType::AstNode { .. }))
            .collect();
        assert!(
            !ast_chunks.is_empty(),
            "SQL must produce AstNode chunks, not fall back to text chunking"
        );

        let table_chunk = ast_chunks.iter().find(|c| match &c.chunk_type {
            ChunkType::AstNode { name, kind, .. } => {
                name == "users" && matches!(kind, AstNodeKind::Struct)
            }
            _ => false,
        });
        assert!(
            table_chunk.is_some(),
            "Should find CREATE TABLE users as a Struct-kind AstNode: {:?}",
            ast_chunks.iter().map(|c| &c.chunk_type).collect::<Vec<_>>()
        );
        assert!(
            table_chunk.unwrap().content.contains("PRIMARY KEY"),
            "Table chunk should contain its column definitions"
        );

        let function_chunk = ast_chunks.iter().find(|c| match &c.chunk_type {
            ChunkType::AstNode { name, kind, .. } => {
                name == "add_numbers" && matches!(kind, AstNodeKind::Function)
            }
            _ => false,
        });
        assert!(
            function_chunk.is_some(),
            "Should find CREATE FUNCTION add_numbers as a Function-kind AstNode"
        );
        assert!(
            function_chunk.unwrap().content.contains("RETURN a + b"),
            "Function chunk should contain the function body"
        );

        let view_chunk = ast_chunks.iter().any(|c| match &c.chunk_type {
            ChunkType::AstNode { name, kind, .. } => {
                name == "active_users" && matches!(kind, AstNodeKind::Other(s) if s == "view")
            }
            _ => false,
        });
        assert!(
            view_chunk,
            "Should find CREATE VIEW active_users as Other(\"view\")"
        );

        let index_chunk = ast_chunks.iter().any(|c| match &c.chunk_type {
            ChunkType::AstNode { name, kind, .. } => {
                name == "idx_users_email" && matches!(kind, AstNodeKind::Other(s) if s == "index")
            }
            _ => false,
        });
        assert!(
            index_chunk,
            "Should find CREATE INDEX idx_users_email as Other(\"index\")"
        );

        // The bare trailing `SELECT * FROM users;` is not a definition — it
        // should surface as gap text, not a spurious AstNode.
        let has_text_window = chunks
            .iter()
            .any(|c| matches!(c.chunk_type, ChunkType::TextWindow { .. }));
        assert!(
            has_text_window,
            "The standalone SELECT statement should remain a TextWindow gap chunk"
        );
    }

    /// Additional SQL DDL coverage beyond the core CREATE TABLE/FUNCTION case:
    /// CREATE SCHEMA (-> Module) and CREATE TRIGGER, whose `object_reference`
    /// name-extraction must pick the trigger's own name, not the table or
    /// function it references.
    #[test]
    fn test_sql_schema_and_trigger_naming() {
        let chunker = AstChunker::new(256, 64);
        let content = r"
CREATE SCHEMA analytics;

CREATE TRIGGER audit_trigger AFTER INSERT ON users FOR EACH ROW EXECUTE FUNCTION log_change();
";
        let chunks = chunker.chunk(Path::new("schema.sql"), content).unwrap();
        let ast_chunks: Vec<_> = chunks
            .iter()
            .filter(|c| matches!(c.chunk_type, ChunkType::AstNode { .. }))
            .collect();

        let has_schema = ast_chunks.iter().any(|c| match &c.chunk_type {
            ChunkType::AstNode { name, kind, .. } => {
                name == "analytics" && matches!(kind, AstNodeKind::Module)
            }
            _ => false,
        });
        assert!(
            has_schema,
            "Should find CREATE SCHEMA analytics as a Module-kind AstNode: {:?}",
            ast_chunks.iter().map(|c| &c.chunk_type).collect::<Vec<_>>()
        );

        let has_trigger = ast_chunks.iter().any(|c| match &c.chunk_type {
            ChunkType::AstNode { name, kind, .. } => {
                name == "audit_trigger" && matches!(kind, AstNodeKind::Other(s) if s == "trigger")
            }
            _ => false,
        });
        assert!(
            has_trigger,
            "Trigger name extraction must pick the trigger's own name (first \
             object_reference), not the referenced table/function: {:?}",
            ast_chunks.iter().map(|c| &c.chunk_type).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_extract_name_hcl_block_joins_type_and_labels() {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_hcl::LANGUAGE.into())
            .unwrap();
        let source = br#"resource "aws_instance" "web" {
  ami = "ami-123456"
}"#;
        let tree = parser.parse(source, None).unwrap();

        // HCL structure is config_file -> body -> block
        let root = tree.root_node();
        let body = root.children(&mut root.walk()).next().unwrap();
        let block = body
            .children(&mut body.walk())
            .find(|n| n.kind() == "block")
            .expect("must find a block node");
        assert_eq!(
            extract_name(&block, source, FileType::Terraform),
            Some(r#"resource "aws_instance" "web""#.to_string())
        );
    }

    #[test]
    fn test_extract_name_proto_message_and_service() {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_proto::LANGUAGE.into())
            .unwrap();
        let source = b"message Invoice {\n  string id = 1;\n}\n\nservice Billing {\n}\n";
        let tree = parser.parse(source, None).unwrap();
        let root = tree.root_node();
        let message = root
            .children(&mut root.walk())
            .find(|n| n.kind() == "message")
            .expect("must find a message node");
        assert_eq!(
            extract_name(&message, source, FileType::Protobuf),
            Some("message Invoice".to_string())
        );
        let service = root
            .children(&mut root.walk())
            .find(|n| n.kind() == "service")
            .expect("must find a service node");
        assert_eq!(
            extract_name(&service, source, FileType::Protobuf),
            Some("service Billing".to_string())
        );
    }

    #[test]
    fn test_extract_name_graphql_type_and_enum() {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_graphql::LANGUAGE.into())
            .unwrap();
        let source = b"type User {\n  id: ID!\n}\n\nenum Role {\n  ADMIN\n}\n";
        let tree = parser.parse(source, None).unwrap();
        let root = tree.root_node();

        // Search recursively for the GraphQL type definitions
        let mut obj_type = None;
        let mut enum_type = None;
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            if node.kind() == "object_type_definition" && obj_type.is_none() {
                obj_type = Some(node);
            }
            if node.kind() == "enum_type_definition" && enum_type.is_none() {
                enum_type = Some(node);
            }
            if obj_type.is_some() && enum_type.is_some() {
                break;
            }
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i as u32) {
                    stack.push(child);
                }
            }
        }

        let obj_type = obj_type.expect("must find object_type_definition");
        assert_eq!(
            extract_name(&obj_type, source, FileType::GraphQl),
            Some("type User".to_string())
        );
        let enum_type = enum_type.expect("must find enum_type_definition");
        assert_eq!(
            extract_name(&enum_type, source, FileType::GraphQl),
            Some("enum Role".to_string())
        );
    }

    #[test]
    fn test_extract_name_graphql_named_and_anonymous_operations() {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_graphql::LANGUAGE.into())
            .unwrap();
        let source = b"query GetUser {\n  user { id }\n}\n\nmutation {\n  createUser { id }\n}\n";
        let tree = parser.parse(source, None).unwrap();
        // operation_definition nodes may be nested (same discovery as the
        // other GraphQL/Starlark/PowerShell tests) -- search the whole tree,
        // not just direct root children.
        let mut stack = vec![tree.root_node()];
        let mut operations = Vec::new();
        while let Some(node) = stack.pop() {
            if node.kind() == "operation_definition" {
                operations.push(node);
            }
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i as u32) {
                    stack.push(child);
                }
            }
        }
        assert_eq!(
            operations.len(),
            2,
            "should find both operation_definition nodes"
        );

        let names: Vec<Option<String>> = operations
            .iter()
            .map(|n| extract_name(n, source, FileType::GraphQl))
            .collect();
        assert!(
            names.contains(&Some("query GetUser".to_string())),
            "named query must extract as 'query GetUser', got {names:?}"
        );
        assert!(
            names.contains(&Some("mutation".to_string())),
            "anonymous mutation must extract as bare 'mutation' (no name to append), got {names:?}"
        );
    }

    #[test]
    fn test_extract_name_starlark_call_with_name_kwarg() {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_starlark::LANGUAGE.into())
            .unwrap();
        let source = b"go_library(\n    name = \"mylib\",\n    srcs = [\"main.go\"],\n)\n";
        let tree = parser.parse(source, None).unwrap();
        let root = tree.root_node();

        // Search recursively for call node
        let mut call = None;
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            if node.kind() == "call" {
                call = Some(node);
                break;
            }
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i as u32) {
                    stack.push(child);
                }
            }
        }
        let call = call.expect("must find a call node");
        assert_eq!(
            extract_name(&call, source, FileType::Starlark),
            Some("go_library \"mylib\"".to_string())
        );
    }

    #[test]
    fn test_extract_name_starlark_call_without_name_kwarg_is_none() {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_starlark::LANGUAGE.into())
            .unwrap();
        let source = b"glob([\"*.go\"])\n";
        let tree = parser.parse(source, None).unwrap();
        let root = tree.root_node();

        // Search recursively for call node
        let mut call = None;
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            if node.kind() == "call" {
                call = Some(node);
                break;
            }
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i as u32) {
                    stack.push(child);
                }
            }
        }
        let call = call.expect("must find a call node");
        assert_eq!(extract_name(&call, source, FileType::Starlark), None);
    }

    #[test]
    fn test_extract_name_cmake_function_and_macro() {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_cmake::LANGUAGE.into())
            .unwrap();
        let source = b"function(add_component name)\nendfunction()\n\nmacro(enable_feature feature_name)\nendmacro()\n";
        let tree = parser.parse(source, None).unwrap();
        let root = tree.root_node();
        let func_def = root
            .children(&mut root.walk())
            .find(|n| n.kind() == "function_def")
            .expect("must find function_def");
        assert_eq!(
            extract_name(&func_def, source, FileType::Cmake),
            Some("add_component".to_string())
        );
        let macro_def = root
            .children(&mut root.walk())
            .find(|n| n.kind() == "macro_def")
            .expect("must find macro_def");
        assert_eq!(
            extract_name(&macro_def, source, FileType::Cmake),
            Some("enable_feature".to_string())
        );
    }

    #[test]
    fn test_extract_name_ini_section_keeps_brackets() {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_ini::LANGUAGE.into())
            .unwrap();
        let source = b"[database]\nhost = localhost\n";
        let tree = parser.parse(source, None).unwrap();
        let root = tree.root_node();
        let section = root
            .children(&mut root.walk())
            .find(|n| n.kind() == "section")
            .expect("must find a section node");
        assert_eq!(
            extract_name(&section, source, FileType::Ini),
            Some("[database]".to_string())
        );
    }

    #[test]
    fn test_extract_name_powershell_function_and_class() {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_powershell::LANGUAGE.into())
            .unwrap();
        let source = b"function Get-Greeting {\n    return \"hi\"\n}\n\nclass Greeter {\n    [string]$Name\n}\n";
        let tree = parser.parse(source, None).unwrap();
        let root = tree.root_node();

        // Search recursively for function_statement and class_statement
        let mut func = None;
        let mut class = None;
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            if node.kind() == "function_statement" && func.is_none() {
                func = Some(node);
            }
            if node.kind() == "class_statement" && class.is_none() {
                class = Some(node);
            }
            if func.is_some() && class.is_some() {
                break;
            }
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i as u32) {
                    stack.push(child);
                }
            }
        }

        let func = func.expect("must find function_statement");
        assert_eq!(
            extract_name(&func, source, FileType::PowerShell),
            Some("Get-Greeting".to_string())
        );
        let class = class.expect("must find class_statement");
        assert_eq!(
            extract_name(&class, source, FileType::PowerShell),
            Some("Greeter".to_string())
        );
    }

    #[test]
    fn test_classify_node_kind_config_as_code() {
        assert!(matches!(classify_node_kind("message"), AstNodeKind::Struct));
        assert!(matches!(classify_node_kind("enum"), AstNodeKind::Enum));
        assert!(matches!(classify_node_kind("service"), AstNodeKind::Other(s) if s == "service"));
        assert!(matches!(
            classify_node_kind("object_type_definition"),
            AstNodeKind::Other(s) if s == "type"
        ));
        assert!(matches!(
            classify_node_kind("interface_type_definition"),
            AstNodeKind::Interface
        ));
        assert!(matches!(
            classify_node_kind("enum_type_definition"),
            AstNodeKind::Enum
        ));
        assert!(matches!(
            classify_node_kind("input_object_type_definition"),
            AstNodeKind::Other(s) if s == "input"
        ));
        assert!(matches!(
            classify_node_kind("union_type_definition"),
            AstNodeKind::Other(s) if s == "union"
        ));
        assert!(matches!(
            classify_node_kind("scalar_type_definition"),
            AstNodeKind::Other(s) if s == "scalar"
        ));
        assert!(matches!(
            classify_node_kind("schema_definition"),
            AstNodeKind::Other(s) if s == "schema"
        ));
        assert!(matches!(
            classify_node_kind("directive_definition"),
            AstNodeKind::Other(s) if s == "directive"
        ));
        assert!(matches!(
            classify_node_kind("operation_definition"),
            AstNodeKind::Other(s) if s == "operation"
        ));
        assert!(matches!(
            classify_node_kind("fragment_definition"),
            AstNodeKind::Other(s) if s == "fragment"
        ));
        assert!(matches!(
            classify_node_kind("function_def"),
            AstNodeKind::Function
        ));
        assert!(matches!(
            classify_node_kind("macro_def"),
            AstNodeKind::Function
        ));
        assert!(matches!(
            classify_node_kind("function_statement"),
            AstNodeKind::Function
        ));
        assert!(matches!(
            classify_node_kind("class_statement"),
            AstNodeKind::Class
        ));
        // Deliberately left as the generic fallback (no clean AstNodeKind
        // equivalent, or — for Starlark's "call" — consistent with the
        // existing unmapped FileType::Elixir "call" precedent):
        assert!(matches!(classify_node_kind("block"), AstNodeKind::Other(s) if s == "block"));
        assert!(matches!(classify_node_kind("section"), AstNodeKind::Other(s) if s == "section"));
        assert!(matches!(classify_node_kind("call"), AstNodeKind::Other(s) if s == "call"));
        // Bash's function_definition and Groovy's function_definition/
        // method_declaration/class_declaration are exact-string matches to
        // EXISTING arms shared with other languages — no new arms needed.
        assert!(matches!(
            classify_node_kind("function_definition"),
            AstNodeKind::Function
        ));
        assert!(matches!(
            classify_node_kind("method_declaration"),
            AstNodeKind::Method
        ));
        assert!(matches!(
            classify_node_kind("class_declaration"),
            AstNodeKind::Class
        ));
    }

    /// Shared invariant check for the config-as-code language tests: every
    /// chunk's line range must be in-bounds within `content`, and every
    /// AstNode chunk must have a non-empty name. Adapted from
    /// `lightonai/colgrep`'s `assert_extractor_invariants` test pattern
    /// (`colgrep/src/parser/tests/common.rs`, Apache-2.0).
    fn assert_ast_chunk_invariants(chunks: &[Chunk], content: &str) {
        let total_lines = content.lines().count() as u32;
        for chunk in chunks {
            if let ChunkType::AstNode { name, .. } = &chunk.chunk_type {
                assert!(!name.is_empty(), "AstNode chunk must have a non-empty name");
            }
            assert!(
                chunk.start_line >= 1,
                "start_line must be 1-based (>=1), got {}",
                chunk.start_line
            );
            assert!(
                chunk.start_line <= chunk.end_line,
                "chunk start_line must not exceed end_line: {} > {}",
                chunk.start_line,
                chunk.end_line
            );
            assert!(
                chunk.end_line <= total_lines,
                "chunk end_line {} exceeds file length {} lines",
                chunk.end_line,
                total_lines
            );
        }
    }

    #[test]
    fn test_bash_function_chunking() {
        let chunker = AstChunker::new(256, 64);
        let content = r#"#!/usr/bin/env bash
function greet() {
    echo "Hello, $1!"
}

deploy() {
    echo "Deploying..."
    greet "world"
}
"#;
        let chunks = chunker.chunk(Path::new("deploy.sh"), content).unwrap();
        assert_ast_chunk_invariants(&chunks, content);
        let ast_chunks: Vec<_> = chunks
            .iter()
            .filter(|c| matches!(c.chunk_type, ChunkType::AstNode { .. }))
            .collect();
        assert!(
            !ast_chunks.is_empty(),
            "Bash must produce AstNode chunks, not fall back to text chunking"
        );

        let greet = ast_chunks.iter().find(|c| match &c.chunk_type {
            ChunkType::AstNode { name, kind, .. } => {
                name == "greet" && matches!(kind, AstNodeKind::Function)
            }
            _ => false,
        });
        assert!(
            greet.is_some(),
            "should find greet() as a Function-kind AstNode"
        );
        assert!(
            greet.unwrap().content.contains("Hello, $1"),
            "greet chunk should contain its body"
        );

        let deploy = ast_chunks.iter().any(|c| match &c.chunk_type {
            ChunkType::AstNode { name, kind, .. } => {
                name == "deploy" && matches!(kind, AstNodeKind::Function)
            }
            _ => false,
        });
        assert!(
            deploy,
            "should find the bare `deploy() {{...}}` form as a Function-kind AstNode too"
        );
    }

    #[test]
    fn test_groovy_class_method_and_function_chunking() {
        let chunker = AstChunker::new(256, 64);
        let content = r#"class Greeter {
    String greet(String name) {
        return "Hello, ${name}!"
    }
}

def standalone() {
    println "standalone"
}
"#;
        let chunks = chunker.chunk(Path::new("Task.groovy"), content).unwrap();
        assert_ast_chunk_invariants(&chunks, content);
        let ast_chunks: Vec<_> = chunks
            .iter()
            .filter(|c| matches!(c.chunk_type, ChunkType::AstNode { .. }))
            .collect();
        assert!(
            !ast_chunks.is_empty(),
            "Groovy must produce AstNode chunks, not fall back to text chunking"
        );

        // Greeter's method_declaration is structurally nested inside its
        // class_declaration span, so the existing outermost-wins dedup pass
        // (ast_chunker.rs's overlap-removal, same as Java/C#/PHP/Kotlin/Scala)
        // keeps only the Class chunk -- greet does NOT get its own separate
        // Method chunk. We assert the whole method is still captured, just
        // as part of Greeter's chunk content, not as a standalone chunk.
        let class = ast_chunks.iter().find(|c| match &c.chunk_type {
            ChunkType::AstNode { name, kind, .. } => {
                name == "Greeter" && matches!(kind, AstNodeKind::Class)
            }
            _ => false,
        });
        assert!(
            class.is_some(),
            "should find Greeter as a Class-kind AstNode"
        );
        assert!(
            class.unwrap().content.contains("Hello, ${name}"),
            "Greeter's chunk must contain its nested greet() method, even though \
             greet doesn't get its own separate chunk (outermost-wins dedup)"
        );

        let func = ast_chunks.iter().any(|c| match &c.chunk_type {
            ChunkType::AstNode { name, kind, .. } => {
                name == "standalone" && matches!(kind, AstNodeKind::Function)
            }
            _ => false,
        });
        assert!(
            func,
            "standalone() is not nested inside anything, so it must get its own \
             Function-kind AstNode chunk (unaffected by the dedup pass)"
        );
    }

    #[test]
    fn test_terraform_block_chunking() {
        let chunker = AstChunker::new(256, 64);
        let content = r#"resource "aws_instance" "web" {
  ami           = "ami-123456"
  instance_type = "t2.micro"
}

variable "region" {
  default = "us-east-1"
}
"#;
        let chunks = chunker.chunk(Path::new("main.tf"), content).unwrap();
        assert_ast_chunk_invariants(&chunks, content);
        let ast_chunks: Vec<_> = chunks
            .iter()
            .filter(|c| matches!(c.chunk_type, ChunkType::AstNode { .. }))
            .collect();
        assert!(
            !ast_chunks.is_empty(),
            "Terraform must produce AstNode chunks, not fall back to text chunking"
        );

        let resource = ast_chunks.iter().find(|c| match &c.chunk_type {
            ChunkType::AstNode { name, .. } => name == r#"resource "aws_instance" "web""#,
            _ => false,
        });
        assert!(
            resource.is_some(),
            "should find the resource block named with its type and labels: {:?}",
            ast_chunks
                .iter()
                .map(|c| c.symbol_name())
                .collect::<Vec<_>>()
        );
        assert!(
            resource.unwrap().content.contains("ami-123456"),
            "resource chunk must keep its attributes folded in (no recursion into the block body)"
        );

        let variable = ast_chunks.iter().any(|c| match &c.chunk_type {
            ChunkType::AstNode { name, .. } => name == r#"variable "region""#,
            _ => false,
        });
        assert!(
            variable,
            "should find the variable block named with its label"
        );
    }

    #[test]
    fn test_protobuf_message_and_service_chunking() {
        let chunker = AstChunker::new(256, 64);
        let content = r#"syntax = "proto3";

message Invoice {
  string id = 1;
  double amount = 2;
}

service Billing {
  rpc GetInvoice (InvoiceRequest) returns (Invoice);
}
"#;
        let chunks = chunker.chunk(Path::new("billing.proto"), content).unwrap();
        assert_ast_chunk_invariants(&chunks, content);
        let ast_chunks: Vec<_> = chunks
            .iter()
            .filter(|c| matches!(c.chunk_type, ChunkType::AstNode { .. }))
            .collect();
        assert!(
            !ast_chunks.is_empty(),
            "Protobuf must produce AstNode chunks, not fall back to text chunking"
        );

        let message = ast_chunks.iter().find(|c| match &c.chunk_type {
            ChunkType::AstNode { name, kind, .. } => {
                name == "message Invoice" && matches!(kind, AstNodeKind::Struct)
            }
            _ => false,
        });
        assert!(
            message.is_some(),
            "should find message Invoice as a Struct-kind AstNode"
        );
        assert!(message.unwrap().content.contains("double amount"));

        let service = ast_chunks.iter().find(|c| match &c.chunk_type {
            ChunkType::AstNode { name, kind, .. } => {
                name == "service Billing" && matches!(kind, AstNodeKind::Other(s) if s == "service")
            }
            _ => false,
        });
        assert!(
            service.is_some(),
            "should find service Billing as an Other(\"service\")-kind AstNode"
        );
        assert!(service.unwrap().content.contains("GetInvoice"));
    }
}
