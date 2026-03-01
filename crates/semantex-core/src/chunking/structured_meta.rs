use serde::{Deserialize, Serialize};

/// A structured doc-comment tag (e.g. `@param`, `@returns`, `@throws`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocTag {
    pub tag: String,
    pub name: Option<String>,
    pub text: String,
}

/// A reference to a type found in a chunk (parameter, return, field, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeRef {
    pub type_name: String,
    pub context: TypeRefContext,
}

/// Where a type reference was found.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TypeRefContext {
    Param,
    Return,
    Field,
    Local,
    Generic,
}

/// A trait/interface implementation relationship.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplRelation {
    pub implementor: String,
    pub trait_name: String,
}

/// High-level semantic role of a code chunk, inferred from naming conventions
/// and structural patterns.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SemanticRole {
    Constructor,
    Destructor,
    Validator,
    Transformer,
    Fetcher,
    Persister,
    Handler,
    Middleware,
    ErrorHandler,
    Sanitizer,
    Orchestrator,
}

impl SemanticRole {
    /// Return a space-separated label of search-relevant synonyms for this role.
    #[must_use]
    pub fn as_label(&self) -> &str {
        match self {
            Self::Constructor => "constructor initializer",
            Self::Destructor => "destructor cleanup teardown",
            Self::Validator => "validator checker verifier",
            Self::Transformer => "transformer converter serializer parser",
            Self::Fetcher => "fetcher getter reader loader",
            Self::Persister => "persister writer saver storage",
            Self::Handler => "handler processor dispatcher",
            Self::Middleware => "middleware interceptor filter guard",
            Self::ErrorHandler => "error handler recovery fallback retry",
            Self::Sanitizer => "sanitizer redactor cleaner masker",
            Self::Orchestrator => "orchestrator coordinator pipeline runner",
        }
    }
}

/// 6-layer structured metadata extracted from deep code analysis of a chunk.
///
/// Layer 1 -- AST: name, signature, params, return type, docstring, inheritance, doc tags
/// Layer 2 -- Call Graph: calls (outgoing), called_by (incoming, filled post-pass)
/// Layer 3 -- Control Flow: complexity, loops, branches, error handling
/// Layer 4 -- Data Flow: local variables, state mutations
/// Layer 5 -- Dependencies: imports, external references, type refs, impl relations
/// Layer 6 -- Semantic Role: high-level purpose classification
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StructuredChunkMeta {
    // -- Layer 1: AST --
    /// Function/class/method name.
    pub name: Option<String>,
    /// Qualified name (e.g., `ClassName.methodName`).
    pub qualified_name: Option<String>,
    /// Full signature text (up to opening brace).
    pub signature: Option<String>,
    /// Parameter names/types.
    pub params: Vec<String>,
    /// Return type annotation (if present).
    pub return_type: Option<String>,
    /// Leading docstring/comment.
    pub docstring: Option<String>,
    /// Parent class/struct/impl (for methods).
    pub parent_class: Option<String>,
    /// Inheritance: extends/implements.
    pub inherits: Vec<String>,
    /// Structured doc-comment tags (`@param`, `@returns`, `@throws`, etc.).
    #[serde(default)]
    pub doc_tags: Vec<DocTag>,

    // -- Layer 2: Call Graph --
    /// Functions/methods called within this chunk (outgoing edges).
    pub calls: Vec<String>,
    /// Functions/methods that call this chunk (incoming edges, filled by post-pass).
    pub called_by: Vec<String>,

    // -- Layer 3: Control Flow --
    /// Cyclomatic complexity estimate (branch/loop count + 1).
    pub complexity: u32,
    /// Contains loop constructs (for, while, loop, etc.).
    pub has_loops: bool,
    /// Contains conditional branches (if, match, switch, etc.).
    pub has_branches: bool,
    /// Contains error handling (try/catch, Result match, ?, etc.).
    pub has_error_handling: bool,

    // -- Layer 4: Data Flow --
    /// Local variable names declared in this chunk.
    pub local_vars: Vec<String>,
    /// State mutations (assignments to fields, globals, etc.).
    pub state_mutations: Vec<String>,

    // -- Layer 5: Dependencies --
    /// Import/use statements in scope.
    pub imports: Vec<String>,
    /// External references (types, modules not defined locally).
    pub external_refs: Vec<String>,
    /// Fully resolved import paths (e.g. `crate::db::ConnectionPool`).
    #[serde(default)]
    pub resolved_imports: Vec<String>,
    /// Type references with context (parameter, return, field, etc.).
    #[serde(default)]
    pub type_refs: Vec<TypeRef>,
    /// Trait/interface implementation relationships.
    #[serde(default)]
    pub implements: Vec<ImplRelation>,

    // -- Layer 6: Semantic Role --
    /// High-level semantic role inferred from naming/structure.
    #[serde(default)]
    pub semantic_role: Option<SemanticRole>,

    // -- Generated --
    /// Human-readable kind label (e.g. "function", "class", "struct").
    /// Set from `AstNodeKind::label()` during AST chunking.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// Generated NL summary for BM25 enrichment (from all 6 layers).
    pub nl_summary: String,
}

impl StructuredChunkMeta {
    /// Generate a natural-language summary from all 6 analysis layers.
    ///
    /// Rule-based (no LLM): concatenates expanded names, calls, callers,
    /// control flow hints, dependency information, and semantic role labels.
    pub fn generate_nl_summary(&mut self) {
        let mut parts: Vec<String> = Vec::new();

        // Layer 1: AST identity
        if let Some(ref name) = self.name {
            // Expand camelCase/snake_case: "allSettled" -> "all settled"
            parts.push(format!("{} {}", self.kind_label(), expand_identifier(name)));
        }
        if let Some(ref parent) = self.parent_class {
            parts.push(format!("in {}", expand_identifier(parent)));
        }
        if !self.inherits.is_empty() {
            let expanded: Vec<String> =
                self.inherits.iter().map(|i| expand_identifier(i)).collect();
            parts.push(format!("extends {}", expanded.join(", ")));
        }
        if !self.params.is_empty() {
            parts.push(format!("parameters: {}", self.params.join(", ")));
        }
        if let Some(ref ret) = self.return_type {
            parts.push(format!("returns {ret}"));
        }

        // Docstring (first sentence) — surfaced early so summaries are human-readable
        if let Some(ref doc) = self.docstring {
            let first_sentence = doc.split('.').next().unwrap_or(doc);
            if first_sentence.len() < 200 {
                parts.push(first_sentence.trim().to_string());
            }
        }

        // Semantic role — surfaced early for quick triage
        if let Some(ref role) = self.semantic_role {
            parts.push(format!("role: {}", role.as_label()));
        }

        // Layer 2: Call graph (most impactful for vocabulary bridging)
        if !self.calls.is_empty() {
            let expanded_calls: Vec<String> =
                self.calls.iter().map(|c| expand_identifier(c)).collect();
            parts.push(format!("calls {}", expanded_calls.join(", ")));
        }
        if !self.called_by.is_empty() {
            let expanded_callers: Vec<String> = self
                .called_by
                .iter()
                .map(|c| expand_identifier(c))
                .collect();
            parts.push(format!("called by {}", expanded_callers.join(", ")));
        }

        // Layer 3: Control flow (searchable patterns)
        if self.has_error_handling {
            parts.push("handles errors".to_string());
        }
        if self.has_loops && self.complexity > 3 {
            parts.push("complex iteration".to_string());
        }

        // Layer 4: Data flow (state mutations are architecturally relevant)
        if !self.state_mutations.is_empty() {
            let expanded: Vec<String> = self
                .state_mutations
                .iter()
                .map(|m| expand_identifier(m))
                .collect();
            parts.push(format!("mutates {}", expanded.join(", ")));
        }

        // Layer 5: Dependencies (key external refs)
        if !self.external_refs.is_empty() {
            let expanded: Vec<String> = self
                .external_refs
                .iter()
                .take(5) // Limit to avoid bloating BM25 content
                .map(|r| expand_identifier(r))
                .collect();
            parts.push(format!("uses {}", expanded.join(", ")));
        }

        // Layer 5 enhanced: type references
        if !self.type_refs.is_empty() {
            let expanded: Vec<String> = self
                .type_refs
                .iter()
                .map(|tr| expand_identifier(&tr.type_name))
                .collect();
            parts.push(format!("uses types {}", expanded.join(", ")));
        }

        // Layer 5 enhanced: implementation relationships
        for rel in &self.implements {
            parts.push(format!(
                "{} implements {}",
                expand_identifier(&rel.implementor),
                expand_identifier(&rel.trait_name)
            ));
        }

        // Layer 5 enhanced: resolved imports (last path segment, up to 8)
        if !self.resolved_imports.is_empty() {
            let segments: Vec<String> = self
                .resolved_imports
                .iter()
                .take(8)
                .filter_map(|imp| {
                    imp.rsplit([':', '/', '.'])
                        .next()
                        .filter(|s| !s.is_empty())
                        .map(expand_identifier)
                })
                .collect();
            if !segments.is_empty() {
                parts.push(format!("imports {}", segments.join(", ")));
            }
        }

        // Doc tags (Layer 1 enhanced)
        for dt in &self.doc_tags {
            match dt.tag.as_str() {
                "returns" | "return" => {
                    parts.push(format!("returns {}", dt.text));
                }
                "throws" | "exception" | "raise" | "raises" => {
                    parts.push(format!("may throw {}", dt.text));
                }
                "deprecated" => {
                    parts.push("deprecated".to_string());
                }
                "see" | "link" => {
                    parts.push(format!("related to {}", dt.text));
                }
                _ => {}
            }
        }

        self.nl_summary = parts.join("; ");
    }

    /// Return BM25 expansion text (prepended to chunk content for indexing).
    pub fn bm25_expansion(&self) -> String {
        self.nl_summary.clone()
    }

    /// Return a human-readable label for the kind of code element.
    pub fn kind_label(&self) -> &str {
        self.kind.as_deref().unwrap_or("function")
    }

    /// Generate a compact, agent-optimized display summary.
    ///
    /// Unlike `nl_summary` (designed for BM25 index enrichment), this output
    /// preserves original identifier casing and filters trivial calls to
    /// minimize tokens while maximizing actionable information.
    ///
    /// Format:
    /// ```text
    /// {signature or reconstructed sig}
    ///   {docstring first sentence}
    ///   calls: {non-trivial calls, max 5}
    ///   called_by: {callers, max 5}
    ///   implements: {trait names}
    ///   [{semantic_role}]
    /// ```
    pub fn display_summary(&self) -> String {
        let mut parts: Vec<String> = Vec::new();

        // Line 1: signature (or reconstructed fallback)
        if let Some(ref sig) = self.signature {
            parts.push(sig.clone());
        } else if let Some(ref name) = self.name {
            let kind = self.kind.as_deref();
            let kind_prefix = match kind {
                Some("fn" | "function" | "method") => "fn ",
                Some("class") => "class ",
                Some("struct") => "struct ",
                Some("enum") => "enum ",
                Some("interface") => "interface ",
                Some("module") => "mod ",
                Some(other) => {
                    parts.push(format!("{other} {name}"));
                    ""
                }
                None => "fn ",
            };
            if !kind_prefix.is_empty() {
                let mut sig = format!("{kind_prefix}{name}");
                let is_type_decl = matches!(
                    kind,
                    Some("class" | "struct" | "enum" | "interface" | "module")
                );
                if !self.params.is_empty() && !is_type_decl {
                    sig.push('(');
                    sig.push_str(&self.params.join(", "));
                    sig.push(')');
                }
                if let Some(ref ret) = self.return_type {
                    sig.push_str(" -> ");
                    sig.push_str(ret);
                }
                parts.push(sig);
            }
        }

        // Line 2: docstring first sentence (max 120 chars)
        if let Some(ref doc) = self.docstring {
            let first = doc.split('.').next().unwrap_or(doc).trim();
            if !first.is_empty() && first.len() <= 120 {
                parts.push(format!("  {first}"));
            }
        }

        // Line 3: filtered calls (non-trivial, max 5)
        let filtered_calls: Vec<&str> = self
            .calls
            .iter()
            .filter(|c| !is_trivial_call(c))
            .take(5)
            .map(String::as_str)
            .collect();
        if !filtered_calls.is_empty() {
            parts.push(format!("  calls: {}", filtered_calls.join(", ")));
        }

        // Line 4: called_by (max 5)
        if !self.called_by.is_empty() {
            let callers: Vec<&str> = self.called_by.iter().take(5).map(String::as_str).collect();
            parts.push(format!("  called_by: {}", callers.join(", ")));
        }

        // Line 5: implements (for structs/classes)
        if !self.implements.is_empty() {
            let traits: Vec<&str> = self
                .implements
                .iter()
                .map(|r| r.trait_name.as_str())
                .collect();
            parts.push(format!("  implements: {}", traits.join(", ")));
        }

        // Line 6: semantic role
        if let Some(ref role) = self.semantic_role {
            let label = match role {
                SemanticRole::Constructor => "constructor",
                SemanticRole::Destructor => "destructor",
                SemanticRole::Validator => "validator",
                SemanticRole::Transformer => "transformer",
                SemanticRole::Fetcher => "fetcher",
                SemanticRole::Persister => "persister",
                SemanticRole::Handler => "handler",
                SemanticRole::Middleware => "middleware",
                SemanticRole::ErrorHandler => "error_handler",
                SemanticRole::Sanitizer => "sanitizer",
                SemanticRole::Orchestrator => "orchestrator",
            };
            parts.push(format!("  [{label}]"));
        }

        parts.join("\n")
    }
}

/// Ubiquitous stdlib/builtin calls that provide no architectural insight.
/// Used by `display_summary()` to filter noise from the calls list.
const TRIVIAL_CALLS: &[&str] = &[
    // Collection operations
    "push",
    "pop",
    "insert",
    "remove",
    "get",
    "set",
    "len",
    "is_empty",
    "contains",
    "iter",
    "into_iter",
    "collect",
    "extend",
    "clear",
    "first",
    "last",
    "next",
    // String operations
    "to_string",
    "as_str",
    "to_owned",
    "trim",
    "split",
    "join",
    "starts_with",
    "ends_with",
    "replace",
    "chars",
    "lines",
    // Formatting
    "format",
    "write",
    "writeln",
    "println",
    "eprintln",
    "print",
    "display",
    // Option/Result
    "unwrap",
    "unwrap_or",
    "unwrap_or_default",
    "unwrap_or_else",
    "expect",
    "ok",
    "err",
    "map",
    "and_then",
    "or_else",
    "map_or",
    "map_or_else",
    "map_err",
    "is_some",
    "is_none",
    "is_ok",
    "is_err",
    "as_ref",
    "as_deref",
    // Conversion
    "into",
    "from",
    "try_into",
    "try_from",
    "as_mut",
    "clone",
    "to_vec",
    // Comparison
    "eq",
    "ne",
    "cmp",
    "partial_cmp",
    "min",
    "max",
    // Memory
    "drop",
    "take",
    "swap",
    "replace",
    // Type checking (dynamic languages)
    "typeof",
    "instanceof",
    // Logging (cross-language)
    "log",
    "debug",
    "info",
    "warn",
    "error",
    "trace",
];

/// Check if a call name is a ubiquitous stdlib operation (no architectural insight).
///
/// Extracts the last segment after `.` or `::` and checks against `TRIVIAL_CALLS`.
pub fn is_trivial_call(call: &str) -> bool {
    let segment = call.rsplit(['.', ':']).next().unwrap_or(call);
    TRIVIAL_CALLS.contains(&segment)
}

/// Expand a programming identifier into space-separated words.
///
/// Handles `camelCase`, `snake_case`, and `dot.notation` by splitting on
/// non-alphanumeric characters and camelCase boundaries, then lowercasing.
///
/// # Examples
///
/// ```
/// use semantex_core::chunking::structured_meta::expand_identifier;
///
/// assert_eq!(expand_identifier("allSettled"), "all settled");
/// assert_eq!(expand_identifier("Promise.allSettled"), "promise all settled");
/// assert_eq!(expand_identifier("get_user_by_id"), "get user by id");
/// assert_eq!(expand_identifier("ConnectionPool"), "connection pool");
/// ```
pub fn expand_identifier(ident: &str) -> String {
    // Split on non-alphanumeric, then camelCase split each part
    let mut words = Vec::new();
    for part in ident.split(|c: char| !c.is_alphanumeric()) {
        if part.is_empty() {
            continue;
        }
        words.extend(camel_case_split(part));
    }
    words.join(" ").to_lowercase()
}

/// Split a string on camelCase boundaries, including acronym boundaries.
///
/// Handles both standard camelCase (`"allSettled"` → `["all", "Settled"]`)
/// and consecutive uppercase / acronyms (`"HTMLParser"` → `["HTML", "Parser"]`).
fn camel_case_split(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = s.chars().collect();

    for i in 0..chars.len() {
        if i > 0 {
            let prev_upper = chars[i - 1].is_uppercase();
            let curr_upper = chars[i].is_uppercase();

            // Standard camelCase boundary: lowerUpper (e.g. "allSettled" at 'S')
            if !prev_upper && curr_upper {
                if !current.is_empty() {
                    parts.push(current.clone());
                    current.clear();
                }
            }
            // Acronym boundary: consecutive uppercase followed by lowercase
            // (e.g. "HTMLParser" at 'a' → split "HTML" from "Parser")
            else if prev_upper && !curr_upper && current.len() > 1 {
                let split_point = current.len() - 1;
                let prefix = current[..split_point].to_string();
                current = current[split_point..].to_string();
                if !prefix.is_empty() {
                    parts.push(prefix);
                }
            }
        }
        current.push(chars[i]);
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_camel_case() {
        assert_eq!(expand_identifier("allSettled"), "all settled");
    }

    #[test]
    fn test_expand_dotted_camel() {
        assert_eq!(
            expand_identifier("Promise.allSettled"),
            "promise all settled"
        );
    }

    #[test]
    fn test_expand_snake_case() {
        assert_eq!(expand_identifier("get_user_by_id"), "get user by id");
    }

    #[test]
    fn test_expand_pascal_case() {
        assert_eq!(expand_identifier("ConnectionPool"), "connection pool");
    }

    #[test]
    fn test_expand_single_word() {
        assert_eq!(expand_identifier("fetch"), "fetch");
    }

    #[test]
    fn test_expand_empty() {
        assert_eq!(expand_identifier(""), "");
    }

    #[test]
    fn test_generate_nl_summary_basic() {
        let mut meta = StructuredChunkMeta {
            name: Some("fetchUserData".to_string()),
            parent_class: Some("UserService".to_string()),
            params: vec!["userId: String".to_string()],
            calls: vec!["db.query".to_string(), "validateId".to_string()],
            has_error_handling: true,
            ..Default::default()
        };
        meta.generate_nl_summary();
        assert!(meta.nl_summary.contains("fetch user data"));
        assert!(meta.nl_summary.contains("in user service"));
        assert!(meta.nl_summary.contains("parameters: userId: String"));
        assert!(meta.nl_summary.contains("calls db query, validate id"));
        assert!(meta.nl_summary.contains("handles errors"));
    }

    #[test]
    fn test_generate_nl_summary_with_called_by() {
        let mut meta = StructuredChunkMeta {
            name: Some("validate".to_string()),
            called_by: vec!["processOrder".to_string()],
            ..Default::default()
        };
        meta.generate_nl_summary();
        assert!(meta.nl_summary.contains("called by process order"));
    }

    #[test]
    fn test_bm25_expansion_returns_summary() {
        let mut meta = StructuredChunkMeta {
            name: Some("init".to_string()),
            ..Default::default()
        };
        meta.generate_nl_summary();
        assert_eq!(meta.bm25_expansion(), meta.nl_summary);
    }

    #[test]
    fn test_kind_label_default() {
        let meta = StructuredChunkMeta::default();
        assert_eq!(meta.kind_label(), "function");
    }

    #[test]
    fn test_kind_label_class() {
        let meta = StructuredChunkMeta {
            kind: Some("class".to_string()),
            ..Default::default()
        };
        assert_eq!(meta.kind_label(), "class");
    }

    #[test]
    fn test_kind_label_struct() {
        let meta = StructuredChunkMeta {
            kind: Some("struct".to_string()),
            ..Default::default()
        };
        assert_eq!(meta.kind_label(), "struct");
    }

    #[test]
    fn test_expand_acronym() {
        assert_eq!(expand_identifier("HTMLParser"), "html parser");
    }

    #[test]
    fn test_expand_acronym_all_caps() {
        assert_eq!(expand_identifier("JSON"), "json");
    }

    #[test]
    fn test_docstring_first_sentence_included() {
        let mut meta = StructuredChunkMeta {
            name: Some("parse".to_string()),
            docstring: Some("Parse the input string. Returns parsed result.".to_string()),
            ..Default::default()
        };
        meta.generate_nl_summary();
        assert!(meta.nl_summary.contains("Parse the input string"));
        assert!(!meta.nl_summary.contains("Returns parsed result"));
    }

    #[test]
    fn test_docstring_too_long_excluded() {
        let long_doc = "x".repeat(250);
        let mut meta = StructuredChunkMeta {
            name: Some("f".to_string()),
            docstring: Some(long_doc),
            ..Default::default()
        };
        meta.generate_nl_summary();
        // The long docstring has no '.', so first_sentence is the whole thing (>200 chars)
        // and should be excluded
        assert_eq!(meta.nl_summary, "function f");
    }

    #[test]
    fn test_semantic_role_label() {
        assert_eq!(
            SemanticRole::Constructor.as_label(),
            "constructor initializer"
        );
        assert_eq!(
            SemanticRole::Sanitizer.as_label(),
            "sanitizer redactor cleaner masker"
        );
    }

    #[test]
    fn test_nl_summary_with_semantic_role() {
        let mut meta = StructuredChunkMeta {
            name: Some("sanitizeInput".to_string()),
            semantic_role: Some(SemanticRole::Sanitizer),
            ..Default::default()
        };
        meta.generate_nl_summary();
        assert!(meta.nl_summary.contains("sanitize input"));
        assert!(
            meta.nl_summary
                .contains("role: sanitizer redactor cleaner masker")
        );
    }

    #[test]
    fn test_nl_summary_with_type_refs() {
        let mut meta = StructuredChunkMeta {
            name: Some("process".to_string()),
            type_refs: vec![
                TypeRef {
                    type_name: "ConnectionPool".to_string(),
                    context: TypeRefContext::Param,
                },
                TypeRef {
                    type_name: "Result".to_string(),
                    context: TypeRefContext::Return,
                },
            ],
            ..Default::default()
        };
        meta.generate_nl_summary();
        assert!(
            meta.nl_summary
                .contains("uses types connection pool, result")
        );
    }

    #[test]
    fn test_nl_summary_with_implements() {
        let mut meta = StructuredChunkMeta {
            name: Some("MyService".to_string()),
            kind: Some("struct".to_string()),
            implements: vec![ImplRelation {
                implementor: "MyService".to_string(),
                trait_name: "ServiceTrait".to_string(),
            }],
            ..Default::default()
        };
        meta.generate_nl_summary();
        assert!(
            meta.nl_summary
                .contains("my service implements service trait")
        );
    }

    #[test]
    fn test_nl_summary_with_resolved_imports() {
        let mut meta = StructuredChunkMeta {
            name: Some("handler".to_string()),
            resolved_imports: vec![
                "crate::db::ConnectionPool".to_string(),
                "std::sync::Arc".to_string(),
            ],
            ..Default::default()
        };
        meta.generate_nl_summary();
        assert!(meta.nl_summary.contains("imports"));
        assert!(meta.nl_summary.contains("connection pool"));
        assert!(meta.nl_summary.contains("arc"));
    }

    #[test]
    fn test_nl_summary_with_doc_tags() {
        let mut meta = StructuredChunkMeta {
            name: Some("divide".to_string()),
            doc_tags: vec![
                DocTag {
                    tag: "returns".to_string(),
                    name: None,
                    text: "the quotient".to_string(),
                },
                DocTag {
                    tag: "throws".to_string(),
                    name: Some("ArithmeticError".to_string()),
                    text: "division by zero".to_string(),
                },
                DocTag {
                    tag: "deprecated".to_string(),
                    name: None,
                    text: "use safe_divide instead".to_string(),
                },
                DocTag {
                    tag: "see".to_string(),
                    name: None,
                    text: "safe_divide".to_string(),
                },
            ],
            ..Default::default()
        };
        meta.generate_nl_summary();
        assert!(meta.nl_summary.contains("returns the quotient"));
        assert!(meta.nl_summary.contains("may throw division by zero"));
        assert!(meta.nl_summary.contains("deprecated"));
        assert!(meta.nl_summary.contains("related to safe_divide"));
    }

    #[test]
    fn test_backward_compat_default() {
        // Ensure old serialized data (without new fields) deserializes correctly
        let json = r#"{"name":"foo","nl_summary":"test","params":[],"inherits":[],"calls":[],"called_by":[],"complexity":0,"has_loops":false,"has_branches":false,"has_error_handling":false,"local_vars":[],"state_mutations":[],"imports":[],"external_refs":[]}"#;
        let meta: StructuredChunkMeta = serde_json::from_str(json).unwrap();
        assert!(meta.doc_tags.is_empty());
        assert!(meta.type_refs.is_empty());
        assert!(meta.implements.is_empty());
        assert!(meta.resolved_imports.is_empty());
        assert!(meta.semantic_role.is_none());
    }

    // --- display_summary tests ---

    #[test]
    fn test_display_summary_with_signature() {
        let meta = StructuredChunkMeta {
            name: Some("handle_search".to_string()),
            signature: Some("fn handle_search(&self, req: SearchRequest) -> Response".to_string()),
            docstring: Some("Handle an incoming search request. Returns response.".to_string()),
            calls: vec![
                "self.searcher.search".to_string(),
                "format".to_string(), // trivial — filtered
                "push".to_string(),   // trivial — filtered
                "search_count.fetch_add".to_string(),
            ],
            called_by: vec!["handle".to_string()],
            semantic_role: Some(SemanticRole::Handler),
            kind: Some("fn".to_string()),
            ..Default::default()
        };
        let summary = meta.display_summary();
        assert!(
            summary.contains("fn handle_search(&self, req: SearchRequest) -> Response"),
            "signature missing: {summary}"
        );
        assert!(
            summary.contains("Handle an incoming search request"),
            "docstring missing: {summary}"
        );
        assert!(
            summary.contains("calls: self.searcher.search, search_count.fetch_add"),
            "calls missing: {summary}"
        );
        assert!(
            !summary.contains("format"),
            "trivial call 'format' should be filtered"
        );
        assert!(
            !summary.contains("push"),
            "trivial call 'push' should be filtered"
        );
        assert!(
            summary.contains("called_by: handle"),
            "called_by missing: {summary}"
        );
        assert!(summary.contains("[handler]"), "role missing: {summary}");
    }

    #[test]
    fn test_display_summary_fallback_no_signature() {
        let meta = StructuredChunkMeta {
            name: Some("validate_input".to_string()),
            params: vec!["input: &str".to_string(), "strict: bool".to_string()],
            return_type: Some("Result<()>".to_string()),
            kind: Some("fn".to_string()),
            ..Default::default()
        };
        let summary = meta.display_summary();
        assert!(
            summary.contains("fn validate_input(input: &str, strict: bool) -> Result<()>"),
            "fallback sig wrong: {summary}"
        );
    }

    #[test]
    fn test_display_summary_struct() {
        let meta = StructuredChunkMeta {
            name: Some("HybridSearcher".to_string()),
            signature: Some("pub struct HybridSearcher".to_string()),
            docstring: Some("Combines dense and sparse search. With reranking.".to_string()),
            implements: vec![ImplRelation {
                implementor: "HybridSearcher".to_string(),
                trait_name: "Drop".to_string(),
            }],
            kind: Some("struct".to_string()),
            ..Default::default()
        };
        let summary = meta.display_summary();
        assert!(summary.contains("pub struct HybridSearcher"), "{summary}");
        assert!(
            summary.contains("Combines dense and sparse search"),
            "{summary}"
        );
        assert!(summary.contains("implements: Drop"), "{summary}");
    }

    #[test]
    fn test_display_summary_minimal() {
        let meta = StructuredChunkMeta {
            name: Some("main".to_string()),
            kind: Some("fn".to_string()),
            ..Default::default()
        };
        let summary = meta.display_summary();
        assert_eq!(summary, "fn main");
    }

    #[test]
    fn test_display_summary_no_name() {
        let meta = StructuredChunkMeta::default();
        let summary = meta.display_summary();
        assert!(summary.is_empty(), "empty meta should yield empty summary");
    }

    #[test]
    fn test_display_summary_all_trivial_calls() {
        let meta = StructuredChunkMeta {
            name: Some("f".to_string()),
            calls: vec!["push".to_string(), "len".to_string(), "clone".to_string()],
            kind: Some("fn".to_string()),
            ..Default::default()
        };
        let summary = meta.display_summary();
        assert!(
            !summary.contains("calls:"),
            "all-trivial calls should produce no calls line"
        );
    }

    #[test]
    fn test_display_summary_calls_max_5() {
        let meta = StructuredChunkMeta {
            name: Some("f".to_string()),
            calls: vec![
                "alpha".to_string(),
                "beta".to_string(),
                "gamma".to_string(),
                "delta".to_string(),
                "epsilon".to_string(),
                "zeta".to_string(),
            ],
            kind: Some("fn".to_string()),
            ..Default::default()
        };
        let summary = meta.display_summary();
        assert!(
            summary.contains("calls: alpha, beta, gamma, delta, epsilon"),
            "{summary}"
        );
        assert!(!summary.contains("zeta"), "6th call should be truncated");
    }

    #[test]
    fn test_display_summary_docstring_too_long_excluded() {
        let meta = StructuredChunkMeta {
            name: Some("f".to_string()),
            docstring: Some("x".repeat(200)),
            kind: Some("fn".to_string()),
            ..Default::default()
        };
        let summary = meta.display_summary();
        // No '.' so first sentence = whole string > 120 chars = excluded
        assert_eq!(
            summary, "fn f",
            "long docstring should be excluded: {summary}"
        );
    }

    #[test]
    fn test_display_summary_enum() {
        let meta = StructuredChunkMeta {
            name: Some("Request".to_string()),
            signature: Some("pub enum Request".to_string()),
            kind: Some("enum".to_string()),
            ..Default::default()
        };
        let summary = meta.display_summary();
        assert_eq!(summary, "pub enum Request");
    }

    #[test]
    fn test_display_summary_module_fallback() {
        let meta = StructuredChunkMeta {
            name: Some("server".to_string()),
            kind: Some("module".to_string()),
            docstring: Some("TCP daemon server for handling search requests.".to_string()),
            ..Default::default()
        };
        let summary = meta.display_summary();
        assert!(summary.contains("mod server"), "{summary}");
        assert!(
            summary.contains("TCP daemon server for handling search requests"),
            "{summary}"
        );
    }

    #[test]
    fn test_display_summary_method_kind() {
        let meta = StructuredChunkMeta {
            name: Some("run".to_string()),
            kind: Some("method".to_string()),
            params: vec!["&self".to_string()],
            return_type: Some("()".to_string()),
            ..Default::default()
        };
        let summary = meta.display_summary();
        assert!(summary.contains("fn run(&self) -> ()"), "{summary}");
    }

    // --- is_trivial_call tests ---

    #[test]
    fn test_is_trivial_call_simple() {
        assert!(is_trivial_call("push"));
        assert!(is_trivial_call("unwrap"));
        assert!(is_trivial_call("clone"));
        assert!(is_trivial_call("format"));
    }

    #[test]
    fn test_is_trivial_call_qualified() {
        assert!(is_trivial_call("self.results.push"));
        assert!(is_trivial_call("Vec::len"));
        assert!(is_trivial_call("option.unwrap_or"));
    }

    #[test]
    fn test_is_trivial_call_non_trivial() {
        assert!(!is_trivial_call("search"));
        assert!(!is_trivial_call("self.searcher.search"));
        assert!(!is_trivial_call("validate_credentials"));
        assert!(!is_trivial_call("db.query"));
        assert!(!is_trivial_call("new")); // constructors carry information
    }
}
