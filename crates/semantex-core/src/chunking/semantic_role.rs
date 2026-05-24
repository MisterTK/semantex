//! Rule-based semantic role classification for code chunks.
//!
//! Classifies functions/methods by their behavioral purpose (constructor, validator,
//! transformer, etc.) using name patterns and structural signals.
//!
//! Also hosts `synthesize_nl_annotation` — the E3 ExCS-style chunk annotation
//! synthesizer (see v0.3 SOTA spec). It produces a rule-derived natural-language
//! block from existing signals (no LLM call) that is indexed in Tantivy's
//! `nl_annotation` field to close the NL → code vocabulary gap.

use super::structured_meta::{
    SemanticRole, StructuredChunkMeta, expand_identifier, is_trivial_call,
};

/// Classify a chunk's semantic role based on name patterns and structural signals.
///
/// Returns `None` if no role can be confidently determined.
#[allow(clippy::too_many_lines)]
pub fn classify_semantic_role(meta: &StructuredChunkMeta, _content: &str) -> Option<SemanticRole> {
    let raw_name = meta.name.as_deref()?;
    let name = raw_name.to_lowercase();
    let expanded = expand_identifier(raw_name).to_lowercase();

    // Priority ordering: more specific patterns first

    // Sanitizer: sanitize, redact, mask, strip_pii, clean, scrub
    if matches_any(
        &expanded,
        &[
            "sanitiz",
            "redact",
            "mask pii",
            "strip pii",
            "clean pii",
            "scrub",
        ],
    ) {
        return Some(SemanticRole::Sanitizer);
    }
    // Be careful with "mask", "clean", "strip" alone -- too generic.
    // Only match if combined with sanitization context.
    if matches_any(&name, &["sanitize", "redact", "scrub"]) {
        return Some(SemanticRole::Sanitizer);
    }

    // ErrorHandler: catch, recover, fallback, retry, on_error
    if matches_any(
        &expanded,
        &[
            "error handler",
            "catch error",
            "recover",
            "fallback",
            "retry",
        ],
    ) {
        return Some(SemanticRole::ErrorHandler);
    }
    if name.starts_with("on_error") || name.starts_with("handle_error") {
        return Some(SemanticRole::ErrorHandler);
    }

    // Middleware: intercept, filter, guard, middleware
    if matches_any(
        &expanded,
        &[
            "intercept",
            "middlewar",
            "guard request",
            "before request",
            "after request",
        ],
    ) {
        return Some(SemanticRole::Middleware);
    }
    if matches_any(&name, &["interceptor", "middleware"]) {
        return Some(SemanticRole::Middleware);
    }

    // Constructor: new, create, init, build, make, setup, from
    if matches_any(&name, &["new", "create", "init", "build", "make", "setup"])
        || name.starts_with("create_")
        || name.starts_with("build_")
        || name.starts_with("new_")
        || name.starts_with("from_")
        || expanded.starts_with("create ")
        || expanded.starts_with("build ")
    {
        return Some(SemanticRole::Constructor);
    }

    // Destructor: drop, close, dispose, shutdown, destroy, cleanup, teardown
    if matches_any(
        &expanded,
        &[
            "close",
            "dispos",
            "shutdown",
            "shut down",
            "destroy",
            "cleanup",
            "clean up",
            "teardown",
            "tear down",
        ],
    ) {
        return Some(SemanticRole::Destructor);
    }
    if name == "drop" || name.starts_with("drop_") {
        return Some(SemanticRole::Destructor);
    }

    // Validator: validate, check, verify, assert, ensure, is_valid
    if matches_any(&expanded, &["validat", "verif", "is valid"]) {
        return Some(SemanticRole::Validator);
    }
    if name.starts_with("check_") || name.starts_with("ensure_") || name.starts_with("assert_") {
        return Some(SemanticRole::Validator);
    }

    // Transformer: convert, transform, serialize, deserialize, parse, format, encode, decode
    if matches_any(
        &expanded,
        &[
            "convert",
            "transform",
            "serializ",
            "deserializ",
            "encod",
            "decod",
        ],
    ) {
        return Some(SemanticRole::Transformer);
    }
    if name.starts_with("to_")
        || name.starts_with("into_")
        || name.starts_with("parse_")
        || name.starts_with("format_")
    {
        return Some(SemanticRole::Transformer);
    }

    // Fetcher: get, fetch, load, read, find, search, retrieve, lookup
    if matches_any(&expanded, &["fetch", "retriev", "lookup", "look up"]) {
        return Some(SemanticRole::Fetcher);
    }
    if name.starts_with("get_")
        || name.starts_with("load_")
        || name.starts_with("read_")
        || name.starts_with("find_")
    {
        return Some(SemanticRole::Fetcher);
    }

    // Persister: save, store, write, insert, update, delete, put, persist, upsert
    if matches_any(&expanded, &["persist", "upsert"]) {
        return Some(SemanticRole::Persister);
    }
    if name.starts_with("save_")
        || name.starts_with("store_")
        || name.starts_with("write_")
        || name.starts_with("insert_")
        || name.starts_with("delete_")
        || name.starts_with("put_")
    {
        return Some(SemanticRole::Persister);
    }
    if matches_any(
        &name,
        &[
            "save", "store", "write", "insert", "delete", "persist", "upsert",
        ],
    ) {
        return Some(SemanticRole::Persister);
    }

    // Handler: handle, process, dispatch (generic, check late)
    if matches_any(&expanded, &["handl", "dispatch"]) || name.starts_with("process_") {
        return Some(SemanticRole::Handler);
    }
    if name.starts_with("on_") {
        return Some(SemanticRole::Handler);
    }

    // Orchestrator: orchestrate, coordinate, pipeline, workflow
    if matches_any(
        &expanded,
        &["orchestrat", "coordinat", "pipeline", "workflow"],
    ) {
        return Some(SemanticRole::Orchestrator);
    }
    // "run"/"execute" only if chunk calls 3+ other functions (orchestration signal)
    if (name.starts_with("run_")
        || name.starts_with("execute_")
        || name == "run"
        || name == "execute")
        && meta.calls.len() >= 3
    {
        return Some(SemanticRole::Orchestrator);
    }

    None
}

fn matches_any(text: &str, patterns: &[&str]) -> bool {
    patterns.iter().any(|p| text.contains(p))
}

// ─────────────────────────────────────────────────────────────────────────────
// E3 — ExCS-style chunk annotation synthesis (no LLM)
// ─────────────────────────────────────────────────────────────────────────────

/// Synthesize an ExCS-style natural-language annotation block for a chunk.
///
/// This is the E3 synthesizer from the semantex v0.3 SOTA spec. It bridges the
/// NL → code vocabulary gap (e.g. a query like *"parallel failure handling"*
/// matching a chunk that uses `Promise.allSettled`) by emitting human-shaped
/// prose that BM25 can score against, without any LLM call.
///
/// The annotation is composed from existing chunk signals:
///
/// 1. **Docstring** (first sentence, if present) — leading authorial intent.
/// 2. **Kind + name** — `function foo`, `struct Bar`, etc., with camelCase /
///    snake_case expansion so identifier-style words land in BM25 as English.
/// 3. **Purpose line** — a one-sentence summary derived from the signature
///    (kind + expanded name + return type if any).
/// 4. **Semantic role label** — e.g. `parallel failure handling`,
///    `validator checker`, drawn from `semantic_role.rs` mappings.
/// 5. **Calls** — what this chunk *does* (filtered to non-trivial calls, then
///    expanded camelCase → English).
/// 6. **Called by** — what *invokes* this chunk (also expanded). Surfacing
///    callers lets NL queries about consumers find providers.
/// 7. **Implements** / **inherits** — type relationships for class-y kinds.
///
/// The output is a multi-line block joined by `\n`. Empty fields are dropped.
///
/// # Parameters
///
/// - `meta`: the chunk's structured metadata (carries docstring, role, etc.).
/// - `calls`: outbound calls — typically `meta.calls`, but accepted separately
///   so callers can post-process (e.g. fold trivial calls).
/// - `called_by`: incoming callers, normally filled by the post-graph-resolve
///   pass — pass an empty slice if not yet resolved at synthesis time.
///
/// # Example
///
/// ```text
/// Parse the input string and return the decoded value.
/// purpose: function parse json — parses, returns Value
/// role: transformer converter serializer parser
/// calls: tokenize, decode value
/// called by: load config, read input
/// ```
#[must_use]
pub fn synthesize_nl_annotation(
    meta: &StructuredChunkMeta,
    calls: &[String],
    called_by: &[String],
) -> String {
    let mut lines: Vec<String> = Vec::with_capacity(8);

    // 1. Docstring (first sentence) — surfaces author-provided intent first.
    if let Some(ref doc) = meta.docstring {
        let trimmed = doc.trim();
        if !trimmed.is_empty() {
            let first = first_sentence(trimmed);
            if !first.is_empty() && first.len() <= 200 {
                lines.push(first.to_string());
            }
        }
    }

    // 2 + 3. Purpose line (kind + expanded name + return type when present).
    if let Some(ref name) = meta.name {
        let kind = meta.kind_label();
        let expanded = expand_identifier(name);
        let mut purpose = format!("purpose: {kind} {expanded}");
        if let Some(ref ret) = meta.return_type {
            let ret_trim = ret.trim();
            if !ret_trim.is_empty() && ret_trim.len() <= 80 {
                purpose.push_str(" — returns ");
                purpose.push_str(ret_trim);
            }
        }
        lines.push(purpose);
    }

    // 4. Semantic role label (e.g. "transformer converter serializer parser").
    if let Some(ref role) = meta.semantic_role {
        lines.push(format!("role: {}", role.as_label()));
    }

    // 5. Outbound calls — filtered to architecturally-meaningful targets,
    // then expanded so camelCase identifiers land as English.
    let interesting_calls: Vec<String> = calls
        .iter()
        .filter(|c| !is_trivial_call(c))
        .take(8)
        .map(|c| expand_identifier(c))
        .collect();
    if !interesting_calls.is_empty() {
        lines.push(format!("calls: {}", interesting_calls.join(", ")));
    }

    // 6. Incoming callers — likewise expanded. Useful for "who consumes X" queries.
    let expanded_callers: Vec<String> = called_by
        .iter()
        .take(8)
        .map(|c| expand_identifier(c))
        .collect();
    if !expanded_callers.is_empty() {
        lines.push(format!("called by: {}", expanded_callers.join(", ")));
    }

    // 7. Type relationships — surfacing impls and inheritance lifts the
    // vocabulary of trait-/interface-level queries.
    if !meta.implements.is_empty() {
        let impls: Vec<String> = meta
            .implements
            .iter()
            .map(|r| {
                format!(
                    "{} implements {}",
                    expand_identifier(&r.implementor),
                    expand_identifier(&r.trait_name)
                )
            })
            .collect();
        lines.push(impls.join("; "));
    }
    if !meta.inherits.is_empty() {
        let parents: Vec<String> = meta.inherits.iter().map(|p| expand_identifier(p)).collect();
        lines.push(format!("inherits: {}", parents.join(", ")));
    }

    lines.join("\n")
}

/// Extract the first sentence from a docstring or comment.
///
/// "Sentence" here is loosely defined: ends at the first `.`, `!`, `?`, or
/// double newline. Common doc-comment leaders (`///`, `//`, `*`) are stripped
/// from the start of each line.
fn first_sentence(s: &str) -> &str {
    let trimmed = strip_doc_leaders(s);
    for (i, ch) in trimmed.char_indices() {
        if matches!(ch, '.' | '!' | '?') {
            return &trimmed[..i];
        }
    }
    // Fall back: stop at the first blank line if no terminator was found.
    if let Some(idx) = trimmed.find("\n\n") {
        return &trimmed[..idx];
    }
    trimmed
}

/// Strip leading doc-comment markers (`///`, `//`, `*`) from a doc string.
/// Only touches the first line — subsequent lines keep their original shape
/// because we never read past the first sentence anyway.
fn strip_doc_leaders(s: &str) -> &str {
    let t = s.trim_start();
    if let Some(rest) = t.strip_prefix("///") {
        return rest.trim_start();
    }
    if let Some(rest) = t.strip_prefix("//") {
        return rest.trim_start();
    }
    if let Some(rest) = t.strip_prefix("/**") {
        return rest.trim_start();
    }
    if let Some(rest) = t.strip_prefix("*") {
        return rest.trim_start();
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta_with_name(name: &str) -> StructuredChunkMeta {
        StructuredChunkMeta {
            name: Some(name.to_string()),
            ..StructuredChunkMeta::default()
        }
    }

    fn meta_with_name_and_calls(name: &str, call_count: usize) -> StructuredChunkMeta {
        let mut meta = meta_with_name(name);
        meta.calls = (0..call_count).map(|i| format!("fn_{i}")).collect();
        meta
    }

    #[test]
    fn test_constructor_patterns() {
        assert_eq!(
            classify_semantic_role(&meta_with_name("new"), ""),
            Some(SemanticRole::Constructor)
        );
        assert_eq!(
            classify_semantic_role(&meta_with_name("create_user"), ""),
            Some(SemanticRole::Constructor)
        );
        assert_eq!(
            classify_semantic_role(&meta_with_name("buildConfig"), ""),
            Some(SemanticRole::Constructor)
        );
        assert_eq!(
            classify_semantic_role(&meta_with_name("from_str"), ""),
            Some(SemanticRole::Constructor)
        );
    }

    #[test]
    fn test_destructor_patterns() {
        assert_eq!(
            classify_semantic_role(&meta_with_name("close"), ""),
            Some(SemanticRole::Destructor)
        );
        assert_eq!(
            classify_semantic_role(&meta_with_name("shutdown"), ""),
            Some(SemanticRole::Destructor)
        );
        assert_eq!(
            classify_semantic_role(&meta_with_name("cleanup"), ""),
            Some(SemanticRole::Destructor)
        );
        assert_eq!(
            classify_semantic_role(&meta_with_name("dispose"), ""),
            Some(SemanticRole::Destructor)
        );
    }

    #[test]
    fn test_sanitizer_patterns() {
        assert_eq!(
            classify_semantic_role(&meta_with_name("sanitize"), ""),
            Some(SemanticRole::Sanitizer)
        );
        assert_eq!(
            classify_semantic_role(&meta_with_name("redact"), ""),
            Some(SemanticRole::Sanitizer)
        );
        assert_eq!(
            classify_semantic_role(&meta_with_name("stripPII"), ""),
            Some(SemanticRole::Sanitizer)
        );
    }

    #[test]
    fn test_orchestrator_needs_calls() {
        assert_eq!(
            classify_semantic_role(&meta_with_name_and_calls("run", 1), ""),
            None
        );
        assert_eq!(
            classify_semantic_role(&meta_with_name_and_calls("run", 3), ""),
            Some(SemanticRole::Orchestrator)
        );
        // pipeline/workflow don't need call count
        assert_eq!(
            classify_semantic_role(&meta_with_name("runPipeline"), ""),
            Some(SemanticRole::Orchestrator)
        );
    }

    #[test]
    fn test_validator_patterns() {
        assert_eq!(
            classify_semantic_role(&meta_with_name("validate_input"), ""),
            Some(SemanticRole::Validator)
        );
        assert_eq!(
            classify_semantic_role(&meta_with_name("check_bounds"), ""),
            Some(SemanticRole::Validator)
        );
        assert_eq!(
            classify_semantic_role(&meta_with_name("ensure_valid"), ""),
            Some(SemanticRole::Validator)
        );
    }

    #[test]
    fn test_transformer_patterns() {
        assert_eq!(
            classify_semantic_role(&meta_with_name("to_string"), ""),
            Some(SemanticRole::Transformer)
        );
        assert_eq!(
            classify_semantic_role(&meta_with_name("parse_json"), ""),
            Some(SemanticRole::Transformer)
        );
        assert_eq!(
            classify_semantic_role(&meta_with_name("serialize"), ""),
            Some(SemanticRole::Transformer)
        );
    }

    #[test]
    fn test_fetcher_patterns() {
        assert_eq!(
            classify_semantic_role(&meta_with_name("get_user"), ""),
            Some(SemanticRole::Fetcher)
        );
        assert_eq!(
            classify_semantic_role(&meta_with_name("load_config"), ""),
            Some(SemanticRole::Fetcher)
        );
        assert_eq!(
            classify_semantic_role(&meta_with_name("fetchData"), ""),
            Some(SemanticRole::Fetcher)
        );
    }

    #[test]
    fn test_persister_patterns() {
        assert_eq!(
            classify_semantic_role(&meta_with_name("save_record"), ""),
            Some(SemanticRole::Persister)
        );
        assert_eq!(
            classify_semantic_role(&meta_with_name("write_to_disk"), ""),
            Some(SemanticRole::Persister)
        );
        assert_eq!(
            classify_semantic_role(&meta_with_name("upsert"), ""),
            Some(SemanticRole::Persister)
        );
    }

    #[test]
    fn test_handler_patterns() {
        assert_eq!(
            classify_semantic_role(&meta_with_name("handleRequest"), ""),
            Some(SemanticRole::Handler)
        );
        assert_eq!(
            classify_semantic_role(&meta_with_name("on_click"), ""),
            Some(SemanticRole::Handler)
        );
        assert_eq!(
            classify_semantic_role(&meta_with_name("process_event"), ""),
            Some(SemanticRole::Handler)
        );
    }

    #[test]
    fn test_error_handler_patterns() {
        assert_eq!(
            classify_semantic_role(&meta_with_name("on_error"), ""),
            Some(SemanticRole::ErrorHandler)
        );
        assert_eq!(
            classify_semantic_role(&meta_with_name("handle_error"), ""),
            Some(SemanticRole::ErrorHandler)
        );
        assert_eq!(
            classify_semantic_role(&meta_with_name("retryRequest"), ""),
            Some(SemanticRole::ErrorHandler)
        );
    }

    #[test]
    fn test_middleware_patterns() {
        assert_eq!(
            classify_semantic_role(&meta_with_name("interceptor"), ""),
            Some(SemanticRole::Middleware)
        );
        assert_eq!(
            classify_semantic_role(&meta_with_name("middleware"), ""),
            Some(SemanticRole::Middleware)
        );
        assert_eq!(
            classify_semantic_role(&meta_with_name("authInterceptor"), ""),
            Some(SemanticRole::Middleware)
        );
    }

    #[test]
    fn test_no_match() {
        assert_eq!(classify_semantic_role(&meta_with_name("foo"), ""), None);
        assert_eq!(classify_semantic_role(&meta_with_name("bar_baz"), ""), None);
    }

    #[test]
    fn test_no_name_returns_none() {
        let meta = StructuredChunkMeta::default();
        assert_eq!(classify_semantic_role(&meta, ""), None);
    }

    // ─────────────────────────────────────────────────────────────────────
    // synthesize_nl_annotation tests
    // ─────────────────────────────────────────────────────────────────────

    use super::super::structured_meta::{ImplRelation, SemanticRole as SR};

    #[test]
    fn test_synthesize_basic_function() {
        let meta = StructuredChunkMeta {
            name: Some("parseJson".to_string()),
            kind: Some("function".to_string()),
            docstring: Some("Parse the input string. Returns the decoded value.".to_string()),
            return_type: Some("Value".to_string()),
            semantic_role: Some(SR::Transformer),
            ..Default::default()
        };
        let calls = vec!["tokenize".to_string(), "decodeValue".to_string()];
        let callers = vec!["loadConfig".to_string()];
        let out = synthesize_nl_annotation(&meta, &calls, &callers);

        assert!(out.contains("Parse the input string"), "{out}");
        assert!(out.contains("purpose: function parse json"), "{out}");
        assert!(out.contains("returns Value"), "{out}");
        assert!(out.contains("role: transformer"), "{out}");
        assert!(out.contains("calls: tokenize, decode value"), "{out}");
        assert!(out.contains("called by: load config"), "{out}");
    }

    #[test]
    fn test_synthesize_class_with_impls() {
        let meta = StructuredChunkMeta {
            name: Some("ConnectionPool".to_string()),
            kind: Some("struct".to_string()),
            implements: vec![ImplRelation {
                implementor: "ConnectionPool".to_string(),
                trait_name: "Drop".to_string(),
            }],
            inherits: vec!["BasePool".to_string()],
            ..Default::default()
        };
        let out = synthesize_nl_annotation(&meta, &[], &[]);
        assert!(out.contains("purpose: struct connection pool"), "{out}");
        assert!(out.contains("connection pool implements drop"), "{out}");
        assert!(out.contains("inherits: base pool"), "{out}");
    }

    #[test]
    fn test_synthesize_module_with_only_docstring() {
        let meta = StructuredChunkMeta {
            name: Some("server".to_string()),
            kind: Some("module".to_string()),
            docstring: Some("TCP daemon server for handling search requests.".to_string()),
            ..Default::default()
        };
        let out = synthesize_nl_annotation(&meta, &[], &[]);
        assert!(
            out.contains("TCP daemon server for handling search requests"),
            "{out}"
        );
        assert!(out.contains("purpose: module server"), "{out}");
    }

    #[test]
    fn test_synthesize_filters_trivial_calls() {
        let meta = StructuredChunkMeta {
            name: Some("collect".to_string()),
            kind: Some("function".to_string()),
            ..Default::default()
        };
        let calls = vec![
            "push".to_string(),   // trivial
            "unwrap".to_string(), // trivial
            "validateInput".to_string(),
        ];
        let out = synthesize_nl_annotation(&meta, &calls, &[]);
        assert!(out.contains("validate input"), "{out}");
        assert!(
            !out.contains("calls: push") && !out.contains(", push"),
            "trivial 'push' should be filtered: {out}"
        );
    }

    #[test]
    fn test_synthesize_empty_meta_returns_empty() {
        let meta = StructuredChunkMeta::default();
        let out = synthesize_nl_annotation(&meta, &[], &[]);
        assert!(
            out.is_empty(),
            "empty meta should yield empty annotation: {out}"
        );
    }

    #[test]
    fn test_synthesize_no_llm_no_async_no_io() {
        // The function is synchronous, allocation-only — proven by the fact
        // that it takes only borrowed references and returns a String built
        // from string concatenation. We verify there are no surprising
        // allocations like network calls by ensuring it runs in <1 ms on
        // typical inputs.
        use std::time::Instant;
        let meta = StructuredChunkMeta {
            name: Some("complexExample".to_string()),
            kind: Some("function".to_string()),
            docstring: Some("Does many things.".to_string()),
            return_type: Some("Result<()>".to_string()),
            semantic_role: Some(SR::Orchestrator),
            ..Default::default()
        };
        let calls: Vec<String> = (0..20).map(|i| format!("doStep{i}")).collect();
        let callers: Vec<String> = (0..20).map(|i| format!("invoker{i}")).collect();

        let start = Instant::now();
        for _ in 0..100 {
            let _ = synthesize_nl_annotation(&meta, &calls, &callers);
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 50,
            "100 syntheses should be sub-50ms (no LLM, no IO); got {}ms",
            elapsed.as_millis()
        );
    }

    #[test]
    fn test_first_sentence_strips_doc_leader() {
        assert_eq!(first_sentence("/// Hello there. More text."), "Hello there");
        assert_eq!(first_sentence("// short comment"), "short comment");
        assert_eq!(first_sentence("Normal text. tail"), "Normal text");
    }

    #[test]
    fn test_first_sentence_long_docstring_truncates() {
        let s = "First sentence ends here. Second sentence.";
        assert_eq!(first_sentence(s), "First sentence ends here");
    }
}
