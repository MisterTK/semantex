//! Rule-based semantic role classification for code chunks.
//!
//! Classifies functions/methods by their behavioral purpose (constructor, validator,
//! transformer, etc.) using name patterns and structural signals.

use super::structured_meta::{SemanticRole, StructuredChunkMeta, expand_identifier};

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
}
