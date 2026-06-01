//! Operator-gated end-to-end smoke test for the LLM integration (Spec L §6.4).
//!
//! Runs only when `SEMANTEX_LLM_TEST_OLLAMA=1` or `SEMANTEX_LLM_TEST_CLI=<kind>`
//! is set in the environment. Default: skipped silently so CI passes on
//! machines without LLM credentials.
//!
//! To run against a local Ollama instance, set:
//!   `SEMANTEX_LLM_TEST_OLLAMA=1 SEMANTEX_LLM_MODEL=qwen2.5-coder:7b`
//!   (provider is inferred from the model-name prefix; the cloud prefixes
//!    `claude-`/`gpt-`/`gemini-` route to API-key providers, model names
//!    starting with `qwen`/`llama`/`mistral`/`phi`/etc. default to Ollama.)
//!   then: `cargo test --test llm_smoke --features semantex-core/llm`
//!
//! To run against the Claude CLI, set:
//!   `SEMANTEX_LLM_TEST_CLI=claude SEMANTEX_LLM_BACKEND=cli:claude`
//!   then: `cargo test --test llm_smoke --features semantex-core/llm`

#![cfg(feature = "llm")]

use semantex_core::llm::LlmBackend;

/// End-to-end probe: classify a structural query and synthesize a HyDE doc.
///
/// The classify assertion is intentionally loose — we accept any valid route
/// in the Structural/ExactSymbol neighbourhood because the LLM may pick either.
/// The HyDE assertion checks only that the response mentions the key concept.
#[tokio::test]
async fn smoke_classify_and_hyde() {
    let ollama_enabled = std::env::var("SEMANTEX_LLM_TEST_OLLAMA").is_ok();
    let cli_enabled = std::env::var("SEMANTEX_LLM_TEST_CLI").is_ok();

    if !ollama_enabled && !cli_enabled {
        eprintln!(
            "smoke_classify_and_hyde: skipped \
             (set SEMANTEX_LLM_TEST_OLLAMA=1 or SEMANTEX_LLM_TEST_CLI=<kind> to run)"
        );
        return;
    }

    let backend = LlmBackend::from_env()
        .expect("LlmBackend::from_env() failed")
        .expect("expected Some(backend) — set SEMANTEX_LLM_MODEL or SEMANTEX_LLM_BACKEND");

    // The inner backend types are `pub(crate)`; external callers must use the
    // public `into_arc()` accessor rather than pattern-matching the variants
    // (see `LlmBackend` doc comment). Label first via the trait object.
    let cap: std::sync::Arc<dyn semantex_core::llm::LlmCapability> = backend.into_arc();
    eprintln!("Smoke test against backend: {}", cap.label());

    // ── classify ─────────────────────────────────────────────────────────────
    let route = cap
        .classify_route("who calls handle_request?")
        .await
        .expect("classify_route failed");

    use semantex_core::search::agent_classifier::AgentRoute;
    assert!(
        matches!(
            route,
            AgentRoute::Structural | AgentRoute::ExactSymbol | AgentRoute::Deep
        ),
        "expected Structural/ExactSymbol/Deep for a callers query, got {route:?}"
    );
    eprintln!("classify_route result: {route:?} ✓");

    // ── synthesize_hyde_doc ──────────────────────────────────────────────────
    let hyde = cap
        .synthesize_hyde_doc("code that handles timeouts")
        .await
        .expect("synthesize_hyde_doc failed");

    assert!(!hyde.trim().is_empty(), "HyDE response must not be empty");

    let hyde_lower = hyde.to_lowercase();
    assert!(
        hyde_lower.contains("timeout")
            || hyde_lower.contains("duration")
            || hyde_lower.contains("deadline"),
        "HyDE response should mention timeout/duration/deadline; got:\n{hyde}"
    );
    eprintln!("synthesize_hyde_doc returned {} chars ✓", hyde.len());
}
