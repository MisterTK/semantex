//! `semantex llm-status` — show LLM backend configuration and run a health-
//! check classify call.
//!
//! Only compiled when built with `--features llm`.

use anyhow::Result;
use semantex_core::llm::LlmBackend;

const PROBE_QUERY: &str = "who calls handle_request?";

/// Default health-check timeout. Sized for cli:claude / cli:codex which
/// include subprocess spawn cost (~3-5 s for the LLM call alone). Matches
/// the daemon's classifier budget; overridable via
/// `SEMANTEX_LLM_CLASSIFY_TIMEOUT_MS` so Ollama/local users can dial down.
const HEALTH_CHECK_DEFAULT_MS: u64 = 8_000;

fn health_check_timeout() -> std::time::Duration {
    std::env::var("SEMANTEX_LLM_CLASSIFY_TIMEOUT_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .map_or_else(
            || std::time::Duration::from_millis(HEALTH_CHECK_DEFAULT_MS),
            std::time::Duration::from_millis,
        )
}

pub fn run() -> Result<()> {
    match LlmBackend::from_env()? {
        None => {
            println!("LLM backend: not configured");
            println!("Set SEMANTEX_LLM_MODEL or SEMANTEX_LLM_BACKEND=cli:<kind> to enable.");
            return Ok(());
        }
        Some(backend) => {
            let cap = backend.into_arc();
            println!("LLM backend:  {}", cap.label());

            // Run a 1-token health-check classify call with a timeout.
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            let timeout = health_check_timeout();
            let result = rt.block_on(async {
                tokio::time::timeout(timeout, cap.classify_route(PROBE_QUERY)).await
            });

            match result {
                Ok(Ok(route)) => {
                    println!("Health check: OK (route = {route})");
                }
                Ok(Err(e)) => {
                    println!("Health check: FAILED — {e}");
                }
                Err(_) => {
                    println!("Health check: TIMED OUT (>{}ms)", timeout.as_millis());
                }
            }
        }
    }
    Ok(())
}
