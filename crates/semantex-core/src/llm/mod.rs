// async-trait dep added by W-LLM-Genai in Cargo.toml under [features].llm
#![cfg(feature = "llm")]

mod genai_backend;
mod prompts;
// `Antigravity` variant is a placeholder pending Team 1 removal; suppress
// the dead-code lint here to avoid failing `-D warnings` in CI.
#[allow(dead_code)]
mod subscription_cli;

pub(crate) use genai_backend::GenaiBackend;
pub(crate) use prompts::{
    CLASSIFIER_SYSTEM_PROMPT, HYDE_SYSTEM_PROMPT, build_classify_prompt, build_hyde_prompt,
    parse_route_from_llm_output,
};
pub(crate) use subscription_cli::SubscriptionCliBackend;

use crate::search::agent_classifier::AgentRoute;

/// What we ask an LLM to do.
///
/// Implemented by every backend (`GenaiBackend`, future `SubscriptionCliBackend`).
/// Held as `Arc<dyn LlmCapability>` across the daemon so backends are
/// swappable without touching call sites.
#[async_trait::async_trait]
pub trait LlmCapability: Send + Sync {
    /// Pick the best [`AgentRoute`] for `query`.
    ///
    /// Used as an optional pre-classifier; the caller falls back to the
    /// keyword classifier on `Err` (including timeouts).
    async fn classify_route(&self, query: &str) -> anyhow::Result<AgentRoute>;

    /// Generate a hypothetical code snippet that would answer `query`.
    ///
    /// Used by the HyDE retrieval channel; the caller falls back to the base
    /// search on `Err`.
    async fn synthesize_hyde_doc(&self, query: &str) -> anyhow::Result<String>;

    /// Human-readable backend name for the startup banner and logs.
    ///
    /// Examples: `"genai/anthropic/claude-sonnet-4-6"`, `"cli:claude"`.
    fn label(&self) -> &str;
}

/// Concrete dispatcher.
///
/// Constructed once at daemon startup and held in `Arc<dyn LlmCapability>`
/// form across the rest of the codebase.
///
/// The inner types (`GenaiBackend`, `SubscriptionCliBackend`) are `pub(crate)`
/// because no external caller needs to name them directly; the public API is
/// `from_env` / `into_arc` / `as_capability`. Pattern-matching on `Genai(b)`
/// still works in tests and integration code without naming the type.
#[allow(private_interfaces)]
pub enum LlmBackend {
    Genai(GenaiBackend),
    SubscriptionCli(SubscriptionCliBackend),
}

impl LlmBackend {
    /// Detect from env vars. Returns `Ok(None)` if no LLM is configured.
    ///
    /// Probe order:
    /// 1. `GenaiBackend` — explicit model via `SEMANTEX_LLM_MODEL` (Phase 1).
    /// 2. `SubscriptionCliBackend` — `SEMANTEX_LLM_BACKEND=cli:*` (Phase 2
    ///    subscription-quota fallback).
    pub fn from_env() -> anyhow::Result<Option<Self>> {
        if let Some(b) = GenaiBackend::from_env()? {
            return Ok(Some(Self::Genai(b)));
        }
        if let Some(b) = SubscriptionCliBackend::from_env()? {
            return Ok(Some(Self::SubscriptionCli(b)));
        }
        Ok(None)
    }

    /// Borrow as a trait object.
    pub fn as_capability(&self) -> &dyn LlmCapability {
        match self {
            Self::Genai(b) => b,
            Self::SubscriptionCli(b) => b,
        }
    }

    /// Consume into a shareable `Arc<dyn LlmCapability>`.
    ///
    /// Used by daemon / MCP / CLI startup paths that need to clone the
    /// backend across requests. Replaces the per-call `match { Genai => Arc,
    /// SubscriptionCli => Arc }` pattern that would otherwise drift if a new
    /// variant is added.
    #[must_use]
    pub fn into_arc(self) -> std::sync::Arc<dyn LlmCapability> {
        match self {
            Self::Genai(b) => std::sync::Arc::new(b),
            Self::SubscriptionCli(b) => std::sync::Arc::new(b),
        }
    }
}

/// Cross-module env-mutation lock for tests under this crate.
///
/// `std::env::set_var` / `remove_var` became `unsafe` in Rust 2024 because
/// process env is global mutable state. Cargo runs all `#[test]` functions
/// from a single crate in the same process and may parallelise them, so any
/// test that flips `SEMANTEX_LLM_*` must serialise. `genai_backend.rs` and
/// `subscription_cli.rs` both mutate overlapping vars — a per-file lock
/// isn't enough.
///
/// **Variables guarded by this lock**: `SEMANTEX_LLM_MODEL`, `SEMANTEX_LLM_PROVIDER`,
/// `SEMANTEX_LLM_ENDPOINT`, `SEMANTEX_LLM_BACKEND`.
///
/// **Independent env-mutation tests**: `memory::tests::env_cap_behaviour_serial`
/// mutates `SEMANTEX_MAX_RSS_MB` and does NOT hold this lock — the two variable
/// families are disjoint so they can coexist safely.  If any future test ever
/// needs to mutate BOTH `SEMANTEX_LLM_*` vars AND `SEMANTEX_MAX_RSS_MB` in the
/// same test function, it MUST hold `TEST_ENV_LOCK` for the whole duration to
/// prevent races with the LLM tests.
#[cfg(test)]
pub(crate) static TEST_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
