// crates/semantex-core/src/llm/mod.rs
// Local-LLM scaffold (v0.6 Item 9).
//
// This module exists so the rest of the crate can be written against
// trait objects regardless of whether a real LLM backend is compiled in.
// The traits live at the module root (always present); the concrete
// `OnnxLlm` loader + classifier + HyDE synthesizer live in feature-gated
// submodules.
//
// Design constraints:
//   - Default `cargo build` MUST stay green and MUST NOT pull in any new
//     runtime dependency. The traits below carry only types already
//     reachable through the existing dep graph (`anyhow`, `AgentRoute`).
//   - `--features local-llm` opts into the ONNX-backed implementation.
//     No model file is bundled in either build — `OnnxLlm::load` reads
//     `SEMANTEX_LLM_PATH` at runtime and returns `Ok(None)` if the file
//     is missing.
//   - Both traits are intentionally synchronous: ONNX inference is a
//     blocking call and the rest of `semantex-core` is sync (no tokio
//     runtime in the default dep graph). Promoting these to `async fn`
//     would force a runtime into the default build for no benefit; if a
//     future caller needs to overlap inference with other I/O, the
//     methods can be wrapped in `spawn_blocking` at that callsite.
//
// Spec reference: docs/superpowers/specs/2026-05-26-semantex-v0.3.1-v0.5-refactor.md
// §6 Item 9 + §9 R3 (risk: install size > 1.5 GB → opt-in feature flag).

use anyhow::Result;

use crate::search::agent_classifier::AgentRoute;

#[cfg(feature = "local-llm")]
pub mod classifier;
#[cfg(feature = "local-llm")]
pub mod hyde;
#[cfg(feature = "local-llm")]
pub mod loader;

/// Classify a natural-language query into an `AgentRoute` using a local
/// LLM. Implementations should be cheap to call (single forward pass on a
/// quantized small model) and MUST be safe to call concurrently from
/// multiple search requests — the daemon shares a single classifier
/// instance across connections.
///
/// Errors are non-fatal: the wrapping `classify_with_llm` helper falls
/// back to the deterministic keyword classifier on any `Err`.
pub trait LlmClassifier: Send + Sync {
    fn classify(&self, query: &str) -> Result<AgentRoute>;
}

/// Synthesize a hypothetical document that would answer `query`, used by
/// the HyDE (Hypothetical Document Embeddings) retrieval channel.
///
/// The output is fed through the existing ColBERT/BM25 pipeline as if it
/// were the user's query — so it should be code-shaped prose, not chat.
/// Implementations MUST be safe for concurrent use; same rationale as
/// `LlmClassifier`.
pub trait LlmHyDE: Send + Sync {
    fn synthesize_doc(&self, query: &str) -> Result<String>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock used by the agent_classifier tests too. Keeping a copy here
    /// proves the trait is shaped right for downstream consumers.
    struct AlwaysSemantic;
    impl LlmClassifier for AlwaysSemantic {
        fn classify(&self, _query: &str) -> Result<AgentRoute> {
            Ok(AgentRoute::Semantic)
        }
    }
    struct AlwaysDoc;
    impl LlmHyDE for AlwaysDoc {
        fn synthesize_doc(&self, _query: &str) -> Result<String> {
            Ok("synthetic hypothetical document".into())
        }
    }

    #[test]
    fn classifier_trait_is_object_safe() {
        let c: Box<dyn LlmClassifier> = Box::new(AlwaysSemantic);
        assert_eq!(c.classify("x").unwrap(), AgentRoute::Semantic);
    }

    #[test]
    fn hyde_trait_is_object_safe() {
        let h: Box<dyn LlmHyDE> = Box::new(AlwaysDoc);
        assert!(!h.synthesize_doc("x").unwrap().is_empty());
    }
}
