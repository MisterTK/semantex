// crates/semantex-core/src/llm/classifier.rs
// `LlmClassifier` impl for `OnnxLlm` (v0.6 Item 9, behind `--features local-llm`).
//
// The full classify-prompt → tokenize → forward → argmax pipeline is a
// stub in this scaffold iteration: the impl returns an `Err` whose
// message documents that the LLM-classifier path is not yet wired to
// model output. `classify_with_llm` interprets the error as "fall back to
// the keyword classifier", so the visible behaviour of the daemon is
// unchanged regardless of whether a model is loaded.
//
// When a real model artifact arrives, the body of `classify` becomes:
//   1. Build a few-shot prompt with the AgentRoute schema.
//   2. Tokenize via the model's bundled tokenizer.
//   3. Run a single `ort::Session::run` and decode the highest-probability
//      label that maps to an `AgentRoute` variant.
//   4. Return the variant, or `Err` if decoding fails.
// The trait stays sync because step 3 is a blocking ORT call.

use anyhow::Result;

use crate::llm::LlmClassifier;
use crate::llm::loader::OnnxLlm;
use crate::search::agent_classifier::AgentRoute;

impl LlmClassifier for OnnxLlm {
    fn classify(&self, _query: &str) -> Result<AgentRoute> {
        // Scaffold: real implementation deferred. Returning an error keeps
        // `classify_with_llm` on its fallback path, which is the desired
        // behaviour while a model artifact is not yet bundled.
        Err(anyhow::anyhow!(
            "local-llm classifier not yet wired to model output (path={})",
            self.model_path.display()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// MockLlmClassifier used by the cross-module tests: lets the test
    /// caller dictate the route or signal an error. Uses `AtomicUsize`
    /// for the call counter so the trait's `Sync` bound is satisfied.
    pub(crate) struct MockLlmClassifier {
        pub route: Result<AgentRoute>,
        pub calls: AtomicUsize,
    }
    impl MockLlmClassifier {
        pub fn ok(route: AgentRoute) -> Self {
            Self {
                route: Ok(route),
                calls: AtomicUsize::new(0),
            }
        }
        pub fn err() -> Self {
            Self {
                route: Err(anyhow::anyhow!("synthetic LLM failure")),
                calls: AtomicUsize::new(0),
            }
        }
        pub fn call_count(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }
    impl LlmClassifier for MockLlmClassifier {
        fn classify(&self, _query: &str) -> Result<AgentRoute> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            // anyhow::Error isn't Clone, so re-create the same shape per call.
            match &self.route {
                Ok(r) => Ok(*r),
                Err(_) => Err(anyhow::anyhow!("synthetic LLM failure")),
            }
        }
    }

    #[test]
    fn mock_returns_feature_planning_when_configured() {
        let m = MockLlmClassifier::ok(AgentRoute::FeaturePlanning);
        assert_eq!(m.classify("anything").unwrap(), AgentRoute::FeaturePlanning);
        assert_eq!(m.call_count(), 1);
    }

    #[test]
    fn classify_with_llm_uses_mock_route() {
        let m = MockLlmClassifier::ok(AgentRoute::FeaturePlanning);
        // Query that would otherwise route to Semantic.
        let route =
            crate::search::agent_classifier::classify_with_llm("authentication middleware", &m);
        assert_eq!(route, AgentRoute::FeaturePlanning);
    }

    #[test]
    fn classify_with_llm_falls_back_when_mock_errs() {
        let m = MockLlmClassifier::err();
        // Query the keyword classifier routes to Deep.
        let route =
            crate::search::agent_classifier::classify_with_llm("how does authentication work?", &m);
        assert_eq!(route, AgentRoute::Deep);
        // Mock was still consulted even though it failed.
        assert_eq!(m.call_count(), 1);
    }

    #[test]
    fn onnx_classifier_returns_err_in_scaffold() {
        // The real OnnxLlm impl always errors in the scaffold; this test
        // pins that behaviour so the fallback path stays exercised until
        // the real pipeline lands.
        let llm = OnnxLlm {
            model_path: std::path::PathBuf::from("/tmp/does-not-matter.onnx"),
        };
        assert!(llm.classify("anything").is_err());
    }
}
