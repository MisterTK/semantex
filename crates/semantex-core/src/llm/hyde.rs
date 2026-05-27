// crates/semantex-core/src/llm/hyde.rs
// `LlmHyDE` impl for `OnnxLlm` (v0.6 Item 9, behind `--features local-llm`).
//
// HyDE = Hypothetical Document Embeddings. The flow is:
//   1. The LLM is asked to synthesize a short code-shaped document that
//      *would* answer the user's natural-language query.
//   2. That synthesized doc is then passed through the existing
//      ColBERT/BM25 retrieval pipeline as if it were the query.
//   3. The synthesized doc's matches are fused with the original query's
//      matches; this closes the NL→code gap on queries like "find code
//      that handles timeouts" which contain zero literal token overlap
//      with `tokio::time::Duration` call sites.
//
// As with `classifier.rs`, the prompt-tokenize-decode pipeline is stubbed
// here so the default `--features local-llm` build can compile and run
// tests without a bundled model. The trait impl returns `Err` so the
// caller falls back to the original-query-only search path.

use anyhow::Result;

use crate::llm::LlmHyDE;
use crate::llm::loader::OnnxLlm;

impl LlmHyDE for OnnxLlm {
    fn synthesize_doc(&self, _query: &str) -> Result<String> {
        // Scaffold: real implementation deferred. Returning an error keeps
        // the HyDE path off; `hybrid_search_with_hyde` will short-circuit
        // to a plain search.
        Err(anyhow::anyhow!(
            "local-llm HyDE synthesizer not yet wired to model output (path={})",
            self.model_path.display()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// MockLlmHyDE that returns a fixed document, with call counting so
    /// tests can assert the synthesis path was actually invoked. Uses
    /// `AtomicUsize` for the counter so the trait's `Sync` bound holds.
    pub(crate) struct MockLlmHyDE {
        pub doc: Result<String>,
        pub calls: AtomicUsize,
    }
    impl MockLlmHyDE {
        pub fn ok(doc: impl Into<String>) -> Self {
            Self {
                doc: Ok(doc.into()),
                calls: AtomicUsize::new(0),
            }
        }
        pub fn err() -> Self {
            Self {
                doc: Err(anyhow::anyhow!("synthetic HyDE failure")),
                calls: AtomicUsize::new(0),
            }
        }
        pub fn call_count(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }
    impl LlmHyDE for MockLlmHyDE {
        fn synthesize_doc(&self, _query: &str) -> Result<String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            match &self.doc {
                Ok(d) => Ok(d.clone()),
                Err(_) => Err(anyhow::anyhow!("synthetic HyDE failure")),
            }
        }
    }

    #[test]
    fn mock_returns_doc() {
        let m = MockLlmHyDE::ok("fn handle_timeout(d: Duration) {}");
        let doc = m.synthesize_doc("find timeout handling").unwrap();
        assert!(doc.contains("Duration"));
        assert_eq!(m.call_count(), 1);
    }

    #[test]
    fn mock_propagates_err() {
        let m = MockLlmHyDE::err();
        assert!(m.synthesize_doc("anything").is_err());
        assert_eq!(m.call_count(), 1);
    }

    #[test]
    fn onnx_hyde_returns_err_in_scaffold() {
        let llm = OnnxLlm {
            model_path: std::path::PathBuf::from("/tmp/does-not-matter.onnx"),
        };
        assert!(llm.synthesize_doc("anything").is_err());
    }
}
