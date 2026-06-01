pub mod adaptive;
pub mod agent;
pub mod agent_classifier;
pub mod agent_formatter;
pub mod code_tokenizer;
pub mod colbert_plaid_backend;
pub mod deep;
pub mod dense_backend;
pub mod fastembed_reranker;
pub mod graph_propagation;
pub mod hybrid;
pub mod mmr;
pub mod onnx_reranker;
pub mod path_signals;
pub mod plaid_search;
pub mod planner;
pub mod query_classifier;
pub mod query_expander;
pub mod regex_semantic;
pub mod reranker_download;
pub mod reranker_engine;
pub mod reranker_model;
pub mod ripgrep_fallback;
pub mod simd;
pub mod sparse_search;
pub mod summarize;
pub mod triple_fusion;

use crate::types::{FileFilter, SearchResult};
use anyhow::Result;

/// One process-wide test mutex serializing every `with_env` helper that mutates
/// the shared `SEMANTEX_RERANKER` / `SEMANTEX_RERANKER_MODEL` env vars across the
/// reranker test modules (`fastembed_reranker`, `onnx_reranker`, `reranker_model`,
/// `reranker_engine`). Per-module `static`s do NOT serialize each other within
/// one test binary, which made env-mutating reranker tests flaky; all four helpers
/// lock THIS one. (Mirrors the `crate::llm::TEST_ENV_LOCK` pattern, but lives here
/// so it is available on the default, no-`llm` build.)
#[cfg(test)]
pub(crate) static RERANKER_TEST_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
use serde::{Deserialize, Serialize};

/// Measured performance metrics from a single search invocation.
///
/// NOTE: `skip_serializing_if` is intentionally omitted from Optional fields.
/// This struct is serialized via postcard (positional encoding) for the binary
/// daemon protocol. Skipping fields in positional encoding causes field misalignment
/// during deserialization. `#[serde(default)]` is kept for JSON compatibility
/// (fills in `null` fields when deserializing older responses).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchMetrics {
    pub total_ms: u64,
    #[serde(default)]
    pub dense_ms: Option<u64>,
    #[serde(default)]
    pub sparse_ms: Option<u64>,
    #[serde(default)]
    pub exact_ms: Option<u64>,
    #[serde(default)]
    pub fusion_ms: Option<u64>,
    #[serde(default)]
    pub rerank_ms: Option<u64>,
    pub dense_count: usize,
    pub sparse_count: usize,
    pub exact_count: usize,
    pub fused_count: usize,
    pub result_count: usize,
    pub query_type: String,
    #[serde(default)]
    pub response_bytes: Option<usize>,
}

/// Search results bundled with performance metrics.
pub struct SearchOutput {
    pub results: Vec<SearchResult>,
    pub metrics: SearchMetrics,
}

/// Search configuration for a single query
#[derive(Debug, Clone)]
pub struct SearchQuery {
    pub text: String,
    pub max_results: usize,
    pub use_dense: bool,
    pub use_sparse: bool,
    pub use_rerank: bool,
    pub file_filter: Option<FileFilter>,
    /// Grep parity mode: exact+sparse only, no dense, no rerank, permissive threshold
    pub grep_mode: bool,
    /// Optional regex pattern for hybrid regex+semantic mode (-e flag).
    pub regex_pattern: Option<String>,
}

impl SearchQuery {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            max_results: 10,
            use_dense: true,
            use_sparse: true,
            use_rerank: true,
            file_filter: None,
            grep_mode: false,
            regex_pattern: None,
        }
    }

    pub fn max_results(mut self, n: usize) -> Self {
        self.max_results = n;
        self
    }

    pub fn dense_only(mut self) -> Self {
        self.use_sparse = false;
        self.use_rerank = false;
        self
    }

    pub fn sparse_only(mut self) -> Self {
        self.use_dense = false;
        self.use_rerank = false;
        self
    }

    pub fn no_rerank(mut self) -> Self {
        self.use_rerank = false;
        self
    }

    /// Enable grep parity mode: exact+sparse only, exhaustive, no reranking.
    pub fn grep_mode(mut self) -> Self {
        self.grep_mode = true;
        self.use_dense = false;
        self.use_sparse = true;
        self.use_rerank = false;
        self.max_results = 50;
        self
    }

    pub fn regex_pattern(mut self, pattern: Option<String>) -> Self {
        self.regex_pattern = pattern;
        self
    }

    pub fn include_types(mut self, extensions: Vec<String>) -> Self {
        let filter = self.file_filter.get_or_insert_with(FileFilter::default);
        filter.include_extensions = extensions;
        self
    }

    pub fn exclude_types(mut self, extensions: Vec<String>) -> Self {
        let filter = self.file_filter.get_or_insert_with(FileFilter::default);
        filter.exclude_extensions = extensions;
        self
    }

    pub fn code_only(self) -> Self {
        self.exclude_types(
            FileFilter::NON_CODE_EXTENSIONS
                .iter()
                .map(std::string::ToString::to_string)
                .collect(),
        )
    }
}

/// Trait for search backends
pub trait Searcher: Send + Sync {
    fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>>;
}
