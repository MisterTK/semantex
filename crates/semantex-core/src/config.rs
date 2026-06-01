use crate::search::adaptive::AdaptiveConfig;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// ColBERT model selection.
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ColbertModelChoice {
    /// LateOn-Code-edge: 48d per-token, INT8 quantized, ~17MB
    #[default]
    LateOnCodeEdge,
}

/// Global semantex configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SemantexConfig {
    /// Maximum number of search results
    pub max_count: usize,
    /// Show content snippets in output
    pub content: bool,
    /// Lines of context around matches
    pub context_lines: usize,
    /// Enable cross-encoder reranking
    pub rerank: bool,
    /// Maximum file size to index (bytes)
    pub max_file_size: u64,
    /// Maximum number of files to index
    pub max_file_count: usize,
    /// Text chunk size in tokens
    pub chunk_size: usize,
    /// Chunk overlap in tokens
    pub chunk_overlap: usize,
    /// RRF fusion constant k
    pub rrf_k: f32,
    /// Number of candidates to fetch before reranking
    pub rerank_candidates: usize,
    /// Custom model directory override
    pub model_dir: Option<PathBuf>,
    /// Enable adaptive result sizing (dynamically adjust result count based on score distribution)
    pub adaptive_sizing: bool,
    /// Enable per-file deduplication (keep only top chunk per file unless scores are close)
    pub adaptive_dedup: bool,
    /// Score gap threshold for keeping multiple chunks from same file (0.0-1.0)
    pub adaptive_dedup_gap: f32,
    /// Minimum confidence score for Identifier queries (relative to top score, 0.0-1.0)
    pub min_score_identifier: f32,
    /// Minimum confidence score for Keyword queries (relative to top score, 0.0-1.0)
    pub min_score_keyword: f32,
    /// Minimum confidence score for Semantic queries (relative to top score, 0.0-1.0)
    pub min_score_semantic: f32,
    /// Minimum confidence score for Mixed queries (relative to top score, 0.0-1.0)
    pub min_score_mixed: f32,
    /// ColBERT model choice
    pub colbert_model: ColbertModelChoice,
    /// PLAID quantization bits (2 or 4, default: 4)
    pub plaid_nbits: usize,
    /// Whether to apply the English Snowball stemmer to BM25 index tokens.
    /// Default `true` (legacy behavior). Code-identifier-heavy corpora MAY
    /// benefit from disabling: with stemming on, `retry` -> `retri`,
    /// `handle` -> `handl`, which can hurt exact identifier matching.
    ///
    /// MUST match the value the index was built with. As of v0.4.1 W-Index #4
    /// the indexer persists this flag in `meta.json` and `SparseIndex::open`
    /// re-validates it against the runtime config — a mismatch is a hard
    /// error at startup, not a silent recall regression. Run
    /// `semantex index --rebuild` after toggling.
    pub use_bm25_stemmer: bool,
    /// Selected dense search backend identity (e.g. `"colbert-plaid"`).
    /// MUST match the value the index was built with — `HybridSearcher::open`
    /// re-validates it against the persisted `IndexMeta.dense_backend` and
    /// refuses to load on mismatch (mirrors `use_bm25_stemmer`). Override via
    /// `SEMANTEX_DENSE_BACKEND`. Default `"colbert-plaid"`.
    pub dense_backend: String,
    /// Active embedder model id (registry lookup key). Default `"lateon-colbert"`
    /// (D4: PLAID is the shipped dense default until the Phase-3 cutover flips
    /// this to `"coderank-137m"`). The embedder spec id is model-descriptive and
    /// distinct from the dense backend name it routes to via capabilities
    /// (`lateon-colbert` → `colbert-plaid`; `coderank-137m` → `coderank-hnsw`).
    /// Override via `SEMANTEX_EMBEDDER`. A change here triggers a versioned dense
    /// rebuild + atomic switchover (S8) — the re-embedding compute is inherent.
    pub embedder: String,
    /// Active reranker model id (registry lookup key). Default
    /// `"bge-reranker-v2-m3"`. Override via `SEMANTEX_RERANKER_MODEL`. A change
    /// here is a query-time live swap — no reindex.
    pub reranker_model: String,
    /// Active LLM model id (registry lookup key). Empty = none. Override via
    /// `SEMANTEX_LLM_MODEL`. Only meaningful with the `llm` feature.
    pub llm_model: String,
}

impl Default for SemantexConfig {
    fn default() -> Self {
        Self {
            max_count: 15,
            content: false,
            context_lines: 0,
            rerank: false,
            max_file_size: 1_048_576, // 1MB
            max_file_count: 50_000,
            chunk_size: 512,
            chunk_overlap: 128,
            rrf_k: 30.0,
            rerank_candidates: 100,
            model_dir: None,
            // Adaptive result sizing
            adaptive_sizing: true,
            adaptive_dedup: true,
            adaptive_dedup_gap: 0.10,
            min_score_identifier: 0.08,
            min_score_keyword: 0.15,
            min_score_semantic: 0.10,
            min_score_mixed: 0.10,
            colbert_model: ColbertModelChoice::default(),
            plaid_nbits: 4,
            use_bm25_stemmer: true,
            dense_backend: "colbert-plaid".to_string(),
            embedder: "lateon-colbert".to_string(),
            reranker_model: "bge-reranker-v2-m3".to_string(),
            llm_model: String::new(),
        }
    }
}

impl SemantexConfig {
    /// Load config from file, falling back to defaults
    pub fn load(project_path: Option<&Path>) -> Result<Self> {
        let mut config = Self::default();

        // Load global config
        let global_config = Self::global_config_path();
        if global_config.exists() {
            let content = std::fs::read_to_string(&global_config)
                .with_context(|| format!("Failed to read {}", global_config.display()))?;
            config = serde_yml::from_str(&content)
                .with_context(|| format!("Failed to parse {}", global_config.display()))?;
        }

        // Overlay project config
        if let Some(project) = project_path {
            let project_config = project.join(".semantexrc.yaml");
            if project_config.exists() {
                let content = std::fs::read_to_string(&project_config)
                    .with_context(|| format!("Failed to read {}", project_config.display()))?;
                let project_cfg: SemantexConfig = serde_yml::from_str(&content)
                    .with_context(|| format!("Failed to parse {}", project_config.display()))?;
                config = project_cfg;
            }
        }

        // Environment variable overrides
        if let Ok(v) = std::env::var("SEMANTEX_MAX_COUNT") {
            config.max_count = v.parse().unwrap_or(config.max_count);
        }
        if let Ok(v) = std::env::var("SEMANTEX_CONTENT") {
            config.content = v == "1" || v.to_lowercase() == "true";
        }
        if let Ok(v) = std::env::var("SEMANTEX_RERANK") {
            config.rerank = v != "0" && v.to_lowercase() != "false";
        }
        if let Ok(v) = std::env::var("SEMANTEX_MAX_FILE_SIZE") {
            config.max_file_size = v.parse().unwrap_or(config.max_file_size);
        }
        if let Ok(v) = std::env::var("SEMANTEX_MODEL_DIR") {
            config.model_dir = Some(PathBuf::from(v));
        }
        if let Ok(v) = std::env::var("SEMANTEX_DENSE_BACKEND") {
            let trimmed = v.trim();
            if !trimmed.is_empty() {
                config.dense_backend = trimmed.to_string();
            }
        }
        // S8: registry selection keys. `SEMANTEX_RERANKER_MODEL` is the same
        // selection key S3 already reads — the registry now reads it via config.
        config.embedder = env_string("SEMANTEX_EMBEDDER", &config.embedder);
        config.reranker_model = env_string("SEMANTEX_RERANKER_MODEL", &config.reranker_model);
        config.llm_model = env_string("SEMANTEX_LLM_MODEL", &config.llm_model);

        Ok(config)
    }

    /// Default semantex home directory (cross-platform)
    pub fn semantex_home() -> PathBuf {
        if let Ok(val) = std::env::var("SEMANTEX_HOME") {
            return PathBuf::from(val);
        }
        dirs::home_dir().map_or_else(
            || std::env::temp_dir().join("semantex"),
            |h| h.join(".semantex"),
        )
    }

    /// Models directory
    pub fn models_dir(&self) -> PathBuf {
        self.model_dir
            .clone()
            .unwrap_or_else(|| Self::semantex_home().join("models"))
    }

    /// Compute project index directory: `<project>/.semantex/`
    pub fn project_index_dir(project_path: &Path) -> PathBuf {
        project_path.join(".semantex")
    }

    /// Build an AdaptiveConfig from the current settings
    pub fn adaptive_config(&self) -> AdaptiveConfig {
        AdaptiveConfig {
            enabled: self.adaptive_sizing,
            min_score_identifier: self.min_score_identifier,
            min_score_keyword: self.min_score_keyword,
            min_score_semantic: self.min_score_semantic,
            min_score_mixed: self.min_score_mixed,
            deduplicate: self.adaptive_dedup,
            dedup_score_gap: self.adaptive_dedup_gap,
            exhaustive: false, // set per-query in hybrid.rs
        }
    }

    /// Global config file path (cross-platform)
    fn global_config_path() -> PathBuf {
        if let Ok(val) = std::env::var("XDG_CONFIG_HOME") {
            return PathBuf::from(val).join("semantex").join("config.yaml");
        }
        dirs::config_dir().map_or_else(
            || Self::semantex_home().join("config.yaml"),
            |d| d.join("semantex").join("config.yaml"),
        )
    }
}

/// Read a positive `usize` tuning knob from an environment variable.
///
/// Returns the parsed value when `key` is set to a positive integer; a missing,
/// unparseable, or zero/negative value falls back to `default`. Callers pass a
/// `default` that is already a sane positive number, so the result is always
/// usable as a thread count / batch size / concurrency limit.
pub(crate) fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default)
}

/// Read a non-empty string tuning knob from an environment variable.
///
/// Returns the trimmed value when `key` is set to a non-empty string; a
/// missing or whitespace-only value falls back to `default`.
// Added in S1 for S8's ModelRegistry selection (per integration §3 item 5);
// S8's `load()` overlays now consume it (SEMANTEX_EMBEDDER / _RERANKER_MODEL /
// _LLM_MODEL).
pub(crate) fn env_string(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| default.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// v0.4 Item 18: BM25 stemmer ON by default (preserves legacy behavior).
    #[test]
    fn default_config_enables_bm25_stemmer() {
        let cfg = SemantexConfig::default();
        assert!(
            cfg.use_bm25_stemmer,
            "Default must preserve legacy stemming behavior (true) per v0.4 spec §9.2.3"
        );
    }

    #[test]
    fn default_dense_backend_is_colbert_plaid() {
        let cfg = SemantexConfig::default();
        assert_eq!(
            cfg.dense_backend, "colbert-plaid",
            "default dense backend must stay colbert-plaid until S2 + harness flip it (D4)"
        );
    }

    #[test]
    fn default_selection_fields() {
        let cfg = SemantexConfig::default();
        // D4: PLAID stays the shipped dense default until the Phase-3 cutover.
        assert_eq!(cfg.embedder, "lateon-colbert");
        assert_eq!(cfg.reranker_model, "bge-reranker-v2-m3");
        assert_eq!(
            cfg.llm_model, "",
            "no LLM selected by default (zero-LLM-deps build)"
        );
    }

    #[test]
    fn env_string_reads_value_or_default() {
        // Unset key falls back to the provided default.
        assert_eq!(
            env_string("SEMANTEX_DENSE_BACKEND_TEST_UNSET_KEY", "colbert-plaid"),
            "colbert-plaid"
        );
    }
}
