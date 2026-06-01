use crate::search::adaptive::AdaptiveConfig;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

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
    /// Base retrieval-candidate pool width. This is the number of fused
    /// candidates retrieval targets for EVERY query (then oversampled per query
    /// type — Identifier ×5, Keyword ×2, Semantic ×2–3, Mixed ×1 — see
    /// `hybrid.rs`). Despite the name it is NOT rerank-specific: it governs the
    /// default (rerank-off) search path too. Distinct from `rerank_top_n`, which
    /// is the cross-encoder scoring window. Do not conflate the two.
    pub rerank_candidates: usize,
    /// Number of top fused candidates the cross-encoder reranker scores. Rerank
    /// latency is ~linear in this value, so it bounds the (CPU-bound) rerank
    /// cost. Only used when reranking is enabled (`rerank` + per-query
    /// `use_rerank`); never scores fewer than `max_count` (we never rerank fewer
    /// than we return). Tune via `SEMANTEX_RERANK_CANDIDATES`. Distinct from
    /// `rerank_candidates`, which is the pre-rerank retrieval pool width.
    pub rerank_top_n: usize,
    /// Custom model directory override
    pub model_dir: Option<PathBuf>,
    /// Enable adaptive result sizing. Runs post-fusion for ALL searches (including
    /// `--sparse-only`), applying three stages: (1) confidence-threshold filtering
    /// (drops results below `top_score × min_score_<type>`), (2) per-file
    /// deduplication (keeps only the top chunk per file unless scores are close),
    /// and (3) elbow-detection output cap (≤ 15 results non-exhaustive, ≤ 25
    /// exhaustive). Stages are applied identically across all A/B arms, so
    /// relative comparisons remain fair — but absolute Recall@k is bounded by the
    /// cap regardless of `-m k`. Set `SEMANTEX_ADAPTIVE_SIZING=0` (or `false`) to
    /// disable for clean Recall@k measurement; any other value (including `1` /
    /// `true`) re-enables it. Default: `true`.
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
    /// DEPRECATED dense-backend alias / selection knob. Default `"coderank-hnsw"`
    /// (the [`DenseBackendKind::default`] name + sole built-in backend). When it
    /// parses to a known backend name that alias wins; an UNKNOWN value (e.g. a
    /// stale `"colbert-plaid"` from an old config) falls through to the canonical
    /// `embedder` selection (see [`ModelRegistry::resolve_dense_backend`]). Set
    /// via `SEMANTEX_DENSE_BACKEND`. MUST match the value the index was built with
    /// — `HybridSearcher::open` re-validates it against the persisted
    /// `IndexMeta.dense_backend` and refuses to load on mismatch (mirrors
    /// `use_bm25_stemmer`).
    pub dense_backend: String,
    /// Active embedder model id (registry lookup key). Default `"coderank-137m"`
    /// (CodeRankEmbed-137M, single-vector → the `coderank-hnsw` dense backend).
    /// The embedder spec id is model-descriptive and distinct from the dense
    /// backend name it routes to via capabilities. Override via
    /// `SEMANTEX_EMBEDDER`. A change here triggers a versioned dense rebuild +
    /// atomic switchover (S8) — the re-embedding compute is inherent.
    pub embedder: String,
    /// Active reranker model id (registry lookup key). Default
    /// `"bge-reranker-v2-m3"`. Override via `SEMANTEX_RERANKER_MODEL`. A change
    /// here is a query-time live swap — no reindex.
    pub reranker_model: String,
    /// Active LLM model id (registry lookup key). Empty = none. Override via
    /// `SEMANTEX_LLM_MODEL`. Only meaningful with the `llm` feature.
    pub llm_model: String,
    /// HNSW `ef_search` OVERRIDE for the coderank-hnsw backend (higher = better
    /// recall, slower). `0` ⇒ "use the preset's ef_search" (the `default` preset
    /// is 64). Set explicitly via `SEMANTEX_HNSW_EF_SEARCH` to override the
    /// preset. Kept `0` by default so `SEMANTEX_HNSW_PRESET=high_recall` actually
    /// takes effect (ef_search bakes into the graph at build time, so a silent
    /// clobber would discard the preset's recall intent).
    pub hnsw_ef_search: usize,
    /// HNSW tuning preset: `default | high_recall | low_latency | memory_optimized`.
    /// Override via `SEMANTEX_HNSW_PRESET`. The preset's `ef_search` is used
    /// unless `SEMANTEX_HNSW_EF_SEARCH` is set non-zero (which then overrides it).
    pub hnsw_preset: String,
    /// fp32-rescore the top `dense_rescore_k` ANN candidates. `0` ⇒ derive 4×k
    /// at query time. Override via `SEMANTEX_DENSE_RESCORE_K`.
    pub dense_rescore_k: usize,
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
            rrf_k: 60.0,
            rerank_candidates: 100,
            rerank_top_n: 25,
            model_dir: None,
            // Adaptive result sizing
            adaptive_sizing: true,
            adaptive_dedup: true,
            adaptive_dedup_gap: 0.10,
            min_score_identifier: 0.08,
            min_score_keyword: 0.15,
            min_score_semantic: 0.10,
            min_score_mixed: 0.10,
            use_bm25_stemmer: true,
            // Default = DenseBackendKind::default() name (the sole built-in
            // backend). The canonical `embedder` selection resolves to the same
            // backend; an explicit SEMANTEX_DENSE_BACKEND can still override.
            dense_backend: "coderank-hnsw".to_string(),
            // coderank-137m → coderank-hnsw is the default dense path.
            embedder: "coderank-137m".to_string(),
            reranker_model: "bge-reranker-v2-m3".to_string(),
            llm_model: String::new(),
            hnsw_ef_search: 0, // 0 ⇒ use the preset's ef_search (default preset = 64)
            hnsw_preset: "default".to_string(),
            dense_rescore_k: 0,
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
        if let Ok(v) = std::env::var("SEMANTEX_ADAPTIVE_SIZING") {
            config.adaptive_sizing = v != "0" && !v.eq_ignore_ascii_case("false");
        }
        // Bounds the cross-encoder scoring window (NOT the retrieval pool, which
        // stays `rerank_candidates`). Rerank latency is ~linear in this value.
        config.rerank_top_n = env_usize("SEMANTEX_RERANK_CANDIDATES", config.rerank_top_n);
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

        // S2: coderank-hnsw tuning knobs.
        config.hnsw_ef_search = env_usize("SEMANTEX_HNSW_EF_SEARCH", config.hnsw_ef_search);
        config.hnsw_preset = env_string("SEMANTEX_HNSW_PRESET", &config.hnsw_preset);
        // `dense_rescore_k` uses a raw parse because `0` is a meaningful value
        // (= derive 4×k at query time) that `env_usize` would reject.
        config.dense_rescore_k = std::env::var("SEMANTEX_DENSE_RESCORE_K")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(config.dense_rescore_k);

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

    #[test]
    fn rrf_k_default_is_60_per_integration_doc_d_rrf_k() {
        // Integration doc D-rrf-k: weighted-RRF default k aligns with the
        // parameter-free RRF_K = 60.0 so A/Bs are apples-to-apples.
        let cfg = SemantexConfig::default();
        assert!(
            (cfg.rrf_k - 60.0).abs() < f32::EPSILON,
            "rrf_k default = {}",
            cfg.rrf_k
        );
    }

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
    fn default_dense_backend_alias_is_coderank_hnsw() {
        let cfg = SemantexConfig::default();
        // D4: the `dense_backend` alias default is the DenseBackendKind::default()
        // name ("coderank-hnsw"), the sole built-in backend. The all-default
        // resolution yields coderank-hnsw — see
        // resolve_dense_backend_all_default_is_coderank_hnsw in registry.rs.
        assert_eq!(
            cfg.dense_backend, "coderank-hnsw",
            "alias default is the sole built-in backend name"
        );
    }

    #[test]
    fn default_selection_fields() {
        let cfg = SemantexConfig::default();
        // D4 cutover: coderank-137m (→ coderank-hnsw) is the shipped default.
        assert_eq!(cfg.embedder, "coderank-137m");
        assert_eq!(cfg.reranker_model, "bge-reranker-v2-m3");
        assert_eq!(
            cfg.llm_model, "",
            "no LLM selected by default (zero-LLM-deps build)"
        );
    }

    #[test]
    fn dense_tuning_defaults() {
        let cfg = SemantexConfig::default();
        assert_eq!(
            cfg.hnsw_ef_search, 0,
            "0 means 'use the preset's ef_search' so SEMANTEX_HNSW_PRESET is honored"
        );
        assert_eq!(cfg.hnsw_preset, "default");
        assert_eq!(cfg.dense_rescore_k, 0, "0 means derive 4×k at query time");
    }

    #[test]
    fn env_string_reads_value_or_default() {
        // Unset key falls back to the provided default.
        assert_eq!(
            env_string("SEMANTEX_DENSE_BACKEND_TEST_UNSET_KEY", "coderank-hnsw"),
            "coderank-hnsw"
        );
    }

    #[test]
    fn rerank_top_n_default_is_25() {
        // The cross-encoder scoring window defaults to 25 so reranking fits a
        // deployable latency budget (latency is ~linear in this value).
        let cfg = SemantexConfig::default();
        assert_eq!(cfg.rerank_top_n, 25);
    }

    #[test]
    fn rerank_candidates_default_is_still_100() {
        // INVARIANT: the rerank scoring window is decoupled from the retrieval
        // pool. Introducing `rerank_top_n` must NOT shrink the base
        // retrieval-candidate pool used by the default (rerank-off) search path.
        let cfg = SemantexConfig::default();
        assert_eq!(
            cfg.rerank_candidates, 100,
            "retrieval pool width must remain 100 (default search path unchanged)"
        );
    }

    /// Default adaptive_sizing is true; SEMANTEX_ADAPTIVE_SIZING env overlay works.
    #[test]
    fn adaptive_sizing_default_and_env_overlay() {
        // Default must be true — the cap is on by default.
        let cfg = SemantexConfig::default();
        assert!(
            cfg.adaptive_sizing,
            "adaptive_sizing default must be true (post-fusion cap is on by default)"
        );

        // Serialize env mutations with the shared reranker lock (same global env).
        let _g = crate::search::RERANKER_TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let prior = std::env::var("SEMANTEX_ADAPTIVE_SIZING").ok();

        let assert_load = |val: &str| -> bool {
            // SAFETY: guarded by RERANKER_TEST_ENV_LOCK.
            unsafe { std::env::set_var("SEMANTEX_ADAPTIVE_SIZING", val) };
            SemantexConfig::load(None).expect("load").adaptive_sizing
        };

        // Falsy values disable adaptive sizing.
        assert!(!assert_load("0"), "\"0\" should disable adaptive_sizing");
        assert!(
            !assert_load("false"),
            "\"false\" should disable adaptive_sizing"
        );
        assert!(
            !assert_load("False"),
            "\"False\" (mixed-case) should disable adaptive_sizing"
        );
        assert!(
            !assert_load("FALSE"),
            "\"FALSE\" should disable adaptive_sizing"
        );

        // Truthy values keep it enabled.
        assert!(assert_load("1"), "\"1\" should enable adaptive_sizing");
        assert!(
            assert_load("true"),
            "\"true\" should enable adaptive_sizing"
        );
        assert!(
            assert_load("True"),
            "\"True\" should enable adaptive_sizing"
        );

        // Restore original env state.
        // SAFETY: guarded by RERANKER_TEST_ENV_LOCK.
        unsafe {
            match prior {
                Some(v) => std::env::set_var("SEMANTEX_ADAPTIVE_SIZING", v),
                None => std::env::remove_var("SEMANTEX_ADAPTIVE_SIZING"),
            }
        }
    }

    #[test]
    fn semantex_rerank_candidates_env_sets_rerank_top_n() {
        // `SEMANTEX_RERANK_CANDIDATES` overlays `rerank_top_n` (the scoring
        // window), NOT `rerank_candidates` (the retrieval pool). Serialize env
        // mutation against the shared reranker test lock to avoid flakiness.
        let _g = crate::search::RERANKER_TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let prior = std::env::var("SEMANTEX_RERANK_CANDIDATES").ok();
        // SAFETY: guarded by RERANKER_TEST_ENV_LOCK.
        unsafe {
            std::env::set_var("SEMANTEX_RERANK_CANDIDATES", "12");
        }
        let cfg = SemantexConfig::load(None).expect("load");
        // SAFETY: guarded by RERANKER_TEST_ENV_LOCK.
        unsafe {
            match prior {
                Some(v) => std::env::set_var("SEMANTEX_RERANK_CANDIDATES", v),
                None => std::env::remove_var("SEMANTEX_RERANK_CANDIDATES"),
            }
        }
        // The "env doesn't clobber the retrieval pool" invariant is covered by
        // `rerank_candidates_default_is_still_100` (uses `default()`, immune to
        // any global config file); asserting the pool here would be fragile
        // because `load(None)` also reads the global config file.
        assert_eq!(cfg.rerank_top_n, 12, "env overlay sets the scoring window");
    }
}
