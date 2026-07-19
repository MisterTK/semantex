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
    /// PLAID residual-quantization bits for the `lateon-colbert` / `colbert-plaid`
    /// dense backend (2 or 4; default 4) — the shipped default backend. Ignored by
    /// the opt-in `coderank-hnsw` backend, which stores int8 single vectors, not
    /// PLAID residuals.
    pub plaid_nbits: usize,
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
    /// DEPRECATED dense-backend alias / explicit override knob. Default EMPTY:
    /// when unset, the canonical `embedder` selection decides the backend (the
    /// default `lateon-colbert` → `colbert-plaid`), so `SEMANTEX_EMBEDDER` is the
    /// primary selector. When set (via `SEMANTEX_DENSE_BACKEND`) to a known
    /// backend name, that alias WINS directly — overriding the embedder path; an
    /// UNKNOWN value falls through to the embedder path (see
    /// [`ModelRegistry::resolve_dense_backend`]). MUST match the value the index
    /// was built with — `HybridSearcher::open` re-validates the RESOLVED backend
    /// against the persisted `IndexMeta.dense_backend` and refuses to load on
    /// mismatch (mirrors `use_bm25_stemmer`).
    pub dense_backend: String,
    /// Active embedder model id (registry lookup key). Default `"lateon-colbert"`
    /// (LateOn-Code-edge, multi-vector → the `colbert-plaid` late-interaction
    /// backend). The embedder spec id is model-descriptive and distinct from the
    /// dense backend name it routes to via capabilities. Override via
    /// `SEMANTEX_EMBEDDER` (e.g. `coderank-137m` → `coderank-hnsw`). A change here
    /// triggers a versioned dense rebuild + atomic switchover (S8) — the
    /// re-embedding compute is inherent.
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
            plaid_nbits: 4,
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
            // EMPTY by default: the deprecated alias is unset, so the canonical
            // `embedder` selection decides the backend. An explicit
            // SEMANTEX_DENSE_BACKEND overrides it. (A non-empty default here would
            // permanently shadow SEMANTEX_EMBEDDER — e.g. make lateon-colbert
            // unselectable — since the alias wins whenever it parses.)
            dense_backend: String::new(),
            // lateon-colbert → colbert-plaid (late-interaction PLAID) is the
            // default dense path, resolved via the embedder capabilities (the
            // alias above is empty). Cutover 2026-06-02 on the measured
            // chunked-real-pipeline A/B (+6.2% nDCG@10 / +12% MRR@10 over
            // coderank, ~10x lower query latency, ~19x smaller index — see
            // docs/superpowers/plans/2026-06-02-item3-realrepo-ab-results.md).
            // coderank-137m / coderank-hnsw is kept registered as a legacy opt-in
            // (SEMANTEX_EMBEDDER=coderank-137m) for anyone who explicitly wants it,
            // but per the 2026-07-16 lateon-vs-coderank head-to-head
            // (results/lateon-vs-coderank-quality/report.md) it is NOT a target for
            // further investment: no quality edge on the larger/more representative
            // benchmark, consistently slower, and more timeout-prone on large repos.
            // Treat it the same as qwen3-embed-0.6b — available, not promoted.
            embedder: "lateon-colbert".to_string(),
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

/// Read a boolean tuning/experiment flag from an environment variable.
///
/// `true` when `key` is set to `"1"` or a case-insensitive `"true"`; missing
/// or any other value is `false`. This is the same truthy convention already
/// applied ad hoc at several call sites (`SEMANTEX_DENSE_CONTEXT` in
/// `model/registry.rs::dense_context_enabled`, `SEMANTEX_SEMANTIC_CACHE` in
/// `search/semantic_cache.rs::is_enabled`) — new boolean env flags should call
/// this helper instead of re-deriving the parse.
pub(crate) fn env_bool(key: &str) -> bool {
    std::env::var(key).is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

/// Read a DEFAULT-ON boolean flag from an environment variable.
///
/// Inverse-default sibling of [`env_bool`]: returns `true` when `key` is missing,
/// and `false` ONLY when it is explicitly set to a recognized falsy token —
/// `"0"` or a case-insensitive `"false"`. Any other value (including `"1"` /
/// `"true"` / `"yes"` / an empty string) is `true`. This is the same
/// `v != "0" && !v.eq_ignore_ascii_case("false")` opt-out convention already
/// applied ad hoc to the default-ON flags in [`SemantexConfig::load`]
/// (`SEMANTEX_RERANK`, `SEMANTEX_ADAPTIVE_SIZING`), surfaced as a reusable helper.
///
/// Note the deliberate asymmetry with `env_bool`'s truthy set: `env_bool` only
/// treats `"1"`/`"true"` as truthy (a default-OFF opt-IN), whereas here anything
/// that is not an explicit falsy token opts IN, because absent must mean "on".
pub(crate) fn env_bool_default_true(key: &str) -> bool {
    match std::env::var(key) {
        Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
        Err(_) => true,
    }
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
    fn default_dense_backend_alias_is_empty() {
        let cfg = SemantexConfig::default();
        // The deprecated alias defaults EMPTY so it does NOT shadow the canonical
        // `embedder` selection (a non-empty default would make SEMANTEX_EMBEDDER
        // ineffective — e.g. lateon-colbert unselectable). The all-default
        // resolution still yields coderank-hnsw via the embedder path — see
        // resolve_dense_backend_all_default_is_coderank_hnsw in registry.rs.
        assert_eq!(
            cfg.dense_backend, "",
            "deprecated alias default must be empty (canonical embedder decides)"
        );
    }

    #[test]
    fn default_selection_fields() {
        let cfg = SemantexConfig::default();
        // 2026-06-02 cutover: lateon-colbert (→ colbert-plaid) is the shipped default.
        assert_eq!(cfg.embedder, "lateon-colbert");
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

    /// Dedicated lock for `env_bool`'s own env-mutating test — a private probe
    /// key untouched by any other test, so this doesn't need to share a
    /// broader family lock like `RERANKER_TEST_ENV_LOCK`.
    static ENV_BOOL_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn env_bool_recognizes_truthy_and_falsy_values() {
        const KEY: &str = "SEMANTEX_TEST_ENV_BOOL_PROBE";
        let _g = ENV_BOOL_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: guarded by ENV_BOOL_TEST_LOCK; KEY is private to this test.
        unsafe {
            std::env::remove_var(KEY);
        }
        assert!(!env_bool(KEY), "unset must default to false");
        for truthy in ["1", "true", "TRUE", "True"] {
            // SAFETY: guarded by ENV_BOOL_TEST_LOCK.
            unsafe {
                std::env::set_var(KEY, truthy);
            }
            assert!(env_bool(KEY), "{truthy:?} must be truthy");
        }
        for falsy in ["0", "false", "yes", ""] {
            // SAFETY: guarded by ENV_BOOL_TEST_LOCK.
            unsafe {
                std::env::set_var(KEY, falsy);
            }
            assert!(!env_bool(KEY), "{falsy:?} must be falsy");
        }
        // SAFETY: guarded by ENV_BOOL_TEST_LOCK.
        unsafe {
            std::env::remove_var(KEY);
        }
    }

    #[test]
    fn env_bool_default_true_defaults_on_and_opts_out_on_falsy() {
        const KEY: &str = "SEMANTEX_TEST_ENV_BOOL_DEFAULT_TRUE_PROBE";
        let _g = ENV_BOOL_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: guarded by ENV_BOOL_TEST_LOCK; KEY is private to this test.
        unsafe {
            std::env::remove_var(KEY);
        }
        // Absent → ON (the whole point of the default-true helper).
        assert!(env_bool_default_true(KEY), "unset must default to true");

        // ONLY "0" / case-insensitive "false" opt out.
        for falsy in ["0", "false", "False", "FALSE"] {
            // SAFETY: guarded by ENV_BOOL_TEST_LOCK.
            unsafe {
                std::env::set_var(KEY, falsy);
            }
            assert!(
                !env_bool_default_true(KEY),
                "{falsy:?} must opt OUT (false)"
            );
        }
        // Everything else stays ON — note "yes" and "" differ from env_bool,
        // which treats them as falsy; here anything not an explicit falsy token
        // means "on".
        for truthy in ["1", "true", "True", "yes", "", "on", "anything"] {
            // SAFETY: guarded by ENV_BOOL_TEST_LOCK.
            unsafe {
                std::env::set_var(KEY, truthy);
            }
            assert!(env_bool_default_true(KEY), "{truthy:?} must stay ON (true)");
        }
        // SAFETY: guarded by ENV_BOOL_TEST_LOCK.
        unsafe {
            std::env::remove_var(KEY);
        }
    }
}
