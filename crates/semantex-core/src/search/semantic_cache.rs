//! Semantic query cache (S7). Daemon-scoped (owned by `HybridSearcher`).
//! Lookup order: exact-match fast path → embed query → cosine ≥ threshold linear
//! scan over a capped LRU → reuse `(results, metrics)`. MUST flush on reindex /
//! schema-version change (stamped with `IndexMeta.updated_at` + `schema_version`),
//! NOT TTL-only — stale file results are wrong for code.
//! Repo-agnostic; no per-corpus tuning.

use crate::search::SearchMetrics;
use crate::types::SearchResult;
use std::collections::VecDeque;
use std::path::Path;

/// Identity stamp that ties cached entries to a specific index build. A change
/// in either field (reindex rewrites `updated_at`; migration bumps
/// `schema_version`) invalidates the whole cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheStamp {
    pub updated_at: String,
    pub schema_version: u32,
}

/// One cached query → results association, with the embedding used to match
/// semantically-similar future queries.
struct CacheEntry {
    query: String,
    embedding: Vec<f32>,
    results: Vec<SearchResult>,
    metrics: SearchMetrics,
}

/// Capped-LRU semantic query cache. Daemon-scoped: one instance lives on the
/// `HybridSearcher` for the daemon's lifetime. NOT thread-safe on its own —
/// `HybridSearcher` wraps it in a `Mutex`.
pub struct SemanticCache {
    /// Front = most-recently-used. `store` pushes front; eviction pops back.
    entries: VecDeque<CacheEntry>,
    capacity: usize,
    /// The index build these entries belong to. `None` until first `store`.
    /// A `lookup`/`store` with a different stamp flushes all entries.
    stamp: Option<CacheStamp>,
}

impl SemanticCache {
    /// Create an empty cache holding at most `capacity` entries.
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(capacity.min(64)),
            capacity: capacity.max(1),
            stamp: None,
        }
    }

    /// Number of cached entries (test/diagnostics).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when the cache holds no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Drop all entries (called on stamp mismatch — reindex/schema change).
    fn flush(&mut self) {
        self.entries.clear();
    }

    /// Enforce the stamp: if `incoming` differs from the cached stamp, flush and
    /// adopt the new stamp. Called by both `lookup` and `store` so a reindex
    /// (new `updated_at`) invalidates the cache even without a searcher swap.
    fn enforce_stamp(&mut self, incoming: &CacheStamp) {
        match &self.stamp {
            Some(existing) if existing == incoming => {}
            _ => {
                self.flush();
                self.stamp = Some(incoming.clone());
            }
        }
    }

    /// Look up a query. Exact-text match wins immediately; otherwise the highest
    /// cosine match ≥ `threshold` is returned. Returns the cloned cached
    /// `(results, metrics)` on hit, `None` on miss. Promotes the hit to MRU.
    pub fn lookup(
        &mut self,
        query: &str,
        query_vec: &[f32],
        threshold: f32,
        stamp: &CacheStamp,
    ) -> Option<(Vec<SearchResult>, SearchMetrics)> {
        self.enforce_stamp(stamp);
        if self.entries.is_empty() {
            return None;
        }

        // 1) Exact-text fast path.
        if let Some(idx) = self.entries.iter().position(|e| e.query == query) {
            let entry = self.entries.remove(idx)?;
            let out = (entry.results.clone(), entry.metrics.clone());
            self.entries.push_front(entry);
            return Some(out);
        }

        // 2) Cosine linear scan for the best match ≥ threshold.
        let mut best: Option<(usize, f32)> = None;
        for (idx, e) in self.entries.iter().enumerate() {
            let sim = crate::search::mmr::cosine(query_vec, &e.embedding);
            if sim >= threshold && best.is_none_or(|(_, b)| sim > b) {
                best = Some((idx, sim));
            }
        }
        let (idx, _) = best?;
        let entry = self.entries.remove(idx)?;
        let out = (entry.results.clone(), entry.metrics.clone());
        self.entries.push_front(entry);
        Some(out)
    }

    /// Store a query → results association, evicting the LRU entry past capacity.
    pub fn store(
        &mut self,
        query: &str,
        query_vec: &[f32],
        results: Vec<SearchResult>,
        metrics: SearchMetrics,
        stamp: &CacheStamp,
    ) {
        self.enforce_stamp(stamp);
        // Drop any existing entry for the same exact text (avoid duplicates).
        if let Some(idx) = self.entries.iter().position(|e| e.query == query) {
            self.entries.remove(idx);
        }
        self.entries.push_front(CacheEntry {
            query: query.to_string(),
            embedding: query_vec.to_vec(),
            results,
            metrics,
        });
        while self.entries.len() > self.capacity {
            self.entries.pop_back();
        }
    }
}

/// Read the current index stamp from `<index_dir>/meta.json`. Returns `None`
/// if meta.json is missing or unparseable (the cache then declines to operate
/// — better no cache than a wrong stamp).
pub fn read_stamp(index_dir: &Path) -> Option<CacheStamp> {
    let meta_str = std::fs::read_to_string(index_dir.join("meta.json")).ok()?;
    let meta: crate::types::IndexMeta = serde_json::from_str(&meta_str).ok()?;
    Some(CacheStamp {
        updated_at: meta.updated_at,
        schema_version: meta.schema_version,
    })
}

/// Whether the semantic cache is enabled. OFF by default (spec S7: gate behind
/// env until A/B'd). Enabled by `SEMANTEX_SEMANTIC_CACHE=1` (or `true`).
pub fn is_enabled() -> bool {
    std::env::var("SEMANTEX_SEMANTIC_CACHE")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

/// Cosine threshold for a semantic hit. Default 0.85 (spec S7). Override with
/// `SEMANTEX_SEMANTIC_CACHE_THRESHOLD`; non-finite / out-of-[0,1] falls back.
pub fn threshold_from_env() -> f32 {
    const DEFAULT: f32 = 0.85;
    std::env::var("SEMANTEX_SEMANTIC_CACHE_THRESHOLD")
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|x| x.is_finite() && (0.0..=1.0).contains(x))
        .unwrap_or(DEFAULT)
}

/// LRU capacity. Default ~1000 (spec S7). Override with
/// `SEMANTEX_SEMANTIC_CACHE_CAP`; uses `config::env_usize` semantics.
pub fn capacity_from_env() -> usize {
    crate::config::env_usize("SEMANTEX_SEMANTIC_CACHE_CAP", 1000)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Chunk, ChunkType, Confidence, SearchResult, SearchSource};
    use std::path::PathBuf;

    fn stamp(updated: &str) -> CacheStamp {
        CacheStamp {
            updated_at: updated.to_string(),
            schema_version: 10,
        }
    }

    fn metrics() -> SearchMetrics {
        SearchMetrics {
            total_ms: 1,
            dense_ms: None,
            sparse_ms: None,
            exact_ms: None,
            fusion_ms: None,
            rerank_ms: None,
            dense_count: 0,
            sparse_count: 0,
            exact_count: 0,
            fused_count: 0,
            result_count: 1,
            query_type: "Semantic".to_string(),
            response_bytes: None,
        }
    }

    fn one_result(id: u64) -> Vec<SearchResult> {
        vec![SearchResult {
            chunk: Chunk {
                id,
                file_path: PathBuf::from("a.rs"),
                content: "x".to_string(),
                start_line: 1,
                end_line: 2,
                chunk_type: ChunkType::TextWindow { window_index: 0 },
            },
            score: 1.0,
            source: SearchSource::Hybrid,
            score_dense: 0.0,
            score_sparse: 0.0,
            score_exact: 0.0,
            confidence: Confidence::Inferred,
            confidence_score: 0.0,
        }]
    }

    #[test]
    fn exact_match_returns_stored_results() {
        let mut cache = SemanticCache::new(10);
        let st = stamp("100");
        cache.store("auth flow", &[1.0, 0.0], one_result(7), metrics(), &st);
        let hit = cache.lookup(
            "auth flow",
            &[0.0, 1.0], /* wrong vec; exact path wins */
            0.85,
            &st,
        );
        assert!(
            hit.is_some(),
            "identical query text must hit via exact-match fast path"
        );
        assert_eq!(hit.unwrap().0[0].chunk.id, 7);
    }

    #[test]
    fn cosine_near_match_hits_above_threshold() {
        let mut cache = SemanticCache::new(10);
        let st = stamp("100");
        cache.store(
            "how is auth handled",
            &[1.0, 0.0],
            one_result(9),
            metrics(),
            &st,
        );
        // Different text, near-parallel embedding (cos ≈ 0.9995 > 0.85) → hit.
        let hit = cache.lookup("auth handling overview", &[0.999, 0.01], 0.85, &st);
        assert!(
            hit.is_some(),
            "near-parallel embedding above threshold must hit"
        );
        assert_eq!(hit.unwrap().0[0].chunk.id, 9);
    }

    #[test]
    fn cosine_below_threshold_misses() {
        let mut cache = SemanticCache::new(10);
        let st = stamp("100");
        cache.store(
            "how is auth handled",
            &[1.0, 0.0],
            one_result(9),
            metrics(),
            &st,
        );
        // Orthogonal embedding (cos = 0) < 0.85 → miss.
        let hit = cache.lookup("database migrations", &[0.0, 1.0], 0.85, &st);
        assert!(hit.is_none());
    }

    #[test]
    fn lru_evicts_oldest_over_capacity() {
        let mut cache = SemanticCache::new(2);
        let st = stamp("100");
        cache.store("q1", &[1.0, 0.0], one_result(1), metrics(), &st);
        cache.store("q2", &[0.0, 1.0], one_result(2), metrics(), &st);
        cache.store("q3", &[1.0, 1.0], one_result(3), metrics(), &st); // evicts q1
        assert_eq!(cache.len(), 2);
        // q1 evicted: exact lookup of "q1" with an orthogonal probe vector misses.
        assert!(cache.lookup("q1", &[0.0, 0.0], 0.85, &st).is_none());
        // q3 present.
        assert!(cache.lookup("q3", &[0.0, 0.0], 0.85, &st).is_some());
    }

    #[test]
    fn stamp_change_flushes_cache() {
        // THE reindex-correctness invariant: a changed updated_at invalidates
        // the cache. A query that hit under stamp "100" must MISS under "200".
        let mut cache = SemanticCache::new(10);
        let st_old = stamp("100");
        cache.store("auth flow", &[1.0, 0.0], one_result(7), metrics(), &st_old);
        assert!(
            cache
                .lookup("auth flow", &[1.0, 0.0], 0.85, &st_old)
                .is_some()
        );

        // Reindex bumps updated_at → new stamp.
        let st_new = stamp("200");
        assert!(
            cache
                .lookup("auth flow", &[1.0, 0.0], 0.85, &st_new)
                .is_none(),
            "reindex (new updated_at) MUST invalidate the cache"
        );
        assert_eq!(cache.len(), 0, "stamp change flushes all entries");
    }

    #[test]
    fn schema_version_change_flushes_cache() {
        let mut cache = SemanticCache::new(10);
        let st_v10 = CacheStamp {
            updated_at: "100".into(),
            schema_version: 10,
        };
        cache.store("q", &[1.0, 0.0], one_result(1), metrics(), &st_v10);
        let st_v11 = CacheStamp {
            updated_at: "100".into(),
            schema_version: 11,
        };
        assert!(cache.lookup("q", &[1.0, 0.0], 0.85, &st_v11).is_none());
    }

    #[test]
    fn read_stamp_from_meta_json() {
        let tmp = tempfile::TempDir::new().unwrap();
        let index_dir = tmp.path();
        let meta = crate::types::IndexMeta {
            schema_version: crate::types::IndexMeta::CURRENT_SCHEMA_VERSION,
            project_path: index_dir.to_path_buf(),
            created_at: "0".to_string(),
            updated_at: "1717000000".to_string(),
            file_count: 1,
            chunk_count: 2,
            embedding_model: "LateOn-Code-edge".to_string(),
            embedding_dim: 48,
            use_bm25_stemmer: true,
            dense_backend: "colbert-plaid".to_string(), // S1 field
            embedder_fingerprint: "test-fp".to_string(), // S8 field
        };
        std::fs::write(
            index_dir.join("meta.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();
        let st = read_stamp(index_dir).expect("meta.json present → Some stamp");
        assert_eq!(st.updated_at, "1717000000");
        assert_eq!(
            st.schema_version,
            crate::types::IndexMeta::CURRENT_SCHEMA_VERSION
        );
    }

    #[test]
    fn read_stamp_missing_meta_is_none() {
        let tmp = tempfile::TempDir::new().unwrap();
        assert!(read_stamp(tmp.path()).is_none());
    }

    #[test]
    fn cache_disabled_by_default_enabled_by_env() {
        unsafe {
            std::env::remove_var("SEMANTEX_SEMANTIC_CACHE");
        }
        assert!(!is_enabled(), "semantic cache is OFF by default");
        unsafe {
            std::env::set_var("SEMANTEX_SEMANTIC_CACHE", "1");
        }
        assert!(is_enabled());
        unsafe {
            std::env::remove_var("SEMANTEX_SEMANTIC_CACHE");
        }
    }

    #[test]
    fn threshold_default_is_point85() {
        unsafe {
            std::env::remove_var("SEMANTEX_SEMANTIC_CACHE_THRESHOLD");
        }
        assert!((threshold_from_env() - 0.85).abs() < f32::EPSILON);
    }
}
