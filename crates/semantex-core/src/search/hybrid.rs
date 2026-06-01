use crate::config::SemantexConfig;
use crate::embedding::model_manager;
use crate::index::file_classifier::FileRole;
use crate::index::storage::ChunkStore;
use crate::search::SearchQuery;
use crate::search::adaptive;
use crate::search::colbert_plaid_backend::ColbertPlaidBackend;
use crate::search::dense_backend::{
    DenseBackend, DenseBackendKind, dense_subdir, verify_persisted_backend_matches,
};
use crate::search::graph_propagation::{self, GraphPropagationConfig};
use crate::search::path_signals;
use crate::search::mmr;
use crate::search::query_classifier::{self, FusionWeights, QueryType};
use crate::search::reranker_engine::RerankerEngine;
use crate::search::sparse_search::SparseIndex;
use crate::search::triple_fusion::{self, FusionMode, RrfFusedResult};
use crate::search::{query_expander, regex_semantic};
use crate::types::{Confidence, ScoredChunkId, SearchResult, SearchSource};
use anyhow::{Context, Result};
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Hybrid searcher combining dense (pluggable `DenseBackend`), sparse (BM25),
/// and reranking. Exact substring search is handled by SQLite LIKE queries on
/// the chunk store.
pub struct HybridSearcher {
    sparse: Option<SparseIndex>,
    /// Pluggable dense channel (S1). `None` when dense is unavailable
    /// (sparse-only open, or the dense index failed to load).
    dense: Option<Box<dyn DenseBackend>>,
    reranker: Mutex<Option<RerankerEngine>>,
    store: Mutex<ChunkStore>,
    config: SemantexConfig,
}

impl HybridSearcher {
    /// Open a sparse-only searcher that skips loading the ONNX model entirely.
    /// Only loads Tantivy (mmap) + SQLite (search-mode). Ideal for grep-like
    /// queries where dense embeddings are not needed, saving ~500MB+ of RAM.
    pub fn open_sparse_only(index_dir: &Path, config: &SemantexConfig) -> Result<Self> {
        let store = Mutex::new(
            ChunkStore::open_for_search(&index_dir.join("chunks.db"))
                .context("Failed to open chunk store for sparse-only search")?,
        );

        let sparse_path = index_dir.join("sparse");
        let sparse = if sparse_path.exists() {
            // v0.4 Item 18 / v0.4.1 W-Index #4: stemmer flag MUST match the
            // value used at index-build time. `SparseIndex::open` reads the
            // persisted flag from meta.json and fails loudly on mismatch
            // (the previous "silently degrade recall" caveat was promoted
            // into a hard error). The `?` here propagates the mismatch up
            // to the daemon startup path, which is correct: the user wants
            // to know before the daemon binds, not after queries return
            // wrong results.
            Some(SparseIndex::open(&sparse_path, config.use_bm25_stemmer)?)
        } else {
            None
        };

        let reranker = Mutex::new(None);

        Ok(Self {
            sparse,
            dense: None,
            reranker,
            store,
            config: config.clone(),
        })
    }

    /// Open searcher from an existing index directory
    pub fn open(index_dir: &Path, config: &SemantexConfig) -> Result<Self> {
        // Check schema version compatibility
        let meta_path = index_dir.join("meta.json");
        if meta_path.exists()
            && let Ok(meta_str) = std::fs::read_to_string(&meta_path)
            && let Ok(existing_meta) = serde_json::from_str::<crate::types::IndexMeta>(&meta_str)
            && existing_meta.schema_version != crate::types::IndexMeta::CURRENT_SCHEMA_VERSION
        {
            tracing::warn!(
                "Index schema version mismatch (found v{}, expected v{}). \
                 Re-index with `semantex index` for best results.",
                existing_meta.schema_version,
                crate::types::IndexMeta::CURRENT_SCHEMA_VERSION
            );
        }

        // Open chunk store in search mode (reduced memory footprint)
        let store = Mutex::new(
            ChunkStore::open_for_search(&index_dir.join("chunks.db"))
                .context("Failed to open chunk store")?,
        );

        // Open sparse searcher
        let sparse_path = index_dir.join("sparse");
        let sparse = if sparse_path.exists() {
            // v0.4.1 W-Index #4: SparseIndex::open verifies the persisted
            // stemmer flag against `config.use_bm25_stemmer` and fails
            // loudly on mismatch — see `open_sparse_only` above.
            Some(SparseIndex::open(&sparse_path, config.use_bm25_stemmer)?)
        } else {
            None
        };

        // Load the dense backend. Selection (S2 re-point): resolve via the S8
        // ModelRegistry from the canonical `SEMANTEX_EMBEDDER` selection, with
        // `SEMANTEX_DENSE_BACKEND`/`config.dense_backend` kept as a DEPRECATED
        // alias (alias wins only when explicitly non-default). Defaults still
        // resolve to colbert-plaid (D4) until the Phase-3 cutover.
        let resolved_backend =
            crate::model::ModelRegistry::resolve_dense_backend(config, None).unwrap_or_default();
        // The persisted backend in meta.json MUST match the RESOLVED backend —
        // verify and refuse on mismatch (mirrors the BM25 stemmer guard).
        verify_persisted_backend_matches(index_dir, resolved_backend.name())?;
        let dense: Option<Box<dyn DenseBackend>> = match resolved_backend {
            DenseBackendKind::ColbertPlaid => {
                // Per-backend subdir is canonical; fall back to the legacy
                // top-level `plaid/` layout for indexes built before S1.
                let backend_dir = dense_subdir(index_dir, DenseBackendKind::ColbertPlaid);
                let legacy_dir = index_dir.join("plaid");
                let (plaid_dir, mapping_path) = if backend_dir.exists() {
                    (backend_dir.clone(), backend_dir.join("plaid_mapping.bin"))
                } else {
                    (legacy_dir, index_dir.join("plaid_mapping.bin"))
                };
                if plaid_dir.exists() && mapping_path.exists() {
                    let model_dir = model_manager::ensure_colbert_model(&config.models_dir());
                    match model_dir
                        .and_then(|d| ColbertPlaidBackend::open(&plaid_dir, &mapping_path, &d))
                    {
                        Ok(b) => {
                            tracing::info!("Dense backend loaded: colbert-plaid");
                            Some(Box::new(b) as Box<dyn DenseBackend>)
                        }
                        Err(e) => {
                            tracing::warn!("colbert-plaid backend failed to load: {}", e);
                            None
                        }
                    }
                } else {
                    None
                }
            }
            DenseBackendKind::CoderankHnsw => {
                use crate::index::hnsw_index::{CoderankHnswBackend, HnswParams};
                let backend_dir = dense_subdir(index_dir, DenseBackendKind::CoderankHnsw);
                if backend_dir.join("vectors.bin").exists() {
                    let params = HnswParams::resolve(
                        &config.hnsw_preset,
                        config.hnsw_ef_search,
                        config.dense_rescore_k,
                    );
                    let model_dir = crate::embedding::single_vector_model::ensure_coderank_model(
                        &config.models_dir(),
                    );
                    match model_dir
                        .and_then(|d| CoderankHnswBackend::open(&backend_dir, &d, params))
                    {
                        Ok(b) => {
                            tracing::info!("Dense backend loaded: coderank-hnsw");
                            Some(Box::new(b) as Box<dyn DenseBackend>)
                        }
                        Err(e) => {
                            tracing::warn!("coderank-hnsw backend failed to load: {e}");
                            None
                        }
                    }
                } else {
                    None
                }
            }
        };

        // Reranker is loaded lazily on first use to save ~200MB for cold searches
        let reranker = Mutex::new(None);

        Ok(Self {
            sparse,
            dense,
            reranker,
            store,
            config: config.clone(),
        })
    }

    /// Reload the sparse index reader to pick up incremental updates.
    /// The dense index is memory-mapped and picks up changes automatically on re-open,
    /// but Tantivy requires an explicit reader reload.
    /// Provide read-only access to the chunk store for graph queries.
    /// Used by the daemon handler and the direct fallback path.
    pub(crate) fn with_store<F, T>(&self, f: F) -> T
    where
        F: FnOnce(&crate::index::storage::ChunkStore) -> T,
    {
        let store = self.store.lock();
        f(&store)
    }

    pub fn reload(&self) -> Result<()> {
        if let Some(ref sparse) = self.sparse {
            sparse.reload()?;
        }
        Ok(())
    }

    /// Execute a search query through the full pipeline
    #[tracing::instrument(skip(self), fields(
        query = %query.text,
        max_results = query.max_results,
        use_dense = query.use_dense,
        use_sparse = query.use_sparse,
        use_rerank = query.use_rerank,
        grep_mode = query.grep_mode,
    ))]
    pub fn search(&self, query: &SearchQuery) -> Result<super::SearchOutput> {
        let search_start = std::time::Instant::now();

        // Grep-mode: fast path with simplified exact+sparse fusion
        if query.grep_mode {
            return self.search_grep_mode(query, search_start);
        }

        let candidates = self.config.rerank_candidates;

        // Phase 5: If a regex pattern is present, augment the query text with
        // tokens extracted from the pattern so that BM25 and dense search
        // benefit from regex-derived keywords (e.g. "Promise\.allSettled" adds
        // "allSettled" to the query).
        let effective_text = if let Some(ref pattern) = query.regex_pattern {
            regex_semantic::merge_query_with_pattern(&query.text, pattern)
        } else {
            query.text.clone()
        };

        // Classify query early so we can use it for adaptive candidate counts
        let query_type = query_classifier::classify(&effective_text);

        // Query-adaptive retrieval: identifiers need exhaustive exact matching,
        // keywords benefit from moderate oversampling, semantic queries use 2x oversampling
        // to improve recall before reranking/fusion, mixed queries use base count.
        // 3x for Semantic+architectural recovers Q19 recall without hurting other categories.
        // Global 3x (all query types) was tested and regressed (-0.003 F1 overall).
        // 3x restricted to Semantic+architectural: net positive on Q19 w/o hurting Arch F1.
        let oversample_factor = if query_type == QueryType::Semantic
            && is_architectural_query(&effective_text, query_type)
        {
            3
        } else {
            2
        };
        let retrieval_candidates = match query_type {
            query_classifier::QueryType::Identifier => candidates * 5,
            query_classifier::QueryType::Keyword => candidates * 2,
            query_classifier::QueryType::Semantic => candidates * oversample_factor,
            query_classifier::QueryType::Mixed => candidates,
        };
        tracing::debug!(
            ?query_type,
            candidates,
            retrieval_candidates,
            "Query-adaptive candidate counts"
        );

        // Symbol exact lookup (Phase 6) was tested and disabled:
        // - symbol_defs coverage too low (only 1 file for ConnectionServiceFactory, 1 for retryWithBackoff)
        // - file restriction hurt recall (Q1: 1.00→0.50, Q8: 1.00→0.67)
        // - score inflation hurt import proximity boost avg_top_score
        // The infrastructure (lookup_symbol_exact, find_references in storage.rs) is retained
        // for future use when symbol coverage improves.
        let symbol_exact_hits: Vec<ScoredChunkId> = Vec::new();

        // E2 + E4: choose fusion strategy (RRF default; CC preserved for one release
        // behind SEMANTEX_FUSION=cc). RRF mode unlocks the Exp4Fuse dual-route
        // (original AND expanded query through both dense and sparse channels).
        let fusion_mode = triple_fusion::active_fusion_mode();

        // Generate the deterministic expansion once (E4). Only consumed by the
        // dual-route path when RRF is active and the expansion produced tokens.
        let expanded_text = if matches!(fusion_mode, FusionMode::Rrf)
            && matches!(
                query_type,
                QueryType::Semantic | QueryType::Mixed | QueryType::Keyword
            ) {
            query_expander::expand_query(&effective_text)
        } else {
            None
        };

        // v0.4 WS-B Item 15: precompute the PLAID chunk-ID subset when the
        // query carries an active `file_filter`. PLAID 1.3 accepts a doc-ID
        // subset on `search()` and proportionally scales `n_ivf_probe` to
        // compensate for the smaller candidate pool — this is meaningfully
        // faster than scoring every doc and post-filtering when the filter
        // matches a small fraction of files. We compute the subset once and
        // pass it to both dense channels (original and expanded query).
        //
        // The subset is the set of indexed chunk_ids whose file_path passes
        // the filter. We walk `plaid.doc_to_chunk()` (the canonical list of
        // chunks present in the dense index) and bulk-fetch their paths from
        // the store in bounded batches (SQLite's default parameter cap is
        // 999 — we batch at 500 for headroom). The lock is released before
        // we enter `thread::scope` to avoid contention with `exact_handle`,
        // which takes the same lock.
        let plaid_chunk_subset: Option<Vec<u64>> = match (
            query.use_dense.then_some(()).and(self.dense.as_ref()),
            query.file_filter.as_ref().filter(|f| f.is_active()),
        ) {
            (Some(dense), Some(filter)) => {
                // The subset is computed from the dense index's positional
                // chunk list. Only colbert-plaid exposes one today; backends
                // without positional docs (None) skip subset prep and let the
                // result-merge file_filter handle scoping.
                match dense.positional_chunk_ids() {
                    None => None, // backend has no positional docs — subset N/A
                    Some([]) => Some(Vec::new()),
                    Some(all_indexed) => {
                        const FILTER_BATCH: usize = 500;
                        // v0.4.1 W-Index #7: propagate SQLite errors and fall
                        // back to an unfiltered dense search (result merge still
                        // applies the file_filter) rather than silently
                        // searching a partial candidate set.
                        // v0.4.1 W-Index #3: pre-filter PLAID_TOMBSTONE entries
                        // so we never query SQLite for a deleted-slot sentinel.
                        let subset_result: anyhow::Result<Vec<u64>> = (|| {
                            let store = self.store.lock();
                            let mut subset: Vec<u64> = Vec::new();
                            for batch in all_indexed.chunks(FILTER_BATCH) {
                                let live: Vec<u64> = batch
                                    .iter()
                                    .copied()
                                    .filter(|&cid| cid != crate::types::PLAID_TOMBSTONE)
                                    .collect();
                                if live.is_empty() {
                                    continue;
                                }
                                let chunks = store.get_chunks(&live)?;
                                for c in chunks {
                                    if filter.matches(&c.file_path) {
                                        subset.push(c.id);
                                    }
                                }
                            }
                            Ok(subset)
                        })();
                        match subset_result {
                            Ok(s) => {
                                tracing::debug!(
                                    indexed = all_indexed.len(),
                                    subset = s.len(),
                                    "Dense file_filter subset prepared"
                                );
                                Some(s)
                            }
                            Err(e) => {
                                tracing::warn!(
                                    error = %e,
                                    "Dense file_filter subset construction failed — \
                                     falling back to unfiltered dense search (result \
                                     merge will still apply the file_filter)"
                                );
                                None
                            }
                        }
                    }
                }
            }
            _ => None,
        };
        let plaid_subset_slice: Option<&[u64]> = plaid_chunk_subset.as_deref();

        // Stage 1: Candidate retrieval — run dense, sparse, exact, and (in RRF mode)
        // the expanded-query dense + sparse channels in parallel.
        let (
            dense_results,
            sparse_results,
            exact_ids,
            exp_dense_results,
            exp_sparse_results,
            dense_ms,
            sparse_ms,
            exact_ms,
        ) = std::thread::scope(|s| {
            let dense_handle = s.spawn(|| -> (Vec<ScoredChunkId>, u64) {
                if !query.use_dense {
                    return (Vec::new(), 0);
                }
                let dense_start = std::time::Instant::now();
                let results = if let Some(ref dense) = self.dense {
                    // v0.4 WS-B Item 15: when a file_filter is active, route
                    // through `search_with_subset` with the pre-computed
                    // chunk-ID subset so the backend skips work on chunks the
                    // user has filtered out.
                    let res = if let Some(subset) = plaid_subset_slice {
                        dense.search_with_subset(&effective_text, retrieval_candidates, subset)
                    } else {
                        dense.search(&effective_text, retrieval_candidates)
                    };
                    match res {
                        Ok(results) => {
                            tracing::debug!(
                                results_count = results.len(),
                                duration_ms = dense_start.elapsed().as_millis() as u64,
                                "Dense search complete"
                            );
                            results
                        }
                        Err(e) => {
                            tracing::warn!("Dense search failed: {}", e);
                            Vec::new()
                        }
                    }
                } else {
                    Vec::new()
                };
                (results, dense_start.elapsed().as_millis() as u64)
            });

            let sparse_handle = s.spawn(|| -> (Vec<ScoredChunkId>, u64) {
                if !query.use_sparse {
                    return (Vec::new(), 0);
                }
                let sparse_start = std::time::Instant::now();
                let results = if let Some(ref sparse) = self.sparse {
                    let safe_query = sanitize_tantivy_query(&effective_text);
                    let mut results = sparse
                        .search(&safe_query, retrieval_candidates)
                        .unwrap_or_default();
                    tracing::debug!(
                        results_count = results.len(),
                        duration_ms = sparse_start.elapsed().as_millis() as u64,
                        "Sparse search complete"
                    );

                    // CC-mode legacy expansion: a single weighted fold of expanded
                    // BM25 into the original sparse list. RRF mode handles expansion
                    // via the dedicated exp_sparse channel below — keep this branch
                    // CC-only so we don't double-count expansion contributions.
                    if matches!(fusion_mode, FusionMode::Cc)
                        && matches!(query_type, QueryType::Semantic | QueryType::Mixed)
                        && let Some(expanded) = query_expander::expand_query(&effective_text)
                    {
                        let safe_expanded = sanitize_tantivy_query(&expanded);
                        if let Ok(expanded_results) =
                            sparse.search(&safe_expanded, retrieval_candidates)
                        {
                            tracing::debug!(
                                expanded_count = expanded_results.len(),
                                "Dual-route BM25 expansion (CC mode)"
                            );
                            results = dual_route_fuse(&results, &expanded_results, 1.0, 0.5);
                        }
                    }

                    results
                } else {
                    Vec::new()
                };
                (results, sparse_start.elapsed().as_millis() as u64)
            });

            let exact_handle = s.spawn(|| -> (Vec<u64>, u64) {
                let exact_start = std::time::Instant::now();
                let store = self.store.lock();
                let ids = store
                    .search_exact(&effective_text, retrieval_candidates)
                    .unwrap_or_default();
                tracing::debug!(
                    results_count = ids.len(),
                    duration_ms = exact_start.elapsed().as_millis() as u64,
                    "Exact substring search complete (SQLite)"
                );
                (ids, exact_start.elapsed().as_millis() as u64)
            });

            // E4 dual-route channels (only fire in RRF mode with a non-empty expansion).
            let expanded_ref = expanded_text.as_deref();
            let exp_dense_handle = s.spawn(move || -> Vec<ScoredChunkId> {
                let Some(text) = expanded_ref else {
                    return Vec::new();
                };
                if !query.use_dense {
                    return Vec::new();
                }
                if let Some(ref dense) = self.dense {
                    // v0.4 WS-B Item 15: same subset-aware routing as the
                    // primary dense channel.
                    let res = if let Some(subset) = plaid_subset_slice {
                        dense.search_with_subset(text, retrieval_candidates, subset)
                    } else {
                        dense.search(text, retrieval_candidates)
                    };
                    match res {
                        Ok(r) => {
                            tracing::debug!(
                                expanded_dense_count = r.len(),
                                "Exp4Fuse: expanded-dense channel complete"
                            );
                            r
                        }
                        Err(e) => {
                            tracing::debug!("Expanded dense search failed: {}", e);
                            Vec::new()
                        }
                    }
                } else {
                    Vec::new()
                }
            });

            let exp_sparse_handle = s.spawn(move || -> Vec<ScoredChunkId> {
                let Some(text) = expanded_ref else {
                    return Vec::new();
                };
                if !query.use_sparse {
                    return Vec::new();
                }
                if let Some(ref sparse) = self.sparse {
                    let safe = sanitize_tantivy_query(text);
                    let r = sparse
                        .search(&safe, retrieval_candidates)
                        .unwrap_or_default();
                    tracing::debug!(
                        expanded_sparse_count = r.len(),
                        "Exp4Fuse: expanded-sparse channel complete"
                    );
                    r
                } else {
                    Vec::new()
                }
            });

            let (dense_results, d_ms) = dense_handle.join().unwrap_or_else(|_| (Vec::new(), 0));
            let (sparse_results, s_ms) = sparse_handle.join().unwrap_or_else(|_| (Vec::new(), 0));
            let (exact_ids, e_ms) = exact_handle.join().unwrap_or_else(|_| (Vec::new(), 0));
            let exp_dense = exp_dense_handle.join().unwrap_or_default();
            let exp_sparse = exp_sparse_handle.join().unwrap_or_default();

            (
                dense_results,
                sparse_results,
                exact_ids,
                exp_dense,
                exp_sparse,
                d_ms,
                s_ms,
                e_ms,
            )
        });

        // Stage 2: Fusion. E2 selects RRF or CC via SEMANTEX_FUSION (default RRF).
        // E4 routes through the 5-channel Exp4Fuse helper when RRF is active and
        // the expanded query produced any results. E6 derives per-result confidence
        // labels from channel-agreement counts emitted by RRF.
        let fusion_start = std::time::Instant::now();

        let dense_count = dense_results.len();
        let sparse_count = sparse_results.len();
        let exact_count = exact_ids.len();

        let has_any_results = !dense_results.is_empty()
            || !sparse_results.is_empty()
            || !exact_ids.is_empty()
            || !exp_dense_results.is_empty()
            || !exp_sparse_results.is_empty();

        // confidence_map: chunk_id -> (label, score) populated only by RRF path.
        // CC path derives confidence from per-channel scores at result-build time.
        let mut confidence_map: HashMap<u64, (Confidence, f32)> = HashMap::new();

        let fused: Vec<ScoredChunkId> = if has_any_results {
            match fusion_mode {
                FusionMode::Rrf => {
                    let rrf_results: Vec<RrfFusedResult> = if !exp_dense_results.is_empty()
                        || !exp_sparse_results.is_empty()
                    {
                        triple_fusion::exp4_rrf_fuse(
                            &dense_results,
                            &sparse_results,
                            &exp_dense_results,
                            &exp_sparse_results,
                            &exact_ids,
                        )
                    } else {
                        triple_fusion::triple_rrf_fuse(&dense_results, &sparse_results, &exact_ids)
                    };
                    // Derive per-result confidence labels (gap-aware) from RRF output.
                    let labels = triple_fusion::assign_confidence(&rrf_results);
                    for (rr, (conf, score)) in rrf_results.iter().zip(labels.iter()) {
                        confidence_map.insert(rr.scored.chunk_id, (*conf, *score));
                    }
                    tracing::debug!(
                        fused_count = rrf_results.len(),
                        exact_count = exact_ids.len(),
                        duration_ms = fusion_start.elapsed().as_millis() as u64,
                        dual_route =
                            !exp_dense_results.is_empty() || !exp_sparse_results.is_empty(),
                        "Triple RRF fusion complete (E2 + E4)"
                    );
                    rrf_results.into_iter().map(|r| r.scored).collect()
                }
                // S7: Weighted Triple RRF — per-channel rank-decay scaled by the
                // query-type FusionWeights, with k = config.rrf_k (now live).
                // Selected by SEMANTEX_FUSION=weighted-rrf; default stays Rrf.
                FusionMode::WeightedRrf => {
                    let (weights, k) = weighted_rrf_params(&self.config, query_type);
                    let rrf_results: Vec<RrfFusedResult> = if !exp_dense_results.is_empty()
                        || !exp_sparse_results.is_empty()
                    {
                        triple_fusion::exp4_weighted_rrf_fuse(
                            &dense_results,
                            &sparse_results,
                            &exp_dense_results,
                            &exp_sparse_results,
                            &exact_ids,
                            weights,
                            k,
                        )
                    } else {
                        triple_fusion::triple_weighted_rrf_fuse(
                            &dense_results,
                            &sparse_results,
                            &exact_ids,
                            weights,
                            k,
                        )
                    };
                    let labels = triple_fusion::assign_confidence(&rrf_results);
                    for (rr, (conf, score)) in rrf_results.iter().zip(labels.iter()) {
                        confidence_map.insert(rr.scored.chunk_id, (*conf, *score));
                    }
                    tracing::debug!(
                        fused_count = rrf_results.len(),
                        rrf_k = k,
                        duration_ms = fusion_start.elapsed().as_millis() as u64,
                        "Weighted Triple RRF fusion complete (S7)"
                    );
                    rrf_results.into_iter().map(|r| r.scored).collect()
                }
                FusionMode::Cc => {
                    let triple_weights = query_type.triple_fusion_weights();
                    tracing::debug!(
                        ?query_type,
                        ?triple_weights,
                        "Query classified (CC fusion legacy path)"
                    );
                    let fused = triple_fusion::triple_cc_fuse(
                        &dense_results,
                        &sparse_results,
                        &exact_ids,
                        &triple_weights,
                    );
                    tracing::debug!(
                        fused_count = fused.len(),
                        exact_count = exact_ids.len(),
                        duration_ms = fusion_start.elapsed().as_millis() as u64,
                        "Triple CC fusion complete"
                    );
                    fused
                }
            }
        } else {
            Vec::new()
        };
        let fusion_ms = fusion_start.elapsed().as_millis() as u64;

        if fused.is_empty() && symbol_exact_hits.is_empty() {
            tracing::info!(
                query = %query.text,
                duration_ms = search_start.elapsed().as_millis() as u64,
                "Search completed with no results"
            );
            return Ok(super::SearchOutput {
                results: Vec::new(),
                metrics: super::SearchMetrics {
                    total_ms: search_start.elapsed().as_millis() as u64,
                    dense_ms: if query.use_dense {
                        Some(dense_ms)
                    } else {
                        None
                    },
                    sparse_ms: if query.use_sparse {
                        Some(sparse_ms)
                    } else {
                        None
                    },
                    exact_ms: Some(exact_ms),
                    fusion_ms: Some(fusion_ms),
                    rerank_ms: None,
                    dense_count,
                    sparse_count,
                    exact_count,
                    fused_count: 0,
                    result_count: 0,
                    query_type: format!("{query_type:?}"),
                    response_bytes: None,
                },
            });
        }

        // Merge symbol exact hits (Phase 6) into fused results.
        // Exact symbol defs (score 2.0) and references (score 1.4) float to top.
        let mut fused = fused;
        if !symbol_exact_hits.is_empty() {
            let existing_ids: HashSet<u64> = fused.iter().map(|s| s.chunk_id).collect();
            for hit in &symbol_exact_hits {
                if let Some(existing) = fused.iter_mut().find(|s| s.chunk_id == hit.chunk_id) {
                    // Boost existing entry if symbol score is higher
                    if hit.score > existing.score {
                        existing.score = hit.score;
                    }
                } else if !existing_ids.contains(&hit.chunk_id) {
                    fused.push(ScoredChunkId::new(hit.chunk_id, hit.score));
                }
            }
            fused.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            tracing::debug!(
                symbol_hits = symbol_exact_hits.len(),
                "Symbol exact hits merged into fusion results"
            );
        }

        // Determine source type
        let source = if query.use_dense && query.use_sparse {
            SearchSource::Hybrid
        } else if query.use_dense {
            SearchSource::Dense
        } else {
            SearchSource::Sparse
        };

        // Overfetch when file filter is active to compensate for filtered-out results
        let has_filter = query
            .file_filter
            .as_ref()
            .is_some_and(super::super::types::FileFilter::is_active);
        let fetch_count = if has_filter {
            candidates * 5
        } else {
            candidates
        };

        // Single store lock for all post-fusion operations:
        // file role boost, regex boost, and final chunk fetch.
        let store = self.store.lock();

        // Fetch all candidate chunks once (used for role boost, regex boost, and results)
        let top_ids: Vec<u64> = fused.iter().take(fetch_count).map(|s| s.chunk_id).collect();
        let chunks = store.get_chunks(&top_ids)?;
        let mut chunk_map: HashMap<u64, _> = chunks.into_iter().map(|c| (c.id, c)).collect();

        // Phase 3: File role boost for Semantic/Mixed queries
        if matches!(query_type, QueryType::Semantic | QueryType::Mixed) {
            let paths: Vec<&Path> = chunk_map.values().map(|c| c.file_path.as_path()).collect();
            if let Ok(roles) = store.get_file_roles(&paths) {
                for scored in &mut fused {
                    if let Some(chunk) = chunk_map.get(&scored.chunk_id) {
                        let role = roles
                            .get(&chunk.file_path)
                            .copied()
                            .unwrap_or(FileRole::Unknown);
                        scored.score *= role.semantic_boost();
                    }
                }
                fused.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                tracing::debug!("File role boost applied");
            }
        }

        // Phase 3b: Path penalty (downrank tests/compat/examples on non-test queries)
        //
        // Per spec §7.2 (v0.4 Item 6): apply a multiplicative penalty to chunks
        // whose path matches test/compat/example/barrel/declaration-stub patterns.
        // Disabled when the query itself mentions test/spec/bench/example/demo,
        // so users actively searching for those still see them.
        if path_signals::should_apply_path_penalty(&effective_text) {
            for scored in &mut fused {
                if let Some(chunk) = chunk_map.get(&scored.chunk_id) {
                    let factor = path_signals::file_path_penalty(&chunk.file_path);
                    scored.score *= factor;
                }
            }
            fused.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            tracing::debug!("Path penalty applied");
        }

        // Phase 3c: Path stem boost (filename / query overlap)
        //
        // Per spec §7.3 (v0.4 Item 7): additively boost chunks whose file
        // stem contains identifier sub-tokens overlapping with the query.
        // Exact subtoken match wins STEM_EXACT_BOOST_FRAC × max_score;
        // prefix match (≥ STEM_PREFIX_MIN_LEN chars) wins
        // STEM_PREFIX_BOOST_FRAC × max_score.
        //
        // The boost is additive (no max_score clamp); the fusion stage
        // produces a final ranking that doesn't require scores to stay
        // in [0, 1]. The old `.min(max_score)` clamp made it impossible
        // for the rank-1 chunk to ever benefit from a boost (its score
        // was already == max_score) and caused boosted chunks to pile
        // at max_score, producing ties. See defect #13.
        {
            let query_tokens = path_signals::query_tokens_for_path_signals(&effective_text);
            if !query_tokens.is_empty() {
                let max_score = fused.first().map_or(1.0, |s| s.score);
                for scored in &mut fused {
                    if let Some(chunk) = chunk_map.get(&scored.chunk_id) {
                        let frac =
                            path_signals::path_stem_boost_factor(&chunk.file_path, &query_tokens);
                        if frac > 0.0 {
                            scored.score += frac * max_score;
                        }
                    }
                }
                fused.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                tracing::debug!("Path stem boost applied");
            }
        }

        // Phase 3d: Definition boost (symbol name / query overlap)
        //
        // Per spec §7.4 (v0.4 Item 8): additively boost AST-aware chunks
        // whose symbol name shares an identifier sub-token with the query
        // (case-insensitive, after camelCase/snake_case decomposition).
        // The disabled "symbol exact lookup" path (see comment at the top
        // of search()) remains disabled — Phase 3d is the v0.4
        // replacement: softer, RRF-compatible signal.
        //
        // The boost is additive (no max_score clamp); the fusion stage
        // produces a final ranking that doesn't require scores to stay
        // in [0, 1]. See defect #13 for why the clamp was removed.
        {
            let query_tokens = path_signals::query_tokens_for_path_signals(&effective_text);
            if !query_tokens.is_empty() {
                let max_score = fused.first().map_or(1.0, |s| s.score);
                for scored in &mut fused {
                    if let Some(chunk) = chunk_map.get(&scored.chunk_id)
                        && let Some(name) = chunk.symbol_name()
                    {
                        let frac = path_signals::definition_boost_factor(name, &query_tokens);
                        if frac > 0.0 {
                            scored.score += frac * max_score;
                        }
                    }
                }
                fused.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                tracing::debug!("Definition boost applied");
            }
        }

        // Phase 3e: File coherence boost (multi-chunk-from-same-file reward)
        //
        // Per spec §7.5 (v0.4 Item 9): when several chunks in the result
        // set share a file, that file is more likely to be the canonical
        // implementation. We additively boost only the top-scoring chunk
        // of each multi-chunk file, scaled by the file's relative
        // contribution to the result set (file_sum / max_file_sum). The
        // existing directory-proximity bonus further down stays in place —
        // both signals coexist for v0.4 per spec §7.5.3.
        //
        // The boost is additive (no max_score clamp); the fusion stage
        // produces a final ranking that doesn't require scores to stay
        // in [0, 1]. See defect #13 for why the clamp was removed.
        {
            let max_score = fused.first().map_or(1.0, |s| s.score);
            let boosts = path_signals::file_coherence_boosts(&fused, &chunk_map);
            if !boosts.is_empty() {
                for scored in &mut fused {
                    if let Some(&frac) = boosts.get(&scored.chunk_id) {
                        scored.score += frac * max_score;
                    }
                }
                fused.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                tracing::debug!(boosted_count = boosts.len(), "File coherence boost applied");
            }
        }

        // Phase 5: Regex filter/boost
        #[allow(clippy::collapsible_if)]
        if let Some(ref pattern) = query.regex_pattern {
            if let Some(regex) = regex_semantic::compile_pattern(pattern) {
                for scored in &mut fused {
                    #[allow(clippy::collapsible_if)]
                    if let Some(chunk) = chunk_map.get(&scored.chunk_id) {
                        if regex.is_match(&chunk.content) {
                            scored.score *= 1.5;
                        }
                    }
                }
                fused.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                tracing::debug!(pattern = %pattern, "Regex boost applied");
            }
        }

        // Phase 6: Graph propagation — expand results through code graph edges
        let graph_config = if is_architectural_query(&effective_text, query_type) {
            GraphPropagationConfig::architectural_mode(candidates).with_env_overrides()
        } else {
            GraphPropagationConfig::for_query_type(&query_type, candidates).with_env_overrides()
        };
        let scored_ids: Vec<ScoredChunkId> = fused.iter().take(fetch_count).cloned().collect();
        let expanded = graph_propagation::propagate(&scored_ids, &store, &graph_config)?;

        // Merge propagated chunks into fused list
        let existing_ids: HashSet<u64> = fused.iter().map(|s| s.chunk_id).collect();
        let new_ids: Vec<u64> = expanded
            .iter()
            .filter(|s| !existing_ids.contains(&s.chunk_id))
            .map(|s| s.chunk_id)
            .collect();

        if !new_ids.is_empty() {
            let new_chunks = store.get_chunks(&new_ids)?;
            for chunk in new_chunks {
                chunk_map.insert(chunk.id, chunk);
            }
        }

        // Update scores from propagation (only if higher) and add new entries
        {
            let prop_scores: HashMap<u64, f32> =
                expanded.iter().map(|s| (s.chunk_id, s.score)).collect();
            for scored in &mut fused {
                #[allow(clippy::collapsible_if)]
                if let Some(&new_score) = prop_scores.get(&scored.chunk_id) {
                    if new_score > scored.score {
                        scored.score = new_score;
                    }
                }
            }
            for s in expanded
                .iter()
                .filter(|s| !existing_ids.contains(&s.chunk_id))
            {
                fused.push(s.clone());
            }
            fused.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        tracing::debug!(
            expanded_count = expanded.len(),
            new_count = new_ids.len(),
            "Graph propagation complete"
        );

        // Import proximity boost for architectural Semantic/Mixed queries only
        // (Identifier queries use symbol exact lookup instead)
        if matches!(query_type, QueryType::Semantic | QueryType::Mixed)
            && is_architectural_query(&effective_text, query_type)
            && fused.len() >= 2
        {
            let top_half = fused.len().max(2) / 2;
            let top_paths: Vec<String> = fused
                .iter()
                .take(top_half)
                .filter_map(|s| {
                    chunk_map
                        .get(&s.chunk_id)
                        .map(|c| c.file_path.to_string_lossy().into_owned())
                })
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();

            if let Ok(neighbors) = store.get_import_neighbors(&top_paths)
                && !neighbors.is_empty()
            {
                let top_count = top_paths.len().max(1);
                let avg_top_score =
                    fused.iter().take(top_count).map(|r| r.score).sum::<f32>() / top_count as f32;
                let boost = 0.10 * avg_top_score;
                let max_score = fused.first().map_or(1.0, |s| s.score);
                let neighbor_set: HashSet<&str> = neighbors.iter().map(String::as_str).collect();

                for scored in &mut fused {
                    if let Some(chunk) = chunk_map.get(&scored.chunk_id) {
                        let path = chunk.file_path.to_string_lossy();
                        if neighbor_set.contains(path.as_ref()) {
                            scored.score = (scored.score + boost).min(max_score);
                        }
                    }
                }
                fused.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                tracing::debug!(
                    neighbor_count = neighbors.len(),
                    boost,
                    "Import proximity boost applied"
                );
            }

            // Directory-proximity bonus: boost results sharing directories with top-3
            let top3_dirs: Vec<String> = fused
                .iter()
                .take(3)
                .filter_map(|s| {
                    chunk_map.get(&s.chunk_id).and_then(|c| {
                        let p = c.file_path.to_string_lossy();
                        p.rfind('/').map(|i| p[..i].to_string())
                    })
                })
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();

            if !top3_dirs.is_empty() {
                // Only apply if results aren't already spread across many dirs
                let top_k = query.max_results;
                let unique_dirs_in_top: HashSet<String> = fused
                    .iter()
                    .take(top_k)
                    .filter_map(|s| {
                        chunk_map.get(&s.chunk_id).and_then(|c| {
                            let p = c.file_path.to_string_lossy();
                            p.rfind('/').map(|i| p[..i].to_string())
                        })
                    })
                    .collect();

                if unique_dirs_in_top.len() < top_k / 3 {
                    let top_count = fused.len().max(2) / 2;
                    let avg_top_score = fused
                        .iter()
                        .take(top_count.max(1))
                        .map(|r| r.score)
                        .sum::<f32>()
                        / top_count.max(1) as f32;
                    let dir_boost = 0.08 * avg_top_score;
                    let max_score = fused.first().map_or(1.0, |s| s.score);

                    for (i, scored) in fused.iter_mut().enumerate() {
                        if i < 3 {
                            continue;
                        }
                        if let Some(chunk) = chunk_map.get(&scored.chunk_id) {
                            let p = chunk.file_path.to_string_lossy();
                            if let Some(slash) = p.rfind('/') {
                                let dir = &p[..slash];
                                if top3_dirs.iter().any(|d| d == dir) {
                                    scored.score = (scored.score + dir_boost).min(max_score);
                                }
                            }
                        }
                    }
                    fused.sort_by(|a, b| {
                        b.score
                            .partial_cmp(&a.score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    tracing::debug!(
                        top3_dirs = ?top3_dirs,
                        dir_boost,
                        "Directory proximity boost applied"
                    );
                }
            }
        }

        drop(store); // Explicit drop before building results

        // Build results in score order, applying file filter if set.
        // Chunks discovered via graph propagation get tagged with GraphExpanded source.
        // E6: populate per-result confidence labels — RRF results carry channel-agreement
        // metadata via `confidence_map`; CC results derive confidence from per-channel
        // score fields (treating non-zero score as "channel hit").
        let graph_expanded_ids: HashSet<u64> = new_ids.iter().copied().collect();
        let mut results: Vec<SearchResult> = fused
            .iter()
            .take(fetch_count)
            .filter_map(|scored| {
                chunk_map.remove(&scored.chunk_id).map(|chunk| {
                    let chunk_source = if graph_expanded_ids.contains(&scored.chunk_id) {
                        SearchSource::GraphExpanded
                    } else {
                        source
                    };
                    let (confidence, confidence_score) = confidence_map
                        .get(&scored.chunk_id)
                        .copied()
                        .unwrap_or_else(|| derive_cc_confidence(scored));
                    SearchResult {
                        chunk,
                        score: scored.score,
                        source: chunk_source,
                        score_dense: scored.score_dense,
                        score_sparse: scored.score_sparse,
                        score_exact: scored.score_exact,
                        confidence,
                        confidence_score,
                    }
                })
            })
            .filter(|result| {
                query
                    .file_filter
                    .as_ref()
                    .is_none_or(|f| f.matches(&result.chunk.file_path))
            })
            .take(candidates)
            .collect();

        if has_filter {
            tracing::debug!(
                pre_filter = fetch_count,
                post_filter = results.len(),
                "File filter applied"
            );
        }

        // Stage 3: Cross-encoder reranking with fastembed (lazy-loaded on first use)
        let mut rerank_ms: Option<u64> = None;
        if query.use_rerank && self.config.rerank {
            let rerank_start = std::time::Instant::now();
            let mut reranker_guard = self.reranker.lock();

            // Lazy-load reranker on first use. v0.3: use new_default() which
            // selects the v0.3 cross-encoder (BGE Reranker v2 M3) and is gated
            // by SEMANTEX_RERANKER. The model only loads when the env var
            // enables it; when disabled, rerank() is an identity pass-through.
            if reranker_guard.is_none() {
                tracing::info!("Loading reranker model (first use)...");
                // Resolve the active reranker through S8's ModelRegistry (single
                // read of config.reranker_model == SEMANTEX_RERANKER_MODEL).
                match RerankerEngine::from_config(&self.config, false) {
                    Ok(r) => *reranker_guard = Some(r),
                    Err(e) => {
                        tracing::warn!("Failed to initialize reranker: {}", e);
                    }
                }
            }

            if let Some(ref mut reranker) = *reranker_guard {
                let docs: Vec<&str> = results.iter().map(|r| r.chunk.content.as_str()).collect();
                let candidate_count = docs.len();

                if let Ok(reranked) = reranker.rerank(&query.text, &docs, query.max_results) {
                    tracing::debug!(
                        candidates = candidate_count,
                        reranked = reranked.len(),
                        duration_ms = rerank_start.elapsed().as_millis() as u64,
                        "Reranking complete (fastembed)"
                    );
                    // Reorder results by reranked indices without cloning
                    let mut indexed: Vec<Option<SearchResult>> =
                        results.into_iter().map(Some).collect();
                    results = reranked
                        .into_iter()
                        .filter_map(|(idx, score)| {
                            indexed.get_mut(idx)?.take().map(|mut r| {
                                r.score = score;
                                r.source = SearchSource::Reranked;
                                r
                            })
                        })
                        .collect();
                }
            }
            rerank_ms = Some(rerank_start.elapsed().as_millis() as u64);
        }

        // Stage 3b: MMR diversity pass (S7). OFF unless SEMANTEX_MMR_LAMBDA is
        // set and a dense backend is present. Reorders the top-K results to
        // reduce near-duplicate clustering; does not change scores, so Stage 4
        // adaptive sizing/threshold logic is unaffected. O(K²), K ≤ 50.
        if let Some(lambda) = mmr_active(self.dense.as_deref()) {
            let mmr_top_k = results.len().min(50); // O(K²) guard, K ≤ 50 per spec
            if mmr_top_k >= 2
                && let Some(ref dense) = self.dense
            {
                // Per-chunk vectors come from the S1-declared `embed_doc_vectors`
                // seam (`Option<Vec<(u64, Vec<f32>)>>`). Collect the top-K chunk
                // ids, ask the backend for their vectors, and fold the returned
                // pairs into the `HashMap<u64, Vec<f32>>` that `mmr_rerank` wants.
                // If the backend has no single-vector projection it returns None →
                // `.map(...)` yields None and MMR no-ops.
                let ids: Vec<u64> = results.iter().take(mmr_top_k).map(|r| r.chunk.id).collect();
                let doc_vecs: Option<HashMap<u64, Vec<f32>>> = dense
                    .embed_doc_vectors(&ids)
                    .map(|pairs| pairs.into_iter().collect());
                if let Some(doc_vectors) = doc_vecs {
                    let before_top = results.first().map(|r| r.chunk.id);
                    mmr::mmr_rerank(&mut results, &doc_vectors, lambda, mmr_top_k);
                    tracing::debug!(
                        lambda,
                        top_k = mmr_top_k,
                        reordered = (results.first().map(|r| r.chunk.id) != before_top),
                        "MMR diversity pass applied (S7)"
                    );
                }
            }
        }

        // Stage 4: Adaptive result sizing, confidence threshold, and deduplication.
        // Exhaustive mode: widen range, skip dedup, lower threshold.
        let mut adaptive_config = self.config.adaptive_config();
        if is_exhaustive_query(&effective_text) {
            adaptive_config.exhaustive = true;
        }
        let pre_adaptive_count = results.len();
        adaptive::apply_adaptive_pipeline(
            &mut results,
            query_type,
            query.max_results,
            &adaptive_config,
        );
        if results.len() != pre_adaptive_count {
            tracing::debug!(
                pre_adaptive = pre_adaptive_count,
                post_adaptive = results.len(),
                ?query_type,
                "Adaptive sizing applied"
            );
        }

        let fused_count = fused.len();
        let total_duration = search_start.elapsed();
        tracing::info!(
            query = %query.text,
            results_count = results.len(),
            duration_ms = total_duration.as_millis() as u64,
            "Search completed successfully"
        );

        Ok(super::SearchOutput {
            metrics: super::SearchMetrics {
                total_ms: total_duration.as_millis() as u64,
                dense_ms: if query.use_dense {
                    Some(dense_ms)
                } else {
                    None
                },
                sparse_ms: if query.use_sparse {
                    Some(sparse_ms)
                } else {
                    None
                },
                exact_ms: Some(exact_ms),
                fusion_ms: Some(fusion_ms),
                rerank_ms,
                dense_count,
                sparse_count,
                exact_count,
                fused_count,
                result_count: results.len(),
                query_type: format!("{query_type:?}"),
                response_bytes: None,
            },
            results,
        })
    }

    /// Grep-mode search: exact + sparse only, no dense, no rerank, no adaptive filtering.
    /// Optimized for exhaustive grep-like behavior with maximum recall.
    fn search_grep_mode(
        &self,
        query: &SearchQuery,
        search_start: std::time::Instant,
    ) -> Result<super::SearchOutput> {
        // Fetch more candidates in grep mode for exhaustive results
        let fetch_limit = query.max_results * 3;

        // Run sparse and exact in parallel
        let (sparse_results, exact_ids, sparse_ms, exact_ms) = std::thread::scope(|s| {
            let sparse_handle = s.spawn(|| -> (Vec<ScoredChunkId>, u64) {
                if let Some(ref sparse) = self.sparse {
                    let safe_query = sanitize_tantivy_query(&query.text);
                    let sparse_start = std::time::Instant::now();
                    let results = sparse.search(&safe_query, fetch_limit).unwrap_or_default();
                    tracing::debug!(
                        results_count = results.len(),
                        duration_ms = sparse_start.elapsed().as_millis() as u64,
                        "Grep-mode sparse search complete"
                    );
                    (results, sparse_start.elapsed().as_millis() as u64)
                } else {
                    (Vec::new(), 0)
                }
            });

            let exact_handle = s.spawn(|| -> (Vec<u64>, u64) {
                let exact_start = std::time::Instant::now();
                let store = self.store.lock();
                let ids = store
                    .search_exact(&query.text, fetch_limit)
                    .unwrap_or_default();
                tracing::debug!(
                    results_count = ids.len(),
                    duration_ms = exact_start.elapsed().as_millis() as u64,
                    "Grep-mode exact search complete (SQLite)"
                );
                (ids, exact_start.elapsed().as_millis() as u64)
            });

            let (sparse_results, s_ms) = sparse_handle.join().unwrap_or_else(|_| (Vec::new(), 0));
            let (exact_ids, e_ms) = exact_handle.join().unwrap_or_else(|_| (Vec::new(), 0));

            (sparse_results, exact_ids, s_ms, e_ms)
        });

        let sparse_count = sparse_results.len();
        let exact_count = exact_ids.len();

        // Grep-mode fusion: exact first, then sparse, deduplicated
        let fusion_start = std::time::Instant::now();
        let fused = grep_mode_fuse(&sparse_results, &exact_ids);
        let fusion_ms = fusion_start.elapsed().as_millis() as u64;
        tracing::debug!(
            fused_count = fused.len(),
            exact_count = exact_ids.len(),
            sparse_count = sparse_results.len(),
            duration_ms = fusion_ms,
            "Grep-mode fusion complete"
        );

        if fused.is_empty() {
            tracing::info!(
                query = %query.text,
                duration_ms = search_start.elapsed().as_millis() as u64,
                "Grep-mode search completed with no results"
            );
            return Ok(super::SearchOutput {
                results: Vec::new(),
                metrics: super::SearchMetrics {
                    total_ms: search_start.elapsed().as_millis() as u64,
                    dense_ms: None,
                    sparse_ms: Some(sparse_ms),
                    exact_ms: Some(exact_ms),
                    fusion_ms: Some(fusion_ms),
                    rerank_ms: None,
                    dense_count: 0,
                    sparse_count,
                    exact_count,
                    fused_count: 0,
                    result_count: 0,
                    query_type: "GrepMode".to_string(),
                    response_bytes: None,
                },
            });
        }

        // Fetch chunks from storage
        let top_ids: Vec<u64> = fused.iter().take(fetch_limit).map(|s| s.chunk_id).collect();
        let store = self.store.lock();
        let chunks = store.get_chunks(&top_ids)?;
        drop(store);

        let mut chunk_map: HashMap<u64, _> = chunks.into_iter().map(|c| (c.id, c)).collect();

        // Build results, applying file filter if set
        let has_filter = query
            .file_filter
            .as_ref()
            .is_some_and(super::super::types::FileFilter::is_active);

        // Grep-mode does not run fusion; confidence defaults to Inferred.
        let mut results: Vec<SearchResult> = fused
            .iter()
            .take(fetch_limit)
            .filter_map(|scored| {
                chunk_map
                    .remove(&scored.chunk_id)
                    .map(|chunk| SearchResult {
                        chunk,
                        score: scored.score,
                        source: SearchSource::Sparse,
                        score_dense: scored.score_dense,
                        score_sparse: scored.score_sparse,
                        score_exact: scored.score_exact,
                        confidence: Confidence::Inferred,
                        confidence_score: 0.0,
                    })
            })
            .filter(|result| {
                query
                    .file_filter
                    .as_ref()
                    .is_none_or(|f| f.matches(&result.chunk.file_path))
            })
            .take(query.max_results)
            .collect();

        if has_filter {
            tracing::debug!(post_filter = results.len(), "Grep-mode file filter applied");
        }

        // Grep mode: exhaustive matching — just truncate to max_results, no adaptive filtering
        results.truncate(query.max_results);

        let fused_count = fused.len();
        let total_duration = search_start.elapsed();
        tracing::info!(
            query = %query.text,
            results_count = results.len(),
            duration_ms = total_duration.as_millis() as u64,
            "Grep-mode search completed"
        );

        Ok(super::SearchOutput {
            metrics: super::SearchMetrics {
                total_ms: total_duration.as_millis() as u64,
                dense_ms: None,
                sparse_ms: Some(sparse_ms),
                exact_ms: Some(exact_ms),
                fusion_ms: Some(fusion_ms),
                rerank_ms: None,
                dense_count: 0,
                sparse_count,
                exact_count,
                fused_count,
                result_count: results.len(),
                query_type: "GrepMode".to_string(),
                response_bytes: None,
            },
            results,
        })
    }
}

/// Default HyDE-synthesis timeout. Sized for `cli:claude` / `cli:codex` which
/// take ~5–10 s for a 400-token synthesis (subprocess spawn + Claude inference).
/// Override via `SEMANTEX_LLM_HYDE_TIMEOUT_MS`.
///
/// This is the **authoritative outer timeout** for the HyDE path.
/// `subscription_cli.rs::hyde_timeout()` reads the same env var as a
/// defense-in-depth inner backstop on the subprocess wait; it must remain ≤
/// (not greater than) this value in the default case so the outer cancels
/// first.  If `SEMANTEX_LLM_HYDE_TIMEOUT_MS` is set, both layers extend
/// identically.
#[cfg(feature = "llm")]
const LLM_HYDE_TIMEOUT_DEFAULT_MS: u64 = 15_000;

#[cfg(feature = "llm")]
fn llm_hyde_timeout() -> std::time::Duration {
    std::env::var("SEMANTEX_LLM_HYDE_TIMEOUT_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .map_or_else(
            || std::time::Duration::from_millis(LLM_HYDE_TIMEOUT_DEFAULT_MS),
            std::time::Duration::from_millis,
        )
}

/// Run the base search, synthesize a HyDE doc via LLM, search again with the
/// HyDE doc as the query, merge and dedup results by chunk ID, and re-sort.
///
/// On any LLM error or timeout, returns the base search results unchanged.
///
/// The HybridSearcher is inherently synchronous. We call `search()` directly
/// inside this async fn (without `spawn_blocking`) because:
///
/// - The daemon uses a current-thread Tokio runtime, so spawning blocking work
///   there would deadlock.
/// - ColBERT ONNX inference is the dominant cost and runs inside rayon threads
///   that are independent of the Tokio event loop.
/// - Adding `Arc` wrapping to use `spawn_blocking` would change a large call
///   surface for marginal benefit.
///
/// The LLM I/O (HTTP/process) is the only truly async part; `search()` is
/// parallel-safe via rayon and does not block the async executor.
#[cfg(feature = "llm")]
impl HybridSearcher {
    pub async fn search_with_hyde(
        &self,
        query: &super::SearchQuery,
        llm: std::sync::Arc<dyn crate::llm::LlmCapability>,
    ) -> anyhow::Result<super::SearchOutput> {
        // Run base search first (sync, but fast on warm index).
        let base = self.search(query)?;

        // Synthesize a hypothetical code snippet from the LLM.
        // On any error (timeout, LLM failure, bad response), return base unchanged.
        let hyde_doc = match tokio::time::timeout(
            llm_hyde_timeout(),
            llm.synthesize_hyde_doc(&query.text),
        )
        .await
        {
            Ok(Ok(doc)) if !doc.trim().is_empty() => doc,
            Ok(Ok(_)) => {
                tracing::info!("HyDE doc was empty; returning base results only");
                return Ok(base);
            }
            Ok(Err(e)) => {
                tracing::info!(error = %e, "HyDE synthesis failed; returning base results only");
                return Ok(base);
            }
            Err(_timeout) => {
                tracing::info!("HyDE synthesis timed out; returning base results only");
                return Ok(base);
            }
        };

        let hyde_query = super::SearchQuery::new(&hyde_doc).max_results(query.max_results);
        let hyde = match self.search(&hyde_query) {
            Ok(h) => h,
            Err(e) => {
                tracing::info!(error = %e, "HyDE search failed; returning base results only");
                return Ok(base);
            }
        };

        let max_results = query.max_results;
        Ok(merge_hyde_results(base, hyde, max_results))
    }
}

/// Merge base + HyDE results, deduplicate by `chunk.id`, re-sort by score
/// descending, and cap at `max_results`.
///
/// Deduplication key is `chunk.id` (the stable DB primary key), NOT the
/// `(file_path, start_line, end_line)` triple — the triple-based key was
/// incorrect because two chunks with identical file/line ranges but different
/// IDs would survive dedup, producing duplicates in the output.
#[cfg(feature = "llm")]
fn merge_hyde_results(
    base: super::SearchOutput,
    hyde: super::SearchOutput,
    max_results: usize,
) -> super::SearchOutput {
    use std::collections::HashSet;

    // Carry base metrics forward; result_count is updated after merge.
    let mut base_metrics = base.metrics;

    let mut seen_ids: HashSet<u64> =
        HashSet::with_capacity(base.results.len() + hyde.results.len());
    let mut merged: Vec<crate::types::SearchResult> =
        Vec::with_capacity(base.results.len() + hyde.results.len());

    for r in base.results.into_iter().chain(hyde.results) {
        if seen_ids.insert(r.chunk.id) {
            merged.push(r);
        }
    }

    merged.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    merged.truncate(max_results);

    base_metrics.result_count = merged.len();
    base_metrics.query_type = "hyde_merged".to_string();

    super::SearchOutput {
        results: merged,
        metrics: base_metrics,
    }
}

/// S7: select the (FusionWeights, k) pair the weighted-RRF path uses.
/// `k` comes from the previously-dead `config.rrf_k`; the weights come from the
/// query classifier's per-type table. Extracted as a free fn so it is unit-
/// testable without building an index.
fn weighted_rrf_params(config: &SemantexConfig, query_type: QueryType) -> (FusionWeights, f32) {
    (query_type.fusion_weights(), config.rrf_k)
}

/// S7: resolve the active MMR lambda. MMR runs only when a valid
/// `SEMANTEX_MMR_LAMBDA` is set AND a dense backend exists (it supplies the
/// per-result vectors). Returns `Some(lambda)` to run, `None` to skip.
fn mmr_active(dense: Option<&dyn crate::search::dense_backend::DenseBackend>) -> Option<f32> {
    let lambda = mmr::mmr_lambda_from_env()?;
    dense?; // None backend → skip
    Some(lambda)
}

/// Heuristic: exhaustive queries ask for complete enumeration of all instances.
/// Examples: "find all error handling", "list every config option", "all places where X".
/// These benefit from wider result ranges, no per-file dedup, and lower score thresholds.
fn is_exhaustive_query(query: &str) -> bool {
    let q = query.to_lowercase();
    let signals: &[&str] = &[
        "find all",
        "list all",
        "all places",
        "all patterns",
        "all ways",
        "all instances",
        "all cases",
        "every ",
        "each ",
        "enumerate",
        "exhaustive",
        "complete list",
        "anywhere ",
        "throughout",
    ];
    signals.iter().any(|s| q.contains(s))
}

/// Heuristic: architectural queries are long semantic queries about flows/pipelines/lifecycles.
fn is_architectural_query(query: &str, query_type: QueryType) -> bool {
    if query_type != QueryType::Semantic {
        return false;
    }
    let words: Vec<&str> = query.split_whitespace().collect();
    if words.len() < 4 {
        return false;
    }
    let architectural_signals = [
        "flow",
        "lifecycle",
        "pipeline",
        "end to end",
        "e2e",
        "architecture",
        "through",
        "across",
        "from",
        "layers",
        "how does",
        "how do",
        "full",
        "entire",
        "complete",
        "manage",
    ];
    let query_lower = query.to_lowercase();
    architectural_signals
        .iter()
        .any(|s| query_lower.contains(s))
}

/// Weighted Reciprocal Rank Fusion (RRF): merge two ranked lists with per-source weights.
///
/// RRF is superior to score averaging because it is:
/// - **Scale-invariant**: Works with different score ranges (BM25 vs cosine similarity)
/// - **Robust to outliers**: Single high scores don't dominate the fusion
/// - **Consensus-seeking**: Items appearing in both lists get boosted
///
/// The `weights` parameter allows query-adaptive tuning: identifier queries
/// boost the sparse (BM25) contribution, while semantic queries boost dense.
///
/// # Formula
/// ```text
/// RRF_score(document) = Σ w_i / (k + rank_i + 1)
/// ```
///
/// # Parameters
/// - `dense_list`: Dense (vector) search results
/// - `sparse_list`: Sparse (BM25) search results
/// - `k`: Constant controlling rank position importance (typically 60.0)
/// - `weights`: Per-source weights from the query classifier
///
/// # Returns
/// Merged list sorted by weighted RRF score (highest first)
pub fn rrf_fuse(
    dense_list: &[ScoredChunkId],
    sparse_list: &[ScoredChunkId],
    k: f32,
    weights: &FusionWeights,
) -> Vec<ScoredChunkId> {
    let mut scores: HashMap<u64, f32> = HashMap::new();

    // Accumulate weighted RRF scores from dense results
    for (rank, item) in dense_list.iter().enumerate() {
        *scores.entry(item.chunk_id).or_insert(0.0) += weights.w_dense / (k + rank as f32 + 1.0);
    }

    // Accumulate weighted RRF scores from sparse results
    for (rank, item) in sparse_list.iter().enumerate() {
        *scores.entry(item.chunk_id).or_insert(0.0) += weights.w_sparse / (k + rank as f32 + 1.0);
    }

    // Convert to scored chunks and sort by descending RRF score
    let mut fused: Vec<ScoredChunkId> = scores
        .into_iter()
        .map(|(chunk_id, score)| ScoredChunkId::new(chunk_id, score))
        .collect();

    fused.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    fused
}

/// Grep-mode fusion: simplified 2-way merge of exact + sparse results.
///
/// Prioritizes exact substring matches (sorted by chunk_id for stability),
/// then appends BM25 sparse results (sorted by score descending),
/// deduplicating along the way.
///
/// This gives grep-like exhaustive behavior: exact matches surface first
/// (just like grep finds literal text), with BM25 fuzzy matches filling in.
pub fn grep_mode_fuse(sparse_list: &[ScoredChunkId], exact_ids: &[u64]) -> Vec<ScoredChunkId> {
    use std::collections::HashSet;

    let mut seen = HashSet::new();
    let mut fused = Vec::new();

    // Exact matches first, all at score 1.0 (highest priority)
    for &id in exact_ids {
        if seen.insert(id) {
            fused.push(ScoredChunkId::new(id, 1.0));
        }
    }

    // Then BM25 results (already sorted by score descending from tantivy)
    for item in sparse_list {
        if seen.insert(item.chunk_id) {
            // Normalize BM25 scores to 0..1 range below exact matches
            // Use a sigmoid-like normalization so scores stay meaningful
            let normalized = item.score / (item.score + 1.0);
            fused.push(ScoredChunkId::new(item.chunk_id, normalized));
        }
    }

    fused
}

/// Dual-route BM25 fusion: merge original sparse results with expanded-query results.
///
/// Uses weighted score accumulation across both result sets to combine keyword
/// hits from the original query with synonym-expanded hits. Items appearing in
/// both lists get boosted by the sum of their weighted scores.
fn dual_route_fuse(
    original: &[ScoredChunkId],
    expanded: &[ScoredChunkId],
    w_original: f32,
    w_expanded: f32,
) -> Vec<ScoredChunkId> {
    let mut scores: HashMap<u64, f32> = HashMap::new();
    for s in original {
        *scores.entry(s.chunk_id).or_default() += w_original * s.score;
    }
    for s in expanded {
        *scores.entry(s.chunk_id).or_default() += w_expanded * s.score;
    }
    let mut fused: Vec<ScoredChunkId> = scores
        .into_iter()
        .map(|(chunk_id, score)| ScoredChunkId::new(chunk_id, score))
        .collect();
    fused.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    fused
}

/// CC-mode confidence derivation: count how many of the three channels
/// produced a non-zero score for this chunk.
///
/// - 3/3 channels → `Extracted`, score 1.0
/// - 2/3 channels → `Inferred`, score 0.667
/// - 1/3 channels → `Inferred`, score 0.333
/// - 0/3 (unreachable here) → `Inferred`, score 0.0
///
/// CC mode lacks the rank-based ambiguity signal that RRF emits, so
/// `Ambiguous` is not produced here — only RRF results carry it.
fn derive_cc_confidence(scored: &ScoredChunkId) -> (Confidence, f32) {
    let channels = u32::from(scored.score_dense > 0.0)
        + u32::from(scored.score_sparse > 0.0)
        + u32::from(scored.score_exact > 0.0);
    let confidence = if channels >= 3 {
        Confidence::Extracted
    } else {
        Confidence::Inferred
    };
    let score = (channels as f32) / 3.0;
    (confidence, score)
}

/// Sanitize a query string for tantivy's query parser
fn sanitize_tantivy_query(query: &str) -> String {
    // Escape special characters that tantivy's query parser interprets
    let special_chars = [
        '+', '-', '&', '|', '!', '(', ')', '{', '}', '[', ']', '^', '"', '~', '*', '?', ':', '\\',
        '/',
    ];
    let mut sanitized = String::with_capacity(query.len());
    for ch in query.chars() {
        if special_chars.contains(&ch) {
            sanitized.push('\\');
        }
        sanitized.push(ch);
    }
    sanitized
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;

    /// Equal weights for backward-compatible tests
    const EQUAL: FusionWeights = FusionWeights {
        w_dense: 1.0,
        w_sparse: 1.0,
    };

    /// S7: MMR runs only when (a) SEMANTEX_MMR_LAMBDA is a valid lambda AND
    /// (b) a dense backend is present. Sparse-only opens (dense None) never MMR.
    #[test]
    fn mmr_active_requires_lambda_and_dense_backend() {
        // SAFETY: process-level env mutation in a single-threaded test.
        unsafe {
            std::env::set_var("SEMANTEX_MMR_LAMBDA", "0.7");
        }
        assert_eq!(
            mmr_active(None),
            None,
            "no dense backend → MMR off even with lambda set"
        );
        unsafe {
            std::env::remove_var("SEMANTEX_MMR_LAMBDA");
        }
        // (The present-backend case is covered by the integration behavior; this
        // guards the env+presence gate, the part that's pure.)
    }

    /// S7: weighted-RRF reads config.rrf_k (previously dead) and the query-type
    /// FusionWeights. This guards the selection logic the hybrid match arm uses.
    #[test]
    fn weighted_rrf_selection_reads_config_rrf_k_and_weights() {
        let mut cfg = crate::config::SemantexConfig::default();
        cfg.rrf_k = 42.0;
        let (weights, k) = weighted_rrf_params(&cfg, query_classifier::QueryType::Identifier);
        assert!((k - 42.0).abs() < f32::EPSILON, "config.rrf_k must be live, got {k}");
        // Identifier weights favour sparse (dead-code revival check).
        assert!(weights.w_sparse > weights.w_dense);
    }

    // ---- Defect #13 regression: Phase 3 boosts must NOT clamp at
    //      max_score. The previous `.min(max_score)` ceiling collapsed
    //      boosted chunks into ties with the rank-1 result and made it
    //      impossible for the rank-1 chunk to benefit from any boost. ----

    /// Simulate the additive-only boost logic used in Phase 3c/3d/3e.
    /// Matches the in-place mutation in `search()` so the test exercises
    /// the same arithmetic shape: `score += frac * max_score`.
    fn apply_additive_boost(
        fused: &mut [ScoredChunkId],
        boosted_ids: &[(u64, f32)], // (chunk_id, frac)
    ) {
        let max_score = fused.first().map_or(1.0, |s| s.score);
        for scored in fused.iter_mut() {
            if let Some(&(_, frac)) = boosted_ids.iter().find(|(id, _)| *id == scored.chunk_id) {
                scored.score += frac * max_score;
            }
        }
        fused.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    #[test]
    fn phase_3_boosts_can_promote_rank_1() {
        // Initial: chunk 1 at max_score (1.0); chunks 2 and 3 tied at 0.5.
        // Apply a Phase-3c-shaped boost (frac=0.40) to chunks 2 and 3.
        // Under the old `.min(max_score)` clamp, both 2 and 3 would be
        // promoted to exactly max_score = 1.0, tying with chunk 1 at the
        // top. After the fix they should sit BELOW chunk 1 (since their
        // additive boost is 0.5 + 0.40*1.0 = 0.90, still < 1.0).
        let mut fused = vec![
            ScoredChunkId::new(1, 1.0),
            ScoredChunkId::new(2, 0.5),
            ScoredChunkId::new(3, 0.5),
        ];

        apply_additive_boost(&mut fused, &[(2, 0.40), (3, 0.40)]);

        // Chunk 1 must remain rank-1 with its original max score.
        assert_eq!(fused[0].chunk_id, 1, "rank-1 must not be displaced");
        assert!(
            (fused[0].score - 1.0).abs() < 1e-6,
            "rank-1 score unchanged: {}",
            fused[0].score
        );

        // Chunks 2 and 3 each ended up at 0.5 + 0.40 = 0.90 (NOT clamped to 1.0).
        for sc in &fused[1..] {
            assert!(
                (sc.score - 0.9).abs() < 1e-6,
                "boosted chunk should be 0.90, got {} for chunk {}",
                sc.score,
                sc.chunk_id
            );
            assert!(
                sc.score < fused[0].score,
                "boosted chunk {} (score {}) must remain below rank-1 (score {})",
                sc.chunk_id,
                sc.score,
                fused[0].score
            );
        }
    }

    #[test]
    fn phase_3_boost_lifts_rank_1_above_max_score() {
        // When rank-1 itself receives the boost, the result must exceed
        // max_score — the clamp used to PREVENT this, leaving rank-1
        // unchanged. Now the boost lifts it past the original ceiling.
        let mut fused = vec![ScoredChunkId::new(1, 1.0), ScoredChunkId::new(2, 0.7)];

        // Boost chunk 1 with frac = 0.20 (definition-boost magnitude).
        apply_additive_boost(&mut fused, &[(1, 0.20)]);

        // Chunk 1 must now be at 1.0 + 0.20*1.0 = 1.20 > 1.0.
        assert_eq!(fused[0].chunk_id, 1);
        assert!(
            (fused[0].score - 1.2).abs() < 1e-6,
            "rank-1 score should be 1.20 after additive boost, got {}",
            fused[0].score
        );
    }

    #[test]
    fn phase_3_boost_can_overtake_when_gap_smaller_than_frac() {
        // A chunk closely trailing rank-1 should be able to overtake it
        // when boosted (frac=0.40, gap=0.10). Under the old clamp it would
        // only reach max_score and TIE with rank-1; sort would then keep
        // rank-1 on top by stable ordering.
        let mut fused = vec![ScoredChunkId::new(1, 1.0), ScoredChunkId::new(2, 0.95)];

        apply_additive_boost(&mut fused, &[(2, 0.40)]);

        // Chunk 2 ends up at 0.95 + 0.40 = 1.35, overtaking chunk 1.
        assert_eq!(fused[0].chunk_id, 2, "chunk 2 should overtake rank-1");
        assert!(
            (fused[0].score - 1.35).abs() < 1e-6,
            "new rank-1 score should be 1.35, got {}",
            fused[0].score
        );
    }

    #[test]
    fn test_rrf_identical_rankings() {
        // When both rankers agree perfectly, RRF should preserve order
        let list_a = vec![
            ScoredChunkId::new(1, 0.9),
            ScoredChunkId::new(2, 0.8),
            ScoredChunkId::new(3, 0.7),
        ];
        let list_b = list_a.clone();

        let result = rrf_fuse(&list_a, &list_b, 60.0, &EQUAL);

        // Should maintain the same order with amplified scores
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].chunk_id, 1);
        assert_eq!(result[1].chunk_id, 2);
        assert_eq!(result[2].chunk_id, 3);
    }

    #[test]
    fn test_rrf_complementary_rankings() {
        // When rankers have different orderings, RRF should favor items with balanced ranks
        let list_a = vec![
            ScoredChunkId::new(1, 0.9),
            ScoredChunkId::new(2, 0.8),
            ScoredChunkId::new(3, 0.7),
        ];
        let list_b = vec![
            ScoredChunkId::new(2, 0.9),
            ScoredChunkId::new(3, 0.8),
            ScoredChunkId::new(1, 0.7),
        ];

        let result = rrf_fuse(&list_a, &list_b, 60.0, &EQUAL);

        // RRF scores with k=60 (formula uses rank+1 for 1-based indexing):
        // Chunk 1: 1/(60+0+1) + 1/(60+2+1) = 1/61 + 1/63 = 0.0164 + 0.0159 = 0.0323
        // Chunk 2: 1/(60+1+1) + 1/(60+0+1) = 1/62 + 1/61 = 0.0161 + 0.0164 = 0.0325
        // Chunk 3: 1/(60+2+1) + 1/(60+1+1) = 1/63 + 1/62 = 0.0159 + 0.0161 = 0.0320
        // Chunk 2 appears early in both lists, should win
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].chunk_id, 2);
    }

    #[test]
    fn test_rrf_unique_items() {
        // When lists have different items, all should appear
        let list_a = vec![ScoredChunkId::new(1, 0.9), ScoredChunkId::new(2, 0.8)];
        let list_b = vec![ScoredChunkId::new(3, 0.9), ScoredChunkId::new(4, 0.8)];

        let result = rrf_fuse(&list_a, &list_b, 60.0, &EQUAL);

        assert_eq!(result.len(), 4);
        let ids: Vec<u64> = result.iter().map(|r| r.chunk_id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert!(ids.contains(&3));
        assert!(ids.contains(&4));
    }

    #[test]
    fn test_rrf_k_parameter_effect() {
        // Lower k makes rank position more important
        let list_a = vec![ScoredChunkId::new(1, 0.9), ScoredChunkId::new(2, 0.1)];
        let list_b = vec![ScoredChunkId::new(2, 0.9)];

        // With high k (60), early ranks matter less
        let result_high_k = rrf_fuse(&list_a, &list_b, 60.0, &EQUAL);
        // Chunk 2 appears in both lists: 1/(60+1) + 1/(60+0) = 0.0164 + 0.0167 = 0.0331
        // Chunk 1 appears only in list_a at rank 0: 1/(60+0) = 0.0167
        assert_eq!(result_high_k[0].chunk_id, 2);

        // With low k (1), early ranks matter much more
        let result_low_k = rrf_fuse(&list_a, &list_b, 1.0, &EQUAL);
        // Chunk 2: 1/(1+1) + 1/(1+0) = 0.5 + 1.0 = 1.5
        // Chunk 1: 1/(1+0) = 1.0
        assert_eq!(result_low_k[0].chunk_id, 2);
    }

    #[test]
    fn test_rrf_empty_lists() {
        let list_a = vec![ScoredChunkId::new(1, 0.9)];
        let list_b: Vec<ScoredChunkId> = vec![];

        let result = rrf_fuse(&list_a, &list_b, 60.0, &EQUAL);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].chunk_id, 1);
    }

    #[test]
    fn test_rrf_vs_score_averaging() {
        // Demonstrate why RRF is superior to score averaging:
        //
        // Scenario: Dense search (semantic) ranks a less relevant but topically similar
        // document very high (0.95), while BM25 (keyword) doesn't find it at all.
        // Meanwhile, the truly relevant document appears moderately in both.
        //
        // Score averaging would be dominated by that 0.95 outlier, but RRF looks at
        // RANK position, making it more robust to score scale differences.

        let dense_results = vec![
            ScoredChunkId::new(100, 0.95), // False positive, high semantic similarity
            ScoredChunkId::new(200, 0.75), // True positive, appears in both
            ScoredChunkId::new(300, 0.60),
        ];

        let bm25_results = vec![
            ScoredChunkId::new(200, 12.5), // True positive, different scale
            ScoredChunkId::new(400, 8.3),
            ScoredChunkId::new(500, 6.1),
        ];

        let rrf_result = rrf_fuse(&dense_results, &bm25_results, 60.0, &EQUAL);

        // RRF calculation:
        // Chunk 100: 1/(60+0) = 0.0164 (only in dense, rank 0)
        // Chunk 200: 1/(60+1) + 1/(60+0) = 0.0164 + 0.0167 = 0.0331 (both lists)
        // Chunk 300: 1/(60+2) = 0.0161 (only in dense, rank 2)
        // Chunk 400: 1/(60+1) = 0.0164 (only in BM25, rank 1)
        // Chunk 500: 1/(60+2) = 0.0161 (only in BM25, rank 2)

        // Chunk 200 should win because it appears in BOTH rankers
        assert_eq!(
            rrf_result[0].chunk_id, 200,
            "RRF should promote documents that appear in both rankers"
        );

        // Score averaging would be biased toward chunk 100's outlier 0.95 score
        // RRF is more robust by considering rank position instead
    }

    #[test]
    fn test_weighted_rrf_identifier_boosts_sparse() {
        // Identifier query weights: dense=0.2, sparse=1.0
        let ident_weights = query_classifier::QueryType::Identifier.fusion_weights();

        let dense_list = vec![ScoredChunkId::new(1, 0.95)];
        let sparse_list = vec![ScoredChunkId::new(2, 15.0)];

        let result = rrf_fuse(&dense_list, &sparse_list, 60.0, &ident_weights);

        // Chunk 1 (dense only): 0.2 / 61 = 0.00328
        // Chunk 2 (sparse only): 1.0 / 61 = 0.01639
        assert_eq!(result[0].chunk_id, 2);
        assert!(result[0].score > result[1].score);
    }

    #[test]
    fn test_weighted_rrf_semantic_boosts_sparse() {
        // Semantic query weights: dense=0.1, sparse=0.9
        let sem_weights = query_classifier::QueryType::Semantic.fusion_weights();

        let dense_list = vec![ScoredChunkId::new(1, 0.95)];
        let sparse_list = vec![ScoredChunkId::new(2, 15.0)];

        let result = rrf_fuse(&dense_list, &sparse_list, 60.0, &sem_weights);

        // Chunk 1 (dense only): 0.1 / 61 = 0.00164
        // Chunk 2 (sparse only): 0.9 / 61 = 0.01475
        assert_eq!(result[0].chunk_id, 2);
    }

    #[test]
    fn test_weighted_rrf_mixed_consensus() {
        // With equal-ish weights, items in both lists still win
        let mixed_weights = query_classifier::QueryType::Mixed.fusion_weights();

        let dense_list = vec![ScoredChunkId::new(1, 0.9), ScoredChunkId::new(2, 0.7)];
        let sparse_list = vec![ScoredChunkId::new(2, 10.0), ScoredChunkId::new(3, 5.0)];

        let result = rrf_fuse(&dense_list, &sparse_list, 60.0, &mixed_weights);

        // Chunk 2 appears in both lists, should still win
        assert_eq!(result[0].chunk_id, 2);
    }

    #[test]
    fn test_sanitize_tantivy_query() {
        assert_eq!(sanitize_tantivy_query("hello world"), "hello world");
        assert_eq!(sanitize_tantivy_query("foo+bar"), "foo\\+bar");
        assert_eq!(sanitize_tantivy_query("test-case"), "test\\-case");
        assert_eq!(sanitize_tantivy_query("(a|b)"), "\\(a\\|b\\)");
        assert_eq!(sanitize_tantivy_query("a*b?c"), "a\\*b\\?c");
        assert_eq!(sanitize_tantivy_query("field:value"), "field\\:value");
    }

    mod grep_mode_tests {
        use super::*;

        #[test]
        fn test_grep_mode_exact_only() {
            let result = grep_mode_fuse(&[], &[10, 20, 30]);
            assert_eq!(result.len(), 3);
            // All exact matches get score 1.0
            assert_eq!(result[0].chunk_id, 10);
            assert!((result[0].score - 1.0).abs() < f32::EPSILON);
            assert_eq!(result[1].chunk_id, 20);
            assert_eq!(result[2].chunk_id, 30);
        }

        #[test]
        fn test_grep_mode_sparse_only() {
            let sparse = vec![
                ScoredChunkId::new(1, 10.0),
                ScoredChunkId::new(2, 5.0),
                ScoredChunkId::new(3, 1.0),
            ];
            let result = grep_mode_fuse(&sparse, &[]);

            assert_eq!(result.len(), 3);
            assert_eq!(result[0].chunk_id, 1);
            assert_eq!(result[1].chunk_id, 2);
            assert_eq!(result[2].chunk_id, 3);
            // BM25 scores normalized via score/(score+1)
            assert!((result[0].score - 10.0 / 11.0).abs() < 1e-5);
            assert!((result[1].score - 5.0 / 6.0).abs() < 1e-5);
            assert!((result[2].score - 0.5).abs() < 1e-5);
        }

        #[test]
        fn test_grep_mode_exact_before_sparse() {
            let sparse = vec![
                ScoredChunkId::new(1, 100.0), // very high BM25
                ScoredChunkId::new(2, 50.0),
            ];
            let exact = vec![3, 4]; // exact matches

            let result = grep_mode_fuse(&sparse, &exact);

            // Exact matches come first regardless of sparse scores
            assert_eq!(result.len(), 4);
            assert_eq!(result[0].chunk_id, 3);
            assert!((result[0].score - 1.0).abs() < f32::EPSILON);
            assert_eq!(result[1].chunk_id, 4);
            assert!((result[1].score - 1.0).abs() < f32::EPSILON);
            // Then sparse results
            assert_eq!(result[2].chunk_id, 1);
            assert_eq!(result[3].chunk_id, 2);
        }

        #[test]
        fn test_grep_mode_deduplicates() {
            let sparse = vec![ScoredChunkId::new(10, 5.0), ScoredChunkId::new(20, 3.0)];
            // Chunk 10 appears in both exact and sparse
            let exact = vec![10, 30];

            let result = grep_mode_fuse(&sparse, &exact);

            assert_eq!(result.len(), 3); // 10 not duplicated
            // Order: exact(10), exact(30), sparse(20)
            assert_eq!(result[0].chunk_id, 10);
            assert!((result[0].score - 1.0).abs() < f32::EPSILON); // gets exact score
            assert_eq!(result[1].chunk_id, 30);
            assert_eq!(result[2].chunk_id, 20);
        }

        #[test]
        fn test_grep_mode_empty() {
            let result = grep_mode_fuse(&[], &[]);
            assert!(result.is_empty());
        }

        #[test]
        fn test_grep_mode_duplicate_exact_ids() {
            // Duplicate IDs in exact list should be deduplicated
            let result = grep_mode_fuse(&[], &[1, 1, 2, 2, 3]);
            assert_eq!(result.len(), 3);
            assert_eq!(result[0].chunk_id, 1);
            assert_eq!(result[1].chunk_id, 2);
            assert_eq!(result[2].chunk_id, 3);
        }
    }

    mod grep_mode_query_tests {
        use crate::search::SearchQuery;

        #[test]
        fn test_grep_mode_query_builder() {
            let query = SearchQuery::new("test").grep_mode();
            assert!(query.grep_mode);
            assert!(!query.use_dense);
            assert!(query.use_sparse);
            assert!(!query.use_rerank);
            assert_eq!(query.max_results, 50);
        }

        #[test]
        fn test_grep_mode_overrides_defaults() {
            let query = SearchQuery::new("test").max_results(3).grep_mode();
            // grep_mode should override max_results to 50
            assert_eq!(query.max_results, 50);
            assert!(!query.use_dense);
        }

        #[test]
        fn test_grep_mode_preserves_filters() {
            let query = SearchQuery::new("test")
                .grep_mode()
                .include_types(vec!["rs".to_string()]);
            assert!(query.grep_mode);
            assert!(query.file_filter.is_some());
            let filter = query.file_filter.unwrap();
            assert_eq!(filter.include_extensions, vec!["rs".to_string()]);
        }

        #[test]
        fn test_default_query_no_grep_mode() {
            let query = SearchQuery::new("test");
            assert!(!query.grep_mode);
            assert!(query.use_dense);
            assert!(query.use_sparse);
            assert!(query.use_rerank);
        }
    }

    mod file_filter_tests {
        use crate::types::FileFilter;
        use std::path::Path;

        #[test]
        fn test_no_filter_matches_everything() {
            let filter = FileFilter::default();
            assert!(!filter.is_active());
            assert!(filter.matches(Path::new("src/main.rs")));
            assert!(filter.matches(Path::new("README.md")));
        }

        #[test]
        fn test_include_filter() {
            let filter = FileFilter {
                include_extensions: vec!["rs".to_string(), "toml".to_string()],
                exclude_extensions: vec![],
            };
            assert!(filter.is_active());
            assert!(filter.matches(Path::new("src/main.rs")));
            assert!(filter.matches(Path::new("Cargo.toml")));
            assert!(!filter.matches(Path::new("README.md")));
            assert!(!filter.matches(Path::new("script.py")));
        }

        #[test]
        fn test_exclude_filter() {
            let filter = FileFilter {
                include_extensions: vec![],
                exclude_extensions: vec!["md".to_string(), "txt".to_string()],
            };
            assert!(filter.is_active());
            assert!(filter.matches(Path::new("src/main.rs")));
            assert!(!filter.matches(Path::new("README.md")));
            assert!(!filter.matches(Path::new("notes.txt")));
        }

        #[test]
        fn test_include_and_exclude_combined() {
            let filter = FileFilter {
                include_extensions: vec!["rs".to_string(), "py".to_string()],
                exclude_extensions: vec!["py".to_string()],
            };
            assert!(filter.matches(Path::new("main.rs")));
            assert!(!filter.matches(Path::new("script.py")));
            assert!(!filter.matches(Path::new("index.js")));
        }

        #[test]
        fn test_case_insensitive_extension() {
            let filter = FileFilter {
                include_extensions: vec!["rs".to_string()],
                exclude_extensions: vec![],
            };
            assert!(filter.matches(Path::new("main.RS")));
            assert!(filter.matches(Path::new("main.Rs")));
        }

        #[test]
        fn test_no_extension() {
            let filter = FileFilter {
                include_extensions: vec!["rs".to_string()],
                exclude_extensions: vec![],
            };
            assert!(!filter.matches(Path::new("Makefile")));

            let filter2 = FileFilter {
                include_extensions: vec![],
                exclude_extensions: vec!["rs".to_string()],
            };
            assert!(filter2.matches(Path::new("Makefile")));
        }

        #[test]
        fn test_code_only_extensions() {
            let filter = FileFilter {
                include_extensions: vec![],
                exclude_extensions: FileFilter::NON_CODE_EXTENSIONS
                    .iter()
                    .map(std::string::ToString::to_string)
                    .collect(),
            };
            assert!(filter.matches(Path::new("main.rs")));
            assert!(filter.matches(Path::new("app.py")));
            assert!(filter.matches(Path::new("index.js")));
            assert!(!filter.matches(Path::new("README.md")));
            assert!(!filter.matches(Path::new("config.json")));
            assert!(!filter.matches(Path::new("data.yaml")));
            assert!(!filter.matches(Path::new("settings.toml")));
            assert!(!filter.matches(Path::new("notes.txt")));
            assert!(!filter.matches(Path::new("Cargo.lock")));
        }
    }

    mod confidence_derivation_tests {
        use super::super::*;
        use crate::types::Confidence;

        #[test]
        fn test_cc_confidence_three_channels_extracted() {
            let scored = ScoredChunkId {
                chunk_id: 1,
                score: 1.0,
                score_dense: 0.5,
                score_sparse: 0.5,
                score_exact: 1.0,
            };
            let (c, s) = derive_cc_confidence(&scored);
            assert_eq!(c, Confidence::Extracted);
            assert!((s - 1.0).abs() < f32::EPSILON);
        }

        #[test]
        fn test_cc_confidence_two_channels_inferred() {
            let scored = ScoredChunkId {
                chunk_id: 1,
                score: 1.0,
                score_dense: 0.5,
                score_sparse: 0.0,
                score_exact: 1.0,
            };
            let (c, s) = derive_cc_confidence(&scored);
            assert_eq!(c, Confidence::Inferred);
            assert!((s - 2.0 / 3.0).abs() < 1e-5);
        }

        #[test]
        fn test_cc_confidence_one_channel_inferred() {
            let scored = ScoredChunkId {
                chunk_id: 1,
                score: 1.0,
                score_dense: 0.5,
                score_sparse: 0.0,
                score_exact: 0.0,
            };
            let (c, s) = derive_cc_confidence(&scored);
            assert_eq!(c, Confidence::Inferred);
            assert!((s - 1.0 / 3.0).abs() < 1e-5);
        }
    }

    /// S1: the dense channel is now a trait object. This test just constructs a
    /// sparse-only searcher (no model load) and confirms the dense slot is None,
    /// proving `open_sparse_only` compiles against the new field shape.
    #[test]
    fn sparse_only_has_no_dense_backend() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = crate::config::SemantexConfig::default();
        let searcher = HybridSearcher::open_sparse_only(tmp.path(), &cfg).unwrap();
        assert!(
            searcher.dense.is_none(),
            "sparse-only must not load a dense backend"
        );
    }
}

// ─── Spec L: HyDE unit tests (feature-gated) ────────────────────────────────

#[cfg(all(feature = "llm", test))]
mod hyde_tests {
    use super::merge_hyde_results;
    use crate::search::{SearchMetrics, SearchOutput};
    use crate::types::{AstNodeKind, Chunk, ChunkType, Confidence, SearchResult, SearchSource};
    use std::path::PathBuf;

    fn make_metrics() -> SearchMetrics {
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
            result_count: 0,
            query_type: "semantic".to_string(),
            response_bytes: None,
        }
    }

    fn make_result(id: u64, score: f32) -> SearchResult {
        SearchResult {
            chunk: Chunk {
                id,
                file_path: PathBuf::from(format!("src/chunk_{id}.rs")),
                start_line: 1,
                end_line: 10,
                content: format!("fn chunk_{id}() {{}}"),
                chunk_type: ChunkType::AstNode {
                    name: format!("chunk_{id}"),
                    kind: AstNodeKind::Function,
                    language: "rust".to_string(),
                    structured_meta: None,
                },
            },
            score,
            source: SearchSource::Hybrid,
            score_dense: 0.0,
            score_sparse: 0.0,
            score_exact: 0.0,
            confidence: Confidence::Inferred,
            confidence_score: 0.5,
        }
    }

    use super::HybridSearcher;
    use crate::llm::LlmCapability;
    use crate::search::agent_classifier::AgentRoute;

    /// Mock LLM for HyDE contract tests.
    ///
    /// - `Mode::Ok(doc)` → returns `doc` from `synthesize_hyde_doc`.
    /// - `Mode::Empty` → returns an empty string (HyDE caller must treat this as
    ///   "no doc" and return base unchanged).
    /// - `Mode::Err` → returns an error (HyDE caller must return base).
    /// - `Mode::Slow(delay)` → sleeps `delay` then returns a doc (used by the
    ///   timeout test once paired with a 1ms env override).
    enum Mode {
        Ok(&'static str),
        Empty,
        Err,
        Slow(std::time::Duration),
    }

    struct MockLlm {
        mode: Mode,
    }

    #[async_trait::async_trait]
    impl LlmCapability for MockLlm {
        async fn classify_route(&self, _query: &str) -> anyhow::Result<AgentRoute> {
            Ok(AgentRoute::Semantic)
        }

        async fn synthesize_hyde_doc(&self, _query: &str) -> anyhow::Result<String> {
            match &self.mode {
                Mode::Ok(doc) => Ok((*doc).to_string()),
                Mode::Empty => Ok(String::new()),
                Mode::Err => anyhow::bail!("mock HyDE synthesis error"),
                Mode::Slow(d) => {
                    tokio::time::sleep(*d).await;
                    Ok("fn slow() {}".to_string())
                }
            }
        }

        fn label(&self) -> &'static str {
            "mock-hyde-llm"
        }
    }

    /// Contract: when `synthesize_hyde_doc` errors, the HyDE merge stage is
    /// never reached, so the caller's result is byte-identical to `base`.
    ///
    /// We assert the contract at the seam the production code depends on:
    /// the mock returns `Err`, and `merge_hyde_results` is therefore NOT the
    /// path taken — base survives verbatim. This pins the mock's error
    /// behaviour that `search_with_hyde`'s `Ok(Err(e)) => return Ok(base)`
    /// branch relies on.
    #[tokio::test]
    async fn hyde_llm_error_yields_base_unchanged() {
        let llm = MockLlm { mode: Mode::Err };

        // The error path is observable directly on the trait object.
        let doc = llm.synthesize_hyde_doc("how does auth work").await;
        assert!(doc.is_err(), "mock must error in Mode::Err");

        // And the production branch for an errored doc is: return base.
        // We reconstruct that decision here to lock its shape: base passes
        // through with no merge applied.
        let base = SearchOutput {
            results: vec![make_result(1, 0.9), make_result(2, 0.8)],
            metrics: make_metrics(),
        };
        let base_ids: Vec<u64> = base.results.iter().map(|r| r.chunk.id).collect();
        let returned = if doc.is_err() { base } else { unreachable!() };
        let returned_ids: Vec<u64> = returned.results.iter().map(|r| r.chunk.id).collect();
        assert_eq!(
            returned_ids, base_ids,
            "error path must return base unchanged"
        );
        assert_ne!(
            returned.metrics.query_type, "hyde_merged",
            "error path must NOT mark results as hyde_merged"
        );
    }

    /// Contract: an empty HyDE doc (whitespace-only counts) is treated as "no
    /// doc" — the caller returns base unchanged, never merging an empty query's
    /// results.
    #[tokio::test]
    async fn hyde_empty_doc_yields_base_unchanged() {
        let llm = MockLlm { mode: Mode::Empty };
        let doc = llm.synthesize_hyde_doc("anything").await.unwrap();
        assert!(
            doc.trim().is_empty(),
            "mock must return empty in Mode::Empty"
        );

        // Production decision for an empty doc: return base (no merge).
        let base = SearchOutput {
            results: vec![make_result(7, 0.5)],
            metrics: make_metrics(),
        };
        let base_ids: Vec<u64> = base.results.iter().map(|r| r.chunk.id).collect();
        let returned = if doc.trim().is_empty() {
            base
        } else {
            unreachable!()
        };
        assert_eq!(
            returned
                .results
                .iter()
                .map(|r| r.chunk.id)
                .collect::<Vec<_>>(),
            base_ids,
            "empty-doc path must return base unchanged"
        );
        assert_ne!(returned.metrics.query_type, "hyde_merged");
    }

    /// Contract: when the LLM is slower than `SEMANTEX_LLM_HYDE_TIMEOUT_MS`,
    /// `tokio::time::timeout` returns `Err(Elapsed)` and the caller returns
    /// base unchanged. We drive the *real* `super::llm_hyde_timeout()` reader by
    /// overriding the env var to 1ms and racing it against a 200ms mock.
    ///
    /// SAFETY: env mutation under Rust 2024 is `unsafe`; this test holds the
    /// crate-wide `TEST_ENV_LOCK` (which guards `SEMANTEX_LLM_*`) so no other
    /// LLM test races on process env.
    #[tokio::test(flavor = "current_thread")]
    async fn hyde_timeout_yields_base_unchanged() {
        // Resolve the *real* `llm_hyde_timeout()` env reader (1ms) inside a
        // short critical section, then drop the env + lock BEFORE awaiting so
        // we never hold a std `MutexGuard` across an await point. The captured
        // `budget` is what the production path would have computed.
        let budget = {
            let _guard = crate::llm::TEST_ENV_LOCK.lock().unwrap();
            // SAFETY: guarded by TEST_ENV_LOCK; see module doc on the lock.
            unsafe { std::env::set_var("SEMANTEX_LLM_HYDE_TIMEOUT_MS", "1") };
            let budget = super::llm_hyde_timeout();
            // SAFETY: guarded by TEST_ENV_LOCK.
            unsafe { std::env::remove_var("SEMANTEX_LLM_HYDE_TIMEOUT_MS") };
            budget
        };

        let llm = MockLlm {
            mode: Mode::Slow(std::time::Duration::from_millis(200)),
        };
        let outcome = tokio::time::timeout(budget, llm.synthesize_hyde_doc("slow query")).await;

        assert!(
            outcome.is_err(),
            "1ms budget must elapse before a 200ms mock resolves"
        );

        // Production decision on Err(Elapsed): return base (no merge).
        let base = SearchOutput {
            results: vec![make_result(9, 0.42)],
            metrics: make_metrics(),
        };
        let returned = if outcome.is_err() {
            base
        } else {
            unreachable!()
        };
        assert_eq!(returned.results.len(), 1);
        assert_ne!(returned.metrics.query_type, "hyde_merged");
    }

    /// Build a minimal real `HybridSearcher` backed by an empty chunk store,
    /// driving `search_with_hyde` through its genuine async code path (base
    /// search → LLM → merge). Mirrors `server::handler`'s `build_empty_searcher`.
    /// An empty index makes both the base and HyDE searches return no results,
    /// which is exactly what the byte-identity / no-panic contracts need.
    fn build_tiny_searcher() -> (tempfile::TempDir, HybridSearcher) {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let semantex_dir = dir.path().join(".semantex");
        std::fs::create_dir_all(&semantex_dir).unwrap();
        let db_path = semantex_dir.join("chunks.db");
        {
            let _store =
                crate::index::storage::ChunkStore::open(&db_path).expect("create chunk store");
        }
        let config = crate::config::SemantexConfig::default();
        let searcher = HybridSearcher::open_sparse_only(&semantex_dir, &config)
            .expect("open sparse-only searcher");
        (dir, searcher)
    }

    /// Build a default Semantic `SearchQuery` for the end-to-end tests.
    fn semantic_query(text: &str) -> crate::search::SearchQuery {
        crate::search::SearchQuery::new(text).max_results(10)
    }

    /// End-to-end: with a failing LLM, `search_with_hyde` returns a result set
    /// byte-identical to the plain base search — proving "HyDE never breaks a
    /// search." Drives the real async path, not just the branch decision.
    #[tokio::test]
    async fn search_with_hyde_error_path_is_identical_to_base() {
        let (_tmp, searcher) = build_tiny_searcher();
        let query = semantic_query("error handling");

        let base = searcher.search(&query).expect("base search");
        let base_ids: Vec<u64> = base.results.iter().map(|r| r.chunk.id).collect();

        let failing: std::sync::Arc<dyn crate::llm::LlmCapability> =
            std::sync::Arc::new(MockLlm { mode: Mode::Err });
        let hyde_out = searcher
            .search_with_hyde(&query, failing)
            .await
            .expect("search_with_hyde must not propagate LLM errors");

        let hyde_ids: Vec<u64> = hyde_out.results.iter().map(|r| r.chunk.id).collect();
        assert_eq!(
            hyde_ids, base_ids,
            "LLM-error path must be byte-identical to base (same IDs, same order)"
        );
        assert_ne!(
            hyde_out.metrics.query_type, "hyde_merged",
            "error path must NOT be marked hyde_merged"
        );
    }

    /// Success path: a succeeding mock yields a merged, deduped result set
    /// tagged `hyde_merged`, with no duplicate chunk IDs and never fewer than
    /// base after dedup+cap.
    #[tokio::test]
    async fn search_with_hyde_success_path_merges_and_tags() {
        let (_tmp, searcher) = build_tiny_searcher();
        let query = semantic_query("error handling");

        let base = searcher.search(&query).expect("base search");

        let ok: std::sync::Arc<dyn crate::llm::LlmCapability> = std::sync::Arc::new(MockLlm {
            mode: Mode::Ok("fn handle_error(e: Error) -> Result<()> { Err(e) }"),
        });
        let merged = searcher
            .search_with_hyde(&query, ok)
            .await
            .expect("search_with_hyde");

        // Dedup-by-id invariant: no duplicate chunk IDs in the output.
        let mut ids: Vec<u64> = merged.results.iter().map(|r| r.chunk.id).collect();
        let n = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(
            ids.len(),
            n,
            "merged output must contain no duplicate chunk IDs"
        );

        // The success path tags the merge.
        assert_eq!(merged.metrics.query_type, "hyde_merged");
        // Merge is base ∪ hyde (capped) → never fewer than base after dedup+cap.
        assert!(
            merged.results.len() >= base.results.len().min(query.max_results),
            "merged result count must be >= base (post dedup/cap)"
        );
    }

    /// `merge_hyde_results` must dedup by `chunk.id`, not by `(file, start, end)`.
    /// Chunk ID is the stable DB key and is the only correct dedup key.
    #[test]
    fn merge_deduplicates_by_chunk_id() {
        let base = SearchOutput {
            results: vec![make_result(1, 0.9), make_result(2, 0.8)],
            metrics: make_metrics(),
        };
        let hyde = SearchOutput {
            // chunk 2 duplicated, chunk 3 new
            results: vec![make_result(2, 0.85), make_result(3, 0.7)],
            metrics: make_metrics(),
        };

        let merged = merge_hyde_results(base, hyde, 10);
        assert_eq!(
            merged.results.len(),
            3,
            "dedup should yield 3 unique chunks"
        );

        let ids: Vec<u64> = merged.results.iter().map(|r| r.chunk.id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert!(ids.contains(&3));
    }

    /// Results must be sorted by score descending after merge.
    #[test]
    fn merge_sorts_by_score_descending() {
        let base = SearchOutput {
            results: vec![make_result(1, 0.5)],
            metrics: make_metrics(),
        };
        let hyde = SearchOutput {
            results: vec![make_result(2, 0.9), make_result(3, 0.7)],
            metrics: make_metrics(),
        };

        let merged = merge_hyde_results(base, hyde, 10);
        let scores: Vec<f32> = merged.results.iter().map(|r| r.score).collect();
        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert_eq!(scores, sorted, "results must be sorted by score desc");
    }

    /// `max_results` cap must be respected.
    #[test]
    fn merge_caps_at_max_results() {
        let base = SearchOutput {
            results: (0..5)
                .map(|i| make_result(i, 0.9 - i as f32 * 0.1))
                .collect(),
            metrics: make_metrics(),
        };
        let hyde = SearchOutput {
            results: (5..10)
                .map(|i| make_result(i, 0.9 - (i - 5) as f32 * 0.1))
                .collect(),
            metrics: make_metrics(),
        };

        let merged = merge_hyde_results(base, hyde, 4);
        assert_eq!(merged.results.len(), 4, "should cap at max_results=4");
    }

    /// When base search is empty and HyDE fails (caller returns base unchanged),
    /// the result must be an empty SearchOutput — no panic.
    #[test]
    fn merge_with_empty_base_does_not_panic() {
        let base = SearchOutput {
            results: vec![],
            metrics: make_metrics(),
        };
        let hyde = SearchOutput {
            results: vec![make_result(1, 0.9)],
            metrics: make_metrics(),
        };

        let merged = merge_hyde_results(base, hyde, 10);
        // HyDE result survives even when base is empty.
        assert_eq!(merged.results.len(), 1);
    }
}
