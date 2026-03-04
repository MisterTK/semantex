use crate::config::SemantexConfig;
use crate::embedding::colbert::ColbertEmbedder;
use crate::embedding::model_manager;
use crate::index::file_classifier::FileRole;
use crate::index::storage::ChunkStore;
use crate::search::SearchQuery;
use crate::search::adaptive;
use crate::search::fastembed_reranker::FastembedReranker;
use crate::search::graph_propagation::{self, GraphPropagationConfig};
use crate::search::plaid_search::PlaidSearcher;
use crate::search::query_classifier::{self, FusionWeights, QueryType};
use crate::search::sparse_search::SparseIndex;
use crate::search::triple_fusion;
use crate::search::{query_expander, regex_semantic};
use crate::types::{ScoredChunkId, SearchResult, SearchSource};
use anyhow::{Context, Result};
use fastembed::RerankerModel;
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Hybrid searcher combining dense (ColBERT/PLAID), sparse (BM25), and reranking.
/// Exact substring search is handled by SQLite LIKE queries on the chunk store.
pub struct HybridSearcher {
    sparse: Option<SparseIndex>,
    plaid: Option<PlaidSearcher>,
    colbert: Option<&'static ColbertEmbedder>,
    reranker: Mutex<Option<FastembedReranker>>,
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
            Some(SparseIndex::open(&sparse_path)?)
        } else {
            None
        };

        let reranker = Mutex::new(None);

        Ok(Self {
            sparse,
            plaid: None,
            colbert: None,
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
            Some(SparseIndex::open(&sparse_path)?)
        } else {
            None
        };

        // Load PLAID index for ColBERT late-interaction search
        let (plaid, colbert) = {
            let plaid_dir = index_dir.join("plaid");
            let mapping_path = index_dir.join("plaid_mapping.bin");
            if plaid_dir.exists() && mapping_path.exists() {
                match PlaidSearcher::open(&plaid_dir, &mapping_path) {
                    Ok(ps) => {
                        // Initialize ColBERT embedder for query encoding
                        let colbert_model_dir =
                            model_manager::ensure_colbert_model(&config.models_dir());
                        match colbert_model_dir.and_then(|d| ColbertEmbedder::global(&d)) {
                            Ok(emb) => {
                                tracing::info!("PLAID index loaded with ColBERT encoder");
                                (Some(ps), Some(emb))
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "PLAID index found but ColBERT encoder failed: {}",
                                    e
                                );
                                (None, None)
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Failed to open PLAID index: {}", e);
                        (None, None)
                    }
                }
            } else {
                (None, None)
            }
        };

        // Reranker is loaded lazily on first use to save ~200MB for cold searches
        let reranker = Mutex::new(None);

        Ok(Self {
            sparse,
            plaid,
            colbert,
            reranker,
            store,
            config: config.clone(),
        })
    }

    /// Reload the sparse index reader to pick up incremental updates.
    /// The PLAID index is memory-mapped and picks up changes automatically on re-open,
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

        // Stage 1: Candidate retrieval — run dense, sparse, and exact in parallel
        let (dense_results, sparse_results, exact_ids, dense_ms, sparse_ms, exact_ms) =
            std::thread::scope(|s| {
                let dense_handle = s.spawn(|| -> (Vec<ScoredChunkId>, u64) {
                    if !query.use_dense {
                        return (Vec::new(), 0);
                    }
                    let dense_start = std::time::Instant::now();
                    let results = if let (Some(plaid), Some(colbert)) = (&self.plaid, self.colbert)
                    {
                        match plaid.search(colbert, &effective_text, retrieval_candidates) {
                            Ok(results) => {
                                tracing::debug!(
                                    results_count = results.len(),
                                    duration_ms = dense_start.elapsed().as_millis() as u64,
                                    "PLAID (ColBERT) search complete"
                                );
                                results
                            }
                            Err(e) => {
                                tracing::warn!("PLAID search failed: {}", e);
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

                        #[allow(clippy::collapsible_if)]
                        if matches!(query_type, QueryType::Semantic | QueryType::Mixed) {
                            if let Some(expanded) = query_expander::expand_query(&effective_text) {
                                let safe_expanded = sanitize_tantivy_query(&expanded);
                                if let Ok(expanded_results) =
                                    sparse.search(&safe_expanded, retrieval_candidates)
                                {
                                    tracing::debug!(
                                        expanded_count = expanded_results.len(),
                                        "Dual-route BM25 expansion"
                                    );
                                    results =
                                        dual_route_fuse(&results, &expanded_results, 1.0, 0.5);
                                }
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

                let (dense_results, d_ms) = dense_handle.join().unwrap_or_else(|_| (Vec::new(), 0));
                let (sparse_results, s_ms) =
                    sparse_handle.join().unwrap_or_else(|_| (Vec::new(), 0));
                let (exact_ids, e_ms) = exact_handle.join().unwrap_or_else(|_| (Vec::new(), 0));

                (dense_results, sparse_results, exact_ids, d_ms, s_ms, e_ms)
            });

        // Stage 2: Query-adaptive weighted Triple CC Fusion
        let fusion_start = std::time::Instant::now();
        let triple_weights = query_type.triple_fusion_weights();
        tracing::debug!(
            ?query_type,
            ?triple_weights,
            "Query classified (triple fusion)"
        );

        let dense_count = dense_results.len();
        let sparse_count = sparse_results.len();
        let exact_count = exact_ids.len();

        let has_any_results =
            !dense_results.is_empty() || !sparse_results.is_empty() || !exact_ids.is_empty();

        let fused = if has_any_results {
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
        let scored_ids: Vec<ScoredChunkId> = fused
            .iter()
            .take(fetch_count)
            .cloned()
            .collect();
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
                    SearchResult {
                        chunk,
                        score: scored.score,
                        source: chunk_source,
                        score_dense: scored.score_dense,
                        score_sparse: scored.score_sparse,
                        score_exact: scored.score_exact,
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

            // Lazy-load reranker on first use
            if reranker_guard.is_none() {
                tracing::info!("Loading reranker model (first use)...");
                match FastembedReranker::new(RerankerModel::JINARerankerV1TurboEn, false) {
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
        let list_a = vec![
            ScoredChunkId::new(1, 0.9),
            ScoredChunkId::new(2, 0.8),
        ];
        let list_b = vec![
            ScoredChunkId::new(3, 0.9),
            ScoredChunkId::new(4, 0.8),
        ];

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
        let list_a = vec![
            ScoredChunkId::new(1, 0.9),
            ScoredChunkId::new(2, 0.1),
        ];
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

        let dense_list = vec![
            ScoredChunkId::new(1, 0.9),
            ScoredChunkId::new(2, 0.7),
        ];
        let sparse_list = vec![
            ScoredChunkId::new(2, 10.0),
            ScoredChunkId::new(3, 5.0),
        ];

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
            let sparse = vec![
                ScoredChunkId::new(10, 5.0),
                ScoredChunkId::new(20, 3.0),
            ];
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
}
