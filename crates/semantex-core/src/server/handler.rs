use crate::index::storage::ChunkStore;
use crate::search::SearchQuery;
use crate::search::hybrid::HybridSearcher;
use crate::types::{Chunk, ChunkType};

use super::protocol::{
    DeepResponseMetrics, DeepSearchRequest, DeepSearchResponse, DeepSearchSource, ErrorResponse,
    GraphWalkRequest, GraphWalkResponse, HealthResponse, MultiSearchRequest, MultiSearchResponse,
    Request, Response, SearchRequest, SearchResponse, SearchResultItem, ShutdownResponse,
};
use crate::search::deep as deep_search_module;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Handles incoming requests using a HybridSearcher
pub struct Handler<'a> {
    searcher: &'a HybridSearcher,
    search_count: &'a AtomicU64,
}

impl<'a> Handler<'a> {
    pub fn new(searcher: &'a HybridSearcher, search_count: &'a AtomicU64) -> Self {
        Self {
            searcher,
            search_count,
        }
    }

    pub fn handle(&self, request: Request, start_time: Instant) -> Response {
        match request {
            Request::Search(req) => self.handle_search(req),
            Request::Health => self.handle_health(start_time),
            Request::Shutdown => Response::Shutdown(ShutdownResponse {
                status: "shutting_down".to_string(),
            }),
            Request::GraphWalk(req) => self.handle_graph_walk(&req),
            Request::MultiSearch(req) => self.handle_multi_search(req),
            Request::DeepSearch(ref req) => self.handle_deep_search(req),
        }
    }

    fn handle_search(&self, req: SearchRequest) -> Response {
        let start = Instant::now();

        let mut query = SearchQuery::new(&req.query).max_results(req.max_results);
        if req.grep_mode {
            query = query.grep_mode();
        } else {
            if !req.use_dense {
                query.use_dense = false;
            }
            if !req.use_sparse {
                query.use_sparse = false;
            }
            if !req.use_rerank {
                query.use_rerank = false;
            }
        }
        if req.code_only {
            query = query.code_only();
        }
        if !req.include_types.is_empty() {
            query = query.include_types(req.include_types);
        }
        if !req.exclude_types.is_empty() {
            query = query.exclude_types(req.exclude_types);
        }
        if req.regex_pattern.is_some() {
            query = query.regex_pattern(req.regex_pattern);
        }

        match self.searcher.search(&query) {
            Ok(output) => {
                self.search_count.fetch_add(1, Ordering::Relaxed);
                let duration_ms = start.elapsed().as_millis() as u64;
                let metrics = output.metrics;

                let mut items: Vec<SearchResultItem> = output
                    .results
                    .iter()
                    .map(|r| {
                        let (chunk_type_str, name, language, kind, summary) =
                            match &r.chunk.chunk_type {
                                ChunkType::AstNode {
                                    name,
                                    kind,
                                    language,
                                    structured_meta,
                                    ..
                                } => {
                                    let kind_str = Some(kind.to_string());
                                    let summary_str = structured_meta
                                        .as_ref()
                                        .map(|meta| meta.display_summary())
                                        .filter(|s| !s.is_empty());
                                    (
                                        "AstNode".to_string(),
                                        Some(name.clone()),
                                        Some(language.clone()),
                                        kind_str,
                                        summary_str,
                                    )
                                }
                                ChunkType::TextWindow { .. } => (
                                    "TextWindow".to_string(),
                                    None,
                                    None,
                                    Some("text".to_string()),
                                    None,
                                ),
                                ChunkType::PdfPage { .. } => (
                                    "PdfPage".to_string(),
                                    None,
                                    None,
                                    Some("pdf".to_string()),
                                    None,
                                ),
                            };

                        let content = if req.include_content {
                            if req.snippet {
                                Some(make_snippet(&r.chunk.content, &r.chunk.chunk_type))
                            } else {
                                Some(r.chunk.content.clone())
                            }
                        } else {
                            None
                        };

                        SearchResultItem {
                            file: r.chunk.file_path.display().to_string(),
                            start_line: r.chunk.start_line,
                            end_line: r.chunk.end_line,
                            score: r.score,
                            source: format!("{:?}", r.source),
                            chunk_type: chunk_type_str,
                            name,
                            language,
                            content,
                            kind,
                            summary,
                        }
                    })
                    .collect();

                // Auto-peek: if requested, populate content for the top result only
                if req.auto_peek_top
                    && !req.include_content
                    && !items.is_empty()
                    && items[0].content.is_none()
                {
                    // Re-fetch content from the search results (still in scope as `output.results`)
                    if let Some(top_result) = output.results.first() {
                        let snippet = top_result
                            .chunk
                            .content
                            .lines()
                            .take(5)
                            .collect::<Vec<_>>()
                            .join("\n");
                        let snippet = if top_result.chunk.content.lines().count() > 5 {
                            format!("{snippet}\n...")
                        } else {
                            snippet
                        };
                        items[0].content = Some(snippet);
                    }
                }

                let confidence = if items.is_empty() {
                    Some("none".to_string())
                } else {
                    let top_score = items[0].score;
                    let second_score = items.get(1).map_or(0.0, |i| i.score);
                    let gap = top_score - second_score;

                    if top_score > 0.5 && gap > 0.15 {
                        Some("high".to_string())
                    } else if top_score > 0.25 {
                        Some("medium".to_string())
                    } else {
                        Some("low".to_string())
                    }
                };

                Response::Search(SearchResponse {
                    results: items,
                    duration_ms,
                    dense_count: metrics.dense_count,
                    sparse_count: metrics.sparse_count,
                    fused_count: metrics.fused_count,
                    metrics: Some(metrics),
                    confidence,
                })
            }
            Err(e) => Response::Error(ErrorResponse {
                message: format!("Search failed: {e}"),
            }),
        }
    }

    fn handle_multi_search(&self, req: MultiSearchRequest) -> Response {
        let responses: Vec<SearchResponse> = req
            .queries
            .into_iter()
            .map(|q| match self.handle_search(q) {
                Response::Search(r) => r,
                _ => unreachable!(),
            })
            .collect();
        Response::MultiSearch(MultiSearchResponse { responses })
    }

    fn handle_deep_search(&self, req: &DeepSearchRequest) -> Response {
        match deep_search_module::deep_search(
            self.searcher,
            &req.query,
            req.max_results,
            req.use_graph,
        ) {
            Ok(result) => {
                let sources = result
                    .sources
                    .into_iter()
                    .map(|s| DeepSearchSource {
                        file: s.file,
                        start_line: s.start_line,
                        end_line: s.end_line,
                        name: s.name,
                        kind: s.kind,
                    })
                    .collect();
                Response::DeepSearch(DeepSearchResponse {
                    answer: result.answer,
                    sources,
                    metrics: DeepResponseMetrics {
                        search_ms: result.metrics.search_ms,
                        triage_ms: result.metrics.triage_ms,
                        graph_ms: result.metrics.graph_ms,
                        read_ms: result.metrics.read_ms,
                        summarize_ms: result.metrics.summarize_ms,
                        total_ms: result.metrics.total_ms,
                        chunks_searched: result.metrics.chunks_searched,
                        chunks_read: result.metrics.chunks_read,
                    },
                })
            }
            Err(e) => Response::Error(ErrorResponse {
                message: format!("Deep search failed: {e}"),
            }),
        }
    }

    fn handle_health(&self, start_time: Instant) -> Response {
        Response::Health(HealthResponse {
            status: "ok".to_string(),
            uptime_s: start_time.elapsed().as_secs(),
            searches: self.search_count.load(Ordering::Relaxed),
        })
    }

    fn handle_graph_walk(&self, req: &GraphWalkRequest) -> Response {
        match self
            .searcher
            .with_store(|store| graph_walk_from_store(store, &req.symbol))
        {
            Ok(resp) => Response::GraphWalk(resp),
            Err(e) => Response::Error(ErrorResponse {
                message: format!("Graph walk failed: {e}"),
            }),
        }
    }
}

/// Perform a graph walk from a direct ChunkStore reference.
/// Used by both the daemon handler and the direct (no-daemon) fallback path.
pub(crate) fn graph_walk_from_store(
    store: &ChunkStore,
    symbol: &str,
) -> anyhow::Result<GraphWalkResponse> {
    // 3. Query all graph directions (limit each to 20 results)
    const MAX_PER_SECTION: usize = 20;

    // 1. Resolve symbol to chunk IDs
    let matches = store.lookup_symbol_exact(symbol)?;
    if matches.is_empty() {
        // Symbol not found — return empty response
        return Ok(GraphWalkResponse {
            target: vec![],
            callers: vec![],
            callees: vec![],
            type_refs: vec![],
            hierarchy: vec![],
        });
    }

    let target_ids: Vec<u64> = matches.iter().map(|(id, _, _)| *id as u64).collect();

    // 2. Resolve target chunks
    let target_chunks = store.get_chunks(&target_ids)?;
    let target_items: Vec<SearchResultItem> = target_chunks.iter().map(chunk_to_item).collect();

    let callee_edges = store.get_call_edges_from(&target_ids)?;
    let callee_ids: Vec<u64> = callee_edges
        .iter()
        .map(|(_, callee)| *callee)
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .take(MAX_PER_SECTION)
        .collect();

    let incoming_edges = store.get_call_edges_to(&target_ids)?;
    let incoming_ids: Vec<u64> = incoming_edges
        .iter()
        .map(|(_, src)| *src)
        .filter(|id| !target_ids.contains(id))
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .take(MAX_PER_SECTION)
        .collect();

    let type_usage_edges = store.get_type_ref_edges_from_usages(&target_ids)?;
    let type_def_edges = store.get_type_ref_edges_to_defs(&target_ids)?;
    let type_ref_ids: Vec<u64> = type_usage_edges
        .iter()
        .map(|(_, def)| *def)
        .chain(type_def_edges.iter().map(|(_, usage)| *usage))
        .filter(|id| !target_ids.contains(id))
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .take(MAX_PER_SECTION)
        .collect();

    let hierarchy_edges = store.get_hierarchy_edges_for(&target_ids)?;
    let hierarchy_ids: Vec<u64> = hierarchy_edges
        .iter()
        .map(|(_, related)| *related)
        .filter(|id| !target_ids.contains(id))
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .take(MAX_PER_SECTION)
        .collect();

    // 4. Resolve chunk IDs to items
    let callee_chunks = store.get_chunks(&callee_ids)?;
    let incoming_chunks = store.get_chunks(&incoming_ids)?;
    let type_ref_chunks = store.get_chunks(&type_ref_ids)?;
    let hierarchy_chunks = store.get_chunks(&hierarchy_ids)?;

    Ok(GraphWalkResponse {
        target: target_items,
        callers: incoming_chunks.iter().map(chunk_to_item).collect(),
        callees: callee_chunks.iter().map(chunk_to_item).collect(),
        type_refs: type_ref_chunks.iter().map(chunk_to_item).collect(),
        hierarchy: hierarchy_chunks.iter().map(chunk_to_item).collect(),
    })
}

/// Convert a Chunk to a SearchResultItem (with kind/summary populated)
pub(crate) fn chunk_to_item(chunk: &Chunk) -> SearchResultItem {
    let (chunk_type_str, name, language, kind, summary) = match &chunk.chunk_type {
        crate::types::ChunkType::AstNode {
            name,
            kind,
            language,
            structured_meta,
            ..
        } => {
            let kind_str = Some(kind.to_string());
            let summary_str = structured_meta
                .as_ref()
                .map(|meta| meta.display_summary())
                .filter(|s| !s.is_empty());
            (
                "AstNode".to_string(),
                Some(name.clone()),
                Some(language.clone()),
                kind_str,
                summary_str,
            )
        }
        crate::types::ChunkType::TextWindow { .. } => (
            "TextWindow".to_string(),
            None,
            None,
            Some("text".to_string()),
            None,
        ),
        crate::types::ChunkType::PdfPage { .. } => (
            "PdfPage".to_string(),
            None,
            None,
            Some("pdf".to_string()),
            None,
        ),
    };
    SearchResultItem {
        file: chunk.file_path.display().to_string(),
        start_line: chunk.start_line,
        end_line: chunk.end_line,
        score: 0.0,
        source: "GraphWalk".to_string(),
        chunk_type: chunk_type_str,
        name,
        language,
        content: None,
        kind,
        summary,
    }
}

/// Create a short snippet from chunk content based on chunk type.
fn make_snippet(content: &str, chunk_type: &ChunkType) -> String {
    match chunk_type {
        ChunkType::AstNode { .. } => {
            let snippet: String = content.lines().take(5).collect::<Vec<_>>().join("\n");
            if content.lines().count() > 5 {
                format!("{snippet}\n...")
            } else {
                snippet
            }
        }
        ChunkType::TextWindow { .. } => {
            let snippet: String = content.lines().take(3).collect::<Vec<_>>().join("\n");
            if content.lines().count() > 3 {
                format!("{snippet}\n...")
            } else {
                snippet
            }
        }
        ChunkType::PdfPage { .. } => {
            if content.len() > 100 {
                format!("{}...", &content[..content.floor_char_boundary(100)])
            } else {
                content.to_string()
            }
        }
    }
}
