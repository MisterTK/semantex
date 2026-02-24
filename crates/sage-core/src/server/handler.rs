use crate::search::SearchQuery;
use crate::search::hybrid::HybridSearcher;
use crate::types::ChunkType;

use super::protocol::{
    ErrorResponse, HealthResponse, Request, Response, SearchRequest, SearchResponse,
    SearchResultItem, ShutdownResponse,
};
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
            Ok(results) => {
                self.search_count.fetch_add(1, Ordering::Relaxed);
                let duration_ms = start.elapsed().as_millis() as u64;

                let items: Vec<SearchResultItem> = results
                    .iter()
                    .map(|r| {
                        let (chunk_type_str, name, language) = match &r.chunk.chunk_type {
                            ChunkType::AstNode {
                                name,
                                kind: _,
                                language,
                                ..
                            } => (
                                "AstNode".to_string(),
                                Some(name.clone()),
                                Some(language.clone()),
                            ),
                            ChunkType::TextWindow { .. } => ("TextWindow".to_string(), None, None),
                            ChunkType::PdfPage { .. } => ("PdfPage".to_string(), None, None),
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
                        }
                    })
                    .collect();

                Response::Search(SearchResponse {
                    results: items,
                    duration_ms,
                    dense_count: 0,
                    sparse_count: 0,
                    fused_count: 0,
                })
            }
            Err(e) => Response::Error(ErrorResponse {
                message: format!("Search failed: {e}"),
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
