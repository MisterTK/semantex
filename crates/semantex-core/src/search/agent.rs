use std::path::{Path, PathBuf};
use crate::search::SearchQuery;
use crate::search::hybrid::HybridSearcher;
use crate::server::protocol::{
    AgentRequest, AgentResponse, AgentMetrics, DeepSearchResponse, DeepSearchSource,
    DeepResponseMetrics, GraphWalkResponse, SearchResponse, SearchResultItem,
};
use super::agent_classifier::{AgentRoute, classify_agent_query, extract_symbol};
use super::agent_formatter::{
    DEFAULT_BUDGET, FormatStyle, format_search_results, format_deep_results,
    format_graph_results, format_code_blocks,
};
use crate::server::handler::{search_result_to_item, graph_walk_from_store};
use crate::search::deep as deep_search_module;

/// Result from an individual handler method.
pub(crate) struct HandlerResult {
    formatted: String,
    fallback_used: bool,
    result_count: usize,
}

/// Orchestrates agent queries: classify → dispatch → fallback → format.
pub struct AgentPipeline<'a> {
    searcher: &'a HybridSearcher,
    project_root: PathBuf,
}

impl<'a> AgentPipeline<'a> {
    pub fn new(searcher: &'a HybridSearcher, project_root: PathBuf) -> Self {
        Self { searcher, project_root }
    }

    pub fn handle(&self, request: &AgentRequest) -> AgentResponse {
        let start = std::time::Instant::now();
        let classify_start = std::time::Instant::now();

        let route = request.route.unwrap_or_else(|| classify_agent_query(&request.query));
        let classify_us = classify_start.elapsed().as_micros() as u64;

        let budget = request.budget.unwrap_or(DEFAULT_BUDGET);

        let search_start = std::time::Instant::now();
        let result = match route {
            AgentRoute::FilePattern => self.handle_file_pattern(&request.query),
            AgentRoute::Regex       => self.handle_regex(&request.query, budget),
            AgentRoute::ExactSymbol => self.handle_exact_symbol(&request.query, budget),
            AgentRoute::Structural  => self.handle_structural(&request.query, budget),
            AgentRoute::Deep        => self.handle_deep(&request.query, budget, false),
            AgentRoute::Analytical  => self.handle_analytical(&request.query, budget),
            AgentRoute::Semantic    => self.handle_semantic(&request.query, budget, false),
        };
        let search_ms = search_start.elapsed().as_millis() as u64;

        let format_start = std::time::Instant::now();
        let formatted = format!("[route: {route}]\n\n{}", result.formatted);
        let format_ms = format_start.elapsed().as_millis() as u64;

        AgentResponse {
            route,
            formatted,
            metrics: AgentMetrics {
                classify_us,
                search_ms,
                format_ms,
                total_ms: start.elapsed().as_millis() as u64,
                fallback_used: result.fallback_used,
                result_count: result.result_count,
            },
        }
    }

    fn handle_semantic(&self, query: &str, budget: usize, is_fallback: bool) -> HandlerResult {
        let sq = SearchQuery::new(query).max_results(10);
        match self.searcher.search(&sq) {
            Ok(output) if !output.results.is_empty() => {
                let items: Vec<SearchResultItem> = output.results.iter()
                    .map(|r| search_result_to_item(r, true))
                    .collect();
                let confidence = compute_confidence(&items);
                let resp = SearchResponse {
                    results: items.clone(),
                    duration_ms: output.metrics.total_ms,
                    dense_count: output.metrics.dense_count,
                    sparse_count: output.metrics.sparse_count,
                    fused_count: output.metrics.fused_count,
                    metrics: None,
                    confidence: Some(confidence),
                };
                HandlerResult {
                    formatted: format_search_results(&resp, FormatStyle::Default, budget),
                    fallback_used: is_fallback,
                    result_count: items.len(),
                }
            }
            _ => {
                if !is_fallback {
                    // Fallback to deep search
                    match deep_search_module::deep_search(self.searcher, query, 20, true) {
                        Ok(result) if !result.answer.is_empty() => {
                            let resp = deep_result_to_response(result);
                            let count = resp.sources.len();
                            HandlerResult {
                                formatted: format_deep_results(&resp, budget),
                                fallback_used: true,
                                result_count: count,
                            }
                        }
                        _ => HandlerResult {
                            formatted: format!("No results found for: {query}"),
                            fallback_used: is_fallback,
                            result_count: 0,
                        },
                    }
                } else {
                    HandlerResult {
                        formatted: format!("No results found for: {query}"),
                        fallback_used: is_fallback,
                        result_count: 0,
                    }
                }
            }
        }
    }

    fn handle_deep(&self, query: &str, budget: usize, is_fallback: bool) -> HandlerResult {
        match deep_search_module::deep_search(self.searcher, query, 20, true) {
            Ok(result) if !result.answer.is_empty() => {
                let resp = deep_result_to_response(result);
                let count = resp.sources.len();
                HandlerResult {
                    formatted: format_deep_results(&resp, budget),
                    fallback_used: is_fallback,
                    result_count: count,
                }
            }
            _ => {
                if !is_fallback {
                    self.handle_semantic(query, budget, true)
                } else {
                    HandlerResult {
                        formatted: format!("No results found for: {query}"),
                        fallback_used: true,
                        result_count: 0,
                    }
                }
            }
        }
    }

    fn handle_exact_symbol(&self, query: &str, budget: usize) -> HandlerResult {
        let symbol = query.trim_matches(|c| c == '`' || c == '"' || c == '\'');
        let sq = SearchQuery::new(symbol).max_results(5);
        if let Ok(output) = self.searcher.search(&sq) {
            if !output.results.is_empty() {
                let items: Vec<SearchResultItem> = output.results.iter()
                    .map(|r| search_result_to_item(r, true))
                    .collect();
                let confidence = compute_confidence(&items);
                let resp = SearchResponse {
                    results: items.clone(),
                    duration_ms: output.metrics.total_ms,
                    dense_count: output.metrics.dense_count,
                    sparse_count: output.metrics.sparse_count,
                    fused_count: output.metrics.fused_count,
                    metrics: None,
                    confidence: Some(confidence),
                };
                return HandlerResult {
                    formatted: format_search_results(&resp, FormatStyle::Default, budget),
                    fallback_used: false,
                    result_count: items.len(),
                };
            }
        }
        // Fallback: grep mode
        let sq = SearchQuery::new(symbol).grep_mode().max_results(10);
        match self.searcher.search(&sq) {
            Ok(output) if !output.results.is_empty() => {
                let items: Vec<SearchResultItem> = output.results.iter()
                    .map(|r| search_result_to_item(r, true))
                    .collect();
                let resp = SearchResponse {
                    results: items.clone(),
                    duration_ms: output.metrics.total_ms,
                    dense_count: 0,
                    sparse_count: output.metrics.sparse_count,
                    fused_count: output.metrics.fused_count,
                    metrics: None,
                    confidence: Some("medium".into()),
                };
                HandlerResult {
                    formatted: format_search_results(&resp, FormatStyle::Default, budget),
                    fallback_used: true,
                    result_count: items.len(),
                }
            }
            _ => HandlerResult {
                formatted: format!("Symbol not found: {symbol}"),
                fallback_used: true,
                result_count: 0,
            },
        }
    }

    fn handle_structural(&self, query: &str, budget: usize) -> HandlerResult {
        if let Some(symbol) = extract_symbol(query) {
            let resp: GraphWalkResponse = self.searcher.with_store(|store| {
                graph_walk_from_store(store, &symbol).unwrap_or_else(|_| GraphWalkResponse {
                    target: vec![],
                    callers: vec![],
                    callees: vec![],
                    type_refs: vec![],
                    hierarchy: vec![],
                })
            });
            let total = resp.target.len() + resp.callers.len() + resp.callees.len()
                + resp.type_refs.len() + resp.hierarchy.len();
            if total > 0 {
                return HandlerResult {
                    formatted: format_graph_results(&resp),
                    fallback_used: false,
                    result_count: total,
                };
            }
        }
        // Fallback to deep
        self.handle_deep(query, budget, true)
    }

    fn handle_analytical(&self, query: &str, budget: usize) -> HandlerResult {
        let sq = SearchQuery::new(query).max_results(5).code_only();
        match self.searcher.search(&sq) {
            Ok(output) if !output.results.is_empty() => {
                let items: Vec<SearchResultItem> = output.results.iter()
                    .map(|r| search_result_to_item(r, true))
                    .collect();
                let code_contents: Vec<String> = items.iter()
                    .map(|i| i.content.clone().unwrap_or_default())
                    .collect();
                let formatted = format_code_blocks(&items, &code_contents, budget);
                if formatted == "No code blocks to display." {
                    return self.handle_deep(query, budget, true);
                }
                HandlerResult {
                    formatted,
                    fallback_used: false,
                    result_count: items.len(),
                }
            }
            _ => self.handle_deep(query, budget, true),
        }
    }

    fn handle_regex(&self, query: &str, budget: usize) -> HandlerResult {
        let sq = SearchQuery::new(query).grep_mode().max_results(20);
        let (items, duration_ms, sparse_count, fused_count) = match self.searcher.search(&sq) {
            Ok(output) => {
                let items: Vec<SearchResultItem> = output.results.iter()
                    .map(|r| search_result_to_item(r, true))
                    .collect();
                (items, output.metrics.total_ms, output.metrics.sparse_count, output.metrics.fused_count)
            }
            Err(_) => (vec![], 0u64, 0usize, 0usize),
        };
        let resp = SearchResponse {
            results: items.clone(),
            duration_ms,
            dense_count: 0,
            sparse_count,
            fused_count,
            metrics: None,
            confidence: None,
        };
        HandlerResult {
            formatted: format_search_results(&resp, FormatStyle::Grep, budget),
            fallback_used: false,
            result_count: items.len(),
        }
    }

    fn handle_file_pattern(&self, query: &str) -> HandlerResult {
        glob_files(&self.project_root, query)
    }
}

/// Perform a filesystem glob walk and return matching file paths.
/// Extracted as a standalone fn to enable unit testing without constructing AgentPipeline.
pub(crate) fn glob_files(root: &Path, pattern: &str) -> HandlerResult {
    use ignore::WalkBuilder;

    // Try to compile as a globset pattern
    let matcher = globset::GlobBuilder::new(pattern)
        .case_insensitive(false)
        .build()
        .and_then(|g| {
            let mut b = globset::GlobSetBuilder::new();
            b.add(g);
            b.build()
        });

    let Ok(glob_set) = matcher else {
        return HandlerResult {
            formatted: format!("Invalid glob pattern: {pattern}"),
            fallback_used: false,
            result_count: 0,
        };
    };

    let mut files: Vec<String> = Vec::new();
    let walker = WalkBuilder::new(root)
        .hidden(false)
        .git_ignore(true)
        .git_global(false)
        .build();

    for entry in walker.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        // Skip .git, .semantex, node_modules, target
        let should_skip = path.components().any(|c| {
            matches!(c.as_os_str().to_str(), Some(".git" | ".semantex" | "node_modules" | "target"))
        });
        if should_skip {
            continue;
        }
        // Get relative path
        let rel = match path.strip_prefix(root) {
            Ok(r) => r,
            Err(_) => continue,
        };
        if glob_set.is_match(rel) {
            files.push(rel.display().to_string());
        }
    }

    files.sort();
    let total = files.len();
    let mut lines: Vec<String> = files.into_iter().take(50).collect();
    if total > 50 {
        lines.push(format!("... and {} more files", total - 50));
    }

    HandlerResult {
        formatted: lines.join("\n"),
        fallback_used: false,
        result_count: total,
    }
}

/// Helper: compute confidence string from search result items.
fn compute_confidence(items: &[SearchResultItem]) -> String {
    if items.is_empty() {
        return "none".into();
    }
    let top = items[0].score;
    let second = items.get(1).map_or(0.0, |i| i.score);
    let gap = top - second;
    if top > 0.5 && gap > 0.15 {
        "high".into()
    } else if top > 0.25 {
        "medium".into()
    } else {
        "low".into()
    }
}

/// Helper: convert a deep_search result to DeepSearchResponse protocol type.
fn deep_result_to_response(result: crate::search::deep::DeepResult) -> DeepSearchResponse {
    let sources = result.sources.into_iter().map(|s| DeepSearchSource {
        file: s.file,
        start_line: s.start_line,
        end_line: s.end_line,
        name: s.name,
        kind: s.kind,
    }).collect();
    DeepSearchResponse {
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
            confidence_zone: result.metrics.confidence_zone,
        },
        confidence: result.confidence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatch_routes_file_pattern() {
        use super::super::agent_classifier::{classify_agent_query, AgentRoute};
        assert_eq!(classify_agent_query("*.rs"), AgentRoute::FilePattern);
    }

    #[test]
    fn test_handle_file_pattern_finds_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("foo.rs"), "").unwrap();
        std::fs::write(dir.path().join("bar.rs"), "").unwrap();
        std::fs::write(dir.path().join("baz.py"), "").unwrap();

        let result = glob_files(dir.path(), "*.rs");
        assert!(result.formatted.contains("foo.rs"));
        assert!(result.formatted.contains("bar.rs"));
        assert!(!result.formatted.contains("baz.py"));
        assert_eq!(result.result_count, 2);
    }

    #[test]
    fn test_handle_file_pattern_excludes_git() {
        let dir = tempfile::tempdir().unwrap();
        let git = dir.path().join(".git");
        std::fs::create_dir(&git).unwrap();
        std::fs::write(git.join("config"), "").unwrap();
        std::fs::write(dir.path().join("main.rs"), "").unwrap();

        let result = glob_files(dir.path(), "*");
        assert!(result.formatted.contains("main.rs"));
        assert!(!result.formatted.contains(".git"));
    }

    #[test]
    fn test_handle_file_pattern_caps_at_50() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..60 {
            std::fs::write(dir.path().join(format!("file{i:03}.rs")), "").unwrap();
        }
        let result = glob_files(dir.path(), "*.rs");
        assert!(result.formatted.contains("... and 10 more files"));
        assert_eq!(result.result_count, 60);
    }
}
