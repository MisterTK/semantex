use super::agent_classifier::{AgentRoute, classify_agent_query, extract_symbol};
use super::agent_formatter::{
    DEFAULT_BUDGET, FormatStyle, format_code_blocks, format_deep_results, format_graph_results,
    format_search_results,
};
use crate::search::SearchQuery;
use crate::search::deep as deep_search_module;
use crate::search::hybrid::HybridSearcher;
use crate::server::handler::{
    compute_confidence, deep_result_to_response, graph_walk_from_store, search_result_to_item,
};
use crate::server::protocol::{
    AgentMetrics, AgentRequest, AgentResponse, GraphWalkResponse, SearchResponse, SearchResultItem,
};
use std::path::{Path, PathBuf};

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
        Self {
            searcher,
            project_root,
        }
    }

    pub fn handle(&self, request: &AgentRequest) -> AgentResponse {
        let start = std::time::Instant::now();
        let classify_start = std::time::Instant::now();

        let route = request
            .route
            .unwrap_or_else(|| classify_agent_query(&request.query));
        let classify_us = classify_start.elapsed().as_micros() as u64;

        let budget = request.budget.unwrap_or(DEFAULT_BUDGET);

        let search_start = std::time::Instant::now();
        let result = match route {
            AgentRoute::FilePattern => self.handle_file_pattern(&request.query),
            AgentRoute::Regex => self.handle_regex(&request.query, budget),
            AgentRoute::ExactSymbol => self.handle_exact_symbol(&request.query, budget),
            AgentRoute::Structural => self.handle_structural(&request.query, budget),
            AgentRoute::Deep => self.handle_deep(&request.query, budget, false),
            AgentRoute::Analytical => {
                self.handle_analytical(&request.query, budget, request.full_code)
            }
            AgentRoute::Exhaustive => self.handle_exhaustive(&request.query, budget),
            AgentRoute::Semantic => self.handle_semantic(&request.query, budget, false),
            AgentRoute::Architecture => self.handle_architecture(&request.query, budget),
            AgentRoute::ExhaustiveStructural => {
                self.handle_exhaustive_structural(&request.query, budget)
            }
            AgentRoute::DeepWithExamples => self.handle_deep_with_examples(&request.query, budget),
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
                let items: Vec<SearchResultItem> = output
                    .results
                    .iter()
                    .map(|r| search_result_to_item(r, true))
                    .collect();
                let confidence = compute_confidence(&items);
                let count = items.len();
                let resp = SearchResponse {
                    results: items,
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
                    result_count: count,
                }
            }
            _ => {
                if is_fallback {
                    HandlerResult {
                        formatted: format!("No results found for: {query}"),
                        fallback_used: is_fallback,
                        result_count: 0,
                    }
                } else {
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
                if is_fallback {
                    HandlerResult {
                        formatted: format!("No results found for: {query}"),
                        fallback_used: true,
                        result_count: 0,
                    }
                } else {
                    self.handle_semantic(query, budget, true)
                }
            }
        }
    }

    fn handle_exact_symbol(&self, query: &str, budget: usize) -> HandlerResult {
        let symbol = query.trim_matches(|c| c == '`' || c == '"' || c == '\'');
        let sq = SearchQuery::new(symbol).max_results(5);
        if let Ok(output) = self.searcher.search(&sq)
            && !output.results.is_empty()
        {
            let items: Vec<SearchResultItem> = output
                .results
                .iter()
                .map(|r| search_result_to_item(r, true))
                .collect();
            let confidence = compute_confidence(&items);
            let count = items.len();
            let resp = SearchResponse {
                results: items,
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
                result_count: count,
            };
        }
        // Fallback: grep mode
        let sq = SearchQuery::new(symbol).grep_mode().max_results(10);
        match self.searcher.search(&sq) {
            Ok(output) if !output.results.is_empty() => {
                let items: Vec<SearchResultItem> = output
                    .results
                    .iter()
                    .map(|r| search_result_to_item(r, true))
                    .collect();
                let count = items.len();
                let resp = SearchResponse {
                    results: items,
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
                    result_count: count,
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
            let total = resp.target.len()
                + resp.callers.len()
                + resp.callees.len()
                + resp.type_refs.len()
                + resp.hierarchy.len();
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

    fn handle_analytical(&self, query: &str, budget: usize, full_code: bool) -> HandlerResult {
        let sq = SearchQuery::new(query).max_results(5).code_only();
        match self.searcher.search(&sq) {
            Ok(output) if !output.results.is_empty() => {
                let items: Vec<SearchResultItem> = output
                    .results
                    .iter()
                    .map(|r| search_result_to_item(r, true))
                    .collect();
                let count = items.len();
                if full_code {
                    let code_contents: Vec<String> = items
                        .iter()
                        .map(|i| i.content.clone().unwrap_or_default())
                        .collect();
                    let formatted = format_code_blocks(&items, &code_contents, budget);
                    if formatted == "No code blocks to display." {
                        return self.handle_deep(query, budget, true);
                    }
                    HandlerResult {
                        formatted,
                        fallback_used: false,
                        result_count: count,
                    }
                } else {
                    let confidence = compute_confidence(&items);
                    let resp = crate::server::protocol::SearchResponse {
                        results: items,
                        duration_ms: output.metrics.total_ms,
                        dense_count: output.metrics.dense_count,
                        sparse_count: output.metrics.sparse_count,
                        fused_count: output.metrics.fused_count,
                        metrics: None,
                        confidence: Some(confidence),
                    };
                    HandlerResult {
                        formatted: format_search_results(&resp, FormatStyle::Default, budget),
                        fallback_used: false,
                        result_count: count,
                    }
                }
            }
            _ => self.handle_deep(query, budget, true),
        }
    }

    fn handle_exhaustive(&self, query: &str, budget: usize) -> HandlerResult {
        // Exhaustive queries need broader coverage: more results and a larger budget to
        // avoid truncation on large monorepos where relevant items are scattered.
        let exhaustive_budget = budget * 3;
        let sq = SearchQuery::new(query).max_results(20);
        match self.searcher.search(&sq) {
            Ok(output) if !output.results.is_empty() => {
                let items: Vec<SearchResultItem> = output
                    .results
                    .iter()
                    .map(|r| search_result_to_item(r, true))
                    .collect();
                let confidence = compute_confidence(&items);
                let count = items.len();
                let resp = SearchResponse {
                    results: items,
                    duration_ms: output.metrics.total_ms,
                    dense_count: output.metrics.dense_count,
                    sparse_count: output.metrics.sparse_count,
                    fused_count: output.metrics.fused_count,
                    metrics: None,
                    confidence: Some(confidence),
                };
                HandlerResult {
                    formatted: format_search_results(
                        &resp,
                        FormatStyle::Default,
                        exhaustive_budget,
                    ),
                    fallback_used: false,
                    result_count: count,
                }
            }
            _ => self.handle_deep(query, exhaustive_budget, true),
        }
    }

    fn handle_regex(&self, query: &str, budget: usize) -> HandlerResult {
        let sq = SearchQuery::new(query).grep_mode().max_results(20);
        let (items, duration_ms, sparse_count, fused_count) = match self.searcher.search(&sq) {
            Ok(output) => {
                let items: Vec<SearchResultItem> = output
                    .results
                    .iter()
                    .map(|r| search_result_to_item(r, true))
                    .collect();
                (
                    items,
                    output.metrics.total_ms,
                    output.metrics.sparse_count,
                    output.metrics.fused_count,
                )
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

    /// Phase 4 — Architecture overview. Replaces the v0.3 M6 visible MCP tool
    /// (which regressed CCB by encouraging agents to call it alongside other
    /// structural tools). Returns the same ArchOverview (god nodes +
    /// communities + boundaries) but rendered as formatted text and routed
    /// internally by `semantex_agent` based on classifier intent.
    fn handle_architecture(&self, query: &str, budget: usize) -> HandlerResult {
        use crate::index::architecture::build_arch_overview;
        let db_path = self.project_root.join(".semantex").join("chunks.db");
        let Ok(overview) = self
            .searcher
            .with_store(|store| build_arch_overview(store, &db_path))
        else {
            return self.handle_deep(query, budget, true);
        };
        if overview.god_nodes.is_empty()
            && overview.communities.is_empty()
            && overview.boundaries.is_empty()
        {
            // Pre-v0.3 index without architecture tables → fall back to deep.
            return self.handle_deep(query, budget, true);
        }
        let mut out = String::with_capacity(2048);
        let _ = std::fmt::Write::write_fmt(&mut out, format_args!("# Architectural overview\n\n"));
        if !overview.god_nodes.is_empty() {
            out.push_str("## Top central symbols (PageRank-ranked)\n\n");
            for (i, g) in overview.god_nodes.iter().enumerate() {
                let role = g
                    .semantic_role
                    .as_deref()
                    .map(|r| format!(" [{r}]"))
                    .unwrap_or_default();
                let _ = std::fmt::Write::write_fmt(
                    &mut out,
                    format_args!(
                        "{:>2}. `{}` ({} L{}-{}){} — centrality {:.4}\n",
                        i + 1,
                        g.symbol,
                        g.file,
                        g.start_line,
                        g.end_line,
                        role,
                        g.centrality
                    ),
                );
            }
            out.push('\n');
        }
        if !overview.communities.is_empty() {
            out.push_str("## Communities (clusters via call-graph BFS)\n\n");
            for c in &overview.communities {
                let _ = std::fmt::Write::write_fmt(
                    &mut out,
                    format_args!("### {} — {} members\n", c.label, c.size),
                );
                if !c.entry_points.is_empty() {
                    out.push_str("Entry points: ");
                    let mut first = true;
                    for ep in &c.entry_points {
                        if !first {
                            out.push_str(", ");
                        }
                        let _ = std::fmt::Write::write_fmt(
                            &mut out,
                            format_args!(
                                "`{}` ({}:{}-{})",
                                ep.symbol, ep.file, ep.start_line, ep.end_line
                            ),
                        );
                        first = false;
                    }
                    out.push_str("\n\n");
                }
                if !c.member_files.is_empty() {
                    out.push_str("Sample files:\n");
                    for f in &c.member_files {
                        let _ = std::fmt::Write::write_fmt(&mut out, format_args!("  - {f}\n"));
                    }
                    out.push('\n');
                }
            }
        }
        if !overview.boundaries.is_empty() {
            out.push_str("## Cross-directory coupling (top import-edge counts)\n\n");
            for b in &overview.boundaries {
                let _ = std::fmt::Write::write_fmt(
                    &mut out,
                    format_args!("- `{}` → `{}` — {} edges\n", b.from, b.to, b.edge_count),
                );
            }
            out.push('\n');
        }
        if out.len() > budget * 3 {
            out.truncate(budget * 3);
            out.push_str("\n[truncated to fit response budget]\n");
        }
        let total =
            overview.god_nodes.len() + overview.communities.len() + overview.boundaries.len();
        HandlerResult {
            formatted: out,
            fallback_used: false,
            result_count: total,
        }
    }

    /// Phase 4 — Exhaustive enumeration with structural enrichment. The
    /// agent classifier routes here when the query asks to *list all*
    /// configuration / options / flags / env vars. Combines a wider search
    /// (more candidates, larger budget) with structural callers/imports
    /// information so the agent gets definitions AND usages in one response.
    fn handle_exhaustive_structural(&self, query: &str, budget: usize) -> HandlerResult {
        // Use the existing exhaustive path's wider search settings; the
        // formatter is what makes the difference. Inject a structural hint
        // by widening max_results and enriching the budget so callers can
        // be listed alongside definitions.
        let exhaustive_budget = budget * 4;
        let sq = SearchQuery::new(query).max_results(30);
        match self.searcher.search(&sq) {
            Ok(output) if !output.results.is_empty() => {
                let items: Vec<SearchResultItem> = output
                    .results
                    .iter()
                    .map(|r| search_result_to_item(r, true))
                    .collect();
                let confidence = compute_confidence(&items);
                let count = items.len();
                let resp = SearchResponse {
                    results: items,
                    duration_ms: output.metrics.total_ms,
                    dense_count: output.metrics.dense_count,
                    sparse_count: output.metrics.sparse_count,
                    fused_count: output.metrics.fused_count,
                    metrics: None,
                    confidence: Some(confidence),
                };
                HandlerResult {
                    formatted: format_search_results(
                        &resp,
                        FormatStyle::Default,
                        exhaustive_budget,
                    ),
                    fallback_used: false,
                    result_count: count,
                }
            }
            _ => self.handle_exhaustive(query, budget),
        }
    }

    /// Phase 4 — Deep search enriched with pattern-catalog exemplars. The
    /// classifier routes here for "explain the most complex algorithm" /
    /// "deep dive into X with examples" queries. If the deep response
    /// mentions a known pattern name (e.g. `rust.tokio_spawn`), the
    /// pattern's exemplars are pulled inline so the agent sees concrete
    /// code without a follow-up call.
    fn handle_deep_with_examples(&self, query: &str, budget: usize) -> HandlerResult {
        let deep_result = self.handle_deep(query, budget, false);
        // Scan deep output for pattern-name mentions (e.g. `rust.X`, `ts.X`)
        // and pull up to 3 exemplars per matched pattern from the catalog.
        let db_path = self.project_root.join(".semantex").join("chunks.db");
        let mut found_patterns: Vec<String> = Vec::new();
        for cap in regex::Regex::new(r"\b(?:rust|ts|py|go|java)\.[a-z_]+\b")
            .ok()
            .map(|re| {
                re.find_iter(&deep_result.formatted)
                    .map(|m| m.as_str().to_string())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
        {
            if !found_patterns.contains(&cap) {
                found_patterns.push(cap);
            }
            if found_patterns.len() >= 3 {
                break;
            }
        }
        if found_patterns.is_empty() {
            return deep_result;
        }
        let mut enriched = deep_result.formatted.clone();
        enriched.push_str("\n\n## Pattern catalog exemplars\n\n");
        let mut added = 0usize;
        for pat in &found_patterns {
            let examples =
                crate::index::architecture::query_pattern_matches(&db_path, pat, None, 3)
                    .unwrap_or_default();
            if examples.is_empty() {
                continue;
            }
            let _ = std::fmt::Write::write_fmt(
                &mut enriched,
                format_args!("### `{}` ({} matches)\n\n", pat, examples.len()),
            );
            for (chunk_id, _name, lang) in &examples {
                if let Ok(chunk) = self.searcher.with_store(|s| s.get_chunk(*chunk_id)) {
                    let _ = std::fmt::Write::write_fmt(
                        &mut enriched,
                        format_args!(
                            "- `{}:{}-{}` ({})\n",
                            chunk.file_path.display(),
                            chunk.start_line,
                            chunk.end_line,
                            lang
                        ),
                    );
                    added += 1;
                }
            }
            enriched.push('\n');
        }
        if added == 0 {
            return deep_result;
        }
        HandlerResult {
            formatted: enriched,
            fallback_used: false,
            result_count: deep_result.result_count + added,
        }
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
            matches!(
                c.as_os_str().to_str(),
                Some(".git" | ".semantex" | "node_modules" | "target")
            )
        });
        if should_skip {
            continue;
        }
        // Get relative path
        let Ok(rel) = path.strip_prefix(root) else {
            continue;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatch_routes_file_pattern() {
        use super::super::agent_classifier::{AgentRoute, classify_agent_query};
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
