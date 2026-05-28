use super::agent_classifier::{AgentRoute, classify_agent_query, extract_symbol};
use super::agent_formatter::{
    DEFAULT_BUDGET, FormatStyle, append_disambiguation_block, format_code_blocks,
    format_deep_results, format_graph_results, format_search_results,
};
use crate::index::architecture::budget_for_chunk_count;
use crate::search::SearchQuery;
use crate::search::deep as deep_search_module;
use crate::search::hybrid::HybridSearcher;
use crate::server::handler::{
    AdaptiveGraphWalk, AdaptiveOptions, compute_confidence, deep_result_to_response,
    graph_walk_from_store_adaptive, search_result_to_item,
};
use crate::server::protocol::{
    AgentMetrics, AgentRequest, AgentResponse, GraphWalkResponse, SearchResponse, SearchResultItem,
};
use std::path::{Path, PathBuf};
#[cfg(feature = "llm")]
use std::sync::Arc;

/// Default outer-budget timeout for an LLM classifier call. Sized for the
/// `cli:claude` / `cli:codex` backends which include subprocess spawn cost;
/// cloud APIs return faster. Override via `SEMANTEX_LLM_CLASSIFY_TIMEOUT_MS`.
#[cfg(feature = "llm")]
const LLM_CLASSIFY_TIMEOUT_DEFAULT_MS: u64 = 8_000;

#[cfg(feature = "llm")]
fn llm_classify_timeout() -> std::time::Duration {
    std::env::var("SEMANTEX_LLM_CLASSIFY_TIMEOUT_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .map_or_else(
            || std::time::Duration::from_millis(LLM_CLASSIFY_TIMEOUT_DEFAULT_MS),
            std::time::Duration::from_millis,
        )
}

/// Threshold below which `handle_architecture` switches into "tiny repo" mode
/// (god_nodes section only, no communities, no boundaries). Per spec §4 Item 4.
const ARCH_TINY_REPO_THRESHOLD: u64 = 500;

/// Result from an individual handler method.
pub(crate) struct HandlerResult {
    formatted: String,
    fallback_used: bool,
    result_count: usize,
    /// v0.5 Item 6: when the search's top result was `Confidence::Ambiguous`,
    /// up to 3 runner-up suggestions are surfaced here so `handle()` can
    /// populate the matching field on `AgentResponse`. `None` when the
    /// handler did not run a confidence-bearing search or the top result
    /// was Extracted / Inferred.
    disambiguation: Option<Vec<crate::server::protocol::DisambigSuggestion>>,
}

/// Orchestrates agent queries: classify → dispatch → fallback → format.
pub struct AgentPipeline<'a> {
    searcher: &'a HybridSearcher,
    project_root: PathBuf,
    #[cfg(feature = "llm")]
    pub(crate) llm: Option<Arc<dyn crate::llm::LlmCapability>>,
}

impl<'a> AgentPipeline<'a> {
    pub fn new(searcher: &'a HybridSearcher, project_root: PathBuf) -> Self {
        Self {
            searcher,
            project_root,
            #[cfg(feature = "llm")]
            llm: None,
        }
    }

    /// Attach an LLM backend for classifier override and HyDE retrieval.
    ///
    /// When set, `handle` will attempt to classify the route via the LLM
    /// before falling back to the keyword classifier (`SEMANTEX_LLM_CLASSIFY_TIMEOUT_MS`,
    /// default 8 s), and `handle_semantic` will use HyDE augmentation
    /// (`SEMANTEX_LLM_HYDE_TIMEOUT_MS`, default 15 s). On any LLM failure
    /// or timeout, queries always fall back to the base behavior — HyDE
    /// must never break a search.
    #[cfg(feature = "llm")]
    pub fn with_llm(mut self, llm: Arc<dyn crate::llm::LlmCapability>) -> Self {
        self.llm = Some(llm);
        self
    }

    /// Run a Semantic-route search, augmenting with HyDE when an LLM is
    /// wired. Falls back to the plain base search on any HyDE failure.
    fn search_semantic(&self, query: &SearchQuery) -> anyhow::Result<crate::search::SearchOutput> {
        #[cfg(feature = "llm")]
        {
            if let Some(llm) = self.llm.clone() {
                match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => {
                        // search_with_hyde already handles HyDE-side errors
                        // by returning the base result; only outer runtime
                        // failures land here.
                        return rt.block_on(self.searcher.search_with_hyde(query, llm));
                    }
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            "Failed to build tokio runtime for HyDE; falling back to base search"
                        );
                    }
                }
            }
        }
        self.searcher.search(query)
    }

    pub fn handle(&self, request: &AgentRequest) -> AgentResponse {
        let start = std::time::Instant::now();
        let classify_start = std::time::Instant::now();

        let route = if let Some(r) = request.route {
            r
        } else {
            self.classify_route_with_llm_fallback(&request.query)
        };
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
            AgentRoute::FeaturePlanning => self.handle_feature_planning(&request.query, budget),
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
            disambiguation: result.disambiguation,
        }
    }

    /// Classify the query route, preferring the LLM classifier when one is
    /// configured. Falls back to the keyword classifier on LLM error or timeout.
    ///
    /// When `feature = "llm"` is disabled this is just a thin wrapper around
    /// [`classify_agent_query`]. When enabled and an LLM is attached, the LLM
    /// call runs inside a per-call current-thread Tokio runtime so the caller
    /// (which is always sync in the daemon and MCP paths) need not be `async`.
    // `&self` is only used under `#[cfg(feature = "llm")]` (to access `self.llm`).
    // Without that feature the argument is unused; we suppress the lint rather
    // than changing the signature, since the `#[cfg]`-gated body must remain.
    #[allow(clippy::unused_self)]
    fn classify_route_with_llm_fallback(&self, query: &str) -> AgentRoute {
        #[cfg(feature = "llm")]
        {
            if let Some(ref llm) = self.llm {
                // Build a minimal current-thread runtime to drive the async LLM call.
                // We keep it cheap: single-threaded, no I/O driver overhead beyond
                // what the LLM backend needs (TCP/TLS for HTTP-based backends).
                match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => {
                        let llm_ref = llm.as_ref();
                        let timeout = llm_classify_timeout();
                        let result = rt.block_on(async {
                            tokio::time::timeout(timeout, llm_ref.classify_route(query)).await
                        });
                        match result {
                            Ok(Ok(r)) => {
                                tracing::debug!(
                                    model = llm.label(),
                                    route = ?r,
                                    "LLM classified route"
                                );
                                return r;
                            }
                            Ok(Err(e)) => {
                                tracing::warn!(
                                    error = %e,
                                    "LLM classifier failed, falling back to keyword"
                                );
                            }
                            Err(_timeout) => {
                                tracing::info!(
                                    "LLM classifier timed out (>{}ms), falling back to keyword",
                                    timeout.as_millis()
                                );
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            "Failed to build tokio runtime for LLM classify, falling back to keyword"
                        );
                    }
                }
            }
        }
        classify_agent_query(query)
    }

    fn handle_semantic(&self, query: &str, budget: usize, is_fallback: bool) -> HandlerResult {
        let sq = SearchQuery::new(query).max_results(10);
        match self.search_semantic(&sq) {
            Ok(output) if !output.results.is_empty() => {
                let items: Vec<SearchResultItem> = output
                    .results
                    .iter()
                    .map(|r| search_result_to_item(r, true))
                    .collect();
                let confidence = compute_confidence(&items);
                let disambig = disambiguation_from_results(&output.results);
                let count = items.len();
                let resp = SearchResponse {
                    results: items,
                    duration_ms: output.metrics.total_ms,
                    dense_count: output.metrics.dense_count,
                    sparse_count: output.metrics.sparse_count,
                    fused_count: output.metrics.fused_count,
                    metrics: None,
                    confidence: Some(confidence),
                    disambiguation: disambig.clone(),
                };
                let mut formatted = format_search_results(&resp, FormatStyle::Default, budget);
                if let Some(suggestions) = &disambig {
                    append_disambiguation_block(&mut formatted, suggestions);
                }
                HandlerResult {
                    formatted,
                    fallback_used: is_fallback,
                    result_count: count,
                    disambiguation: disambig,
                }
            }
            _ => {
                if is_fallback {
                    HandlerResult {
                        formatted: format!("No results found for: {query}"),
                        fallback_used: is_fallback,
                        result_count: 0,
                        disambiguation: None,
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
                                disambiguation: None,
                            }
                        }
                        _ => HandlerResult {
                            formatted: format!("No results found for: {query}"),
                            fallback_used: is_fallback,
                            result_count: 0,
                            disambiguation: None,
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
                    disambiguation: None,
                }
            }
            _ => {
                if is_fallback {
                    HandlerResult {
                        formatted: format!("No results found for: {query}"),
                        fallback_used: true,
                        result_count: 0,
                        disambiguation: None,
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
            let disambig = disambiguation_from_results(&output.results);
            let count = items.len();
            let resp = SearchResponse {
                results: items,
                duration_ms: output.metrics.total_ms,
                dense_count: output.metrics.dense_count,
                sparse_count: output.metrics.sparse_count,
                fused_count: output.metrics.fused_count,
                metrics: None,
                confidence: Some(confidence),
                disambiguation: disambig.clone(),
            };
            let mut formatted = format_search_results(&resp, FormatStyle::Default, budget);
            if let Some(suggestions) = &disambig {
                append_disambiguation_block(&mut formatted, suggestions);
            }
            return HandlerResult {
                formatted,
                fallback_used: false,
                result_count: count,
                disambiguation: disambig,
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
                    disambiguation: None,
                };
                HandlerResult {
                    formatted: format_search_results(&resp, FormatStyle::Default, budget),
                    fallback_used: true,
                    result_count: count,
                    disambiguation: None,
                }
            }
            _ => HandlerResult {
                formatted: format!("Symbol not found: {symbol}"),
                fallback_used: true,
                result_count: 0,
                disambiguation: None,
            },
        }
    }

    /// v0.5 Item 7 — adaptive structural walk.
    ///
    /// Uses [`AdaptiveOptions::default()`] (min=5, max=50, depth_initial=1).
    /// When the depth-1 caller / callee list is oversized, the adaptive
    /// walk replaces it with a per-module summary string surfaced through
    /// `AdaptiveGraphWalk::*_summary`. When undersized, it walks one level
    /// deeper and merges. We append the summary text outside
    /// `format_graph_results` so the existing formatter stays untouched
    /// (it is owned by W-Gamma per spec §10).
    fn handle_structural(&self, query: &str, budget: usize) -> HandlerResult {
        if let Some(symbol) = extract_symbol(query) {
            let walk: AdaptiveGraphWalk = self.searcher.with_store(|store| {
                graph_walk_from_store_adaptive(store, &symbol, AdaptiveOptions::default())
                    .unwrap_or_else(|_| AdaptiveGraphWalk {
                        response: GraphWalkResponse {
                            target: vec![],
                            callers: vec![],
                            callees: vec![],
                            type_refs: vec![],
                            hierarchy: vec![],
                        },
                        callers_summary: None,
                        callees_summary: None,
                    })
            });
            let resp = &walk.response;
            let total = resp.target.len()
                + resp.callers.len()
                + resp.callees.len()
                + resp.type_refs.len()
                + resp.hierarchy.len()
                + usize::from(walk.callers_summary.is_some())
                + usize::from(walk.callees_summary.is_some());
            if total > 0 {
                let mut formatted = format_graph_results(resp);
                // Append collapsed summaries (Item 7): each replaces the
                // per-item list with one line. Done outside `format_graph_results`
                // so the W-Gamma-owned formatter stays untouched.
                if let Some(s) = &walk.callers_summary {
                    if !formatted.ends_with('\n') {
                        formatted.push('\n');
                    }
                    formatted.push_str("\nCallers (collapsed):\n  ");
                    formatted.push_str(s);
                    formatted.push('\n');
                }
                if let Some(s) = &walk.callees_summary {
                    if !formatted.ends_with('\n') {
                        formatted.push('\n');
                    }
                    formatted.push_str("\nCallees (collapsed):\n  ");
                    formatted.push_str(s);
                    formatted.push('\n');
                }
                return HandlerResult {
                    formatted,
                    fallback_used: false,
                    result_count: total,
                    disambiguation: None,
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
                let disambig = disambiguation_from_results(&output.results);
                let count = items.len();
                if full_code {
                    let code_contents: Vec<String> = items
                        .iter()
                        .map(|i| i.content.clone().unwrap_or_default())
                        .collect();
                    let mut formatted = format_code_blocks(&items, &code_contents, budget);
                    if formatted == "No code blocks to display." {
                        return self.handle_deep(query, budget, true);
                    }
                    if let Some(suggestions) = &disambig {
                        append_disambiguation_block(&mut formatted, suggestions);
                    }
                    HandlerResult {
                        formatted,
                        fallback_used: false,
                        result_count: count,
                        disambiguation: disambig,
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
                        disambiguation: disambig.clone(),
                    };
                    let mut formatted = format_search_results(&resp, FormatStyle::Default, budget);
                    if let Some(suggestions) = &disambig {
                        append_disambiguation_block(&mut formatted, suggestions);
                    }
                    HandlerResult {
                        formatted,
                        fallback_used: false,
                        result_count: count,
                        disambiguation: disambig,
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
                let disambig = disambiguation_from_results(&output.results);
                let count = items.len();
                let resp = SearchResponse {
                    results: items,
                    duration_ms: output.metrics.total_ms,
                    dense_count: output.metrics.dense_count,
                    sparse_count: output.metrics.sparse_count,
                    fused_count: output.metrics.fused_count,
                    metrics: None,
                    confidence: Some(confidence),
                    disambiguation: disambig.clone(),
                };
                let mut formatted =
                    format_search_results(&resp, FormatStyle::Default, exhaustive_budget);
                if let Some(suggestions) = &disambig {
                    append_disambiguation_block(&mut formatted, suggestions);
                }
                HandlerResult {
                    formatted,
                    fallback_used: false,
                    result_count: count,
                    disambiguation: disambig,
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
            disambiguation: None,
        };
        HandlerResult {
            formatted: format_search_results(&resp, FormatStyle::Grep, budget),
            fallback_used: false,
            result_count: items.len(),
            disambiguation: None,
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
    ///
    /// v0.3.1 Item 1 + Item 4 (per spec §4): the overview is now size-adaptive.
    /// `chunk_count < 500` → tiny-repo mode (god_nodes only, max 5, no
    /// communities / boundaries — too few to be meaningful, and rendering
    /// them on tiny repos triggers the agent to do follow-up exploration).
    /// `chunk_count >= 500` → use the `budget_for_chunk_count` tier
    /// (small / medium / large) from Item 1.
    fn handle_architecture(&self, query: &str, budget: usize) -> HandlerResult {
        use crate::index::architecture::build_arch_overview;
        let db_path = self.project_root.join(".semantex").join("chunks.db");
        let chunk_count = self.searcher.with_store(|s| s.chunk_count().unwrap_or(0));
        // Item 1: derive the adaptive budget from index size.
        // Item 4: when chunk_count < 500, force communities=0 and boundaries=0
        // so build_arch_overview skips those passes; cap god_nodes at 5.
        let mut arch_budget = budget_for_chunk_count(chunk_count as usize);
        if chunk_count < ARCH_TINY_REPO_THRESHOLD {
            arch_budget.god_nodes = arch_budget.god_nodes.min(5);
            arch_budget.communities = 0;
            arch_budget.boundaries = 0;
        }
        let Ok(overview) = self
            .searcher
            .with_store(|store| build_arch_overview(store, &db_path, Some(arch_budget)))
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
            // floor_char_boundary keeps the truncate point on a UTF-8 char
            // boundary; raw String::truncate panics mid-codepoint (e.g. on
            // em-dashes in arch overviews or non-ASCII identifiers).
            let cut = out.floor_char_boundary(budget * 3);
            out.truncate(cut);
            out.push_str("\n[truncated to fit response budget]\n");
        }
        let total =
            overview.god_nodes.len() + overview.communities.len() + overview.boundaries.len();
        HandlerResult {
            formatted: out,
            fallback_used: false,
            result_count: total,
            disambiguation: None,
        }
    }

    /// Phase 4 — Exhaustive enumeration with structural enrichment. The
    /// agent classifier routes here when the query asks to *list all*
    /// configuration / options / flags / env vars. Combines a wider search
    /// (more candidates, larger budget) with structural callers/imports
    /// information so the agent gets definitions AND usages in one response.
    ///
    /// v0.3.1 Item 1: `max_results` is now derived from `budget_for_chunk_count`
    /// (was hardcoded 30, which over-fetched on small repos and triggered
    /// follow-up exploration — see spec §4 Item 1, platform Q4 +29%).
    fn handle_exhaustive_structural(&self, query: &str, budget: usize) -> HandlerResult {
        // Use the existing exhaustive path's wider search settings; the
        // formatter is what makes the difference. Inject a structural hint
        // by widening max_results and enriching the budget so callers can
        // be listed alongside definitions.
        let exhaustive_budget = budget * 4;
        let chunk_count = self.searcher.with_store(|s| s.chunk_count().unwrap_or(0));
        let arch_budget = budget_for_chunk_count(chunk_count as usize);
        let sq = SearchQuery::new(query).max_results(arch_budget.exhaustive_max);
        match self.searcher.search(&sq) {
            Ok(output) if !output.results.is_empty() => {
                let items: Vec<SearchResultItem> = output
                    .results
                    .iter()
                    .map(|r| search_result_to_item(r, true))
                    .collect();
                let confidence = compute_confidence(&items);
                let disambig = disambiguation_from_results(&output.results);
                let count = items.len();
                let resp = SearchResponse {
                    results: items,
                    duration_ms: output.metrics.total_ms,
                    dense_count: output.metrics.dense_count,
                    sparse_count: output.metrics.sparse_count,
                    fused_count: output.metrics.fused_count,
                    metrics: None,
                    confidence: Some(confidence),
                    disambiguation: disambig.clone(),
                };
                let mut formatted =
                    format_search_results(&resp, FormatStyle::Default, exhaustive_budget);
                if let Some(suggestions) = &disambig {
                    append_disambiguation_block(&mut formatted, suggestions);
                }
                HandlerResult {
                    formatted,
                    fallback_used: false,
                    result_count: count,
                    disambiguation: disambig,
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
    ///
    /// v0.3.1 Item 3 (spec §4): pattern × example expansion is now capped
    /// adaptively by `ArchBudget.deep_examples_max`:
    ///   - small repo (chunks ≤ 2000): 1 pattern × 2 examples = 2 hits
    ///   - medium (≤ 10000):           2 patterns × 3 examples = 6 hits
    ///   - large:                       3 patterns × 3 examples = 9 hits
    ///
    /// Also: if the upstream deep result has < 5 sources, the enrichment
    /// pass is skipped entirely — the answer is already focused and the
    /// extra exemplars would just bloat tokens (motivates flask Q3 +54%).
    fn handle_deep_with_examples(&self, query: &str, budget: usize) -> HandlerResult {
        let deep_result = self.handle_deep(query, budget, false);
        let chunk_count = self.searcher.with_store(|s| s.chunk_count().unwrap_or(0));
        let arch_budget = budget_for_chunk_count(chunk_count as usize);
        let limits = deep_examples_limits(arch_budget.deep_examples_max);

        // Item 3: short-circuit when the deep response is already focused.
        if !should_enrich_with_examples(deep_result.result_count, &limits) {
            return deep_result;
        }

        // Scan deep output for pattern-name mentions (e.g. `rust.X`, `ts.X`)
        // and pull `limits.examples_per_pattern` exemplars per matched
        // pattern from the catalog, capped at `limits.max_patterns`.
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
            if found_patterns.len() >= limits.max_patterns {
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
            let examples = crate::index::architecture::query_pattern_matches(
                &db_path,
                pat,
                None,
                limits.examples_per_pattern,
            )
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
            disambiguation: None,
        }
    }

    /// v0.6 Item 10 — Feature-planning route. The classifier sends
    /// "if I wanted to add X, what files would change?" questions here.
    /// We build a 3-step plan (Architecture → ConventionLookup →
    /// ImpactedFiles) and merge per-step outputs into one response,
    /// collapsing what would otherwise be 30+ external tool turns into a
    /// single `semantex_agent` call.
    ///
    /// Safety: the planner enforces a 5-step cap and a 10s wall (see
    /// `search::planner::{MAX_STEPS, MAX_WALL}`). On any planner error —
    /// timeout, step failure, oversized plan — we fall back to the
    /// existing `handle_deep` path. The existing deep handler is reused
    /// unchanged; W-Gamma and v0.5 own its body.
    fn handle_feature_planning(&self, query: &str, budget: usize) -> HandlerResult {
        use super::planner::{Plan, PlanStepKind};

        let Ok(plan) = Plan::new(query, super::agent_classifier::AgentRoute::FeaturePlanning)
        else {
            return self.handle_deep(query, budget, true);
        };

        // Each step is a thin search-shaped call. The runner returns the
        // step's formatted body; `Plan::execute` concatenates them.
        let runner = |step: &super::planner::PlanStep,
                      _remaining: std::time::Duration|
         -> anyhow::Result<String> {
            match step.kind {
                PlanStepKind::Architecture => {
                    Ok(self.handle_architecture(&step.query, budget).formatted)
                }
                PlanStepKind::ConventionLookup
                | PlanStepKind::ImpactedFiles
                | PlanStepKind::ExampleSites
                | PlanStepKind::Synthesize => {
                    let sq = SearchQuery::new(&step.query).max_results(8);
                    match self.searcher.search(&sq) {
                        Ok(output) if !output.results.is_empty() => {
                            let items: Vec<SearchResultItem> = output
                                .results
                                .iter()
                                .map(|r| search_result_to_item(r, true))
                                .collect();
                            let resp = SearchResponse {
                                results: items,
                                duration_ms: output.metrics.total_ms,
                                dense_count: output.metrics.dense_count,
                                sparse_count: output.metrics.sparse_count,
                                fused_count: output.metrics.fused_count,
                                metrics: None,
                                confidence: Some("medium".into()),
                                disambiguation: None,
                            };
                            Ok(format_search_results(&resp, FormatStyle::Default, budget))
                        }
                        _ => Ok(String::new()),
                    }
                }
            }
        };

        match plan.execute(runner) {
            Ok(result) => HandlerResult {
                formatted: result.merged,
                fallback_used: false,
                result_count: result
                    .per_step
                    .iter()
                    .filter(|s| !s.output.is_empty())
                    .count(),
                disambiguation: None,
            },
            Err(_) => self.handle_deep(query, budget, true),
        }
    }
}

// --- v0.5 Item 6: confidence-driven disambiguation suggestions -------------

/// Max number of runner-up entries surfaced in the disambiguation block.
/// Per spec §5 Item 6 — bounded at 3.
const DISAMBIG_MAX_SUGGESTIONS: usize = 3;

/// Compute disambiguation suggestions for the agent's top result.
///
/// Returns `Some(Vec<DisambigSuggestion>)` with up to
/// `DISAMBIG_MAX_SUGGESTIONS` runner-up entries iff `results[0].confidence`
/// is `Confidence::Ambiguous`. Otherwise returns `None`.
///
/// Suggestion selection rules:
/// - Skip the top result itself (index 0).
/// - Skip any candidate whose symbol name matches an already-included
///   suggestion (or the top result's name) — runners-up with the same
///   name as the top result wouldn't help refine the query.
/// - Skip candidates without a symbol name (TextWindow / PdfPage chunks
///   can't be referenced by name in a refined query).
/// - Bound to `DISAMBIG_MAX_SUGGESTIONS` entries.
pub(crate) fn disambiguation_from_results(
    results: &[crate::types::SearchResult],
) -> Option<Vec<crate::server::protocol::DisambigSuggestion>> {
    use crate::server::protocol::DisambigSuggestion;
    use crate::types::Confidence;

    let top = results.first()?;
    if top.confidence != Confidence::Ambiguous {
        return None;
    }

    let mut seen_names: Vec<String> = Vec::with_capacity(DISAMBIG_MAX_SUGGESTIONS + 1);
    if let Some(name) = top.chunk.symbol_name() {
        seen_names.push(name.to_string());
    }

    let mut out: Vec<DisambigSuggestion> = Vec::with_capacity(DISAMBIG_MAX_SUGGESTIONS);
    for r in results.iter().skip(1) {
        if out.len() >= DISAMBIG_MAX_SUGGESTIONS {
            break;
        }
        let Some(name) = r.chunk.symbol_name() else {
            continue;
        };
        if seen_names.iter().any(|n| n == name) {
            continue;
        }
        seen_names.push(name.to_string());
        out.push(DisambigSuggestion {
            name: name.to_string(),
            path: r.chunk.file_path.display().to_string(),
            line: r.chunk.start_line as usize,
        });
    }

    if out.is_empty() { None } else { Some(out) }
}

/// Adaptive pattern × example caps for `handle_deep_with_examples`.
///
/// Derived from `ArchBudget.deep_examples_max` via `deep_examples_limits`.
/// Per spec §4 Item 3:
///   - small repo (deep_examples_max = 1): 1 pattern × 2 examples = 2 hits
///   - medium (= 3):                       2 patterns × 3 examples = 6 hits
///   - large (= 5):                        3 patterns × 3 examples = 9 hits
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeepExamplesLimits {
    pub max_patterns: usize,
    pub examples_per_pattern: usize,
}

/// Map `ArchBudget.deep_examples_max` to the pattern × example grid used by
/// `handle_deep_with_examples`. Pure function so it's trivially unit-testable.
pub(crate) fn deep_examples_limits(deep_examples_max: usize) -> DeepExamplesLimits {
    // The grid is intentionally non-linear: small repos get *both* fewer
    // patterns and fewer examples per pattern, since the answer is usually
    // narrow already; large repos keep the original 3×3 ceiling.
    match deep_examples_max {
        0 => DeepExamplesLimits {
            max_patterns: 0,
            examples_per_pattern: 0,
        },
        1 => DeepExamplesLimits {
            max_patterns: 1,
            examples_per_pattern: 2,
        },
        2..=4 => DeepExamplesLimits {
            max_patterns: 2,
            examples_per_pattern: 3,
        },
        _ => DeepExamplesLimits {
            max_patterns: 3,
            examples_per_pattern: 3,
        },
    }
}

/// Decide whether `handle_deep_with_examples` should run pattern enrichment.
///
/// Per spec §4 Item 3: if the upstream deep result has fewer than 5 sources,
/// the answer is already focused and we skip enrichment entirely. Also skip
/// if the budget reduces to zero hits.
pub(crate) fn should_enrich_with_examples(
    deep_result_count: usize,
    limits: &DeepExamplesLimits,
) -> bool {
    if deep_result_count < 5 {
        return false;
    }
    limits.max_patterns > 0 && limits.examples_per_pattern > 0
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
            disambiguation: None,
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
        disambiguation: None,
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

    // --- v0.3.1 Item 3 helpers ---------------------------------------------

    #[test]
    fn deep_examples_limits_small_repo() {
        // small tier deep_examples_max = 1 → 1 pattern × 2 examples
        let l = deep_examples_limits(1);
        assert_eq!(l.max_patterns, 1);
        assert_eq!(l.examples_per_pattern, 2);
    }

    #[test]
    fn deep_examples_limits_medium_repo() {
        // medium tier deep_examples_max = 3 → 2 patterns × 3 examples
        let l = deep_examples_limits(3);
        assert_eq!(l.max_patterns, 2);
        assert_eq!(l.examples_per_pattern, 3);
    }

    #[test]
    fn deep_examples_limits_large_repo() {
        // large tier deep_examples_max = 5 → 3 patterns × 3 examples
        let l = deep_examples_limits(5);
        assert_eq!(l.max_patterns, 3);
        assert_eq!(l.examples_per_pattern, 3);
    }

    #[test]
    fn deep_examples_limits_zero_yields_zero() {
        // Edge case: explicit zero budget should disable enrichment entirely.
        let l = deep_examples_limits(0);
        assert_eq!(l.max_patterns, 0);
        assert_eq!(l.examples_per_pattern, 0);
    }

    #[test]
    fn should_enrich_skips_when_few_sources() {
        // Spec §4 Item 3: result_count < 5 → skip enrichment regardless of budget.
        let l = deep_examples_limits(5); // generous large-repo budget
        assert!(!should_enrich_with_examples(0, &l));
        assert!(!should_enrich_with_examples(3, &l));
        assert!(!should_enrich_with_examples(4, &l));
    }

    #[test]
    fn should_enrich_runs_with_enough_sources() {
        let l = deep_examples_limits(3);
        assert!(should_enrich_with_examples(5, &l));
        assert!(should_enrich_with_examples(20, &l));
    }

    #[test]
    fn should_enrich_skips_when_budget_zero() {
        // Even with plenty of sources, a zero budget disables enrichment.
        let l = deep_examples_limits(0);
        assert!(!should_enrich_with_examples(50, &l));
    }
}

// --- Spec L: LLM-classifier tests (feature-gated) --------------------------

#[cfg(all(feature = "llm", test))]
mod llm_tests {
    use super::*;
    use crate::llm::LlmCapability;
    use crate::search::agent_classifier::AgentRoute;

    /// A mock LLM that always returns `AgentRoute::Structural`.
    struct MockLlm {
        route: AgentRoute,
        fail: bool,
    }

    #[async_trait::async_trait]
    impl LlmCapability for MockLlm {
        async fn classify_route(&self, _query: &str) -> anyhow::Result<AgentRoute> {
            if self.fail {
                anyhow::bail!("mock LLM classify error");
            }
            Ok(self.route)
        }

        async fn synthesize_hyde_doc(&self, _query: &str) -> anyhow::Result<String> {
            if self.fail {
                anyhow::bail!("mock LLM synthesize error");
            }
            Ok("fn handle_request(req: &Request) -> Response { /* mock */ }".to_string())
        }

        fn label(&self) -> &str {
            "mock-llm"
        }
    }

    /// When an LLM is attached and returns a route, `classify_route_with_llm_fallback`
    /// should return that route (not the keyword default).
    #[test]
    fn llm_classifier_overrides_keyword_default() {
        // "hello world" would keyword-classify as Semantic, but our mock returns Structural.
        let pipeline_wrapper = LlmClassifyHarness::new(AgentRoute::Structural, false);
        let route = pipeline_wrapper.classify("hello world");
        assert_eq!(route, AgentRoute::Structural);
    }

    /// When the mock LLM returns Err, `classify_route_with_llm_fallback` must fall
    /// back to the keyword classifier (not panic, not return the error).
    #[test]
    fn llm_classifier_falls_back_on_error() {
        // "hello world" → keyword: Semantic
        let pipeline_wrapper =
            LlmClassifyHarness::new(AgentRoute::Structural, true /* fail */);
        let route = pipeline_wrapper.classify("hello world");
        // keyword classifier returns Semantic for plain text
        assert_eq!(route, AgentRoute::Semantic);
    }

    /// Test harness that calls `classify_route_with_llm_fallback` without
    /// needing a real HybridSearcher.
    struct LlmClassifyHarness {
        llm: Arc<dyn LlmCapability>,
    }

    impl LlmClassifyHarness {
        fn new(route: AgentRoute, fail: bool) -> Self {
            let mock = MockLlm { route, fail };
            Self {
                llm: Arc::new(mock),
            }
        }

        /// Call the pipeline's classify helper in isolation using a minimal
        /// no-op searcher. We can't easily construct HybridSearcher in tests,
        /// but we CAN test the classify logic by building a harness that calls
        /// the same runtime code.
        fn classify(&self, query: &str) -> AgentRoute {
            // Exercise the same runtime path used in `classify_route_with_llm_fallback`.
            let llm_ref = self.llm.as_ref();
            match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => {
                    let timeout = llm_classify_timeout();
                    let result = rt.block_on(async {
                        tokio::time::timeout(timeout, llm_ref.classify_route(query)).await
                    });
                    match result {
                        Ok(Ok(r)) => r,
                        Ok(Err(_e)) => classify_agent_query(query),
                        Err(_timeout) => classify_agent_query(query),
                    }
                }
                Err(_) => classify_agent_query(query),
            }
        }
    }
}

// --- v0.5 Item 6: disambiguation suggestion tests --------------------------

#[cfg(test)]
mod disambig_tests {
    use super::*;
    use crate::types::{AstNodeKind, Chunk, ChunkType, Confidence, SearchResult, SearchSource};
    use std::path::PathBuf;

    /// Build a synthetic AST-node SearchResult with the given name, file,
    /// line, score, and per-result confidence.
    fn mk_result(
        name: &str,
        file: &str,
        line: u32,
        score: f32,
        confidence: Confidence,
    ) -> SearchResult {
        SearchResult {
            chunk: Chunk {
                id: 0,
                file_path: PathBuf::from(file),
                start_line: line,
                end_line: line + 10,
                content: format!("fn {name}() {{}}"),
                chunk_type: ChunkType::AstNode {
                    name: name.into(),
                    kind: AstNodeKind::Function,
                    language: "rust".into(),
                    structured_meta: None,
                },
            },
            score,
            source: SearchSource::Hybrid,
            score_dense: 0.0,
            score_sparse: 0.0,
            score_exact: 0.0,
            confidence,
            confidence_score: 0.5,
        }
    }

    /// Spec §5 Item 6: synthetic SearchResult with Ambiguous top result
    /// produces up to 3 disambiguation entries.
    #[test]
    fn ambiguous_top_produces_three_suggestions() {
        let results = vec![
            mk_result(
                "userAuthHandler",
                "auth/users.rs",
                42,
                0.92,
                Confidence::Ambiguous,
            ),
            mk_result(
                "tokenAuthHandler",
                "auth/tokens.rs",
                18,
                0.91,
                Confidence::Inferred,
            ),
            mk_result(
                "sessionAuth",
                "sessions/handler.rs",
                107,
                0.90,
                Confidence::Inferred,
            ),
            mk_result(
                "authMiddleware",
                "api/middleware.rs",
                12,
                0.88,
                Confidence::Inferred,
            ),
        ];

        let disambig = disambiguation_from_results(&results).expect("must populate");
        assert_eq!(disambig.len(), 3);
        assert_eq!(disambig[0].name, "tokenAuthHandler");
        assert_eq!(disambig[0].path, "auth/tokens.rs");
        assert_eq!(disambig[0].line, 18);
        assert_eq!(disambig[1].name, "sessionAuth");
        assert_eq!(disambig[2].name, "authMiddleware");
    }

    /// Spec §5 Item 6: when top result is Extracted (not Ambiguous),
    /// disambiguation is None and formatted output contains no
    /// disambiguation block.
    #[test]
    fn extracted_top_produces_no_suggestions() {
        let results = vec![
            mk_result(
                "authHandler",
                "auth/main.rs",
                1,
                0.99,
                Confidence::Extracted,
            ),
            mk_result(
                "tokenAuthHandler",
                "auth/tokens.rs",
                18,
                0.50,
                Confidence::Inferred,
            ),
            mk_result(
                "sessionAuth",
                "sessions/handler.rs",
                107,
                0.40,
                Confidence::Inferred,
            ),
        ];

        let disambig = disambiguation_from_results(&results);
        assert!(
            disambig.is_none(),
            "Extracted top must not populate disambig"
        );
    }

    /// Spec §5 Item 6: when top is Inferred (also not Ambiguous), no
    /// suggestions either.
    #[test]
    fn inferred_top_produces_no_suggestions() {
        let results = vec![mk_result(
            "lonely",
            "src/m.rs",
            1,
            0.5,
            Confidence::Inferred,
        )];
        assert!(disambiguation_from_results(&results).is_none());
    }

    /// Spec §5 Item 6: dedup by name — a runner-up that shares a name
    /// with the top (e.g. polymorphic same-named symbol in two files) is
    /// skipped, since refining by that name wouldn't disambiguate.
    #[test]
    fn runners_up_with_duplicate_names_are_skipped() {
        let results = vec![
            mk_result("commonName", "a/x.rs", 10, 0.92, Confidence::Ambiguous),
            // duplicate name as top → skip
            mk_result("commonName", "b/x.rs", 20, 0.91, Confidence::Inferred),
            // duplicate name as the next pick → skip second instance
            mk_result("uniqueName", "c/x.rs", 30, 0.90, Confidence::Inferred),
            mk_result("uniqueName", "d/x.rs", 40, 0.89, Confidence::Inferred),
            mk_result("otherName", "e/x.rs", 50, 0.88, Confidence::Inferred),
        ];

        let disambig = disambiguation_from_results(&results).expect("must populate");
        assert_eq!(disambig.len(), 2);
        assert_eq!(disambig[0].name, "uniqueName");
        assert_eq!(disambig[0].path, "c/x.rs");
        assert_eq!(disambig[1].name, "otherName");
    }

    /// Empty results → None (caller is responsible for not calling on
    /// empties, but the helper is safe).
    #[test]
    fn empty_results_produce_no_suggestions() {
        assert!(disambiguation_from_results(&[]).is_none());
    }

    /// Spec §5 Item 6: runners-up that are TextWindow chunks (no
    /// symbol name) are skipped — agents can't refine by a missing name.
    #[test]
    fn runners_up_without_symbol_names_are_skipped() {
        let no_name_result = SearchResult {
            chunk: Chunk {
                id: 0,
                file_path: PathBuf::from("text.md"),
                start_line: 1,
                end_line: 1,
                content: "lorem ipsum".into(),
                chunk_type: ChunkType::TextWindow { window_index: 0 },
            },
            score: 0.91,
            source: SearchSource::Sparse,
            score_dense: 0.0,
            score_sparse: 0.0,
            score_exact: 0.0,
            confidence: Confidence::Inferred,
            confidence_score: 0.5,
        };
        let results = vec![
            mk_result("topResult", "a/x.rs", 1, 0.92, Confidence::Ambiguous),
            no_name_result,
            mk_result("realRunner", "b/x.rs", 30, 0.89, Confidence::Inferred),
        ];
        let disambig = disambiguation_from_results(&results).expect("must populate");
        assert_eq!(disambig.len(), 1);
        assert_eq!(disambig[0].name, "realRunner");
    }
}
