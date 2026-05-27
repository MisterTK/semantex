use crate::index::storage::ChunkStore;
use crate::search::SearchQuery;
use crate::search::hybrid::HybridSearcher;
use crate::types::{Chunk, ChunkType};

use super::protocol::{
    AgentRequest, DeepResponseMetrics, DeepSearchRequest, DeepSearchResponse, DeepSearchSource,
    ErrorResponse, GraphWalkRequest, GraphWalkResponse, HealthResponse, MultiSearchRequest,
    MultiSearchResponse, Request, Response, SearchRequest, SearchResponse, SearchResultItem,
    ShutdownResponse,
};
use crate::search::deep as deep_search_module;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Handles incoming requests using a HybridSearcher
pub struct Handler<'a> {
    searcher: &'a HybridSearcher,
    search_count: &'a AtomicU64,
    project_root: PathBuf,
}

impl<'a> Handler<'a> {
    pub fn new(
        searcher: &'a HybridSearcher,
        search_count: &'a AtomicU64,
        project_root: PathBuf,
    ) -> Self {
        Self {
            searcher,
            search_count,
            project_root,
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
            Request::Agent(ref req) => self.handle_agent(req),
        }
    }

    fn handle_agent(&self, req: &AgentRequest) -> Response {
        use crate::search::agent::AgentPipeline;
        let pipeline = AgentPipeline::new(self.searcher, self.project_root.clone());
        Response::Agent(pipeline.handle(req))
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
                            chunk_type_meta(&r.chunk.chunk_type);

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

                let confidence = Some(compute_confidence(&items));
                // v0.5 Item 6: surface disambiguation suggestions for the
                // programmatic search path too, so MCP / CLI clients reading
                // `SearchResponse` directly (not via the agent) can pick up
                // structured runner-ups.
                let disambiguation =
                    crate::search::agent::disambiguation_from_results(&output.results);

                Response::Search(SearchResponse {
                    results: items,
                    duration_ms,
                    dense_count: metrics.dense_count,
                    sparse_count: metrics.sparse_count,
                    fused_count: metrics.fused_count,
                    metrics: Some(metrics),
                    confidence,
                    disambiguation,
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
            Ok(result) => Response::DeepSearch(deep_result_to_response(result)),
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

/// Compute a confidence string from sorted search result items.
/// Items must be sorted by descending score.
pub(crate) fn compute_confidence(items: &[SearchResultItem]) -> String {
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

/// Convert a deep search result into the wire response type.
pub(crate) fn deep_result_to_response(
    result: crate::search::deep::DeepResult,
) -> DeepSearchResponse {
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

/// Perform a graph walk from a direct ChunkStore reference.
/// Used by both the daemon handler and the direct (no-daemon) fallback path.
///
/// Behaviour is unchanged from earlier releases: each section is capped at
/// `MAX_PER_SECTION = 20` and no adaptive collapse / expand is performed.
/// The agent pipeline calls `graph_walk_from_store_adaptive` instead, which
/// takes [`AdaptiveOptions`] and applies the v0.5 Item 7 logic.
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

// --- v0.5 Item 7: adaptive structural walk ---------------------------------

/// Generous cap on the initial fetch in the adaptive walk so the collapse
/// pass has real data to summarize. The collapse pass replaces the per-item
/// list with a summary, so this larger cap doesn't blow up the response
/// size for the happy path (it's only paid when we end up collapsing).
const MAX_PER_SECTION_ADAPTIVE: usize = 200;

/// Cap for the type-refs / hierarchy sections in the adaptive walk. Those
/// sections don't carry the response-size regressions that motivated Item 7,
/// so they keep the legacy depth-1 / 20-cap behavior.
const TYPE_AND_HIERARCHY_CAP: usize = 20;

/// Options for [`graph_walk_from_store_adaptive`] (v0.5 Item 7).
///
/// Tunes how the agent's structural-walk handler reacts to caller/callee
/// list size:
/// - `> max_callers`: collapse the per-caller list to a per-module summary
///   so a heavily-used utility doesn't flood the response.
/// - `< min_callers`: re-walk at `depth_initial + 1` and merge results so
///   small-fan-in symbols still surface meaningful context.
///
/// Defaults (`min=5`, `max=50`, `depth_initial=1`) preserve current behavior
/// for callers in `[5, 50]` — only out-of-band sizes trigger the adaptive
/// passes.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveOptions {
    pub min_callers: usize,
    pub max_callers: usize,
    pub depth_initial: usize,
}

impl Default for AdaptiveOptions {
    fn default() -> Self {
        Self {
            min_callers: 5,
            max_callers: 50,
            depth_initial: 1,
        }
    }
}

/// Output of [`graph_walk_from_store_adaptive`].
///
/// `response` is the standard wire-format [`GraphWalkResponse`]. When a
/// collapse pass fires, the corresponding `Vec` (callers / callees) is
/// emptied and the human-readable per-module summary is placed in the
/// matching `*_summary` slot. The agent-side formatter substitutes the
/// summary text in place of the per-item list. When an expand pass fires,
/// the `Vec` grows and `*_summary` stays `None`.
#[derive(Debug, Clone)]
pub struct AdaptiveGraphWalk {
    pub response: GraphWalkResponse,
    /// Pre-rendered "N callers across M files: ..." string when callers were collapsed.
    pub callers_summary: Option<String>,
    /// Pre-rendered "N callees across M files: ..." string when callees were collapsed.
    pub callees_summary: Option<String>,
}

/// Adaptive variant of [`graph_walk_from_store`] (v0.5 Item 7).
///
/// 1. Performs the initial walk at `opts.depth_initial`. Each section is
///    capped at `MAX_PER_SECTION_ADAPTIVE = 200` to leave headroom for the
///    collapse / expand passes.
/// 2. If `callers.len() > opts.max_callers`, the per-caller list is
///    replaced by a "N callers across M files: src/a/* (k), …" summary
///    surfaced via [`AdaptiveGraphWalk::callers_summary`].
/// 3. If `callers.len() < opts.min_callers`, a depth-2 walk is performed
///    on the depth-1 callers and the new caller chunks are merged in
///    (deduped by chunk id; never re-adds a `target_ids` entry).
/// 4. The same logic mirrors over to callees.
///
/// `type_refs` and `hierarchy` are unchanged — their sizes have not been
/// observed to trigger the same response-size regressions.
pub(crate) fn graph_walk_from_store_adaptive(
    store: &ChunkStore,
    symbol: &str,
    opts: AdaptiveOptions,
) -> anyhow::Result<AdaptiveGraphWalk> {
    let matches = store.lookup_symbol_exact(symbol)?;
    if matches.is_empty() {
        return Ok(AdaptiveGraphWalk {
            response: GraphWalkResponse {
                target: vec![],
                callers: vec![],
                callees: vec![],
                type_refs: vec![],
                hierarchy: vec![],
            },
            callers_summary: None,
            callees_summary: None,
        });
    }

    let target_ids: Vec<u64> = matches.iter().map(|(id, _, _)| *id as u64).collect();
    let target_chunks = store.get_chunks(&target_ids)?;
    let target_items: Vec<SearchResultItem> = target_chunks.iter().map(chunk_to_item).collect();

    // Depth-1 incoming (callers) and outgoing (callees) neighbors.
    let mut incoming_ids = unique_neighbors(
        &store.get_call_edges_to(&target_ids)?,
        |(_, src)| *src,
        &target_ids,
        MAX_PER_SECTION_ADAPTIVE,
    );
    let mut outgoing_ids = unique_neighbors(
        &store.get_call_edges_from(&target_ids)?,
        |(_, callee)| *callee,
        &target_ids,
        MAX_PER_SECTION_ADAPTIVE,
    );

    // Expand pass: if the initial walk returned too few neighbors on either
    // side, walk one more level out and merge.
    let mut visited: std::collections::HashSet<u64> = target_ids.iter().copied().collect();
    visited.extend(incoming_ids.iter().copied());
    visited.extend(outgoing_ids.iter().copied());
    if incoming_ids.len() < opts.min_callers && opts.depth_initial < 2 && !incoming_ids.is_empty() {
        let depth2 = unique_neighbors(
            &store.get_call_edges_to(&incoming_ids)?,
            |(_, src)| *src,
            &[],
            MAX_PER_SECTION_ADAPTIVE,
        );
        for id in depth2 {
            if visited.insert(id) {
                incoming_ids.push(id);
            }
        }
    }
    if outgoing_ids.len() < opts.min_callers && opts.depth_initial < 2 && !outgoing_ids.is_empty() {
        let depth2 = unique_neighbors(
            &store.get_call_edges_from(&outgoing_ids)?,
            |(_, callee)| *callee,
            &[],
            MAX_PER_SECTION_ADAPTIVE,
        );
        for id in depth2 {
            if visited.insert(id) {
                outgoing_ids.push(id);
            }
        }
    }

    // Resolve neighbor IDs to items.
    let incoming_chunks = store.get_chunks(&incoming_ids)?;
    let outgoing_chunks = store.get_chunks(&outgoing_ids)?;
    let mut incoming_items: Vec<SearchResultItem> =
        incoming_chunks.iter().map(chunk_to_item).collect();
    let mut outgoing_items: Vec<SearchResultItem> =
        outgoing_chunks.iter().map(chunk_to_item).collect();

    // Collapse pass: oversized lists get replaced by a per-module summary.
    let (in_summary, out_summary) =
        collapse_if_oversized(&mut incoming_items, &mut outgoing_items, opts.max_callers);

    // Other sections keep the legacy depth-1 / 20-cap behavior.
    let type_ref_ids = {
        let usage = store.get_type_ref_edges_from_usages(&target_ids)?;
        let def = store.get_type_ref_edges_to_defs(&target_ids)?;
        usage
            .iter()
            .map(|(_, def)| *def)
            .chain(def.iter().map(|(_, usage)| *usage))
            .filter(|id| !target_ids.contains(id))
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .take(TYPE_AND_HIERARCHY_CAP)
            .collect::<Vec<_>>()
    };
    let hierarchy_ids = store
        .get_hierarchy_edges_for(&target_ids)?
        .iter()
        .map(|(_, related)| *related)
        .filter(|id| !target_ids.contains(id))
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .take(TYPE_AND_HIERARCHY_CAP)
        .collect::<Vec<_>>();
    let type_ref_chunks = store.get_chunks(&type_ref_ids)?;
    let hierarchy_chunks = store.get_chunks(&hierarchy_ids)?;

    Ok(AdaptiveGraphWalk {
        response: GraphWalkResponse {
            target: target_items,
            callers: incoming_items,
            callees: outgoing_items,
            type_refs: type_ref_chunks.iter().map(chunk_to_item).collect(),
            hierarchy: hierarchy_chunks.iter().map(chunk_to_item).collect(),
        },
        callers_summary: in_summary,
        callees_summary: out_summary,
    })
}

/// Apply the Item 7 collapse pass to a depth-1 (caller, callee) item pair.
/// Each list that exceeds `max` is replaced by a per-module summary string
/// and the original `Vec` is cleared. Returns `(callers_summary, callees_summary)`.
fn collapse_if_oversized(
    incoming: &mut Vec<SearchResultItem>,
    outgoing: &mut Vec<SearchResultItem>,
    max: usize,
) -> (Option<String>, Option<String>) {
    let mut in_summary = None;
    if incoming.len() > max {
        in_summary = Some(format_module_summary(incoming, "callers"));
        incoming.clear();
    }
    let mut out_summary = None;
    if outgoing.len() > max {
        out_summary = Some(format_module_summary(outgoing, "callees"));
        outgoing.clear();
    }
    (in_summary, out_summary)
}

/// Build a deduped, filtered, capped neighbor ID list from a slice of edges.
/// Pulled out so callers/callees use identical semantics (and to keep
/// `graph_walk_from_store_adaptive` readable).
fn unique_neighbors(
    edges: &[(u64, u64)],
    extract: impl Fn(&(u64, u64)) -> u64,
    exclude: &[u64],
    max: usize,
) -> Vec<u64> {
    let mut set: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut out: Vec<u64> = Vec::with_capacity(edges.len().min(max));
    for e in edges {
        let id = extract(e);
        if exclude.contains(&id) {
            continue;
        }
        if set.insert(id) {
            out.push(id);
            if out.len() >= max {
                break;
            }
        }
    }
    out
}

/// Render a per-module summary for a list of `SearchResultItem`s.
///
/// Format (matches v0.5 spec §5 Item 7):
/// `"78 callers across 12 files: src/auth/* (24), src/api/* (18), src/utils/* (8), ..."`
///
/// "module" here means the directory of the file path, with a `/*` suffix.
/// Items are grouped by directory and the top 3 directories by count are
/// listed; a tail count covers the remainder. The function is pure so it's
/// trivially unit-testable from the adaptive-walk tests.
pub(crate) fn format_module_summary(items: &[SearchResultItem], label: &str) -> String {
    use std::collections::BTreeMap;
    use std::fmt::Write as _;
    if items.is_empty() {
        return format!("0 {label}");
    }
    let total = items.len();
    let mut by_dir: BTreeMap<String, usize> = BTreeMap::new();
    for it in items {
        let dir = module_dir_of(&it.file);
        *by_dir.entry(dir).or_insert(0) += 1;
    }
    let file_count = by_dir.len();
    let mut entries: Vec<(String, usize)> = by_dir.into_iter().collect();
    // Sort by count descending, then by directory name for stable output.
    entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let top: Vec<&(String, usize)> = entries.iter().take(3).collect();
    let listed: usize = top.iter().map(|(_, c)| *c).sum();
    let mut out = format!("{total} {label} across {file_count} files: ");
    let mut first = true;
    for (dir, count) in &top {
        if !first {
            out.push_str(", ");
        }
        let _ = write!(out, "{dir}/* ({count})");
        first = false;
    }
    if listed < total {
        let tail = total - listed;
        let _ = write!(out, ", ... +{tail} more");
    }
    out
}

/// Extract a "module" (directory) label from a file path. Falls back to
/// "<root>" if the path has no directory component (e.g. `main.rs`).
fn module_dir_of(file: &str) -> String {
    let path = std::path::Path::new(file);
    match path.parent().and_then(|p| p.to_str()) {
        Some(p) if !p.is_empty() => p.to_string(),
        _ => "<root>".to_string(),
    }
}

/// Extract display metadata from a ChunkType.
/// Returns `(chunk_type_str, name, language, kind, summary)`.
pub(crate) fn chunk_type_meta(
    ct: &crate::types::ChunkType,
) -> (
    String,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
) {
    match ct {
        crate::types::ChunkType::AstNode {
            name,
            kind,
            language,
            structured_meta,
            ..
        } => {
            let summary = structured_meta
                .as_ref()
                .map(|meta| meta.display_summary())
                .filter(|s| !s.is_empty());
            (
                "AstNode".to_string(),
                Some(name.clone()),
                Some(language.clone()),
                Some(kind.to_string()),
                summary,
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
    }
}

/// Convert a Chunk to a SearchResultItem (with kind/summary populated)
pub(crate) fn chunk_to_item(chunk: &Chunk) -> SearchResultItem {
    let (chunk_type_str, name, language, kind, summary) = chunk_type_meta(&chunk.chunk_type);
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

/// Convert a SearchResult to a SearchResultItem (with optional content).
/// Used by the agent pipeline to convert search output to protocol items.
pub(crate) fn search_result_to_item(
    result: &crate::types::SearchResult,
    include_content: bool,
) -> SearchResultItem {
    let (chunk_type_str, name, language, kind, summary) = chunk_type_meta(&result.chunk.chunk_type);

    let content = if include_content {
        Some(result.chunk.content.clone())
    } else {
        None
    };

    SearchResultItem {
        file: result.chunk.file_path.display().to_string(),
        start_line: result.chunk.start_line,
        end_line: result.chunk.end_line,
        score: result.score,
        source: format!("{:?}", result.source),
        chunk_type: chunk_type_str,
        name,
        language,
        content,
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

// --- v0.5 Item 7 tests: adaptive structural walk ---------------------------

#[cfg(test)]
mod adaptive_walk_tests {
    use super::*;
    use crate::types::{Chunk, ChunkType};
    use std::path::PathBuf;
    use tempfile::TempDir;

    /// Insert a code-bearing chunk into the store and return its row id.
    /// We tag it as `AstNode` with the supplied symbol name + kind so the
    /// adaptive walk's `lookup_symbol_exact` join hits a real chunk.
    fn insert_named_chunk(store: &ChunkStore, file: &str, symbol: &str) -> u64 {
        let chunk = Chunk {
            id: 0,
            file_path: PathBuf::from(file),
            start_line: 1,
            end_line: 5,
            content: format!("fn {symbol}() {{}}"),
            chunk_type: ChunkType::AstNode {
                name: symbol.into(),
                kind: crate::types::AstNodeKind::Function,
                language: "rust".into(),
                structured_meta: None,
            },
        };
        let cid = store.insert_chunk(&chunk, 0xdead_beef, 0).unwrap();
        store
            .insert_symbol_def(cid, symbol, "function", file)
            .unwrap();
        cid
    }

    /// Test 1 (spec §5 Item 7): a graph with 100 callers collapses to a
    /// per-module summary string and the callers list is emptied.
    #[test]
    fn adaptive_walk_collapses_oversized_caller_list() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("chunks.db");
        let store = ChunkStore::open(&db_path).unwrap();

        let target = insert_named_chunk(&store, "src/util/format.rs", "format_string");
        // 100 callers, 60 in src/auth/, 30 in src/api/, 10 in src/utils/
        for i in 0..60 {
            let cid = insert_named_chunk(&store, &format!("src/auth/m{i}.rs"), &format!("a{i}"));
            store
                .store_call_graph_edge(cid, "format_string", Some(target))
                .unwrap();
        }
        for i in 0..30 {
            let cid = insert_named_chunk(&store, &format!("src/api/m{i}.rs"), &format!("b{i}"));
            store
                .store_call_graph_edge(cid, "format_string", Some(target))
                .unwrap();
        }
        for i in 0..10 {
            let cid = insert_named_chunk(&store, &format!("src/utils/m{i}.rs"), &format!("c{i}"));
            store
                .store_call_graph_edge(cid, "format_string", Some(target))
                .unwrap();
        }

        let walk =
            graph_walk_from_store_adaptive(&store, "format_string", AdaptiveOptions::default())
                .unwrap();
        // Callers list is emptied; summary string is populated and references the
        // grouped modules with their counts.
        assert!(
            walk.response.callers.is_empty(),
            "expected callers list to be replaced by summary, got {} items",
            walk.response.callers.len()
        );
        let summary = walk.callers_summary.expect("collapse should fire");
        assert!(summary.contains("100 callers"), "summary: {summary}");
        assert!(summary.contains("src/auth"), "summary: {summary}");
        assert!(summary.contains("(60)"), "summary: {summary}");
        assert!(summary.contains("src/api"), "summary: {summary}");
    }

    /// Test 2 (spec §5 Item 7): a graph with 2 direct callers walks one
    /// more level out and merges the depth-2 callers in.
    #[test]
    fn adaptive_walk_expands_when_undersized() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("chunks.db");
        let store = ChunkStore::open(&db_path).unwrap();

        let target = insert_named_chunk(&store, "src/lib.rs", "tiny_target");
        // 2 direct callers at depth=1
        let d1a = insert_named_chunk(&store, "src/a/m1.rs", "d1a");
        let d1b = insert_named_chunk(&store, "src/a/m2.rs", "d1b");
        store
            .store_call_graph_edge(d1a, "tiny_target", Some(target))
            .unwrap();
        store
            .store_call_graph_edge(d1b, "tiny_target", Some(target))
            .unwrap();

        // 3 depth-2 callers (each calling d1a or d1b, not the target directly)
        let d2a = insert_named_chunk(&store, "src/b/m1.rs", "d2a");
        let d2b = insert_named_chunk(&store, "src/b/m2.rs", "d2b");
        let d2c = insert_named_chunk(&store, "src/b/m3.rs", "d2c");
        store.store_call_graph_edge(d2a, "d1a", Some(d1a)).unwrap();
        store.store_call_graph_edge(d2b, "d1a", Some(d1a)).unwrap();
        store.store_call_graph_edge(d2c, "d1b", Some(d1b)).unwrap();

        let walk =
            graph_walk_from_store_adaptive(&store, "tiny_target", AdaptiveOptions::default())
                .unwrap();
        // No collapse (under max).
        assert!(walk.callers_summary.is_none());
        // 2 direct + 3 depth-2 = 5 unique callers, all merged in.
        let names: Vec<String> = walk
            .response
            .callers
            .iter()
            .filter_map(|i| i.name.clone())
            .collect();
        assert!(names.contains(&"d1a".to_string()), "names: {names:?}");
        assert!(names.contains(&"d1b".to_string()), "names: {names:?}");
        assert!(names.contains(&"d2a".to_string()), "names: {names:?}");
        assert!(names.contains(&"d2b".to_string()), "names: {names:?}");
        assert!(names.contains(&"d2c".to_string()), "names: {names:?}");
        assert!(
            walk.response.callers.len() >= 5,
            "expected ≥5 callers after expand, got {}",
            walk.response.callers.len()
        );
    }

    /// Test 3 (spec §5 Item 7): a graph with 20 callers is in the
    /// [min, max] band and the output is unchanged — no collapse summary,
    /// no extra depth-2 entries.
    #[test]
    fn adaptive_walk_passes_through_midsized_list() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("chunks.db");
        let store = ChunkStore::open(&db_path).unwrap();

        let target = insert_named_chunk(&store, "src/lib.rs", "mid_target");
        for i in 0..20 {
            let cid = insert_named_chunk(&store, &format!("src/m{i}.rs"), &format!("c{i}"));
            store
                .store_call_graph_edge(cid, "mid_target", Some(target))
                .unwrap();
        }

        let walk = graph_walk_from_store_adaptive(&store, "mid_target", AdaptiveOptions::default())
            .unwrap();
        assert!(walk.callers_summary.is_none(), "no collapse expected");
        assert_eq!(walk.response.callers.len(), 20);
    }

    /// Unit test on the pure summary formatter: top-3 modules listed in
    /// descending count order, tail count covers the remainder, the
    /// "<root>" bucket is used for files without a directory.
    #[test]
    fn format_module_summary_renders_top3_and_tail() {
        let items: Vec<SearchResultItem> = ["src/a/x.rs"; 5]
            .iter()
            .chain(["src/b/x.rs"; 4].iter())
            .chain(["src/c/x.rs"; 3].iter())
            .chain(["src/d/x.rs"; 2].iter())
            .chain(["src/e/x.rs"; 1].iter())
            .map(|f| SearchResultItem {
                file: (*f).into(),
                start_line: 1,
                end_line: 1,
                score: 0.0,
                source: "GraphWalk".into(),
                chunk_type: "AstNode".into(),
                name: None,
                language: None,
                content: None,
                kind: None,
                summary: None,
            })
            .collect();
        let s = format_module_summary(&items, "callers");
        assert!(s.starts_with("15 callers across 5 files: "), "{s}");
        assert!(s.contains("src/a/* (5)"), "{s}");
        assert!(s.contains("src/b/* (4)"), "{s}");
        assert!(s.contains("src/c/* (3)"), "{s}");
        // The tail covers the remaining 2 + 1 = 3 items not in the top-3.
        assert!(s.contains("... +3 more"), "{s}");
    }

    #[test]
    fn format_module_summary_handles_root_files() {
        let items = vec![SearchResultItem {
            file: "main.rs".into(),
            start_line: 1,
            end_line: 1,
            score: 0.0,
            source: "GraphWalk".into(),
            chunk_type: "AstNode".into(),
            name: None,
            language: None,
            content: None,
            kind: None,
            summary: None,
        }];
        let s = format_module_summary(&items, "callees");
        assert!(s.contains("<root>/* (1)"), "{s}");
    }

    /// `AdaptiveOptions::default()` defaults must match the spec text
    /// (`min=5`, `max=50`, `depth_initial=1`).
    #[test]
    fn adaptive_options_defaults_match_spec() {
        let o = AdaptiveOptions::default();
        assert_eq!(o.min_callers, 5);
        assert_eq!(o.max_callers, 50);
        assert_eq!(o.depth_initial, 1);
    }
}
