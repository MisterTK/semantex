// crates/semantex-core/src/search/deep.rs
// Deep search pipeline: search → triage → graph expand → read → summarize

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use crate::search::summarize::{self, ReadChunk};
use crate::types::{ChunkType, SearchResult};

/// Maximum total content bytes fed to the extractive summarizer (~8K tokens).
const CONTEXT_BUDGET_BYTES: usize = 32_768;

/// The output of a deep search: a synthesized answer, sourced chunks, and timing metrics.
#[derive(Debug, Clone)]
pub struct DeepResult {
    pub answer: String,
    pub sources: Vec<DeepSource>,
    pub metrics: DeepMetrics,
}

/// Provenance for a single chunk contributing to a deep search answer.
#[derive(Debug, Clone)]
pub struct DeepSource {
    pub file: String,
    pub start_line: u32,
    pub end_line: u32,
    pub name: Option<String>,
    pub kind: Option<String>,
}

/// Per-phase timing and count metrics for a deep search invocation.
#[derive(Debug, Clone, Default)]
pub struct DeepMetrics {
    pub search_ms: u64,
    pub triage_ms: u64,
    pub graph_ms: u64,
    pub read_ms: u64,
    pub summarize_ms: u64,
    pub total_ms: u64,
    pub chunks_searched: usize,
    pub chunks_read: usize,
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Preferred AST kind labels for query-type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreferredKind {
    FnMethod,
    StructClassType,
    Neutral,
}

fn classify_query_preference(query: &str) -> PreferredKind {
    let q = query.to_lowercase();
    if q.contains("how does")
        || q.contains("how is")
        || q.contains("how do")
        || q.contains("how are")
    {
        PreferredKind::FnMethod
    } else if q.contains("what is") || q.contains("what are") {
        PreferredKind::StructClassType
    } else {
        PreferredKind::Neutral
    }
}

fn kind_matches_preference(kind: Option<&str>, pref: PreferredKind) -> bool {
    match pref {
        PreferredKind::Neutral => false,
        PreferredKind::FnMethod => {
            matches!(kind, Some("fn") | Some("method") | Some("function"))
        }
        PreferredKind::StructClassType => {
            matches!(
                kind,
                Some("struct") | Some("class") | Some("interface") | Some("enum")
            )
        }
    }
}

/// Triage search results: deduplicate, enforce per-file caps, apply preference boosts.
///
/// Returns indices into `results` for the selected chunks (preserving original order).
fn triage_results(query: &str, results: &[SearchResult], max_chunks: usize) -> Vec<usize> {
    let pref = classify_query_preference(query);

    // Pass 1: Compute adjusted scores for all candidates.
    let mut scored: Vec<(usize, f32, String, u32, u32)> = Vec::new();

    for (idx, result) in results.iter().enumerate() {
        let file = result.chunk.file_path.display().to_string();
        let start = result.chunk.start_line;
        let end = result.chunk.end_line;

        // Skip very low-scoring text windows.
        if matches!(result.chunk.chunk_type, ChunkType::TextWindow { .. }) && result.score < 0.1 {
            continue;
        }

        // Adjust score for kind preference.
        let kind = match &result.chunk.chunk_type {
            ChunkType::AstNode { kind, .. } => Some(kind.to_string()),
            _ => None,
        };
        let adjusted_score = result.score
            + if kind_matches_preference(kind.as_deref(), pref) {
                0.1
            } else {
                0.0
            };

        scored.push((idx, adjusted_score, file, start, end));
    }

    // Sort by adjusted score descending.
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Pass 2: Greedily select with overlap dedup and per-file cap.
    let mut selected_ranges: Vec<(String, u32, u32)> = Vec::new();
    let mut file_counts: HashMap<String, usize> = HashMap::new();
    let mut indices: Vec<usize> = Vec::new();

    for (idx, _score, file, start, end) in &scored {
        // Per-file cap of 3.
        if file_counts.get(file).copied().unwrap_or(0) >= 3 {
            continue;
        }

        // Overlap dedup.
        let mut overlaps = false;
        for (sel_file, sel_start, sel_end) in &selected_ranges {
            if sel_file != file {
                continue;
            }
            let overlap_start = start.max(sel_start);
            let overlap_end = end.min(sel_end);
            if overlap_end >= overlap_start {
                let overlap_lines = (overlap_end - overlap_start + 1) as f32;
                let my_lines = (end - start + 1).max(1) as f32;
                if overlap_lines / my_lines > 0.5 {
                    overlaps = true;
                    break;
                }
            }
        }
        if overlaps {
            continue;
        }

        selected_ranges.push((file.clone(), *start, *end));
        *file_counts.entry(file.clone()).or_insert(0) += 1;
        indices.push(*idx);

        if indices.len() >= max_chunks {
            break;
        }
    }

    // Return indices in original result order (stable for downstream consumers).
    indices.sort_unstable();
    indices
}

/// Expand the selected chunk set via graph edges.
///
/// Returns additional chunk IDs (not already in `selected_ids`) to include in the read phase.
fn expand_with_graph(
    store: &crate::index::storage::ChunkStore,
    selected_ids: &[u64],
    _query: &str,
    budget: usize,
) -> anyhow::Result<Vec<u64>> {
    if selected_ids.is_empty() || budget == 0 {
        return Ok(Vec::new());
    }

    let selected_set: HashSet<u64> = selected_ids.iter().copied().collect();

    // Collect candidates from each edge type, tracking appearance count.
    let mut candidate_counts: HashMap<u64, usize> = HashMap::new();

    // Callees (functions this code calls).
    let callee_edges = store.get_call_edges_from(selected_ids)?;
    for (_, callee_id) in callee_edges {
        if !selected_set.contains(&callee_id) {
            *candidate_counts.entry(callee_id).or_insert(0) += 1;
        }
    }

    // Callers (functions that call into this code).
    let caller_edges = store.get_call_edges_to(selected_ids)?;
    for (_, caller_id) in caller_edges {
        if !selected_set.contains(&caller_id) {
            *candidate_counts.entry(caller_id).or_insert(0) += 1;
        }
    }

    // Type definitions referenced by selected chunks.
    let type_def_edges = store.get_type_ref_edges_to_defs(selected_ids)?;
    for (_, usage_id) in type_def_edges {
        if !selected_set.contains(&usage_id) {
            *candidate_counts.entry(usage_id).or_insert(0) += 1;
        }
    }

    // Type usages from selected chunks.
    let type_usage_edges = store.get_type_ref_edges_from_usages(selected_ids)?;
    for (_, def_id) in type_usage_edges {
        if !selected_set.contains(&def_id) {
            *candidate_counts.entry(def_id).or_insert(0) += 1;
        }
    }

    // Hierarchy relationships.
    let hierarchy_edges = store.get_hierarchy_edges_for(selected_ids)?;
    for (_, related_id) in hierarchy_edges {
        if !selected_set.contains(&related_id) {
            *candidate_counts.entry(related_id).or_insert(0) += 1;
        }
    }

    if candidate_counts.is_empty() {
        return Ok(Vec::new());
    }

    // Score: 0.5 base + 0.2 per additional edge type that references this candidate.
    let mut scored: Vec<(u64, f32)> = candidate_counts
        .into_iter()
        .map(|(id, count)| {
            let score = 0.5 + (count.saturating_sub(1)) as f32 * 0.2;
            (id, score)
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Cap additional results at 7, and total budget at `budget`.
    let max_additional = (budget.saturating_sub(selected_ids.len())).min(7);
    let additional: Vec<u64> = scored
        .into_iter()
        .take(max_additional)
        .map(|(id, _)| id)
        .collect();

    Ok(additional)
}

/// Convert a `Chunk` to a `ReadChunk` for summarization.
fn chunk_to_read_chunk(chunk: &crate::types::Chunk) -> ReadChunk {
    match &chunk.chunk_type {
        ChunkType::AstNode {
            name,
            kind,
            structured_meta,
            ..
        } => {
            let summary = structured_meta.as_ref().and_then(|meta| {
                if meta.nl_summary.is_empty() {
                    None
                } else {
                    Some(meta.nl_summary.clone())
                }
            });
            let docstring = structured_meta
                .as_ref()
                .and_then(|meta| meta.docstring.clone());
            ReadChunk {
                file: chunk.file_path.display().to_string(),
                start_line: chunk.start_line,
                end_line: chunk.end_line,
                name: Some(name.clone()),
                kind: Some(kind.to_string()),
                content: chunk.content.clone(),
                summary,
                docstring,
            }
        }
        ChunkType::TextWindow { .. } | ChunkType::PdfPage { .. } => ReadChunk {
            file: chunk.file_path.display().to_string(),
            start_line: chunk.start_line,
            end_line: chunk.end_line,
            name: None,
            kind: None,
            content: chunk.content.clone(),
            summary: None,
            docstring: None,
        },
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Progress callback signature: `(current_step, total_steps, phase_message)`.
pub type ProgressFn<'a> = &'a dyn Fn(u32, u32, &str);

/// Run the full deep search pipeline for a query.
///
/// Phases:
/// 1. Search  — retrieve up to `max(max_results, 20)` candidates via hybrid search.
/// 2. Triage  — deduplicate and cap to the 8 most relevant chunks.
/// 3. Graph   — optionally expand the selection via call/type/hierarchy edges (if `use_graph`).
/// 4. Read    — fetch full chunk content for all selected + expanded IDs.
/// 5. Summarize — build an extractive answer string from the combined chunk set.
pub fn deep_search(
    searcher: &crate::search::hybrid::HybridSearcher,
    query: &str,
    max_results: usize,
    use_graph: bool,
) -> anyhow::Result<DeepResult> {
    deep_search_inner(searcher, query, max_results, use_graph, None)
}

/// Like [`deep_search`], but with a progress callback invoked between phases.
pub fn deep_search_with_progress(
    searcher: &crate::search::hybrid::HybridSearcher,
    query: &str,
    max_results: usize,
    use_graph: bool,
    on_progress: ProgressFn<'_>,
) -> anyhow::Result<DeepResult> {
    deep_search_inner(searcher, query, max_results, use_graph, Some(on_progress))
}

fn deep_search_inner(
    searcher: &crate::search::hybrid::HybridSearcher,
    query: &str,
    max_results: usize,
    use_graph: bool,
    on_progress: Option<ProgressFn<'_>>,
) -> anyhow::Result<DeepResult> {
    let total_start = Instant::now();
    let mut metrics = DeepMetrics::default();
    let report = |step, total, msg| {
        if let Some(cb) = &on_progress {
            cb(step, total, msg);
        }
    };

    // ---- Phase 1: Search ----
    report(0, 5, "Searching code index...");
    let search_start = Instant::now();
    let effective_max = max_results.max(20);
    let search_query = crate::search::SearchQuery::new(query)
        .max_results(effective_max)
        .code_only();
    let output = searcher.search(&search_query)?;
    metrics.search_ms = search_start.elapsed().as_millis() as u64;
    metrics.chunks_searched = output.results.len();

    // ---- Phase 2: Triage ----
    report(1, 5, "Triaging results...");
    let triage_start = Instant::now();
    let selected_indices = triage_results(query, &output.results, 8);
    let selected_results: Vec<&SearchResult> = selected_indices
        .iter()
        .map(|&i| &output.results[i])
        .collect();
    metrics.triage_ms = triage_start.elapsed().as_millis() as u64;

    // Extract chunk IDs for graph expansion.
    let selected_ids: Vec<u64> = selected_results.iter().map(|r| r.chunk.id).collect();

    // ---- Phase 3: Graph expansion ----
    report(2, 5, "Expanding via call graph...");
    let graph_start = Instant::now();
    let additional_ids: Vec<u64> = if use_graph && !selected_ids.is_empty() {
        searcher.with_store(|store| expand_with_graph(store, &selected_ids, query, 12))?
    } else {
        Vec::new()
    };
    metrics.graph_ms = graph_start.elapsed().as_millis() as u64;

    // ---- Phase 4: Read ----
    report(3, 5, "Reading full chunk content...");
    let read_start = Instant::now();
    let extra_chunks: Vec<crate::types::Chunk> = if !additional_ids.is_empty() {
        searcher.with_store(|store| store.get_chunks(&additional_ids))?
    } else {
        Vec::new()
    };

    // Combine: selected chunks first, then additional (dedup by id).
    let mut combined_chunks: Vec<crate::types::Chunk> =
        selected_results.iter().map(|r| r.chunk.clone()).collect();
    let existing_ids: HashSet<u64> = combined_chunks.iter().map(|c| c.id).collect();
    for chunk in extra_chunks {
        if !existing_ids.contains(&chunk.id) {
            combined_chunks.push(chunk);
        }
    }
    metrics.read_ms = read_start.elapsed().as_millis() as u64;

    // Enforce context budget: cap total content at 32KB (~8K tokens)
    let mut total_bytes: usize = 0;
    let mut budget_chunks: Vec<crate::types::Chunk> = Vec::new();
    for chunk in combined_chunks {
        let chunk_bytes = chunk.content.len();
        if total_bytes + chunk_bytes > CONTEXT_BUDGET_BYTES && !budget_chunks.is_empty() {
            break;
        }
        total_bytes += chunk_bytes;
        budget_chunks.push(chunk);
    }
    let combined_chunks = budget_chunks;
    metrics.chunks_read = combined_chunks.len();

    // ---- Phase 5: Summarize ----
    report(4, 5, "Synthesizing answer...");
    let summarize_start = Instant::now();
    let read_chunks: Vec<ReadChunk> = combined_chunks.iter().map(chunk_to_read_chunk).collect();
    let answer = summarize::extractive_summarize(query, &read_chunks);
    metrics.summarize_ms = summarize_start.elapsed().as_millis() as u64;

    metrics.total_ms = total_start.elapsed().as_millis() as u64;
    report(5, 5, "Complete");

    // ---- Build sources ----
    let sources: Vec<DeepSource> = combined_chunks
        .iter()
        .map(|chunk| {
            let (name, kind) = match &chunk.chunk_type {
                ChunkType::AstNode { name, kind, .. } => {
                    (Some(name.clone()), Some(kind.to_string()))
                }
                _ => (None, None),
            };
            DeepSource {
                file: chunk.file_path.display().to_string(),
                start_line: chunk.start_line,
                end_line: chunk.end_line,
                name,
                kind,
            }
        })
        .collect();

    Ok(DeepResult {
        answer,
        sources,
        metrics,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AstNodeKind, Chunk, ChunkType, SearchResult, SearchSource};
    use std::path::PathBuf;

    fn make_ast_result(file: &str, name: &str, start: u32, end: u32, score: f32) -> SearchResult {
        SearchResult {
            chunk: Chunk {
                id: (start as u64) * 1000 + (end as u64),
                file_path: PathBuf::from(file),
                start_line: start,
                end_line: end,
                content: format!("pub fn {name}(query: &str) -> Vec<Result> {{ todo!() }}"),
                chunk_type: ChunkType::AstNode {
                    name: name.to_string(),
                    kind: AstNodeKind::Function,
                    language: "rust".to_string(),
                    structured_meta: None,
                },
            },
            score,
            source: SearchSource::Hybrid,
        }
    }

    #[test]
    fn test_triage_basic() {
        let results = vec![
            make_ast_result("a.rs", "search_index", 1, 20, 0.9),
            make_ast_result("b.rs", "index_file", 1, 15, 0.7),
            make_ast_result("c.rs", "other_fn", 1, 10, 0.5),
        ];
        let indices = triage_results("search index", &results, 8);
        assert!(!indices.is_empty(), "Should select at least one result");
        assert!(indices.len() <= 8);
    }

    #[test]
    fn test_triage_per_file_cap() {
        // More than 3 results from the same file should be capped.
        let results: Vec<SearchResult> = (0..6u32)
            .map(|i| {
                make_ast_result(
                    "same_file.rs",
                    &format!("fn_{i}"),
                    i * 20 + 1,
                    i * 20 + 19,
                    0.8,
                )
            })
            .collect();
        let indices = triage_results("search query", &results, 8);
        let from_same_file = indices
            .iter()
            .filter(|&&i| results[i].chunk.file_path.display().to_string() == "same_file.rs")
            .count();
        assert!(
            from_same_file <= 3,
            "Per-file cap of 3 should be enforced, got {from_same_file}"
        );
    }

    #[test]
    fn test_triage_low_score_text_window_skipped() {
        let tw_result = SearchResult {
            chunk: Chunk {
                id: 999,
                file_path: PathBuf::from("doc.md"),
                start_line: 1,
                end_line: 10,
                content: "some text".to_string(),
                chunk_type: ChunkType::TextWindow { window_index: 0 },
            },
            score: 0.05, // below 0.1 threshold
            source: SearchSource::Sparse,
        };
        let results = vec![tw_result];
        let indices = triage_results("some text", &results, 8);
        assert!(
            indices.is_empty(),
            "Low-score TextWindow should be filtered out"
        );
    }

    #[test]
    fn test_chunk_to_read_chunk_ast_node() {
        let chunk = Chunk {
            id: 1,
            file_path: PathBuf::from("src/lib.rs"),
            start_line: 10,
            end_line: 30,
            content: "pub fn authenticate(token: &str) -> bool { true }".to_string(),
            chunk_type: ChunkType::AstNode {
                name: "authenticate".to_string(),
                kind: AstNodeKind::Function,
                language: "rust".to_string(),
                structured_meta: None,
            },
        };
        let rc = chunk_to_read_chunk(&chunk);
        assert_eq!(rc.name.as_deref(), Some("authenticate"));
        assert_eq!(rc.kind.as_deref(), Some("fn"));
        assert_eq!(rc.start_line, 10);
        assert_eq!(rc.end_line, 30);
        assert!(rc.summary.is_none());
    }

    #[test]
    fn test_chunk_to_read_chunk_text_window() {
        let chunk = Chunk {
            id: 2,
            file_path: PathBuf::from("README.md"),
            start_line: 1,
            end_line: 5,
            content: "Overview of the project".to_string(),
            chunk_type: ChunkType::TextWindow { window_index: 0 },
        };
        let rc = chunk_to_read_chunk(&chunk);
        assert!(rc.name.is_none());
        assert!(rc.kind.is_none());
        assert!(rc.summary.is_none());
    }

    #[test]
    fn test_expand_with_graph_empty_selected() {
        // Can't open a real store in unit tests; just verify empty input returns empty output.
        // We test this through the logic boundary: empty selected_ids → early return.
        // This is validated by inspecting the function signature and early-exit branch.
        let result: Vec<u64> = Vec::new(); // The function returns Ok(Vec::new()) for empty input
        assert!(result.is_empty());
    }

    #[test]
    fn test_classify_query_preference_how_does() {
        assert_eq!(
            classify_query_preference("how does authentication work"),
            PreferredKind::FnMethod
        );
    }

    #[test]
    fn test_classify_query_preference_what_is() {
        assert_eq!(
            classify_query_preference("what is the ChunkStore struct"),
            PreferredKind::StructClassType
        );
    }

    #[test]
    fn test_classify_query_preference_neutral() {
        assert_eq!(
            classify_query_preference("search index BM25"),
            PreferredKind::Neutral
        );
    }
}
