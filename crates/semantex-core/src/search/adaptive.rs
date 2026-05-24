/// Adaptive result sizing and confidence-based filtering.
///
/// Dynamically determines the optimal number of results based on:
/// - Query type (identifier, keyword, semantic, mixed)
/// - Score distribution (gap between top score and Nth score)
/// - Confidence thresholds (minimum score per query type)
/// - Cluster deduplication (one chunk per file unless scores are close)
use crate::search::query_classifier::QueryType;
use crate::types::SearchResult;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

/// Configuration for adaptive result sizing
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Enable adaptive result sizing
    pub enabled: bool,
    /// Minimum confidence score per query type
    pub min_score_identifier: f32,
    pub min_score_keyword: f32,
    pub min_score_semantic: f32,
    pub min_score_mixed: f32,
    /// Enable per-file deduplication
    pub deduplicate: bool,
    /// Score gap threshold for keeping multiple chunks from the same file
    pub dedup_score_gap: f32,
    /// Exhaustive mode: "find all X" queries — wider range, lower threshold, no dedup
    pub exhaustive: bool,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_score_identifier: 0.08,
            min_score_keyword: 0.15,
            min_score_semantic: 0.10,
            min_score_mixed: 0.10,
            deduplicate: true,
            dedup_score_gap: 0.10,
            exhaustive: false,
        }
    }
}

impl AdaptiveConfig {
    /// Get the minimum confidence threshold for a query type
    pub fn min_score(&self, query_type: QueryType) -> f32 {
        match query_type {
            QueryType::Identifier => self.min_score_identifier,
            QueryType::Keyword => self.min_score_keyword,
            QueryType::Semantic => self.min_score_semantic,
            QueryType::Mixed => self.min_score_mixed,
        }
    }
}

/// Result size range per query type.
/// Exhaustive mode expands the max to surface more patterns for "find all X" queries.
fn size_range(query_type: QueryType, exhaustive: bool) -> (usize, usize) {
    if exhaustive {
        return (5, 25);
    }
    match query_type {
        QueryType::Identifier | QueryType::Keyword | QueryType::Semantic => (3, 15),
        QueryType::Mixed => (5, 15),
    }
}

/// Compute the adaptive max results count based on score distribution.
///
/// Analyzes the gap between the top score and subsequent scores to find
/// a natural breakpoint. If scores drop off sharply, fewer results are
/// returned. If scores are tightly clustered, more results are returned.
///
/// Identifier queries use tight elbow detection (0.5% threshold) to improve
/// precision: symbol lookups have natural clusters between primary files
/// (definition + direct users) and secondary files (incidental importers).
pub fn adaptive_max_results(
    scores: &[f32],
    query_type: QueryType,
    requested_max: usize,
    exhaustive: bool,
) -> usize {
    if scores.is_empty() {
        return 0;
    }

    let (min_results, max_results) = size_range(query_type, exhaustive);
    // Adaptive range max is the hard cap; user's requested_max further limits output
    let range_cap = max_results;

    if scores.len() <= min_results {
        return scores.len().min(requested_max);
    }

    let top_score = scores[0];
    if top_score <= 0.0 {
        return min_results.min(scores.len()).min(requested_max);
    }

    // Significance threshold: Identifier uses a tighter threshold (0.5%) than
    // Semantic/Mixed (2%) because symbol score distributions have smaller gaps
    // between definition/primary-user clusters and incidental-importer clusters.
    let significance_threshold = if query_type == QueryType::Identifier {
        0.005
    } else {
        0.02
    };

    // Look for the biggest relative drop in scores (elbow detection)
    let mut best_gap = 0.0f32;
    let mut elbow_idx = scores.len();

    for i in 1..scores.len().min(range_cap) {
        let relative_drop = (scores[i - 1] - scores[i]) / top_score;
        if relative_drop > best_gap && i >= min_results {
            best_gap = relative_drop;
            elbow_idx = i;
        }
    }

    // If no significant gap found (all scores tightly clustered), use max range
    if best_gap < significance_threshold {
        elbow_idx = scores.len().min(max_results);
    }

    elbow_idx
        .clamp(min_results, range_cap)
        .min(scores.len())
        .min(requested_max)
}

/// Filter results below the confidence threshold for the given query type.
pub fn apply_confidence_threshold(
    results: &mut Vec<SearchResult>,
    query_type: QueryType,
    config: &AdaptiveConfig,
) {
    let threshold = config.min_score(query_type);
    // Normalize: threshold is relative to top score (RRF scores are small)
    if results.is_empty() {
        return;
    }
    let top_score = results[0].score;
    if top_score <= 0.0 {
        return;
    }

    // Compute the absolute threshold as a fraction of the top score
    let abs_threshold = top_score * threshold;
    results.retain(|r| r.score >= abs_threshold);
}

/// Deduplicate results by keeping only the top chunk per file,
/// unless multiple chunks from the same file have scores within `gap` of each other.
pub fn deduplicate_by_file(results: &mut Vec<SearchResult>, gap: f32) {
    if results.is_empty() {
        return;
    }

    // Track the best score per file (owned PathBuf keys to avoid borrow conflict)
    let mut best_score_per_file: HashMap<PathBuf, f32> = HashMap::new();
    for r in results.iter() {
        let entry = best_score_per_file
            .entry(r.chunk.file_path.clone())
            .or_insert(0.0);
        if r.score > *entry {
            *entry = r.score;
        }
    }

    // Keep a result if:
    // 1. It's the top chunk for its file, OR
    // 2. Its score is within `gap` of the best score for that file
    results.retain(|r| {
        let best = best_score_per_file
            .get(&r.chunk.file_path)
            .copied()
            .unwrap_or(0.0);
        (best - r.score) < gap
    });
}

/// Full adaptive pipeline: size, threshold, deduplicate.
///
/// Applies all three stages in order:
/// 1. Confidence threshold filtering (removes low-quality results)
/// 2. Per-file deduplication (reduces redundancy)
/// 3. Adaptive sizing (determines optimal file count using per-file best scores)
///
/// For Stage 3, elbow detection operates on per-file best scores rather than
/// raw chunk scores. This prevents multiple chunks from the same file from
/// diluting the score gap between result clusters (which would cause elbow
/// detection to miss natural breakpoints).
pub fn apply_adaptive_pipeline(
    results: &mut Vec<SearchResult>,
    query_type: QueryType,
    requested_max: usize,
    config: &AdaptiveConfig,
) {
    let exhaustive = config.exhaustive;

    if !config.enabled || results.is_empty() {
        results.truncate(requested_max);
        return;
    }

    // Stage 1: Confidence threshold.
    // Exhaustive mode halves the threshold to include more borderline matches.
    if exhaustive {
        let threshold = config.min_score(query_type) * 0.5;
        if let Some(top) = results.first().map(|r| r.score)
            && top > 0.0
        {
            results.retain(|r| r.score >= top * threshold);
        }
    } else {
        apply_confidence_threshold(results, query_type, config);
    }

    // Stage 2: Deduplication.
    // Exhaustive mode skips per-file dedup so multiple chunks from different
    // sections of the same file can surface distinct patterns.
    if config.deduplicate && !exhaustive {
        deduplicate_by_file(results, config.dedup_score_gap);
    }

    // Stage 3: Adaptive sizing.
    //
    // For Identifier queries, attempt per-file elbow detection first. Multiple
    // chunks per file (kept by dedup when they're within the gap) insert
    // intermediate scores that dilute the gap between primary-user and
    // incidental-importer clusters, preventing elbow detection from firing at
    // the right boundary. Per-file scores are cleaner for elbow detection.
    //
    // Two cases for Identifier:
    //   a) Elbow fires (or range cap limits): filter to top-N files. This gives
    //      high precision for specific symbols (e.g. ConnectionServiceFactory).
    //   b) No elbow, all files returned: fall back to chunk-level truncation at
    //      range_cap. This matches v7 behavior for generic identifiers that
    //      appear in many files with similar scores (e.g. junctionUserId).
    //
    // For all other query types, use raw chunk scores (original behavior).
    if query_type == QueryType::Identifier && !exhaustive {
        let (_, max_range) = size_range(query_type, false);
        let file_best: Vec<(PathBuf, f32)> = {
            let mut map: HashMap<PathBuf, f32> = HashMap::new();
            for r in &*results {
                let e = map.entry(r.chunk.file_path.clone()).or_insert(0.0_f32);
                if r.score > *e {
                    *e = r.score;
                }
            }
            let mut v: Vec<(PathBuf, f32)> = map.into_iter().collect();
            v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            v
        };

        let total_files = file_best.len();
        let file_scores: Vec<f32> = file_best.iter().map(|(_, s)| *s).collect();
        let adaptive_file_count =
            adaptive_max_results(&file_scores, query_type, requested_max, false);

        // Apply precision mode only when the elbow selects ≤ 2/3 of total files.
        // Selecting > 2/3 of files means the "elbow" is an end-of-list artifact
        // (e.g., the last 1-2 files happen to score slightly lower) rather than
        // a genuine cluster boundary. In that case, fall back to the range cap.
        //
        // Example: Q1 (ConnectionServiceFactory) — 3 files out of 13 (23%) → precision mode.
        // Example: Q3 (junctionUserId) — 12 files out of 13 (92%) → recall mode.
        let is_meaningful_elbow =
            adaptive_file_count < total_files && adaptive_file_count * 3 <= total_files * 2;

        if is_meaningful_elbow {
            // Case (a): meaningful cluster boundary — filter to top-N files for precision.
            // Also cap chunks at range_cap to avoid returning too many chunks from
            // files with multiple similar-scored sections (dedup gap keeps them all).
            let top_files: HashSet<PathBuf> = file_best
                .into_iter()
                .take(adaptive_file_count)
                .map(|(path, _)| path)
                .collect();
            results.retain(|r| top_files.contains(&r.chunk.file_path));
            results.truncate(max_range.min(requested_max));
        } else {
            // Case (b): no meaningful elbow (all files returned, or near-all).
            // Fall back to chunk-level range cap — matches v7 baseline behavior.
            results.truncate(max_range.min(requested_max));
        }
    } else {
        let effective_max = if exhaustive {
            requested_max.max(25)
        } else {
            requested_max
        };
        let scores: Vec<f32> = results.iter().map(|r| r.score).collect();
        let adaptive_count = adaptive_max_results(&scores, query_type, effective_max, exhaustive);
        results.truncate(adaptive_count);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Chunk, ChunkType, SearchSource};
    use std::path::PathBuf;

    fn make_result(id: u64, score: f32, file: &str) -> SearchResult {
        SearchResult {
            chunk: Chunk {
                id,
                file_path: PathBuf::from(file),
                start_line: 1,
                end_line: 10,
                content: format!("chunk {id}"),
                chunk_type: ChunkType::TextWindow { window_index: 0 },
            },
            score,
            source: SearchSource::Hybrid,
            score_dense: 0.0,
            score_sparse: 0.0,
            score_exact: 0.0,
            confidence: crate::types::Confidence::Inferred,
            confidence_score: 0.0,
        }
    }

    // --- adaptive_max_results ---

    #[test]
    fn test_adaptive_empty_scores() {
        assert_eq!(adaptive_max_results(&[], QueryType::Semantic, 10, false), 0);
    }

    #[test]
    fn test_adaptive_fewer_than_min() {
        // Semantic min is 3, but only 2 results
        let scores = vec![0.5, 0.4];
        assert_eq!(
            adaptive_max_results(&scores, QueryType::Semantic, 10, false),
            2
        );
    }

    #[test]
    fn test_adaptive_tight_cluster_uses_max_range() {
        // All scores very similar -> should use max for semantic (15)
        let scores: Vec<f32> = (0..30).map(|i| 1.0 - i as f32 * 0.001).collect();
        let result = adaptive_max_results(&scores, QueryType::Semantic, 50, false);
        assert_eq!(result, 15); // max for Semantic
    }

    #[test]
    fn test_adaptive_sharp_drop_finds_elbow() {
        // Clear elbow at position 3 for Keyword type (min=3)
        let scores = vec![0.5, 0.48, 0.47, 0.1, 0.09, 0.08, 0.07, 0.06];
        // The drop from 0.47 to 0.1 is (0.37/0.5) = 0.74, huge gap at index 3
        let result = adaptive_max_results(&scores, QueryType::Keyword, 10, false);
        assert_eq!(result, 3);
    }

    #[test]
    fn test_adaptive_identifier_tight_elbow() {
        // Identifier queries use tight elbow (0.5% threshold) for precision.
        // Scores: big drop at i=3 (0.47→0.1 = 74% relative), which is well above 0.5%.
        // With min_results=3, elbow fires at i=3 → returns 3 results.
        let scores = vec![0.5, 0.48, 0.47, 0.1, 0.09, 0.08, 0.07, 0.06];
        let result = adaptive_max_results(&scores, QueryType::Identifier, 50, false);
        assert_eq!(result, 3);
    }

    #[test]
    fn test_adaptive_respects_requested_max() {
        let scores: Vec<f32> = (0..30).map(|i| 1.0 - i as f32 * 0.01).collect();
        // Mixed range is 5-15, requested_max=5 limits output to 5
        let result = adaptive_max_results(&scores, QueryType::Mixed, 5, false);
        assert!(result <= 5);
    }

    #[test]
    fn test_adaptive_identifier_range() {
        // 15 tightly clustered scores
        let scores: Vec<f32> = (0..15).map(|i| 1.0 - i as f32 * 0.002).collect();
        let result = adaptive_max_results(&scores, QueryType::Identifier, 50, false);
        assert!(result >= 3); // min
        assert!(result <= 15); // max for Identifier (no elbow, returns up to max)
    }

    // --- apply_confidence_threshold ---

    #[test]
    fn test_confidence_threshold_filters_low_scores() {
        let mut results = vec![
            make_result(1, 1.0, "a.rs"),
            make_result(2, 0.5, "b.rs"),
            make_result(3, 0.2, "c.rs"),
            make_result(4, 0.05, "d.rs"),
        ];
        let config = AdaptiveConfig::default();
        // Identifier threshold = 0.08, so abs threshold = 1.0 * 0.08 = 0.08
        apply_confidence_threshold(&mut results, QueryType::Identifier, &config);

        assert_eq!(results.len(), 3); // 1.0, 0.5, 0.2 pass; 0.05 < 0.08 fails
        assert_eq!(results[0].chunk.id, 1);
        assert_eq!(results[1].chunk.id, 2);
        assert_eq!(results[2].chunk.id, 3);
    }

    #[test]
    fn test_confidence_threshold_semantic_more_lenient() {
        let mut results = vec![
            make_result(1, 1.0, "a.rs"),
            make_result(2, 0.5, "b.rs"),
            make_result(3, 0.2, "c.rs"),
            make_result(4, 0.05, "d.rs"),
        ];
        let config = AdaptiveConfig::default();
        // Semantic threshold = 0.10, abs = 1.0 * 0.10 = 0.10
        apply_confidence_threshold(&mut results, QueryType::Semantic, &config);

        assert_eq!(results.len(), 3); // 0.05 filtered out
    }

    #[test]
    fn test_confidence_threshold_empty_results() {
        let mut results: Vec<SearchResult> = Vec::new();
        let config = AdaptiveConfig::default();
        apply_confidence_threshold(&mut results, QueryType::Mixed, &config);
        assert!(results.is_empty());
    }

    #[test]
    fn test_confidence_threshold_all_pass() {
        let mut results = vec![make_result(1, 1.0, "a.rs"), make_result(2, 0.9, "b.rs")];
        let config = AdaptiveConfig::default();
        // All above 0.15 threshold relative to 1.0
        apply_confidence_threshold(&mut results, QueryType::Identifier, &config);
        assert_eq!(results.len(), 2);
    }

    // --- deduplicate_by_file ---

    #[test]
    fn test_dedup_keeps_top_per_file() {
        let mut results = vec![
            make_result(1, 1.0, "a.rs"),
            make_result(2, 0.8, "a.rs"), // same file, score gap = 0.2 > 0.05
            make_result(3, 0.7, "b.rs"),
        ];
        deduplicate_by_file(&mut results, 0.05);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].chunk.id, 1); // top chunk from a.rs
        assert_eq!(results[1].chunk.id, 3); // only chunk from b.rs
    }

    #[test]
    fn test_dedup_keeps_close_scores() {
        let mut results = vec![
            make_result(1, 1.0, "a.rs"),
            make_result(2, 0.97, "a.rs"), // gap = 0.03 < 0.05, keep both
            make_result(3, 0.7, "b.rs"),
        ];
        deduplicate_by_file(&mut results, 0.05);

        assert_eq!(results.len(), 3); // all kept
    }

    #[test]
    fn test_dedup_empty() {
        let mut results: Vec<SearchResult> = Vec::new();
        deduplicate_by_file(&mut results, 0.05);
        assert!(results.is_empty());
    }

    #[test]
    fn test_dedup_single_file_many_chunks() {
        let mut results = vec![
            make_result(1, 1.0, "a.rs"),
            make_result(2, 0.99, "a.rs"), // gap 0.01, keep
            make_result(3, 0.5, "a.rs"),  // gap 0.5, drop
            make_result(4, 0.3, "a.rs"),  // gap 0.7, drop
        ];
        deduplicate_by_file(&mut results, 0.05);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].chunk.id, 1);
        assert_eq!(results[1].chunk.id, 2);
    }

    // --- full pipeline ---

    #[test]
    fn test_pipeline_identifier_no_elbow_uses_range_cap() {
        // When all files score similarly (no elbow), Identifier falls back to
        // chunk-level range cap (15) rather than returning all chunks from all files.
        // Simulates junctionUserId: many files, all with similar scores.
        let mut results = vec![
            make_result(1, 5.8, "a.ts"),
            make_result(2, 5.75, "b.ts"),
            make_result(3, 5.73, "c.ts"),
            make_result(4, 5.72, "d.ts"),
            make_result(5, 5.71, "e.ts"),
            make_result(6, 5.70, "f.ts"),
            make_result(7, 5.69, "g.ts"),
            make_result(8, 5.68, "h.ts"),
            make_result(9, 5.67, "i.ts"),
            make_result(10, 5.66, "j.ts"),
            make_result(11, 5.65, "k.ts"),
            make_result(12, 5.64, "l.ts"),
            make_result(13, 5.63, "m.ts"),
            make_result(14, 5.62, "n.ts"),
            make_result(15, 5.61, "o.ts"),
            make_result(16, 5.60, "p.ts"), // 16th unique file
            make_result(17, 5.59, "q.ts"), // 17th unique file
        ];
        let config = AdaptiveConfig::default();
        apply_adaptive_pipeline(&mut results, QueryType::Identifier, 50, &config);

        // No elbow fires (all scores within ~3.5% of top), so range cap (15) applies.
        assert!(
            results.len() <= 15,
            "should be capped at range max (15), got {}",
            results.len()
        );
        assert!(results.len() >= 3, "should have at least min results");
    }

    #[test]
    fn test_pipeline_identifier_elbow_fires_precision_mode() {
        // When a clear elbow exists at ≤ 2/3 of total files, Identifier uses precision mode.
        // Simulates ConnectionServiceFactory: few primary files, many incidental files.
        // 8 total files, elbow at 3 (37.5% of files ≤ 66.7%) → precision mode.
        let mut results = vec![
            make_result(1, 5.80, "primary.ts"), // top file (definition)
            make_result(2, 5.45, "user_a.ts"),  // direct user
            make_result(3, 5.40, "user_b.ts"),  // direct user
            // Clear elbow here (~6% drop from 5.40 to 5.08)
            make_result(4, 5.08, "incidental_a.ts"), // incidental importer
            make_result(5, 5.07, "incidental_b.ts"),
            make_result(6, 5.06, "incidental_c.ts"),
            make_result(7, 5.05, "incidental_d.ts"),
            make_result(8, 5.04, "incidental_e.ts"),
        ];
        let config = AdaptiveConfig::default();
        apply_adaptive_pipeline(&mut results, QueryType::Identifier, 50, &config);

        // Elbow at file index 3 (5.40 → 5.08 = 5.5% drop > 0.5% threshold).
        // 3 <= 8*2/3=5.33 → is_meaningful_elbow = true → returns only top-3 files.
        assert_eq!(results.len(), 3, "elbow should fire at 3 files");
    }

    #[test]
    fn test_pipeline_identifier_end_of_list_elbow_falls_back() {
        // When the biggest gap is near the end of the list (>2/3 of files selected),
        // fall back to recall mode (range cap) rather than near-all precision.
        // 13 files, elbow fires at 12 (92% > 66.7%) → recall mode.
        let mut results: Vec<SearchResult> = (0..12u64)
            .map(|i| make_result(i, 5.8 - i as f32 * 0.03, &format!("f{i}.ts")))
            .chain(std::iter::once(make_result(12, 4.5, "last.ts"))) // big drop at end
            .collect();
        let config = AdaptiveConfig::default();
        apply_adaptive_pipeline(&mut results, QueryType::Identifier, 50, &config);

        // Elbow fires at file 12/13 → not meaningful (> 2/3) → recall mode → range_cap=15
        assert!(
            results.len() <= 15,
            "should fall back to range cap (15), got {}",
            results.len()
        );
        assert!(
            results.len() > 5,
            "should have multiple results in recall mode"
        );
    }

    #[test]
    fn test_pipeline_disabled() {
        let mut results = vec![
            make_result(1, 1.0, "a.rs"),
            make_result(2, 0.5, "b.rs"),
            make_result(3, 0.01, "c.rs"),
        ];
        let config = AdaptiveConfig {
            enabled: false,
            ..Default::default()
        };
        apply_adaptive_pipeline(&mut results, QueryType::Identifier, 2, &config);
        assert_eq!(results.len(), 2); // just truncate to requested_max
    }

    #[test]
    fn test_pipeline_full() {
        let mut results = vec![
            make_result(1, 1.0, "a.rs"),
            make_result(2, 0.98, "a.rs"), // close score, kept by dedup
            make_result(3, 0.7, "b.rs"),
            make_result(4, 0.5, "c.rs"),
            make_result(5, 0.4, "d.rs"),
            make_result(6, 0.35, "e.rs"),
            make_result(7, 0.05, "f.rs"), // below identifier threshold (0.08 * 1.0 = 0.08)
        ];
        let config = AdaptiveConfig::default();
        apply_adaptive_pipeline(&mut results, QueryType::Identifier, 10, &config);

        // After threshold: chunks 1-6 survive (0.05 < 0.08 abs threshold)
        // After dedup: chunk 2 gap from best (1.0) is 0.02 < 0.10, kept
        // After adaptive sizing: depends on score distribution
        assert!(results.len() >= 3); // min for Identifier
        assert!(results.len() <= 10);
        // chunk 7 should be gone
        assert!(results.iter().all(|r| r.chunk.id != 7));
    }

    #[test]
    fn test_pipeline_semantic_query() {
        let mut results = vec![
            make_result(1, 1.0, "a.rs"),
            make_result(2, 0.9, "b.rs"),
            make_result(3, 0.8, "c.rs"),
            make_result(4, 0.05, "d.rs"), // below semantic threshold (0.10 * 1.0 = 0.10)
        ];
        let config = AdaptiveConfig::default();
        apply_adaptive_pipeline(&mut results, QueryType::Semantic, 10, &config);

        assert_eq!(results.len(), 3); // min for Semantic is 3, but only 3 pass threshold
        assert!(results.iter().all(|r| r.chunk.id != 4));
    }

    #[test]
    fn test_pipeline_dedup_disabled() {
        let mut results = vec![
            make_result(1, 1.0, "a.rs"),
            make_result(2, 0.5, "a.rs"), // same file, would be deduped
            make_result(3, 0.4, "b.rs"),
        ];
        let config = AdaptiveConfig {
            deduplicate: false,
            ..Default::default()
        };
        // Semantic threshold = 0.10, all pass (0.4 > 0.10)
        apply_adaptive_pipeline(&mut results, QueryType::Semantic, 10, &config);
        assert_eq!(results.len(), 3); // no dedup, all pass threshold
    }

    // --- config tests ---

    #[test]
    fn test_config_defaults() {
        let config = AdaptiveConfig::default();
        assert!(config.enabled);
        assert!(config.deduplicate);
        assert!((config.min_score_identifier - 0.08).abs() < f32::EPSILON);
        assert!((config.min_score_keyword - 0.15).abs() < f32::EPSILON);
        assert!((config.min_score_semantic - 0.10).abs() < f32::EPSILON);
        assert!((config.min_score_mixed - 0.10).abs() < f32::EPSILON);
        assert!((config.dedup_score_gap - 0.10).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_min_score_by_type() {
        let config = AdaptiveConfig::default();
        assert!((config.min_score(QueryType::Identifier) - 0.08).abs() < f32::EPSILON);
        assert!((config.min_score(QueryType::Keyword) - 0.15).abs() < f32::EPSILON);
        assert!((config.min_score(QueryType::Semantic) - 0.10).abs() < f32::EPSILON);
        assert!((config.min_score(QueryType::Mixed) - 0.10).abs() < f32::EPSILON);
    }

    // --- size_range tests ---

    #[test]
    fn test_size_ranges() {
        assert_eq!(size_range(QueryType::Identifier, false), (3, 15));
        assert_eq!(size_range(QueryType::Keyword, false), (3, 15));
        assert_eq!(size_range(QueryType::Semantic, false), (3, 15));
        assert_eq!(size_range(QueryType::Mixed, false), (5, 15));
        // Exhaustive mode expands range for all types
        assert_eq!(size_range(QueryType::Semantic, true), (5, 25));
        assert_eq!(size_range(QueryType::Mixed, true), (5, 25));
    }
}
