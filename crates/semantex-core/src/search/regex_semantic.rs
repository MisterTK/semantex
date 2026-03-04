use crate::types::ScoredChunkId;
use regex::Regex;
use std::collections::HashMap;

/// Strip regex metacharacters to extract semantic-meaningful tokens.
///
/// Converts a regex pattern into plain words suitable for semantic search.
///
/// # Examples
///
/// - `"fn\\s+\\w+"` becomes `"fn"`
/// - `"Promise\\.allSettled"` becomes `"Promise allSettled"`
pub fn strip_regex_for_semantic(pattern: &str) -> String {
    // Remove common regex metacharacters and character classes
    let stripped = pattern
        .replace("\\s+", " ")
        .replace("\\w+", "")
        .replace("\\d+", "")
        .replace("\\b", "")
        .replace(".*", " ")
        .replace(".+", " ")
        .replace("\\.", " ");

    // Remove remaining metacharacters
    let cleaned: String = stripped
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace() || *c == '_')
        .collect();

    // Collapse whitespace and trim
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Merge a semantic query with tokens extracted from a regex pattern.
///
/// Deduplicates tokens to avoid BM25 over-weighting. Tokens from the regex
/// pattern that already appear in the semantic query (case-insensitive) are
/// not added again.
pub fn merge_query_with_pattern(semantic_query: &str, regex_pattern: &str) -> String {
    let pattern_tokens = strip_regex_for_semantic(regex_pattern);
    let query_lower = semantic_query.to_lowercase();

    let mut tokens: Vec<&str> = semantic_query.split_whitespace().collect();
    for token in pattern_tokens.split_whitespace() {
        if !query_lower.contains(&token.to_lowercase()) {
            tokens.push(token);
        }
    }
    tokens.join(" ")
}

/// Merge regex-matched results with semantic results.
///
/// For each unique chunk, takes the **higher** weighted score (not the sum).
/// Results are returned sorted by descending score.
pub fn merge_regex_semantic(
    regex_results: &[ScoredChunkId],
    semantic_results: &[ScoredChunkId],
    w_regex: f32,
    w_semantic: f32,
) -> Vec<ScoredChunkId> {
    let mut scores: HashMap<u64, f32> = HashMap::new();

    for s in regex_results {
        let entry = scores.entry(s.chunk_id).or_insert(0.0);
        *entry = entry.max(w_regex * s.score);
    }
    for s in semantic_results {
        let entry = scores.entry(s.chunk_id).or_insert(0.0);
        *entry = entry.max(w_semantic * s.score);
    }

    let mut merged: Vec<ScoredChunkId> = scores
        .into_iter()
        .map(|(chunk_id, score)| ScoredChunkId::new(chunk_id, score))
        .collect();
    merged.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    merged
}

/// Compile a user-provided regex pattern, returning `None` on invalid regex.
pub fn compile_pattern(pattern: &str) -> Option<Regex> {
    Regex::new(pattern).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_basic_regex() {
        assert_eq!(strip_regex_for_semantic(r"fn\s+\w+"), "fn");
    }

    #[test]
    fn strip_escaped_dot() {
        assert_eq!(
            strip_regex_for_semantic(r"Promise\.allSettled"),
            "Promise allSettled"
        );
    }

    #[test]
    fn strip_complex_pattern() {
        assert_eq!(strip_regex_for_semantic(r"async\s+fn\s+\w+"), "async fn");
    }

    #[test]
    fn merge_query_deduplicates() {
        let merged = merge_query_with_pattern("find Promise usage", r"Promise\.allSettled");
        // "Promise" already in query, should not be duplicated
        let promise_count = merged.matches("Promise").count();
        assert_eq!(promise_count, 1);
        assert!(merged.contains("allSettled"));
    }

    #[test]
    fn merge_results_uses_max_score() {
        let regex = vec![ScoredChunkId::new(1, 0.8), ScoredChunkId::new(2, 0.5)];
        let semantic = vec![ScoredChunkId::new(1, 0.6), ScoredChunkId::new(3, 0.9)];

        let merged = merge_regex_semantic(&regex, &semantic, 1.0, 1.0);
        assert_eq!(merged.len(), 3);

        // Chunk 1 appears in both: max(0.8, 0.6) = 0.8
        let chunk1 = merged.iter().find(|s| s.chunk_id == 1).expect("chunk 1");
        assert!((chunk1.score - 0.8).abs() < f32::EPSILON);

        // Results sorted by descending score
        assert!(merged[0].score >= merged[1].score);
        assert!(merged[1].score >= merged[2].score);
    }

    #[test]
    fn merge_results_with_weights() {
        let regex = vec![ScoredChunkId::new(1, 1.0)];
        let semantic = vec![ScoredChunkId::new(1, 1.0)];

        let merged = merge_regex_semantic(&regex, &semantic, 0.5, 0.8);
        let chunk1 = merged.iter().find(|s| s.chunk_id == 1).expect("chunk 1");
        // max(0.5 * 1.0, 0.8 * 1.0) = 0.8
        assert!((chunk1.score - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn compile_valid_pattern() {
        assert!(compile_pattern(r"fn\s+\w+").is_some());
    }

    #[test]
    fn compile_invalid_pattern() {
        assert!(compile_pattern(r"[invalid").is_none());
    }
}
