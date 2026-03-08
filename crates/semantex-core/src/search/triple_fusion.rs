use crate::search::query_classifier::QueryType;
use crate::types::ScoredChunkId;
use std::collections::HashMap;
use std::sync::LazyLock;

/// Per-source weights for triple CC fusion (dense + sparse + exact).
#[derive(Debug, Clone, Copy)]
pub struct TripleFusionWeights {
    pub w_dense: f32,
    pub w_sparse: f32,
    pub w_exact: f32,
}

impl TripleFusionWeights {
    /// Theoretical max fused score: all three channels at normalized 1.0.
    pub fn max_possible(&self) -> f32 {
        self.w_dense + self.w_sparse + self.w_exact
    }
}

struct CachedWeightOverrides {
    identifier: Option<TripleFusionWeights>,
    keyword: Option<TripleFusionWeights>,
    semantic: Option<TripleFusionWeights>,
    mixed: Option<TripleFusionWeights>,
}

static WEIGHT_OVERRIDES: LazyLock<CachedWeightOverrides> = LazyLock::new(|| {
    fn parse_weights(env_key: &str) -> Option<TripleFusionWeights> {
        let val = std::env::var(env_key).ok()?;
        let parts: Vec<f32> = val
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        if parts.len() == 3 {
            Some(TripleFusionWeights {
                w_dense: parts[0],
                w_sparse: parts[1],
                w_exact: parts[2],
            })
        } else {
            None
        }
    }

    CachedWeightOverrides {
        identifier: parse_weights("SEMANTEX_WEIGHTS_IDENTIFIER"),
        keyword: parse_weights("SEMANTEX_WEIGHTS_KEYWORD"),
        semantic: parse_weights("SEMANTEX_WEIGHTS_SEMANTIC"),
        mixed: parse_weights("SEMANTEX_WEIGHTS_MIXED"),
    }
});

impl QueryType {
    /// Return recommended weights for triple CC fusion (dense + sparse + exact).
    ///
    /// Supports env var overrides for tuning: `SEMANTEX_WEIGHTS_{TYPE}=dense,sparse,exact`
    /// e.g. `SEMANTEX_WEIGHTS_SEMANTIC=0.4,0.5,1.5`
    ///
    /// Overrides are cached via LazyLock — read once at first access.
    pub fn triple_fusion_weights(self) -> TripleFusionWeights {
        let cached = match self {
            QueryType::Identifier => WEIGHT_OVERRIDES.identifier,
            QueryType::Keyword => WEIGHT_OVERRIDES.keyword,
            QueryType::Semantic => WEIGHT_OVERRIDES.semantic,
            QueryType::Mixed => WEIGHT_OVERRIDES.mixed,
        };
        if let Some(weights) = cached {
            return weights;
        }
        match self {
            QueryType::Identifier => TripleFusionWeights {
                w_dense: 0.2,
                w_sparse: 0.6,
                w_exact: 5.0,
            },
            QueryType::Keyword => TripleFusionWeights {
                w_dense: 0.3,
                w_sparse: 0.6,
                w_exact: 2.0,
            },
            QueryType::Semantic => TripleFusionWeights {
                w_dense: 0.4,
                w_sparse: 0.5,
                w_exact: 0.8,
            },
            QueryType::Mixed => TripleFusionWeights {
                w_dense: 0.5,
                w_sparse: 0.4,
                w_exact: 0.8,
            },
        }
    }
}

/// Top-score normalize a list of scored chunks to the [0, 1] range.
///
/// - Empty list → empty result
/// - Single item → score 1.0
/// - All same score → all 1.0
/// - Otherwise → `score / max`, preserving relative magnitudes
fn top_score_normalize(list: &[ScoredChunkId]) -> Vec<(u64, f32)> {
    if list.is_empty() {
        return Vec::new();
    }
    if list.len() == 1 {
        return vec![(list[0].chunk_id, 1.0)];
    }
    let max = list
        .iter()
        .map(|s| s.score)
        .fold(f32::NEG_INFINITY, f32::max);
    if max <= f32::EPSILON {
        return list.iter().map(|s| (s.chunk_id, 0.0)).collect();
    }
    list.iter().map(|s| (s.chunk_id, s.score / max)).collect()
}

/// Adjust fusion weights based on the relative confidence of each retrieval channel.
/// If one channel has a very strong top result, boost that channel's weight.
fn adapt_weights(
    weights: TripleFusionWeights,
    dense_top: f32,
    sparse_top: f32,
    exact_top: f32,
) -> TripleFusionWeights {
    let mut w = weights;

    // If exact match is very strong (near-exact symbol/string hit), boost it significantly
    if exact_top > 0.8 {
        w.w_exact *= 1.5;
    }

    // If sparse is dominant vs dense (BM25 much stronger), lean more on sparse
    if sparse_top > 0.0 && dense_top > 0.0 {
        let ratio = sparse_top / dense_top;
        if ratio > 2.5 {
            w.w_sparse *= 1.3;
            w.w_dense *= 0.8;
        } else if ratio < 0.4 {
            // Dense much stronger: lean on dense
            w.w_dense *= 1.3;
            w.w_sparse *= 0.8;
        }
    }

    w
}

/// Triple Convex Combination fusion: merge dense + sparse + exact results
/// using weighted normalized scores.
///
/// Unlike RRF which is rank-based and discards score magnitudes, CC preserves
/// score information by top-score normalizing each source to [0, 1] and then
/// combining with per-source weights.
///
/// # Parameters
/// - `dense_list`: Dense (vector) search results with cosine similarity scores
/// - `sparse_list`: Sparse (BM25) search results with BM25 scores
/// - `exact_ids`: Exact substring match chunk IDs (assigned score 1.0)
/// - `weights`: Per-source weights from the query classifier
///
/// # Returns
/// Merged list sorted by weighted CC score (highest first)
pub fn triple_cc_fuse(
    dense_list: &[ScoredChunkId],
    sparse_list: &[ScoredChunkId],
    exact_ids: &[u64],
    weights: &TripleFusionWeights,
) -> Vec<ScoredChunkId> {
    // Track per-channel normalized scores alongside total
    struct ChannelScores {
        total: f32,
        dense: f32,
        sparse: f32,
        exact: f32,
    }

    // Pre-compute normalized scores for each channel so we can inspect top values
    // before choosing the final weights.
    let dense_normalized = top_score_normalize(dense_list);
    let sparse_normalized = top_score_normalize(sparse_list);

    // Top-1 normalized score per channel (used by DAT-lite weight adaptation).
    // Dense and sparse are already normalized to [0,1] by top_score_normalize.
    // Exact is binary: 1.0 if any exact hits exist, 0.0 otherwise.
    let dense_top = dense_normalized
        .iter()
        .map(|&(_, s)| s)
        .fold(0.0_f32, f32::max);
    let sparse_top = sparse_normalized
        .iter()
        .map(|&(_, s)| s)
        .fold(0.0_f32, f32::max);
    let exact_top = if exact_ids.is_empty() { 0.0 } else { 1.0 };

    // DAT-lite: dynamically adjust weights based on per-channel confidence.
    let weights = adapt_weights(*weights, dense_top, sparse_top, exact_top);

    let mut scores: HashMap<u64, ChannelScores> = HashMap::new();

    // Accumulate dense scores (already normalized)
    for (chunk_id, norm_score) in dense_normalized {
        let entry = scores.entry(chunk_id).or_insert(ChannelScores {
            total: 0.0,
            dense: 0.0,
            sparse: 0.0,
            exact: 0.0,
        });
        entry.dense = norm_score;
        entry.total += weights.w_dense * norm_score;
    }

    // Accumulate sparse scores (already normalized)
    for (chunk_id, norm_score) in sparse_normalized {
        let entry = scores.entry(chunk_id).or_insert(ChannelScores {
            total: 0.0,
            dense: 0.0,
            sparse: 0.0,
            exact: 0.0,
        });
        entry.sparse = norm_score;
        entry.total += weights.w_sparse * norm_score;
    }

    // Exact matches: always score 1.0 (they are binary: match or no match)
    for &chunk_id in exact_ids {
        let entry = scores.entry(chunk_id).or_insert(ChannelScores {
            total: 0.0,
            dense: 0.0,
            sparse: 0.0,
            exact: 0.0,
        });
        entry.exact = 1.0;
        entry.total += weights.w_exact;
    }

    // Convert to scored chunks and sort by descending CC score
    let mut fused: Vec<ScoredChunkId> = scores
        .into_iter()
        .map(|(chunk_id, cs)| ScoredChunkId {
            chunk_id,
            score: cs.total,
            score_dense: cs.dense,
            score_sparse: cs.sparse,
            score_exact: cs.exact,
        })
        .collect();

    fused.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    fused
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(id: u64, score: f32) -> ScoredChunkId {
        ScoredChunkId::new(id, score)
    }

    // --- top_score_normalize tests ---

    #[test]
    fn test_top_score_normalize_empty() {
        let result = top_score_normalize(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_top_score_normalize_single() {
        let result = top_score_normalize(&[s(42, 0.5)]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 42);
        assert!((result[0].1 - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_top_score_normalize_all_same() {
        let list = vec![s(1, 5.0), s(2, 5.0), s(3, 5.0)];
        let result = top_score_normalize(&list);
        assert_eq!(result.len(), 3);
        for (_, norm) in &result {
            assert!((*norm - 1.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_top_score_normalize_normal() {
        let list = vec![s(1, 10.0), s(2, 5.0), s(3, 0.0)];
        let result = top_score_normalize(&list);
        assert_eq!(result.len(), 3);
        // id=1: 10/10 = 1.0
        assert_eq!(result[0].0, 1);
        assert!((result[0].1 - 1.0).abs() < f32::EPSILON);
        // id=2: 5/10 = 0.5
        assert_eq!(result[1].0, 2);
        assert!((result[1].1 - 0.5).abs() < f32::EPSILON);
        // id=3: 0/10 = 0.0
        assert_eq!(result[2].0, 3);
        assert!(result[2].1.abs() < f32::EPSILON);
    }

    // --- triple_cc_fuse tests ---

    #[test]
    fn test_triple_cc_fuse_basic() {
        let weights = TripleFusionWeights {
            w_dense: 1.0,
            w_sparse: 1.0,
            w_exact: 1.0,
        };
        // Dense: chunk 1 highest, chunk 2 lowest
        let dense = vec![s(1, 0.9), s(2, 0.1)];
        // Sparse: chunk 2 highest, chunk 1 lowest
        let sparse = vec![s(2, 10.0), s(1, 1.0)];

        let result = triple_cc_fuse(&dense, &sparse, &[], &weights);

        assert_eq!(result.len(), 2);
        // Top-score norm:
        // Chunk 1: dense=0.9/0.9=1.0, sparse=1/10=0.1 → 1.1
        // Chunk 2: dense=0.1/0.9≈0.111, sparse=10/10=1.0 → ≈1.111
        // Chunk 2 wins (sparse advantage outweighs dense advantage)
        assert_eq!(result[0].chunk_id, 2);
        assert!(result[0].score > result[1].score);
    }

    #[test]
    fn test_triple_cc_fuse_exact_boost() {
        let weights = TripleFusionWeights {
            w_dense: 0.5,
            w_sparse: 0.5,
            w_exact: 2.0,
        };

        let dense = vec![s(1, 0.9), s(2, 0.8)];
        let result = triple_cc_fuse(&dense, &[], &[2], &weights);

        // DAT-lite: dense_top=1.0, sparse_top=0.0, exact_top=1.0.
        // exact_top > 0.8 → w_exact *= 1.5 = 3.0.
        // sparse_top=0.0 → no ratio adjustment.
        // Effective weights: dense=0.5, sparse=0.5, exact=3.0.
        // Top-score norm: dense max=0.9
        // Chunk 1: dense=0.9/0.9=1.0, score = 0.5*1.0 = 0.5
        // Chunk 2: dense=0.8/0.9≈0.889, exact=1.0, score = 0.5*0.889 + 3.0*1.0 ≈ 3.444
        assert_eq!(result[0].chunk_id, 2);
        let expected_chunk2 = 0.5 * (0.8_f32 / 0.9) + 3.0;
        assert!((result[0].score - expected_chunk2).abs() < 1e-5);
        assert_eq!(result[1].chunk_id, 1);
        assert!((result[1].score - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_triple_cc_fuse_multi_source() {
        let weights = TripleFusionWeights {
            w_dense: 1.0,
            w_sparse: 1.0,
            w_exact: 1.0,
        };

        // Chunk 5 appears in all three sources
        let dense = vec![s(5, 0.9), s(10, 0.5)];
        let sparse = vec![s(5, 10.0), s(20, 5.0)];

        let result = triple_cc_fuse(&dense, &sparse, &[5], &weights);

        // DAT-lite: dense_top=1.0, sparse_top=1.0, exact_top=1.0.
        // exact_top > 0.8 → w_exact *= 1.5 = 1.5.
        // ratio = 1.0 → no dense/sparse adjustment.
        // Top-score norm:
        // Chunk 5: dense=0.9/0.9=1.0 + sparse=10/10=1.0 + exact=1.0 → 1.0*1.0 + 1.0*1.0 + 1.5*1.0 = 3.5
        // Chunk 10: dense=0.5/0.9≈0.556 → 1.0*0.556
        // Chunk 20: sparse=5/10=0.5 → 1.0*0.5
        assert_eq!(result[0].chunk_id, 5);
        assert!((result[0].score - 3.5).abs() < 1e-5);

        // Per-channel scores: chunk 5 should have all three > 0
        assert!(result[0].score_dense > 0.0, "chunk 5 should have dense > 0");
        assert!(
            result[0].score_sparse > 0.0,
            "chunk 5 should have sparse > 0"
        );
        assert!(
            (result[0].score_exact - 1.0).abs() < f32::EPSILON,
            "chunk 5 exact should be 1.0"
        );
    }

    #[test]
    fn test_triple_cc_fuse_empty_sources() {
        let weights = TripleFusionWeights {
            w_dense: 1.0,
            w_sparse: 1.0,
            w_exact: 1.0,
        };

        // All empty
        let result = triple_cc_fuse(&[], &[], &[], &weights);
        assert!(result.is_empty());

        // Only exact
        // DAT-lite: dense_top=0.0, sparse_top=0.0, exact_top=1.0.
        // exact_top > 0.8 → w_exact *= 1.5 = 1.5.
        let result = triple_cc_fuse(&[], &[], &[10, 20], &weights);
        assert_eq!(result.len(), 2);
        assert!((result[0].score - 1.5).abs() < f32::EPSILON);

        // Only dense
        let result = triple_cc_fuse(&[s(1, 0.5)], &[], &[], &weights);
        assert_eq!(result.len(), 1);
        // Single item normalizes to 1.0
        assert!((result[0].score - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_identifier_weights_exact_dominates() {
        let weights = QueryType::Identifier.triple_fusion_weights();

        // Dense finds chunk 1, sparse finds chunk 2, exact finds chunk 3
        let dense = vec![s(1, 0.95)];
        let sparse = vec![s(2, 15.0)];
        let exact = vec![3];

        let result = triple_cc_fuse(&dense, &sparse, &exact, &weights);

        // Chunk 3 (exact only): w_exact * 1.0 = 5.0
        // Chunk 1 (dense only, single item → norm 1.0): w_dense * 1.0 = 0.2
        // Chunk 2 (sparse only, single item → norm 1.0): w_sparse * 1.0 = 0.6
        assert_eq!(result[0].chunk_id, 3);
    }

    #[test]
    fn test_semantic_weights_exact_dominates() {
        let weights = QueryType::Semantic.triple_fusion_weights();

        let dense = vec![s(1, 0.95)];
        let sparse = vec![s(2, 15.0)];
        let exact = vec![3];

        let result = triple_cc_fuse(&dense, &sparse, &exact, &weights);

        // Chunk 1 (dense only, single → 1.0): 0.4 * 1.0 = 0.4
        // Chunk 2 (sparse only, single → 1.0): 0.5 * 1.0 = 0.5
        // Chunk 3 (exact only): 0.8 * 1.0 = 0.8
        assert_eq!(result[0].chunk_id, 3);
    }

    #[test]
    fn test_consensus_wins() {
        let weights = TripleFusionWeights {
            w_dense: 1.0,
            w_sparse: 1.0,
            w_exact: 1.0,
        };

        // Chunk 5 appears in dense + sparse + exact, others in only one source
        let dense = vec![s(5, 0.9), s(10, 0.7)];
        let sparse = vec![s(5, 10.0), s(20, 8.0)];
        let exact = vec![5, 30];

        let result = triple_cc_fuse(&dense, &sparse, &exact, &weights);

        // Chunk 5 gets contributions from all three → highest score
        assert_eq!(result[0].chunk_id, 5);
    }

    #[test]
    fn test_mixed_weights() {
        let weights = QueryType::Mixed.triple_fusion_weights();
        assert!((weights.w_dense - 0.5).abs() < f32::EPSILON);
        assert!((weights.w_sparse - 0.4).abs() < f32::EPSILON);
        assert!((weights.w_exact - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_keyword_weights() {
        let weights = QueryType::Keyword.triple_fusion_weights();
        assert!((weights.w_dense - 0.3).abs() < f32::EPSILON);
        assert!((weights.w_sparse - 0.6).abs() < f32::EPSILON);
        assert!((weights.w_exact - 2.0).abs() < f32::EPSILON);
    }

    // --- per-channel score tests ---

    #[test]
    fn test_per_channel_scores_preserved() {
        let weights = TripleFusionWeights {
            w_dense: 0.4,
            w_sparse: 0.5,
            w_exact: 0.8,
        };

        // Chunk 1: dense only; Chunk 2: sparse only; Chunk 3: exact only; Chunk 4: all three
        let dense = vec![s(1, 0.8), s(4, 0.6)];
        let sparse = vec![s(2, 5.0), s(4, 3.0)];
        let exact = vec![3, 4];

        let result = triple_cc_fuse(&dense, &sparse, &exact, &weights);
        let by_id: HashMap<u64, &ScoredChunkId> = result.iter().map(|r| (r.chunk_id, r)).collect();

        // Chunk 1: dense only
        let c1 = by_id[&1];
        assert!((c1.score_dense - 1.0).abs() < f32::EPSILON); // top-score normalized (single → 1.0 ... wait, there are 2 dense items)
        // Actually dense has 2 items: [0.8, 0.6], max=0.8. So chunk 1 = 0.8/0.8 = 1.0, chunk 4 = 0.6/0.8 = 0.75
        assert!((c1.score_dense - 1.0).abs() < f32::EPSILON);
        assert!(c1.score_sparse.abs() < f32::EPSILON);
        assert!(c1.score_exact.abs() < f32::EPSILON);

        // Chunk 2: sparse only
        let c2 = by_id[&2];
        assert!(c2.score_dense.abs() < f32::EPSILON);
        assert!((c2.score_sparse - 1.0).abs() < f32::EPSILON); // 5/5 = 1.0
        assert!(c2.score_exact.abs() < f32::EPSILON);

        // Chunk 3: exact only
        let c3 = by_id[&3];
        assert!(c3.score_dense.abs() < f32::EPSILON);
        assert!(c3.score_sparse.abs() < f32::EPSILON);
        assert!((c3.score_exact - 1.0).abs() < f32::EPSILON);

        // Chunk 4: all three
        let c4 = by_id[&4];
        assert!(c4.score_dense > 0.0);
        assert!(c4.score_sparse > 0.0);
        assert!((c4.score_exact - 1.0).abs() < f32::EPSILON);
    }

    // --- max_possible tests ---

    #[test]
    fn test_max_possible_identifier() {
        let mp = QueryType::Identifier.triple_fusion_weights().max_possible();
        assert!(
            (mp - 5.8).abs() < f32::EPSILON,
            "Identifier max_possible should be 5.8, got {mp}"
        );
    }

    #[test]
    fn test_max_possible_semantic() {
        let mp = QueryType::Semantic.triple_fusion_weights().max_possible();
        assert!(
            (mp - 1.7).abs() < f32::EPSILON,
            "Semantic max_possible should be 1.7, got {mp}"
        );
    }

    // --- adapt_weights tests ---

    #[test]
    fn test_adapt_weights_high_exact_boosts_exact() {
        let base = TripleFusionWeights {
            w_dense: 0.4,
            w_sparse: 0.5,
            w_exact: 0.8,
        };
        // exact_top > 0.8 → w_exact *= 1.5
        let adapted = adapt_weights(base, 0.5, 0.5, 0.9);
        assert!(
            (adapted.w_exact - 1.2).abs() < 1e-5,
            "w_exact should be boosted to 1.2, got {}",
            adapted.w_exact
        );
        // dense and sparse unchanged (ratio = 1.0)
        assert!((adapted.w_dense - 0.4).abs() < 1e-5);
        assert!((adapted.w_sparse - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_adapt_weights_low_exact_no_boost() {
        let base = TripleFusionWeights {
            w_dense: 0.4,
            w_sparse: 0.5,
            w_exact: 0.8,
        };
        // exact_top = 0.0 → no boost
        let adapted = adapt_weights(base, 0.5, 0.5, 0.0);
        assert!((adapted.w_exact - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_adapt_weights_sparse_dominant_boosts_sparse() {
        let base = TripleFusionWeights {
            w_dense: 0.4,
            w_sparse: 0.5,
            w_exact: 0.8,
        };
        // sparse_top / dense_top = 0.9 / 0.3 = 3.0 > 2.5 → boost sparse, dampen dense
        let adapted = adapt_weights(base, 0.3, 0.9, 0.0);
        assert!(
            (adapted.w_sparse - 0.5 * 1.3).abs() < 1e-5,
            "w_sparse should be boosted, got {}",
            adapted.w_sparse
        );
        assert!(
            (adapted.w_dense - 0.4 * 0.8).abs() < 1e-5,
            "w_dense should be dampened, got {}",
            adapted.w_dense
        );
    }

    #[test]
    fn test_adapt_weights_dense_dominant_boosts_dense() {
        let base = TripleFusionWeights {
            w_dense: 0.4,
            w_sparse: 0.5,
            w_exact: 0.8,
        };
        // sparse_top / dense_top = 0.1 / 0.9 ≈ 0.11 < 0.4 → boost dense, dampen sparse
        let adapted = adapt_weights(base, 0.9, 0.1, 0.0);
        assert!(
            (adapted.w_dense - 0.4 * 1.3).abs() < 1e-5,
            "w_dense should be boosted, got {}",
            adapted.w_dense
        );
        assert!(
            (adapted.w_sparse - 0.5 * 0.8).abs() < 1e-5,
            "w_sparse should be dampened, got {}",
            adapted.w_sparse
        );
    }

    #[test]
    fn test_adapt_weights_balanced_no_adjustment() {
        let base = TripleFusionWeights {
            w_dense: 0.4,
            w_sparse: 0.5,
            w_exact: 0.8,
        };
        // ratio = 0.6 / 0.6 = 1.0 → no adjustment
        let adapted = adapt_weights(base, 0.6, 0.6, 0.0);
        assert!((adapted.w_dense - 0.4).abs() < 1e-5);
        assert!((adapted.w_sparse - 0.5).abs() < 1e-5);
        assert!((adapted.w_exact - 0.8).abs() < 1e-5);
    }
}
