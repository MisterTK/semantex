use crate::search::query_classifier::QueryType;
use crate::types::{Confidence, ScoredChunkId};
use std::collections::HashMap;
use std::sync::LazyLock;

/// RRF rank-decay constant (E2). The named value in the spec — do not invent
/// a different k. The standard 60 chosen by Cormack/Clarke/Buettcher (2009) is
/// robust across query types and corpus sizes.
pub const RRF_K: f32 = 60.0;

/// Score-gap threshold for tagging a result as `Confidence::Ambiguous` (E6).
/// If `(score - next_score) / score < AMBIGUOUS_GAP_THRESHOLD` the result is
/// considered too close to its neighbour to discriminate confidently.
pub const AMBIGUOUS_GAP_THRESHOLD: f32 = 0.05;

/// Selectable fusion strategy.
///
/// RRF is the v0.3 default per spec E2. CC is preserved behind
/// `SEMANTEX_FUSION=cc` for one release; removed in v0.4.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum FusionMode {
    /// Reciprocal Rank Fusion (default, parameter-free).
    #[default]
    Rrf,
    /// Weighted RRF: per-channel rank-decay scaled by query-type FusionWeights
    /// and a configurable `k` (`config.rrf_k`). Spec S7 — revives the dead
    /// adaptive weights on the RRF path.
    WeightedRrf,
    /// Triple Convex Combination (legacy, weighted normalized scores).
    Cc,
}

impl FusionMode {
    /// Parse a fusion mode from an env-var value.
    /// Anything unrecognised (or empty) falls back to the default (RRF).
    pub fn from_env_value(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            // Weighted-RRF uses explicit spellings so the legacy "weighted"
            // (= CC) alias below stays distinct.
            "weighted-rrf" | "wrrf" => Self::WeightedRrf,
            "cc" | "convex" | "weighted" => Self::Cc,
            // Accept "rrf" or anything unrecognised → default RRF.
            _ => Self::Rrf,
        }
    }
}

static FUSION_MODE: LazyLock<FusionMode> = LazyLock::new(|| {
    std::env::var("SEMANTEX_FUSION")
        .ok()
        .map_or(FusionMode::default(), |v| FusionMode::from_env_value(&v))
});

/// Return the active fusion mode (cached via LazyLock; reads SEMANTEX_FUSION once).
pub fn active_fusion_mode() -> FusionMode {
    *FUSION_MODE
}

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

/// Parse a comma-separated weight triple from the given env var.
///
/// Finding 15 mitigation: `str::parse::<f32>` accepts "NaN", "inf", "-inf"
/// verbatim. Allowing those into fusion weights propagates NaN into result
/// scores, and the descending-score sort previously used
/// `partial_cmp(...).unwrap_or(Equal)` which violates total-order on NaN
/// pairs. Since Rust 1.81, `slice::sort_by` panics in debug builds when the
/// comparator is non-total. Filter non-finite values so the override either
/// takes a clean 3-element vector or is rejected entirely. Emits a
/// `tracing::warn!` when at least one value was rejected so a user who set
/// the env var with `NaN`/`inf` can see why their override was dropped.
fn parse_weight_override(env_key: &str) -> Option<TripleFusionWeights> {
    let val = std::env::var(env_key).ok()?;
    let raw: Vec<&str> = val.split(',').map(str::trim).collect();
    let mut parts: Vec<f32> = Vec::with_capacity(raw.len());
    let mut rejected = 0usize;
    for token in &raw {
        match token.parse::<f32>() {
            Ok(x) if x.is_finite() => parts.push(x),
            Ok(_) | Err(_) => rejected += 1,
        }
    }
    if rejected > 0 {
        tracing::warn!(
            env_var = env_key,
            value = %val,
            rejected,
            "ignoring non-finite or unparseable values in fusion weight override; \
             falling back to defaults unless exactly 3 finite values remain"
        );
    }
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

static WEIGHT_OVERRIDES: LazyLock<CachedWeightOverrides> =
    LazyLock::new(|| CachedWeightOverrides {
        identifier: parse_weight_override("SEMANTEX_WEIGHTS_IDENTIFIER"),
        keyword: parse_weight_override("SEMANTEX_WEIGHTS_KEYWORD"),
        semantic: parse_weight_override("SEMANTEX_WEIGHTS_SEMANTIC"),
        mixed: parse_weight_override("SEMANTEX_WEIGHTS_MIXED"),
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

    // Defensive against Finding 15: even though parse_weights now filters
    // non-finite env values, NaN can still slip in via channel score inputs.
    // `f32::total_cmp` is a total order on all f32 values (including NaN) and
    // never panics — strictly safer than `partial_cmp(...).unwrap_or(Equal)`.
    fused.sort_by(|a, b| b.score.total_cmp(&a.score));
    fused
}

// =====================================================================
// E2 — Reciprocal Rank Fusion (RRF)
// E4 — Exp4Fuse dual-route variant (5 channels)
// E6 — Per-result confidence labels
// =====================================================================

/// Fused result enriched with channel-agreement metadata.
///
/// `channels_hit` counts how many input channels surfaced this chunk_id.
/// `channels_fired` is the total number of channels that produced any results
/// in this fusion call (used to derive the confidence label downstream).
#[derive(Debug, Clone)]
pub struct RrfFusedResult {
    pub scored: ScoredChunkId,
    /// How many channels surfaced this chunk.
    pub channels_hit: u32,
    /// How many channels produced any results in this fusion call.
    pub channels_fired: u32,
}

impl RrfFusedResult {
    /// Derive the per-result confidence label using channel-agreement +
    /// score-gap rules (E6). The next-result score is required to determine
    /// `Ambiguous`; pass `None` for the last result in the list.
    ///
    /// Rules:
    /// - **Extracted**: all fired channels found this result (channels_hit == channels_fired,
    ///   and at least 2 channels fired so consensus is meaningful)
    /// - **Ambiguous**: score gap to next result < 5% of this score (when next score is provided)
    /// - **Inferred**: otherwise (single-channel hit, or partial multi-channel agreement
    ///   without an ambiguous gap)
    pub fn confidence(&self, next_score: Option<f32>) -> Confidence {
        let extracted = self.channels_fired >= 2 && self.channels_hit == self.channels_fired;

        // Ambiguous overrides Inferred but is overridden by Extracted —
        // a result with full channel agreement is high-confidence regardless of gap.
        if !extracted
            && let Some(next) = next_score
            && self.scored.score > 0.0
        {
            let gap = (self.scored.score - next).abs() / self.scored.score;
            if gap < AMBIGUOUS_GAP_THRESHOLD {
                return Confidence::Ambiguous;
            }
        }

        if extracted {
            Confidence::Extracted
        } else {
            Confidence::Inferred
        }
    }

    /// Numeric confidence in [0.0, 1.0] — channels-hit / channels-fired.
    /// Returns 0.0 when no channels fired (guards divide-by-zero).
    pub fn confidence_score(&self) -> f32 {
        if self.channels_fired == 0 {
            0.0
        } else {
            self.channels_hit as f32 / self.channels_fired as f32
        }
    }
}

/// Per-chunk RRF accumulator.
struct RrfAccum {
    total: f32,
    dense: f32,
    sparse: f32,
    exact: f32,
    /// Bitset of channels that surfaced this chunk (one bit per channel).
    /// Lower bits are reserved for original-query channels (dense, sparse, exact),
    /// higher bits for expanded-query channels in Exp4Fuse mode.
    channels_hit_mask: u32,
}

impl RrfAccum {
    fn new() -> Self {
        Self {
            total: 0.0,
            dense: 0.0,
            sparse: 0.0,
            exact: 0.0,
            channels_hit_mask: 0,
        }
    }
}

/// Accumulate RRF rank-decayed contributions from one channel into a score map.
/// `channel_bit` marks the channel for agreement-tracking (E6).
fn accumulate_rrf_channel(
    scores: &mut HashMap<u64, RrfAccum>,
    ranked: &[ScoredChunkId],
    channel_bit: u32,
    score_field: ChannelKind,
) {
    for (rank, item) in ranked.iter().enumerate() {
        let contribution = 1.0 / (RRF_K + rank as f32 + 1.0);
        let entry = scores.entry(item.chunk_id).or_insert_with(RrfAccum::new);
        entry.total += contribution;
        entry.channels_hit_mask |= channel_bit;
        match score_field {
            ChannelKind::Dense => entry.dense += contribution,
            ChannelKind::Sparse => entry.sparse += contribution,
        }
    }
}

/// Accumulate RRF contributions from an exact-id list (no score, rank by list order).
fn accumulate_rrf_exact(scores: &mut HashMap<u64, RrfAccum>, ids: &[u64], channel_bit: u32) {
    for (rank, &id) in ids.iter().enumerate() {
        let contribution = 1.0 / (RRF_K + rank as f32 + 1.0);
        let entry = scores.entry(id).or_insert_with(RrfAccum::new);
        entry.total += contribution;
        entry.exact += contribution;
        entry.channels_hit_mask |= channel_bit;
    }
}

/// Weighted variant of `accumulate_rrf_channel` (S7). Each rank contributes
/// `weight * 1/(k + rank + 1)` instead of the parameter-free `1/(RRF_K + rank + 1)`.
/// `weight` is the query-type per-channel weight (dense or sparse); `k` is the
/// configurable decay constant (`config.rrf_k`). Channel-agreement tracking is
/// identical to the unweighted path so E6 confidence labels are unaffected.
fn accumulate_weighted_rrf_channel(
    scores: &mut HashMap<u64, RrfAccum>,
    ranked: &[ScoredChunkId],
    channel_bit: u32,
    score_field: ChannelKind,
    weight: f32,
    k: f32,
) {
    for (rank, item) in ranked.iter().enumerate() {
        let contribution = weight * (1.0 / (k + rank as f32 + 1.0));
        let entry = scores.entry(item.chunk_id).or_insert_with(RrfAccum::new);
        entry.total += contribution;
        entry.channels_hit_mask |= channel_bit;
        match score_field {
            ChannelKind::Dense => entry.dense += contribution,
            ChannelKind::Sparse => entry.sparse += contribution,
        }
    }
}

/// Weighted variant of `accumulate_rrf_exact` (S7). The exact channel has no
/// per-channel `FusionWeights` field (those are dense/sparse only), so callers
/// pass an explicit `exact_weight` — `triple_weighted_rrf_fuse` uses `1.0`,
/// preserving the exact channel's full rank-decay contribution.
fn accumulate_weighted_rrf_exact(
    scores: &mut HashMap<u64, RrfAccum>,
    ids: &[u64],
    channel_bit: u32,
    exact_weight: f32,
    k: f32,
) {
    for (rank, &id) in ids.iter().enumerate() {
        let contribution = exact_weight * (1.0 / (k + rank as f32 + 1.0));
        let entry = scores.entry(id).or_insert_with(RrfAccum::new);
        entry.total += contribution;
        entry.exact += contribution;
        entry.channels_hit_mask |= channel_bit;
    }
}

/// Internal: which per-channel field receives a rank contribution.
/// The `Exact` channel uses a dedicated `accumulate_rrf_exact` path because
/// exact-id lists have no scores; only `Dense` and `Sparse` flow through
/// `accumulate_rrf_channel`.
#[derive(Debug, Clone, Copy)]
enum ChannelKind {
    Dense,
    Sparse,
}

/// Triple Reciprocal Rank Fusion (E2): merge dense + sparse + exact results.
///
/// RRF score: `Σ 1/(RRF_K + rank_c + 1)` across channels, where `rank_c` is the
/// 0-based rank of the chunk in channel `c`. `RRF_K = 60` is the spec-named
/// constant (Cormack/Clarke/Buettcher 2009).
///
/// Properties:
/// - **Parameter-free**: No per-channel weights. The spec requires this — RRF
///   preserves the channel-weighting concept via rank position alone.
/// - **Scale-invariant**: Mixes cosine, BM25, and binary exact scores without
///   normalisation.
/// - **Consensus-seeking**: Chunks surfaced by multiple channels rise to the top.
///
/// The returned `RrfFusedResult` carries channel-agreement counts so the caller
/// can derive per-result confidence labels (E6).
pub fn triple_rrf_fuse(
    dense_list: &[ScoredChunkId],
    sparse_list: &[ScoredChunkId],
    exact_ids: &[u64],
) -> Vec<RrfFusedResult> {
    let mut scores: HashMap<u64, RrfAccum> = HashMap::new();

    // Bits 0/1/2 represent dense/sparse/exact respectively.
    let dense_fired = !dense_list.is_empty();
    let sparse_fired = !sparse_list.is_empty();
    let exact_fired = !exact_ids.is_empty();

    if dense_fired {
        accumulate_rrf_channel(&mut scores, dense_list, 0b001, ChannelKind::Dense);
    }
    if sparse_fired {
        accumulate_rrf_channel(&mut scores, sparse_list, 0b010, ChannelKind::Sparse);
    }
    if exact_fired {
        accumulate_rrf_exact(&mut scores, exact_ids, 0b100);
    }

    let channels_fired = u32::from(dense_fired) + u32::from(sparse_fired) + u32::from(exact_fired);

    let mut fused: Vec<RrfFusedResult> = scores
        .into_iter()
        .map(|(chunk_id, acc)| {
            let channels_hit = acc.channels_hit_mask.count_ones();
            RrfFusedResult {
                scored: ScoredChunkId {
                    chunk_id,
                    score: acc.total,
                    score_dense: acc.dense,
                    score_sparse: acc.sparse,
                    score_exact: acc.exact,
                },
                channels_hit,
                channels_fired,
            }
        })
        .collect();

    // Defensive against Finding 15: use `total_cmp` so a NaN score (e.g. from a
    // pathological env-var override or upstream channel input) sorts to a stable
    // position instead of triggering a panic in Rust ≥ 1.81's sort.
    fused.sort_by(|a, b| b.scored.score.total_cmp(&a.scored.score));
    fused
}

/// Exp4Fuse dual-route RRF (E4): five channels.
///
/// Inputs:
/// - Original query: dense, sparse
/// - Expanded query: dense, sparse
/// - Exact substring matches (shared, query-text-based)
///
/// All five rank lists contribute under the same RRF formula. Channel-agreement
/// is tracked across all five for the E6 confidence label.
///
/// When either of the expanded lists is empty (e.g. the expansion produced
/// nothing for this query), those channels simply don't contribute — the
/// `channels_fired` count reflects only channels with at least one result.
///
/// Pass empty slices for any expanded channel to fall back to single-route RRF.
pub fn exp4_rrf_fuse(
    orig_dense: &[ScoredChunkId],
    orig_sparse: &[ScoredChunkId],
    exp_dense: &[ScoredChunkId],
    exp_sparse: &[ScoredChunkId],
    exact_ids: &[u64],
) -> Vec<RrfFusedResult> {
    let mut scores: HashMap<u64, RrfAccum> = HashMap::new();

    let orig_dense_active = !orig_dense.is_empty();
    let orig_sparse_active = !orig_sparse.is_empty();
    let exp_dense_active = !exp_dense.is_empty();
    let exp_sparse_active = !exp_sparse.is_empty();
    let exact_active = !exact_ids.is_empty();

    if orig_dense_active {
        accumulate_rrf_channel(&mut scores, orig_dense, 0b0_0001, ChannelKind::Dense);
    }
    if orig_sparse_active {
        accumulate_rrf_channel(&mut scores, orig_sparse, 0b0_0010, ChannelKind::Sparse);
    }
    if exp_dense_active {
        accumulate_rrf_channel(&mut scores, exp_dense, 0b0_0100, ChannelKind::Dense);
    }
    if exp_sparse_active {
        accumulate_rrf_channel(&mut scores, exp_sparse, 0b0_1000, ChannelKind::Sparse);
    }
    if exact_active {
        accumulate_rrf_exact(&mut scores, exact_ids, 0b1_0000);
    }

    let channels_fired = u32::from(orig_dense_active)
        + u32::from(orig_sparse_active)
        + u32::from(exp_dense_active)
        + u32::from(exp_sparse_active)
        + u32::from(exact_active);

    let mut fused: Vec<RrfFusedResult> = scores
        .into_iter()
        .map(|(chunk_id, acc)| {
            let channels_hit = acc.channels_hit_mask.count_ones();
            RrfFusedResult {
                scored: ScoredChunkId {
                    chunk_id,
                    score: acc.total,
                    score_dense: acc.dense,
                    score_sparse: acc.sparse,
                    score_exact: acc.exact,
                },
                channels_hit,
                channels_fired,
            }
        })
        .collect();

    // Defensive against Finding 15 (see triple_rrf_fuse).
    fused.sort_by(|a, b| b.scored.score.total_cmp(&a.scored.score));
    fused
}

/// Helper for callers: derive confidence labels for a sorted slice of
/// `RrfFusedResult`s using each result's gap to its successor.
///
/// The last result has no successor — its confidence is derived without the
/// score-gap check (cannot be `Ambiguous`).
pub fn assign_confidence(fused: &[RrfFusedResult]) -> Vec<(Confidence, f32)> {
    fused
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let next = fused.get(i + 1).map(|n| n.scored.score);
            (r.confidence(next), r.confidence_score())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(id: u64, score: f32) -> ScoredChunkId {
        ScoredChunkId::new(id, score)
    }

    // --- S7 weighted-RRF accumulation tests ---

    #[test]
    fn test_weighted_accumulate_scales_by_weight_and_k() {
        // One dense channel, weight 2.0, k=30. Rank-0 contribution must be
        // 2.0 * 1/(30+0+1) = 2/31. Rank-1 must be 2.0 * 1/(30+1+1) = 2/32.
        let mut scores: HashMap<u64, RrfAccum> = HashMap::new();
        let dense = vec![s(1, 0.9), s(2, 0.5)];
        accumulate_weighted_rrf_channel(&mut scores, &dense, 0b001, ChannelKind::Dense, 2.0, 30.0);

        let c1 = &scores[&1];
        let c2 = &scores[&2];
        assert!((c1.total - 2.0 / 31.0).abs() < 1e-6, "rank0 total = {}", c1.total);
        assert!((c1.dense - 2.0 / 31.0).abs() < 1e-6, "rank0 dense = {}", c1.dense);
        assert!((c2.total - 2.0 / 32.0).abs() < 1e-6, "rank1 total = {}", c2.total);
        assert_eq!(c1.channels_hit_mask, 0b001);
    }

    #[test]
    fn test_weighted_accumulate_exact_scales_by_weight() {
        let mut scores: HashMap<u64, RrfAccum> = HashMap::new();
        accumulate_weighted_rrf_exact(&mut scores, &[42, 7], 0b100, 3.0, 60.0);
        // rank-0 (id 42): 3.0 * 1/(60+0+1) = 3/61
        assert!((scores[&42].total - 3.0 / 61.0).abs() < 1e-6);
        assert!((scores[&42].exact - 3.0 / 61.0).abs() < 1e-6);
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

    // =================================================================
    // E2 — Triple RRF Fusion tests
    // =================================================================

    #[test]
    fn test_triple_rrf_basic() {
        // Dense ranks 1>2, sparse ranks 2>1. Chunk that appears in both
        // should be boosted by consensus.
        let dense = vec![s(1, 0.9), s(2, 0.1)];
        let sparse = vec![s(2, 10.0), s(1, 1.0)];

        let fused = triple_rrf_fuse(&dense, &sparse, &[]);

        assert_eq!(fused.len(), 2);
        // Two channels fired (no exact)
        assert_eq!(fused[0].channels_fired, 2);
        // Both chunks appear in both channels
        assert_eq!(fused[0].channels_hit, 2);
        assert_eq!(fused[1].channels_hit, 2);

        // RRF formula: chunk 1 = 1/(60+0+1) + 1/(60+1+1) = 1/61 + 1/62
        // chunk 2 = 1/(60+1+1) + 1/(60+0+1) = 1/62 + 1/61
        // They tie — but the sort is stable enough that the higher of the two should win.
        // Both should have basically equal scores.
        let expected = 1.0 / 61.0 + 1.0 / 62.0;
        assert!((fused[0].scored.score - expected).abs() < 1e-5);
        assert!((fused[1].scored.score - expected).abs() < 1e-5);
    }

    #[test]
    fn test_triple_rrf_consensus_wins() {
        // Chunk 5 appears in all three channels; others appear only in one.
        // RRF should rank chunk 5 first.
        let dense = vec![s(5, 0.9), s(10, 0.5)];
        let sparse = vec![s(5, 10.0), s(20, 5.0)];
        let exact = vec![5, 30];

        let fused = triple_rrf_fuse(&dense, &sparse, &exact);

        assert_eq!(fused[0].scored.chunk_id, 5);
        assert_eq!(fused[0].channels_hit, 3);
        assert_eq!(fused[0].channels_fired, 3);
    }

    #[test]
    fn test_triple_rrf_parameter_free_k_named_60() {
        // Validate the spec-named constant. Do not invent a different k.
        assert!((RRF_K - 60.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_triple_rrf_empty_inputs() {
        let fused = triple_rrf_fuse(&[], &[], &[]);
        assert!(fused.is_empty());
    }

    #[test]
    fn test_triple_rrf_only_exact() {
        let fused = triple_rrf_fuse(&[], &[], &[42, 7]);
        assert_eq!(fused.len(), 2);
        // 42 at rank 0, 7 at rank 1 in the exact list → 42 wins
        assert_eq!(fused[0].scored.chunk_id, 42);
        assert_eq!(fused[0].channels_fired, 1);
        assert_eq!(fused[0].channels_hit, 1);
    }

    #[test]
    fn test_triple_rrf_no_per_channel_weighting() {
        // RRF is parameter-free across query types per spec. A high-score outlier
        // in one channel must NOT dominate a chunk that ranks well in multiple
        // channels. This is the core RRF property.

        // Dense: chunk 100 ranks high (would dominate with weighted scoring)
        let dense = vec![s(100, 999.0), s(200, 0.1)];
        // Sparse: chunk 200 ranks high (cross-confirmed)
        let sparse = vec![s(200, 0.1), s(100, 0.01)];

        let fused = triple_rrf_fuse(&dense, &sparse, &[]);

        // RRF: both at rank 0+1, both in both channels, equal RRF score.
        // Critically: the 999.0 outlier doesn't dominate.
        // chunk 100 = 1/61 + 1/62
        // chunk 200 = 1/62 + 1/61
        let chunk_100 = fused.iter().find(|r| r.scored.chunk_id == 100).unwrap();
        let chunk_200 = fused.iter().find(|r| r.scored.chunk_id == 200).unwrap();
        assert!((chunk_100.scored.score - chunk_200.scored.score).abs() < 1e-6);
    }

    // =================================================================
    // E2 — RRF vs CC behavioural comparison
    // =================================================================

    #[test]
    fn test_rrf_vs_cc_perfect_agreement_both_pick_same_winner() {
        // When channels perfectly agree, RRF and CC must converge on the same
        // winner. This is the simple-case equivalence test.
        let dense = vec![s(1, 0.9), s(2, 0.5), s(3, 0.1)];
        let sparse = vec![s(1, 10.0), s(2, 5.0), s(3, 1.0)];
        let exact: Vec<u64> = vec![];

        let cc_weights = TripleFusionWeights {
            w_dense: 1.0,
            w_sparse: 1.0,
            w_exact: 1.0,
        };
        let cc = triple_cc_fuse(&dense, &sparse, &exact, &cc_weights);
        let rrf = triple_rrf_fuse(&dense, &sparse, &exact);

        // Both should pick chunk 1 as winner.
        assert_eq!(cc[0].chunk_id, 1);
        assert_eq!(rrf[0].scored.chunk_id, 1);
    }

    // =================================================================
    // E4 — Exp4Fuse dual-route RRF tests
    // =================================================================

    #[test]
    fn test_exp4_basic_five_channels() {
        // All five channels surface chunk 1, only the original-dense surfaces chunk 2.
        // chunk 1 must win + be Extracted.
        let orig_dense = vec![s(1, 0.9), s(2, 0.5)];
        let orig_sparse = vec![s(1, 10.0)];
        let exp_dense = vec![s(1, 0.8)];
        let exp_sparse = vec![s(1, 8.0)];
        let exact = vec![1u64];

        let fused = exp4_rrf_fuse(&orig_dense, &orig_sparse, &exp_dense, &exp_sparse, &exact);

        assert_eq!(fused[0].scored.chunk_id, 1);
        assert_eq!(fused[0].channels_fired, 5);
        assert_eq!(fused[0].channels_hit, 5);
    }

    #[test]
    fn test_exp4_with_no_expansion_falls_back() {
        // Empty expanded slices → equivalent to triple_rrf_fuse on the
        // original three channels (orig dense + orig sparse + exact).
        let dense = vec![s(5, 0.9)];
        let sparse = vec![s(7, 10.0)];
        let exact = vec![5u64];

        let triple = triple_rrf_fuse(&dense, &sparse, &exact);
        let exp4 = exp4_rrf_fuse(&dense, &sparse, &[], &[], &exact);

        // Same number of unique chunks
        assert_eq!(triple.len(), exp4.len());

        // Channels fired count must match (both = 3 active channels)
        assert_eq!(triple[0].channels_fired, exp4[0].channels_fired);
        assert_eq!(triple[0].channels_fired, 3);
    }

    #[test]
    fn test_exp4_dual_route_finds_more_than_single_route() {
        // A query expansion that surfaces a chunk neither the original dense
        // nor the original sparse channel found must appear in the exp4 output.
        // Without expansion, single-route would miss it.

        let orig_dense = vec![s(1, 0.9)];
        let orig_sparse = vec![s(1, 10.0)];
        // Chunk 99 only surfaces via the expanded query — synonym discovery.
        let exp_dense = vec![s(99, 0.7)];
        let exp_sparse = vec![s(99, 5.0)];
        let exact: Vec<u64> = vec![];

        let single = triple_rrf_fuse(&orig_dense, &orig_sparse, &exact);
        let dual = exp4_rrf_fuse(&orig_dense, &orig_sparse, &exp_dense, &exp_sparse, &exact);

        let single_ids: std::collections::HashSet<u64> =
            single.iter().map(|r| r.scored.chunk_id).collect();
        let dual_ids: std::collections::HashSet<u64> =
            dual.iter().map(|r| r.scored.chunk_id).collect();

        assert!(
            !single_ids.contains(&99),
            "Chunk 99 should be missing from single-route"
        );
        assert!(
            dual_ids.contains(&99),
            "Chunk 99 should be found by dual-route via expansion"
        );
    }

    // =================================================================
    // E6 — Per-result confidence label tests
    // =================================================================

    #[test]
    fn test_confidence_extracted_when_all_channels_agree() {
        // Chunk hit by all 3 channels (3 of 3 fired) → Extracted
        let dense = vec![s(1, 0.9)];
        let sparse = vec![s(1, 10.0)];
        let exact = vec![1u64];

        let fused = triple_rrf_fuse(&dense, &sparse, &exact);
        let confidence = fused[0].confidence(None);
        assert_eq!(confidence, Confidence::Extracted);
        assert!((fused[0].confidence_score() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_confidence_inferred_when_single_channel() {
        // Chunk hit by only one of two fired channels → Inferred
        let dense = vec![s(1, 0.9), s(2, 0.8)];
        let sparse = vec![s(1, 10.0)]; // only chunk 1
        // Pass a non-ambiguous next_score so we hit the Inferred branch
        let fused = triple_rrf_fuse(&dense, &sparse, &[]);

        let chunk_2 = fused.iter().find(|r| r.scored.chunk_id == 2).unwrap();
        // chunk 2 in only dense channel → 1 of 2 fired
        assert_eq!(chunk_2.channels_hit, 1);
        assert_eq!(chunk_2.channels_fired, 2);
        assert_eq!(chunk_2.confidence(None), Confidence::Inferred);
        assert!((chunk_2.confidence_score() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_confidence_ambiguous_threshold() {
        // Two results with score gap < 5% → Ambiguous label on the higher one
        // (when computed with the next-result score). The lower has no next so
        // cannot be Ambiguous.
        let r1 = RrfFusedResult {
            scored: ScoredChunkId::new(1, 1.000),
            channels_hit: 1,
            channels_fired: 2,
        };
        let r2 = RrfFusedResult {
            scored: ScoredChunkId::new(2, 0.970), // gap = 0.030 / 1.0 = 3% < 5%
            channels_hit: 1,
            channels_fired: 2,
        };
        assert_eq!(r1.confidence(Some(r2.scored.score)), Confidence::Ambiguous);

        // Wider gap: above threshold → Inferred (single channel)
        let r3 = RrfFusedResult {
            scored: ScoredChunkId::new(3, 0.900), // gap = 0.1 / 1.0 = 10% > 5%
            channels_hit: 1,
            channels_fired: 2,
        };
        assert_eq!(r1.confidence(Some(r3.scored.score)), Confidence::Inferred);
    }

    #[test]
    fn test_confidence_extracted_overrides_ambiguous() {
        // Even with a tight score gap, full channel agreement → Extracted,
        // not Ambiguous. Channel-consensus is a stronger signal.
        let extracted = RrfFusedResult {
            scored: ScoredChunkId::new(1, 1.000),
            channels_hit: 3,
            channels_fired: 3,
        };
        let _next_close = 0.999_f32;
        assert_eq!(
            extracted.confidence(Some(_next_close)),
            Confidence::Extracted
        );
    }

    #[test]
    fn test_confidence_single_channel_no_consensus_possible() {
        // Only 1 channel fired → cannot be Extracted (requires ≥2 fired channels).
        let solo = RrfFusedResult {
            scored: ScoredChunkId::new(1, 0.5),
            channels_hit: 1,
            channels_fired: 1,
        };
        assert_eq!(solo.confidence(None), Confidence::Inferred);
        assert!((solo.confidence_score() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_confidence_zero_channels_fired_safe() {
        let empty = RrfFusedResult {
            scored: ScoredChunkId::new(1, 0.0),
            channels_hit: 0,
            channels_fired: 0,
        };
        assert!(empty.confidence_score().abs() < f32::EPSILON);
    }

    #[test]
    fn test_assign_confidence_propagates_to_list() {
        // chunk 1 (all 3 channels), chunk 2 (1 of 3 channels) with wide gap → Inferred
        let dense = vec![s(1, 0.9), s(2, 0.05)];
        let sparse = vec![s(1, 10.0)];
        let exact = vec![1u64];
        let fused = triple_rrf_fuse(&dense, &sparse, &exact);

        let labels = assign_confidence(&fused);
        assert_eq!(labels.len(), fused.len());
        assert_eq!(labels[0].0, Confidence::Extracted);
        assert!((labels[0].1 - 1.0).abs() < f32::EPSILON);
        // chunk 2 has only dense channel → Inferred, score = 1/3 ≈ 0.333
        let chunk_2_pos = fused.iter().position(|r| r.scored.chunk_id == 2).unwrap();
        assert_eq!(labels[chunk_2_pos].0, Confidence::Inferred);
        assert!((labels[chunk_2_pos].1 - 1.0 / 3.0).abs() < 1e-5);
    }

    // =================================================================
    // FusionMode env-var parsing
    // =================================================================

    #[test]
    fn test_fusion_mode_default_is_rrf() {
        assert_eq!(FusionMode::default(), FusionMode::Rrf);
    }

    #[test]
    fn test_fusion_mode_parse_cc() {
        assert_eq!(FusionMode::from_env_value("cc"), FusionMode::Cc);
        assert_eq!(FusionMode::from_env_value("CC"), FusionMode::Cc);
        assert_eq!(FusionMode::from_env_value("  cc  "), FusionMode::Cc);
        assert_eq!(FusionMode::from_env_value("convex"), FusionMode::Cc);
        assert_eq!(FusionMode::from_env_value("weighted"), FusionMode::Cc);
    }

    #[test]
    fn test_fusion_mode_parse_rrf() {
        assert_eq!(FusionMode::from_env_value("rrf"), FusionMode::Rrf);
        assert_eq!(FusionMode::from_env_value("RRF"), FusionMode::Rrf);
    }

    #[test]
    fn test_fusion_mode_unknown_falls_back_to_default() {
        // Spec: anything unrecognised → default RRF
        assert_eq!(FusionMode::from_env_value(""), FusionMode::Rrf);
        assert_eq!(FusionMode::from_env_value("zzz"), FusionMode::Rrf);
    }

    #[test]
    fn test_fusion_mode_parse_weighted_rrf() {
        assert_eq!(
            FusionMode::from_env_value("weighted-rrf"),
            FusionMode::WeightedRrf
        );
        assert_eq!(FusionMode::from_env_value("wrrf"), FusionMode::WeightedRrf);
        assert_eq!(
            FusionMode::from_env_value("  Weighted-RRF  "),
            FusionMode::WeightedRrf
        );
    }

    #[test]
    fn test_fusion_mode_weighted_does_not_collide_with_cc() {
        // "weighted" historically aliased CC (convex). Keep that alias for CC;
        // weighted-RRF uses the explicit "weighted-rrf"/"wrrf" spellings so the
        // legacy SEMANTEX_FUSION=weighted users still get CC, unchanged.
        assert_eq!(FusionMode::from_env_value("weighted"), FusionMode::Cc);
    }

    // =================================================================
    // Finding 15 — NaN env weights must not panic on sort
    // =================================================================
    //
    // `str::parse::<f32>` accepts "NaN", "inf", "-inf" verbatim. If any of
    // those reach the fusion weights, they propagate into the per-chunk
    // score, and `slice::sort_by` panics in Rust ≥ 1.81 on a non-total
    // comparator. `parse_weight_override` must drop the non-finite tokens
    // and either reject the override or keep three finite values.
    //
    // Each test below uses a UNIQUE env var name so it doesn't race with
    // sibling tests or pollute the cached `WEIGHT_OVERRIDES` LazyLock used
    // by `QueryType::triple_fusion_weights()`.

    /// All three values NaN/inf → fewer than 3 finite values → None.
    #[test]
    fn test_parse_weight_override_rejects_all_non_finite() {
        let key = "SEMANTEX_WEIGHTS_FINDING15_ALL_NAN";
        // SAFETY: test process-level env mutation; key is unique to this test.
        unsafe {
            std::env::set_var(key, "NaN,inf,-inf");
        }
        let parsed = parse_weight_override(key);
        unsafe {
            std::env::remove_var(key);
        }
        assert!(
            parsed.is_none(),
            "all non-finite weights must yield None, got {parsed:?}"
        );
    }

    /// Mixed: 1 NaN + 2 finite → only 2 finite remain → None (length != 3).
    /// The override is rejected rather than silently coerced.
    #[test]
    fn test_parse_weight_override_rejects_mixed_nan() {
        let key = "SEMANTEX_WEIGHTS_FINDING15_MIXED";
        unsafe {
            std::env::set_var(key, "NaN,1.0,0.5");
        }
        let parsed = parse_weight_override(key);
        unsafe {
            std::env::remove_var(key);
        }
        assert!(
            parsed.is_none(),
            "partial-finite weights must yield None, got {parsed:?}"
        );
    }

    /// Clean input still parses correctly.
    #[test]
    fn test_parse_weight_override_accepts_three_finite() {
        let key = "SEMANTEX_WEIGHTS_FINDING15_CLEAN";
        unsafe {
            std::env::set_var(key, "0.4,0.5,1.5");
        }
        let parsed = parse_weight_override(key);
        unsafe {
            std::env::remove_var(key);
        }
        let w = parsed.expect("three finite values must parse");
        assert!((w.w_dense - 0.4).abs() < f32::EPSILON);
        assert!((w.w_sparse - 0.5).abs() < f32::EPSILON);
        assert!((w.w_exact - 1.5).abs() < f32::EPSILON);
        assert!(w.w_dense.is_finite() && w.w_sparse.is_finite() && w.w_exact.is_finite());
    }

    /// Even if NaN somehow reached the fusion weights, the sort must not
    /// panic — both `triple_cc_fuse` and `triple_rrf_fuse` now use
    /// `total_cmp` which is a total order on all f32 values including NaN.
    #[test]
    fn test_sort_does_not_panic_on_nan_scores() {
        let weights = TripleFusionWeights {
            w_dense: f32::NAN,
            w_sparse: 0.5,
            w_exact: 0.8,
        };
        let dense = vec![s(1, 0.9), s(2, 0.5)];
        let sparse = vec![s(2, 5.0), s(3, 1.0)];
        // Must not panic.
        let cc = triple_cc_fuse(&dense, &sparse, &[1], &weights);
        assert!(!cc.is_empty());
        let rrf = triple_rrf_fuse(&dense, &sparse, &[1]);
        assert!(!rrf.is_empty());
    }
}
