//! MMR diversity pass (S7). After rerank, before return: greedily reorder the
//! top-K results to maximise `λ·relevance − (1−λ)·max_similarity_to_selected`,
//! reducing near-duplicate clustering on exhaustive queries. O(K²), K ≤ top_k.
//! Repo-agnostic; no per-corpus tuning. Distance math is plain scalar f32 (S6
//! SIMD kernels are a drop-in optimization behind the same `cosine` signature).

use crate::types::SearchResult;
use std::collections::HashMap;

/// Cosine similarity of two equal-length vectors. Returns 0.0 for a zero-norm
/// vector (never NaN). Vectors of differing length return 0.0 (defensive — the
/// caller only ever passes embeddings from the same backend/dim).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= f32::EPSILON || nb <= f32::EPSILON {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Read `SEMANTEX_MMR_LAMBDA`. Returns `Some(λ)` for a finite value in `[0, 1]`,
/// otherwise `None` (the OFF state — MMR does not run). λ trades relevance
/// (1.0 = pure relevance, original order) against diversity (0.0 = pure
/// novelty). Spec S7 suggests ~0.7 once A/B'd; OFF by default.
pub fn mmr_lambda_from_env() -> Option<f32> {
    let v = std::env::var("SEMANTEX_MMR_LAMBDA").ok()?;
    let lambda: f32 = v.trim().parse().ok()?;
    if lambda.is_finite() && (0.0..=1.0).contains(&lambda) {
        Some(lambda)
    } else {
        None
    }
}

/// Reorder `results` in place by Maximal Marginal Relevance over the top
/// `top_k` (the tail beyond `top_k` is left untouched). Greedy O(K²):
/// repeatedly pick the candidate maximising `λ·rel − (1−λ)·max_sim_to_selected`.
///
/// `doc_vectors` maps `chunk.id` → its embedding. If ANY of the top-`top_k`
/// results lacks an embedding, MMR is skipped entirely (order unchanged) — we
/// never reorder on partial similarity information. Relevance is the current
/// `result.score` (post-rerank). This function does not change scores, only
/// order, so downstream adaptive sizing/threshold logic is unaffected.
pub fn mmr_rerank(
    results: &mut Vec<SearchResult>,
    doc_vectors: &HashMap<u64, Vec<f32>>,
    lambda: f32,
    top_k: usize,
) {
    let k = top_k.min(results.len());
    if k < 2 {
        return;
    }
    // Bail out (no-op) if any candidate in the window lacks a vector.
    if results[..k]
        .iter()
        .any(|r| !doc_vectors.contains_key(&r.chunk.id))
    {
        return;
    }

    // Work on the top-k window; keep the tail in place.
    let mut pool: Vec<SearchResult> = results.drain(..k).collect();
    let mut selected: Vec<SearchResult> = Vec::with_capacity(k);

    // Seed with the highest-relevance candidate (pool[0] — results were sorted
    // by score on entry).
    selected.push(pool.remove(0));

    while !pool.is_empty() {
        let mut best_idx = 0usize;
        let mut best_mmr = f32::NEG_INFINITY;
        for (idx, cand) in pool.iter().enumerate() {
            let cand_vec = &doc_vectors[&cand.chunk.id];
            let max_sim = selected
                .iter()
                .map(|s| cosine(cand_vec, &doc_vectors[&s.chunk.id]))
                .fold(0.0f32, f32::max);
            let mmr = lambda * cand.score - (1.0 - lambda) * max_sim;
            if mmr > best_mmr {
                best_mmr = mmr;
                best_idx = idx;
            }
        }
        selected.push(pool.remove(best_idx));
    }

    // Prepend the reordered window back ahead of the untouched tail.
    selected.append(results); // `results` now holds only the tail (drained above)
    *results = selected;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Chunk, ChunkType, Confidence, SearchResult, SearchSource};
    use std::path::PathBuf;

    fn result(id: u64, score: f32, content: &str) -> SearchResult {
        SearchResult {
            chunk: Chunk {
                id,
                file_path: PathBuf::from(format!("f{id}.rs")),
                content: content.to_string(),
                start_line: 1,
                end_line: 2,
                chunk_type: ChunkType::TextWindow { window_index: 0 },
            },
            score,
            source: SearchSource::Hybrid,
            score_dense: 0.0,
            score_sparse: 0.0,
            score_exact: 0.0,
            confidence: Confidence::Inferred,
            confidence_score: 0.0,
        }
    }

    #[test]
    fn cosine_orthogonal_is_zero_identical_is_one() {
        assert!((cosine(&[1.0, 0.0], &[0.0, 1.0])).abs() < 1e-6);
        assert!((cosine(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_zero_vector_is_zero_not_nan() {
        assert!((cosine(&[0.0, 0.0], &[1.0, 1.0])).abs() < 1e-6);
    }

    #[test]
    fn mmr_keeps_rank1_and_demotes_near_duplicate() {
        // 3 results: r1 (top), r2 nearly identical to r1, r3 distinct.
        // With low lambda (diversity-heavy), the distinct r3 should be promoted
        // above the near-duplicate r2.
        let mut results = vec![
            result(1, 1.00, "alpha"),
            result(2, 0.98, "alpha"), // near-duplicate of r1
            result(3, 0.90, "zeta"),  // distinct
        ];
        let mut vecs: HashMap<u64, Vec<f32>> = HashMap::new();
        vecs.insert(1, vec![1.0, 0.0]);
        vecs.insert(2, vec![0.99, 0.01]); // ~parallel to r1
        vecs.insert(3, vec![0.0, 1.0]); // orthogonal to r1
        mmr_rerank(&mut results, &vecs, 0.3, 10);
        assert_eq!(results[0].chunk.id, 1, "rank-1 (highest relevance) stays first");
        assert_eq!(results[1].chunk.id, 3, "distinct result promoted over near-dup");
        assert_eq!(results[2].chunk.id, 2);
    }

    #[test]
    fn mmr_lambda_one_preserves_relevance_order() {
        // lambda = 1.0 → pure relevance → original order unchanged.
        let mut results = vec![
            result(1, 1.0, "a"),
            result(2, 0.9, "b"),
            result(3, 0.8, "c"),
        ];
        let mut vecs: HashMap<u64, Vec<f32>> = HashMap::new();
        vecs.insert(1, vec![1.0, 0.0]);
        vecs.insert(2, vec![1.0, 0.0]);
        vecs.insert(3, vec![1.0, 0.0]);
        mmr_rerank(&mut results, &vecs, 1.0, 10);
        assert_eq!(
            results.iter().map(|r| r.chunk.id).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn mmr_noop_when_a_vector_is_missing() {
        // If any top-K result lacks an embedding, MMR must leave order untouched
        // (it can't compute similarity safely → skip rather than guess).
        let mut results = vec![result(1, 1.0, "a"), result(2, 0.9, "b")];
        let vecs: HashMap<u64, Vec<f32>> = HashMap::new(); // empty → missing
        mmr_rerank(&mut results, &vecs, 0.3, 10);
        assert_eq!(
            results.iter().map(|r| r.chunk.id).collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[test]
    fn mmr_lambda_from_env_parses_and_clamps() {
        // SAFETY: process-level env mutation; unique key per assertion.
        unsafe {
            std::env::set_var("SEMANTEX_MMR_LAMBDA", "0.7");
        }
        assert_eq!(mmr_lambda_from_env(), Some(0.7));
        unsafe {
            std::env::set_var("SEMANTEX_MMR_LAMBDA", "9.0");
        } // out of range
        assert_eq!(mmr_lambda_from_env(), None, "out-of-[0,1] lambda is rejected");
        unsafe {
            std::env::remove_var("SEMANTEX_MMR_LAMBDA");
        }
        assert_eq!(mmr_lambda_from_env(), None, "unset = OFF");
    }
}
