//! Named post-fusion pipeline stage: code-graph expansion.
//!
//! Wraps `graph_propagation::propagate` with the merge bookkeeping that the
//! hybrid boost chain needs: run propagation on the top seeds, pull in the
//! newly-discovered chunks, update fused scores (keep max), append new entries,
//! re-sort, and report which chunk ids were newly introduced so the caller can
//! tag them `SearchSource::GraphExpanded`.

use std::collections::{HashMap, HashSet};

use anyhow::Result;

use crate::index::storage::ChunkStore;
use crate::search::graph_propagation::{self, GraphPropagationConfig};
use crate::types::{Chunk, ScoredChunkId};

/// Run the code-graph expansion stage in place.
///
/// `fused` is the current fused ranking (sorted desc by score). The stage
/// seeds propagation with the top `fetch_count` entries, merges any newly
/// discovered chunks (fetching them into `chunk_map`), updates scores keeping
/// the max, re-sorts `fused`, and returns the set of chunk ids that were newly
/// introduced by the graph (for `GraphExpanded` tagging by the caller).
///
/// Returns an empty set with `fused` untouched when the config is disabled or
/// no new chunks are discovered.
pub fn run_graph_stage(
    fused: &mut Vec<ScoredChunkId>,
    chunk_map: &mut HashMap<u64, Chunk>,
    store: &ChunkStore,
    config: &GraphPropagationConfig,
    fetch_count: usize,
) -> Result<HashSet<u64>> {
    if config.disabled || fused.is_empty() {
        return Ok(HashSet::new());
    }

    let scored_ids: Vec<ScoredChunkId> = fused.iter().take(fetch_count).cloned().collect();
    let expanded = graph_propagation::propagate(&scored_ids, store, config)?;

    let existing_ids: HashSet<u64> = fused.iter().map(|s| s.chunk_id).collect();
    let new_ids: Vec<u64> = expanded
        .iter()
        .filter(|s| !existing_ids.contains(&s.chunk_id))
        .map(|s| s.chunk_id)
        .collect();

    if !new_ids.is_empty() {
        let new_chunks = store.get_chunks(&new_ids)?;
        for chunk in new_chunks {
            chunk_map.insert(chunk.id, chunk);
        }
    }

    // Update scores from propagation (only if higher) and add new entries.
    let prop_scores: HashMap<u64, f32> = expanded.iter().map(|s| (s.chunk_id, s.score)).collect();
    for scored in fused.iter_mut() {
        if let Some(&new_score) = prop_scores.get(&scored.chunk_id) {
            if new_score > scored.score {
                scored.score = new_score;
            }
        }
    }
    for s in expanded
        .iter()
        .filter(|s| !existing_ids.contains(&s.chunk_id))
    {
        fused.push(s.clone());
    }
    fused.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Optional centrality prior (off by default; SEMANTEX_GRAPH_CENTRALITY_WEIGHT).
    let centrality_weight = centrality_weight_from_env();
    if centrality_weight > 0.0 {
        apply_centrality_prior(fused, store, centrality_weight)?;
    }

    Ok(new_ids.into_iter().collect())
}

/// Read the centrality-prior weight from `SEMANTEX_GRAPH_CENTRALITY_WEIGHT`.
/// Default `0.0` (off). Negative or unparseable values are treated as off.
fn centrality_weight_from_env() -> f32 {
    std::env::var("SEMANTEX_GRAPH_CENTRALITY_WEIGHT")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|w| *w > 0.0)
        .unwrap_or(0.0)
}

/// Apply the stored-PageRank centrality prior to `fused` in place.
///
/// Min–max normalizes the present centrality values to `[0,1]` and adds
/// `weight * max_seed_score * normalized_centrality` to each chunk's score,
/// keeping the prior commensurate with the propagation bonuses (same
/// `max_seed_score` convention as `propagate`). Re-sorts `fused` desc.
///
/// No-op when `weight <= 0.0`, when no centrality rows exist, or when all
/// present centrality values are equal (degenerate normalization).
fn apply_centrality_prior(
    fused: &mut [ScoredChunkId],
    store: &ChunkStore,
    weight: f32,
) -> Result<()> {
    if weight <= 0.0 || fused.is_empty() {
        return Ok(());
    }
    let max_seed_score = fused
        .iter()
        .map(|s| s.score)
        .fold(f32::NEG_INFINITY, f32::max);
    if max_seed_score <= 0.0 {
        return Ok(());
    }

    let ids: Vec<u64> = fused.iter().map(|s| s.chunk_id).collect();
    let cen = store.get_centrality_scores(&ids)?;
    if cen.is_empty() {
        return Ok(());
    }

    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in cen.values() {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let span = hi - lo;
    if span <= f32::EPSILON {
        // All present centralities equal → no discriminative signal.
        return Ok(());
    }

    for scored in fused.iter_mut() {
        if let Some(&v) = cen.get(&scored.chunk_id) {
            let norm = (v - lo) / span;
            scored.score += weight * max_seed_score * norm;
        }
    }
    fused.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(())
}

#[cfg(test)]
pub(crate) mod test_support {
    use crate::index::storage::ChunkStore;
    use crate::types::{Chunk, ChunkType};
    use std::path::Path;
    use std::path::PathBuf;

    fn chunk(file: &str) -> Chunk {
        Chunk {
            id: 0,
            file_path: PathBuf::from(file),
            start_line: 1,
            end_line: 10,
            content: "fn placeholder() {}".to_string(),
            chunk_type: ChunkType::TextWindow { window_index: 0 },
        }
    }

    /// Build a tiny ChunkStore with chunks 1,2,3 where chunk 1 calls chunk 2.
    /// Uses the real ChunkStore construction + call-graph insert path the indexer
    /// uses (`ChunkStore::open` + `insert_chunk` + `store_call_graph_edge`).
    pub fn build_call_edge_store(dir: &Path) -> ChunkStore {
        let db_path = dir.join("chunks.db");
        let store = ChunkStore::open(&db_path).unwrap();
        let id1 = store.insert_chunk(&chunk("src/caller.rs"), 0x1111, 0).unwrap();
        let id2 = store.insert_chunk(&chunk("src/callee.rs"), 0x2222, 0).unwrap();
        let _id3 = store
            .insert_chunk(&chunk("src/unrelated.rs"), 0x3333, 0)
            .unwrap();
        // chunk 1 (caller) calls chunk 2 (callee), resolved.
        store
            .store_call_graph_edge(id1, "callee_fn", Some(id2))
            .unwrap();
        store
    }

    /// Seed a PageRank centrality row (creating the aux table on first call).
    pub fn add_centrality(store: &ChunkStore, chunk_id: u64, value: f64) {
        store.insert_centrality_score(chunk_id, value).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::graph_propagation::GraphPropagationConfig;
    use crate::types::ScoredChunkId;
    use std::collections::HashMap;

    // Builds a 3-chunk in-memory index: chunk 1 calls chunk 2; chunk 3 unrelated.
    fn store_with_call_edge(dir: &std::path::Path) -> ChunkStore {
        test_support::build_call_edge_store(dir)
    }

    #[test]
    fn test_stage_pulls_in_callee_and_tags_it_new() {
        let tmp = tempfile::tempdir().unwrap();
        let store = store_with_call_edge(tmp.path());

        // Seed only chunk 1 (the caller). Chunk 2 (callee) is NOT a seed.
        let mut fused = vec![ScoredChunkId::new(1, 10.0)];
        let mut chunk_map: HashMap<u64, Chunk> = HashMap::new();
        for c in store.get_chunks(&[1]).unwrap() {
            chunk_map.insert(c.id, c);
        }

        let config = GraphPropagationConfig::localization_mode(10);
        let new_ids = run_graph_stage(&mut fused, &mut chunk_map, &store, &config, 10).unwrap();

        // Callee (chunk 2) must now be present and flagged new.
        assert!(fused.iter().any(|s| s.chunk_id == 2), "callee not merged");
        assert!(
            new_ids.contains(&2),
            "callee not reported as graph-expanded"
        );
        assert!(
            chunk_map.contains_key(&2),
            "callee chunk not fetched into map"
        );
        // Seed keeps its original (higher) score and stays ranked first.
        assert_eq!(fused[0].chunk_id, 1);
        assert!((fused[0].score - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_stage_noop_when_disabled() {
        let tmp = tempfile::tempdir().unwrap();
        let store = store_with_call_edge(tmp.path());
        let mut fused = vec![ScoredChunkId::new(1, 10.0)];
        let mut chunk_map: HashMap<u64, Chunk> = HashMap::new();
        for c in store.get_chunks(&[1]).unwrap() {
            chunk_map.insert(c.id, c);
        }
        let mut config = GraphPropagationConfig::localization_mode(10);
        config.disabled = true;
        let new_ids = run_graph_stage(&mut fused, &mut chunk_map, &store, &config, 10).unwrap();
        assert!(new_ids.is_empty());
        assert_eq!(fused.len(), 1);
    }

    #[test]
    fn test_centrality_prior_weight_zero_is_byte_identical() {
        let tmp = tempfile::tempdir().unwrap();
        let store = store_with_call_edge(tmp.path());
        test_support::add_centrality(&store, 2, 0.9);
        test_support::add_centrality(&store, 1, 0.1);

        let mut fused = vec![ScoredChunkId::new(1, 10.0), ScoredChunkId::new(2, 4.0)];
        let before = fused.clone();
        // weight 0.0 → no-op, scores unchanged.
        apply_centrality_prior(&mut fused, &store, 0.0).unwrap();
        assert_eq!(fused.len(), before.len());
        for (a, b) in fused.iter().zip(before.iter()) {
            assert_eq!(a.chunk_id, b.chunk_id);
            assert!((a.score - b.score).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_centrality_prior_lifts_high_centrality_chunk() {
        let tmp = tempfile::tempdir().unwrap();
        let store = store_with_call_edge(tmp.path());
        // chunk 2 has the highest centrality; chunk 1 the lowest.
        test_support::add_centrality(&store, 2, 1.0);
        test_support::add_centrality(&store, 1, 0.0);

        let mut fused = vec![ScoredChunkId::new(1, 10.0), ScoredChunkId::new(2, 4.0)];
        let max_seed = 10.0_f32;
        let weight = 0.2_f32;
        apply_centrality_prior(&mut fused, &store, weight).unwrap();

        // min–max normalized centrality: chunk2 -> 1.0, chunk1 -> 0.0.
        // chunk2 score = 4.0 + weight * max_seed * 1.0 = 4.0 + 2.0 = 6.0.
        let s2 = fused.iter().find(|s| s.chunk_id == 2).unwrap().score;
        let s1 = fused.iter().find(|s| s.chunk_id == 1).unwrap().score;
        assert!((s2 - (4.0 + weight * max_seed)).abs() < 1e-4, "got {s2}");
        // chunk1 has normalized centrality 0 → unchanged.
        assert!((s1 - 10.0).abs() < f32::EPSILON, "got {s1}");
        // re-sorted desc: chunk1 (10.0) still first.
        assert_eq!(fused[0].chunk_id, 1);
    }
}
