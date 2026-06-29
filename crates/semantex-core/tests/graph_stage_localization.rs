//! End-to-end check that the code-graph stage surfaces structurally-related
//! chunks on a real index. Skips when no `.semantex` index is present so unit
//! CI stays hermetic.

use std::collections::HashMap;
use std::path::Path;

use semantex_core::index::storage::ChunkStore;
use semantex_core::search::graph_propagation::GraphPropagationConfig;
use semantex_core::search::graph_stage::run_graph_stage;
use semantex_core::types::{Chunk, ScoredChunkId};

/// Locate this repo's index db. The semantex workspace index lives at
/// `<repo>/.semantex/chunks.db`. `CARGO_MANIFEST_DIR` = crates/semantex-core; the
/// workspace root is two levels up.
fn open_repo_store() -> Option<ChunkStore> {
    let db = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../.semantex")
        .join("chunks.db");
    if !db.exists() {
        return None;
    }
    ChunkStore::open_for_search(&db).ok()
}

#[test]
fn graph_stage_expands_real_index_seeds() {
    let Some(store) = open_repo_store() else {
        eprintln!("skipping: no .semantex index present");
        return;
    };

    // Seed with a real chunk id that has outgoing call edges.
    let all_ids = store.get_all_chunk_ids().expect("chunk ids");
    let Some(seed_id) = all_ids.iter().copied().find(|id| {
        store
            .get_call_edges_from(&[*id])
            .map(|e| !e.is_empty())
            .unwrap_or(false)
    }) else {
        eprintln!("skipping: no chunk with outgoing call edges in the index");
        return;
    };

    let mut fused = vec![ScoredChunkId::new(seed_id, 10.0)];
    let mut chunk_map: HashMap<u64, Chunk> = HashMap::new();
    for c in store.get_chunks(&[seed_id]).unwrap() {
        chunk_map.insert(c.id, c);
    }

    let config = GraphPropagationConfig::localization_mode(20);
    let before = fused.len();
    let new_ids = run_graph_stage(&mut fused, &mut chunk_map, &store, &config, 20).unwrap();

    assert!(
        !new_ids.is_empty(),
        "localization stage discovered no related chunks"
    );
    assert!(fused.len() > before, "fused list did not grow");
    // Every newly-introduced id has a fetched chunk (for GraphExpanded tagging).
    for id in &new_ids {
        assert!(
            chunk_map.contains_key(id),
            "new id {id} missing from chunk_map"
        );
    }
    // Seed retains a presence in the ranking.
    assert!(fused.iter().any(|s| s.chunk_id == seed_id));
}

#[test]
fn graph_stage_disabled_is_noop_on_real_index() {
    let Some(store) = open_repo_store() else {
        eprintln!("skipping: no .semantex index present");
        return;
    };
    let all_ids = store.get_all_chunk_ids().expect("chunk ids");
    let Some(&seed_id) = all_ids.first() else {
        eprintln!("skipping: empty index");
        return;
    };
    let mut fused = vec![ScoredChunkId::new(seed_id, 10.0)];
    let mut chunk_map: HashMap<u64, Chunk> = HashMap::new();
    for c in store.get_chunks(&[seed_id]).unwrap() {
        chunk_map.insert(c.id, c);
    }
    let mut config = GraphPropagationConfig::localization_mode(20);
    config.disabled = true;
    let new_ids = run_graph_stage(&mut fused, &mut chunk_map, &store, &config, 20).unwrap();
    assert!(new_ids.is_empty());
    assert_eq!(fused.len(), 1);
}
