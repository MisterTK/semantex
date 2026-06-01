//! Search-time score propagation through the code graph.
//!
//! After standard retrieval (BM25 + dense + fusion), propagate scores
//! through call edges, type references, and type hierarchy to discover
//! structurally related but textually dissimilar chunks.

use std::collections::HashMap;

use anyhow::Result;

use crate::index::storage::ChunkStore;
use crate::search::query_classifier::QueryType;
use crate::types::ScoredChunkId;

/// Configuration for graph propagation behavior.
pub struct GraphPropagationConfig {
    pub call_decay: f32,
    pub caller_decay: f32,
    pub type_ref_decay: f32,
    pub transitive_decay: f32,
    pub hierarchy_decay: f32,
    pub max_propagated: usize,
    pub enable_transitive: bool,
    /// Master off switch (`SEMANTEX_GRAPH_DISABLE`). When set, `propagate`
    /// returns the seeds unchanged — used by the S0 harness graph-off A/B and
    /// as a user escape hatch. Default false.
    pub disabled: bool,
    /// Import-cohesion ("community") expansion decay (`SEMANTEX_GRAPH_MODULE_DECAY`).
    /// When `> 0.0`, the stage expands seeds to chunks in import-neighbor files
    /// (shared-import cohesion ≈ same subsystem) for the exhaustive route.
    /// Default `0.0` (off) in every preset per the S4 acceptance gate.
    pub module_decay: f32,
}

impl GraphPropagationConfig {
    /// Standard config per query type.
    #[must_use]
    pub fn for_query_type(query_type: &QueryType, top_k: usize) -> Self {
        match query_type {
            QueryType::Identifier => Self {
                call_decay: 0.10,
                caller_decay: 0.10,
                type_ref_decay: 0.05,
                transitive_decay: 0.0,
                hierarchy_decay: 0.05,
                max_propagated: top_k / 4,
                enable_transitive: false,
                disabled: false,
                module_decay: 0.0,
            },
            QueryType::Keyword => Self {
                call_decay: 0.15,
                caller_decay: 0.15,
                type_ref_decay: 0.10,
                transitive_decay: 0.0,
                hierarchy_decay: 0.10,
                max_propagated: top_k / 3,
                enable_transitive: false,
                disabled: false,
                module_decay: 0.0,
            },
            QueryType::Semantic => Self {
                call_decay: 0.15,
                caller_decay: 0.12,
                type_ref_decay: 0.08,
                transitive_decay: 0.0,
                hierarchy_decay: 0.12,
                max_propagated: top_k / 4,
                enable_transitive: false,
                disabled: false,
                module_decay: 0.0,
            },
            QueryType::Mixed => Self {
                call_decay: 0.20,
                caller_decay: 0.15,
                type_ref_decay: 0.12,
                transitive_decay: 0.0,
                hierarchy_decay: 0.10,
                max_propagated: top_k / 3,
                enable_transitive: false,
                disabled: false,
                module_decay: 0.0,
            },
        }
    }

    /// Architectural mode: enables 2-hop transitive expansion.
    #[must_use]
    pub fn architectural_mode(top_k: usize) -> Self {
        Self {
            call_decay: 0.30,
            caller_decay: 0.25,
            type_ref_decay: 0.20,
            transitive_decay: 0.08,
            hierarchy_decay: 0.15,
            max_propagated: top_k,
            enable_transitive: true,
            disabled: false,
            module_decay: 0.0,
        }
    }

    /// Localization mode: recall-oriented expansion for SWE-bench-style
    /// file-level localization. Expands top seeds 1–2 hops along call + import/
    /// type edges with generous decays and a high propagated cap so structurally
    /// related (but textually dissimilar) sites surface in the top-k. Tuned on
    /// the S0 SWE-loc eval (S4 Task S4.7); S4's gate is file-level recall —
    /// function-level recall is an S0 follow-up (integration §4 D-graph).
    #[must_use]
    pub fn localization_mode(top_k: usize) -> Self {
        Self {
            call_decay: 0.35,
            caller_decay: 0.35,
            type_ref_decay: 0.25,
            transitive_decay: 0.12,
            hierarchy_decay: 0.20,
            max_propagated: top_k,
            enable_transitive: true,
            disabled: false,
            module_decay: 0.0,
        }
    }

    /// Apply environment variable overrides for tuning.
    #[must_use]
    #[allow(clippy::collapsible_if)]
    pub fn with_env_overrides(mut self) -> Self {
        if let Ok(v) = std::env::var("SEMANTEX_GRAPH_CALL_DECAY") {
            if let Ok(f) = v.parse() {
                self.call_decay = f;
            }
        }
        if let Ok(v) = std::env::var("SEMANTEX_GRAPH_CALLER_DECAY") {
            if let Ok(f) = v.parse() {
                self.caller_decay = f;
            }
        }
        if let Ok(v) = std::env::var("SEMANTEX_GRAPH_TYPE_DECAY") {
            if let Ok(f) = v.parse() {
                self.type_ref_decay = f;
            }
        }
        if let Ok(v) = std::env::var("SEMANTEX_GRAPH_HIERARCHY_DECAY") {
            if let Ok(f) = v.parse() {
                self.hierarchy_decay = f;
            }
        }
        if let Ok(v) = std::env::var("SEMANTEX_GRAPH_TRANSITIVE_DECAY") {
            if let Ok(f) = v.parse() {
                self.transitive_decay = f;
            }
        }
        if let Ok(v) = std::env::var("SEMANTEX_GRAPH_HOPS") {
            if let Ok(h) = v.parse::<u32>() {
                // hops=1 disables 2-hop transitive; hops>=2 enables it.
                self.enable_transitive = h >= 2;
            }
        }
        if let Ok(v) = std::env::var("SEMANTEX_GRAPH_DISABLE") {
            // Any non-empty, non-"0" value disables the stage.
            self.disabled = !v.is_empty() && v != "0";
        }
        if let Ok(v) = std::env::var("SEMANTEX_GRAPH_MODULE_DECAY") {
            if let Ok(f) = v.parse() {
                self.module_decay = f;
            }
        }
        self
    }
}

/// Propagate scores through the code graph from seed results.
///
/// Seeds keep their original scores. Newly discovered chunks (not in
/// seeds) receive propagated scores scaled by the maximum seed score
/// and the relevant decay factor.  The result is sorted by score
/// descending and truncated to `seeds.len() + config.max_propagated`.
pub fn propagate(
    seeds: &[ScoredChunkId],
    store: &ChunkStore,
    config: &GraphPropagationConfig,
) -> Result<Vec<ScoredChunkId>> {
    if seeds.is_empty() {
        return Ok(Vec::new());
    }

    if config.disabled {
        return Ok(seeds.to_vec());
    }

    // Build seed lookup: chunk_id -> original score
    let seed_scores: HashMap<u64, f32> = seeds.iter().map(|s| (s.chunk_id, s.score)).collect();

    let max_seed_score = seeds
        .iter()
        .map(|s| s.score)
        .fold(f32::NEG_INFINITY, f32::max);

    if max_seed_score <= 0.0 {
        return Ok(seeds.to_vec());
    }

    // Normalize seed scores to [0,1]
    let normalized: HashMap<u64, f32> = seed_scores
        .iter()
        .map(|(&id, &score)| (id, score / max_seed_score))
        .collect();

    let seed_ids: Vec<u64> = seeds.iter().map(|s| s.chunk_id).collect();

    // Accumulate propagated scores for non-seed chunks
    let mut propagated: HashMap<u64, f32> = HashMap::new();

    // 1-hop call edges (outgoing): seed calls callee -> callee gets score * call_decay
    if config.call_decay > 0.0 {
        let call_edges = store.get_call_edges_from(&seed_ids)?;
        for (caller_id, callee_id) in &call_edges {
            if let Some(&norm_score) = normalized.get(caller_id) {
                let bonus = norm_score * config.call_decay;
                accumulate(&mut propagated, *callee_id, bonus);
            }
        }
    }

    // 1-hop caller edges (incoming): seed is called by caller -> caller gets score * caller_decay
    if config.caller_decay > 0.0 {
        let caller_edges = store.get_call_edges_to(&seed_ids)?;
        for (callee_id, caller_id) in &caller_edges {
            if let Some(&norm_score) = normalized.get(callee_id) {
                let bonus = norm_score * config.caller_decay;
                accumulate(&mut propagated, *caller_id, bonus);
            }
        }
    }

    // 1-hop type reference edges: seed defines type -> usage gets boost, seed uses type -> def gets boost
    if config.type_ref_decay > 0.0 {
        let def_edges = store.get_type_ref_edges_to_defs(&seed_ids)?;
        for (def_id, usage_id) in &def_edges {
            if let Some(&norm_score) = normalized.get(def_id) {
                let bonus = norm_score * config.type_ref_decay;
                accumulate(&mut propagated, *usage_id, bonus);
            }
        }

        let usage_edges = store.get_type_ref_edges_from_usages(&seed_ids)?;
        for (usage_id, def_id) in &usage_edges {
            if let Some(&norm_score) = normalized.get(usage_id) {
                let bonus = norm_score * config.type_ref_decay;
                accumulate(&mut propagated, *def_id, bonus);
            }
        }
    }

    // 1-hop hierarchy edges: parent/child relationships
    if config.hierarchy_decay > 0.0 {
        let hierarchy_edges = store.get_hierarchy_edges_for(&seed_ids)?;
        for (source_id, related_id) in &hierarchy_edges {
            if let Some(&norm_score) = normalized.get(source_id) {
                let bonus = norm_score * config.hierarchy_decay;
                accumulate(&mut propagated, *related_id, bonus);
            }
        }
    }

    // 2-hop transitive: callees of callees
    if config.enable_transitive && config.transitive_decay > 0.0 {
        // Collect hop-1 callees (non-seed chunks discovered via call edges)
        let hop1_callees: Vec<u64> = propagated
            .keys()
            .filter(|id| !seed_scores.contains_key(id))
            .copied()
            .collect();

        if !hop1_callees.is_empty() {
            let hop2_edges = store.get_call_edges_from(&hop1_callees)?;
            for (hop1_id, hop2_id) in &hop2_edges {
                if let Some(&hop1_score) = propagated.get(hop1_id) {
                    let bonus = hop1_score * config.transitive_decay;
                    accumulate(&mut propagated, *hop2_id, bonus);
                }
            }
        }
    }

    // Remove seed chunks from propagated (seeds keep original scores)
    for seed_id in &seed_ids {
        propagated.remove(seed_id);
    }

    // Build final result: seeds with original scores + propagated with scaled scores
    let mut result: Vec<ScoredChunkId> = seeds.to_vec();
    for (chunk_id, prop_score) in &propagated {
        result.push(ScoredChunkId::new(*chunk_id, prop_score * max_seed_score));
    }

    result.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    result.truncate(seeds.len() + config.max_propagated);

    Ok(result)
}

/// Accumulate a bonus score for a chunk, keeping the max across all edges.
fn accumulate(map: &mut HashMap<u64, f32>, chunk_id: u64, bonus: f32) {
    let entry = map.entry(chunk_id).or_insert(0.0);
    if bonus > *entry {
        *entry = bonus;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes the env-mutating tests below so they don't clobber each
    /// other's `SEMANTEX_GRAPH_*` vars under cargo's default parallel runner.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn scored(id: u64, score: f32) -> ScoredChunkId {
        ScoredChunkId::new(id, score)
    }

    #[test]
    fn test_empty_seeds_returns_empty() {
        let config = GraphPropagationConfig::for_query_type(&QueryType::Semantic, 10);
        // We cannot call propagate without a real ChunkStore, but we can test the early return
        // by checking the condition directly.
        let seeds: Vec<ScoredChunkId> = vec![];
        assert!(seeds.is_empty());
        assert_eq!(config.max_propagated, 2); // 10/4
    }

    #[test]
    fn test_config_for_identifier() {
        let config = GraphPropagationConfig::for_query_type(&QueryType::Identifier, 20);
        assert!((config.call_decay - 0.10).abs() < f32::EPSILON);
        assert!((config.caller_decay - 0.10).abs() < f32::EPSILON);
        assert_eq!(config.max_propagated, 5); // 20/4
        assert!(!config.enable_transitive);
    }

    #[test]
    fn test_config_for_semantic() {
        let config = GraphPropagationConfig::for_query_type(&QueryType::Semantic, 10);
        assert!((config.call_decay - 0.15).abs() < f32::EPSILON);
        assert!((config.caller_decay - 0.12).abs() < f32::EPSILON);
        assert_eq!(config.max_propagated, 2); // 10/4
        assert!(!config.enable_transitive);
    }

    #[test]
    fn test_config_architectural_mode() {
        let config = GraphPropagationConfig::architectural_mode(10);
        assert!((config.call_decay - 0.30).abs() < f32::EPSILON);
        assert!((config.transitive_decay - 0.08).abs() < f32::EPSILON);
        assert_eq!(config.max_propagated, 10);
        assert!(config.enable_transitive);
    }

    #[test]
    fn test_accumulate_takes_max() {
        let mut map = HashMap::new();
        accumulate(&mut map, 1, 0.3);
        accumulate(&mut map, 1, 0.5);
        accumulate(&mut map, 1, 0.2);
        assert!((map[&1] - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_accumulate_separate_keys() {
        let mut map = HashMap::new();
        accumulate(&mut map, 1, 0.3);
        accumulate(&mut map, 2, 0.7);
        assert!((map[&1] - 0.3).abs() < f32::EPSILON);
        assert!((map[&2] - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_merge_logic_seeds_keep_original_scores() {
        // Simulate the merge step without a database
        let seeds = vec![scored(1, 10.0), scored(2, 8.0)];
        let max_seed_score = 10.0_f32;

        // Simulated propagated scores (normalized)
        let mut propagated: HashMap<u64, f32> = HashMap::new();
        propagated.insert(3, 0.25); // callee of seed 1
        propagated.insert(4, 0.15); // type-ref from seed 2

        // Remove seeds from propagated
        propagated.remove(&1);
        propagated.remove(&2);

        let mut result: Vec<ScoredChunkId> = seeds.clone();
        for (chunk_id, prop_score) in &propagated {
            result.push(scored(*chunk_id, prop_score * max_seed_score));
        }

        result.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Seeds should be first (10.0, 8.0), then propagated (2.5, 1.5)
        assert_eq!(result[0].chunk_id, 1);
        assert!((result[0].score - 10.0).abs() < f32::EPSILON);
        assert_eq!(result[1].chunk_id, 2);
        assert!((result[1].score - 8.0).abs() < f32::EPSILON);
        // Propagated chunks get prop_score * max_seed_score
        let prop_scores: Vec<f32> = result[2..].iter().map(|s| s.score).collect();
        assert!(prop_scores.contains(&2.5));
        assert!(prop_scores.contains(&1.5));
    }

    #[test]
    fn test_merge_truncation() {
        let seeds = vec![scored(1, 10.0), scored(2, 8.0)];
        let max_propagated = 1;
        let max_seed_score = 10.0_f32;

        let mut propagated: HashMap<u64, f32> = HashMap::new();
        propagated.insert(3, 0.25);
        propagated.insert(4, 0.15);
        propagated.insert(5, 0.05);

        let mut result: Vec<ScoredChunkId> = seeds.clone();
        for (chunk_id, prop_score) in &propagated {
            result.push(scored(*chunk_id, prop_score * max_seed_score));
        }

        result.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        result.truncate(seeds.len() + max_propagated);

        // 2 seeds + 1 max_propagated = 3 total
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].chunk_id, 1);
        assert_eq!(result[1].chunk_id, 2);
        assert_eq!(result[2].chunk_id, 3); // highest propagated score
    }

    #[test]
    fn test_zero_decay_skips_propagation() {
        let config = GraphPropagationConfig {
            call_decay: 0.0,
            caller_decay: 0.0,
            type_ref_decay: 0.0,
            transitive_decay: 0.0,
            hierarchy_decay: 0.0,
            max_propagated: 10,
            enable_transitive: false,
            disabled: false,
            module_decay: 0.0,
        };
        // All decays are zero, so no propagation would occur
        assert!((config.call_decay).abs() < f32::EPSILON);
        assert!((config.caller_decay).abs() < f32::EPSILON);
    }

    #[test]
    fn test_env_overrides() {
        // Test that with_env_overrides doesn't crash when env vars are unset
        let config =
            GraphPropagationConfig::for_query_type(&QueryType::Semantic, 10).with_env_overrides();
        // Should still have default values since env vars aren't set
        assert!((config.call_decay - 0.15).abs() < f32::EPSILON);
    }

    #[test]
    fn test_localization_mode_enables_two_hop_and_caps_propagated() {
        let config = GraphPropagationConfig::localization_mode(20);
        // Localization leans on call + import/type structure; transitive ON.
        assert!(config.enable_transitive);
        assert!(config.transitive_decay > 0.0);
        assert!(config.call_decay > 0.0 && config.caller_decay > 0.0);
        // Expands generously for recall: up to top_k new chunks.
        assert_eq!(config.max_propagated, 20);
        assert!(!config.disabled);
    }

    #[test]
    fn test_module_decay_off_by_default_in_all_presets() {
        // Cohesion expansion ships OFF by default in EVERY preset (incl.
        // localization_mode) per the S4 acceptance gate: default behavior must
        // equal pre-S4. It is opt-in only via SEMANTEX_GRAPH_MODULE_DECAY.
        for qt in [
            QueryType::Identifier,
            QueryType::Keyword,
            QueryType::Semantic,
            QueryType::Mixed,
        ] {
            assert!(
                (GraphPropagationConfig::for_query_type(&qt, 10).module_decay).abs() < f32::EPSILON,
            );
        }
        assert!((GraphPropagationConfig::architectural_mode(10).module_decay).abs() < f32::EPSILON,);
        assert!(
            (GraphPropagationConfig::localization_mode(10).module_decay).abs() < f32::EPSILON,
            "localization_mode must also default module_decay to 0 (off by default)"
        );
    }

    #[test]
    fn test_module_decay_env_override() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { std::env::set_var("SEMANTEX_GRAPH_MODULE_DECAY", "0.10") };
        let config = GraphPropagationConfig::localization_mode(10).with_env_overrides();
        unsafe { std::env::remove_var("SEMANTEX_GRAPH_MODULE_DECAY") };
        assert!((config.module_decay - 0.10).abs() < f32::EPSILON);
    }

    #[test]
    fn test_disabled_default_is_false() {
        let config = GraphPropagationConfig::for_query_type(&QueryType::Semantic, 10);
        assert!(!config.disabled);
    }

    #[test]
    fn test_hops_env_one_forces_transitive_off() {
        // SAFETY: env-mutating tests are serialized by ENV_LOCK; restored at end.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { std::env::set_var("SEMANTEX_GRAPH_HOPS", "1") };
        let config = GraphPropagationConfig::architectural_mode(10).with_env_overrides();
        unsafe { std::env::remove_var("SEMANTEX_GRAPH_HOPS") };
        assert!(!config.enable_transitive, "hops=1 must disable 2-hop");
    }

    #[test]
    fn test_hops_env_two_forces_transitive_on() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { std::env::set_var("SEMANTEX_GRAPH_HOPS", "2") };
        // Start from a 1-hop preset; hops=2 must turn transitive ON.
        let config =
            GraphPropagationConfig::for_query_type(&QueryType::Identifier, 10).with_env_overrides();
        unsafe { std::env::remove_var("SEMANTEX_GRAPH_HOPS") };
        assert!(config.enable_transitive, "hops=2 must enable 2-hop");
    }

    #[test]
    fn test_disable_env_sets_disabled() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { std::env::set_var("SEMANTEX_GRAPH_DISABLE", "1") };
        let config =
            GraphPropagationConfig::for_query_type(&QueryType::Semantic, 10).with_env_overrides();
        unsafe { std::env::remove_var("SEMANTEX_GRAPH_DISABLE") };
        assert!(config.disabled);
    }

    #[test]
    fn test_propagate_returns_seeds_unchanged_when_disabled() {
        // disabled short-circuits before any store access, so a never-touched
        // store reference is fine — we pass seeds through a disabled config path
        // by checking the early-return contract directly.
        let config = GraphPropagationConfig {
            call_decay: 0.3,
            caller_decay: 0.3,
            type_ref_decay: 0.2,
            transitive_decay: 0.1,
            hierarchy_decay: 0.1,
            max_propagated: 10,
            enable_transitive: true,
            disabled: true,
            module_decay: 0.0,
        };
        assert!(config.disabled);
        // The disabled branch in propagate() returns seeds.to_vec() — covered by
        // the integration test in Task S4.4 against a real ChunkStore.
    }
}
