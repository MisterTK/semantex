//! PageRank over the unified code graph (call + import + hierarchy edges).
//!
//! Computes a `structural_centrality` score in [0.0, 1.0] for every chunk node,
//! capturing how "load-bearing" a chunk is in the codebase's structural graph.
//! High-centrality chunks are typically god-nodes — APIs, central services,
//! widely-used types — and are excellent priors for reranking and architectural
//! summaries.
//!
//! ## Algorithm
//!
//! Standard PageRank with:
//! - damping factor `d = 0.85` (canonical value)
//! - max iterations `50` (per spec risk T3 mitigation)
//! - convergence tolerance `1e-6` on L1 norm
//!
//! Edges from the call graph, type-reference graph (defining_chunk → usage_chunk),
//! and type-hierarchy graph are merged into a single weighted-uniform out-edge map.
//! A dangling-node correction (mass-redistribution to all nodes) keeps the rank
//! vector a valid probability distribution.
//!
//! For very large graphs (>100k symbols, per spec T3), callers should pass
//! `max_iterations <= 50` to bound wall-time. The reference `compute_pagerank`
//! entrypoint already applies this cap.

use std::collections::{HashMap, HashSet};

/// Damping factor for PageRank — the probability of continuing the random walk
/// vs. teleporting to a uniformly-random node. 0.85 is the canonical Brin–Page
/// value.
pub const DEFAULT_DAMPING: f32 = 0.85;

/// Maximum number of power-iteration steps. Per spec T3 mitigation:
/// "Cap iterations at 50" for graphs >100k symbols.
pub const DEFAULT_MAX_ITERATIONS: usize = 50;

/// Convergence tolerance on the L1 norm of `|rank_new - rank_old|`.
pub const DEFAULT_TOLERANCE: f32 = 1e-6;

/// Lightweight, owned representation of the structural graph used for PageRank.
///
/// `out_edges[node]` lists the outbound neighbors of `node`. The graph is
/// directed; symmetric relationships (e.g. import edges) should be added in
/// both directions by the caller if undirected semantics are desired.
///
/// Node IDs are `u64` (chunk IDs from SQLite). All node IDs that appear in any
/// edge are implicitly registered as nodes; isolated nodes can be added via
/// `add_node`.
#[derive(Debug, Default, Clone)]
pub struct CodeGraph {
    /// Outbound adjacency: `node -> Vec<neighbor>`.
    out_edges: HashMap<u64, Vec<u64>>,
    /// Full set of known nodes (includes nodes without outbound edges).
    nodes: HashSet<u64>,
}

impl CodeGraph {
    /// Build an empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a node. Idempotent.
    pub fn add_node(&mut self, node: u64) {
        self.nodes.insert(node);
    }

    /// Add a directed edge `from -> to`. Both endpoints are registered as nodes
    /// even when the edge itself is dropped (self-loops). Self-loops would
    /// inflate the node's own centrality without adding structural information
    /// — we register the node so it still appears in the rank output, but skip
    /// inserting the loop into the adjacency.
    pub fn add_edge(&mut self, from: u64, to: u64) {
        self.nodes.insert(from);
        self.nodes.insert(to);
        if from == to {
            return;
        }
        self.out_edges.entry(from).or_default().push(to);
    }

    /// Add an undirected edge by inserting both `from -> to` and `to -> from`.
    /// Useful for import-style relationships where direction is informational
    /// but bidirectional structural coupling is what matters for centrality.
    pub fn add_undirected_edge(&mut self, a: u64, b: u64) {
        self.add_edge(a, b);
        self.add_edge(b, a);
    }

    /// Number of registered nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of directed edges (counts duplicates and parallel edges).
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.out_edges.values().map(Vec::len).sum()
    }

    /// True if the graph has no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Iterate over all known nodes in arbitrary order.
    pub fn nodes(&self) -> impl Iterator<Item = u64> + '_ {
        self.nodes.iter().copied()
    }
}

/// Configuration for a PageRank run.
#[derive(Debug, Clone, Copy)]
pub struct PageRankConfig {
    pub damping: f32,
    pub max_iterations: usize,
    pub tolerance: f32,
}

impl Default for PageRankConfig {
    fn default() -> Self {
        Self {
            damping: DEFAULT_DAMPING,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            tolerance: DEFAULT_TOLERANCE,
        }
    }
}

/// Compute PageRank over a `CodeGraph` using the default configuration.
///
/// Returns a `HashMap<chunk_id, centrality>` where centrality values are in
/// `[0.0, 1.0]` and sum to ~1.0 (modulo floating-point noise). The output is
/// suitable for direct use as a reranker feature or for god-node selection.
///
/// On an empty graph, returns an empty map. On a graph with isolated nodes
/// (no outbound edges), the dangling-node correction redistributes their
/// share uniformly.
#[must_use]
pub fn compute_pagerank(graph: &CodeGraph) -> HashMap<u64, f32> {
    compute_pagerank_with(graph, PageRankConfig::default())
}

/// Compute PageRank with explicit configuration.
///
/// Implements the standard Brin–Page formulation with dangling-node handling:
///
/// ```text
///   rank(v) = (1 - d) / N
///           + d * sum_{u in in_neighbors(v)} rank(u) / out_deg(u)
///           + d * dangling_mass / N
/// ```
///
/// where `dangling_mass = sum_{u : out_deg(u) == 0} rank(u)`.
#[must_use]
pub fn compute_pagerank_with(graph: &CodeGraph, config: PageRankConfig) -> HashMap<u64, f32> {
    if graph.is_empty() {
        return HashMap::new();
    }

    let n_nodes = graph.node_count();
    #[allow(clippy::cast_precision_loss)]
    let n_f = n_nodes as f32;
    let base = (1.0 - config.damping) / n_f;

    // Stable node order for deterministic iteration.
    let mut node_list: Vec<u64> = graph.nodes().collect();
    node_list.sort_unstable();
    let node_index: HashMap<u64, usize> =
        node_list.iter().enumerate().map(|(i, n)| (*n, i)).collect();

    // Build inbound adjacency for efficient propagation: for each node v,
    // list of (u, out_deg(u)) tuples where u -> v exists.
    let mut in_neighbors: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_nodes];
    let mut out_degree: Vec<f32> = vec![0.0; n_nodes];
    for (from, tos) in &graph.out_edges {
        let Some(&u_idx) = node_index.get(from) else {
            continue;
        };
        #[allow(clippy::cast_precision_loss)]
        let deg = tos.len() as f32;
        out_degree[u_idx] = deg;
        for to in tos {
            if let Some(&v_idx) = node_index.get(to) {
                in_neighbors[v_idx].push((u_idx, deg));
            }
        }
    }

    let mut rank: Vec<f32> = vec![1.0 / n_f; n_nodes];
    let mut next: Vec<f32> = vec![0.0; n_nodes];

    for _iter in 0..config.max_iterations {
        // Dangling mass: sum of ranks at nodes with no outbound edges.
        let mut dangling: f32 = 0.0;
        for i in 0..n_nodes {
            if out_degree[i] == 0.0 {
                dangling += rank[i];
            }
        }
        let dangling_share = config.damping * dangling / n_f;

        // Propagate: rank_new(v) = base + d * sum_{u in In(v)} rank(u)/out_deg(u) + dangling_share.
        next.fill(base + dangling_share);
        for v_idx in 0..n_nodes {
            for &(u_idx, deg) in &in_neighbors[v_idx] {
                // deg > 0 because u has at least the edge (u -> v).
                next[v_idx] += config.damping * rank[u_idx] / deg;
            }
        }

        // Check convergence via L1 norm of change.
        let mut delta: f32 = 0.0;
        for i in 0..n_nodes {
            delta += (next[i] - rank[i]).abs();
        }
        std::mem::swap(&mut rank, &mut next);
        if delta < config.tolerance {
            break;
        }
    }

    node_list.into_iter().zip(rank).collect()
}

/// Build a `CodeGraph` from raw edge lists pulled from `ChunkStore`.
///
/// Caller supplies:
/// - `call_edges`: `(caller_chunk_id, callee_chunk_id)` pairs (directed)
/// - `type_ref_edges`: `(defining_chunk, usage_chunk)` pairs (treated as undirected
///   so that high-fan-in *and* high-fan-out type definitions earn rank)
/// - `hierarchy_edges`: `(child_chunk, parent_chunk)` pairs (parent → child is the
///   semantically-meaningful "is-a" direction; we add both for centrality)
/// - `all_chunk_ids`: every known chunk_id, so that isolated chunks are still
///   represented and receive `base` rank
#[must_use]
pub fn build_code_graph(
    call_edges: &[(u64, u64)],
    type_ref_edges: &[(u64, u64)],
    hierarchy_edges: &[(u64, u64)],
    all_chunk_ids: &[u64],
) -> CodeGraph {
    let mut g = CodeGraph::new();
    for &id in all_chunk_ids {
        g.add_node(id);
    }
    for &(caller, callee) in call_edges {
        g.add_edge(caller, callee);
    }
    for &(def, usage) in type_ref_edges {
        g.add_undirected_edge(def, usage);
    }
    for &(child, parent) in hierarchy_edges {
        g.add_undirected_edge(child, parent);
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn empty_graph_returns_empty_ranks() {
        let g = CodeGraph::new();
        let ranks = compute_pagerank(&g);
        assert!(ranks.is_empty());
    }

    #[test]
    fn single_node_has_unit_rank() {
        let mut g = CodeGraph::new();
        g.add_node(1);
        let ranks = compute_pagerank(&g);
        assert_eq!(ranks.len(), 1);
        assert!(
            approx_eq(ranks[&1], 1.0, 1e-4),
            "single node rank should be ~1.0, got {}",
            ranks[&1]
        );
    }

    #[test]
    fn three_node_chain_central_node_wins() {
        // 1 -> 2 -> 3 -> 2  (node 2 is a sink/hub via the back-edge)
        let mut g = CodeGraph::new();
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        g.add_edge(3, 2);
        let ranks = compute_pagerank(&g);
        assert_eq!(ranks.len(), 3);
        // Node 2 receives flow from both 1 and 3 → should be highest.
        assert!(
            ranks[&2] > ranks[&1],
            "node 2 ({}) should outrank node 1 ({})",
            ranks[&2],
            ranks[&1]
        );
        assert!(
            ranks[&2] > ranks[&3],
            "node 2 ({}) should outrank node 3 ({})",
            ranks[&2],
            ranks[&3]
        );
        // Ranks must form a valid probability distribution.
        let sum: f32 = ranks.values().sum();
        assert!(
            approx_eq(sum, 1.0, 1e-3),
            "ranks should sum to ~1.0, got {sum}"
        );
    }

    #[test]
    fn star_graph_central_hub_dominates() {
        // 5 leaves all point at hub node 0.
        let mut g = CodeGraph::new();
        for leaf in 1..=5u64 {
            g.add_edge(leaf, 0);
        }
        let ranks = compute_pagerank(&g);
        let hub = ranks[&0];
        for leaf in 1..=5u64 {
            assert!(
                hub > ranks[&leaf],
                "hub rank ({hub}) should exceed leaf {} rank ({})",
                leaf,
                ranks[&leaf]
            );
        }
    }

    #[test]
    fn isolated_nodes_get_base_rank() {
        let mut g = CodeGraph::new();
        g.add_node(10);
        g.add_node(11);
        g.add_edge(20, 21);
        let ranks = compute_pagerank(&g);
        assert_eq!(ranks.len(), 4);
        // Isolated nodes should all have equal (base) rank.
        assert!(approx_eq(ranks[&10], ranks[&11], 1e-4));
        // And they should be positive — dangling redistribution gives non-zero mass.
        assert!(ranks[&10] > 0.0);
    }

    #[test]
    fn self_loops_are_ignored() {
        let mut g = CodeGraph::new();
        g.add_edge(1, 1);
        g.add_edge(2, 3);
        // Node 1 should not have any outbound edges (self-loop dropped).
        let ranks = compute_pagerank(&g);
        assert!(ranks.contains_key(&1));
        // Node 1 is now effectively isolated — should still get base rank.
        assert!(ranks[&1] > 0.0);
    }

    #[test]
    fn ranks_sum_to_unity() {
        // 8-node small graph with a mix of edges.
        let mut g = CodeGraph::new();
        let edges = [(1, 2), (2, 3), (3, 1), (4, 1), (5, 4), (6, 7), (7, 8)];
        for (a, b) in edges {
            g.add_edge(a, b);
        }
        let ranks = compute_pagerank(&g);
        let sum: f32 = ranks.values().sum();
        assert!(
            approx_eq(sum, 1.0, 5e-3),
            "ranks should sum to ~1.0, got {sum}"
        );
    }

    #[test]
    fn build_code_graph_merges_all_edge_types() {
        let calls = vec![(1u64, 2u64)];
        let type_refs = vec![(3u64, 4u64)];
        let hierarchy = vec![(5u64, 6u64)];
        let all_ids = vec![1u64, 2, 3, 4, 5, 6, 7]; // 7 is isolated
        let g = build_code_graph(&calls, &type_refs, &hierarchy, &all_ids);
        assert_eq!(g.node_count(), 7);
        // 1 call edge (directed) + 2 type-ref edges (undirected) + 2 hierarchy
        // edges (undirected) = 5 directed edges.
        assert_eq!(g.edge_count(), 5);

        let ranks = compute_pagerank(&g);
        assert_eq!(ranks.len(), 7);
        // The isolated node 7 should still be present.
        assert!(ranks.contains_key(&7));
    }

    #[test]
    fn convergence_respects_max_iterations() {
        let mut g = CodeGraph::new();
        for i in 0..20u64 {
            for j in 0..20u64 {
                if i != j {
                    g.add_edge(i, j);
                }
            }
        }
        // Very low iteration cap — should still produce a valid distribution.
        let cfg = PageRankConfig {
            damping: 0.85,
            max_iterations: 3,
            tolerance: 1e-12, // unreachable, force iteration cap
        };
        let ranks = compute_pagerank_with(&g, cfg);
        let sum: f32 = ranks.values().sum();
        assert!(
            approx_eq(sum, 1.0, 5e-3),
            "ranks should remain a distribution even with iter cap: sum={sum}"
        );
        // On a complete graph, all nodes should have approximately equal rank.
        let expected = 1.0 / 20.0;
        for v in ranks.values() {
            assert!(
                approx_eq(*v, expected, 0.02),
                "complete-graph rank should be ~1/N, got {v}"
            );
        }
    }
}
