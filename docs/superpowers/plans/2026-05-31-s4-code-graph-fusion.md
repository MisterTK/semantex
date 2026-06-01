# S4 — Code-Graph Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote semantex's existing code-graph propagation from an untuned post-fusion boost to a **measured, tuned, first-class pipeline stage**. Concretely: (1) make graph propagation a named stage (`GraphStage`) with explicit, env-tunable decays and a single graph-on/off switch (`SEMANTEX_GRAPH_DISABLE`); (2) gate optional 2-hop transitive expansion per query route (architectural / exhaustive / feature-planning) via route predicates rather than only the existing `is_architectural_query` keyword check; (3) add a localization-oriented expansion preset that, after dense+sparse fusion, expands the top seeds 1–2 hops along call/import/type edges, re-scores, and tags `GraphExpanded`; (4) tune the decays + hop count on the S0 SWE-loc localization eval for best **file-level** Recall@{5,10} **without regressing** CoIR/CSN/in-domain. Acceptance: measurable SWE-loc Recall@{5,10} lift vs graph-off, no net regression on CoIR/CSN/in-domain. (The S0 harness ships file-level recall only; function-level recall is an S0 follow-up per integration §4 D-graph — S4's gate is file-level.)

**Architecture:** S4 is almost entirely a **refactor + extension of two existing files** — `crates/semantex-core/src/search/graph_propagation.rs` (the propagation algorithm + per-query-type `GraphPropagationConfig`) and `crates/semantex-core/src/search/hybrid.rs` (where `graph_propagation::propagate` runs as "Phase 6" of the post-fusion boost chain at lines ~861–913). It is **backend-agnostic**: it operates on the already-fused `Vec<ScoredChunkId>` and the SQLite graph tables (`call_graph`, `type_refs`, `type_hierarchy`, `module_edges`) via `ChunkStore`, so it works identically under `colbert-plaid` and `coderank-hnsw`. The graph edges themselves are extracted at index time by `chunking/structured_meta.rs` (`StructuredChunkMeta.calls` / `called_by` / `type_refs` / `implements` / `resolved_imports`) and resolved into `storage.rs`'s graph tables — **S4 changes none of the indexing/extraction path**, only the search-time consumption.

**Tech Stack:**
- Rust (workspace crate `semantex-core`); build `cargo build -p semantex-core`; test `cargo test -p semantex-core <name>`.
- No new dependencies. No new ONNX models. No LLM features touched.
- The graph tables and `ChunkStore` query methods (`get_call_edges_from/to`, `get_type_ref_edges_*`, `get_hierarchy_edges_for`, `get_import_neighbors`, `get_chunks`) already exist (v7 schema) and are reused verbatim.
- Tuning is measured by the **S0 relevance harness** (`benchmarks/relevance/`, Python) via `python -m scripts.run --dataset swe-loc|csn --ablation hybrid`, with graph behavior toggled through pass-through env vars (`SEMANTEX_GRAPH_DISABLE`, `SEMANTEX_GRAPH_*_DECAY`, `SEMANTEX_GRAPH_HOPS`).

---

## File Structure

```
crates/semantex-core/src/search/
├── graph_propagation.rs    # MODIFY — add SEMANTEX_GRAPH_DISABLE gate, localization_mode()
│                           #   preset, route-gated 2-hop (enable_transitive + transitive_decay),
│                           #   GRAPH_HOPS env, max-1/2 hop cap; the propagate() core stays the
│                           #   same shape but gains a localization entry config.
├── hybrid.rs               # MODIFY — turn "Phase 6" (lines ~861–913) into a single named
│                           #   call to graph_stage::run_graph_stage(...); add route detection
│                           #   for exhaustive/feature-planning to pick localization_mode vs
│                           #   architectural_mode vs per-query-type. GraphExpanded tagging at
│                           #   ~1038–1048 is preserved.
├── graph_stage.rs          # CREATE — the named pipeline-stage wrapper: owns the
│                           #   "pick config → propagate → merge expanded into fused → collect
│                           #   new_ids for GraphExpanded tagging" logic extracted out of
│                           #   hybrid.rs so the stage is unit-testable and the boost chain in
│                           #   hybrid.rs reads as one call.
└── query_classifier.rs     # MODIFY (small) — add pub fn route predicates
                            #   is_exhaustive_query(&str) and is_feature_planning_query(&str)
                            #   (universal English signal words only; repo-agnostic) used by
                            #   hybrid.rs to gate 2-hop expansion. No new QueryType variants.
```

**Module responsibilities (one job each):**
- `graph_propagation.rs` — the propagation algorithm and the decay/hop **config presets** (`for_query_type`, `architectural_mode`, new `localization_mode`), env overrides, and the single on/off gate. Pure over `(&[ScoredChunkId], &ChunkStore, &GraphPropagationConfig)`.
- `graph_stage.rs` — the **named stage**: given the current `fused` list, the chunk map, the store, and the chosen config, run propagation and return `(updated_fused, graph_expanded_ids)`. No scoring policy beyond "keep max" (already in `propagate`); no DB schema knowledge beyond calling `ChunkStore`.
- `hybrid.rs` — **route selection + call site**: decide which config preset to use from the query route, call `graph_stage::run_graph_stage`, and feed `graph_expanded_ids` into the existing `SearchSource::GraphExpanded` tagging.
- `query_classifier.rs` — **route predicates** only (free-function booleans over the query string), repo-agnostic.

**Coordination (per spec §5 sequencing):** S4 edits the **post-fusion region of `hybrid.rs`** (currently "Phase 6", lines ~861–913, plus the `GraphExpanded` tagging at ~1038–1048). Per spec §5, **`hybrid.rs` contention is resolved by landing S1 (the `DenseBackend` seam) first**, then S2 (dense channel) and S7 (fusion) coordinate. S4 "touch[es] distinct regions but rebase[s] on S1." **Before starting Task S4.2 (the first `hybrid.rs` edit), rebase this branch on the merged S1 seam refactor** and re-confirm the line numbers of the Phase-6 block and the `GraphExpanded` tagging block (they may shift; the *logic* — `propagate` → merge → tag — is what to find, not the literal lines). Tasks S4.1 (graph_propagation.rs) and S4.6 (query_classifier.rs) touch files S1 does not, and can proceed in parallel with S1.

---

## Phase 0 — Baseline measurement (no code; establishes the number S4 must beat)

### Task S4.0: Record the graph-off vs graph-on SWE-loc baseline (research-only, no commit)

**Files:** none (Append to (create if first): `docs/superpowers/plans/2026-05-31-research-notes.md` under a `## S4 code-graph fusion` section; never overwrite — sibling streams share this file).

This is the "graph-off baseline" the acceptance gate compares against. It must be captured **before** any S4 code change, on TODAY's `main`, so the lift is attributable. It depends on the S0 harness being built (S0 Tasks through 7.x) and the SWE-bench Phase-A repos being indexed (`benchmarks/swe_bench/scripts/pre_index.py`). If the Phase-A cache is empty, the SWE-loc runner indexes on demand (S0 `ensure_index`), so the first run is slow.

- [ ] **Step 1: Confirm the current graph behavior has NO on/off switch (VERIFY)**

Run:
```bash
grep -rn "SEMANTEX_GRAPH" crates/semantex-core/src/search/graph_propagation.rs
```
Expected output: only the five decay overrides exist —
```
SEMANTEX_GRAPH_CALL_DECAY
SEMANTEX_GRAPH_CALLER_DECAY
SEMANTEX_GRAPH_TYPE_DECAY
SEMANTEX_GRAPH_HIERARCHY_DECAY
SEMANTEX_GRAPH_TRANSITIVE_DECAY
```
Record in research notes: **there is currently no `SEMANTEX_GRAPH_DISABLE` and no hop-count env** — S4 Task S4.1 adds them. The graph-off baseline must therefore be captured by setting all five decays to `0` (which makes `propagate` skip every edge class — verified: each edge block in `propagate` is guarded by `if config.<decay> > 0.0`).

- [ ] **Step 2: Capture the SWE-loc graph-OFF baseline (all decays zeroed)**

From the S0 harness venv (`benchmarks/relevance/.venv`), with `semantex` on PATH and built from current `main`:
```bash
cd benchmarks/relevance && source .venv/bin/activate
export SEMANTEX_BINARY=$(which semantex)
export SEMANTEX_GRAPH_CALL_DECAY=0 SEMANTEX_GRAPH_CALLER_DECAY=0 \
       SEMANTEX_GRAPH_TYPE_DECAY=0 SEMANTEX_GRAPH_HIERARCHY_DECAY=0 \
       SEMANTEX_GRAPH_TRANSITIVE_DECAY=0
python -m scripts.run --dataset swe-loc --ablation hybrid --run-id s4-graphoff --k 10
unset SEMANTEX_GRAPH_CALL_DECAY SEMANTEX_GRAPH_CALLER_DECAY \
      SEMANTEX_GRAPH_TYPE_DECAY SEMANTEX_GRAPH_HIERARCHY_DECAY \
      SEMANTEX_GRAPH_TRANSITIVE_DECAY
```
Expected: `results/s4-graphoff/report.md` with one `swe-loc` row. Record `recall_at_5`, `recall_at_10`, `mrr_at_10` verbatim in research notes as **GRAPH-OFF baseline**.

- [ ] **Step 3: Capture the SWE-loc graph-ON baseline (current defaults)**

```bash
cd benchmarks/relevance && source .venv/bin/activate
export SEMANTEX_BINARY=$(which semantex)
python -m scripts.run --dataset swe-loc --ablation hybrid --run-id s4-graphon-default --k 10
```
Expected: `results/s4-graphon-default/report.md`. Record `recall_at_5/10` + `mrr_at_10` as **GRAPH-ON (current defaults)**. Also capture the CSN baseline (the no-regression guard) under the same default graph config:
```bash
python -m scripts.run --dataset csn --ablation hybrid --run-id s4-csn-default --k 10
```
Record CSN `mrr_at_10` / `ndcg_at_10` / `recall_at_10` as **CSN no-regression baseline**.

- [ ] **Step 4: Write the baseline table to research notes (no commit — controller commits)**

Append to (create if first) `docs/superpowers/plans/2026-05-31-research-notes.md` a small table (graph-off vs graph-on-default for SWE-loc R@5/R@10/MRR@10, plus the CSN no-regression numbers) under the `## S4 code-graph fusion` section; never overwrite (sibling streams share this file). **All later tuning tasks (S4.7) compare against these recorded numbers.** Leave version control to the controller.

**Outputs locked after this task:** the graph-off SWE-loc baseline, the current-default SWE-loc number, and the CSN no-regression baseline — the three numbers the acceptance gate is stated against.

---

## Phase 1 — Stage-ify the config: on/off gate + hop-count env + localization preset

### Task S4.1: Add `SEMANTEX_GRAPH_DISABLE` gate, `SEMANTEX_GRAPH_HOPS` env, and `localization_mode()` preset (TDD)

**Files:**
- Modify: `crates/semantex-core/src/search/graph_propagation.rs`

The current `GraphPropagationConfig` has fields `call_decay, caller_decay, type_ref_decay, transitive_decay, hierarchy_decay, max_propagated, enable_transitive` and presets `for_query_type(&QueryType, top_k)` / `architectural_mode(top_k)` / `with_env_overrides(self)`. We add: a `disabled` flag honored from `SEMANTEX_GRAPH_DISABLE`, a `hops` field (1 or 2) honored from `SEMANTEX_GRAPH_HOPS` that drives `enable_transitive`, and a `localization_mode(top_k)` preset tuned for SWE-loc-style **file-level** localization (function-level recall is an S0 follow-up per integration §4 D-graph; S4's gate is file-level).

- [ ] **Step 1: Write failing tests**

Add these tests inside the existing `#[cfg(test)] mod tests` block at the bottom of `crates/semantex-core/src/search/graph_propagation.rs` (after `test_env_overrides`):

```rust
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
    fn test_disabled_default_is_false() {
        let config = GraphPropagationConfig::for_query_type(&QueryType::Semantic, 10);
        assert!(!config.disabled);
    }

    #[test]
    fn test_hops_env_one_forces_transitive_off() {
        // SAFETY: single-threaded test; env restored at end.
        unsafe { std::env::set_var("SEMANTEX_GRAPH_HOPS", "1") };
        let config = GraphPropagationConfig::architectural_mode(10).with_env_overrides();
        assert!(!config.enable_transitive, "hops=1 must disable 2-hop");
        unsafe { std::env::remove_var("SEMANTEX_GRAPH_HOPS") };
    }

    #[test]
    fn test_hops_env_two_forces_transitive_on() {
        unsafe { std::env::set_var("SEMANTEX_GRAPH_HOPS", "2") };
        // Start from a 1-hop preset; hops=2 must turn transitive ON.
        let config = GraphPropagationConfig::for_query_type(&QueryType::Identifier, 10)
            .with_env_overrides();
        assert!(config.enable_transitive, "hops=2 must enable 2-hop");
        unsafe { std::env::remove_var("SEMANTEX_GRAPH_HOPS") };
    }

    #[test]
    fn test_disable_env_sets_disabled() {
        unsafe { std::env::set_var("SEMANTEX_GRAPH_DISABLE", "1") };
        let config = GraphPropagationConfig::for_query_type(&QueryType::Semantic, 10)
            .with_env_overrides();
        assert!(config.disabled);
        unsafe { std::env::remove_var("SEMANTEX_GRAPH_DISABLE") };
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
        };
        assert!(config.disabled);
        // The disabled branch in propagate() returns seeds.to_vec() — covered by
        // the integration test in Task S4.4 against a real ChunkStore.
    }
```

- [ ] **Step 2: Run — expect failure**

Run: `cargo test -p semantex-core graph_propagation::tests::test_localization_mode_enables_two_hop_and_caps_propagated`
Expected: **FAIL to compile** — `no function or associated item named 'localization_mode'` and `struct GraphPropagationConfig has no field named 'disabled'`.

- [ ] **Step 3: Add the `disabled` field, the env knobs, and `localization_mode()`**

In `crates/semantex-core/src/search/graph_propagation.rs`, add the field to the struct (after `enable_transitive`):

```rust
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
}
```

Every existing struct literal that constructs `GraphPropagationConfig` must now set `disabled: false`. There are four such literals in this file: the four match arms of `for_query_type` (`Identifier`, `Keyword`, `Semantic`, `Mixed`), the `architectural_mode` body, and the two test literals in `mod tests` (`test_zero_decay_skips_propagation` and the new `test_propagate_returns_seeds_unchanged_when_disabled`). Add `disabled: false,` as the last field to each of the production literals (`for_query_type` arms + `architectural_mode`) and to `test_zero_decay_skips_propagation`. (The new Task S4.1 test literal already sets `disabled: true`.)

Add the new preset after `architectural_mode`:

```rust
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
        }
    }
```

Extend `with_env_overrides` to honor the two new knobs. Add this block before the final `self` return (keep the existing `#[allow(clippy::collapsible_if)]` on the function):

```rust
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
```

Add the disabled short-circuit at the very top of `propagate`, immediately after the existing `if seeds.is_empty()` guard:

```rust
    if config.disabled {
        return Ok(seeds.to_vec());
    }
```

- [ ] **Step 4: Run — expect pass**

Run: `cargo test -p semantex-core graph_propagation`
Expected: all `graph_propagation::tests::*` pass, including the six new tests. (Run the whole module to also confirm the four updated struct literals still compile.)

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/search/graph_propagation.rs
git commit -m "$(cat <<'EOF'
feat(graph): SEMANTEX_GRAPH_DISABLE gate, GRAPH_HOPS env, localization_mode preset

Adds a master on/off switch and a hop-count env knob to GraphPropagationConfig,
plus a recall-oriented localization_mode() preset for SWE-loc-style expansion.
disabled short-circuits propagate() to pass seeds through unchanged (graph-off
A/B). No behavior change at default config (disabled=false, hops follow presets).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2 — Route predicates for 2-hop gating (exhaustive / feature-planning)

### Task S4.6: Add `is_exhaustive_query` + `is_feature_planning_query` route predicates (TDD)

**Files:**
- Modify: `crates/semantex-core/src/search/query_classifier.rs`

Spec §4 S4 wants 2-hop transitive expansion "gated per query-route (architectural / exhaustive / feature-planning)". Today only **architectural** is realized (`hybrid.rs::is_architectural_query`, which fires for `QueryType::Semantic` + 4+ words + an architectural signal word). The `QueryType` enum has exactly four variants (`Identifier`, `Keyword`, `Semantic`, `Mixed`) and **adding new variants is out of scope** (it would ripple through `fusion_weights`, every match, and the S1/S2/S7 fusion edits). Instead, add two repo-agnostic free-function predicates over the query string, mirroring the existing `is_architectural_query` shape, that `hybrid.rs` uses to pick `localization_mode` (the 2-hop, recall-oriented preset).

The signal words must be **universal English** (CLAUDE.md rule: repo-agnostic; no product/service/domain terms). "Exhaustive" = "find every / all usages / everywhere" intent. "Feature-planning" = "where would I add / implement / wire up" intent.

- [ ] **Step 1: Write failing tests**

Add to the `#[cfg(test)] mod tests` block in `crates/semantex-core/src/search/query_classifier.rs`:

```rust
    #[test]
    fn test_exhaustive_query_detection() {
        assert!(is_exhaustive_query("find all usages of the retry helper"));
        assert!(is_exhaustive_query("every place that reads the config file"));
        assert!(is_exhaustive_query("everywhere we open a database connection"));
        assert!(is_exhaustive_query("list all callers of the auth middleware"));
    }

    #[test]
    fn test_non_exhaustive_query() {
        assert!(!is_exhaustive_query("login handler"));
        assert!(!is_exhaustive_query("getUserById"));
        assert!(!is_exhaustive_query("parse the request body"));
    }

    #[test]
    fn test_feature_planning_query_detection() {
        assert!(is_feature_planning_query("where should I add rate limiting"));
        assert!(is_feature_planning_query("where to implement the new export endpoint"));
        assert!(is_feature_planning_query("how to wire up a second cache layer"));
        assert!(is_feature_planning_query("where would I hook in a metrics counter"));
    }

    #[test]
    fn test_non_feature_planning_query() {
        assert!(!is_feature_planning_query("connection pool"));
        assert!(!is_feature_planning_query("error handling in the parser"));
        assert!(!is_feature_planning_query("std::io::Read"));
    }
```

- [ ] **Step 2: Run — expect failure**

Run: `cargo test -p semantex-core query_classifier::tests::test_exhaustive_query_detection`
Expected: **FAIL to compile** — `cannot find function 'is_exhaustive_query' in this scope`.

- [ ] **Step 3: Implement the two predicates**

Append to `crates/semantex-core/src/search/query_classifier.rs` (after the `classify` function, before `#[cfg(test)]`):

```rust
/// True if the query expresses an *exhaustive* intent — "find every / all
/// occurrences / callers / usages everywhere". Such queries benefit from 2-hop
/// graph expansion so transitively related sites surface. Universal English
/// signals only (repo-agnostic).
#[must_use]
pub fn is_exhaustive_query(query: &str) -> bool {
    let q = query.to_lowercase();
    const SIGNALS: &[&str] = &[
        "all usages",
        "all callers",
        "all references",
        "all places",
        "all the places",
        "every place",
        "every usage",
        "every caller",
        "everywhere",
        "find all",
        "list all",
        "all occurrences",
    ];
    SIGNALS.iter().any(|s| q.contains(s))
}

/// True if the query expresses a *feature-planning / change-impact* intent —
/// "where should I add / implement / wire up X". These benefit from 2-hop
/// expansion to reveal the surrounding integration surface. Universal English
/// signals only (repo-agnostic).
#[must_use]
pub fn is_feature_planning_query(query: &str) -> bool {
    let q = query.to_lowercase();
    // Require a location interrogative paired with an add/implement intent so we
    // don't fire on every "how to" question.
    let asks_where = q.contains("where should")
        || q.contains("where to")
        || q.contains("where would")
        || q.contains("where do i")
        || q.contains("where can i");
    let intent_add = q.contains("add")
        || q.contains("implement")
        || q.contains("wire up")
        || q.contains("hook in")
        || q.contains("hook up")
        || q.contains("introduce")
        || q.contains("integrate");
    asks_where && intent_add
}
```

- [ ] **Step 4: Run — expect pass**

Run: `cargo test -p semantex-core query_classifier`
Expected: all `query_classifier::tests::*` pass, including the four new tests.

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/search/query_classifier.rs
git commit -m "$(cat <<'EOF'
feat(classifier): repo-agnostic exhaustive/feature-planning route predicates

Free-function predicates over the query string (universal English signals only)
that hybrid.rs uses to gate 2-hop graph expansion. No new QueryType variants.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3 — Extract the named graph stage

### Task S4.2: Create `graph_stage.rs` — the named, testable pipeline stage (TDD)

> **Rebase gate:** This is the first task that depends on `hybrid.rs` internals. **Rebase on the merged S1 seam refactor before starting** (spec §5). The extracted logic below mirrors `hybrid.rs` lines ~861–913 on current `main`; if S1 moved them, port the same `propagate → merge → collect new_ids` shape.

**Files:**
- Create: `crates/semantex-core/src/search/graph_stage.rs`
- Modify: `crates/semantex-core/src/search/mod.rs` (declare the module)

The stage encapsulates the merge logic that currently lives inline in `hybrid.rs`: run `graph_propagation::propagate` on the top `fetch_count` seeds, fetch the newly-discovered chunks, update `fused` scores (keep max), append new entries, re-sort, and return the set of newly-introduced chunk ids (for `GraphExpanded` tagging). Keeping it in its own function makes it unit-testable against a real `ChunkStore` and makes the `hybrid.rs` boost chain read as a single call.

- [ ] **Step 1: Confirm the real `ChunkStore` graph + chunk APIs (VERIFY, no commit)**

Run:
```bash
grep -n "pub fn get_call_edges_from\|pub fn get_call_edges_to\|pub fn get_chunks\|pub fn get_type_ref_edges\|pub fn get_hierarchy_edges_for" crates/semantex-core/src/index/storage.rs
```
Expected (current `main`):
```
get_call_edges_from(&self, caller_ids: &[u64]) -> Result<Vec<(u64, u64)>>   // (caller, callee)
get_call_edges_to(&self, callee_ids: &[u64]) -> Result<Vec<(u64, u64)>>     // (callee, caller)
get_type_ref_edges_to_defs(&self, def_chunk_ids: &[u64]) -> Result<Vec<(u64, u64)>>
get_type_ref_edges_from_usages(&self, usage_chunk_ids: &[u64]) -> Result<Vec<(u64, u64)>>
get_hierarchy_edges_for(&self, chunk_ids: &[u64]) -> Result<Vec<(u64, u64)>>
get_chunks(&self, ids: &[u64]) -> Result<Vec<Chunk>>
```
These are what `propagate` and the stage call. Record nothing new — this step just confirms the signatures the stage code below relies on.

- [ ] **Step 2: Write failing tests**

`crates/semantex-core/src/search/graph_stage.rs` (test module at bottom; the production code comes in Step 4). For now, write the file with ONLY the test module so it fails to compile against the missing `run_graph_stage`:

```rust
//! Named post-fusion pipeline stage: code-graph expansion.
//!
//! Wraps `graph_propagation::propagate` with the merge bookkeeping that the
//! hybrid boost chain needs: run propagation on the top seeds, pull in the
//! newly-discovered chunks, update fused scores (keep max), append new entries,
//! re-sort, and report which chunk ids were newly introduced so the caller can
//! tag them `SearchSource::GraphExpanded`.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::storage::ChunkStore;
    use crate::search::graph_propagation::GraphPropagationConfig;
    use crate::types::ScoredChunkId;
    use std::collections::HashMap;

    // Builds a 3-chunk in-memory index: chunk 1 calls chunk 2; chunk 3 unrelated.
    // Returns an opened ChunkStore over a temp dir.
    fn store_with_call_edge(dir: &std::path::Path) -> ChunkStore {
        crate::search::graph_stage::test_support::build_call_edge_store(dir)
    }

    #[test]
    fn test_stage_pulls_in_callee_and_tags_it_new() {
        let tmp = tempfile::tempdir().unwrap();
        let store = store_with_call_edge(tmp.path());

        // Seed only chunk 1 (the caller). Chunk 2 (callee) is NOT a seed.
        let mut fused = vec![ScoredChunkId::new(1, 10.0)];
        let mut chunk_map: HashMap<u64, crate::types::Chunk> = HashMap::new();
        for c in store.get_chunks(&[1]).unwrap() {
            chunk_map.insert(c.id, c);
        }

        let config = GraphPropagationConfig::localization_mode(10);
        let new_ids = run_graph_stage(&mut fused, &mut chunk_map, &store, &config, 10).unwrap();

        // Callee (chunk 2) must now be present and flagged new.
        assert!(fused.iter().any(|s| s.chunk_id == 2), "callee not merged");
        assert!(new_ids.contains(&2), "callee not reported as graph-expanded");
        assert!(chunk_map.contains_key(&2), "callee chunk not fetched into map");
        // Seed keeps its original (higher) score and stays ranked first.
        assert_eq!(fused[0].chunk_id, 1);
        assert!((fused[0].score - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_stage_noop_when_disabled() {
        let tmp = tempfile::tempdir().unwrap();
        let store = store_with_call_edge(tmp.path());
        let mut fused = vec![ScoredChunkId::new(1, 10.0)];
        let mut chunk_map: HashMap<u64, crate::types::Chunk> = HashMap::new();
        for c in store.get_chunks(&[1]).unwrap() {
            chunk_map.insert(c.id, c);
        }
        let mut config = GraphPropagationConfig::localization_mode(10);
        config.disabled = true;
        let new_ids = run_graph_stage(&mut fused, &mut chunk_map, &store, &config, 10).unwrap();
        assert!(new_ids.is_empty());
        assert_eq!(fused.len(), 1);
    }
}
```

> **Note on the test helper:** `run_graph_stage` operates over a real `ChunkStore`, so the test needs a tiny store with one resolved call edge. Implement `test_support::build_call_edge_store` in Step 4 alongside the production function (guarded by `#[cfg(test)]`), reusing `ChunkStore`'s existing insert/build path. If building a `ChunkStore` from scratch in a unit test proves heavy, **fall back** to the lighter contract test already proven in `graph_propagation.rs` (`test_merge_logic_seeds_keep_original_scores`) and instead exercise `run_graph_stage` end-to-end in Task S4.4's integration test against one of the 6 indexed repos. Pick whichever the existing `storage.rs` test utilities support; do not invent a `ChunkStore` constructor that does not exist — check `crates/semantex-core/src/index/storage.rs` tests first:
> ```bash
> grep -n "fn open\|fn create\|ChunkStore::\|#\[cfg(test)\]\|fn.*-> ChunkStore\|tempdir" crates/semantex-core/src/index/storage.rs | head
> ```

- [ ] **Step 3: Run — expect failure**

Run: `cargo test -p semantex-core graph_stage::tests::test_stage_pulls_in_callee_and_tags_it_new`
Expected: **FAIL to compile** — `cannot find function 'run_graph_stage' in this scope` (and the `test_support` helper unresolved).

- [ ] **Step 4: Implement `run_graph_stage` (and the test support helper)**

Replace the file header doc-comment region of `crates/semantex-core/src/search/graph_stage.rs` with the production function above the test module:

```rust
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
    let prop_scores: HashMap<u64, f32> =
        expanded.iter().map(|s| (s.chunk_id, s.score)).collect();
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

    Ok(new_ids.into_iter().collect())
}

#[cfg(test)]
pub(crate) mod test_support {
    use crate::index::storage::ChunkStore;
    use std::path::Path;

    /// Build a tiny ChunkStore with chunks 1,2,3 where chunk 1 calls chunk 2.
    /// Uses the same ChunkStore construction + graph-table insertion path the
    /// indexer uses; see storage.rs tests for the exact constructor to call.
    pub fn build_call_edge_store(_dir: &Path) -> ChunkStore {
        // IMPLEMENT using the real ChunkStore test constructor confirmed in
        // Task S4.2 Step 2's grep (e.g. ChunkStore::create / open + insert_chunk
        // + the call_graph insert helper). Wire chunk 1 -> chunk 2 as a resolved
        // call edge (callee_chunk_id = 2). Do NOT fabricate an API; match what
        // storage.rs exposes for tests.
        unimplemented!("wire to the real storage.rs test constructor")
    }
}
```

> Replace the `unimplemented!` body with the real construction using whatever `storage.rs` exposes (the Step-2 grep tells you). If `storage.rs` has no test-friendly constructor for inserting resolved call edges, **delete `test_support` and the two `graph_stage::tests` that depend on it**, keep `run_graph_stage` as the production unit, and rely on Task S4.4's integration test (against a real indexed repo) for coverage — note this choice in the commit body.

Declare the module in `crates/semantex-core/src/search/mod.rs` (add alongside the other `pub mod` lines, e.g. next to `pub mod graph_propagation;`):

```rust
pub mod graph_stage;
```

- [ ] **Step 5: Run — expect pass**

Run: `cargo test -p semantex-core graph_stage`
Expected: the `graph_stage::tests::*` pass (or, if you took the fallback, the module compiles and Task S4.4 covers it). Then `cargo build -p semantex-core` to confirm the module is wired.

- [ ] **Step 6: Commit**

```bash
git add crates/semantex-core/src/search/graph_stage.rs crates/semantex-core/src/search/mod.rs
git commit -m "$(cat <<'EOF'
feat(graph): extract run_graph_stage — named, testable code-graph fusion stage

Pulls the inline graph-propagation merge logic out of hybrid.rs into a reusable
stage: propagate top seeds, merge discovered chunks, keep-max scores, re-sort,
report newly-introduced ids for GraphExpanded tagging.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task S4.3: Wire `hybrid.rs` to the named stage + route-gated config selection (TDD)

> **Rebase gate:** still operating on the `hybrid.rs` post-fusion region. Confirm you are rebased on S1.

**Files:**
- Modify: `crates/semantex-core/src/search/hybrid.rs`

Replace the inline "Phase 6" block (the `propagate` call + the two merge blocks at lines ~861–913 on current `main`) with a single `graph_stage::run_graph_stage` call, and choose the config preset from the query route: `localization_mode` for exhaustive / feature-planning queries, `architectural_mode` for architectural queries (existing behavior), else `for_query_type`. The `new_ids` returned by the stage feeds the existing `graph_expanded_ids` set used at lines ~1038–1048.

- [ ] **Step 1: Add a route-selection unit test (TDD via a small pure helper)**

To keep the route choice testable without spinning up a full `HybridSearcher`, extract the preset choice into a free function `select_graph_config` and test it. Add to the `#[cfg(test)] mod tests` block in `hybrid.rs`:

```rust
    #[test]
    fn test_select_graph_config_localization_for_exhaustive() {
        let cfg = select_graph_config("find all usages of the retry helper", QueryType::Semantic, 20);
        assert!(cfg.enable_transitive, "exhaustive should get 2-hop");
        assert_eq!(cfg.max_propagated, 20, "exhaustive should use localization cap");
    }

    #[test]
    fn test_select_graph_config_localization_for_feature_planning() {
        let cfg = select_graph_config("where should I add rate limiting", QueryType::Semantic, 20);
        assert!(cfg.enable_transitive, "feature-planning should get 2-hop");
        assert_eq!(cfg.max_propagated, 20);
    }

    #[test]
    fn test_select_graph_config_architectural_unchanged() {
        // 4+ words + architectural signal "flow" + Semantic -> architectural_mode (transitive on).
        let cfg = select_graph_config("how does the request flow through layers", QueryType::Semantic, 12);
        assert!(cfg.enable_transitive);
        assert_eq!(cfg.max_propagated, 12); // architectural_mode uses top_k
    }

    #[test]
    fn test_select_graph_config_default_per_query_type() {
        // A plain identifier query: no route -> per-query-type config, 1-hop.
        let cfg = select_graph_config("getUserById", QueryType::Identifier, 20);
        assert!(!cfg.enable_transitive);
        assert_eq!(cfg.max_propagated, 5); // 20/4 from for_query_type(Identifier)
    }
```

- [ ] **Step 2: Run — expect failure**

Run: `cargo test -p semantex-core --lib search::hybrid::tests::test_select_graph_config_localization_for_exhaustive`
Expected: **FAIL to compile** — `cannot find function 'select_graph_config' in this scope`.

- [ ] **Step 3: Implement `select_graph_config` and re-wire Phase 6**

Add the helper near `is_architectural_query` in `hybrid.rs` (which is at line ~1515):

```rust
/// Pick the graph-propagation config preset for a query's route.
///
/// Routes (gated by repo-agnostic predicates, not new QueryType variants):
/// - exhaustive ("find all usages…") or feature-planning ("where should I add…")
///   → `localization_mode` (recall-oriented, 2-hop).
/// - architectural ("how does X flow through layers") → `architectural_mode`
///   (existing behavior, 2-hop).
/// - otherwise → `for_query_type` (1-hop, per-type decays).
///
/// `with_env_overrides` is applied last so `SEMANTEX_GRAPH_*` env knobs (incl.
/// `SEMANTEX_GRAPH_DISABLE` and `SEMANTEX_GRAPH_HOPS`) always win for tuning.
fn select_graph_config(
    query: &str,
    query_type: QueryType,
    candidates: usize,
) -> GraphPropagationConfig {
    let base = if query_classifier::is_exhaustive_query(query)
        || query_classifier::is_feature_planning_query(query)
    {
        GraphPropagationConfig::localization_mode(candidates)
    } else if is_architectural_query(query, query_type) {
        GraphPropagationConfig::architectural_mode(candidates)
    } else {
        GraphPropagationConfig::for_query_type(&query_type, candidates)
    };
    base.with_env_overrides()
}
```

Ensure `query_classifier` is in scope (it is used elsewhere in `hybrid.rs`; confirm the `use` at the top includes `query_classifier` — `classify` is already called at line ~201, so the path is available).

Now replace the inline Phase-6 block. The current block (lines ~861–913) is:

```rust
        // Phase 6: Graph propagation — expand results through code graph edges
        let graph_config = if is_architectural_query(&effective_text, query_type) {
            GraphPropagationConfig::architectural_mode(candidates).with_env_overrides()
        } else {
            GraphPropagationConfig::for_query_type(&query_type, candidates).with_env_overrides()
        };
        let scored_ids: Vec<ScoredChunkId> = fused.iter().take(fetch_count).cloned().collect();
        let expanded = graph_propagation::propagate(&scored_ids, &store, &graph_config)?;

        // Merge propagated chunks into fused list
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

        // Update scores from propagation (only if higher) and add new entries
        {
            let prop_scores: HashMap<u64, f32> =
                expanded.iter().map(|s| (s.chunk_id, s.score)).collect();
            for scored in &mut fused {
                #[allow(clippy::collapsible_if)]
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
        }
        tracing::debug!(
            expanded_count = expanded.len(),
            new_count = new_ids.len(),
            "Graph propagation complete"
        );
```

Replace it entirely with:

```rust
        // Phase 6: Graph propagation — named stage over the fused candidates.
        let graph_config = select_graph_config(&effective_text, query_type, candidates);
        let new_ids: Vec<u64> = graph_stage::run_graph_stage(
            &mut fused,
            &mut chunk_map,
            &store,
            &graph_config,
            fetch_count,
        )?
        .into_iter()
        .collect();
        tracing::debug!(
            new_count = new_ids.len(),
            disabled = graph_config.disabled,
            two_hop = graph_config.enable_transitive,
            "Graph propagation stage complete"
        );
```

Update the imports at the top of `hybrid.rs`. The current import is:

```rust
use crate::search::graph_propagation::{self, GraphPropagationConfig};
```

Keep `GraphPropagationConfig` (used by `select_graph_config`), drop the now-unused `self` (no longer calling `graph_propagation::propagate` directly), and add the stage module:

```rust
use crate::search::graph_propagation::GraphPropagationConfig;
use crate::search::graph_stage;
```

The downstream `graph_expanded_ids` set at line ~1038 already consumes `new_ids`:
```rust
        let graph_expanded_ids: HashSet<u64> = new_ids.iter().copied().collect();
```
This still compiles unchanged because `new_ids: Vec<u64>` is preserved. **Do not touch the `GraphExpanded` tagging block (~1038–1048).** Confirm `HashMap`/`HashSet` are still imported (they are used elsewhere in the function).

- [ ] **Step 4: Run — expect pass**

Run: `cargo test -p semantex-core --lib search::hybrid`
Expected: all `search::hybrid::tests::*` pass, including the four new `select_graph_config` tests. Then build: `cargo build -p semantex-core` — expected: compiles with no unused-import warnings for `graph_propagation`.

- [ ] **Step 5: Full crate test (regression guard for the refactor)**

Run: `cargo test -p semantex-core`
Expected: the entire `semantex-core` suite is green — the refactor is behavior-preserving at default config (same `propagate`, same merge math, same `GraphExpanded` tagging), so no existing test should change. If any test flips, the extraction diverged from the original inline logic — diff `run_graph_stage` against the pre-refactor block and reconcile before proceeding (use superpowers:systematic-debugging).

- [ ] **Step 6: Commit**

```bash
git add crates/semantex-core/src/search/hybrid.rs
git commit -m "$(cat <<'EOF'
refactor(hybrid): Phase 6 graph propagation → named run_graph_stage call

Replaces the inline propagate+merge block with graph_stage::run_graph_stage and
adds select_graph_config: localization_mode for exhaustive/feature-planning
routes, architectural_mode for architectural queries (unchanged), else
per-query-type. Behavior-preserving at default config; GraphExpanded tagging
untouched.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 4 — Localization expansion integration test (real index)

### Task S4.4: End-to-end stage test against a real indexed repo (TDD)

**Files:**
- Create: `crates/semantex-core/tests/graph_stage_localization.rs` (integration test)

This proves the stage actually pulls structurally-related chunks into the ranking on a real index — the behavior SWE-loc rewards. It runs against the semantex repo's own `.semantex` index (always present per project memory) or skips if absent, so unit CI stays hermetic. It is the safety net for the Task S4.2 fallback (if the unit `test_support` store was dropped).

- [ ] **Step 1: Confirm how integration tests open an existing index (VERIFY, no commit)**

Run:
```bash
ls crates/semantex-core/tests/ 2>/dev/null
grep -rn "HybridSearcher::open\|ChunkStore::open\|fn open\b" crates/semantex-core/src/search/hybrid.rs crates/semantex-core/src/index/storage.rs | head
```
Record the real `HybridSearcher::open` / `ChunkStore::open` signature(s) the test will call (path arg + config). If there are no existing integration tests under `tests/`, model the new one on the unit-test setup in `hybrid.rs`'s `#[cfg(test)] mod tests` (it constructs searchers/stores there). Do not invent an opener; use the confirmed signature.

- [ ] **Step 2: Write the integration test**

`crates/semantex-core/tests/graph_stage_localization.rs`:

```rust
//! End-to-end check that the code-graph stage surfaces structurally-related
//! chunks on a real index. Skips when no `.semantex` index is present so unit
//! CI stays hermetic.

use std::collections::HashMap;
use std::path::Path;

use semantex_core::index::storage::ChunkStore;
use semantex_core::search::graph_propagation::GraphPropagationConfig;
use semantex_core::search::graph_stage::run_graph_stage;
use semantex_core::types::{Chunk, ScoredChunkId};

/// Locate this repo's index db. Adjust the relative path via the confirmed
/// ChunkStore::open signature from Task S4.4 Step 1.
fn open_repo_store() -> Option<ChunkStore> {
    // The semantex workspace index lives at <repo>/.semantex (project memory).
    // CARGO_MANIFEST_DIR = crates/semantex-core; the workspace root is two up.
    let db = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../.semantex");
    if !db.exists() {
        return None;
    }
    // Replace `ChunkStore::open(&db)` with the real opener confirmed in Step 1.
    ChunkStore::open(&db).ok()
}

#[test]
fn graph_stage_expands_real_index_seeds() {
    let Some(store) = open_repo_store() else {
        eprintln!("skipping: no .semantex index present");
        return;
    };

    // Seed with a few real chunk ids that have outgoing call edges.
    let all_ids = store.get_all_chunk_ids().expect("chunk ids");
    // Find a seed that actually has callees so expansion is non-trivial.
    let seed_id = all_ids
        .iter()
        .copied()
        .find(|id| {
            store
                .get_call_edges_from(&[*id])
                .map(|e| !e.is_empty())
                .unwrap_or(false)
        })
        .expect("expected at least one chunk with outgoing call edges in the semantex index");

    let mut fused = vec![ScoredChunkId::new(seed_id, 10.0)];
    let mut chunk_map: HashMap<u64, Chunk> = HashMap::new();
    for c in store.get_chunks(&[seed_id]).unwrap() {
        chunk_map.insert(c.id, c);
    }

    let config = GraphPropagationConfig::localization_mode(20);
    let before = fused.len();
    let new_ids = run_graph_stage(&mut fused, &mut chunk_map, &store, &config, 20).unwrap();

    assert!(!new_ids.is_empty(), "localization stage discovered no related chunks");
    assert!(fused.len() > before, "fused list did not grow");
    // Every newly-introduced id has a fetched chunk (for GraphExpanded tagging).
    for id in &new_ids {
        assert!(chunk_map.contains_key(id), "new id {id} missing from chunk_map");
    }
    // Seed retains its top score / rank.
    assert!(fused.iter().any(|s| s.chunk_id == seed_id));
}

#[test]
fn graph_stage_disabled_is_noop_on_real_index() {
    let Some(store) = open_repo_store() else {
        eprintln!("skipping: no .semantex index present");
        return;
    };
    let all_ids = store.get_all_chunk_ids().expect("chunk ids");
    let seed_id = *all_ids.first().expect("non-empty index");
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
```

> If Step 1 showed `ChunkStore::open` takes a different argument (e.g. the `.semantex` dir vs the db file, or an extra config), fix `open_repo_store` accordingly. `get_all_chunk_ids`, `get_call_edges_from`, and `get_chunks` are confirmed real `ChunkStore` methods (storage.rs lines 425, 707, 254).

- [ ] **Step 3: Run — expect pass (or skip if no index)**

First ensure the index exists: `cargo build -p semantex-cli && ./target/debug/semantex "graph propagation" -m 1 >/dev/null 2>&1 || true` (builds the index on first call).
Run: `cargo test -p semantex-core --test graph_stage_localization`
Expected: `graph_stage_expands_real_index_seeds` and `graph_stage_disabled_is_noop_on_real_index` pass (the semantex index has rich call edges). If they print "skipping", the index is absent — build it as above and re-run.

- [ ] **Step 4: Commit**

```bash
git add crates/semantex-core/tests/graph_stage_localization.rs
git commit -m "$(cat <<'EOF'
test(graph): e2e localization stage test on a real index (skips if absent)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 5 — Tune on S0 SWE-loc; guard CoIR/CSN

### Task S4.7: Grid-tune decays + hop count on SWE-loc; verify no CoIR/CSN regression (measure-only, no crate commit)

**Files:** none modified in `crates/` (tuning is env-driven). Final tuned defaults, if they differ from the shipped presets, are applied in Task S4.8. Append to (create if first) `docs/superpowers/plans/2026-05-31-research-notes.md` all measurements under the `## S4 code-graph fusion` section (a `→ tuning` subsection); never overwrite (sibling streams share this file).

This is the measured-tuning step that satisfies spec §4 S4's acceptance gate. It uses the S0 harness exactly as in S0 Task 8.x, toggling graph behavior through the env knobs added in Task S4.1. All commands target the real harness; the comparison baselines are the numbers recorded in Task S4.0.

- [ ] **Step 1: Build the tuned binary**

```bash
cargo build -p semantex-cli
export SEMANTEX_BINARY=$(pwd)/target/debug/semantex
```
(Use this `SEMANTEX_BINARY` for every run below so the harness exercises the S4 code, not a stale installed binary.)

- [ ] **Step 2: Sweep hop count on SWE-loc (1-hop vs 2-hop, localization route)**

The SWE-loc queries are `problem_statement` text. To force every query onto the localization preset for the sweep (so the 2-hop effect is measured uniformly rather than depending on per-query route detection), set the decays explicitly and toggle hops via `SEMANTEX_GRAPH_HOPS`:

```bash
cd benchmarks/relevance && source .venv/bin/activate
for HOPS in 1 2; do
  export SEMANTEX_GRAPH_HOPS=$HOPS
  python -m scripts.run --dataset swe-loc --ablation hybrid \
    --run-id s4-hops$HOPS --k 10
done
unset SEMANTEX_GRAPH_HOPS
echo "=== hops=1 ==="; cat results/s4-hops1/report.md
echo "=== hops=2 ==="; cat results/s4-hops2/report.md
```
Record SWE-loc `recall_at_5` / `recall_at_10` / `mrr_at_10` for each. Compare to the Task S4.0 graph-off baseline.

- [ ] **Step 3: Grid-search the decays on SWE-loc**

Sweep the call/caller and type/hierarchy decays around the `localization_mode` defaults (0.35 / 0.35 / 0.25 / 0.20) with the best hop count from Step 2. Keep the grid small (CPU-bound):

```bash
cd benchmarks/relevance && source .venv/bin/activate
export SEMANTEX_GRAPH_HOPS=2   # or 1 if Step 2 favored it
for CALL in 0.25 0.35 0.45; do
  for TYPE in 0.15 0.25; do
    export SEMANTEX_GRAPH_CALL_DECAY=$CALL
    export SEMANTEX_GRAPH_CALLER_DECAY=$CALL
    export SEMANTEX_GRAPH_TYPE_DECAY=$TYPE
    export SEMANTEX_GRAPH_TRANSITIVE_DECAY=0.12
    export SEMANTEX_GRAPH_HIERARCHY_DECAY=0.20
    python -m scripts.run --dataset swe-loc --ablation hybrid \
      --run-id "s4-grid-c${CALL}-t${TYPE}" --k 10
    echo "call=$CALL type=$TYPE:"; grep -A2 "swe-loc" results/s4-grid-c${CALL}-t${TYPE}/report.md
  done
done
unset SEMANTEX_GRAPH_HOPS SEMANTEX_GRAPH_CALL_DECAY SEMANTEX_GRAPH_CALLER_DECAY \
      SEMANTEX_GRAPH_TYPE_DECAY SEMANTEX_GRAPH_TRANSITIVE_DECAY SEMANTEX_GRAPH_HIERARCHY_DECAY
```
Record the full grid (call, type, R@5, R@10, MRR@10). Pick the cell with the best SWE-loc Recall@{5,10} that also clears the no-regression guard in Step 4. **These are universal decay values, not repo-specific tuning** (CLAUDE.md): they parameterize a generic graph-walk, exactly like RRF_K.

- [ ] **Step 4: No-regression guard on CSN with the winning config**

Re-run CSN under the winning decays/hops and compare to the Task S4.0 **CSN no-regression baseline**:

```bash
cd benchmarks/relevance && source .venv/bin/activate
export SEMANTEX_GRAPH_HOPS=<winner> \
       SEMANTEX_GRAPH_CALL_DECAY=<winner> SEMANTEX_GRAPH_CALLER_DECAY=<winner> \
       SEMANTEX_GRAPH_TYPE_DECAY=<winner> SEMANTEX_GRAPH_TRANSITIVE_DECAY=0.12 \
       SEMANTEX_GRAPH_HIERARCHY_DECAY=0.20
python -m scripts.run --dataset csn --ablation hybrid --run-id s4-csn-tuned --k 10
unset SEMANTEX_GRAPH_HOPS SEMANTEX_GRAPH_CALL_DECAY SEMANTEX_GRAPH_CALLER_DECAY \
      SEMANTEX_GRAPH_TYPE_DECAY SEMANTEX_GRAPH_TRANSITIVE_DECAY SEMANTEX_GRAPH_HIERARCHY_DECAY
cat results/s4-csn-tuned/report.md
```
**Acceptance check:** CSN `mrr_at_10` / `ndcg_at_10` / `recall_at_10` must be within noise of the S4.0 CSN baseline (no net regression). SWE-loc `recall_at_5` and `recall_at_10` must be **strictly above** the S4.0 graph-off baseline. Record both deltas. If SWE-loc improves but CSN regresses, prefer a milder decay cell that holds CSN flat — the gate is "lift on SWE-loc, no regression on CoIR/CSN/in-domain," not "max SWE-loc at any cost."

- [ ] **Step 5: (If reachable) CoIR confirmation**

If the S0 CoIR loader is live on this machine (S0 Task 0.1 Step 4 may have deferred CoIR), repeat Step 4 for `--dataset coir`. If CoIR is deferred, record that the CoIR leg of the no-regression gate runs on the machine with HF access, and that CSN (the external-calibration proxy) stands in for the unit gate here.

- [ ] **Step 6: Write the tuning report (no commit — controller commits the research note)**

Append (never overwrite) the hop sweep, the decay grid, the winning config, and the two acceptance deltas (SWE-loc lift vs graph-off; CSN no-regression) to the `## S4 code-graph fusion` section of `docs/superpowers/plans/2026-05-31-research-notes.md`. **Decision recorded here drives Task S4.8.** If no cell beats graph-off without regressing CSN, record that and skip Task S4.8 (ship the refactor + on/off switch only; the spec's "promote to a measured signal" is still satisfied by the named stage + measurement, with defaults unchanged).

---

### Task S4.8: Apply the tuned defaults (only if Task S4.7 found a net-positive config) (TDD)

**Files:**
- Modify: `crates/semantex-core/src/search/graph_propagation.rs`

If Task S4.7 selected decays/hops different from the `localization_mode` defaults shipped in Task S4.1, update the preset to the winners so the gain is on by default (no env required). If S4.7 found the shipped defaults already optimal, **skip this task** (record "defaults unchanged" in the commit log of the PR, not a new commit).

- [ ] **Step 1: Update the failing test to assert the tuned values**

Edit `test_localization_mode_enables_two_hop_and_caps_propagated` (added in Task S4.1) to assert the **winning** decays from S4.7 instead of `> 0.0`. Example, if the winner was `call=0.45, type=0.25, hops=2`:

```rust
    #[test]
    fn test_localization_mode_enables_two_hop_and_caps_propagated() {
        let config = GraphPropagationConfig::localization_mode(20);
        assert!(config.enable_transitive);
        assert!((config.call_decay - 0.45).abs() < f32::EPSILON);
        assert!((config.caller_decay - 0.45).abs() < f32::EPSILON);
        assert!((config.type_ref_decay - 0.25).abs() < f32::EPSILON);
        assert_eq!(config.max_propagated, 20);
        assert!(!config.disabled);
    }
```

- [ ] **Step 2: Run — expect failure**

Run: `cargo test -p semantex-core graph_propagation::tests::test_localization_mode_enables_two_hop_and_caps_propagated`
Expected: **FAIL** — asserted tuned values do not match the Task S4.1 placeholder defaults.

- [ ] **Step 3: Apply the tuned decays to `localization_mode`**

Edit the `localization_mode` body in `graph_propagation.rs` to the winning constants from S4.7 (call/caller/type/hierarchy/transitive decays + `enable_transitive` per the winning hop count). Keep `max_propagated: top_k` and `disabled: false`.

- [ ] **Step 4: Run — expect pass**

Run: `cargo test -p semantex-core graph_propagation`
Expected: green, including the now-tightened localization test.

- [ ] **Step 5: Re-confirm the lift with the new defaults (no env overrides)**

```bash
cargo build -p semantex-cli
export SEMANTEX_BINARY=$(pwd)/target/debug/semantex
cd benchmarks/relevance && source .venv/bin/activate
python -m scripts.run --dataset swe-loc --ablation hybrid --run-id s4-final-default --k 10
python -m scripts.run --dataset csn     --ablation hybrid --run-id s4-csn-final-default --k 10
echo "=== SWE-loc (tuned defaults, no env) ==="; cat results/s4-final-default/report.md
echo "=== CSN (tuned defaults, no env) ==="; cat results/s4-csn-final-default/report.md
```
Expected: SWE-loc `recall_at_5/10` above the S4.0 graph-off baseline **with no env set** (the gain is now the default for localization-routed queries), and CSN within noise of baseline. Append the final numbers (never overwrite) to the `## S4 code-graph fusion` section of research notes.

- [ ] **Step 6: Commit**

```bash
git add crates/semantex-core/src/search/graph_propagation.rs
git commit -m "$(cat <<'EOF'
perf(graph): tune localization_mode decays + hop count on SWE-loc

Sets localization_mode to the S0-SWE-loc-tuned decays (see research notes
2026-05-31 §S4 tuning): measurable SWE-loc Recall@{5,10} lift vs graph-off with
no net CSN regression. Universal, env-overridable graph-walk parameters.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 6 — Verification & integration

### Task S4.9: Full verification + integration handoff

**Files:** none modified.

- [ ] **Step 1: Full `semantex-core` test suite**

Run: `cargo test -p semantex-core`
Expected: green (all of `graph_propagation`, `query_classifier`, `graph_stage`, `hybrid`, and the `graph_stage_localization` integration test pass or skip cleanly).

- [ ] **Step 2: Workspace build + lint + format**

```bash
cargo build --workspace
cargo clippy --all
cargo fmt --all -- --check
```
Expected: builds clean; clippy reports no new warnings (the `graph_propagation` `self` import was dropped from `hybrid.rs`, so there is no unused-import warning); `fmt --check` clean. If `fmt` complains, run `cargo fmt --all` and fold it into the relevant task's commit (do not add a bare formatting commit).

- [ ] **Step 3: Default-build LLM-dep guard (CLAUDE.md rule 8)**

S4 touches no LLM code, but run the guard to be safe:
```bash
cargo tree | grep genai || echo "OK: no genai in default build"
```
Expected: `OK: no genai in default build`.

- [ ] **Step 4: Confirm the env contract is documented for the harness/users**

Verify the five existing + two new env knobs are discoverable. Grep:
```bash
grep -rn "SEMANTEX_GRAPH_DISABLE\|SEMANTEX_GRAPH_HOPS\|SEMANTEX_GRAPH_CALL_DECAY" crates/semantex-core/src/search/graph_propagation.rs
```
Expected: all of `SEMANTEX_GRAPH_DISABLE`, `SEMANTEX_GRAPH_HOPS`, and the five `*_DECAY` knobs present in `with_env_overrides`. (If the project keeps an env-var reference doc, add the two new knobs there; otherwise the inline doc-comments on `with_env_overrides` suffice — do not invent a new doc file per CLAUDE.md's "no proactive docs" rule.)

- [ ] **Step 5: Integration handoff note (no commit — for the controller)**

Summarize for the integration controller: the SWE-loc lift vs graph-off (from S4.7/S4.8), the CSN no-regression delta, the final `localization_mode` defaults, and the new env knobs. Note that **S4 is backend-agnostic** — it operates on fused candidates, so the same lift should hold once S2's `coderank-hnsw` is the dense backend (re-measure during the Phase-3 cutover A/B per spec §5). Flag that S4's `hybrid.rs` edits were rebased on S1 and live in a distinct region from S7's fusion edits (so a clean merge is expected, but a golden-output check after merge is prudent per spec §5).

---

## Coverage vs spec §4 S4

- **(1) Named, config-tunable stage:** `graph_stage::run_graph_stage` (Task S4.2) + the `hybrid.rs` rewrite to a single stage call (Task S4.3). Decays are env-tunable via the pre-existing `SEMANTEX_GRAPH_*_DECAY` knobs (verified present in `with_env_overrides`), now joined by `SEMANTEX_GRAPH_HOPS` and the `SEMANTEX_GRAPH_DISABLE` on/off switch (Task S4.1).
- **(2) Optional 2-hop, gated per query-route:** `is_exhaustive_query` / `is_feature_planning_query` predicates (Task S4.6) + `select_graph_config` route selection (Task S4.3), choosing `localization_mode`/`architectural_mode` (both 2-hop) vs `for_query_type` (1-hop). `SEMANTEX_GRAPH_HOPS` overrides for tuning.
- **(3) Localization expansion + GraphExpanded tag:** `localization_mode` preset (Task S4.1) drives a recall-oriented 1–2-hop expansion of the top fused seeds along call/import/type/hierarchy edges; the existing `SearchSource::GraphExpanded` tagging (`hybrid.rs` ~1038–1048) is preserved and fed by the stage's `new_ids` (Task S4.3); proven end-to-end on a real index (Task S4.4).
- **(4) Tune on SWE-loc, no CoIR/CSN regression:** Task S4.7 (grid/hop sweep against the S0 harness) + Task S4.8 (apply winners). Baselines captured in Task S4.0.
- **Acceptance gate:** measurable SWE-loc Recall@{5,10} lift vs graph-off (Task S4.0 baseline vs S4.7/S4.8 tuned), no net regression on CSN (and CoIR where reachable) — checked in Tasks S4.7 Step 4 / S4.8 Step 5.
- **Backend-agnostic:** the stage consumes `Vec<ScoredChunkId>` + `ChunkStore` graph tables only; no dense-backend coupling (Task S4.9 Step 5 handoff note).

## CLAUDE.md compliance

- **No hardcoded repo-specific graph rules:** the route predicates use universal English signal words only (Task S4.6); the decays are generic graph-walk parameters (like `RRF_K`), env-overridable, tuned on external benchmarks (SWE-loc/CSN), not on any one repo.
- **No hardcoded paths:** the only path is the integration test's `CARGO_MANIFEST_DIR/../../.semantex` (test-only, derived from the build dir, not an absolute `/Users/...` path); `benchmarks/` env-driven tuning is exempt.
- **No new deps; default build stays zero-LLM** (Task S4.9 Step 3).
