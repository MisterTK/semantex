# Release Sequence — v0.3.1 → v0.4 → v0.5 → v0.6

**Date:** 2026-05-26
**Status:** Authoritative ordering for two-spec parallel refactor
**Source specs:**
- `docs/superpowers/specs/2026-05-26-semantex-v0.3.1-v0.5-refactor.md` (quality/benchmark — "Spec Q")
- `docs/v0.4_SPEC.md` (deps & capabilities — "Spec D")

This document resolves the v0.4 label collision between the two specs above, locks the file-ownership order across them, and amends two acceptance gates that are unsafe as written. Subagent dispatch MUST NOT begin until this doc is read and §5's pre-flight is green.

---

## 1. Version remap (authoritative)

| Release | Source | Scope | Bench artifact |
|---|---|---|---|
| **v0.3.1** | Spec Q Tier 1 (Items 1, 2, 3, 4) | Adaptive output budgets, multi-language classifier, deep_with_examples trim, small-repo arch override | `run27-v0.3.1/` (gin + flask + platform subset) |
| **v0.4** | Spec D (all 18 items, 5 workstreams) | Dep upgrades (bincode→postcard, axum 0.8, rusqlite 0.40, tantivy 0.26), PLAID 1.3 wiring, colgrep ranking signals, C# 10 / Scala 3 / Vue / OCaml chunking | `run28-v0.4/` (full 6 repos) |
| **v0.5** | Spec Q Tier 2 (Items 5, 6, 7, 8) | Deep audit, confidence-driven disambiguation, adaptive structural walk, deep dedup | `run29-v0.5/` (full 6 repos) |
| **v0.6** | Spec Q Tier 3 (Items 9, 10, 11, 12) | Local LLM classifier + HyDE, internal multi-step planner, cross-repo daemon (conditional), index-time LLM enrichment | `run30-v0.6/` |
| (strategic) | Spec Q Tier 4 (Items 13–16) | Quality benchmarks, prebuilt indexes, streaming MCP, web UI | n/a |

Anything still saying "Tier 2 = v0.4" or "Tier 3 = v0.5" in Spec Q has been remapped; the spec header has been amended. Disregard older copies.

---

## 2. Execution order

```
v0.3.1 (Spec Q Tier 1) ──→ bench run27 ──→ v0.4 (Spec D, all WS)  ──→ bench run28 ──→ v0.5 (Spec Q Tier 2) ──→ bench run29 ──→ v0.6 (Spec Q Tier 3)
       1-2 days, $15            ~6 h wall (5 parallel WS), $0       1-2 weeks, $80                       1-2 months, $80
```

Rationale:
1. **v0.3.1 first** — small (4 items), surgical, closes the 7 known regressions documented in `docs/BENCHMARK-v0.3-PHASE4-RESULTS.md`. Validates the bench harness before larger surgery and gives Spec D a known-good baseline.
2. **v0.4 second** — three of its items are unavoidable maintenance (bincode poison-pill, axum 0.7 panics, tantivy 0.26 compile break). Several items (colgrep ranking signals) move score composition; doing them before Spec Q Tier 2 means Tier 2 is designed against the real post-ranking baseline.
3. **v0.5 and v0.6** — re-evaluate against `run28-v0.4`. Some Tier 2 items may shrink or grow once Spec D's ranking signals are in.

No step in this sequence runs in parallel with another step. Within v0.4, the 5 workstreams run in parallel per Spec D §3.2.

---

## 3. File-ownership conflicts between Spec Q and Spec D

The two specs overlap on three files. Spec D wins the first two (it lands first); Spec Q wins the third (Spec D doesn't touch it).

### 3.1 `crates/semantex-core/src/server/protocol.rs`

| Owner | Item | Change |
|---|---|---|
| Spec D | Item 2 (bincode → postcard) | Sweeping rewrite of `encode_*`/`decode_*` functions; changes wire format payload |
| Spec Q | W-Delta Item 6 (disambiguation field) | Adds `disambiguation: Option<Vec<DisambigSuggestion>>` to `SearchResponse` / `AgentResponse` |

**Rule:** Spec D Item 2 MUST land before Spec Q Item 6 begins. The disambiguation field is defined as serde-derived, so it transports through postcard without further work. Spec Q W-Delta MUST verify on the v0.4 branch that postcard is the active codec before adding the field.

### 3.2 `crates/semantex-core/src/search/hybrid.rs`

| Owner | Item | Change |
|---|---|---|
| Spec D | W-C Items 6–9 | Adds 4 new score-boost phases (3b path penalty, 3c stem boost, 3d definition boost, 3e file coherence) after existing Phase 3 |
| Spec Q | (read-only) | No writes |

No write conflict, but **measurement conflict**: Spec Q Tier 1 (Items 1, 2, 4) targets the same regressions Spec D's ranking signals partially address (gin Q1 arch, platform Q2 error_handling). Tier 1 measures against run25; Tier 2/3 measure against `run28-v0.4` per §1. This is fine as long as the order in §2 is followed.

### 3.3 `crates/semantex-core/src/search/agent.rs`, `index/architecture.rs`, `agent_classifier.rs`, `deep.rs`, `agent_formatter.rs`

Owned exclusively by Spec Q workstreams. Spec D does not touch these.

### 3.4 `crates/semantex-core/src/search/sparse_search.rs`

Spec D internal: W-A Item 5 (tantivy 0.26) MUST land before W-E Item 18 (configurable stemmer). Already noted in Spec D §3.3.

### 3.5 `crates/semantex-core/src/index/builder.rs`

Spec D internal: W-B Item 13 (use returned doc IDs) MUST land before W-B Item 14 (buffer_size). Already noted in Spec D §3.3.

---

## 4. Amended acceptance gates

### 4.1 Spec D §10 — loosen "every cell neutral-or-better"

Spec D §10 currently requires CCB neutral-or-better for every (repo × question_type) cell vs run25. Run-to-run noise on this benchmark is ±5–10pp per cell (see MEMORY signal-quality table), so a strict per-cell gate will be tripped by noise on most runs.

**Amended gate:** Aggregate CCB Δ on `run28-v0.4` MUST be neutral-or-better than `run27-v0.3.1` (not run25). No single cell may regress by more than **10pp** vs `run27-v0.3.1`. Cells already-regressing in `run27-v0.3.1` are exempt — Spec D is not required to fix Spec Q's open work.

### 4.2 Spec Q Item 2 — investigation-first, gate softened

Spec Q Item 2 (multi-language classifier) is the only Tier 1 item that self-admits the regression may not be in the classifier at all ("the regression may actually be in the Deep handler's behavior on multi-language repos, not the classifier").

**Amended Tier 1 gate for Item 2:**
1. W-Beta MUST run the classifier on the exact Q2 wording from `benchmarks/agent_bench.py::QUESTIONS` and record which route fires, before writing any production code.
2. If the classifier already routes to `Deep` for platform Q2, W-Beta MUST open `coordination_request.md` rather than implementing the spec's `detect_languages` override — the regression source is elsewhere (likely Deep handler), and Tier 2 Item 5 already owns that work.
3. If the classifier routes to `Structural`, W-Beta proceeds with Item 2 as written, and the platform Q2 ≤+20% bench gate stands.
4. If the investigation outcome doesn't fit either branch above, W-Beta MUST escalate via `coordination_request.md`. Do NOT invent a new override path.

### 4.3 Bench cadence

- `run27-v0.3.1/` — subset (gin + flask + platform), runs after Spec Q Tier 1 integration branch builds clean. Gate: Items 1, 3, 4 per their per-cell targets; Item 2 per §4.2.
- `run28-v0.4/` — full 6 repos, runs after all 5 Spec D workstreams merge and the integration branch is green. Gate: §4.1.
- `run29-v0.5/` — full 6 repos, runs after Spec Q Tier 2 integration. Gate: aggregate CCB Δ ≤ −35% vs `run28-v0.4`.

Each bench artifact lives under `benchmarks/results/<name>/` and the comparison script is regenerated from MEMORY notes.

---

## 5. Pre-flight checklist (before dispatching any subagent)

- [ ] Spec Q header reads "v0.3.1 → v0.6", Tier 2 reads "v0.5", Tier 3 reads "v0.6+" (already applied in this edit).
- [ ] This file exists at `docs/RELEASE-SEQUENCE-2026-05.md` and is committed to `main`.
- [ ] No other branch claims the v0.4 label.
- [ ] `main` is at the Phase 4 commit (`446b720`) or later, with `run25` results present at `benchmarks/results/run25/`.
- [ ] Each subagent dispatch prompt includes the line: "Read `docs/RELEASE-SEQUENCE-2026-05.md` §3 (file-ownership) and §4 (amended gates) before reading the spec body."
- [ ] Worktree-leakage prevention from `~/.claude/projects/.../memory/parallel-subagent-worktree-leakage.md` is in effect: each subagent dispatched with `isolation: worktree`, integration checkout reset clean before merging branches in.

---

## 6. Out of scope for this coordination doc

- Tier 4 items (Spec Q Items 13–16) — not on the critical path; sequence with v0.6+ at the team's discretion.
- `docs/v0.4-future_SPEC.md` (referenced in Spec D §13) — that document does not exist yet; do not create it as part of v0.4. The deferred items live in Spec D §13's table only.
- Index format / schema version changes — neither spec bumps the schema; if v0.5 or v0.6 needs one, write a separate migration spec.
- Cross-repo daemon (Spec Q Item 11) — gated on a downstream consumer request, not on this sequence.

---

## 7. Changelog

- **2026-05-26** — Initial coordination doc. Resolves v0.4 label collision (Spec D wins, Spec Q tiers shift by one). Locks `protocol.rs` order (postcard → disambiguation). Loosens Spec D §10 per-cell gate; adds investigation gate to Spec Q Item 2.
