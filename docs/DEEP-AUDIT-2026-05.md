# `semantex_deep` Audit — Item 5, Spec Q (v0.5)

**Date:** 2026-05-26
**Author:** subagent W-Gamma (v0.5)
**Spec reference:** `docs/superpowers/specs/2026-05-26-semantex-v0.3.1-v0.5-refactor.md` §5 Item 5
**Verdict:** Item 5 root-cause **confirmed**. None of the spec's three pre-listed candidate fixes (a/b/c) match the actual cause. The chosen fix is a fourth, principled alternative — a **noise-floor rescue pass** in `triage_results` — described in §3 below.

---

## 1. Method

### 1.1 Files diffed / read
- `benchmarks/results/run22/raw/gin/Q2_treatment.jsonl` — v0.2 baseline (2026-03-08).
- `benchmarks/results/run25/raw/gin/Q2_treatment.jsonl` — Phase 4, current `main`.
- Spot-checked the same comparison on: `platform/Q2`, `flask/Q3`, `gin/Q5`, `platform/Q5` (the four other cells from §2.5 of the spec).
- Also spot-checked `run23` and `run24` for the same `gin/Q2` cell to find when the regression first appeared.

### 1.2 Code paths inspected
- `crates/semantex-core/src/search/deep.rs::deep_search_inner` — lines 616-847 (the full deep pipeline).
- `crates/semantex-core/src/search/deep.rs::triage_results` — lines 136-244 (the triage filter).
- `crates/semantex-core/src/search/deep.rs::CONFIDENCE_NOISE_FLOOR` — line 16 (the constant 0.08).
- `crates/semantex-core/src/search/triple_fusion.rs::triple_fusion_weights` — lines 131-163 (max_possible per query type).
- `crates/semantex-core/src/search/summarize.rs::extractive_summarize` — lines 134-178 (the no-chunks "No relevant code found" message).
- `git show fd267c1` — the commit that introduced the noise floor.

### 1.3 Hypothesis (from spec) vs. data
The spec hypothesis was: "in v0.2 `semantex_deep` returned one comprehensive response; in Phase 4 it returns a smaller response followed by follow-ups." **Confirmed quantitatively** (see §2). But the spec then suggests three possible causes — (a) fewer sources, (b) terse synthesis, (c) E3 annotation noise. **None of those match the data.** The data shows the deep search is returning literally "No relevant code found for this query." even though search retrieved 7–32 candidates. The problem is a hard filter dropping 100% of candidates, not synthesis size or weighting.

---

## 2. Quantitative summary

### 2.1 gin/Q2 (the spec's primary investigation cell)

| Metric | run22 (v0.2) | run25 (Phase 4) | Δ |
|---|---|---|---|
| `semantex_*` tool calls | 3 | 1 | −2 |
| Deep sub-queries issued | 6 | 4 | −2 |
| "No relevant code" sub-responses | 0 | 4 | +4 (i.e. 100% of sub-queries returned empty) |
| First semantex response chars | 14,790 | 417 | −97% |
| Total chunks `searched` (sum over sub-queries) | 67 | 32 | −52% |
| Total chunks `read` (sum over sub-queries) | 65 | 0 | **−100%** |
| Total non-semantex tool calls (Read/Grep/Glob fallback) | 4 | 17 | +13 |
| Total tool-call count (everything) | 7 | 18 | +157% |

The key cell is `read = 0` despite `searched = 32`. The deep pipeline received candidates from hybrid search but its triage filter dropped every single one of them.

### 2.2 Cross-cell consistency

| Repo / Q | run22 searched / read | run25 searched / read | run25 `read = 0`? |
|---|---|---|---|
| gin Q2 | 67 / 65 | 32 / 0 | yes |
| platform Q2 | 47 / 67¹ | 26 / 0 | yes |
| flask Q3 | 23 / 36¹ | 29 / 0 | yes |
| platform Q5 | 0 / 0² | 23 / 0 | yes |
| gin Q5 | 50 / 55¹ | 0 / 0³ | n/a |

¹ `read` > `searched` is possible because deep's graph-expansion phase adds extra IDs after the search phase.
² Agent in run22 didn't call semantex for platform Q5 — used pure Grep/Read.
³ Agent in run25 didn't call semantex for gin Q5 — bailed early to Grep/Read after seeing the empty results on other queries (signal cascade).

The pattern is **fully consistent**: every cell where the agent did call `semantex_deep` in run25 shows `read = 0` despite `searched > 0`. This is a single-cause regression.

### 2.3 When did it first appear?

| Run | gin/Q2 sub_queries | gin/Q2 searched | gin/Q2 read |
|---|---|---|---|
| run22 (v0.2, 2026-03-08) | 6 | 67 | 65 |
| run23 (v0.3 visible-tools) | 0 | 0 | 0 |
| run24 (Phase 3) | 4 | 26 | **0** |
| run25 (Phase 4) | 4 | 32 | **0** |

run23 doesn't reach the deep path at all (visible-tools regression). run24 is the first run where the agent calls deep but `read = 0`. So the regression slipped in somewhere between v0.2 (run22) and Phase 3 (run24).

---

## 3. Root cause

`git log` on `crates/semantex-core/src/search/deep.rs` between v0.2 and current `main` shows commit `fd267c1` ("Implement Cardinal-inspired enhancements: confidence zones, index validation, symbol shortcut, per-channel scores", 2026-03-03) introduced two filters in `triage_results`:

1. **Noise floor (line 157–163 of current deep.rs):**
   ```rust
   if max_possible > 0.0 {
       let normalized = result.score / max_possible;
       if normalized < CONFIDENCE_NOISE_FLOOR {  // 0.08
           continue;
       }
   }
   ```
2. **Consensus bonus** (additive only; not the cause — included for completeness).

The noise floor uses `query_type.triple_fusion_weights().max_possible()` as the denominator. For a `Semantic` query (which "error handling patterns propagation recovery" classifies as), `max_possible = 0.4 + 0.5 + 0.8 = 1.7`. The 0.08 floor therefore demands raw fused score ≥ `0.136`.

In small, focused repos like `gin` (118 files), and on cross-cutting queries with no single dominant match, fused scores cluster in the 0.05–0.13 band. Every candidate from hybrid search falls below 0.136 → every candidate is dropped → `combined_chunks` is empty → `extractive_summarize` is called with `chunks: &[]` (summarize.rs:136) → it returns the sentinel `"No relevant code found for this query."`. The wire response goes back with `chunks_searched > 0` but `chunks_read = 0`, and `answer = "No relevant code found..."`.

This is **not** any of the three causes the spec hypothesized:
- **Not (a) — fewer sources per call.** `max_results.max(20)` is unchanged from v0.2 (line 643 of current deep.rs). The search phase is fine.
- **Not (b) — synthesis too terse.** The synthesis budget (`MAX_LEN = 16_000` in summarize.rs:135) is unchanged from v0.2.
- **Not (c) — E3 annotation noise.** E3 annotations contribute through `nl_summary` matching, which raises BM25 scores; if anything they should raise candidates above the floor, not push them below.

The root cause is **the noise floor itself is mis-calibrated to small-repo / cross-cutting-query distributions**. For a healthy result set (many high-confidence matches), the floor correctly filters bottom-feeders. For a query against a small repo where the topic is genuinely diffuse, the floor filters everything.

---

## 4. Chosen fix — Rescue pass on empty triage

**Implementation:** A single `if scored.is_empty()` branch inside `triage_results`, after the noise-floor + text-window filter. If the filter passed zero candidates AND the input `results` was non-empty, redo Pass 1 without the noise floor (text-window filter still applies). The downstream per-file cap + overlap dedup + `max_chunks` cap still bound output size; nothing else changes.

This is option **(a) — but for a different reason than the spec listed.** The spec said "Deep is returning fewer sources per call now. Fix: bump max_sources from current value to 25-30." The data shows max_sources is fine; the count being returned is zero. The rescue pass restores 1..N sources (where N is the prior `max_chunks` triage cap, currently 5/8/14 by confidence band) in the cases where the noise floor would have returned zero, without changing the count in cases where it returns ≥ 1.

### Why this fix and not a lower constant

- A constant (e.g. lower `CONFIDENCE_NOISE_FLOOR` to 0.02) is arbitrary and tunable to a specific bench — violates `CLAUDE.md` hard rule #2 ("no test-repo metadata") and #4 ("avoid overly generic" tuning).
- A relative noise floor (e.g. "drop scores below X% of top score") changes behavior on every query and risks regressing the cases where the absolute floor was correctly filtering bottom-feeders.
- The rescue pass changes behavior *only* in the regression case (filter would have returned zero) and leaves every other case bit-identical. It is the smallest possible change with measurable scope.

### Why this satisfies CLAUDE.md hard rules

- No hardcoded paths. (Hard rule #1.)
- No repo-specific tuning — applies to any codebase. (Hard rule #2.)
- Touches no synonym table. (Hard rule #3, #4.)
- No benchmark-ground-truth leakage. (Hard rule #5.)

### What gets measured

Per the spec's acceptance criteria (§5 Item 5): gin/flask/qgrep Q2 + Q5 (4 cells). The fix targets the mechanism that's currently zeroing those cells. The benchmark run (run29-v0.5) will measure the magnitude of the recovery, but the unit test in `deep.rs` (see §5 below) asserts the *mechanism*: synthetic results with all-below-noise-floor scores now yield non-empty triage output, where previously they yielded empty.

---

## 5. Test added

`crates/semantex-core/src/search/deep.rs::test_triage_noise_floor_rescue_on_total_filter`:

- Synthetic input: 3 AstNode results, all with `score = 0.05` and `max_possible = 1.0` → normalized 0.05, all below `CONFIDENCE_NOISE_FLOOR = 0.08`.
- Pre-fix expectation: triage returns `[]`.
- Post-fix expectation: triage returns the top-K (bounded by `max_chunks`).

This test would have caught the regression. It does not depend on any specific repo or benchmark.

---

## 6. Item 8 (cross-source dedup) — implementation note

Item 8 lives in the same workstream (W-Gamma). The dedup pass groups selected `SearchResult`s by `(file_path, start_line, end_line)` BEFORE the chunk lift. Highest-score wins; dropped entries' channel labels (Dense/Sparse/Exact) are aggregated into a new internal field `DeepSource::also_matched_via: Vec<String>`.

The wire-format struct `DeepSearchResponse` (owned by W-Delta in `protocol.rs`) is NOT modified — surface to the agent goes through a "Also matched via:" footer appended to `DeepResult::answer` for the deduped sources. The deduped set, not the original set, is fed into `extractive_summarize`, so the synthesis output is smaller without information loss (since the dropped duplicates pointed at the same `(file, start, end)`).

---

## 7. Out-of-scope follow-ups (not part of this fix)

- Re-evaluating whether `CONFIDENCE_NOISE_FLOOR = 0.08` is the right constant for healthy result sets is deferred. The rescue pass leaves the constant intact and only short-circuits the total-filter case. If `run29-v0.5` shows the floor is still too aggressive in mixed cases (`searched = 32, read = 1` rather than `read = 0`), a follow-up spec can revisit the constant.
- The "low confidence" warning at line 826–834 of `deep_search_inner` still fires for the rescued path — appropriately, since the rescued results genuinely *are* low-confidence. The agent will see results + warning, not empty + warning.
- Consensus bonus, kind boost, docstring boost — all left untouched. They influence ranking inside `scored`, not whether `scored` is empty.
