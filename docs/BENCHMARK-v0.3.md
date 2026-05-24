# semantex v0.3 — Benchmark Results

**Status:** TEMPLATE — fill in after the real sweep completes.
**Spec:** `docs/superpowers/specs/2026-05-24-semantex-v0.3-sota-design.md` §4 + §4.7.
**Scaffolding:** `benchmarks/run_public.py`, `benchmarks/latency_bench.py`,
`benchmarks/v0_3/`, `benchmarks/datasets/nl2code-v1/`.

---

## 0. Cap-condition checklist — MARK BEFORE WRITING WINS

Per spec §4.7. Each box must be checked (or left unchecked with a note) BEFORE
the wins table below is filled. If any condition trips and the wins claim
isn't downgraded accordingly, this release is shipping a false SOTA claim.

This checklist is the structural guarantee from spec §6.4 Barrier 2 — the
"honest cap conditions baked in" requirement.

- [ ] **B1 cap (retrieval):** semantex does NOT lose on >2 categories in B1.
      If it loses on >2 → claim is capped at "SOTA on agent tasks, competitive on retrieval".
      → Result: <!-- TO FILL: passes / capped (which condition?) -->
- [ ] **B3 cap (agent CCB):** average CCB reduction is ≥−50%.
      If <−50% → claim is capped at "competitive with Claude Code built-ins, wins on specific question types".
      → Result: <!-- TO FILL: -X% measured, passes / capped -->
- [x] **B4 cap (cold start):** p50 cold start ≤400 ms.
      If >400 ms → claim is capped at "wins on warm latency".
      → Result: **CAPPED** — initial measurement on the semantex repo (n=3, see §4.1):
      cold p50 = 1358 ms (4.5× the 300 ms target, 3.4× the 400 ms cap).
      Cold-start claim downgraded; v0.3 ships with "wins on daemon-ready time"
      and "deferred model load tradeoff" (lazy ONNX moves the cost from daemon-startup
      into the first inference). Real-sweep replication on >1 repo required before
      tagging v0.3.0 — controller flagged this for human review.
- [ ] **B3 variance cap (per-row):** no B3 row has SD > 30% of mean across reps.
      Any rows that trip this are declared "noise-dominated, no claim" and EXCLUDED from wins.
      → Number of rows flagged: <!-- TO FILL: N out of M -->
- [ ] **Honest losses section drafted** by a different actor from the wins section
      (spec §6.7). This rule prevents narrative drift; do not bypass.
      → Wins author: <!-- TO FILL: handle -->
      → Losses author: <!-- TO FILL: handle (MUST differ from wins author) -->

---

## 1. Top-line leaderboard

<!-- TO FILL: replace with actual numbers from the sweep -->

| Benchmark | semantex v0.3 | ripgrep 14.x | graphify 0.8.x | Claude Code built-ins | Best dense baseline | Best hybrid baseline | Winner |
|---|---|---|---|---|---|---|---|
| B1 — CSN adversarial (MRR@10) | <!-- TO FILL --> | n/a | <!-- TO FILL --> | n/a | <!-- TO FILL --> | <!-- TO FILL --> | <!-- TO FILL --> |
| B1 — CoSQA (MRR@10) | <!-- TO FILL --> | n/a | <!-- TO FILL --> | n/a | <!-- TO FILL --> | <!-- TO FILL --> | <!-- TO FILL --> |
| B1 — CodeXGLUE (MRR@10) | <!-- TO FILL --> | n/a | <!-- TO FILL --> | n/a | <!-- TO FILL --> | <!-- TO FILL --> | <!-- TO FILL --> |
| B1 — SWE-bench-Lite (Recall@10) | <!-- TO FILL --> | n/a | <!-- TO FILL --> | n/a | <!-- TO FILL --> | <!-- TO FILL --> | <!-- TO FILL --> |
| B2 — Owned NL→code (F1) | <!-- TO FILL --> | <!-- TO FILL --> | <!-- TO FILL --> | n/a | <!-- TO FILL --> | <!-- TO FILL --> | <!-- TO FILL --> |
| B3 — Agent CCB reduction vs Claude builtins | <!-- TO FILL: −X% --> | n/a | <!-- TO FILL --> | baseline | n/a | n/a | <!-- TO FILL --> |
| B4 — p50 warm latency (ms) | <!-- TO FILL --> | <!-- TO FILL --> | <!-- TO FILL --> | n/a | <!-- TO FILL --> | <!-- TO FILL --> | <!-- TO FILL --> |
| B4 — p50 cold-start (ms) | <!-- TO FILL --> | n/a | <!-- TO FILL --> | n/a | n/a | n/a | <!-- TO FILL --> |
| B5 — Head-to-head win rate (B3 questions) | <!-- TO FILL --> | n/a | <!-- TO FILL --> | n/a | n/a | n/a | <!-- TO FILL --> |

---

## 2. Methodology

* **Spec:** see `docs/superpowers/specs/2026-05-24-semantex-v0.3-sota-design.md` §4.
* **Reproducer:** `python3 benchmarks/run_public.py --tool=all --benchmark=all --output=results/v0.3/`
* **Competitor versions:** see `benchmarks/COMPETITORS.md` for pins.
* **Datasets:** see `benchmarks/datasets/<name>/MANIFEST.json` for SHA pinning.
* **Statistical methodology:** paired t-test with n≥3, alpha=0.05. Effect sizes
  reported with 95% CIs. Rows with p>=0.05 are reported as "not significant" and
  NOT counted as wins.
* **Per-spec §4.7 cap conditions:** see §0 of this doc; checked before §1 wins
  are filled.

---

## 3. Raw data

* `benchmarks/results/v0.3/sweep-<YYYYMMDD>/SUMMARY.json` — top-line summary.
* `benchmarks/results/v0.3/sweep-<YYYYMMDD>/b{1,2,3,4,5}/` — per-sub-benchmark outputs.
* `benchmarks/results/v0.3/sweep-<YYYYMMDD>/b3/rep*/`*`.jsonl` — per-replication transcripts.
* `benchmarks/results/v0.3/sweep-<YYYYMMDD>/COMPETITOR_VERSIONS.txt` — exact tool versions.

---

## 4. Per-category breakdown

<!-- TO FILL: B1 — per-language results (Rust/Python/TS/Go/Java where applicable) -->

<!-- TO FILL: B2 — per-language F1 + the 10-row blind-set summary -->

<!-- TO FILL: B3 — per-question-type CCB delta (architecture/error_handling/deep_technical/exhaustive/feature_planning) -->

### 4.1 B4 — Initial latency measurements (2026-05-24, n=3, semantex repo only)

These numbers were measured on `v0.3-integration` HEAD (commit reachable from the
Phase 1+2 integration) against the indexed semantex repo itself. They are
**preliminary** — the real sweep needs 5+ repos, 3+ reps each, on a CI runner.
They are recorded here because the §0 cap-condition checklist requires
documenting numbers **before** the wins table is written.

| Metric | v0.3 measured (n=3) | Spec target | Spec baseline ("today") | Verdict |
|---|---|---|---|---|
| daemon-ready time (W4 metric) | ~50 ms | n/a (subsumed) | ~540 ms (E8 work) | improved ~10× |
| p50 cold (end-to-end first query, fresh daemon) | **1358 ms** | ≤300 ms | ~540 ms | **regressed** vs baseline |
| p50 warm (subsequent query, hot daemon) | ~29 ms | ≤10 ms | ~17 ms | regressed slightly |
| `grep_mode` cold (no embedding, no daemon) | 20 ms | n/a | n/a | working as intended |

**What changed and why:** W4's E8 work made ColBERT ONNX session creation lazy
(deferred to first inference) and added a background warm-up thread. The
daemon-ready time genuinely dropped to ~50 ms, but the synchronous model load
now happens inside the first user query. If the user waits ~1 s after
`semantex serve` before issuing the first query (background warm-up completes
in ~1064 ms per W4's report), they see warm latency. If they query immediately,
they pay the full deferred cost.

**Honest takeaway:** the cold-start optimization changed the *shape* of cold
start (daemon-ready is fast, but first-query under pressure is slow) without
hitting the spec target on the end-to-end metric the spec actually measures.
This trips the §0 B4 cap; cold-start claim is downgraded accordingly.

**Possible v0.3.1 follow-ups:** (a) trigger synchronous model load inside the
daemon-ready path when the next query is expected within ~100 ms (latency
predictor); (b) ship the model as a memory-mapped shared file across processes;
(c) explicit `semantex warm` command users can wire into shell startup.

<!-- TO FILL: B4 — warm/cold latency distributions with p50/p95/p99 on the full benchmark repo set -->

<!-- TO FILL: B5 — paired t-test results per (question × tool pair); flag insignificant rows -->

---

## 5. Honest losses

**THIS SECTION MUST BE WRITTEN BY A DIFFERENT ACTOR THAN §1 / §4.**

This is the spec §6.7 "honest losses must be authored by a different subagent
than the wins" rule. The point is to prevent narrative drift — the writer of
the wins is unconsciously inclined to under-report losses; routing losses to a
different writer breaks that bias.

The losses author reads the same `SUMMARY.json` as the wins author, plus the
raw transcripts, plus the cap-condition status from §0. They populate the
following template. If they find a result that contradicts a §1 claim, they
edit §1 (or escalate to the human) BEFORE this section ships.

### 5.1 Where semantex loses

<!-- TO FILL (losses author): name every benchmark where semantex does NOT win or place 2nd.
     For each, give the winning tool, the magnitude of the gap, and one paragraph
     explaining the likely cause from the engine architecture. -->

### 5.2 Per-question-type CCB losses

<!-- TO FILL (losses author): for each B3 question type where semantex's CCB delta is positive
     (worse than baseline) or where p>=0.05, name it explicitly and explain the
     evidence. Example: "Architecture-trace queries on Java showed +12% CCB (Δ
     not significant at p=0.18) — semantex_agent appears to re-read the same files
     when the question spans 4+ packages. Tracked in issue #XYZ." -->

### 5.3 Cap-condition trips (if any)

<!-- TO FILL (losses author): if any §0 box is unchecked, explain the consequence
     and document which §1 claim is downgraded. -->

### 5.4 Known reproducibility caveats

<!-- TO FILL (losses author):
     - Any tool that was excluded due to verification failure (per COMPETITORS.md)
     - Any benchmark run that fell below 95% completion rate
     - Any model deprecations (e.g. Sonnet 4.6 → 4.5 fallback)
     - Any per-run variance > 30% that was averaged away in §1 (must be flagged separately) -->

---

## 6. Reproducibility checklist (per spec §4.5)

- [ ] All raw transcripts committed to `benchmarks/results/v0.3/`
- [ ] `COMPETITOR_VERSIONS.txt` committed
- [ ] Per-dataset MANIFEST.json SHAs pinned and committed
- [ ] Statistical methodology documented (n, alpha, CI method)
- [ ] Wins author and losses author are different actors
- [ ] Two external humans have run the reproducer script and confirmed numbers within ±5% (spec §6.4 Barrier 3)
- [ ] Spec §4.7 cap conditions checked before §1 wins were filled

---

## 7. Pointers

* **Spec:** `docs/superpowers/specs/2026-05-24-semantex-v0.3-sota-design.md`
* **Plan:** `benchmarks/BENCHMARK-v0.3-PLAN.md`
* **Competitor pins:** `benchmarks/COMPETITORS.md`
* **Reproducer:** `benchmarks/run_public.py`
* **Latency bench (B4 standalone):** `benchmarks/latency_bench.py`
* **B3 wrapper:** `benchmarks/run_b3.py`
* **Datasets:** `benchmarks/datasets/`
* **Per-sub-benchmark modules:** `benchmarks/v0_3/b{1,2,3,4,5}_*.py`
