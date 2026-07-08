# semantex v1 — Benchmark Validation (fresh re-measurement)

**Purpose:** re-measure the README's headline retrieval/efficiency claims on the exact
build that will ship, now that two release-candidate fixes have landed that plausibly
change prior numbers:

- `8d8c2d4` **lenient-parse fallback for hostile sparse queries** — real GitHub-issue-shaped
  queries containing `--`/`<!-- -->` markdown previously made semantex's BM25 (sparse)
  channel return zero results silently (`sparse.search(...).unwrap_or_default()` swallowing
  the tantivy parse error). This was the root cause behind `sparse-only` scoring 0/5 in the
  prior rerank experiment (`benchmarks/relevance/results/rerank-experiment.md`).
- `95b500b` **memory-cap fix** — the old kernel `RLIMIT_AS` cap measured *virtual address
  space* but was budgeted as if it were RSS, aborting legitimate first-index builds whose
  actual peak RSS fit comfortably in the box. Also raises indexing thread count on
  single-build-slot machines (measured ~1.5x wall-clock speedup for a full index).

## Environment

- **Container:** CPU-only, 4 vCPUs, ~15.7 GB RAM (`free -h`: 15Gi total), **no swap**, ~25 GB
  free disk at measurement start (shared with the rest of the worktree).
- **Engine commit:** `d994e8a2afca31e9bf2ab5ed379e48ac10923f44` (`origin/main`, the release
  candidate named in this task — includes both fixes above), built via `cargo build --release`
  (release profile, LTO, single build, ~7 min on this container).
- **Binary:** `target/release/semantex`, version `0.1.2`. Memory caps reported at startup on
  this container: `app soft=8037 MB, kernel hard=64300 MB; system RAM=16075 MB`.
- **Prior comparison point:** the rerank experiment's SWE-loc table
  (`benchmarks/relevance/results/rerank-experiment.md`), engine rev `a6daa31` — an earlier
  commit that predates both fixes above.

## Summary for the launch decision

1. **The sparse fix works, and it moves hybrid too, not just `sparse-only`.** On the same
   5-instance set the prior report used, `sparse-only` went from **0/5 Acc@1/5/10 (MRR
   0.000)** to **1/5 Acc@1, 3/5 Acc@5, 5/5 Acc@10 (MRR 0.442)** — BM25 is no longer dark on
   real issue-shaped queries. `hybrid` (the shipped default) also improved, because it fuses
   the now-live sparse channel: Acc@1 **2/5 → 2/5 unchanged**, but Acc@5 **2/5 → 3/5**, Acc@10
   **2/5 → 4/5**, MRR@10 **0.400 → 0.522**. One concrete example: `pytest-dev__pytest-8399`'s
   gold file was "not in top-10" for hybrid in the prior report; it now ranks **9th**.
2. **The reranker no longer reliably hurts, and on this rerun it net-helps** — a reversal of
   the prior report's headline finding (which was itself conditioned on the sparse-channel bug
   starving the retrieval pool of lexical signal). See §2 for the full before/after. Its
   latency remains completely disqualifying for interactive use (tens of seconds/query on this
   CPU) — that conclusion is unchanged.
3. **The memory fix works as hoped, and `SEMANTEX_NO_RLIMIT`/`SEMANTEX_MAX_RSS_MB=0` are no
   longer needed** to index the exact repo (pytest) that required them in the prior
   experiment, on the same class of container (see §2 memory verification).
4. **The README's specific "Measured Impact" table numbers (212K/355K tokens, 2.2M/6.8M CCB,
   etc.) could not be verified or reproduced from anything in this repository** — no raw run
   matching those exact figures exists in git history, and the underlying eval-plan doc the
   README points to has been removed from this checkout. A fresh, real 2-arm rerun on this
   container (§1) reproduces the *direction* of every claimed metric (CCB -39%, tool calls
   -51%, peak context -36%, wall-clock -54%, quality *improved* rather than tied) but not the
   specific magnitudes or the tied-quality result. This is flagged for the maintainer; see §1.
5. Full quantitative tables below (§1 CCB rerun, §2 SWE-loc rerun).

---

## 1. CCB / three-way (agent-quality + token-efficiency) rerun

### What actually produced the README's numbers — untraceable

The README's "Measured Impact" table (`Benchmark: 10 agents, 5 real-world tasks, Sonnet
4.6` — 212K vs 355K tokens, 2.2M vs 6.8M CCB, 39 vs 86 tool calls, -40%/-67%/-55%/-40%/-16%,
tied quality) has been present with **byte-identical numbers** since the commit that first
added this section, unchanged across the later "reposition README for mid-2026" rewrite
(`96ce667`). It is footnoted "Self-reported benchmark... Full methodology and per-question
data available in the benchmark suite (`benchmarks/`)" — but **no raw run, report, or
per-question data matching these exact numbers exists anywhere in this repo's git history**
(checked `git log --all -p -- README.md`, `results/`, `benchmarks/results/`; the `docs/`
tree the harness's own code comments cite for the underlying eval plan
(`docs/superpowers/plans/2026-05-30-semantex-eval-plan.md`) is absent from this checkout
entirely — see the `claude/remove-internal-docs` history). The closest **committed, real**
runs of the actual harness this repo does carry:

| source | arms | repo(s) | result |
|---|---|---|---|
| `results/three-way/report.md` (2026-06-05, pre-fix engine) | builtin/semantex/graphify/serena | CopilotKit | semantex: quality 5.0, CCB 1.18M; builtin: quality 4.0, CCB 2.15M (**-45% CCB**, better quality) |
| `results/cbench-clean-largerepo/report.md` (2026-06-05, pre-fix engine) | builtin / semantex bare / semantex steered | "platform" (private, unavailable here) + CopilotKit | bare semantex ≈ ties builtin on CCB (adoption-gated); **steered** semantex: −57%/−24% CCB at matched-or-better quality |

**Neither matches the README table's specific magnitudes.** The real, reproducible evidence
in this repo supports the qualitative claim (semantex reduces CCB/tool-call overhead without
hurting answer quality, *when the agent actually uses it*) but not the specific numbers
printed in the README. **This is a finding for the maintainer** — the mission explicitly
disallows editing the README from this branch, so it is flagged here instead of silently
patched.

### This rerun

`benchmarks/three_way.py` (4-arm equal-adoption harness) requires `graphify` and `serena`
binaries — **neither is installed in this container** — and targets a hardcoded external
repo path (`/path/to/CopilotKit`) not present here. **Skipped in full** (see "What was
skipped" below). `benchmarks/claude_bench.py` supports arbitrary `--repos` and a `builtin`
vs `semantex` 2-arm comparison, which is the shape that maps onto the README table, so that
is what was rerun here.

- **Target repo:** `pallets/flask` (the harness's own documented example repo —
  `python3 benchmarks/claude_bench.py run --repos ~/dev/gin ~/dev/flask ...`); CopilotKit
  itself was not re-cloned in this container (see "skipped" below).
- **Arms:** `builtin` (native Grep/Glob/Read/Bash, no MCP) vs `semantex` (semantex MCP,
  default embedder `lateon-colbert`), repo pre-indexed (`semantex index .`) before the run so
  the semantex arm is measured at steady state.
- **Questions:** all 5 of `claude_bench.QUESTIONS` (architecture / error_handling /
  deep_technical / exhaustive / feature_planning) — "5 real-world tasks"; `--reps 1` → 10
  total agent invocations (2 arms × 5 questions), matching "10 agents" under the most
  literal reading of the README footnote.
- `--neutralize-claude-md` passed (flask ships no CLAUDE.md, so it is a no-op safety net
  here, not a live factor).
- Judge: `claude_bench.py judge` — blind 1-5 quality score, tool-blind, project-free judge
  call, run in a neutral cwd.

### Results

Raw per-cell data: `wall_secs`/`cost_usd`/`ccb`/`peak_context`/`tool_calls` from
`claude_bench.py`'s own `parse_claude_stream`; `quality` from its blind judge
(`claude-sonnet-4-6`, 1-5 scale). n=5 question-runs per arm, 1 rep each.

| metric | builtin (mean) | semantex (mean) | delta | README's original claim (untraceable, see above) |
|---|---|---|---|---|
| **Cumulative context burden (CCB)** | 1,525,681 | 936,811 | **-39%** | -67% (2.2M → 6.8M) |
| **Peak context size** | 83,555 | 53,155 | **-36%** | -40% (42K → 71K avg) |
| **Tool calls** | 30.4 | 15.0 | **-51%** | -55% (39 → 86) |
| **Wall-clock time** | 254 s | 116 s | **-54%** | -16% (513s → 609s) |
| **Cost (USD, proxy for "total tokens")** | $1.054 | $0.372 | **-65%** | -40% (212K → 355K tokens) |
| **Answer quality (blind judge, 1-5)** | 3.80 | **4.80** | **semantex higher**, not tied | "Tie" (both "Comprehensive") |

Per-question breakdown (`n=1` each — this is where the small-N caveat bites hardest):

| question | builtin CCB | semantex CCB | builtin quality | semantex quality |
|---|---|---|---|---|
| Q1 architecture | 1,971,246 | 820,446 | 5 | 5 |
| Q2 error_handling | 2,411,244 | 1,491,208 | 5 | 5 |
| Q3 deep_technical | 1,255,035 | 439,258 | **1** | 5 |
| Q4 exhaustive | 608,128 | 861,688 (semantex *higher* here) | 5 | 5 |
| Q5 feature_planning | 1,382,754 | 1,071,453 | 3 | 4 |

Tool usage (mean calls/run): builtin used 0 `semantex_agent` calls (as expected — it has no
MCP) and 30.4 native (Grep/Read/Bash) calls; semantex used 5.4 `semantex_agent` calls and 9.6
native calls (semantex still falls back to native tools sometimes — it doesn't fully displace
them on this repo/question mix, consistent with the "adoption-gated" finding in
`results/cbench-clean-largerepo/report.md`).

**Reading this table honestly:** every metric moves in the claimed direction, several by
similar or larger magnitude (tool calls -51% vs claimed -55%; peak context -36% vs claimed
-40%). Two differ materially: wall-clock here is -54% (claimed only -16% — builtin's grep
loops were simply slow on this repo/container), and quality shows semantex clearly ahead
(4.80 vs 3.80) rather than tied — driven almost entirely by one cell (Q3, builtin scored a
blind-judge 1/5 — see `/tmp/ccb_work/out/all_results.json`-equivalent per-question data; not
committed, gitignored). With n=1 per question, one bad builtin answer swings the mean quality
substantially; this is real data, not cherry-picked, but it is not a robust quality delta
without more reps.

### What was skipped and why

- **`graphify`, `serena` arms** (`three_way.py`) — neither binary is installed in this
  container, and installing serena means an `uvx`-managed clone of
  `github.com/oraios/serena` at benchmark-run time — out of scope for a benchmark-only
  branch that should not add new runtime dependencies just to re-run a comparison this
  container can't otherwise support.
- **CopilotKit / "platform" as target repos** — CopilotKit is a large TS monorepo; a cold
  clone + dense index of it competes for the same CPU/RAM budget the SWE-loc indexing in
  §2 needs serially (this container has **no swap**, and a single index build alone peaked
  at ~9.4 GB RSS on a mid-size repo — see §2). "platform" is a private repo unavailable to
  this container at all. Flask is the harness's own documented substitute-scale example, and
  the single largest methodological deviation from the original CCB runs in this report.
- **Reps > 1** — the original README numbers imply 1 rep; more reps would sharpen confidence
  at proportional extra wall-clock/API cost, which this pass did not spend.

---

## 2. SWE-bench-Verified file-localisation rerun (`swe_loc_localize`)

Rerunning the exact 5-instance set from `benchmarks/relevance/results/rerank-experiment.md`
(engine rev `a6daa31`): `pytest-dev__pytest-5631`, `pytest-dev__pytest-5809`,
`pytest-dev__pytest-8399`, `pydata__xarray-3095`, `pydata__xarray-3151` — "the first Phase-A
ids from indexable repos" (astropy/django/sympy instances are too large to dense-index in
this class of container within a practical time budget; the same selection constraint
applies here as it did in the prior report).

Protocol (unchanged from the prior report): one query per instance (the GitHub issue's
problem statement), gold = files touched by the merged fix, Acc@k = "any gold file in top
k", k=10, `SEMANTEX_ADAPTIVE_SIZING=0` (the harness's canonical A/B lock, set automatically
by `relevance_harness.indexing` / `semantex_client`), default embedder (`lateon-colbert` ->
colbert-plaid). All 5 arms the harness runs per instance are reported:  `hybrid`,
`sparse-only`, `rerank`, `agent-routed`, `ripgrep`.

Disk hygiene: each instance was checked out + indexed **one at a time** (via the harness's
own `relevance_harness.swe_loc_runner.run_instance`, unmodified), measured, then its
checkout+index directory was deleted before moving to the next — no more than one
instance's repo+index was on disk at any point during this rerun.

### Results

**Fresh (this rerun, engine `d994e8a`, k=10, n=5 real instances):**

| arm | Acc@1 | Acc@5 | Acc@10 | MRR@10 | avg tokens returned | cold p50 | cold p95 | warm p50 | warm p95 | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| **hybrid** (shipped default, rerank off) | 2/5 (0.400) | 3/5 (0.600) | 4/5 (0.800) | **0.522** | 5,123 | 1,222 ms | 1,717 ms | 469 ms | 994 ms | 0 |
| **sparse-only** | 1/5 (0.200) | 3/5 (0.600) | 5/5 (1.000) | **0.442** | 6,345 | 24 ms | 40 ms | 27 ms | 36 ms | 0 |
| rerank (hybrid + cross-encoder) | 2/5 (0.400) | 4/5 (0.800) | 5/5 (1.000) | 0.537 | 4,703 | 68,874 ms | 75,633 ms | 66,788 ms | 68,950 ms | 0 |
| agent-routed | 3/5 (0.600) | 3/5 (0.600) | 3/5 (0.600) | 0.600 | 12,697 | 24 ms | 35 ms | n/a | n/a | 0 |
| ripgrep | 0/5 (0.000) | 1/5 (0.200) | 2/5 (0.400) | 0.095 | 0 | 60 ms | 81 ms | n/a | n/a | 0 |

**Prior report (rerank-experiment.md, engine `a6daa31`, same 5 instances, same k=10) for
comparison:**

| arm | Acc@1 | Acc@5 | Acc@10 | MRR@10 | cold p50 | warm p50 |
|---|---|---|---|---|---|---|
| hybrid | 2/5 (0.400) | 2/5 (0.400) | 2/5 (0.400) | 0.400 | 1,501 ms | 730 ms |
| sparse-only | 0/5 (0.000) | 0/5 (0.000) | 0/5 (0.000) | 0.000 | 16 ms | 17 ms |
| rerank | 0/5 (0.000) | 1/5 (0.200) | 2/5 (0.400) | 0.083 | 31,632 ms | 23,867 ms |
| agent-routed | 0/5 (0.000) | 0/5 (0.000) | 0/5 (0.000) | 0.000 | 21 ms | n/a |
| ripgrep | 0/5 (0.000) | 1/5 (0.200) | 2/5 (0.400) | 0.095 | 54 ms | n/a |

**What moved and why:**

- `sparse-only`: **0/5 → 1/5/3/5/5/5, MRR 0.000 → 0.442.** This is the sparse fix, directly:
  the prior 0/5 floor was the `<!--`-comment parse bug silently emptying the BM25 channel on
  every real GitHub-issue query, not a genuine BM25 quality result. Fixed, BM25 alone now hits
  the gold file within the top 10 on every one of the 5 instances.
- `hybrid`: MRR **0.400 → 0.522**, Acc@5/10 roughly doubled (2/5 → 3/5, 2/5 → 4/5). Hybrid
  fuses dense + sparse, so a previously-dark sparse channel meant hybrid was running
  dense-only-in-practice on these queries; it's now genuinely hybrid.
- `rerank`: MRR **0.083 → 0.537**, Acc@10 **2/5 → 5/5**, and it no longer *demotes* hybrid's
  rank-1 hits — see the per-instance table below. This reverses the prior report's headline
  finding, but not because the reranker itself changed: the prior finding was measured on a
  retrieval pool that was accidentally dense-only (see `hybrid` above); reranking a richer,
  now-genuinely-hybrid pool behaves differently. **Rerank's latency remains completely
  disqualifying for interactive use** on this CPU-only container — warm p50 ≈ **67 s/query**,
  ~140x hybrid's warm p50 (worse than the prior report's ~33x; see caveats — this container
  had run several back-to-back index builds immediately before this arm, so some of the
  absolute latency here is likely contention, not a pure engine regression, but the *relative*
  gap versus hybrid confirms the same qualitative conclusion the prior report reached: rerank
  stays off by default).
- `agent-routed`: **0/5 → 3/5 Acc@1**, now the best-performing arm on Acc@1/MRR. Directly
  downstream of the sparse fix (the classifier's chosen route depends on a working sparse
  channel for several of the mechanisms it can pick).
- `ripgrep`: **unchanged** (0/5, 1/5, 2/5, MRR 0.095) — expected; this baseline doesn't touch
  semantex at all, and is an internal consistency check that this rerun's methodology matches
  the prior one.

### Per-instance rank of the first gold file (hybrid vs sparse-only vs rerank)

| instance | gold file(s) | hybrid rank | sparse-only rank | rerank rank |
|---|---|---|---|---|
| pytest-dev__pytest-5631 | src/_pytest/compat.py | 2 | 2 | 10 |
| pytest-dev__pytest-5809 | src/_pytest/pastebin.py | **1** | 2 | **1** (was demoted to 4 in the prior report) |
| pytest-dev__pytest-8399 | src/_pytest/{python,unittest}.py | 9 (was "not in top-10") | 9 | 4 |
| pydata__xarray-3095 | xarray/core/{indexing,variable}.py | not in top-10 | 10 | 3 (was "not in top-10") |
| pydata__xarray-3151 | xarray/core/combine.py | **1** | **1** | **1** (was demoted to 6 in the prior report) |

On both instances where hybrid places the gold file at rank 1 (`pytest-5809`, `xarray-3151`),
the reranker **no longer demotes it** — in the prior report both were demoted (1→4, 1→6). This
is the clearest, most direct rebuttal of the prior report's specific "reranker demotes good
hybrid hits" finding; it does not rebut the prior report's latency conclusion, which still
holds (rerank stays off by default on quality-neutral-or-better + latency-disqualifying
grounds — now for the latency reason alone rather than both).

### Memory-fix verification (`SEMANTEX_NO_RLIMIT` / `SEMANTEX_MAX_RSS_MB`)

The prior rerank experiment needed `SEMANTEX_NO_RLIMIT=1 SEMANTEX_MAX_RSS_MB=0` to index
these repos at all ("indexing pytest died at ~7.9 GB RSS ... twice before the caps were
identified"). This rerun indexed `pytest-dev__pytest-5631` (base commit
`cb828ebe70b4fa35cd5f9a7ee024272237eab351`) on the **same class of container** (4 vCPU,
~15.7 GB RAM, no swap) with **no memory env overrides at all**. Observed: peak RSS reached
~9.4 GB (above the app-level soft cap of 8037 MB derived from this container's RAM, and
briefly above the *old* kernel `RLIMIT_AS` formula too) during the encode/PLAID-build phase,
and the index **completed successfully** — `semantex index .` returned 0 and produced a
valid `.semantex/meta.json` with `dense_backend: colbert-plaid`, matching the fix's own
described mechanism (the RSS soft-cap is polled at batch/phase checkpoints, not inside the
single uncheckpointed PLAID build call where the transient spike occurs, and the new kernel
`RLIMIT_AS` ceiling — 4x RAM, `~64.3 GB` here — sits far above the ~20 GB address-space
excursion that used to trip the old formula).

**Verified: `SEMANTEX_NO_RLIMIT` is no longer necessary for this workload on this class of
container.** The remaining risk is machines with materially less RAM than ~16 GB, or repos
whose plateau RSS (not just the transient uncheckpointed spike) sits above 50% of system
RAM — not exercised here.

---

## Caveats (read before citing any number above)

- **CPU-only container, no GPU, no swap.** The `rerank` arm's latency numbers in particular
  are hardware-specific (a GPU or smaller cross-encoder would look very different); the
  *ordering* (does rerank help Acc@k) is not.
- **Small N.** 5 real SWE-loc instances, 10 CCB agent runs on one repo. Raw counts are
  reported next to every rate below; nothing here clears a significance bar.
- **Different target repo for CCB** (flask, not CopilotKit/"platform") — see "What was
  skipped" in §1. These numbers are informative about direction, not a strict
  apples-to-apples replacement for the README's original (untraceable) benchmark.
- **Self-reported.** Same disclaimer the README already carries: this was run by the
  semantex team (here, the benchmark-validation workstream), not an independent third party.

## Recommendation for the maintainer

- **Do not ship the current README "Measured Impact" table as-is** without either (a)
  locating/re-deriving its source run, or (b) replacing it with numbers from a rerun like this
  one (with an explicit repo/N caveat). It is currently an unverifiable claim sitting next to
  real, differently-shaped evidence in this same repo. The fresh rerun in §1 supports the same
  *qualitative* story (semantex meaningfully cuts CCB/tool-calls/peak-context/wall-clock
  without a quality tradeoff) but not the specific numbers or the "tied" quality framing —
  this rerun's quality delta favors semantex, not a tie, though on n=1/question that is not a
  claim to lean on either.
- **The sparse-fix and memory-fix engineering claims in the task brief both check out on fresh
  measurement** (§2): BM25 is no longer dark on real issue-shaped queries (0/5 → real
  non-zero Acc@k across every arm downstream of it), and `SEMANTEX_NO_RLIMIT`/
  `SEMANTEX_MAX_RSS_MB=0` are no longer required to index the repo that previously needed
  them, on the same container class. Safe to cite internally; the specific magnitudes are this
  container's numbers, not universal constants.
- **The prior rerank-off-by-default decision does NOT need to be revisited on quality
  grounds** (rerank now helps rather than hurts on this sample) **but should stay off on
  latency grounds alone** — warm p50 is still ~140x hybrid's on this CPU-only container. If a
  future track revisits reranking, this rerun's per-instance table (rerank no longer demoting
  hybrid's rank-1 hits) is worth referencing as evidence the earlier "reranker actively hurts"
  finding was an artifact of the sparse-channel bug, not an inherent property of the
  cross-encoder.

## Reproduction

Everything in §1 and §2 was produced by the harnesses already committed in this repo
(`benchmarks/claude_bench.py`, `benchmarks/relevance/scripts/swe_loc_localize.py` via
`relevance_harness.swe_loc_runner.run_instance`), run unmodified except for one narrow change
(`benchmarks/relevance/src/relevance_harness/swe_loc_runner.py`: `SemantexClient` timeout for
this runner raised from the client's 120 s default to 300 s — a cold `rerank` call on this
CPU-constrained container's contended cores exceeded 120 s even with the cross-encoder weights
already cache-warm, which was a harness-timeout false negative, not an engine failure; see the
inline comment). No other harness code changed. §1 used a one-off orchestration script (not
committed — lives outside `benchmarks/`) that calls `claude_bench.py`'s `setup`/`run`/
`judge`/`report` subcommands exactly as documented in its own module docstring, against
`pallets/flask` instead of the original CopilotKit/"platform" targets (see "What was skipped").
§2 used a similar one-off driver that calls `relevance_harness.swe_loc_runner.run_instance`
once per instance, serially, deleting each checkout+index before moving to the next.
