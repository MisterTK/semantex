# Should reranking be on by default? (2026-07-04)

**Question:** semantex ships a cross-encoder reranker (`bge-reranker-v2-m3`) that is
off by default. The Wave 0 audit flagged "rerank off by default = benchmarks
understate quality" as a risk, but the counter-hypothesis — latency cost not
worth it, or hybrid fusion already good enough — had never been measured.
This experiment measures both sides on the SWE-bench file-localization
harness (`benchmarks/relevance/scripts/swe_loc_localize.py`), which was
extended with a `rerank` arm for this purpose (see "Harness changes" below).

**Bottom line: keep reranking OFF by default.** On every real SWE-bench-Verified
instance measured, reranking either did nothing or actively *demoted* the gold
file that hybrid had already ranked #1 (Acc@1 2/5 → 0/5, MRR@10 0.400 → 0.083)
— and it costs ~24 s per query warm (~33× hybrid's ~0.73 s) on this
container's CPU. The audit's concern ("benchmarks understate quality") is not
supported by this data: with the shipped `bge-reranker-v2-m3` on
code-localisation queries, rerank-ON would make the benchmarks *worse*, not
better. The config default was NOT flipped.

## Environment caveats (read before the numbers)

- **CPU-only container**, 4 vCPUs, ~15.7 GB RAM, shared disk (~29 GB nominal;
  observed as low as ~1 GB free while sibling agents' cargo builds ran
  concurrently). No GPU. The reranker runs its cross-encoder forward pass on
  CPU over up-to-`rerank_top_n`(=25) chunk contents per query; a GPU or a
  much smaller cross-encoder would have a very different (better) latency
  profile. The *ordering* results (reranker demotes gold files) are
  hardware-independent; the *latency* results are CPU-specific.
- **Small N (5 real instances + 1 synthetic).** Raw counts are reported next
  to every percentage; nothing here passes a significance test, and none of
  it is dressed up as if it did. The direction is consistent across all
  instances measured (rerank never improved any Acc@k anywhere), which is
  the honest summary of what small-N can support.
- **Instance selection:** the first N Phase-A instances (sorted by
  instance_id) from repos small enough to dense-index inside this
  container's memory/wall-clock budget (pytest, xarray). Large-repo
  instances (django/sympy/astropy) were out of reach: indexing one repo
  takes 9–30+ min on 4 CPUs and peaks at ~9 GB RSS. This skews the sample
  toward mid-size Python repos; there is no reason to expect the reranker
  to behave differently on larger ones (its input is the same top-25 fused
  candidates either way), but it is a real selection constraint.
- **Memory caps:** `semantex index` on these repos exceeds the binary's
  default self-imposed caps (app soft = 50 % RAM, kernel RLIMIT_AS = 75 %
  RAM). RLIMIT_AS counts *virtual* address space, and mimalloc's arena
  reservations blow through it long before RSS is actually exhausted —
  indexing pytest died at ~7.9 GB RSS with "memory allocation of 156754368
  bytes failed" twice before the caps were identified. All runs here used
  `SEMANTEX_NO_RLIMIT=1 SEMANTEX_MAX_RSS_MB=0` (the documented
  container-with-cgroup-limits escape hatch). Observed peak ≈ 9.2 GB RSS
  indexing pytest (3 201 chunks).
- **Model download is real network I/O.** `bge-reranker-v2-m3` (~2.2 GB on
  disk under `~/.fastembed_cache`) downloads from Hugging Face on first-ever
  use; later runs reuse the cache but each fresh daemon still pays ONNX
  session construction from disk (residual in "cold" timings).

### The three-gate trap (methodological, important for anyone re-running this)

semantex's reranker only actually activates on a daemon-served query when
**three** independent switches line up:

1. the per-query `--rerank` CLI flag (forwarded as `use_rerank` on the wire),
2. `SEMANTEX_RERANK=1` — sets the **daemon's own** `SemantexConfig.rerank`
   at *its* spawn time. The auto-spawned daemon is started as bare
   `serve <path>`, so the CLI's `--rerank` never reaches its config; without
   this env var, `query.use_rerank && self.config.rerank` (hybrid.rs)
   short-circuits false and the rerank stage is silently skipped — no error,
   no weight download, results byte-identical to `hybrid`,
3. `SEMANTEX_RERANKER=on` — the master switch gating reranker construction /
   weight download (the S3 off-by-default safety contract,
   `fastembed_reranker.rs::reranker_enabled`).

During harness development the `rerank` ablation initially set only gate 3;
every "rerank" measurement was silently identical to hybrid. The tell was
`~/.fastembed_cache` staying empty and rerank timings matching plain hybrid.
Fixed in `semantex_client.py::_build_env` (sets gates 2+3 for the `rerank`
ablation only) plus a forced daemon respawn before the rerank arm in
`swe_loc_runner.run_instance` (both env gates are spawn-time-only). The
correct activation was then verified positively: `source: "Reranked"` in the
JSON output, cross-encoder scores replacing RRF scores, and a model download
on first use.

### Pre-existing engine bug found on the way (out of scope, not fixed here)

`sparse-only` and `agent-routed` scored 0 across all real instances — not
because BM25 is that bad, but because **the sparse channel returns zero
results for queries containing `--`-style markdown** (e.g. the HTML comments
`<!-- ... -->` present in most real GitHub issue bodies). Minimal repro:
`semantex --sparse-only --json -- 'unittest setUpClass <!-- comment --> fixture private'`
→ `[]`, while the same query without the comment returns hits.
`hybrid.rs::sanitize_tantivy_query` escapes `-` and `!` with backslashes, but
the resulting query still fails tantivy's parser, and
`sparse.search(...).unwrap_or_default()` swallows the error into an empty
list. Hybrid survives because its dense channel still fires (which is also
why this doesn't invalidate the hybrid-vs-rerank comparison: both arms share
the identical fused retrieval pool). Fixing this is core-search work owned
by another track; flagged here so the `sparse-only` zeros aren't read as a
real BM25 quality measurement. It DOES mean hybrid ran dense-only-in-practice
on these instances, i.e. real-issue queries currently get no lexical signal.

## Harness changes (what was extended)

- New `rerank` arm alongside the existing `hybrid` / `sparse-only` /
  `agent-routed` / `ripgrep` arms in
  `benchmarks/relevance/src/relevance_harness/swe_loc_runner.py`: same
  retrieval pool as `hybrid` (`semantex search`), with `--rerank` plus the
  two env gates above, isolating the reranker's effect on the same candidates.
- Cold/warm per-query latency capture on `ArmQueryResult`: each search-path
  arm times its scored query ("cold") and an immediate identical repeat
  against the same warm daemon/model ("warm"), so the rerank-specific
  steady-state cost is readable separately from one-time daemon-spawn /
  model-load cost.
- `scripts/swe_loc_localize.py::compute_arm_rows` now reports p50/p95 cold
  and warm latency (ms) per arm (`relevance_harness.metrics.percentile`,
  new dependency-free helper).
- Test coverage at parity with the pre-existing arms
  (`tests/test_swe_loc_runner.py`, `tests/test_semantex_client.py`,
  `tests/test_swe_loc_localize.py`, `tests/test_metrics.py`).

## Results

Protocol: one query per instance (the GitHub issue's problem statement),
gold = the files touched by the merged fix's patch, Acc@k = "any gold file in
the top k" (the SweRank/LocAgent file-localisation protocol), k=10,
`SEMANTEX_ADAPTIVE_SIZING=0` (the canonical A/B lock). Engine rev `a6daa31`,
release build, default embedder (`lateon-colbert` → colbert-plaid).

### Offline fixture (3-file synthetic corpus, 1 query — pipeline validation + clean latency floor)

| arm | Acc@1 | cold p50 | warm p50 |
|---|---|---|---|
| hybrid | 1/1 | 779 ms | 20 ms |
| sparse-only | 1/1 | 17 ms | 11 ms |
| **rerank** | 1/1 | 6 436 ms | **2 202 ms** |
| agent-routed | 1/1 | 484 ms | n/a |
| ripgrep | 1/1 | 13 ms | n/a |

Every arm trivially hits on a 3-file corpus; the value here is the latency
floor: even with only 3 files to score, a warm reranker costs ~2.2 s/query
(~110× hybrid warm) on this CPU.

### Real SWE-bench-Verified instances (n = 5)

Instances (first Phase-A ids from indexable repos, deterministic order):
`pytest-dev__pytest-5631`, `pytest-dev__pytest-5809`, `pytest-dev__pytest-8399`,
`pydata__xarray-3095`, `pydata__xarray-3151`.

| arm | Acc@1 | Acc@5 | Acc@10 | MRR@10 | cold p50 | cold p95 | warm p50 | warm p95 | errors |
|---|---|---|---|---|---|---|---|---|---|
| **hybrid** (shipped default, rerank off) | **2/5** | **2/5** | 2/5 | **0.400** | 1 501 ms | 1 647 ms | **730 ms** | **825 ms** | 0 |
| sparse-only | 0/5 | 0/5 | 0/5 | 0.000 | 16 ms | 21 ms | 17 ms | 21 ms | 0 |
| **rerank** (hybrid + cross-encoder) | **0/5** | 1/5 | 2/5 | 0.083 | 31 632 ms | 56 874 ms | **23 867 ms** | **24 447 ms** | 0 |
| agent-routed | 0/5 | 0/5 | 0/5 | 0.000 | 21 ms | 25 ms | n/a | n/a | 0 |
| ripgrep | 0/5 | 1/5 | 2/5 | 0.095 | 54 ms | 99 ms | n/a | n/a | 0 |

(`sparse-only`/`agent-routed` zeros are the engine bug described above, not a
BM25 quality result. Warm rerank p95 is **29.6×** warm hybrid p95 — against
the "< ~2×" acceptance bar.)

Per-instance rank of the first gold file, hybrid vs rerank (the head-to-head
that answers the question — identical retrieval pool, only the final
ordering stage differs):

| instance | gold file(s) | hybrid rank | rerank rank |
|---|---|---|---|
| pytest-dev__pytest-5631 | src/_pytest/compat.py | not in top-10 | not in top-10 |
| pytest-dev__pytest-5809 | src/_pytest/pastebin.py | **1** | **4** (demoted) |
| pytest-dev__pytest-8399 | src/_pytest/{python,unittest}.py | not in top-10 | not in top-10 |
| pydata__xarray-3095 | xarray/core/{indexing,variable}.py | not in top-10 | not in top-10 |
| pydata__xarray-3151 | xarray/core/combine.py | **1** | **6** (demoted) |

On both instances where hybrid placed the gold file at rank 1, the reranker
demoted it (1→4, 1→6), promoting *test files* above the implementation file
in each case (e.g. on xarray-3151, ranks 1–5 after reranking are
`test_dataset.py`, `api.py`, `test_dataset.py`, `test_dataset.py`,
`test_dataarray.py`). Where hybrid missed, rerank missed identically (it can
only reorder the same top-25 pool). Rerank's Acc@1 went 2/5 → 0/5 and MRR@10
0.400 → 0.083 relative to hybrid.

Raw per-instance latencies (cold = first scored call incl. any model/session
load the fresh daemon still owes; warm = immediate identical re-query):

| instance | hybrid cold/warm (s) | rerank cold/warm (s) |
|---|---|---|
| pydata__xarray-3095 | 1.67 / 0.84 | 30.39 / 23.10 |
| pydata__xarray-3151 | 1.58 / 0.77 | 30.35 / 23.09 |
| pytest-dev__pytest-5631 | 1.18 / 0.41 | 31.63 / 23.87 |
| pytest-dev__pytest-5809 | 1.14 / 0.38 | 56.78 / 24.04 |
| pytest-dev__pytest-8399 | 1.50 / 0.73 | 56.90 / 24.55 |

The rerank warm cost is remarkably stable (~23–25 s) across repos — it is
dominated by the cross-encoder forward pass over the top-25 chunk contents,
which is corpus-size-independent. The two ~57 s cold outliers include the
reranker ONNX session build in a fresh daemon.

## Analysis

1. **Reranking never helped.** In no instance did the rerank arm place a gold
   file at a better rank than hybrid. Its only visible effects were (a)
   reordering *within* the same top-10 set, and (b) on
   `pytest-dev__pytest-5809`, **demoting** the gold file hybrid had at
   rank 1 down to rank 4 — i.e. it broke the one clean Acc@1 hit in the
   sample. Mechanically this is easy to explain: the cross-encoder scores
   `(issue text, chunk content)` pairs, and real issue bodies (markdown,
   repro snippets, tracebacks) score *test files* highly (they look like
   repro code), pushing implementation files down.
2. **The latency cost is disqualifying on CPU regardless.** Warm rerank
   p50 ≈ 23.9 s/query on real repos vs ≈ 0.73 s hybrid — a ~33× multiplier
   at p50, ~30× at p95 (the fixture's clean floor: 2.2 s vs 20 ms, ~110×).
   The acceptance bar was "warm p95 < ~2× non-rerank"; measured ≈ 30×.
   Interactive use (the MCP agent path budgets whole tool calls in low
   seconds) cannot absorb a ~24 s search stage.
3. **Acc@k on hybrid is itself low in this sample** (see the sparse-channel
   bug above, plus genuinely hard multi-file gold sets). That is a
   retrieval-pool problem: the reranker can only reorder the top 25 fused
   candidates, so no reranker fixes a pool that doesn't contain the gold
   file. The audit's implicit model ("benchmarks understate quality because
   rerank is off") assumes rerank-on adds quality on top of the pool; the
   data shows the opposite on this task with this model.

## Recommendation

- **Do not flip `rerank` to on-by-default.** Both acceptance criteria fail:
  Acc@k does not improve (it degrades where it changes anything: Acc@1
  2/5 → 0/5), and warm latency is ~30× non-rerank against a < 2× bar. The
  current default (`SemantexConfig::default().rerank = false` + the
  `SEMANTEX_RERANKER` master switch) is the right one. No config change was
  made.
- The reasons are engine-actionable, so recording them for future waves
  (all out of scope for this branch):
  1. `bge-reranker-v2-m3` is a natural-language relevance model; a
     code-tuned or smaller reranker (or one scoring file paths + signatures
     instead of raw chunk bodies) could change the quality story. The
     latency story needs a much smaller model regardless.
  2. Fix the sparse-channel parse failure on `--`-containing queries
     (silently swallowed by `unwrap_or_default`) — it currently removes all
     lexical signal for real-world issue-shaped queries, which hurts the
     retrieval pool that ANY reranker would depend on.
  3. If rerank is ever revisited, gate it per-query-type (e.g. only
     Semantic queries) and benchmark against this harness's `rerank` arm —
     the instrument now exists and costs one flag to run.

## Reproduction

```
cd benchmarks/relevance
source .venv/bin/activate            # see README Setup
# offline fixture (validates pipeline + latency floor; reranker model
# downloads on first use):
export SWE_BENCH_REPO_CACHE=/tmp/tiny_swe_loc_cache
mkdir -p "$SWE_BENCH_REPO_CACHE/tiny__tiny-1"
cp fixtures/tiny_corpus/* "$SWE_BENCH_REPO_CACHE/tiny__tiny-1/"
python -m scripts.swe_loc_localize --local-fixture fixtures/tiny_swe_loc_instance.json

# real instances (each repo is cloned + fully indexed first — 9-30+ min/repo
# on CPU; in a memory-capped container export SEMANTEX_NO_RLIMIT=1
# SEMANTEX_MAX_RSS_MB=0):
cd ../swe_bench && python -m scripts.pre_index --phase a
cd ../relevance && python -m scripts.swe_loc_localize --limit 5
```
