# semantex Relevance Harness

Pure-retrieval relevance measurement for semantex (no LLM confound).
Datasets: CoIR (headline), CodeSearchNet (external calibration), SWE-bench
localization (code-native). Metrics: MRR@10, nDCG@10, Recall@{1,5,10}, MAP.

## Setup
```
cd benchmarks/relevance
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Run
```
# CodeSearchNet, full hybrid, default dense backend
python -m scripts.run --dataset csn --ablation hybrid

# Ablation sweep (sparse vs dense vs hybrid vs hybrid+rerank)
python -m scripts.run --dataset csn --ablation sparse-only
python -m scripts.run --dataset csn --ablation dense-only
python -m scripts.run --dataset csn --ablation rerank

# D4 embedder A/B (env-selected via SEMANTEX_EMBEDDER; canonical per integration §4)
python -m scripts.run --dataset csn --ablation hybrid --embedder lateon-colbert
python -m scripts.run --dataset csn --ablation hybrid --embedder coderank-137m

# SWE-bench localization (reuses benchmarks/swe_bench Phase-A instances)
python -m scripts.run --dataset swe-loc --ablation hybrid

# Acceptance gate: reproduce a published baseline within tolerance
python -m scripts.reproduce_baseline --dataset csn
```

All subsets are seeded and logged; every report records the exact datasets,
sample sizes, dropped items, git rev, and dense backend used.

## SWE-bench file-level localisation (external, reproducible benchmark)

`scripts/swe_loc_localize.py` is the externally-comparable retrieval
benchmark: given an issue's title+body, rank the files the gold patch
touched. It scores **Acc@1 / Acc@5 / Acc@10** (file-level hit rate — did ANY
gold file appear in the top-k, the metric SweRank and LocAgent report) plus
**MRR@10** and an **avg-tokens-returned** estimate, so numbers are directly
comparable to published SweRank/LocAgent tables instead of being
self-referential.

Four arms are run per instance, so semantex is measured against real
baselines in the same pass:

| arm            | what it is                                                                 |
|----------------|------------------------------------------------------------------------------|
| `hybrid`       | `semantex search` — semantex's shipped dense+sparse fusion default        |
| `sparse-only`  | `semantex search --sparse-only` — BM25-only baseline                      |
| `agent-routed` | `semantex agent --json-hits` with **no** forced route — the engine's own keyword classifier (`agent_classifier.rs`) picks the retrieval mechanism, same as an unforced real `agent` call |
| `ripgrep`      | an external keyword baseline (`relevance_harness/ripgrep_baseline.py`) — issue-derived identifiers, ranked by ripgrep match count. No semantex at all: the floor every retrieval claim should beat. |

All four arms are **offline after setup**: no network, no LLM calls. The
`agent` classifier is a pure keyword/regex heuristic (see
`semantex_core::search::agent_classifier`); the default semantex build wires
**zero** LLM dependencies (`semantex-core/Cargo.toml` `default = []`) —
an LLM only activates with an explicit `--features llm` build AND
`SEMANTEX_LLM_BACKEND`/`SEMANTEX_LLM_MODEL` set, neither of which this
harness does. Instance processing order is always sorted by `instance_id`,
so `--limit N` is deterministic.

### Setup (once, per instance set)

```
cd benchmarks/swe_bench
python -m scripts.pre_index --phase a      # clones + indexes the Phase-A instances
                                            # under $SWE_BENCH_REPO_CACHE (default ~/.swe_bench_repos)
```

### Run

```
cd benchmarks/relevance

# offline smoke test against the tiny synthetic fixture (no network, no real
# git history -- see fixtures/tiny_swe_loc_instance.json + fixtures/tiny_corpus).
# Materialise the fixture corpus under a throwaway repo cache first:
export SWE_BENCH_REPO_CACHE=/tmp/tiny_swe_loc_cache
mkdir -p "$SWE_BENCH_REPO_CACHE/tiny__tiny-1"
cp fixtures/tiny_corpus/* "$SWE_BENCH_REPO_CACHE/tiny__tiny-1/"
python -m scripts.swe_loc_localize --local-fixture fixtures/tiny_swe_loc_instance.json

# real SWE-bench-Verified Phase-A instances (after pre_index.py above)
python -m scripts.swe_loc_localize --limit 3     # smoke sample
python -m scripts.swe_loc_localize               # full Phase-A set
```

Indexing cost: large scientific repos (astropy, sympy, ...) take tens of
minutes each to dense-index on a CPU-only machine — `pre_index.py` is the
expensive step and is why indexing is decoupled from scoring. The harness's
per-instance index guard defaults to 2h (`indexing.py::timeout_secs`);
instances whose index isn't ready are skipped and reported in the manifest
rather than silently dropped.

Reports land in `results/<run-id>/report.md` + `report.json` (aggregate,
per-arm Acc@k/MRR/tokens) and `per_instance.json` (every instance's ranked
files per arm, for drill-down). `results/` is gitignored — only the harness
code + this methodology are committed.

### A note on the ONNX Runtime dependency

`semantex index` always builds a dense embedding backend, which needs the
ONNX Runtime shared library. semantex normally auto-provisions this from
`github.com/microsoft/onnxruntime` releases (see `resolve_ort_dylib` in
`semantex-cli/src/main.rs`); in a sandbox where that download is blocked by
an egress policy but `pip` access to PyPI is allowed, install the Python
`onnxruntime` wheel (which bundles the same shared library) and point
`ORT_DYLIB_PATH` at it instead:

```
pip install onnxruntime
export ORT_DYLIB_PATH="$(python -c 'import onnxruntime,os;print(os.path.join(os.path.dirname(onnxruntime.__file__),"capi"))')/libonnxruntime.so.<version>"
```
