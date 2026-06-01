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
