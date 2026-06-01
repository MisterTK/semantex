# Post-Overhaul Status & Follow-Up Specs (read first if you're picking up this work)

- **Date:** 2026-06-01
- **Purpose:** single source of truth for the state AFTER the S0–S8 SOTA overhaul shipped, and the precise spec for each remaining follow-up. A fresh session should read THIS doc + the project memory `sota-overhaul-shipped-2026-06-01` before touching anything.
- **History (do not rewrite):** the design lives in `docs/superpowers/specs/2026-05-31-semantex-sota-overhaul-design.md`; the per-stream TDD plans are `docs/superpowers/plans/2026-05-31-s{0..8}-*.md` and the coordinator is `2026-05-31-integration-and-cutover.md`. Those describe the as-DESIGNED overhaul (colbert-plaid was the starting default) and are intentionally NOT edited to erase that history — they are executed/done.

---

## 1. What shipped (origin/main @ `22ac3b8`)

The 9-stream overhaul (S0–S8) is merged to `origin/main`, built by parallel subagent teams via superpowers:subagent-driven-development with two-stage review (spec → quality) + a cargo/`actually` gate per stream.

**Current architecture / defaults:**
- **Dense default = `coderank-hnsw`** — single-vector CodeRankEmbed int8 ONNX (`hf:MisterTK/CodeRankEmbed-onnx-int8`, dim 768, mean-pool+L2, query prefix `"Represent this query for searching relevant code: "`) → instant-distance HNSW (int8 store + fp32 rescore). Selected via canonical **`SEMANTEX_EMBEDDER=coderank-137m`**. Index **schema v11** (v12 once PR #1 lands).
- **`colbert-plaid` retained, opt-in** via `SEMANTEX_EMBEDDER=lateon-colbert` — **BUT being removed in PR #1** (see §2). After PR #1 merges, coderank-hnsw is the sole dense backend.
- **DenseBackend trait seam** (`crates/semantex-core/src/search/dense_backend.rs`) — pluggable; the `multi_vector` capability + `positional_chunk_ids()`/`search_with_subset` hooks are retained for a future multi-vector/positional backend.
- **Config-driven model registry** (`crates/semantex-core/src/model/`) — every model (embedder/reranker/llm) is data (built-in permissive specs + optional `models.toml`), selected via `SEMANTEX_EMBEDDER`/`SEMANTEX_RERANKER_MODEL`/`SEMANTEX_LLM_MODEL`. Built-ins: `coderank-137m` (default), `qwen3-embed-0.6b`, `bge-reranker-v2-m3`, `qwen3-reranker-0.6b`, feature-gated LLM. (`lateon-colbert` removed in PR #1.)
- **Default build is zero-LLM** (`cargo tree | grep genai` empty without `--features llm`). This is a hard CLAUDE.md rule — keep it.
- Relevance harness `benchmarks/relevance/` (CoIR / CodeSearchNet / SWE-loc) is the measurement judge.

**Off-by-default (each opt-in via env), with the MEASURED reason — do NOT re-flip without a NEW measured win:**
| Feature | Default | Why off (measured 2026-06-01, coderank-hnsw, CoIR-CodeTransDL 180q + CSN-py 200q) |
|---|---|---|
| Reranker (`SEMANTEX_RERANK=1`+`SEMANTEX_RERANKER=on`+`SEMANTEX_RERANKER_MODEL`) | OFF | qwen3 & bge **work** (non-no-op), but CPU latency is fatal: bge ~47 s/query, qwen3 >120 s at the hardcoded 100 candidates (~1000× vs 15-50 ms warm). Gate fails on latency, not quality. → follow-up #15. |
| Weighted-RRF (`SEMANTEX_FUSION=weighted-rrf`) | OFF (default `Rrf`) | Helps CSN (+0.04 nDCG) but **regresses external CoIR anchor** (−0.023 nDCG/−0.10 R@10) — Semantic weights over-weight the weak sparse channel. Not net-non-negative. |
| MMR (`SEMANTEX_MMR_LAMBDA`) | OFF | Hurts both corpora (R@10 collapses) — diversity evicts the single gold doc on precision tasks. Needs an exhaustive/diversity eval to show upside. |
| Graph centrality/cohesion/2-hop (`SEMANTEX_GRAPH_CENTRALITY_WEIGHT`/`_MODULE_DECAY`/`_HOPS=2`) | OFF | No measured SWE-loc lift (SWE-loc infeasible in-budget); on CSN single-gold, centrality is mildly negative. Graph-propagation *default* (graph on, levers off) is unchanged + confirmed neutral vs `SEMANTEX_GRAPH_DISABLE`. → follow-up #16. |
| Semantic cache (`SEMANTEX_SEMANTIC_CACHE=1`) | OFF | Correctness-tested (reindex/schema flush); opt-in latency cache. If ever promoted on-by-default, mtime-gate the per-query meta.json stamp read. |

---

## 2. PR #1 — ColBERT/next-plaid removal (in review, the D4 end-state)

Branch `chore/remove-colbert-plaid`, **PR #1** (https://github.com/MisterTK/semantex/pull/1). Removes the colbert-plaid impl + vendored `next-plaid` entirely (coderank-hnsw becomes sole dense backend; seam preserved; schema 11→12; old colbert indexes degrade to clean reindex). Two-stage review ship-ready; gate green; ort verified to still work without `next-plaid-onnx`. **Status when you read this: confirm whether PR #1 is merged** — it changes whether colbert-plaid code still exists. The follow-ups below are written to be robust either way.

---

## 3. Re-running the harness A/B (READ — there are real gotchas that invalidated naive runs)

Harness: `benchmarks/relevance/` (venv `benchmarks/relevance/.venv`, `pip install -e ".[dev]"`). Build the CLI: `cargo build -p semantex-cli --release` → `target/release/semantex`. On macOS set `ORT_DYLIB_PATH=/opt/homebrew/lib/libonnxruntime.dylib`.

**GOTCHA 1 (follow-up #14 — not yet fixed on main):** the harness `spawn_daemon_if_needed` spawns the bare name `semantex` (PATH → `~/.cargo/bin/semantex`), **NOT** `target/release/semantex`. A query served by a stale installed binary gives invalid A/B numbers. WORKAROUND until #14 lands: pre-start a branch daemon (`target/release/semantex serve …`, absolute path, with the test env baked into the daemon's spawn env) per corpus dir, and verify you're hitting it.
**GOTCHA 2:** the embedder is authoritative at INDEX time. The harness fix `540e4b0` threads `SEMANTEX_EMBEDDER` into the index subprocess + namespaces corpora per embedder — keep that; a coderank search over a colbert index silently returns colbert results.
**GOTCHA 3:** reranking needs `SEMANTEX_RERANK=1` AND `SEMANTEX_RERANKER=on` AND `SEMANTEX_RERANKER_MODEL=<id>` in the DAEMON's env, plus `SEMANTEX_MAX_RSS_MB>=8192` (reranker models push RSS to 3-6 GB and the default 1024 MB cap aborts the daemon). Fusion/MMR/graph env vars are also read daemon-side.
**Datasets:** CoIR `CodeTransOceanDL` (180q — external anchor; MTEB BM25 baseline nDCG@10 = **0.34418**, cited from `mteb/baseline-bm25s`) and CSN python (200q — internal determinism). SWE-loc needs the `benchmarks/swe_bench/` Phase-A pre-indexed corpus (absent → must be built; ~hours).
**Gate (S0):** `python -m scripts.reproduce_baseline` — split tight-internal (semantex's own CodeTransDL nDCG ≈0.188 ±0.025) + loose-external (vs 0.344 ±0.18). A 0.19→0.12 regression fails the tight band.

---

## 4. Follow-up specs (each is its own PR; use subagent-driven-development + two-stage review)

### F1 — Colbert comment scrub (IN PROGRESS, folded into PR #1)
Scrubbing stale/dangling `colbert`/`plaid` references in `crates/` comments (they dangle once `colbert.rs` is deleted on the removal branch). Code + user-facing strings only; `docs/` historical plans left as-is. If PR #1 already merged with this, F1 is done.

### F2 (#15, HIGH leverage, small) — `SEMANTEX_RERANK_CANDIDATES` knob → re-A/B rerank
**Why:** reranking is correct but non-viable on CPU at the hardcoded `rerank_candidates=100` (~47-120 s/query). A smaller candidate window may make it net-positive within a deployable latency budget — this is the §10 "reranking net-positive on code" claim.
**What:** add an env-tunable `SEMANTEX_RERANK_CANDIDATES` (config field + overlay; default e.g. 20-30, NOT 100) where the reranker currently slices the top-100 (in `hybrid.rs`/`reranker_engine.rs`). Then re-run the §6.2 A/B (rerank-off vs qwen3 vs bge) at the smaller count: nDCG@10/MRR delta + per-query latency. **Gate:** flip rerank default-on (with the winning model) only if net-positive nDCG/MRR within an acceptable warm-latency budget; else keep off, document the latency/quality curve. Optionally also wire a CoreML/CUDA execution provider.

### F3 (#14, small) — fix harness daemon-binary spawn
**Why:** GOTCHA 1 above — every future A/B is invalid unless worked around. **What (benchmark-only):** `benchmarks/relevance` `spawn_daemon_if_needed` must spawn the CONFIGURED `semantex_binary` (the branch `target/release/semantex`), not the bare PATH name; ideally assert the daemon's binary path/version matches. Consider consolidating the confusing `SEMANTEX_RERANK` vs `SEMANTEX_RERANKER` env surface while here.

### F4 (#5, small-med) — investigate adaptive pruning under `--sparse-only`
**Why:** `search/adaptive.rs::apply_adaptive_pipeline` (confidence threshold `score ≥ top_score×min_score(query_type)` + elbow `adaptive_max_results`) runs EVEN under `--sparse-only`, trimming ~100 candidates to ~15, so dense-only ≈ hybrid on small corpora and Recall@k for k>15 is bounded. **What:** confirm the behavior; decide whether the harness/measurement should be able to disable adaptive sizing for clean Recall@k (e.g. an env knob), or whether it's applied identically across A/B arms (it is, post-fusion — so relative comparisons are fair, but absolute recall is capped). NOT a product bug per se; matters for measurement fidelity + whether sparse/fusion levers can ever show effect.

### F5 (#10, med) — S8 versioned-dir hot-swap completion
**Why:** the `embedder_fingerprint` is stamped in meta.json but NOT wired into the open-time/staleness guard, and the builder writes to plain `dense_subdir` not the versioned `active_dense_dir(<fingerprint>/)` with the atomic ACTIVE pointer. So toggling `SEMANTEX_DENSE_CONTEXT` (same backend, different fingerprint) on an EXISTING index reuses the wrong index silently. **Not needed for the Phase-3 A/B** (harness builds separate corpora) and `DENSE_CONTEXT` is off-by-default/experimental. **What:** wire `active_dense_dir` + `read/write_active_pointer` into the builder, and fold `embedder_fingerprint` into `index/state.rs::is_stale` (needs config/registry access there) so a fingerprint mismatch triggers a clean rebuild, not a hard error. Helpers already exist in `dense_backend.rs` (S8).

### F6 (#16, HEAVY — hours) — SWE-loc measurement (graph lift + §10 localization claim)
**Why:** §6.4 graph centrality/cohesion/2-hop could NOT be evaluated — SWE-loc is the only corpus that shows graph-localization upside, and its corpus is absent. The §10 "+Z pp SWE-bench localization Recall@10" claim is unmeasured. **What:** stand up the `benchmarks/swe_bench/` Phase-A pre-indexed corpus (`pre_index.py` — clones+indexes 100 instances across ~8 large repos; django/sympy 30-60 min each), then run `scripts/run.py --dataset swe-loc` for graph-off vs graph-on vs graph+centrality/cohesion/2-hop at file-level Recall@{5,10}. **Gate:** flip the graph levers on by default only with a measured SWE-loc lift AND no CoIR/CSN regression; else keep off.

### F7 (trivial) — sweep historical-doc colbert mentions IF desired
Only if you want the design docs to read post-hoc — generally leave them (historical accuracy). Not recommended.

---

## 5. Hard rules that still apply (CLAUDE.md)
`crates/` repo-agnostic, no hardcoded paths, permissive-license defaults, **default build zero-LLM** (`cargo tree | grep genai` empty). Verify every change with `cargo build/test/clippy/fmt` + `actually verify_change`. Never flip a default without a measured win on the S0 harness.
