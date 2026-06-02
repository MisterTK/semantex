# Item 3 — chunked real-repo A/B results: LateOn-colbert vs coderank (2026-06-02)

The roadmap's ONE promotion blocker (`2026-06-01-cpu-optimization-roadmap.md` §3 + query-architecture). Run on the REAL chunked pipeline (`semantex index`+`search`, both backends), CoIR-CodeTransDL (180q, 816-doc corpus, file-level gold + per-file dedup, seed 20260531), **dense-only, adaptive-OFF** (item-1 lock). Binary `target/release/semantex` @ `37b5036`. Reproducible across 2 runs (cached-index re-measure idx_secs=0.0). Driver: `/tmp/item3_measure.py` (harness `run_corpus` at retrieve_k=50).

## Measured

| arm (dense-only, 180q, adaptive-OFF) | nDCG@10 | R@10 | R@50 | MRR@10 | warm ms | cold ms | index s |
|---|---|---|---|---|---|---|---|
| coderank-137m (HNSW, dim-768) | 0.2806 | 0.739 | 1.000 | 0.1480 | **903** | 1433 | 66.3 |
| **lateon-colbert (PLAID, dim-48)** | **0.2981** | 0.750 | 1.000 | **0.1659** | **92** | 722 | 25.3 |
| **Δ (lateon vs coderank)** | **+6.2%** | +1.5% | — | **+12%** | **0.10×** | 0.50× | 0.38× |

Index footprint: PLAID ~23.5 MB vs single-vector HNSW ~438 MB (the D4 RSS objection inverts).

## The three decision inputs (roadmap §query-architecture)

- **(a) chunked quality** — lateon WINS, modestly: **+6.2% nDCG@10 / +12% MRR@10**, R@10 ~tied (~0.74). The whole-doc probe's +22.5% was an UPPER BOUND (predicted); chunking compresses it to ~+6%. The win is in RANKING (nDCG/MRR rank the single gold higher), which single-gold eval can barely register — likely an UNDER-statement of the real benefit.
- **(b) real-repo PLAID latency — ACCEPTABLE; the roadmap's latency worry is FALSIFIED.** lateon warm **92 ms vs coderank 903 ms** (~10× FASTER, reproducible). On CPU the per-query bottleneck is QUERY-ENCODE, not PLAID search: CoIR queries are long code snippets, and the 17M ColBERT model encodes ~10× cheaper than 137M CodeRankEmbed. Index time also faster (25 s vs 66 s). NOTE: subprocess-inclusive (each query = a `semantex` CLI round-trip); the ~810 ms gap is the 137M-vs-17M encode delta. coderank's 903 ms ≠ the memory's "15–50 ms" because that figure is HNSW-search-only, excluding query-encode.
- **(c) coderank recall@50 = 1.000** ≥ lateon R@10 (0.750) → reranker (B) coderank→LateOn-MaxSim is architecturally VIABLE (coderank top-50 contains every gold). BUT (A) LateOn first-stage DOMINATES it here: faster query, faster index, 23.5 MB vs 438+23.5 MB (B keeps both), higher quality. R@50=1.0 is a small-corpus artifact (816 docs).

## Verdict
**(A) LateOn-colbert as first-stage is the measured winner on this corpus: higher quality (+6.2% nDCG / +12% MRR), ~10× lower query latency, faster indexing, ~19× smaller index.** Every concern the roadmap flagged for (A) (PLAID latency, footprint) inverted in its favor. (C) query-time-encode reranking stays dead.

## Caveats before flipping the SHIPPED DEFAULT (why this is a user cutover, not an autonomous flip)
1. **ONE corpus.** CoIR-CodeTransDL is single-gold, code-to-code, 180q. CSN is saturated (useless as A/B). No multi-gold / SWE-loc (CPU-infeasible → VM, item 4). Generalization to "tens of thousands of diverse repos" is unproven on one benchmark.
2. **Quality magnitude is modest** (+6.2% nDCG; R@10 tied). The win is real and direction-robust across runs, but not large.
3. **The latency win may be query-distribution-dependent.** It comes from long CODE queries (137M encode is slow); semantex's real traffic is agent NL+code queries. For SHORT NL queries the encode-cost gap shrinks — the 10× may not hold. Worth a short-query latency check.
4. **dim-48 edge model only.** LateOn-Code 130M (dim-128, probe +46.6%) is the quality-max option, untested here; a default flip should weigh edge-vs-130M.
5. Agent-level CCB/answer-quality (the product's actual metric, `benchmarks/claude_bench.py`) is unmeasured for lateon-colbert.

## Recommended next steps (pending user call on the cutover)
- If flipping toward lateon: first broaden — (i) a short-NL-query latency probe, (ii) re-run on ≥1 other CoIR sub-task or a real indexed repo with known-item queries, (iii) an agent-CCB A/B (claude_bench) lateon vs coderank. Then flip the default with a multi-corpus number.
- Keep lateon-colbert OPT-IN until the above; it ships and works today (`SEMANTEX_EMBEDDER=lateon-colbert`).
- Item 4 (SWE-loc on a VM) remains the venue for the at-scale + multi-gold confirmation.

## Addendum — short-NL-query latency probe (2026-06-02, de-risks the cutover)
Caveat 3 (the latency win may be long-CODE-query-specific) — TESTED and resolved FAVORABLY.
10 short agent-style NL queries ("authenticate a user", "database connection pool", "handle
error and retry", …), warm daemon, subprocess round-trip (equal overhead both arms), CoIR index:

| query type | coderank-137m (HNSW) | lateon-colbert (PLAID) |
|---|---|---|
| long code (CoIR, 180q) | 903 ms warm | 92 ms (~10× faster) |
| **short NL (10q)** | **56 ms** median (51–83) | **48 ms** median (40–52, tighter) |

For SHORT NL queries (semantex's real agent traffic) lateon is still slightly FASTER (48 vs 56 ms)
and more consistent. coderank's 903 ms was specific to LONG code queries (137M encode dominates);
for short queries its encode is cheap (~56 ms) but lateon's 17M encode is cheaper still. **The
latency win holds across query distributions — lateon is never slower.** The default flip's
load-bearing latency argument is confirmed; remaining caveats (one single-gold corpus, modest
quality magnitude) stand. Probe is ad-hoc bash (timing `semantex --dense-only` per query).

## Addendum 2 — hybrid (production fusion) A/B confirms the win (2026-06-02)
The shipped search path is HYBRID (dense+BM25 fused), not dense-only. Re-ran the A/B with
`ablation="hybrid"` on the cached indexes (adaptive-OFF):

| arm (hybrid, 180q) | nDCG@10 | R@10 | MRR@10 | warm ms |
|---|---|---|---|---|
| coderank-137m | 0.2801 | 0.739 | 0.1473 | 901 |
| lateon-colbert | **0.2966 (+5.9%)** | 0.744 | **0.1654 (+12%)** | 88 |

The win HOLDS in the production fusion path (+5.9% nDCG@10, ~same as dense-only's +6.2%) — on
CoIR-CodeTransDL the sparse channel adds little (hybrid ≈ dense; matches the prior "sparse mildly
hurts on code-translation" finding). Latency unchanged (dense query-encode dominates fusion).

**Cutover validation summary:** quality +5.9–6.2% nDCG (dense AND hybrid) / +12% MRR; latency
faster on BOTH long-code (10×) and short-NL (48 vs 56 ms) queries; ~19× smaller index; graceful
migration (no re-chunk). The default flip is thoroughly validated on this corpus. Remaining gap:
generalization (one single-gold corpus) + agent-CCB — the explicit, reversible residual risk.
