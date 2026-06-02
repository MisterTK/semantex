# semantex CPU-Workstation Optimization Roadmap — measured end-to-end (2026-06-01)

Built, ran, and tested the CPU-feasible optimization levers on this MacBook (no GPU — the
product's hard constraint) and synthesized a ranked roadmap. All numbers are CoIR
CodeTransOcean-DL, full 180 queries / 816-doc corpus, file-level gold + per-file dedup,
seed 20260531, CPU-only, verified against on-disk reports. Deltas are vs the
coderank-137m **adaptive-OFF** dense baseline (0.2805 / R@10 0.7333). Probe:
`benchmarks/relevance/scripts/lateon_probe.py` (PyLate). Companion: `2026-06-01-why-no-feature-uplift-rootcause.md`.

## Headline

For a CPU-only workstation tool, the biggest measured levers are (a) **measuring A/Bs
correctly** (adaptive-OFF) and (b) **swapping the dense channel to a late-interaction model**.
Late interaction is the **first clean retrieval A/B win** after reranking, weighted-RRF, MMR,
and graph all failed their gates — and the D4 "multi-vector is too heavy" objection **inverts**:
with PLAID PQ the LateOn index is ~10–19× *smaller* than the current single-vector index.

## Measured results

| Arm | Model (params, dim) | Index | nDCG@10 | R@10 | MRR@10 | Δ nDCG | Median lat (CPU) | Deployable index |
|---|---|---|---|---|---|---|---|---|
| **Baseline — dense (adaptive OFF)** | coderank-137m | HNSW single-vec | **0.2805** | **0.7333** | — | — | (fast) | 438 MB RSS |
| Shipped default — dense (adaptive **ON**) | coderank-137m | HNSW single-vec | 0.1883 | 0.4056 | 0.1237 | **−33%** | — | 438 MB |
| Default **RRF** (parameter-free) | coderank-137m | HNSW+BM25 | 0.2850 | 0.7444 | 0.1523 | +1.6% | — | — |
| Fixed **weighted-RRF** (both audit bugs fixed) | coderank-137m | HNSW+BM25 | 0.2565–0.2603 | 0.69–0.70 | — | **−7 to −9%** | — | — |
| **LateOn-Code-edge** (17M, dim-48) | PLAID (PQ) | — | **0.3436** | 0.8000 | 0.2098 | **+22.5%** | ~500 ms* | **23.5 MB** |
| LateOn-Code-edge (17M, dim-48) | Voyager fp32 | — | 0.3410 | 0.8056 | 0.2040 | +21.6% | **191 ms** | 241 MB (not deployable) |
| **LateOn-Code** (130M, dim-128) | PLAID (PQ) | — | **0.4112** | **0.8611** | 0.2775 | **+46.6%** | ~612 ms* | **50.7 MB** |
| LateOn-Code (130M, dim-128) | Voyager fp32 | — | 0.4018 | 0.8444 | 0.2695 | +43.2% | 313 ms | 347 MB (not deployable) |

\*PLAID per-query latency (~500–612 ms) is a **pessimistic small-corpus floor** — FastPLAID's
centroid+PQ-decode is fixed cost that 816 docs can't amortize; it scales better on real repos.
Index-build (816 docs, CPU): edge encode 8.3 s / total 51.7 s; 130M encode **86.2 s** (10×) — encode dominates real-repo index time, so **edge is the workstation default, 130M a quality-max opt-in**.
130M PLAID (0.4112) lands on its published CodeTransDL ~0.405 → wiring is faithful, not flattering.

Reference ceilings (CoIR paper Table 3, CodeTransDL per-task, NOT the 10-task aggregate):
BM25 0.50 / E5-Base 0.625 / Voyage-Code-002 0.728 / E5-Mistral 0.826. CodeRankEmbed's "0.601" is
the **aggregate** — the wrong target for this hard subtask, which is why "0.28 vs 0.60" overstated the gap.

## Roadmap (ranked by leverage-per-CPU-cost)

1. **[S, free] Lock `SEMANTEX_ADAPTIVE_SIZING=0` as the canonical A/B harness config** (keep adaptive ON in the product — it's the −18% agent-CCB feature). Recovers +38% nDCG / +45% R@10 of measurement distortion; makes every future A/B valid. No production change.
2. **[M] Prototype LateOn-Code-edge (17M) + PLAID as an opt-in `DenseBackend`** (`SEMANTEX_EMBEDDER=lateon-colbert`). The seam is intact post-D4 (DenseBackend/DenseIndexBuilder traits, `positional_chunk_ids()` multi-vector hook, per-backend `dense/<backend>/<fingerprint>/` layout) → "one enum variant + one match arm + two trait impls." Re-vendor `lightonai/next-plaid` (Apache-2.0, pure Rust, ONNX baked in, mmap, 2/4-bit residual quant). +22.5% nDCG / 23.5 MB index.
3. **[M] Run the chunked, real-repo A/B (LateOn-edge PLAID vs coderank, both adaptive-OFF) — the ONE remaining promotion blocker.** Closes (a) the whole-doc-vs-chunk parity gap (the CoIR probe retrieves whole docs → headline deltas are an UPPER BOUND) and (b) the PLAID small-corpus latency artifact. Quality is settled; chunk-parity + real-repo latency are not.
4. **[S] Offer LateOn-Code 130M (dim-128, `--embedding-size 128`) as a quality-max opt-in** (not default): +46.6% nDCG but ~300–612 ms/query + 86 s/816-doc encode.
5. **[S, do-nothing] Keep parameter-free RRF; do NOT ship weighted-RRF.** Even with both audit bugs fixed it loses −0.025 nDCG / −0.044 R@10 to RRF (reproduced on a shared index). Keep the fix diff LOCAL/uncommitted.
6. **[M] Cross-encoder reranking stays OFF on CPU permanently** (bge regresses; qwen3-reranker-0.6b >120 s/query). If reranking is ever wanted, route it through the SAME LateOn MaxSim backend, not a cross-encoder.
7. **[L, cpu-blocked] Do NOT adopt SPLADE-Code as the BM25 replacement.** +0.268 CoIR & ~50× faster query than BM25, BUT CC-BY-NC-SA-4.0 (non-commercial — disqualifying) + no ONNX + 596M encode-per-chunk indexing. BM25 stays the default sparse channel.
8. **[L, parked] qwen3-embed-0.6b dormant** (no last-token pooling in `single_vector.rs` + 4–5× index cost); **MMR / graph dormant** (downside-only on single-gold; need a multi-gold/SWE-loc eval — CPU-infeasible to build locally → VM).

## Query architecture: LateOn first-stage vs LateOn reranker (decide on real-repo data)

The choice is NOT "reranker vs no reranker" — it's **how you query the LateOn multi-vector index
you build either way** (same engineering: re-vendor next-plaid + the seam backend). Late
interaction's CPU advantage comes entirely from **precomputing doc multi-vectors at INDEX time**,
which forces three options:

- **(A) LateOn as first-stage** — PLAID-search the whole corpus. Best recall (measured R@10 0.80
  edge / 0.86 130M vs coderank 0.73) + precision; drops the 438 MB coderank channel (index = 23.5 MB).
  Open question = real-repo PLAID query latency (~500 ms is a micro-corpus fixed-cost floor; should
  amortize, not yet measured).
- **(B) coderank first-stage → LateOn MaxSim rerank top-k** (over PRECOMPUTED LateOn vectors).
  Likely lower query latency (coderank HNSW in ms + MaxSim over k precomputed docs) but **recall-capped
  at coderank's recall@k**, and keeps BOTH indexes (438 MB + 23.5 MB). A latency hedge, not a separate win.
- **(C) reranker that encodes candidates at QUERY time** — DEAD. Re-encoding k×≤2048-tok code docs per
  query (~500 ms–1 s on CPU, every query) is a cross-encoder in disguise; throws away the precompute
  advantage. **Do not build.**

**Decision criterion (measure both in the chunked real-repo A/B):**
1. **real-repo LateOn-PLAID query latency** — if acceptable → (A), it dominates on quality + index size.
2. **coderank's recall@50** = the (B) reranker's quality ceiling. The probe shows LateOn finds golds
   coderank's dense channel misses (R@10 0.80 vs 0.73); some may not be in coderank's top-50 at all, so
   (B) can never recover them. If **coderank R@50 ≥ LateOn R@10** → (B) can match (A)'s quality at lower
   latency → reranker wins. If **coderank R@50 < LateOn R@10** → (A) wins on recall.

**Recommendation:** build the multi-vector backend regardless (same work); make **first-stage-search
vs rerank-top-k a query-time knob**. Default to **(A) first-stage** (dominates on quality + drops 438 MB);
fall back to **(B) coderank→LateOn-MaxSim rerank** only if real-repo PLAID latency demands it. Never (C).

## Decisive conclusions (settled, with reproduced evidence)
- **Adaptive pruning** = measurement confound, not a product bug. OFF→ON = −38% nDCG / −45% R@10. ON in product, OFF for A/Bs.
- **Late interaction is IN** as the next dense channel (edge +22.5%, 130M +46.6%); edge is the workstation default, 130M an opt-in. **PLAID, not Voyager, is the deployable index** (equal/better nDCG at ~10× smaller).
- **The D4 28×-RSS objection is OBSOLETE** for dim-128 + 2-bit residual + mmap (~220–280 MB runtime, same league as 438 MB single-vector). D4 flipped on a quality WASH with the old heavy ColBERT; the leaner LateOn now WINS on quality, so the footprint argument reverses.
- **Weighted-RRF OUT** (confirmed after fixing both audit bugs). **Cross-encoder rerank OUT on CPU** (use LateOn MaxSim instead). **SPLADE-Code dead as a default** (license + ONNX + index cost). **qwen3-embed + MMR + graph dormant.**

## Honest counterpoint
The LateOn headline is an **UPPER BOUND**: the probe retrieves whole CoIR docs (no chunking) while
coderank runs the real chunk+adaptive pipeline; 312/816 docs >1500 chars get un-truncated context
the chunked baseline can't — so the real-repo chunked number will be lower (direction robust; the
130M matching its published 0.405 confirms faithful wiring, not a fair-to-coderank comparison). PLAID's
~500–612 ms is a micro-corpus floor; "amortizes on real repos" is a reasoned extrapolation, not yet
measured. The win is on ONE hard code-to-code subtask (CSN saturated; no multi-gold eval on CPU). Net:
late interaction is a real quality+footprint win and clearly worth building as an opt-in backend — but
the **chunked real-repo A/B + real-repo PLAID latency** are the two facts between "promising prototype"
and "default flip." Everything declared dead is dead for concrete, reproduced reasons; LateOn-on-real-repos is the one live-but-unproven lever.
