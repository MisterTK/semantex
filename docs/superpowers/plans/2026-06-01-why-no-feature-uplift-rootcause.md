# Why none of the retrieval features showed A/B uplift — root cause + remediation (2026-06-01)

Audit (8-agent workflow + live verification) of why semantex's OFF-by-default features
(reranking, weighted-RRF, MMR, graph centrality/cohesion/2-hop, HyDE) all came back
null-or-negative in A/Bs despite the literature reporting meaningful gains. **The effort
was not wasted on bad implementations** — the nulls split into three layered causes, the
two biggest being measurement mismatch and model choice (not the originally-suspected
"how we wired/used/combined it", though those are real for specific features).

## Verdict (ranked root causes)

1. **MEASUREMENT MISMATCH (dominant).** Both working evals (CoIR-CodeTransDL 180q, CSN-py
   200q) are **single-gold** (one correct doc/query); CSN is near-saturated. On single-gold
   nDCG@10, reordering features are precision-only and structurally cannot show their gains —
   and MMR/graph can ONLY show downside.
   - Smoking gun (graph): on CSN, `graph_default == graph_disabled == hops2 == module_decay`
     nDCG = **0.9090621818019904 byte-identical**, recall flat at 0.96 across all 8 conditions
     (`benchmarks/relevance/results/graph-ab/graph_ab_report.json`). Graph provably moves
     nothing on single-gold; on CoIR graph_default (0.21677) actually *beats* graph_off
     (0.21204, +0.0047). The lever works; the benchmark can't register it.
   - MMR is downside-only by construction on single-gold (diversity can only demote the one
     gold). `mmr.rs` is **correct — do not "fix" it.** Same for the graph stage.

2. **SELF-INFLICTED BASE SUPPRESSION (verified live, high-leverage, cheap to fix).** Our own
   adaptive pruning clips ~45% of recoverable recall before any feature runs:
   | CoIR dense-only | nDCG@10 | Recall@10 |
   |---|---|---|
   | adaptive ON (default) | 0.2032 | 0.5056 |
   | adaptive OFF (`SEMANTEX_ADAPTIVE_SIZING=0`) | **0.2805 (+38%)** | **0.7333 (+45%)** |
   The ≤15 cap doesn't bind at k=10, so it's the **confidence threshold + per-file dedup**
   (`adaptive.rs`) discarding correct golds *already in the candidate set*. Every feature A/B
   ran on this pre-clipped base → reranking can't rescue dropped golds; MMR/graph reorder a
   recall-starved set. NOTE: pruning is NOT a bug — it's the terse-output feature behind the
   −18% agent CCB win. The error was measuring feature A/Bs with it ON. (0.28 is still <
   published ~0.60 → ~half the residual gap is our 180q-subset/chunking protocol or a mid-tier
   embedder.)
   - **Phase-1 generalization sweep (verified, both ablations × both corpora):** recovery is
     broad WHERE THERE IS HEADROOM, null where saturated — i.e. it bites exactly the corpus
     that matters for A/B. CoIR dense 0.2032/R@10 0.5056 → 0.2805/0.7333 (+38%/+45%); CoIR
     hybrid 0.2054/0.5111 → 0.2760/0.7222 (+34%/+41%); CSN-py dense & hybrid both ~0.9091/0.96
     → ~0.9106/0.965 (+0.2%/+0.5%, saturated). Corollaries: (a) CSN is SATURATED → useless as
     an A/B corpus (regression gate only); (b) on CoIR adaptive-off, hybrid (0.276) < dense
     (0.281) — sparse mildly HURTS on code-translation, independently confirming weighted-RRF's
     "sparse is the weak channel" finding. **Adaptive-OFF is the correct measurement config for
     all feature A/Bs.**

3. **MODEL CHOICE (real).**
   - Reranker: `bge-reranker-v2-m3` is a generic multilingual TEXT reranker — **CoIR 35.97,
     the weakest code reranker in the 2026 literature** (jina-v3 paper Table 2), 512-token
     hard cap (truncates code), applied to a code→code task that mismatches its NL→code
     training. Measured −73% nDCG / recall@10 0.5278→0.1222 = anti-correlated scoring, not
     weak degradation. `qwen3-reranker-0.6b` is ALREADY built-in (OnnxReranker, yes/no-logit)
     and was never A/B'd; jina-reranker-v3 (CoIR 63.28, +27pts) is the real upgrade.
   - Embedder: `coderank-137m` is mid-tier (CoIR ~60 published; ~0.19-0.28 measured here).
     `qwen3-embed-0.6b` (75.41 MTEB-Code) is a **built-in env swap** (`SEMANTEX_EMBEDDER=
     qwen3-embed-0.6b`, same coderank-hnsw backend, zero new code) — never A/B'd.

## Confirmed bugs (file:line)
1. `crates/semantex-core/src/search/hybrid.rs:301` — weighted-RRF A/B is **asymmetric**:
   `expanded_text` (Exp4Fuse dual-route query expansion) is gated to
   `matches!(fusion_mode, FusionMode::Rrf)`, so the weighted-RRF arm never ran
   `exp4_weighted_rrf_fuse` and was compared against an RRF baseline that DID get expansion.
2. `crates/semantex-core/src/search/query_classifier.rs:30-46` — weighted-RRF weights are
   hand-set with `w_sparse >= w_dense` for every query type (Semantic 0.1/0.9 = 9:1 sparse
   bias), encoding an obsolete "sparse is reliable for code" assumption; up-weights the
   measurably weak sparse channel (sparse-only 0.1884 < hybrid 0.215). Usage/config issue.
3. `benchmarks/relevance/src/relevance_harness/semantex_client.py:23` — HyDE is structurally
   excluded from measurement (`_ABLATIONS` omits it; `semantex search` → `run_with_searcher`
   → plain `searcher.search`, never the AgentPipeline HyDE needs). Its "null" = no experiment.
4. LATENT: `hybrid.rs:1560-1595` `merge_hyde_results` re-sorts base∪hyde by raw `score`
   without re-normalizing the two fused score distributions — would mis-rank if HyDE were
   ever enabled. Dormant because HyDE is never measured.

## Remediation (ranked — do in order)
1. **[S, do first] Re-run ALL feature A/Bs with `SEMANTEX_ADAPTIVE_SIZING=0`.** Free; uses the
   F4 knob shipped today; raises the base ~38-45% AND un-clips the rank region where
   rerank/graph/MMR operate. This is the prerequisite that makes every other A/B meaningful.
2. **[S] Flip `SEMANTEX_EMBEDDER=qwen3-embed-0.6b`** (built-in, +15 MTEB-Code, zero code) and
   re-measure base CoIR/CSN. Restores headroom downstream levers need.
3. **[M] Swap the reranker** to qwen3-reranker (built-in) or jina-reranker-v3; A/B on an
   NL→code corpus (CSN, orientation-matched) at window ≤15 (the "Drowning in Documents"
   degradation curve). Distinguishes "reranking doesn't help" (false) from "bge is wrong for
   code" (the real finding). CPU latency (~47-120s/q) remains a separate deployment blocker.
4. **[L] Build a multi-gold / SWE-loc eval on a GPU VM** — the only benchmark shape that can
   register MMR / graph localization / recall-side rerank gains. Gate reordering-feature A/Bs
   on it; keep CSN/CoIR single-gold as regression gates only, NOT A/B arbiters for reordering.
5. **[M] Fix weighted-RRF** (invert/adapt weights toward dense; drop the `:301` expansion
   guard so the A/B is symmetric) before declaring it dead.
6. **[M] Make HyDE measurable** (add a `hyde` ablation; build `--features llm`; set
   `SEMANTEX_LLM_BACKEND`); generate N=3-5 hypothetical docs and average their embeddings.

## Honest counterpoint (do NOT assume all 6 are hidden wins)
Several nulls are genuinely real: bge should not be the default (and a good reranker still has
deployment-blocking CPU latency); **weighted-RRF may be irrecoverable as a repo-agnostic
feature** (helped CSN where the docstring leaks into the doc → sparse strong; hurt CoIR where
it doesn't → a single global weight table cannot win both regimes, so parameter-free RRF as
the default may be the *correct* outcome); HyDE with an already code-tuned encoder is exactly
where the literature says its gains vanish; MMR has no benefit for single-answer "find the
function" search. **Strongest steelman: for semantex's core use case, "strong code embedder +
parameter-free RRF + (eventually) a code reranker" may be the whole story, with the rest
correctly shipped OFF.** The measurement fixes are worth doing to *know* which features are
dead vs dormant — but the prior should not be that all six are wins.
