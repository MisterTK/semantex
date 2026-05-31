# SOTA Overhaul — Integration, Sequencing & Cross-Stream Reconciliation

> **READ THIS FIRST.** This is the coordination layer for the 8 stream plans. It is authoritative for cross-stream interfaces, build order, lead decisions that resolve gaps the per-stream plans surfaced, and the cutover criteria. Each stream team executes its own plan via **superpowers:subagent-driven-development**; this doc is what keeps them coherent.

- **Date:** 2026-05-31
- **Spec:** `docs/superpowers/specs/2026-05-31-semantex-sota-overhaul-design.md`
- **Execution model:** one subagent team per stream, in the phases below; run the `actually` MCP `verify_change` after each stream merges to the integration branch.

---

## 1. Plan index

| Stream | Plan file | Lang | One-liner |
|--------|-----------|------|-----------|
| S0 | `2026-05-31-s0-relevance-harness.md` | Python | CoIR/CSN/SWE-loc relevance harness (the judge) |
| S1 | `2026-05-31-s1-dense-backend-seam.md` | Rust | `DenseBackend` trait; PLAID → impl #1 `colbert-plaid` |
| S2 | `2026-05-31-s2-coderank-hnsw-dense.md` | Rust | CodeRankEmbed single-vector + HNSW + int8 (`coderank-hnsw`) |
| S3 | `2026-05-31-s3-onnx-reranker-upgrade.md` | Rust | generic ONNX cross-encoder + Qwen3-Reranker-0.6B |
| S4 | `2026-05-31-s4-code-graph-fusion.md` | Rust | promote graph_propagation to a measured signal |
| S5 | `2026-05-31-s5-hyde-wiring.md` | Rust | **MCP-server runtime wiring** (HyDE core already done) |
| S6 | `2026-05-31-s6-simd-kernels.md` | Rust | AVX2/NEON/scalar dot/cosine/l2 (+int8) kernels |
| S7 | `2026-05-31-s7-fusion-polish.md` | Rust | weighted-RRF + MMR + semantic cache |

**Shared file:** `docs/superpowers/plans/2026-05-31-research-notes.md` does not exist yet. The **first spike task that runs creates it**; S2/S3/S5 each append a clearly-headed section (`## S2`, `## S3`, `## S5`). Treat it as append-only.

---

## 2. Build phases & sequencing

```
Phase 1 (unblock, parallel):  S0 (harness)   +   S1 (DenseBackend seam)
Phase 2 (parallel build):     S2 [needs S1, consumes S6] · S6 · S3 · S4 · S5 · S7 [needs S1]
Phase 3 (integration):        run S0 A/B → tune → cutover decisions → schedule ColBERT/next-plaid removal
```

- **S1 lands before S2/S4/S7** (they edit `hybrid.rs`/`builder.rs`/`triple_fusion.rs` through the seam S1 creates).
- **`hybrid.rs` contention:** S1 first; then **S2, S4, S7 each rebase on S1's landed shape** and edit *distinct regions* (S2: dense channel + builder match arms; S4: the post-fusion graph stage ~lines 861-913; S7: rerank/return region + fusion mode). Serialize their `hybrid.rs` merges; run `verify_change` after each.
- **S6 before (or alongside) S2's distance code.** S2 may start on a scalar `search/simd.rs` shim and swap in S6's kernels when they land; the shared contract is fixed in §3.
- **S3, S5 are independent** of the dense work and can land any time in Phase 2.

---

## 3. LOCKED cross-stream interfaces (authoritative — overrides the spec's sketches)

1. **`ScoredChunkId` is the existing 5-field `crate::types::ScoredChunkId`** `{ chunk_id, score, score_dense, score_sparse, score_exact }`, aliased `DenseHit` — **not** the spec §3 two-field sketch. Dense backends populate `chunk_id` + `score`. S1/S2/S7 all use this.

2. **`DenseBackend` trait (final):**
   ```rust
   pub trait DenseBackend: Send + Sync {
       fn name(&self) -> &'static str;
       fn search(&self, query: &str, k: usize) -> Result<Vec<DenseHit>>;
       fn search_with_subset(&self, query: &str, k: usize, subset: &[u64]) -> Result<Vec<DenseHit>>;
       fn positional_chunk_ids(&self) -> Option<&[u64]> { None }
       // ADDED for S7 (MMR / semantic cache). Default None; colbert-plaid returns a
       // mean-pooled+L2-normalized projection; coderank-hnsw returns its exact vectors.
       fn embed_text_vector(&self, _query: &str) -> Option<Vec<f32>> { None }
       fn embed_doc_vectors(&self, _chunk_ids: &[u64]) -> Option<Vec<(u64, Vec<f32>)>> { None }
   }
   pub trait DenseIndexBuilder: Send + Sync {
       fn name(&self) -> &'static str;
       fn build(&mut self, chunks: &[(u64, &str)]) -> Result<()>;
       fn insert(&mut self, chunks: &[(u64, &str)]) -> Result<()>;
       fn delete(&mut self, chunk_ids: &[u64]) -> Result<()>;
       fn persist(&self, dir: &Path) -> Result<()>;
   }
   ```
   **Action:** S1 defines `embed_text_vector`/`embed_doc_vectors` on the trait (default `None`) so S7 doesn't have to retro-patch it; S2 implements them with exact int8-store vectors; the `colbert-plaid` impl provides the mean-pool projection.

3. **`DenseBackendKind`:** `ColbertPlaid` (S1, name `"colbert-plaid"`) + `CoderankHnsw` (S2, name `"coderank-hnsw"`). Selection via `dense_backend` config + `SEMANTEX_DENSE_BACKEND` env; default `colbert-plaid` until Phase 3 cutover. On-disk: `.semantex/dense/<name>/`.

4. **Index schema version:** **S1 bumps 9→10; S2 bumps 10→11.** Final shipped value is **11**. If S1+S2 land together, one bump to 11. Never two competing "10"s.

5. **SIMD kernels (S6) — public API S2 calls:**
   ```rust
   pub fn dot_f32(a: &[f32], b: &[f32]) -> f32;
   pub fn cosine_f32(a: &[f32], b: &[f32]) -> f32;  // SIMILARITY in [-1,1], NOT distance
   pub fn l2_f32(a: &[f32], b: &[f32]) -> f32;       // Euclidean distance
   pub fn dot_i8(a: &[i8], b: &[i8]) -> f32;
   pub fn cosine_i8(a: &[i8], b: &[i8]) -> f32;
   ```
   Module: `crates/semantex-core/src/search/simd.rs`. **S2 must use `1.0 - cosine_f32(..)` where it needs a distance.**

6. **S3 selector:** the spec said "add models to `select_model_from_env`," but that fn returns `fastembed::RerankerModel` (no Qwen3). S3 introduces a `RerankerChoice` enum + `select_reranker_choice_from_env()` layer instead — same intent. The `SEMANTEX_RERANKER` master switch + off-by-default identity pass-through are preserved.

---

## 4. Lead decisions (resolving the gaps the plans surfaced)

- **D-int8 (S2 ↔ S6):** S2's stored int8 vectors use **symmetric quantization (zero-point 0)** so S6's `dot_i8`/`cosine_i8` apply directly without dequant bias. (L2-normalized embeddings are ~zero-centered; symmetric is the right call.)
- **D-rrf-k (S7):** align `config.rrf_k` default **30.0 → 60.0** to match the `RRF_K` const, so weighted-RRF A/B is apples-to-apples vs the current parameter-free RRF.
- **D-mmr-cache (S7):** MMR + semantic-cache yield **exact** behavior only on `coderank-hnsw` (single-vector). Validate these two features on the **S2 backend** primarily; on `colbert-plaid` they use the mean-pool projection (approximate) and may be left off.
- **D-graph (S4):** add a real **`SEMANTEX_GRAPH_DISABLE`** on/off knob (none exists today) so the "graph-off vs graph-on" gate is measurable. The route names "architectural/exhaustive/feature-planning" are **free-function predicates**, not `QueryType` enum variants (the enum stays 4 variants). S4's SWE-loc gate is measured at **file-level** recall (S0 ships file-level; function-level is an S0 follow-up).
- **D-s5 (S5 reframe):** HyDE's core (`search_with_hyde`/`merge_hyde_results`) and the **daemon** path are already complete and safety-correct. S5's real work is: wire a shared Tokio runtime into the **MCP server** (`semantex-mcp/src/server.rs` — `tool_agent` never chains `.with_runtime()`), fix the `semantex-mcp` `llm` feature to pull `tokio` (`llm = ["semantex-core/llm", "dep:tokio"]`), and add the missing end-to-end LLM-error/timeout safety tests. The spec's S5 section is corrected to match.
- **D-s0-gate (S0):** the reproducible acceptance gate anchors on a **deterministic BM25/CSN** baseline (`--sparse-only`, model-independent). CoIR is the **headline** metric where HF access is available; if HF is gated at run time, the CoIR loader is built+unit-tested but the headline number is recorded once reachable. Never silently truncate subsets — log the manifest.

---

## 5. Human / ops prerequisites (block the relevant stream's download path)

- **CodeRankEmbed int8 ONNX (S2)** and **Qwen3-Reranker-0.6B int8 ONNX (S3)** are **not pre-hosted** as ONNX. The spike tasks export + quantize them locally; a human must then **upload the artifacts to a project-controlled, permissively-licensed HF repo** and record the resolved URLs in `2026-05-31-research-notes.md` before the in-product download path works. Until then, S2/S3 default-path integration tests run against the locally-exported artifacts.
- **`tokenizers` C-backend:** the S2/S3 spikes must record a `tokenizers` feature set with the `onig` C dependency **disabled** (airgap / no-C-C++ rule); use the pure-Rust tokenizer path.

---

## 6. Cutover criteria (Phase 3 — the decisions the harness makes)

After S2–S7 land and S0 is green, run the full A/B on the harness and decide:
1. **Dense default (D4):** flip `SEMANTEX_DENSE_BACKEND` default to `coderank-hnsw` **only if** it meets-or-beats `colbert-plaid` on CoIR + CSN Recall@10/nDCG@10. If it loses, keep `colbert-plaid` and reassess (do **not** delete ColBERT).
2. **Reranker (S3):** flip the `SEMANTEX_RERANKER` default to on with the winning model **only if** net-positive nDCG@10/MRR vs rerank-off within the latency budget; else leave off.
3. **Fusion polish (S7):** ship weighted-RRF, MMR, semantic-cache **individually** only where each is net-non-negative on the harness; gate the rest behind env.
4. **Graph (S4):** ship the tuned decays/hops only with a measured SWE-loc lift and no CoIR/CSN regression.
5. **Removal follow-up:** only after `coderank-hnsw` is the proven default, schedule a **separate** PR to delete `colbert-plaid`/vendored next-plaid (D4 end-state).

---

## 7. Execution recommendation

- Use **superpowers:subagent-driven-development**: one team per stream, fresh subagent per task, two-stage review between tasks.
- Phase 1 (S0+S1) → gate → Phase 2 (S2,S3,S4,S5,S6,S7) → integration branch → Phase 3.
- After each stream merges, run the `actually` MCP **`verify_change`**; never mark a stream done while it reports broken.
- **Worktree hygiene** (per project memory): subagent worktrees in `isolation: worktree` can leak into the integration checkout — reset before merging, and `cd` the controller shell back to the integration root before merge/commit.
- Respect CLAUDE.md throughout: repo-agnostic, no hardcoded paths in `crates/`, permissive licenses, default build zero-LLM-deps.
