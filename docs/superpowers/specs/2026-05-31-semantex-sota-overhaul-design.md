# semantex SOTA / Leaderboard Overhaul — Design Spec

- **Date:** 2026-05-31
- **Status:** Draft for review
- **Target:** v0.9 (single-vector dense; schema bump → forced reindex)
- **Authors:** design from the oxirs competitive review + 2026 code-retrieval SOTA gap analysis (see `~/.claude` project memory `oxirs-review-and-sota-gap-2026-05-31`).
- **Execution model:** implemented by parallel subagent teams, one per work-stream, from this spec + its derived implementation plans.
- **Plans + reconciliation:** the 8 per-stream plans live in `docs/superpowers/plans/2026-05-31-s{0..7}-*.md`. **Cross-stream interfaces, lead decisions, and corrections discovered during planning are reconciled in `docs/superpowers/plans/2026-05-31-integration-and-cutover.md` — that doc is authoritative where it differs from this spec's sketches** (notably: the real 5-field `ScoredChunkId`, schema bump to 11, the SIMD signatures, and the S5 reframe below).

---

## 1. Goal & thesis

Move semantex onto the modern code-retrieval leaderboards (CoIR, SWE-bench localization) **without abandoning its identity** — fully local, CPU-only, airgap-friendly, repo-agnostic, permissively licensed.

**Thesis from the gap analysis:** semantex's *design* is already SOTA-shaped. The gap is (a) **model quality per stage**, (b) two proven code-native levers (query reasoning/HyDE, code-graph fusion), and critically (c) **we cannot claim a leaderboard position we do not measure.** This spec closes all three, and folds in the genuinely useful engineering from the oxirs review (SIMD kernels, weighted-RRF, MMR, semantic cache, the recall-metrics harness).

The headline architectural bet (user decision): **replace the late-interaction ColBERT/PLAID dense path with a single-vector code embedder + HNSW ANN.** 2026 code-retrieval leaderboards are won by single-vector code embedders; this aligns semantex with that frontier while a small, MIT-licensed model keeps the local/CPU identity.

---

## 2. Decisions locked (brainstorming outcomes)

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | **Replace ColBERT** with a single-vector dense model as the new default | 2026 code-IR SOTA is single-vector; user chose full replacement |
| D2 | Dense model = **CodeRankEmbed (137M, MIT)** | Code-specialized (60.1 CoIR / 77.9 CSN MRR, beats CodeSage-Large 1.3B), MIT-permissive, ~768-dim, 8k ctx, ~35–70 MB int8 (≈2–4× current 17 MB — preserves local identity) |
| D3 | **Permissive-only** for anything shipped as a default | OSS tool on tens of thousands of repos; Apache-2.0/MIT only as defaults. NC models (jina-code, jina-reranker-v3, SFR) excluded even as opt-ins unless re-confirmed permissive |
| D4 | **Pluggable `DenseBackend` seam**; build single-vector behind it; keep ColBERT/PLAID for A/B; default to harness winner; delete loser in a follow-up | De-risks the biggest change; evidence-backed cutover; reversible |
| D5 | Benchmarks = **CoIR (headline) + CodeSearchNet + SWE-bench localization**, CPU-feasible subsets | CoIR is the current code-IR leaderboard; SWE-loc is the code-native localization signal; CSN gives external calibration |
| D6 | **next-plaid streaming refactor dropped as moot** | Replacing PLAID removes the k-means build-memory problem; HNSW + int8 + streaming inserts solve build-memory natively |
| D7 | ANN index = **HNSW + int8 vectors + fp32 rescore**, brute-force fallback for tiny indexes; **IVF-PQ deferred** | Best recall/latency at repo scale; int8 keeps memory tiny; defer PQ until a real large-repo ceiling appears |
| D8 | Reranker = **Qwen3-Reranker-0.6B (Apache-2.0)** as the code-capable option; **bge-reranker-v2-m3** retained as the permissive, already-integrated fallback; jina-v3 excluded (NC) | Code-trained, permissive; harness decides the shipped default and whether rerank flips on |

---

## 3. End-state architecture

```
query
 └─ query understanding: query_classifier + query_expander + HyDE [llm, optional]
      ├─ DENSE  : CodeRankEmbed (single-vector int8) ─► HNSW ANN (+fp32 rescore) ─► top-k
      ├─ SPARSE : BM25 (Tantivy, unchanged)                                       ─► top-k
      └─ EXACT  : substring (unchanged)                                           ─► top-k
            │
            ▼
     weighted-RRF fusion (revived adaptive weights)
            ▼
     code-graph expansion (call/import/type, 1–2 hop)   [promoted graph_propagation]
            ▼
     MMR diversity pass
            ▼
     cross-encoder rerank (Qwen3-Reranker-0.6B | bge-v2-m3) [default decided by harness]
            ▼
     results
   (semantic query cache wraps the call; flushed on reindex / schema change)
```

**Dense is swappable.** A `DenseBackend` trait abstracts the dense channel; two impls coexist during validation:
- `colbert-plaid` — today's path (LateOn-Code-edge + vendored next-plaid), behavior-preserved.
- `coderank-hnsw` — the new single-vector path.

Selected by config (`SEMANTEX_DENSE_BACKEND` / `dense_backend` in config), so the benchmark harness A/Bs them on identical corpora and we default to the proven winner.

---

## 4. Work-streams

Eight streams. Each lists **purpose, design, interfaces/files, config, acceptance gate, dependencies**. Interfaces and gates are designed so teams work in parallel without colliding on the same files.

### S0 · Benchmark harness `benchmarks/relevance/` — *foundation; gates validation of everything*

**Purpose:** the leaderboard-grade measurement we lack today. Pure-retrieval relevance (no LLM confound), reproducible, ablatable, dense-backend-selectable.

**Design:**
- New Python package `benchmarks/relevance/` (mirrors the style of `benchmarks/swe_bench/`: `pyproject.toml`, `src/`, `tests/`, `fixtures/`, gitignored `results/`).
- **Datasets** (loaders, each with deterministic, *logged* subsetting — never silently truncate):
  - **CoIR** — load the CoIR task suite from HuggingFace; ship a `coir_subset.yaml` selecting CPU-feasible datasets/sample-sizes with a fixed seed; record exactly what was included and what was dropped in every report.
  - **CodeSearchNet** — `code_search_net` (py/js/java/go/php/ruby); query→gold-function pairs.
  - **SWE-bench localization (SWE-loc)** — derive gold files/functions from SWE-bench Verified gold patches (`patch` hunks → changed files/symbols); reuse the **Phase-A 100 pre-indexed instances** from `benchmarks/swe_bench/`. Metric: file-level and function-level Recall@{1,5,10} and MRR.
- **Runner:** for each (dataset, query): build/locate a semantex index of the corpus, call `semantex search --json` (structured path), find the rank of the gold target, accumulate metrics.
- **Metrics module** (`metrics.py`, pure functions): **MRR@10, nDCG@10, Recall@{1,5,10}, MAP** — reimplement from the oxirs `RecallEvaluator` formulas (oxirs `engine/oxirs-vec/src/adaptive_recall_tuner.rs`) as the reference; no oxirs code copied (Python reimpl).
- **Ablation switches** (env, passed through to semantex): `--sparse-only`, `--dense-only`, `--hybrid`, `--rerank on|off`, and **`--dense-backend colbert-plaid|coderank-hnsw`** for the D4 A/B.
- **Report** (`report.py`): per-dataset + aggregate table; emits `report.md` + machine-readable JSON; includes the subset manifest and a `git rev` / model-id stamp for reproducibility.

**Config/IO:** `benchmarks/relevance/config/*.yaml`; results under `benchmarks/relevance/results/runN/` (gitignored). HF datasets cached under `$HF_HOME`.

**Acceptance gate:** harness runs end-to-end on a tiny fixture in tests; on a real subset it **reproduces a published CodeSearchNet (or CoIR) baseline number within a stated tolerance** (e.g. a known CodeBERT/UniXcoder CSN MRR), proving the protocol is correct before any tuning relies on it.

**Dependencies:** none to build. Gates the *defaults/cutover decisions* of S2–S7.

---

### S1 · `DenseBackend` trait + seam — *foundation; blocks S2*

**Purpose:** extract the abstraction that lets the new dense path coexist with the old.

**Design:**
- Define in `crates/semantex-core/src/search/dense_backend.rs`:
  ```rust
  pub struct ScoredChunkId { pub chunk_id: u64, pub score: f32 }

  pub trait DenseBackend: Send + Sync {
      /// Backend identity for on-disk paths + config selection.
      fn name(&self) -> &'static str;                 // "colbert-plaid" | "coderank-hnsw"
      /// Search the dense channel for a text query.
      fn search(&self, query: &str, k: usize) -> Result<Vec<ScoredChunkId>>;
      /// Restrict scoring to a candidate subset (used by graph/exact prefilters).
      fn search_with_subset(&self, query: &str, k: usize, subset: &[u64])
          -> Result<Vec<ScoredChunkId>>;
  }

  pub trait DenseIndexBuilder: Send + Sync {
      fn build(&mut self, chunks: &[(u64, &str)]) -> Result<()>;       // full
      fn insert(&mut self, chunks: &[(u64, &str)]) -> Result<()>;      // incremental
      fn delete(&mut self, chunk_ids: &[u64]) -> Result<()>;
      fn persist(&self, dir: &Path) -> Result<()>;
  }
  ```
- Refactor today's PLAID call sites in `hybrid.rs` (the `dense_handle` channel ~`hybrid.rs:351-389`) and `index/builder.rs` (the PLAID block ~`builder.rs:575-854`) to go through these traits. `PlaidSearcher` becomes impl #1 (`colbert-plaid`), **behavior-identical**.
- Backend selection: `dense_backend` config field + `SEMANTEX_DENSE_BACKEND` env; default stays `colbert-plaid` until S2 + harness flip it (D4).
- On-disk: per-backend subdirs — `.semantex/dense/colbert-plaid/` (today's `plaid/`), `.semantex/dense/coderank-hnsw/`. `meta.json` records the active backend; opening with a mismatched backend triggers a clean rebuild prompt (mirror the stemmer-flag pattern in `sparse_search.rs`).

**Acceptance gate:** full existing test suite green; dense search results **byte-identical** to pre-refactor for `colbert-plaid` (golden test on the 6 indexed repos).

**Dependencies:** none. Blocks S2.

---

### S2 · Single-vector dense path (`coderank-hnsw`) — *the big one; needs S1; consumes S6*

**Purpose:** the new default dense backend.

**Design:**
1. **Model — CodeRankEmbed.**
   - **Spike first** (`docs/superpowers/plans/...-research-notes.md`): export `nomic-ai/CodeRankEmbed` (base `Snowflake/snowflake-arctic-embed-m-long`) to ONNX via `optimum-cli export onnx`; apply ONNX Runtime **dynamic int8** quantization; confirm: embedding **dim** (expected ~768), max context (8k), the **query prefix string** (per model card — code docs get no prefix), and that long-context attention exports cleanly. Record exact dim + prefix; all later steps reference them.
   - Host the exported `model_int8.onnx` + `tokenizer.json` analogously to `model_manager.rs` (download-on-first-use; reuse `runtime_manager.rs` for the ONNX Runtime shared lib). New module `crates/semantex-core/src/embedding/single_vector.rs` (single-vector encoder: tokenize → forward → **mean-pool or CLS per model card** → L2-normalize → optional int8 quant).
   - CPU execution provider pinned (same rationale as `colbert.rs:182-195`); threads via existing `SEMANTEX_ORT_THREADS` / `SEMANTEX_INDEX_ORT_THREADS`.
   - **Embedding inputs:** the dense channel embeds the **raw code chunk** (code side, no prefix); the query gets the CodeRankEmbed query prefix. The BM25 enrichment (NL annotation + expansion in `builder.rs:373-407`) stays on the *sparse* side only — do not feed it to the dense model.
   - **Matryoshka N/A:** CodeRankEmbed (arctic-embed-m-long base) is **not** MRL-trained — the embedding dim is fixed; do not attempt dimension truncation. (MRL would only be an option under a Qwen3-class model, which is out of scope per §7.)
2. **HNSW index.**
   - **Spike:** select a pure-Rust, airgap-clean, MIT/Apache HNSW crate (evaluate `hnsw_rs`, `instant-distance`); criteria: no C/C++ deps, supports incremental insert + delete (or tombstone), serializable to disk, cosine/dot metric, maintained. **Fallback:** vendor oxirs-vec's HNSW (`engine/oxirs-vec/src/hnsw/`, Apache-2.0/MIT — attribute) — note its `parallel_construction.rs` is a stub; build would be sequential.
   - New module `crates/semantex-core/src/index/hnsw_index.rs` implementing `DenseBackend` + `DenseIndexBuilder`.
   - **Storage:** int8-quantized vectors on disk (scale+zero-point per the standard scalar-quant recipe; mmap-friendly) + the HNSW graph. Query: ANN over int8 (fast, approximate) → **fp32 rescore** of the top `rescore_k` candidates for exact ranking (the "approximate prefilter → exact rescore" pattern). Distance kernels from **S6**.
   - **Params (config-exposed, sane defaults):** `M=16`, `ef_construction=200`, `ef_search` tunable (start 64), `rescore_k=4×k`. Expose `default` / `high_recall` / `low_latency` / `memory_optimized` presets (oxirs config-preset pattern).
   - **Brute-force fallback:** below `HNSW_MIN_VECTORS` (e.g. 2,000), skip the graph and do SIMD brute-force int8 + fp32 rescore (oxirs batch-size-gating idea) — exact, and avoids HNSW overhead on small repos.
   - **Build memory (D6):** stream encode → insert; never collect all embeddings then one big call. RSS failsafe (`SEMANTEX_MAX_RSS_MB`) retained. This is how the build-memory ceiling is removed without next-plaid.
3. **Index integration.** Wire the new builder into `index/builder.rs` behind the `DenseIndexBuilder` trait; **bump the index schema version** (forces reindex); update `meta.json`.

**Config/env:** `dense_backend`, `SEMANTEX_DENSE_BACKEND`, `SEMANTEX_HNSW_EF_SEARCH`, `SEMANTEX_HNSW_PRESET`, `SEMANTEX_DENSE_RESCORE_K`.

**Acceptance gate:** indexes all 6 benchmark repos within the RSS budget; on the **S0 harness**, `coderank-hnsw` Recall@10/nDCG@10 **≥ `colbert-plaid` baseline** on CoIR + CSN (this is the A/B that justifies the cutover); query latency within target (cold ≤ current ~540 ms class; warm well under).

**Dependencies:** S1 (trait), S6 (SIMD kernels — may start with scalar, swap in SIMD when ready).

---

### S3 · Reranker upgrade — *independent*

**Purpose:** flip the strongest dormant precision lever from liability to win. The reranker is OFF today because bge-reranker-v2-m3 (35.97 CoIR) is weak on code; a code-trained permissive reranker can change that.

**Design:**
- fastembed 5.9's `RerankerModel` enum lacks the new models → build a **generic ONNX cross-encoder loader** `crates/semantex-core/src/search/onnx_reranker.rs` (download/run via ort + tokenizers; reuse `runtime_manager.rs`). Support **two score-extraction strategies**:
  - *classifier-head* (bge-style sequence-classification: single relevance logit), and
  - *yes/no-logit* (Qwen3-Reranker-style generative: prompt template → logit of the "yes" token).
- Ship **Qwen3-Reranker-0.6B (Apache-2.0)** as the code-capable option (export to ONNX + int8 — **spike**, as it's a 0.6B decoder); keep **bge-reranker-v2-m3** as the guaranteed-working permissive fallback (already integrated, permissive).
- Keep the existing `SEMANTEX_RERANKER` master switch + `SEMANTEX_RERANKER_MODEL` selection (`fastembed_reranker.rs`); add the new models to `select_model_from_env`. The default-off gate **stays until S0 proves a net win**, then flips to on with the winning model.
- Latency guard: rerank only the top `rerank_candidates`; a 0.6B reranker on CPU over ~100 candidates is feasible but must be measured — keep bge as the lighter option if Qwen3 latency is unacceptable.

**Acceptance gate:** on S0, the chosen reranker shows **net-positive nDCG@10/MRR vs rerank-off** on CoIR + CSN + the in-domain suite, within a stated per-query latency budget. If yes → flip default-on with that model; if no → document and leave off.

**Dependencies:** none to build; S0 to decide the default.

---

### S4 · Code-graph fusion promotion — *independent*

**Purpose:** the repo-level SOTA lever (LocAgent/SweRank). semantex already extracts the call/import/type graph (`structured_meta.rs`) and runs `graph_propagation.rs` as an untuned post-fusion boost — promote it to a **measured, tuned, first-class signal**.

**Design:**
- Treat `graph_propagation` as a named pipeline stage with explicit, config-tunable decays (`SEMANTEX_GRAPH_*_DECAY` already exist); add an optional **2-hop transitive** expansion gated per query-route (architectural/exhaustive/feature-planning).
- Add a localization-oriented expansion: after dense+sparse fusion, expand top seeds 1–2 hops along call/import/type edges (SQLite graph tables) and re-score; tag `GraphExpanded`.
- **Tune on SWE-loc** (S0): grid/heuristic-search the decays + hop count for best file/function Recall@k without regressing CoIR/CSN.

**Acceptance gate:** measurable SWE-loc Recall@{5,10} lift vs graph-off; **no net regression** on CoIR/CSN/in-domain.

**Dependencies:** S0 (to measure/tune). Builds independently of S2 (operates on fused candidates, backend-agnostic).

---

### S5 · HyDE call-site wiring — *independent*

**Purpose:** finish the pending v0.7.1 HyDE wiring; +5 to +12 nDCG on hard/conceptual queries (ReasonIR/BRIGHT), matching the LLM-optional design.

**Design (corrected after the S5 planning spike):** the HyDE core (`HybridSearcher::search_with_hyde` / `merge_hyde_results`) **and the daemon TCP path are already complete and safety-correct.** The real residual gap is the **MCP in-process path** (`crates/semantex-mcp/src/server.rs`): `tool_agent` never chains `.with_runtime()`, so every MCP HyDE/classify call builds a fresh Tokio runtime per request.
- Wire a shared Tokio runtime into the MCP server (mirror `listener.rs::bind`: add an `llm_runtime: Option<Arc<Runtime>>`, build once, chain `.with_runtime()` in `tool_agent`).
- Fix the latent Cargo bug: `semantex-mcp`'s `llm` feature must pull tokio — `llm = ["semantex-core/llm", "dep:tokio"]` (today tokio is gated behind `http`, so `--no-default-features --features llm` won't compile).
- Add the missing end-to-end safety tests: any LLM error/timeout (`SEMANTEX_LLM_HYDE_TIMEOUT_MS`, default 15 s) returns base results unchanged — HyDE never breaks a search. Default build remains zero-LLM-deps (feature-gated).

**Acceptance gate:** with `--features llm` + a configured backend (Ollama for airgap test), HyDE-on improves hard-query nDCG on S0's reasoning-heavy slice (CoIR conceptual / BRIGHT-style); HyDE-off and LLM-error paths are byte-identical to base.

**Dependencies:** none to build; S0 to measure. Touches `hybrid.rs` → coordinate ordering with S1/S2/S7 (see §5).

---

### S6 · SIMD distance kernels — *feeds S2's hot path*

**Purpose:** single-vector cosine/dot is the new dense hot path (and the brute-force fallback + rescore). Own a fast, portable kernel.

**Design:**
- New module `crates/semantex-core/src/search/simd.rs` (or `embedding/simd.rs`): `dot_f32`, `cosine_f32`, `l2_f32` with **AVX2 (x86_64) + NEON (aarch64) + scalar fallback**, runtime-dispatched (`is_x86_feature_detected!`), `len & !N` + scalar tail. **Reimplemented from oxirs-core's `simd/{scalar,x86_simd,arm_simd}.rs` as a reference pattern** (oxirs is Apache-2.0/MIT; we write our own to keep the tree clean) — zero external deps (`std::arch` only).
- **Batch-size gating:** below `SIMD_MIN_LEN`, use scalar (avoids setup/horizontal-reduction overhead on tiny inputs).
- Also expose an int8 dot/cosine path for scoring quantized vectors before fp32 rescore.

**Acceptance gate:** parity tests vs scalar within `1e-6` (FMA reorders floats); criterion benchmark shows speedup on representative dims (768) and the build is `unsafe`-audited + `cfg`-gated per arch.

**Dependencies:** none. Consumed by S2 (S2 can start scalar, swap in S6).

---

### S7 · Fusion & search polish — *coordinate with S1/S2 on `hybrid.rs`*

**Purpose:** the cheap, high-leverage oxirs borrows + fixing dead code in fusion.

**Design:**
- **Weighted-RRF + revive adaptive weights.** Today default RRF (`triple_fusion.rs`, `RRF_K=60`) is parameter-free; the per-query-type weight tables (`QueryType::fusion_weights`) are **dead on the RRF path**, and `config.rrf_k` (30.0) is **never read**. Implement weighted RRF `Σ wᵢ/(k+rank+1)` consuming the existing per-type weights, and make `config.rrf_k` live. (oxirs `hybrid_fusion.rs` normalizer/RRF abstraction as reference.)
- **MMR diversity pass.** After rerank, before return: `λ·rel − (1−λ)·max_sim_to_selected` over the top-K (reuse cached chunk embeddings; O(K²) at K≤50). Helps the exhaustive-query weakness. `SEMANTEX_MMR_LAMBDA` (default e.g. 0.7), off-by-default until A/B'd.
- **Semantic query cache.** New `crates/semantex-core/src/search/semantic_cache.rs`: exact-match fast path → embed query → cosine ≥ `threshold` (default 0.85) linear scan over a capped LRU (cap ~1000) → reuse `(results, query_embedding)`. **Must flush on reindex / schema-version change** (not TTL-only — stale file results are wrong for code). Daemon-scoped.

**Acceptance gate:** each sub-feature A/B'd individually on S0; ship only those with **no net regression** (weighted-RRF and MMR may help or hurt per query-type — keep what wins; gate the rest behind env). Semantic cache: correctness test that a reindex invalidates the cache.

**Dependencies:** edits `hybrid.rs`/`triple_fusion.rs` → sequence after S1's refactor lands to avoid merge thrash (see §5).

---

## 5. Sequencing & dependency graph

```
        ┌─────────────── S0 harness (gates all validation/defaults) ───────────────┐
        │                                                                            │
  S1 trait/seam ──► S2 coderank-hnsw ◄── S6 SIMD                                     │
        │                                                                            │
        ├── S3 reranker (independent build) ─────────────────────────────────► validate
        ├── S4 code-graph (independent build) ───────────────────────────────► validate
        ├── S5 HyDE (independent build) ─────────────────────────────────────► validate
        └── S7 fusion polish (after S1 lands; shares hybrid.rs) ──────────────► validate
                                                                                     │
                                              Integration + harness A/B + tuning + cutover
```

**Phase 1 (unblock):** S0 (harness) and S1 (seam) in parallel — they gate judgment and S2.
**Phase 2 (parallel build):** S2, S3, S4, S5, S6, S7. **`hybrid.rs` contention:** S1 lands first; then S2 (dense channel) and S7 (fusion) coordinate — assign them to the *same* team or serialize their `hybrid.rs` edits; S4/S5 touch distinct regions but rebase on S1.
**Phase 3 (decide):** run the full harness A/B → pick the dense-backend default (D4 cutover), decide rerank-on + model (S3), tune graph decays (S4) and fusion knobs (S7). **Only if `coderank-hnsw` wins** do we schedule ColBERT/next-plaid removal (separate follow-up PR).

**Worktree hygiene (from project memory):** parallel teams in `isolation: worktree` must not leak into the integration checkout; reset before merging; the controller shell must `cd` back to the integration root before merge/commit.

---

## 6. Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| CodeRankEmbed ONNX export fails / long-context attention issues | **Spike before S2 proper**; record dim/prefix/ctx. Fallback: `gte-modernbert-base` (Apache, ModernBERT, exportable) as a secondary permissive candidate |
| Single-vector regresses vs ColBERT on identifier/long-file matching | D4 A/B gate: `coderank-hnsw` must **meet-or-beat** `colbert-plaid` on the harness before cutover; otherwise keep ColBERT default and reassess |
| Qwen3-Reranker-0.6B ONNX export heavy / CPU latency too high | bge-reranker-v2-m3 (permissive, integrated) is the guaranteed fallback; rerank stays opt-in until latency+quality both pass |
| HNSW crate has C++ deps / poor maintenance / no delete | Selection spike with hard criteria; fallback to vendored oxirs HNSW (Apache/MIT) |
| `hybrid.rs` merge thrash across S2/S4/S5/S7 | Land S1 first; serialize hybrid.rs edits; golden-output test catches behavior drift |
| CoIR full corpus too big for CPU | Logged, seeded subsets (D5); never silent-truncate; report the manifest |
| Schema bump strands existing indexes | Forced clean reindex on version mismatch (existing pattern); documented in migration notes |

---

## 7. Scope boundaries (YAGNI / explicitly OUT)

- IVF-PQ / product quantization (defer to a real large-repo memory ceiling).
- Qwen3-Embedding-0.6B as the dense model (CodeRankEmbed is the pick; may revisit as opt-in if harness shows a large quality gap worth the 4× size).
- jina-code-embeddings, jina-reranker-v3, SFR-Embedding-Code (NC/research licenses — excluded by D3).
- EmbeddingGemma (Gemma license — not OSI-permissive).
- GPU execution paths.
- next-plaid streaming refactor (moot under D1/D6).
- All oxirs speculative modules (`quantum_rag`, `consciousness/*`, `diffusion_embeddings`, NAS, the ColBERT/cross-encoder stubs).
- Cloud/API embedders or rerankers (break local/airgap).

---

## 8. Permissive-license register

| Artifact | License | Use |
|----------|---------|-----|
| CodeRankEmbed (137M) | **MIT** | new default dense model |
| Qwen3-Reranker-0.6B | **Apache-2.0** | code-capable reranker option |
| bge-reranker-v2-m3 | permissive (confirm; already shipped) | reranker fallback |
| oxirs SIMD / HNSW reference | **Apache-2.0 / MIT** | reimplement-from-reference (SIMD) / vendor-with-attribution fallback (HNSW) |
| HNSW crate (TBD by spike) | must be **MIT/Apache**, no C/C++ | ANN index |

---

## 9. Deliverable flow

This design spec → (writing-plans skill) → a **task-by-task implementation plan** with TDD steps, exact file edits, and per-task acceptance gates, organized by the S0–S7 streams above. That plan is what the subagent teams execute from, stream by stream, with the harness (S0) as the shared judge.

---

## 10. Public claims this unlocks (once measured)

- "**+X% code-search relevance (nDCG@10 / MRR@10)** on CoIR vs the prior architecture and vs BM25-only" (S0 + S2).
- "**A measured position on CoIR**" — the first time semantex has a number on the modern code-IR leaderboard.
- "**+Z pp SWE-bench localization Recall@10**" from code-graph fusion (S4).
- "**Reranking net-positive on code**" with a permissive code-trained reranker (S3).
