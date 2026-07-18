# Cinder — Compiled Encoder-Free Indexing for Late Interaction (Design)

**Date:** 2026-07-18
**Status:** Approved design, pre-implementation
**Predecessors:** Ember Plan A (static token table, Gate 1 PASS — `results/ember-gate1/report.md`), Ember Plan B (frozen universal centroids, Gate 3 report — `results/ember-gate3/report.md`)
**Codename rationale:** Ember → Cinder; the encoder's heat is gone, what remains is compiled residue that still does the work.

## 1. Problem

lateon-colbert is semantex's validated-best embedder (quality + latency + index size), but document-side indexing remains its weak point even after Ember A+B:

- Gate 3 measured the tier0+frozen dense stage at ~120s on CopilotKit (159,013 chunks): ~116s across 307 `update_append` batches — dominated by brute-force nearest-centroid assignment (every token dotted against all 8,192 centroids) plus a per-batch O(index) IVF reload/rewrite (O(N²/B) total).
- Ember Tier-0 quality has a real gap on javascript: 91.5% nDCG@10 retention vs full contextual (go 96.4%, python 101.7%). The 5-token *linear* mix cannot express token-identity-dependent, nonlinear context.
- A ~1.0–1.1GB process RSS floor exists even on a 118-file repo (1.6s build) — unprofiled, not caused by the dense stage.

**Goal:** a truly novel indexing system where the dense stage is near-instant at any repo scale, CPU-only, memory-flat, with quality *better* than Tier-0 — while the query side (real contextual encoder, PlaidSearcher, MaxSim) stays completely unchanged.

## 2. Requirements (locked with the user, 2026-07-18)

| # | Requirement | Value |
|---|---|---|
| R1 | Dense-stage wall clock | Full dense build (Tier-0-class, no background phase) **<5s at 159k-chunk scale; <1s on platform (7.8k chunks)**. Measured as the dense increment over a sparse-only build. The walk/chunk/BM25 pipeline (~22s on CopilotKit) is explicitly out of scope. |
| R2 | Quality | **Close ≥50% of the per-language Tier-0 gap** to full contextual on CSN hybrid: js ≥95.75%, go ≥98.2%, python ≥100% nDCG@10 retention. |
| R3 | Memory | **Dense-stage peak-RSS increment <300MB** at 159k-chunk scale (vs sparse-only build). The pre-existing ~1GB floor gets a profiling workstream, not a gate. |
| R4 | Hardware | CPU-only. No GPU, no ONNX/transformer forward pass at index time. |
| R5 | Compatibility | Query side untouched; on-disk index byte-compatible with next-plaid `MmapIndex`; safe fallback chain; no existing index invalidated (fingerprint untouched). |

## 3. Prior art and novelty claims

Scanned 2026-07-18 (Tavily). Closest work and why it differs:

- **Model2Vec / WordLlama / fastText / NUMEN (2026):** static or hashed-n-gram embeddings, all *single-vector* (mean-pooled) and uncontextualized. None target late interaction; none learn a contextualization operator.
- **AweDist / Token Distillation (2025):** distills hidden-state context into *initializations for new vocabulary tokens* — a different problem (vocab extension), not an indexing-time operator.
- **CDE (Contextual Document Embeddings):** corpus-level conditioning inside a full transformer encoder; not encoder-free.
- **PLAID (2022):** uses centroid ids as compact codes at *query* time; document ingestion still requires the full encoder and k-means.
- **SPFresh / LSM-VEC / DiskANN (2023–25):** dynamic *single-vector* ANN index maintenance; nothing addresses multi-vector/PLAID token-posting construction or vocabulary-keyed ingestion.

**Cinder's original contributions:**

1. **Compiled indexing pipeline:** with a static token table and frozen centroids, embed→contextualize→assign→quantize is a pure function of the token stream; Cinder precompiles it into lookup artifacts + one tiny SIMD kernel. To our knowledge, no late-interaction system indexes documents without any model forward pass.
2. **Contextualization-delta distillation ("micro-mixer"):** distill the *contextualization operator* of a ColBERT-style encoder — not the encoder — into a ~20–30k-param windowed depthwise-separable convolution over static token vectors.
3. **Vocabulary-keyed shortlist assignment:** per-vocab-token precomputed top-m centroid shortlists; ingestion-side nearest-centroid search over the window's shortlist union (≤~160 candidates vs 8,192), with a measured agreement gate.
4. *(Systems contribution)* **Single-pass segmented PLAID construction:** integer-code segments + one k-way merge replaces per-batch IVF rewrites; O(N) instead of O(N²/B).

## 4. Architecture

```
                       OFFLINE (once, generic corpora)                    INDEX TIME (per repo)
 lateon-colbert ──teacher──► distill-static-table ─► static_token_table.bin ─┐
 (contextual)  ──teacher──► distill-mixer        ─► cinder_mixer.bin       ─┼─► tokenize ─► gather rows
               ──teacher──► distill-centroids    ─► frozen_centroids.npy   ─┤     │
                            derive-shortlists    ─► cinder_shortlists.bin  ─┘     ▼
                                                                            micro-mixer (SIMD, ±4 window)
 QUERY TIME (unchanged):                                                          │ L2 norm
 real ColbertEmbedder ─► PlaidSearcher ─► MaxSim                                  ▼
        ▲                                                                   shortlist-union argmax ─► (cid, residual)
        └────────────── reads the same on-disk PLAID format ◄── segment spill + k-way merge ─► ivf/codes/residuals
```

### 4.1 Artifacts (model dir, all offline-trained, all repo-agnostic)

1. `static_token_table.bin` — existing (Ember A, SXST v1, unchanged).
2. `frozen_centroids.npy` — existing (Plan B, [8192, 48] f32 npy, unchanged).
3. **`cinder_mixer.bin` (new).** Micro-mixer weights: `e_i = L2norm(t_i + W_p · GELU(DW(t_{i−4..i+4})))` where `DW` is a depthwise convolution over a 9-token window at d=48 (per-dim 9-tap filters, 9×48 params) and `W_p` is a 48×48 pointwise projection; ≈2.7k core params plus biases (well under the 30k ceiling; a second block may be added if ablation demands, staying within the FLOP budget). Cost ≈5 kFLOPs/token — ~150× cheaper than the 17M-param transformer, ~6× under the R1 FLOP budget. Format: magic `SXCM`, version u32, dims/window/params header with checked-arithmetic loading (per the static-table overflow lesson), int8 weights + f32 scales, f16 acceptable for biases.
4. **`cinder_shortlists.bin` (new).** For every vocab token with a nonzero table row: top-m (m=32 default) centroid ids by dot product of its *static* embedding; 50,370 × 32 × u16 ≈ 3.2MB. Magic `SXCS`, versioned, checked loading. Tokens with zero rows (never seen in distillation) get the global top-m centroids of the mean embedding as a fallback shortlist entry.

### 4.2 Offline training (`distill-mixer` hidden CLI + `derive-shortlists`)

- **Teacher:** full contextual lateon-colbert per-token embeddings via the existing `encode_documents_with_ids` alignment seam (Ember A). Corpus: the Gate-3 recipe (CopilotKit/packages, platform, pub, gin, adk-python, semantex; record SHAs).
- **Student:** micro-mixer over static-table rows for the same token windows. Loss: `1 − cosine(student, teacher)` per token position (teacher rows are unit-norm). Optimizer: hand-rolled Adam with fixed seeds; backprop implemented manually (depthwise conv + GELU + matmul — ~300 lines). **No new ML dependencies** (no candle/burn/torch); hermetically testable via the existing `DocTokenEncoder` fake-encoder seam. Reservoir-sample training pairs to a fixed memory bound (same pattern as `static_distill`/`centroid_train`).
- **Shortlist derivation:** exact top-m per vocab row against `frozen_centroids.npy`; pure linear algebra, runs in seconds.
- **Offline diagnostics emitted at train time:** held-out cosine-to-teacher (mixer vs 5-tap linear baseline), and shortlist agreement rate (fraction of held-out *mixed* embeddings whose exact argmax centroid appears in their window's shortlist union) — see gate C4.

### 4.3 Index-time pipeline (fresh build only)

Per chunk: tokenize (existing skiplist-aware path) → gather static rows → micro-mixer (SIMD f32; document edge positions replicate the center convention used everywhere since Ember A) → L2-normalize → shortlist-union argmax → residual = e − centroid, quantized with next-plaid's `ResidualCodec` cutoff logic (cutoffs computed once from a sampled prefix of the stream, matching next-plaid's own construction) → emit packed `(centroid_id: u16, residual: nbits-packed)` codes into an in-memory segment.

Segments spill to disk every ~512k tokens (~13MB at 4-bit/48-d). Finalize: k-way merge by centroid id → write `centroids.npy`, `ivf.npy`/`ivf_lengths.npy`, codes, residuals, `metadata.json` — **byte-compatible with next-plaid `MmapIndex::load`** (the load-bearing constraint; C4 verifies it). Peak memory: one segment + merge buffers + artifacts (≈50–80MB), independent of corpus size.

Complexity: assignment O(tokens × 160 × 48) instead of O(tokens × 8192 × 48); construction O(N log S) merge instead of O(N²/B) IVF rewrites. Budget at 9.5M tokens / 8 cores: tokenize ~1–2s, mixer ~0.5–1s, assignment ~1s, merge+write ~1s → ~4s worst case (R1).

### 4.4 Scope boundaries

- **Fresh builds only.** Incremental inserts keep the existing `update_append` path (O(batch), frozen-centroid-aware since Plan B). Revisit only if incremental profiling demands it.
- Walk/chunk/sparse pipeline: out of scope (noted as the next wall-clock frontier).
- Background convergence to full contextual quality (old Plan C): out of scope; Cinder's R2 quality is the permanent quality of a Cinder index.
- **Floor autopsy (ungated workstream):** profile the ~1.06GB gin-build floor (ORT provisioning, tantivy, sqlite, allocator arenas), write findings into the evaluation report, take only obvious wins (e.g., skip ORT runtime provisioning entirely on encoder-free builds).

### 4.5 Fallback chain and compatibility

`SEMANTEX_CINDER=1` (via `env_bool`, build-time only):

| State | Behavior |
|---|---|
| Flag off | Byte-identical to today (Ember flags still work independently). |
| Flag on, all 3 artifacts load | Cinder writer. One `tracing::info!` confirmation line. |
| Mixer or shortlists missing/corrupt | `tracing::warn!` → Ember tier0+frozen path (if its flags/artifacts allow) or contextual build. |
| Centroids missing | `tracing::warn!` → per-repo k-means path. |

Never errors from artifact resolution. All artifact loads use checked size arithmetic and reject implausible headers before allocation. Writes are atomic (temp + rename). `SEMANTEX_CINDER` and any future artifact fields stay **out of `EmbedderFingerprint`** (doc-side build option; same rationale and test pattern as `SEMANTEX_STATIC_DOC_EMBED` / `SEMANTEX_FROZEN_CENTROIDS`) — no existing index is ever invalidated.

## 5. Validation gates

| Gate | Criterion | Method |
|---|---|---|
| **C1 Quality (research gate)** | Per-language CSN hybrid nDCG@10 retention vs full contextual: **js ≥95.75%, go ≥98.2%, python ≥100%** (≥50% of each Gate-1 tier0 gap closed). | Standard CSN harness (seed 20260531 subset, unchanged); baseline = Gate-1 full-hybrid recordings. Ablation matrix {5-tap linear, mixer} × {exact argmax, shortlist} to attribute any miss. Contingency: hashed bigram/trigram contextual tables under the mixer, only if the mixer misses. |
| **C2 Speed** | Dense increment **<5s** at 159k chunks; **<1s** on platform. | `/usr/bin/time -l` on `semantex index` with and without the dense stage (sparse-only control), same protocol as Gate 3. |
| **C3 Memory** | Dense-stage peak-RSS increment **<300MB** at 159k chunks. | Same runs as C2: Cinder peak minus sparse-only peak. |
| **C4 Mechanism** | Shortlist agreement **≥99%** vs exact argmax on held-out sample (bump m if missed); segment-merge output **byte-equivalent** to a reference next-plaid `create_index_files` build given identical codes; every fallback row in §4.5 observed in logs. | Offline diagnostics + differential test + log assertions. |

A C1 failure after the contingency lever is a legitimate research finding — report it honestly with the ablation attribution; do not ship a quality regression silently.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Micro-mixer can't close 50% of the js gap (the research bet) | Ablation matrix isolates mixer vs assignment effects; n-gram contingency lever; honest FAIL is an acceptable outcome with attribution. |
| Shortlist misses shift centroid assignments enough to hurt quality | C4 agreement gate measured *before* C1 runs; m is tunable; exact-argmax ablation quantifies the ceiling. |
| Byte-compat drift vs next-plaid's writer | C4 differential test builds the same codes through both writers and compares files. |
| Hand-rolled backprop bugs | Gradient-check test (finite differences vs analytic, fixed seed) is a mandatory unit test; tiny model makes this exact. |
| Tokenization throughput becomes the new bottleneck | Tokenizer already runs in the chunk pipeline; measure early (first plan task includes a profile checkpoint); parallelize across chunks if needed. |
| Residual-codec cutoff mismatch vs next-plaid construction | Reuse next-plaid's own `ResidualCodec` code paths rather than reimplementing. |

## 7. Deliverables

Implementation branch + `results/cinder-gate/report.md` (Gate-1/3 report conventions: raw logs, SHAs, judgment calls, per-gate verdicts, recommendation). CLAUDE.md constraints apply throughout: repo-agnostic crates, generic training corpora, zero new default-build deps, Rust 1.91 CI gates, postcard discipline (no postcard structs are touched by this design).
