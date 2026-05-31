# S2 — CodeRankEmbed + HNSW Dense Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `coderank-hnsw` — a single-vector dense backend (CodeRankEmbed int8 ONNX encoder + pure-Rust HNSW ANN with int8 storage and fp32 rescore) — behind the S1 `DenseBackend`/`DenseIndexBuilder` trait seam, selectable via `SEMANTEX_DENSE_BACKEND=coderank-hnsw`, so the S0 harness can A/B it against `colbert-plaid` and (if it wins) we cut over the default.

**Architecture:** Two new core modules. `embedding/single_vector.rs` is a lazy ONNX encoder (mirrors `colbert.rs`: lazy `OnceLock` session, CPU EP pinned, `SEMANTEX_(INDEX_)ORT_THREADS`) that tokenizes → forwards → mean-pools (per recorded model card) → L2-normalizes → optionally int8-quantizes, embedding **raw code** for documents and **prefix+query** for queries. `index/hnsw_index.rs` implements both S1 traits over a pure-Rust HNSW crate (chosen by Spike 2): int8-quantized vectors stored on disk, ANN over int8 → fp32 rescore of top `rescore_k`, brute-force fallback below `HNSW_MIN_VECTORS`, streaming encode→insert to bound build RSS. `builder.rs` routes the dense build through this backend when selected; the schema bumps to **11**.

**Tech Stack:** Rust 2024, `ort 2.0.0-rc.11` (raw `Session`, same dynamic-load runtime as ColBERT), `tokenizers 0.22` (NEW direct dep — already transitive), `ndarray 0.16`, the HNSW crate chosen in Spike 2, `anyhow`, `serde`/`postcard`, `tempfile` + `cargo test`. Reuses `embedding/runtime_manager.rs` (ONNX Runtime provisioning) and `embedding/model_manager.rs` download pattern. SIMD kernels from S6 are optional (start scalar, swap later).

---

## Reconciled facts (verified against current source + deps — do not re-derive)

Quoted from the real tree / pinned deps at plan-authoring time. Every type/method below exists today, is pinned in `Cargo.lock`, is added by an earlier task in this plan, or is **RECORDED by a Spike task** (Tasks 1–2). No later task invents a crate API or an export detail — it references a recorded value.

### From S1 (LOCKED — consume verbatim, do NOT redefine)

S1 (`docs/superpowers/plans/2026-05-31-s1-dense-backend-seam.md`) lands first and provides, in `crates/semantex-core/src/search/dense_backend.rs`:

- **`DenseHit = crate::types::ScoredChunkId`** — the **5-field** type (`chunk_id, score, score_dense, score_sparse, score_exact`), `crate/types.rs:219-241`. Dense backends populate only `chunk_id` + `score` (via `ScoredChunkId::new(id, score)`); the per-channel fields stay 0.0 and are filled by fusion. **Do NOT introduce a 2-field struct.**
- **`pub trait DenseBackend: Send + Sync`** with:
  - `fn name(&self) -> &'static str;`
  - `fn search(&self, query: &str, k: usize) -> Result<Vec<DenseHit>>;`
  - `fn search_with_subset(&self, query: &str, k: usize, subset: &[u64]) -> Result<Vec<DenseHit>>;`
  - `fn positional_chunk_ids(&self) -> Option<&[u64]> { None }` (default). **Our HNSW backend keeps the default `None`** (G5): no positional doc→chunk array; the `file_filter` subset path then degrades to unfiltered dense + result-merge filtering. Document this in code.
- **`pub trait DenseIndexBuilder: Send + Sync`** with:
  - `fn name(&self) -> &'static str;`
  - `fn build(&mut self, chunks: &[(u64, &str)]) -> Result<()>;`
  - `fn insert(&mut self, chunks: &[(u64, &str)]) -> Result<()>;`
  - `fn delete(&mut self, chunk_ids: &[u64]) -> Result<()>;`
  - `fn persist(&self, dir: &Path) -> Result<()>;`
- **`pub enum DenseBackendKind { #[default] ColbertPlaid }`** with `fn name(self) -> &'static str` and `fn parse(s: &str) -> Option<Self>` (case-insensitive, trimmed). **S2 adds the `CoderankHnsw` variant** (name `"coderank-hnsw"`, parse).
- **`pub fn dense_subdir(index_dir: &Path, backend: DenseBackendKind) -> PathBuf`** → `<index_dir>/dense/<backend.name()>/`. Our on-disk dir is therefore `.semantex/dense/coderank-hnsw/`.
- **`pub fn verify_persisted_backend_matches(index_dir: &Path, expected: &str) -> Result<()>`** — open-time mismatch guard (mirrors stemmer flag). Reused unchanged; just needs `meta.dense_backend` populated, which builder already does.
- **`SemantexConfig.dense_backend: String`** (default `"colbert-plaid"`) + `SEMANTEX_DENSE_BACKEND` env overlay in `SemantexConfig::load()` + `config::env_string(key, default) -> String` helper. Reused; S2 adds three more env knobs.
- **`IndexMeta.dense_backend: String`** field (S1 added it). S1 bumps `CURRENT_SCHEMA_VERSION` **9 → 10**; **S2 bumps it 10 → 11** (G3 — coordinated below).
- **`hybrid.rs`** dense channel is `dense: Option<Box<dyn DenseBackend>>`, selected in `open()` by `DenseBackendKind::parse(&config.dense_backend)`. S2 only adds a `CoderankHnsw` arm to that `match`.
- **`builder.rs`** dense build routes through `DenseBackendKind::parse(...).unwrap_or_default()` → a `match backend_kind { ColbertPlaid => ... }`. S2 adds a `CoderankHnsw` arm.

### From the current tree

- **`ort 2.0.0-rc.11`** (pinned, `Cargo.lock`): `ort::session::Session::builder() -> Result<SessionBuilder>`; `SessionBuilder::with_execution_providers(impl AsRef<[ExecutionProviderDispatch]>) -> Result<Self>`, `with_intra_threads(usize) -> Result<Self>`, `with_optimization_level(GraphOptimizationLevel) -> Result<Self>`, `commit_from_file<P>(self, P) -> Result<Session>`; `Session::run(&mut self, impl Into<SessionInputs<...>>) -> Result<SessionOutputs>`; the `ort::inputs!["name" => value, ...]` macro builds a `Vec<(Cow<str>, SessionInputValue)>`; `ort::value::Tensor::from_array((shape: Vec<i64>, data: Vec<T>)) -> Result<Tensor<T>>`; `SessionOutputs` is `Index<&str>`; `DynValue::try_extract_tensor::<f32>() -> Result<(&Shape, &[f32])>`. EP construction idiom (verbatim from `search/fastembed_reranker.rs:165-187`): `ort::ep::CPU::default().build()`, `ort::ep::CoreML::default().build()`.
- **ONNX Runtime provisioning:** `crates/semantex-cli/src/main.rs::resolve_ort_dylib()` sets `ORT_DYLIB_PATH` at process startup (via `runtime_manager::find_onnxruntime` / `ensure_onnxruntime`) for any command that loads ONNX in-process. A raw `ort::Session` in `single_vector.rs` inherits this — **no extra wiring needed in core**, but the encoder calls `runtime_manager::ensure_onnxruntime(&runtime_root)` defensively on first build (idempotent no-op when already provisioned), exactly like the CLI does. `ONNXRUNTIME_VERSION` is pinned to ort (`runtime_manager.rs:30`).
- **`tokenizers`**: resolved to `0.22.2` transitively (`Cargo.lock`), NOT a direct dep. **Task 3 adds it to `crates/semantex-core/Cargo.toml`.** API: `tokenizers::Tokenizer::from_file<P>(P) -> Result<Tokenizer>`; `Tokenizer::encode(input, add_special_tokens: bool) -> Result<Encoding>`; `Encoding::get_ids() -> &[u32]`, `Encoding::get_attention_mask() -> &[u32]`.
- **`ndarray 0.16`** is the direct dep (`Cargo.toml:89`). `ColbertEmbedder` uses `Array2<f32>`. We use plain `Vec<f32>` for single vectors.
- **`ChunkStore`** (`index/storage.rs`): `get_all_chunk_ids(&self) -> Result<Vec<u64>>` (425), `get_chunks(&self, ids: &[u64]) -> Result<Vec<Chunk>>` (254), `chunk_count(&self) -> Result<u64>` (360). `Chunk` (`types.rs:20-34`): `id: u64`, `content: String`, `file_path: PathBuf`, `start_line/end_line: u32`, `chunk_type`.
- **`memory`**: `crate::memory::check_rss_or_abort(label: &str) -> Result<(), String>`, `crate::memory::purge_allocator()`, `crate::memory::current_rss_mb() -> Option<u64>`. Used at batch boundaries (the RSS failsafe).
- **`colbert.rs` lazy/threads/EP pattern** (the template `single_vector.rs` follows): lazy `OnceLock<Mutex<Session>>` + `build_lock` double-checked locking (lines 200-234); `index_ort_threads()` = `env_usize("SEMANTEX_INDEX_ORT_THREADS", clamp(cores/2,2,8))` (36-39); query threads = `env_usize("SEMANTEX_ORT_THREADS", 4)` (97); CPU EP pinned on every platform, CoreML only via `SEMANTEX_COREML=1` on macOS (182-195). `ColbertEmbedder::global(model_dir) -> Result<&'static ColbertEmbedder>` (258) is the process-global singleton pattern.
- **`builder.rs` dense block** (post-S1) routes through the trait; pre-S1 inline PLAID lives at `builder.rs:570-854`, meta write at `862-874`, stale cleanup at `147-156`. **S2 edits the post-S1 shape** (a `match backend_kind` with a `ColbertPlaid` arm) — add the `CoderankHnsw` arm.
- **Module wiring:** `embedding/mod.rs` declares `colbert`, `model_manager`, `runtime_manager`; `index/mod.rs` declares `builder, storage, state, ...`; `search/mod.rs` declares (post-S1) `dense_backend, colbert_plaid_backend, ...`.
- **S0 harness contract** (`docs/superpowers/plans/2026-05-31-s0-relevance-harness.md`): the A/B is driven purely by the **`SEMANTEX_DENSE_BACKEND` env var** + `semantex search --json` — there is **no `--dense-backend` CLI flag**. So S2's acceptance needs only: (a) `coderank-hnsw` parses + builds + searches end-to-end via the env var, (b) `semantex index` then `semantex search --json` returns results. No CLI changes required in S2.
- **Repo-agnostic test harness pattern** (`tests/search_accuracy_test.rs`): `TempDir` → write synthetic source files → `IndexBuilder::new(&config)?.build(&project_dir)?` → `HybridSearcher::open(&project_dir.join(".semantex"), &config)`. No hardcoded `/Users/` paths (G2, CLAUDE.md Hard Rule #1).

---

## File Structure

Files created or modified, one responsibility each:

- **Modify `docs/superpowers/plans/2026-05-31-research-notes.md`** (create if missing) — the two spikes RECORD facts here: CodeRankEmbed export (dim, max ctx, query-prefix string, pooling, ONNX input/output tensor names, int8 file name, HF repo + base) and the chosen HNSW crate (crate name + version, the exact constructor/insert/search/delete/serialize API, metric, license). Every later task references these recorded values.

- **Modify `crates/semantex-core/Cargo.toml`** — add `tokenizers = { version = "0.22", default-features = false, features = ["onig"] }` (or the feature set Spike confirms compiles offline) and the HNSW crate dependency chosen in Spike 2. No LLM deps; default build stays zero-genai.

- **Create `crates/semantex-core/src/embedding/single_vector_model.rs`** — model provisioning. `ensure_coderank_model(models_dir) -> Result<PathBuf>` + `is_coderank_downloaded(models_dir) -> bool`, mirroring `model_manager.rs`. The HF repo, file list, and dir name are RECORDED constants from Spike 1. Reuses `model_manager::download_file`.

- **Create `crates/semantex-core/src/embedding/single_vector.rs`** — the encoder. `SingleVectorEmbedder` (lazy `OnceLock<Mutex<Session>>` + `build_lock`, threads/EP per `colbert.rs`). `encode_document(&str) -> Result<Vec<f32>>` (raw code, no prefix), `encode_query(&str) -> Result<Vec<f32>>` (RECORDED prefix prepended), `embedding_dim() -> usize` (RECORDED), `for_indexing`/`new`/`global` constructors. Internal: tokenize → run → mean-pool over attention mask → L2-normalize. Plus `quantize_int8(&[f32]) -> (Vec<i8>, f32)` + `dequantize_int8(&[i8], f32) -> Vec<f32>` (symmetric per-vector scalar quant).

- **Create `crates/semantex-core/src/search/simd.rs`** — distance kernels (S2's local copy; S6 may later replace the body). `dot_f32(&[f32],&[f32]) -> f32`, `cosine_f32`, plus `dot_i8(&[i8],&[i8]) -> i32`. Scalar now; `#[inline]`, documented as the swap-in point for S6 SIMD. (If S6 has already landed `search::simd`, SKIP creating this file and import S6's instead — Task 5 notes the check.)

- **Create `crates/semantex-core/src/index/hnsw_index.rs`** — the backend. `HnswParams` (M/ef_construction/ef_search/rescore_k + presets), `HnswDenseIndex` (on-disk int8 vectors + graph + chunk_id map), `CoderankHnswBackend` (impl `DenseBackend`), `CoderankHnswIndexBuilder` (impl `DenseIndexBuilder`). Brute-force fallback below `HNSW_MIN_VECTORS`; ANN→fp32-rescore query path; streaming encode→insert build. The HNSW crate calls are RECORDED from Spike 2.

- **Modify `crates/semantex-core/src/embedding/mod.rs`** — add `pub mod single_vector;` and `pub mod single_vector_model;`.

- **Modify `crates/semantex-core/src/index/mod.rs`** — add `pub mod hnsw_index;`.

- **Modify `crates/semantex-core/src/search/mod.rs`** — add `pub mod simd;` (only if not already added by S6).

- **Modify `crates/semantex-core/src/search/dense_backend.rs`** — add the `CoderankHnsw` variant to `DenseBackendKind` (`name` + `parse` arms). (This is the one S1-owned file S2 edits; coordinate per §5.)

- **Modify `crates/semantex-core/src/config.rs`** — add `hnsw_ef_search`, `hnsw_preset`, `dense_rescore_k` config fields + their `SEMANTEX_HNSW_EF_SEARCH` / `SEMANTEX_HNSW_PRESET` / `SEMANTEX_DENSE_RESCORE_K` env overlays.

- **Modify `crates/semantex-core/src/types.rs`** — bump `CURRENT_SCHEMA_VERSION` 10 → 11; record the v11 reason (single-vector dense layout). Update the `current_schema_version_is_10` test (S1's) → `_is_11`.

- **Modify `crates/semantex-core/src/search/hybrid.rs`** — add the `CoderankHnsw` arm to the dense-backend `match` in `open()`.

- **Modify `crates/semantex-core/src/index/builder.rs`** — add the `CoderankHnsw` arm to the dense-build `match`; record `embedding_model`/`embedding_dim` for the single-vector model in `meta.json` when that backend is active.

- **Create `crates/semantex-core/tests/coderank_hnsw_backend_test.rs`** — the acceptance gate. Synthetic tempdir repo → build with `dense_backend = "coderank-hnsw"` → `HybridSearcher` dense search returns sane, deterministic results; brute-force vs HNSW agree on a tiny corpus; RSS-bounded build. `#[ignore]` (needs the model download). Repo-agnostic.

---

## Phasing note for executors

**Tasks 1–2 are the two SPIKES and MUST run first.** They produce no shippable code — they download/export the model and evaluate the HNSW crate, and write the recorded facts into `2026-05-31-research-notes.md`. Every subsequent task quotes those recorded values (the HF repo, dim, prefix, pooling, tensor names; the crate name + API). If a spike's outcome differs from the placeholders below (e.g. pooling is CLS not mean, or the chosen crate is `instant-distance` not `hnsw_rs`), update the affected task's code to match the recorded fact before implementing — the spike is the source of truth, not this draft.

Tasks 3–6 build the encoder + kernels + model provisioning (pure additions, suite stays green). Task 7 builds the HNSW backend. Tasks 8–11 wire config/enum/schema/builder/hybrid. Task 12 is the end-to-end acceptance gate. Commit after every task.

---

### Task 1: SPIKE — export CodeRankEmbed to ONNX int8 + record model facts

> **EXISTING EXPORTS — verified on HF 2026-05-31; prefer these over a cold export.** MIT-licensed ONNX of `nomic-ai/CodeRankEmbed` already exists: **[`mrsladoje/CodeRankEmbed-onnx-int8`](https://huggingface.co/mrsladoje/CodeRankEmbed-onnx-int8)** (tagged int8; `onnx/model.onnx` + `tokenizer.json` + `config.json`) and **[`sirasagi62/code-rank-embed-onnx`](https://huggingface.co/sirasagi62/code-rank-embed-onnx)** (fp32 `model.onnx`, 896 dls — most vetted). This spike should **verify one of these for parity + int8**, record dim/prefix/pooling from it, then **re-host into a project-controlled MIT repo** (don't point production at a personal repo). The tool **`benchmarks/onnx_models/prepare_models.py`** does verify→int8→re-host in one command. The cold `optimum-cli` export below stays as the fallback if parity verification fails.

**Goal:** Produce `model_int8.onnx` + `tokenizer.json` for `nomic-ai/CodeRankEmbed` and RECORD the facts later tasks depend on. No Rust in this task.

**Files:**
- Modify: `docs/superpowers/plans/2026-05-31-research-notes.md` (create if missing)

- [ ] **Step 1: Export via optimum + quantize to int8**

Run in a throwaway Python venv (NOT committed; this is a one-time artifact-producing spike). Use `uv` or `pip`:

```bash
python3 -m venv /tmp/cre-spike && . /tmp/cre-spike/bin/activate
pip install "optimum[exporters,onnxruntime]" "transformers>=4.40" sentence-transformers onnx onnxruntime
# Export the encoder to ONNX (fp32 first).
optimum-cli export onnx \
  --model nomic-ai/CodeRankEmbed \
  --task feature-extraction \
  --trust-remote-code \
  /tmp/cre-onnx
ls -la /tmp/cre-onnx
```

Then apply ONNX Runtime **dynamic int8** quantization:

```bash
python3 - <<'PY'
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("/tmp/cre-onnx/model.onnx",
                 "/tmp/cre-onnx/model_int8.onnx",
                 weight_type=QuantType.QInt8)
print("int8 written")
import os; print({f: os.path.getsize(f"/tmp/cre-onnx/{f}") for f in os.listdir("/tmp/cre-onnx")})
PY
```

- [ ] **Step 2: Probe dim, ctx, pooling, prefix, tensor names**

```bash
python3 - <<'PY'
import json, onnx, numpy as np, onnxruntime as ort
from transformers import AutoTokenizer, AutoConfig

cfg = AutoConfig.from_pretrained("nomic-ai/CodeRankEmbed", trust_remote_code=True)
print("hidden_size (expect ~768):", cfg.hidden_size)
print("max_position_embeddings (expect 8192):", getattr(cfg, "max_position_embeddings", None))

# ONNX graph input/output tensor names — single_vector.rs MUST use these exact names.
m = onnx.load("/tmp/cre-onnx/model_int8.onnx")
print("INPUTS :", [i.name for i in m.graph.input])
print("OUTPUTS:", [o.name for o in m.graph.output])

tok = AutoTokenizer.from_pretrained("nomic-ai/CodeRankEmbed", trust_remote_code=True)
# Run a doc + a query, compare with the reference sentence-transformers embedding
# to confirm pooling (mean vs CLS) and the required query prefix.
PY
```

**The query prefix is on the model card.** CodeRankEmbed requires the query be prefixed with: `Represent this query for searching relevant code: ` (documents get NO prefix). CONFIRM this exact string from the card / `sentence-transformers` config in the export, and record it verbatim (trailing space included).

Verify pooling: run the ONNX model's `last_hidden_state` through (a) mean-pool over the attention mask and (b) CLS (token 0), L2-normalize each, and compare cosine to the `SentenceTransformer("nomic-ai/CodeRankEmbed", trust_remote_code=True).encode(...)` reference. The pooling whose cosine ≈ 1.0 is the correct one (arctic-embed-m-long uses **CLS**, but CONFIRM empirically).

- [ ] **Step 3: RECORD facts in research-notes**

Append a `## S2 Spike 1 — CodeRankEmbed ONNX export` section to `docs/superpowers/plans/2026-05-31-research-notes.md` with EXACTLY these recorded values (fill the `<...>` from the probes):

```markdown
## S2 Spike 1 — CodeRankEmbed ONNX export (RECORDED <date>)
- HF repo: nomic-ai/CodeRankEmbed   (base: Snowflake/snowflake-arctic-embed-m-long)
- License: MIT
- embedding_dim: <768?>            ← single_vector.rs EMBEDDING_DIM
- max_context_tokens: <8192?>      ← single_vector.rs MAX_CTX (we truncate to this)
- pooling: <CLS | mean>            ← single_vector.rs pooling fn
- query_prefix: "<Represent this query for searching relevant code: >"  ← EXACT, incl trailing space
- document_prefix: "" (none)
- ONNX input tensor names: <["input_ids","attention_mask",("token_type_ids"?)]>
- ONNX output tensor name (token embeddings): <"last_hidden_state">
- int8 file: model_int8.onnx (<size> MB)   tokenizer file: tokenizer.json
- Reference parity: cosine(onnx+<pooling>+L2, sentence-transformers) = <~1.0>
- Long-ctx attention exported cleanly: <yes/no, notes>
```

- [ ] **Step 4: Host the artifacts**

Upload `model_int8.onnx` + `tokenizer.json` to a HF repo under the project's control (e.g. `MisterTK/CodeRankEmbed-onnx-int8`) so `single_vector_model.rs` can download-on-first-use, mirroring `lightonai/LateOn-Code-edge`. Record the resolve base URL in research-notes:

```markdown
- Download base: https://huggingface.co/<owner>/CodeRankEmbed-onnx-int8/resolve/main
- Files: ["model_int8.onnx", "tokenizer.json"]
- Dir name: CodeRankEmbed
```

If the export FAILS or long-context attention does not export cleanly, record the failure and STOP — escalate to the spec owner: the §6 fallback is `gte-modernbert-base` (Apache, ModernBERT, exportable). Re-run this spike against that model and record its facts instead (its prefix/pooling differ — record the actuals).

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/plans/2026-05-31-research-notes.md
git commit -m "docs(s2): record CodeRankEmbed ONNX export spike (dim/ctx/prefix/pooling)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: SPIKE — select a pure-Rust HNSW crate + record its API

**Goal:** Pick the ANN crate and RECORD its exact API so Task 7's code is non-hallucinated. No shippable Rust.

**Files:**
- Modify: `docs/superpowers/plans/2026-05-31-research-notes.md`

- [ ] **Step 1: Evaluate candidates against hard criteria**

For each of `hnsw_rs` and `instant-distance`, check (use `cargo add --dry-run`, the crate's `Cargo.toml`, and docs.rs):

1. **No C/C++ deps** — `cargo tree -p <crate>` shows no `cc`/`*-sys`/bindgen. (semantex must stay airgap/offline-buildable.)
2. **License** MIT or Apache-2.0 (NOT GPL/NC).
3. **Incremental insert + delete (or tombstone)** — needed for `DenseIndexBuilder::insert`/`delete`.
4. **Serializable to disk** — needed for `persist`. (If the crate has no native serialize, we serialize our own int8 vectors + chunk_id map and **rebuild the graph on open** — record this as the plan.)
5. **Cosine or dot metric** — we feed L2-normalized vectors, so dot == cosine.
6. **Maintained** — last release within ~18 months; not yanked.

Create a tiny scratch crate to actually compile+run each:

```bash
mkdir -p /tmp/hnsw-spike && cd /tmp/hnsw-spike && cargo init --name hnsw_spike
cargo add hnsw_rs        # then write a 20-line insert+search smoke test, cargo run
# repeat with: cargo rm hnsw_rs && cargo add instant-distance
```

Smoke test must: build an index from ~1000 random 768-dim vectors, insert, search top-10, and (if supported) delete one id and re-search. Record which crate compiled offline, ran, and supports delete.

- [ ] **Step 2: RECORD the chosen crate + its exact API**

Append to `docs/superpowers/plans/2026-05-31-research-notes.md`:

```markdown
## S2 Spike 2 — HNSW crate selection (RECORDED <date>)
- CHOSEN: <hnsw_rs vX.Y | instant-distance vX.Y>
- License: <MIT|Apache-2.0>
- C/C++ deps: none (cargo tree verified)
- Metric used: <Cosine|DotProduct> over L2-normalized vectors
- Delete support: <native | tombstone-only | NONE → rebuild-on-mutate strategy>
- Serialize support: <native (fn …) | NONE → we persist int8 vecs + map, rebuild graph on open>
- EXACT API (copy real signatures):
    - construct: <e.g. Hnsw::<f32, DistCosine>::new(max_nb_conn, max_elems, max_layer, ef_c, DistCosine)>
    - insert:    <e.g. hnsw.insert((&vec, id))>
    - search:    <e.g. hnsw.search(&query, k, ef_search) -> Vec<Neighbour{ d_id, distance }>>
    - delete:    <exact, or "n/a">
    - id type:   <usize? supports u64?>
    - distance orientation: <smaller = closer? then score = -distance or 1-distance>
- Fallback note: if neither passes, vendor oxirs-vec HNSW (engine/oxirs-vec/src/hnsw/, Apache/MIT,
  attribute) — note parallel_construction.rs is a stub (sequential build).
```

**The recorded "EXACT API" block is what Task 7 codes against.** If the chosen crate uses `usize` ids only, Task 7 maps `chunk_id: u64 → usize` via a side `Vec<u64>` (graph position → chunk_id), exactly like PLAID's `doc_to_chunk` — record that decision here.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-05-31-research-notes.md
git commit -m "docs(s2): record HNSW crate selection spike + exact API

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: add `tokenizers` (+ HNSW crate) deps; `single_vector_model.rs` provisioning

**Files:**
- Modify: `crates/semantex-core/Cargo.toml`
- Create: `crates/semantex-core/src/embedding/single_vector_model.rs`
- Modify: `crates/semantex-core/src/embedding/mod.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/semantex-core/src/embedding/single_vector_model.rs` with the test module first:

```rust
//! Download-on-first-use provisioning for the CodeRankEmbed single-vector ONNX
//! model. Mirrors `model_manager.rs` (the ColBERT downloader). The HF repo +
//! file list are the values RECORDED in
//! `docs/superpowers/plans/2026-05-31-research-notes.md` (S2 Spike 1).

use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coderank_not_downloaded_in_empty_dir() {
        let tmp = tempfile::TempDir::new().unwrap();
        assert!(!is_coderank_downloaded(tmp.path()));
    }

    #[test]
    fn coderank_detected_when_files_present() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = tmp.path().join(CODERANK_DIR);
        fs::create_dir_all(&dir).unwrap();
        for f in CODERANK_FILES {
            fs::write(dir.join(f), b"stub").unwrap();
        }
        assert!(is_coderank_downloaded(tmp.path()));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Add to `crates/semantex-core/src/embedding/mod.rs`:

```rust
pub mod colbert;
pub mod model_manager;
pub mod runtime_manager;
pub mod single_vector_model;
```

Run: `cargo test -p semantex-core single_vector_model::tests 2>&1 | tail -20`
Expected: FAIL — `cannot find value 'CODERANK_DIR'` / `cannot find function 'is_coderank_downloaded'`.

- [ ] **Step 3: Write minimal implementation**

Add the deps to `crates/semantex-core/Cargo.toml` (after the `fastembed` line; tokenizers is already transitive at 0.22.2 — pin the same minor). **Use the exact HNSW crate + version RECORDED in Task 2.**

```toml
# Single-vector dense path (S2): HuggingFace tokenizer for CodeRankEmbed. Already
# in the tree transitively (via fastembed/next-plaid-onnx); declared directly so
# embedding/single_vector.rs can tokenize. `onig` regex backend builds offline.
tokenizers = { version = "0.22", default-features = false, features = ["onig"] }
# Pure-Rust HNSW ANN index for the coderank-hnsw backend (S2 Spike 2). No C/C++.
<hnsw-crate> = "<version>"   # e.g. hnsw_rs = "0.3"  — RECORDED in research-notes
```

Then add to `single_vector_model.rs`, ABOVE the test module (use the RECORDED base URL / files / dir from Spike 1):

```rust
/// HF resolve base for the project-hosted CodeRankEmbed int8 export.
/// RECORDED in research-notes (S2 Spike 1, Task 1 Step 4).
const CODERANK_BASE_URL: &str =
    "https://huggingface.co/MisterTK/CodeRankEmbed-onnx-int8/resolve/main";
/// Files to fetch. RECORDED: the int8 ONNX graph + its tokenizer.
pub(crate) const CODERANK_FILES: &[&str] = &["model_int8.onnx", "tokenizer.json"];
/// On-disk model subdirectory under `models_dir`.
pub(crate) const CODERANK_DIR: &str = "CodeRankEmbed";

/// Download the CodeRankEmbed int8 model if not already cached. Returns the
/// model directory (containing `model_int8.onnx` + `tokenizer.json`).
pub fn ensure_coderank_model(models_dir: &Path) -> Result<PathBuf> {
    let model_dir = models_dir.join(CODERANK_DIR);
    if model_dir.join("model_int8.onnx").exists() && model_dir.join("tokenizer.json").exists() {
        return Ok(model_dir);
    }
    fs::create_dir_all(&model_dir)
        .with_context(|| format!("Failed to create model dir: {}", model_dir.display()))?;
    tracing::info!("Downloading CodeRankEmbed single-vector ONNX model...");
    for file_name in CODERANK_FILES {
        let dest = model_dir.join(file_name);
        if !dest.exists() {
            let url = format!("{CODERANK_BASE_URL}/{file_name}");
            crate::embedding::model_manager::download_file(&url, &dest)
                .with_context(|| format!("Failed to download {file_name} for CodeRankEmbed"))?;
        }
    }
    Ok(model_dir)
}

/// True if every CodeRankEmbed file is present under `models_dir`.
pub fn is_coderank_downloaded(models_dir: &Path) -> bool {
    let model_dir = models_dir.join(CODERANK_DIR);
    CODERANK_FILES.iter().all(|f| model_dir.join(f).exists())
}
```

Note `model_manager::download_file` is already `pub(crate)` (runtime_manager uses it), so this compiles.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core single_vector_model::tests 2>&1 | tail -20`
Expected: PASS — both tests pass; crate compiles with the new deps.

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/Cargo.toml crates/semantex-core/src/embedding/single_vector_model.rs crates/semantex-core/src/embedding/mod.rs
git commit -m "feat(embedding): CodeRankEmbed model provisioning + tokenizers/hnsw deps (S2)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: int8 quantization helpers (pure, testable without the model)

**Files:**
- Create: `crates/semantex-core/src/embedding/single_vector.rs` (quant helpers + tests only; encoder follows in Task 6)
- Modify: `crates/semantex-core/src/embedding/mod.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/semantex-core/src/embedding/single_vector.rs` with the header + quant helpers' tests first:

```rust
//! Single-vector dense encoder for the `coderank-hnsw` backend.
//!
//! Wraps CodeRankEmbed (int8 ONNX) via a raw `ort::Session`. Lazy session init,
//! CPU execution provider pinned, threads from `SEMANTEX_(INDEX_)ORT_THREADS` —
//! the same rationale as `colbert.rs` (a single multi-thread CPU session; `Auto`
//! would collapse to 1 thread on non-macOS, see colbert.rs:147-157).
//!
//! Pipeline: tokenize → forward → pool (per RECORDED model card) → L2-normalize.
//! Documents embed RAW code (no prefix); queries get the RECORDED query prefix.
//! The BM25 enrichment (NL annotation + expansion) is NOT fed here — dense sees
//! only the raw chunk content. No Matryoshka (fixed dim).

use anyhow::Result;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_normalize_unit_length() {
        let mut v = vec![3.0f32, 4.0];
        l2_normalize(&mut v);
        let norm = (v[0] * v[0] + v[1] * v[1]).sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "expected unit norm, got {norm}");
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_zero_vector_is_safe() {
        let mut v = vec![0.0f32, 0.0, 0.0];
        l2_normalize(&mut v); // must not divide by zero / NaN
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn int8_round_trip_preserves_direction() {
        // A normalized vector quantized then dequantized should stay close.
        let mut v = vec![0.1f32, -0.5, 0.8, 0.2, -0.3];
        l2_normalize(&mut v);
        let (q, scale) = quantize_int8(&v);
        assert_eq!(q.len(), v.len());
        assert!(scale > 0.0);
        let back = dequantize_int8(&q, scale);
        // cosine(original, round-trip) should be very high.
        let dot: f32 = v.iter().zip(&back).map(|(a, b)| a * b).sum();
        let nb: f32 = back.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos = dot / nb.max(1e-12);
        assert!(cos > 0.99, "int8 round-trip cosine too low: {cos}");
    }

    #[test]
    fn int8_all_zero_vector_has_nonzero_scale() {
        let (q, scale) = quantize_int8(&[0.0, 0.0, 0.0]);
        assert!(q.iter().all(|&x| x == 0));
        assert!(scale > 0.0, "scale must never be zero (avoids div-by-zero on dequant)");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Add to `crates/semantex-core/src/embedding/mod.rs`:

```rust
pub mod single_vector;
```

Run: `cargo test -p semantex-core embedding::single_vector::tests 2>&1 | tail -20`
Expected: FAIL — `cannot find function 'l2_normalize'` / `quantize_int8` / `dequantize_int8`.

- [ ] **Step 3: Write minimal implementation**

Add to `single_vector.rs`, ABOVE the test module:

```rust
/// L2-normalize in place. A zero vector is left as zeros (no div-by-zero).
pub(crate) fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// Symmetric per-vector int8 quantization. Returns `(quantized, scale)` where
/// `dequantized[i] ≈ quantized[i] as f32 * scale`. The scale is
/// `max(|v|) / 127`, floored to a tiny positive value so dequant never divides
/// by — or multiplies into — zero. This is the standard scalar-quant recipe;
/// L2-normalized inputs keep `max(|v|)` ≈ O(1), so int8 resolution is ample.
pub(crate) fn quantize_int8(v: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = v.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    let scale = (max_abs / 127.0).max(1e-8);
    let q = v
        .iter()
        .map(|&x| {
            let scaled = (x / scale).round();
            scaled.clamp(-127.0, 127.0) as i8
        })
        .collect();
    (q, scale)
}

/// Inverse of [`quantize_int8`].
pub(crate) fn dequantize_int8(q: &[i8], scale: f32) -> Vec<f32> {
    q.iter().map(|&x| f32::from(x) * scale).collect()
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core embedding::single_vector::tests 2>&1 | tail -20`
Expected: PASS — all 4 quant tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/embedding/single_vector.rs crates/semantex-core/src/embedding/mod.rs
git commit -m "feat(embedding): int8 quantization + L2-normalize helpers for single-vector path (S2)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: scalar distance kernels (`search/simd.rs`) — S6 swap-in point

**Files:**
- Create: `crates/semantex-core/src/search/simd.rs`
- Modify: `crates/semantex-core/src/search/mod.rs`

> **First check:** if S6 has already created `crates/semantex-core/src/search/simd.rs`, SKIP this task entirely — verify it exposes `dot_f32`, `cosine_f32`, `dot_i8` with the signatures below; if it does, just reference them in Task 7. Only create the file if it is absent.

- [ ] **Step 1: Write the failing test**

Create `crates/semantex-core/src/search/simd.rs`:

```rust
//! Distance kernels for the single-vector dense path (S2 hot path + brute-force
//! fallback + fp32 rescore). Scalar today; S6 replaces the bodies with
//! runtime-dispatched AVX2/NEON behind these exact signatures (the parity test
//! below is S6's contract — keep it).

/// Dot product of two equal-length f32 slices. Panics on length mismatch.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot_f32 length mismatch");
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Cosine similarity. Inputs need NOT be normalized.
#[inline]
pub fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_f32(a, b);
    let na = dot_f32(a, a).sqrt();
    let nb = dot_f32(b, b).sqrt();
    let denom = (na * nb).max(1e-12);
    dot / denom
}

/// Integer dot product of two equal-length i8 slices (for scoring quantized
/// vectors before fp32 rescore). Accumulates in i32 — for 768-dim int8 the max
/// magnitude is 768 * 127 * 127 ≈ 1.2e7, well within i32.
#[inline]
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len(), "dot_i8 length mismatch");
    a.iter().zip(b).map(|(&x, &y)| i32::from(x) * i32::from(y)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_f32_basic() {
        assert!((dot_f32(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_is_zero_parallel_is_one() {
        assert!(cosine_f32(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert!((cosine_f32(&[1.0, 2.0], &[2.0, 4.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn dot_i8_basic() {
        assert_eq!(dot_i8(&[1, 2, 3], &[4, 5, 6]), 32);
        assert_eq!(dot_i8(&[127, -127], &[127, 127]), 127 * 127 - 127 * 127);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Add to `crates/semantex-core/src/search/mod.rs` (alphabetical-ish, after `regex_semantic`):

```rust
pub mod ripgrep_fallback;
pub mod simd;
pub mod sparse_search;
```

Run: `cargo test -p semantex-core search::simd::tests 2>&1 | tail -20`
Expected: FAIL before adding the `pub mod simd;` line (unresolved module); PASS-shaped once added — so to see a real RED, run BEFORE editing `mod.rs`: `cannot find module`. After adding the module line the test compiles and the bodies are present, so this task's RED is the missing-module state.

- [ ] **Step 3: Write minimal implementation**

Already written in Step 1 (the bodies are complete). No additional code.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core search::simd::tests 2>&1 | tail -20`
Expected: PASS — all 3 kernel tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/search/simd.rs crates/semantex-core/src/search/mod.rs
git commit -m "feat(search): scalar dot/cosine/int8 distance kernels (S2; S6 swap-in point)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: `SingleVectorEmbedder` — lazy ONNX encoder (encode_document / encode_query)

This wires the raw `ort::Session` (CodeRankEmbed) using the RECORDED tensor names, pooling, prefix, and dim from Spike 1. Lazy-init + CPU-EP pin + threads mirror `colbert.rs` verbatim.

**Files:**
- Modify: `crates/semantex-core/src/embedding/single_vector.rs`

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` block in `single_vector.rs` (structural tests that do NOT require the model — the end-to-end encode is exercised by the `#[ignore]` gate in Task 12):

```rust
    #[test]
    fn embedder_new_is_lazy_no_session_at_construction() {
        let tmp = tempfile::TempDir::new().unwrap();
        // Dir exists, no model files — construction must succeed (fail-late).
        let emb = SingleVectorEmbedder::new(tmp.path())
            .expect("constructor must succeed for any existing dir");
        assert!(!emb.is_initialized(), "session must not build at construction");
    }

    #[test]
    fn embedder_rejects_missing_dir() {
        let res = SingleVectorEmbedder::new(Path::new("/nonexistent/s2/model/dir"));
        assert!(res.is_err(), "missing model dir must fail at construction");
    }

    #[test]
    fn query_prefix_is_applied_document_is_raw() {
        // The prefixing logic is pure string work — test it without a session.
        let doc = "fn add(a:i32,b:i32)->i32{a+b}";
        assert_eq!(prefix_document(doc), doc, "documents get NO prefix");
        let q = "add two integers";
        assert_eq!(prefix_query(q), format!("{QUERY_PREFIX}{q}"));
        assert!(QUERY_PREFIX.ends_with(' '), "RECORDED prefix keeps its trailing space");
    }

    #[test]
    fn embedding_dim_is_recorded_constant() {
        assert_eq!(SingleVectorEmbedder::embedding_dim(), EMBEDDING_DIM);
        assert!(EMBEDDING_DIM > 0);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p semantex-core embedding::single_vector::tests::embedder 2>&1 | tail -20`
Expected: FAIL — `cannot find type 'SingleVectorEmbedder'`, `prefix_document`, `QUERY_PREFIX`, `EMBEDDING_DIM`.

- [ ] **Step 3: Write minimal implementation**

Add to `single_vector.rs` (above the test module). **Replace the placeholder `EMBEDDING_DIM`, `QUERY_PREFIX`, `INPUT_IDS`/`ATTENTION_MASK`/`OUTPUT_NAME`, `MAX_CTX`, and the pooling choice with the values RECORDED in Spike 1.** The structure below is grounded in the verified ort rc.11 + tokenizers 0.22 API.

```rust
use crate::search::simd::dot_f32;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use parking_lot::Mutex;
use tokenizers::Tokenizer;

// ── RECORDED from S2 Spike 1 (research-notes) — UPDATE to the probed values ──
/// CodeRankEmbed embedding dimension.
pub(crate) const EMBEDDING_DIM: usize = 768;
/// Max context the encoder accepts; we truncate token ids to this.
const MAX_CTX: usize = 8192;
/// Query prefix (documents get none). EXACT, trailing space included.
pub(crate) const QUERY_PREFIX: &str = "Represent this query for searching relevant code: ";
/// ONNX input tensor names.
const INPUT_IDS: &str = "input_ids";
const ATTENTION_MASK: &str = "attention_mask";
/// ONNX output tensor name holding `[1, seq, dim]` token embeddings.
const OUTPUT_NAME: &str = "last_hidden_state";
/// Whether the model card pooling is CLS (token 0) vs mean over the mask.
const POOL_CLS: bool = true;
// ─────────────────────────────────────────────────────────────────────────────

/// Apply the query prefix (search side).
pub(crate) fn prefix_query(q: &str) -> String {
    format!("{QUERY_PREFIX}{q}")
}
/// Documents are embedded raw (no prefix).
pub(crate) fn prefix_document(d: &str) -> &str {
    d
}

/// Query-tuned ORT intra-op threads (low; many concurrent daemons). Mirrors
/// `colbert.rs` query default.
fn query_threads() -> usize {
    crate::config::env_usize("SEMANTEX_ORT_THREADS", 4)
}
/// Indexing-tuned ORT intra-op threads (throughput-bound; gated by index::gate).
/// Mirrors `colbert.rs::index_ort_threads`.
fn index_threads() -> usize {
    let cores = std::thread::available_parallelism().map_or(4, std::num::NonZeroUsize::get);
    crate::config::env_usize("SEMANTEX_INDEX_ORT_THREADS", (cores / 2).clamp(2, 8))
}

/// Lazy single-vector encoder. The ONNX session is built on first encode
/// (double-checked locking), exactly like `ColbertEmbedder`.
pub struct SingleVectorEmbedder {
    model_dir: PathBuf,
    threads: usize,
    use_coreml: bool,
    session: OnceLock<Mutex<Session>>,
    tokenizer: OnceLock<Tokenizer>,
    build_lock: std::sync::Mutex<()>,
}

static GLOBAL: OnceLock<SingleVectorEmbedder> = OnceLock::new();
static GLOBAL_INIT_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

impl SingleVectorEmbedder {
    /// CodeRankEmbed embedding dimension (RECORDED).
    pub fn embedding_dim() -> usize {
        EMBEDDING_DIM
    }

    /// Construct a lazy embedder (query thread profile). Cheap; no session.
    pub fn new(model_dir: &Path) -> Result<Self> {
        Self::with_threads(model_dir, query_threads())
    }

    /// Construct a lazy embedder tuned for indexing (more threads).
    pub fn for_indexing(model_dir: &Path) -> Result<Self> {
        Self::with_threads(model_dir, index_threads())
    }

    fn with_threads(model_dir: &Path, threads: usize) -> Result<Self> {
        if !model_dir.exists() {
            anyhow::bail!(
                "CodeRankEmbed model dir does not exist: {}",
                model_dir.display()
            );
        }
        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            threads: threads.max(1),
            use_coreml: std::env::var("SEMANTEX_COREML").is_ok_and(|v| v == "1"),
            session: OnceLock::new(),
            tokenizer: OnceLock::new(),
            build_lock: std::sync::Mutex::new(()),
        })
    }

    /// Process-global singleton (query path), mirroring `ColbertEmbedder::global`.
    pub fn global(model_dir: &Path) -> Result<&'static SingleVectorEmbedder> {
        if let Some(e) = GLOBAL.get() {
            return Ok(e);
        }
        let _g = GLOBAL_INIT_LOCK.lock();
        if let Some(e) = GLOBAL.get() {
            return Ok(e);
        }
        let e = Self::new(model_dir)?;
        let _ = GLOBAL.set(e);
        Ok(GLOBAL.get().expect("just set"))
    }

    pub fn is_initialized(&self) -> bool {
        self.session.get().is_some()
    }

    /// CPU execution provider pinned on every platform (CoreML only via
    /// SEMANTEX_COREML on macOS) — same rationale as `colbert.rs`: a single
    /// multi-thread CPU session; `Auto` would silently collapse to 1 thread on
    /// Linux/Windows where no GPU EP is compiled.
    fn execution_providers(&self) -> Vec<ort::ep::ExecutionProviderDispatch> {
        let mut providers = Vec::new();
        #[cfg(target_os = "macos")]
        if self.use_coreml {
            providers.push(ort::ep::CoreML::default().build());
        }
        #[cfg(not(target_os = "macos"))]
        let _ = self.use_coreml;
        providers.push(ort::ep::CPU::default().build());
        providers
    }

    fn build_session(&self) -> Result<Session> {
        // Defensive: ensure the ONNX Runtime shared lib is provisioned (the CLI
        // already sets ORT_DYLIB_PATH at startup; this is a no-op when present).
        let runtime_root = crate::config::SemantexConfig::semantex_home().join("runtime");
        let _ = crate::embedding::runtime_manager::ensure_onnxruntime(&runtime_root);

        let model_path = self.model_dir.join("model_int8.onnx");
        let session = Session::builder()?
            .with_execution_providers(self.execution_providers())?
            .with_intra_threads(self.threads)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(&model_path)?;
        Ok(session)
    }

    fn session(&self) -> Result<&Mutex<Session>> {
        if let Some(s) = self.session.get() {
            return Ok(s);
        }
        let _guard = self
            .build_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(s) = self.session.get() {
            return Ok(s);
        }
        let built = self.build_session()?;
        let _ = self.session.set(Mutex::new(built));
        Ok(self.session.get().expect("session set above"))
    }

    fn tokenizer(&self) -> Result<&Tokenizer> {
        if let Some(t) = self.tokenizer.get() {
            return Ok(t);
        }
        let path = self.model_dir.join("tokenizer.json");
        let tok = Tokenizer::from_file(&path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer {}: {e}", path.display()))?;
        let _ = self.tokenizer.set(tok);
        Ok(self.tokenizer.get().expect("tokenizer set above"))
    }

    /// Encode a document (raw code, no prefix) → L2-normalized embedding.
    pub fn encode_document(&self, text: &str) -> Result<Vec<f32>> {
        self.encode_text(prefix_document(text))
    }

    /// Encode a query (RECORDED prefix prepended) → L2-normalized embedding.
    pub fn encode_query(&self, text: &str) -> Result<Vec<f32>> {
        self.encode_text(&prefix_query(text))
    }

    fn encode_text(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer()?
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenize failed: {e}"))?;

        // Truncate to the model's max context (RECORDED).
        let ids_u32 = encoding.get_ids();
        let mask_u32 = encoding.get_attention_mask();
        let len = ids_u32.len().min(MAX_CTX);
        let ids: Vec<i64> = ids_u32[..len].iter().map(|&x| i64::from(x)).collect();
        let mask: Vec<i64> = mask_u32[..len].iter().map(|&x| i64::from(x)).collect();

        let seq = ids.len() as i64;
        let id_tensor = Tensor::from_array((vec![1_i64, seq], ids))?;
        let mask_tensor = Tensor::from_array((vec![1_i64, seq], mask.clone()))?;

        let session = self.session()?;
        let outputs = session.lock().run(ort::inputs![
            INPUT_IDS => id_tensor,
            ATTENTION_MASK => mask_tensor,
        ])?;

        // Output is [1, seq, dim] f32 token embeddings.
        let (shape, data) = outputs[OUTPUT_NAME].try_extract_tensor::<f32>()?;
        anyhow::ensure!(
            shape.len() == 3 && shape[2] as usize == EMBEDDING_DIM,
            "unexpected encoder output shape {shape:?} (expected [1, seq, {EMBEDDING_DIM}])"
        );
        let seq_len = shape[1] as usize;
        let dim = EMBEDDING_DIM;

        let mut pooled = vec![0.0f32; dim];
        if POOL_CLS {
            // CLS pooling = token 0's row.
            pooled.copy_from_slice(&data[0..dim]);
        } else {
            // Mean pooling over masked tokens.
            let mut count = 0.0f32;
            for t in 0..seq_len {
                if mask.get(t).copied().unwrap_or(0) == 0 {
                    continue;
                }
                count += 1.0;
                let row = &data[t * dim..(t + 1) * dim];
                for (p, &x) in pooled.iter_mut().zip(row) {
                    *p += x;
                }
            }
            if count > 0.0 {
                for p in pooled.iter_mut() {
                    *p /= count;
                }
            }
        }
        l2_normalize(&mut pooled);
        // Keep `dot_f32` referenced for the rescore path (imported for callers).
        debug_assert!(dot_f32(&pooled, &pooled) <= 1.0 + 1e-3);
        Ok(pooled)
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core embedding::single_vector::tests 2>&1 | tail -20`
Expected: PASS — the structural tests pass; crate compiles against ort/tokenizers.

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/embedding/single_vector.rs
git commit -m "feat(embedding): SingleVectorEmbedder lazy CodeRankEmbed ONNX encoder (S2)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: `hnsw_index.rs` — HNSW backend (params, brute-force fallback, ANN→rescore, streaming build)

Implements both S1 traits. **Code the HNSW crate calls against the EXACT API RECORDED in Task 2** — the snippets below use placeholder names (`HnswGraph`, `graph.insert`, `graph.search`) clearly marked; substitute the recorded signatures. The int8 store + chunk_id map + brute-force + rescore + streaming are crate-agnostic and complete as written.

**Files:**
- Create: `crates/semantex-core/src/index/hnsw_index.rs`
- Modify: `crates/semantex-core/src/index/mod.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/semantex-core/src/index/hnsw_index.rs`:

```rust
//! `coderank-hnsw` dense backend: CodeRankEmbed single-vector embeddings stored
//! int8-quantized on disk, searched via HNSW ANN over int8 → fp32 rescore of the
//! top `rescore_k`, with a brute-force fallback below `HNSW_MIN_VECTORS`.
//! Implements the S1 `DenseBackend` + `DenseIndexBuilder` traits.
//!
//! Per S1 G5: `positional_chunk_ids()` returns `None` (no positional doc array),
//! so the `hybrid.rs` file_filter subset path degrades to unfiltered dense +
//! result-merge filtering — documented on the trait impl below.

use crate::embedding::single_vector::{SingleVectorEmbedder, dequantize_int8, quantize_int8};
use crate::search::dense_backend::{DenseBackend, DenseHit, DenseIndexBuilder};
use crate::search::simd::{dot_f32, dot_i8};
use crate::types::ScoredChunkId;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Below this many vectors, skip the HNSW graph and brute-force (exact, and
/// avoids graph overhead on small repos). Overridable via `SEMANTEX_HNSW_MIN_VECTORS`.
const HNSW_MIN_VECTORS_DEFAULT: usize = 2_000;

/// HNSW + rescore tuning. Presets mirror the oxirs config-preset idea.
#[derive(Debug, Clone, Copy)]
pub struct HnswParams {
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    /// fp32-rescore the top `rescore_k` ANN candidates. Effective = rescore_k.max(k).
    pub rescore_k: usize,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self { m: 16, ef_construction: 200, ef_search: 64, rescore_k: 0 }
    }
}

impl HnswParams {
    /// Resolve a named preset. `rescore_k = 0` means "derive 4×k at query time".
    pub fn preset(name: &str) -> Self {
        match name.trim().to_ascii_lowercase().as_str() {
            "high_recall" => Self { m: 32, ef_construction: 400, ef_search: 200, rescore_k: 0 },
            "low_latency" => Self { m: 16, ef_construction: 200, ef_search: 32, rescore_k: 0 },
            "memory_optimized" => Self { m: 8, ef_construction: 100, ef_search: 48, rescore_k: 0 },
            _ => Self::default(), // "default" and unknown
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_names_resolve() {
        assert_eq!(HnswParams::preset("default").m, 16);
        assert_eq!(HnswParams::preset("high_recall").ef_search, 200);
        assert_eq!(HnswParams::preset("low_latency").ef_search, 32);
        assert_eq!(HnswParams::preset("memory_optimized").m, 8);
        assert_eq!(HnswParams::preset("garbage").m, 16); // unknown → default
    }

    #[test]
    fn brute_force_ranks_by_cosine_descending() {
        // 3 unit vectors; query closest to id=10, then 20, then 30.
        let store = Int8VectorStore {
            dim: 2,
            scales: vec![1.0, 1.0, 1.0],
            vectors: vec![
                quantize_int8(&[1.0, 0.0]).0,   // id 10
                quantize_int8(&[0.7071, 0.7071]).0, // id 20
                quantize_int8(&[0.0, 1.0]).0,   // id 30
            ],
            chunk_ids: vec![10, 20, 30],
        };
        let q = vec![1.0f32, 0.0];
        let hits = store.brute_force(&q, 3);
        assert_eq!(hits.iter().map(|h| h.chunk_id).collect::<Vec<_>>(), vec![10, 20, 30]);
        assert!(hits[0].score >= hits[1].score && hits[1].score >= hits[2].score);
    }

    #[test]
    fn empty_store_returns_no_hits() {
        let store = Int8VectorStore { dim: 4, scales: vec![], vectors: vec![], chunk_ids: vec![] };
        assert!(store.brute_force(&[0.0, 0.0, 0.0, 0.0], 5).is_empty());
    }

    #[test]
    fn rescore_reorders_int8_candidates_by_fp32() {
        // Two candidates with near-equal int8 score; fp32 rescore must order them.
        let store = Int8VectorStore {
            dim: 3,
            scales: vec![1.0, 1.0],
            vectors: vec![
                quantize_int8(&[0.9, 0.1, 0.0]).0,
                quantize_int8(&[0.8, 0.2, 0.0]).0,
            ],
            chunk_ids: vec![1, 2],
        };
        let q = vec![1.0f32, 0.0, 0.0];
        // candidate positions 0 and 1, rescore both → id 1 (higher dot) first.
        let hits = store.fp32_rescore(&q, &[0, 1], 2);
        assert_eq!(hits[0].chunk_id, 1);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Add to `crates/semantex-core/src/index/mod.rs` (alphabetical, after `global_graph`):

```rust
pub mod global_graph;
pub mod hnsw_index;
pub mod page_rank;
```

Run: `cargo test -p semantex-core index::hnsw_index::tests 2>&1 | tail -20`
Expected: FAIL — `cannot find type 'Int8VectorStore'` and method `brute_force`/`fp32_rescore`.

- [ ] **Step 3: Write minimal implementation**

Add to `hnsw_index.rs` (above the test module). The `Int8VectorStore` + brute-force + rescore are complete and crate-agnostic; the `HnswGraph`/`graph.*` calls are the RECORDED-API substitution points (clearly marked).

```rust
/// On-disk int8 vector store. Serialized with postcard alongside the HNSW graph.
/// `vectors[i]` is the int8 quantization of chunk `chunk_ids[i]` with per-vector
/// `scales[i]`. Positional: graph node id `i` ↔ this row ↔ `chunk_ids[i]`.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Int8VectorStore {
    pub dim: usize,
    pub scales: Vec<f32>,
    pub vectors: Vec<Vec<i8>>,
    pub chunk_ids: Vec<u64>,
}

impl Int8VectorStore {
    fn len(&self) -> usize {
        self.chunk_ids.len()
    }

    /// Append one quantized vector for `chunk_id`. Returns its row index.
    fn push(&mut self, chunk_id: u64, q: Vec<i8>, scale: f32) -> usize {
        let idx = self.chunk_ids.len();
        self.vectors.push(q);
        self.scales.push(scale);
        self.chunk_ids.push(chunk_id);
        idx
    }

    /// Exact brute-force top-k over int8 (used below `HNSW_MIN_VECTORS` and as
    /// the rescore primitive's sibling). Scores via int8 dot (monotonic in
    /// cosine for L2-normalized inputs) then keeps the fp32-equivalent score.
    fn brute_force(&self, query: &[f32], k: usize) -> Vec<DenseHit> {
        let (q8, _qs) = quantize_int8(query);
        let mut scored: Vec<(usize, i32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, dot_i8(&q8, v)))
            .collect();
        scored.sort_by(|a, b| b.1.cmp(&a.1));
        scored.truncate(k.max(1));
        let positions: Vec<usize> = scored.into_iter().map(|(i, _)| i).collect();
        // Always fp32-rescore the brute-force shortlist for exact ranking.
        self.fp32_rescore(query, &positions, k)
    }

    /// fp32-rescore the given row `positions` (dequantize → cosine via dot on
    /// L2-normalized vectors) and return the top `k` as `DenseHit`s sorted desc.
    fn fp32_rescore(&self, query: &[f32], positions: &[usize], k: usize) -> Vec<DenseHit> {
        let mut hits: Vec<DenseHit> = positions
            .iter()
            .filter_map(|&i| {
                let v = self.vectors.get(i)?;
                let scale = *self.scales.get(i)?;
                let f = dequantize_int8(v, scale);
                // Stored vectors were L2-normalized before quantization; query is
                // normalized by the encoder. dot == cosine.
                let score = dot_f32(query, &f);
                Some(ScoredChunkId::new(self.chunk_ids[i], score))
            })
            .collect();
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        hits.truncate(k.max(1));
        hits
    }
}

fn hnsw_min_vectors() -> usize {
    crate::config::env_usize("SEMANTEX_HNSW_MIN_VECTORS", HNSW_MIN_VECTORS_DEFAULT)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core index::hnsw_index::tests 2>&1 | tail -20`
Expected: PASS — all 4 store/preset tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/index/hnsw_index.rs crates/semantex-core/src/index/mod.rs
git commit -m "feat(index): int8 vector store + HNSW params/presets + brute-force/rescore (S2)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: `HnswDenseIndex` — persisted graph + `DenseBackend`/`DenseIndexBuilder` impls

Now wrap the store + the HNSW crate into the two traits, with persistence, streaming build, and the ANN-or-brute-force query path. **Substitute the RECORDED HNSW API for the `HnswGraph` placeholder calls.**

**Files:**
- Modify: `crates/semantex-core/src/index/hnsw_index.rs`

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` block:

```rust
    #[test]
    fn builder_name_and_empty_build_is_ok() {
        let tmp = tempfile::TempDir::new().unwrap();
        let mut b = CoderankHnswIndexBuilder::new(tmp.path(), HnswParams::default());
        assert_eq!(DenseIndexBuilder::name(&b), "coderank-hnsw");
        // Empty corpus → no-op success, no store file written.
        b.build(&[]).unwrap();
        assert!(!tmp.path().join("vectors.bin").exists());
    }

    #[test]
    fn store_persist_and_reload_round_trips() {
        let tmp = tempfile::TempDir::new().unwrap();
        let mut store = Int8VectorStore { dim: 2, ..Default::default() };
        let (q, s) = quantize_int8(&[0.6, 0.8]);
        store.push(42, q, s);
        let path = tmp.path().join("vectors.bin");
        std::fs::write(&path, postcard::to_stdvec(&store).unwrap()).unwrap();
        let back: Int8VectorStore =
            postcard::from_bytes(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(back.chunk_ids, vec![42]);
        assert_eq!(back.dim, 2);
    }

    #[test]
    fn backend_positional_chunk_ids_is_none() {
        // S1 G5: HNSW has no positional doc array; subset path degrades to merge.
        let tmp = tempfile::TempDir::new().unwrap();
        let store = Int8VectorStore { dim: 4, ..Default::default() };
        std::fs::write(
            tmp.path().join("vectors.bin"),
            postcard::to_stdvec(&store).unwrap(),
        )
        .unwrap();
        let backend = CoderankHnswBackend::open_for_test(tmp.path());
        assert!(backend.positional_chunk_ids().is_none());
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p semantex-core index::hnsw_index::tests::builder_name 2>&1 | tail -20`
Expected: FAIL — `cannot find type 'CoderankHnswIndexBuilder'` / `'CoderankHnswBackend'`.

- [ ] **Step 3: Write minimal implementation**

Add to `hnsw_index.rs`. The persistence (postcard of `Int8VectorStore`) and query dispatch are complete; the `// HNSW-API:` comments mark where the RECORDED crate calls go. Per Spike 2's serialize finding, we persist the int8 vectors + map ourselves and rebuild the graph on open if the crate has no native serialize (the common case) — that keeps persistence crate-independent.

```rust
const STORE_FILE: &str = "vectors.bin";

/// The persisted dense index: the int8 vector store (+ rebuilt HNSW graph in RAM).
pub struct HnswDenseIndex {
    store: Int8VectorStore,
    params: HnswParams,
    /// In-RAM HNSW graph, built from `store` on open when len ≥ HNSW_MIN_VECTORS.
    /// `None` ⇒ brute-force path. (Type is the RECORDED crate's graph.)
    graph: Option<HnswGraph>,
}

// HNSW-API: alias the RECORDED crate's graph type, e.g.
//   use hnsw_rs::prelude::{Hnsw, DistCosine};
//   type HnswGraph = Hnsw<'static, f32, DistCosine>;
// Substitute the real type. The methods used below are: new(...), insert((&[f32], usize)),
// search(&[f32], k, ef) -> Vec<Neighbour { d_id: usize, distance: f32 }>.
type HnswGraph = PlaceholderGraph; // ← REPLACE with the recorded type

impl HnswDenseIndex {
    /// Build the in-RAM graph from the store (dequantized fp32 vectors), or
    /// return `None` for a small store (brute-force path).
    fn build_graph(store: &Int8VectorStore, params: HnswParams) -> Option<HnswGraph> {
        if store.len() < hnsw_min_vectors() {
            return None;
        }
        // HNSW-API: construct + insert every row. Example shape:
        //   let g = Hnsw::<f32, DistCosine>::new(params.m, store.len(), 16,
        //                                        params.ef_construction, DistCosine{});
        //   for i in 0..store.len() {
        //       let f = dequantize_int8(&store.vectors[i], store.scales[i]);
        //       g.insert((&f, i));   // node id i ↔ store row i ↔ chunk_ids[i]
        //   }
        //   Some(g)
        None // ← REPLACE with the recorded construction
    }

    fn load(dir: &Path, params: HnswParams) -> Result<Self> {
        let bytes = std::fs::read(dir.join(STORE_FILE))?;
        let store: Int8VectorStore = postcard::from_bytes(&bytes)?;
        let graph = Self::build_graph(&store, params);
        Ok(Self { store, params, graph })
    }

    fn save(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join(STORE_FILE), postcard::to_stdvec(&self.store)?)?;
        Ok(())
    }

    /// Top-k search: ANN (if graph present) → fp32 rescore; else brute-force.
    fn search(&self, query: &[f32], k: usize) -> Vec<DenseHit> {
        if self.store.len() == 0 {
            return Vec::new();
        }
        let rescore_k = if self.params.rescore_k == 0 { 4 * k } else { self.params.rescore_k }.max(k);
        match &self.graph {
            None => self.store.brute_force(query, k),
            Some(_g) => {
                // HNSW-API: query the graph for `rescore_k` ANN candidates, then
                // fp32-rescore. Example shape:
                //   let neighbours = _g.search(query, rescore_k, self.params.ef_search);
                //   let positions: Vec<usize> = neighbours.iter().map(|n| n.d_id).collect();
                //   self.store.fp32_rescore(query, &positions, k)
                let positions: Vec<usize> = (0..self.store.len().min(rescore_k)).collect(); // ← REPLACE
                self.store.fp32_rescore(query, &positions, k)
            }
        }
    }
}

/// Query-time backend.
pub struct CoderankHnswBackend {
    index: HnswDenseIndex,
    embedder: &'static SingleVectorEmbedder,
}

impl CoderankHnswBackend {
    pub const NAME: &'static str = "coderank-hnsw";

    pub fn open(dir: &Path, model_dir: &Path, params: HnswParams) -> Result<Self> {
        let index = HnswDenseIndex::load(dir, params)?;
        let embedder = SingleVectorEmbedder::global(model_dir)?;
        Ok(Self { index, embedder })
    }

    #[cfg(test)]
    fn open_for_test(dir: &Path) -> Self {
        // Test-only: load the store without a model (embedder unused by the
        // positional-ids / load tests).
        let index = HnswDenseIndex::load(dir, HnswParams::default())
            .expect("test store loads");
        // Leak a dummy embedder pointer is unsafe; instead the only test that
        // calls this asserts positional_chunk_ids(), which never touches the
        // embedder — so we build a global against a temp dir.
        let tmp = std::env::temp_dir();
        let embedder = SingleVectorEmbedder::global(&tmp)
            .or_else(|_| SingleVectorEmbedder::global(&tmp))
            .expect("global embedder (lazy; no session built)");
        Self { index, embedder }
    }
}

impl DenseBackend for CoderankHnswBackend {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn search(&self, query: &str, k: usize) -> Result<Vec<DenseHit>> {
        let q = self.embedder.encode_query(query)?;
        Ok(self.index.search(&q, k))
    }

    fn search_with_subset(&self, query: &str, k: usize, subset: &[u64]) -> Result<Vec<DenseHit>> {
        // HNSW has no positional doc array (positional_chunk_ids → None), so the
        // hybrid.rs subset path normally won't call this with a real subset. If a
        // caller does pass one, honor it by post-filtering the unfiltered top-N
        // (N widened to absorb the filter). This is the documented G5 degradation.
        if subset.is_empty() {
            return Ok(Vec::new());
        }
        let q = self.embedder.encode_query(query)?;
        let wide = self.index.search(&q, (k * 8).max(64));
        let allow: std::collections::HashSet<u64> = subset.iter().copied().collect();
        let mut filtered: Vec<DenseHit> =
            wide.into_iter().filter(|h| allow.contains(&h.chunk_id)).collect();
        filtered.truncate(k);
        Ok(filtered)
    }

    /// S1 G5: no positional doc→chunk array. Returning `None` makes the
    /// `hybrid.rs` file_filter subset path degrade to unfiltered dense + the
    /// result-merge file_filter — which is correct, just not pre-filtered.
    fn positional_chunk_ids(&self) -> Option<&[u64]> {
        None
    }
}

/// Build-time backend. Streams encode→insert: each batch is encoded, quantized,
/// pushed into the store, and the RSS failsafe is checked — never collect all
/// fp32 embeddings then one big call (the D6 build-memory property).
pub struct CoderankHnswIndexBuilder {
    dir: PathBuf,
    params: HnswParams,
    models_dir: PathBuf,
    store: Int8VectorStore,
}

impl CoderankHnswIndexBuilder {
    pub fn new(dir: &Path, params: HnswParams) -> Self {
        Self {
            dir: dir.to_path_buf(),
            params,
            models_dir: crate::config::SemantexConfig::default().models_dir(),
            store: Int8VectorStore { dim: SingleVectorEmbedder::embedding_dim(), ..Default::default() },
        }
    }

    pub fn with_models_dir(mut self, models_dir: PathBuf) -> Self {
        self.models_dir = models_dir;
        self
    }

    /// Encode + quantize a batch into the store, with an RSS check.
    fn encode_into_store(&mut self, embedder: &SingleVectorEmbedder, chunks: &[(u64, &str)]) -> Result<()> {
        const BATCH: usize = 32;
        for batch in chunks.chunks(BATCH) {
            if let Err(e) = crate::memory::check_rss_or_abort("HNSW encode batch") {
                anyhow::bail!("Indexing aborted: {e}");
            }
            for &(chunk_id, content) in batch {
                let emb = embedder.encode_document(content)?;
                let (q, scale) = quantize_int8(&emb);
                self.store.push(chunk_id, q, scale);
            }
        }
        Ok(())
    }

    fn load_existing_store(&self) -> Int8VectorStore {
        std::fs::read(self.dir.join(STORE_FILE))
            .ok()
            .and_then(|b| postcard::from_bytes::<Int8VectorStore>(&b).ok())
            .unwrap_or_else(|| Int8VectorStore {
                dim: SingleVectorEmbedder::embedding_dim(),
                ..Default::default()
            })
    }
}

impl DenseIndexBuilder for CoderankHnswIndexBuilder {
    fn name(&self) -> &'static str {
        CoderankHnswBackend::NAME
    }

    fn build(&mut self, chunks: &[(u64, &str)]) -> Result<()> {
        if chunks.is_empty() {
            tracing::info!("No chunks to encode for coderank-hnsw index");
            return Ok(());
        }
        let model_dir = crate::embedding::single_vector_model::ensure_coderank_model(&self.models_dir)?;
        let embedder = SingleVectorEmbedder::for_indexing(&model_dir)?;
        // Fresh full build.
        self.store = Int8VectorStore { dim: SingleVectorEmbedder::embedding_dim(), ..Default::default() };
        self.encode_into_store(&embedder, chunks)?;
        crate::memory::purge_allocator();
        self.persist(&self.dir.clone())?;
        tracing::info!("coderank-hnsw index built ({} vectors)", self.store.len());
        Ok(())
    }

    fn insert(&mut self, chunks: &[(u64, &str)]) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }
        let model_dir = crate::embedding::single_vector_model::ensure_coderank_model(&self.models_dir)?;
        let embedder = SingleVectorEmbedder::for_indexing(&model_dir)?;
        self.store = self.load_existing_store();
        self.encode_into_store(&embedder, chunks)?;
        self.persist(&self.dir.clone())?;
        Ok(())
    }

    fn delete(&mut self, chunk_ids: &[u64]) -> Result<()> {
        if chunk_ids.is_empty() {
            return Ok(());
        }
        self.store = self.load_existing_store();
        let remove: std::collections::HashSet<u64> = chunk_ids.iter().copied().collect();
        // Rebuild the store without the removed ids (compact; no tombstones —
        // HNSW graph is rebuilt on open from the compacted store).
        let keep: Vec<usize> = (0..self.store.len())
            .filter(|&i| !remove.contains(&self.store.chunk_ids[i]))
            .collect();
        let mut compact = Int8VectorStore { dim: self.store.dim, ..Default::default() };
        for i in keep {
            compact.push(
                self.store.chunk_ids[i],
                std::mem::take(&mut self.store.vectors[i]),
                self.store.scales[i],
            );
        }
        self.store = compact;
        self.persist(&self.dir.clone())?;
        Ok(())
    }

    fn persist(&self, dir: &Path) -> Result<()> {
        let index = HnswDenseIndex { store: std::mem::take(&mut self.store.clone_for_persist()), params: self.params, graph: None };
        index.save(dir)
    }
}
```

Add a tiny `clone_for_persist` helper + a `PlaceholderGraph` zero-sized stand-in so the file compiles BEFORE the HNSW crate substitution (the placeholder is replaced in Step 3's HNSW-API edits; keeping it makes the TDD step compile):

```rust
impl Int8VectorStore {
    /// Clone the store for persistence without consuming `self` (persist takes &self).
    fn clone_for_persist(&self) -> Int8VectorStore {
        Int8VectorStore {
            dim: self.dim,
            scales: self.scales.clone(),
            vectors: self.vectors.clone(),
            chunk_ids: self.chunk_ids.clone(),
        }
    }
}

/// Zero-sized placeholder until the RECORDED HNSW crate type is substituted.
/// Replaced in the HNSW-API edits; brute-force path is used while this stands in.
#[doc(hidden)]
pub struct PlaceholderGraph;
```

> **HNSW-API substitution (do this in the same step, against Task 2's recorded API):** replace `type HnswGraph = PlaceholderGraph;` with the real graph type, fill `build_graph`'s construction/insert loop, and fill `search`'s `graph.search(...)` call + `Neighbour → d_id` mapping. Re-run the tests after substitution. Until substitution, `build_graph` returns `None` and everything goes through the (correct, exact) brute-force path — so the suite is green either way; the graph just isn't exercised until wired.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core index::hnsw_index::tests 2>&1 | tail -20`
Expected: PASS — builder name/empty-build, persist round-trip, and `positional_chunk_ids is None` all pass.

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/index/hnsw_index.rs
git commit -m "feat(index): CoderankHnswBackend + builder (persist, streaming build, ANN/brute-force) (S2)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: `CoderankHnsw` variant in `DenseBackendKind` + three config knobs

**Files:**
- Modify: `crates/semantex-core/src/search/dense_backend.rs` (S1-owned — coordinate per §5)
- Modify: `crates/semantex-core/src/config.rs`

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` block in `dense_backend.rs`:

```rust
    #[test]
    fn parse_coderank_hnsw_backend() {
        assert_eq!(
            DenseBackendKind::parse("coderank-hnsw"),
            Some(DenseBackendKind::CoderankHnsw)
        );
        assert_eq!(
            DenseBackendKind::parse("  Coderank-HNSW "),
            Some(DenseBackendKind::CoderankHnsw)
        );
        assert_eq!(DenseBackendKind::CoderankHnsw.name(), "coderank-hnsw");
    }

    #[test]
    fn coderank_dense_subdir() {
        let p = dense_subdir(Path::new("/x/.semantex"), DenseBackendKind::CoderankHnsw);
        assert_eq!(p, Path::new("/x/.semantex/dense/coderank-hnsw"));
    }
```

Add to the `#[cfg(test)] mod tests` block in `config.rs`:

```rust
    #[test]
    fn dense_tuning_defaults() {
        let cfg = SemantexConfig::default();
        assert_eq!(cfg.hnsw_ef_search, 64);
        assert_eq!(cfg.hnsw_preset, "default");
        assert_eq!(cfg.dense_rescore_k, 0, "0 means derive 4×k at query time");
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p semantex-core dense_backend::tests::parse_coderank config::tests::dense_tuning 2>&1 | tail -20`
Expected: FAIL — no variant `CoderankHnsw`; no fields `hnsw_ef_search`/`hnsw_preset`/`dense_rescore_k`.

- [ ] **Step 3: Write minimal implementation**

In `dense_backend.rs`, add the variant + arms:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DenseBackendKind {
    /// ColBERT late-interaction over a vendored next-plaid PLAID index.
    #[default]
    ColbertPlaid,
    /// CodeRankEmbed single-vector embeddings over a pure-Rust HNSW index (S2).
    CoderankHnsw,
}
```

In `DenseBackendKind::name`:

```rust
    pub fn name(self) -> &'static str {
        match self {
            DenseBackendKind::ColbertPlaid => "colbert-plaid",
            DenseBackendKind::CoderankHnsw => "coderank-hnsw",
        }
    }
```

In `DenseBackendKind::parse`:

```rust
        match s.trim().to_ascii_lowercase().as_str() {
            "colbert-plaid" => Some(DenseBackendKind::ColbertPlaid),
            "coderank-hnsw" => Some(DenseBackendKind::CoderankHnsw),
            _ => None,
        }
```

In `config.rs`, add the fields to `SemantexConfig` (after `dense_backend: String,`):

```rust
    /// HNSW `ef_search` for the coderank-hnsw backend (higher = better recall,
    /// slower). Override via `SEMANTEX_HNSW_EF_SEARCH`. Default 64.
    pub hnsw_ef_search: usize,
    /// HNSW tuning preset: `default | high_recall | low_latency | memory_optimized`.
    /// Override via `SEMANTEX_HNSW_PRESET`. When set, the preset's `ef_search`
    /// is used unless `SEMANTEX_HNSW_EF_SEARCH` is also set (which wins).
    pub hnsw_preset: String,
    /// fp32-rescore the top `dense_rescore_k` ANN candidates. `0` ⇒ derive 4×k
    /// at query time. Override via `SEMANTEX_DENSE_RESCORE_K`.
    pub dense_rescore_k: usize,
```

In `impl Default for SemantexConfig` (after `dense_backend: "colbert-plaid".to_string(),`):

```rust
            hnsw_ef_search: 64,
            hnsw_preset: "default".to_string(),
            dense_rescore_k: 0,
```

In `load()`, after the existing `SEMANTEX_DENSE_BACKEND` overlay (added by S1), append:

```rust
        config.hnsw_ef_search = env_usize("SEMANTEX_HNSW_EF_SEARCH", config.hnsw_ef_search);
        config.hnsw_preset = env_string("SEMANTEX_HNSW_PRESET", &config.hnsw_preset);
        config.dense_rescore_k =
            std::env::var("SEMANTEX_DENSE_RESCORE_K")
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(config.dense_rescore_k);
```

(`env_usize` and `env_string` already exist — `env_string` was added by S1 Task 2. `dense_rescore_k` uses the raw parse because `0` is a meaningful value that `env_usize` would reject.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core dense_backend::tests config::tests 2>&1 | tail -20`
Expected: PASS — new variant/subdir/defaults tests pass; existing dense_backend + config tests still pass.

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/search/dense_backend.rs crates/semantex-core/src/config.rs
git commit -m "feat(search): CoderankHnsw backend variant + HNSW/rescore config knobs (S2)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: bump schema 10 → 11

S1 bumped 9 → 10. S2 changes the dense on-disk layout for the new backend, so it bumps to 11 (forces a clean reindex; G3).

**Files:**
- Modify: `crates/semantex-core/src/types.rs`

- [ ] **Step 1: Write the failing test**

Replace S1's `current_schema_version_is_10` test in `types.rs` with:

```rust
    /// S2: schema bumped 10 → 11 for the single-vector dense on-disk layout
    /// (`dense/coderank-hnsw/vectors.bin`). Older indexes become `Stale`.
    #[test]
    fn current_schema_version_is_11() {
        assert_eq!(IndexMeta::CURRENT_SCHEMA_VERSION, 11);
    }
```

(If S1's test name differs, find the `assert_eq!(IndexMeta::CURRENT_SCHEMA_VERSION, 10)` line and update it.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p semantex-core types::tests::current_schema_version 2>&1 | tail -20`
Expected: FAIL — `assertion failed: CURRENT_SCHEMA_VERSION == 11` (currently 10).

- [ ] **Step 3: Write minimal implementation**

In `types.rs`, bump the constant and extend its doc:

```rust
    /// v11 (S2): the single-vector dense backend (`coderank-hnsw`) introduces a
    /// new on-disk layout (`dense/coderank-hnsw/vectors.bin`). Bumping forces a
    /// clean reindex so an old PLAID-only index isn't half-read by the new path.
    pub const CURRENT_SCHEMA_VERSION: u32 = 11;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core types:: 2>&1 | tail -20`
Expected: PASS — `current_schema_version_is_11` passes; `synthetic_v8_meta_is_stale` (and any S1 stale test) still passes (any version < 11 is Stale).

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/types.rs
git commit -m "feat(index): bump schema 10->11 for single-vector dense layout (S2)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: wire `CoderankHnsw` into `hybrid.rs` open + `builder.rs` build

Add the `CoderankHnsw` arm to the two backend `match`es S1 created. Behavior for `colbert-plaid` is untouched.

**Files:**
- Modify: `crates/semantex-core/src/search/hybrid.rs` (the dense-load `match` in `open()`)
- Modify: `crates/semantex-core/src/index/builder.rs` (the dense-build `match`; meta model/dim)

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` block in `builder.rs`:

```rust
    /// S2: building with dense_backend = "coderank-hnsw" writes the per-backend
    /// subdir `.semantex/dense/coderank-hnsw/vectors.bin` and records the model
    /// + dim in meta.json. `#[ignore]` — needs the CodeRankEmbed model download.
    #[test]
    #[ignore]
    fn coderank_hnsw_build_writes_subdir_and_meta() {
        use crate::config::SemantexConfig;
        let tmp = tempfile::TempDir::new().unwrap();
        let project = tmp.path().join("repo");
        std::fs::create_dir_all(&project).unwrap();
        std::fs::write(project.join("a.rs"), "pub fn hello() -> u32 { 41 + 1 }\n").unwrap();

        let cfg = SemantexConfig { dense_backend: "coderank-hnsw".to_string(), ..SemantexConfig::default() };
        IndexBuilder::new(&cfg).unwrap().build(&project).unwrap();

        assert!(project.join(".semantex/dense/coderank-hnsw/vectors.bin").exists());
        let meta: crate::types::IndexMeta = serde_json::from_str(
            &std::fs::read_to_string(project.join(".semantex/meta.json")).unwrap(),
        ).unwrap();
        assert_eq!(meta.dense_backend, "coderank-hnsw");
        assert_eq!(meta.schema_version, 11);
        assert_eq!(meta.embedding_model, "CodeRankEmbed");
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p semantex-core index::builder::tests::coderank_hnsw_build -- --ignored 2>&1 | tail -20`
Expected: FAIL — the `match backend_kind` in `builder.rs` has no `CoderankHnsw` arm (non-exhaustive match compile error), and `embedding_model` is hardcoded `"LateOn-Code-edge"`.

- [ ] **Step 3: Write minimal implementation**

**(a) `builder.rs` — add the build arm.** In the dense-build `match backend_kind { ... }` that S1 introduced (replacing the old inline PLAID block), add:

```rust
                DenseBackendKind::CoderankHnsw => {
                    use crate::index::hnsw_index::{CoderankHnswIndexBuilder, HnswParams};
                    let mut params = HnswParams::preset(&self.config.hnsw_preset);
                    params.ef_search = self.config.hnsw_ef_search;
                    params.rescore_k = self.config.dense_rescore_k;
                    let mut dense_builder =
                        CoderankHnswIndexBuilder::new(&dense_dir, params)
                            .with_models_dir(self.config.models_dir());
                    if dense_missing {
                        let _slot = crate::index::gate::acquire(|| {
                            tracing::info!(
                                "Waiting for a free index-build slot (max {} concurrent full builds)",
                                crate::index::gate::max_concurrent_builds()
                            );
                        });
                        let all_ids = store.get_all_chunk_ids()?;
                        if all_ids.is_empty() {
                            return Ok(());
                        }
                        let all_chunks = store.get_chunks(&all_ids)?;
                        let pairs: Vec<(u64, &str)> =
                            all_chunks.iter().map(|c| (c.id, c.content.as_str())).collect();
                        dense_builder.build(&pairs)?;
                    } else {
                        if !removed_chunk_ids.is_empty() {
                            dense_builder.delete(&removed_chunk_ids)?;
                        }
                        if !new_chunk_ids.is_empty() {
                            let new_chunks = store.get_chunks(&new_chunk_ids)?;
                            let pairs: Vec<(u64, &str)> =
                                new_chunks.iter().map(|c| (c.id, c.content.as_str())).collect();
                            dense_builder.insert(&pairs)?;
                        }
                    }
                    dense_builder.persist(&dense_dir)?;
                    tracing::info!(
                        added = new_chunk_ids.len(),
                        removed = removed_chunk_ids.len(),
                        "Dense index (coderank-hnsw) build complete"
                    );
                }
```

**(b) `builder.rs` — model/dim in meta.** The `IndexMeta` literal (S1 left `embedding_model: "LateOn-Code-edge"`, `embedding_dim: 48`) must reflect the active backend. Replace those two fields with backend-aware values computed just before the literal:

```rust
        let backend_kind = DenseBackendKind::parse(&self.config.dense_backend).unwrap_or_default();
        let (embedding_model, embedding_dim) = match backend_kind {
            DenseBackendKind::ColbertPlaid => ("LateOn-Code-edge".to_string(), 48u32),
            DenseBackendKind::CoderankHnsw => (
                "CodeRankEmbed".to_string(),
                crate::embedding::single_vector::SingleVectorEmbedder::embedding_dim() as u32,
            ),
        };
```

and in the literal:

```rust
            embedding_model,
            embedding_dim,
```

(If `builder.rs` already computes `backend_kind` earlier in the S1 dense block, reuse that binding instead of recomputing — keep one binding to avoid an unused-variable warning.)

**(c) `hybrid.rs` — add the open arm.** In the dense-load `match DenseBackendKind::parse(&config.dense_backend)` that S1 introduced in `open()`, add a `CoderankHnsw` arm next to `ColbertPlaid`:

```rust
            Some(DenseBackendKind::CoderankHnsw) => {
                use crate::index::hnsw_index::{CoderankHnswBackend, HnswParams};
                let backend_dir = dense_subdir(index_dir, DenseBackendKind::CoderankHnsw);
                if backend_dir.join("vectors.bin").exists() {
                    let mut params = HnswParams::preset(&config.hnsw_preset);
                    params.ef_search = config.hnsw_ef_search;
                    params.rescore_k = config.dense_rescore_k;
                    let model_dir =
                        crate::embedding::single_vector_model::ensure_coderank_model(&config.models_dir());
                    match model_dir.and_then(|d| CoderankHnswBackend::open(&backend_dir, &d, params)) {
                        Ok(b) => {
                            tracing::info!("Dense backend loaded: coderank-hnsw");
                            Some(Box::new(b) as Box<dyn DenseBackend>)
                        }
                        Err(e) => {
                            tracing::warn!("coderank-hnsw backend failed to load: {e}");
                            None
                        }
                    }
                } else {
                    None
                }
            }
```

- [ ] **Step 4: Run test to verify it passes**

First, the default build path is unchanged — confirm the whole suite still green:

Run: `cargo test -p semantex-core 2>&1 | tail -15`
Expected: PASS — lib suite green; `colbert-plaid` golden test (S1's) unaffected.

Then the new ignored gate (requires the model download):

Run: `cargo test -p semantex-core index::builder::tests::coderank_hnsw_build -- --ignored --nocapture 2>&1 | tail -25`
Expected: PASS — `vectors.bin` written under `dense/coderank-hnsw/`; meta records `coderank-hnsw`, schema 11, model `CodeRankEmbed`.

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/search/hybrid.rs crates/semantex-core/src/index/builder.rs
git commit -m "feat(index,search): wire coderank-hnsw into builder + hybrid open (S2)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: acceptance gate — end-to-end `coderank-hnsw` search + determinism + RSS

The S2 gate: a synthetic repo indexed with `dense_backend = "coderank-hnsw"`, searched through `HybridSearcher`, returns sane, deterministic dense results; brute-force (tiny corpus) is exact; the build stays within the RSS budget. Repo-agnostic (tempdir + inline sources). `#[ignore]` (needs the model). The leaderboard A/B (Recall@10/nDCG@10 ≥ colbert-plaid on CoIR/CSN) is run by the **S0 harness** via `SEMANTEX_DENSE_BACKEND=coderank-hnsw` — this test proves the path works end-to-end so S0 can measure it.

**Files:**
- Create: `crates/semantex-core/tests/coderank_hnsw_backend_test.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/semantex-core/tests/coderank_hnsw_backend_test.rs`:

```rust
//! S2 acceptance gate: the `coderank-hnsw` dense backend indexes a synthetic
//! repo, searches end-to-end via HybridSearcher, and returns deterministic,
//! relevant dense results. Repo-agnostic (tempdir + inline sources). `#[ignore]`
//! because it downloads + runs the CodeRankEmbed ONNX model.
//!
//! Run: cargo test -p semantex-core --test coderank_hnsw_backend_test -- --ignored --nocapture
#![allow(clippy::unwrap_used)]

use anyhow::Result;
use semantex_core::config::SemantexConfig;
use semantex_core::index::builder::IndexBuilder;
use semantex_core::search::SearchQuery;
use semantex_core::search::hybrid::HybridSearcher;
use std::fs;
use std::path::Path;
use tempfile::TempDir;

fn write_repo(root: &Path) {
    fs::write(
        root.join("fib.rs"),
        "/// nth fibonacci number\npub fn fibonacci(n: u32) -> u64 {\n    if n < 2 { n as u64 } else { fibonacci(n-1) + fibonacci(n-2) }\n}\n",
    ).unwrap();
    fs::write(
        root.join("search.py"),
        "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n",
    ).unwrap();
    fs::write(
        root.join("db.go"),
        "package db\nimport \"database/sql\"\nfunc OpenConn(dsn string) (*sql.DB, error) {\n\treturn sql.Open(\"postgres\", dsn)\n}\n",
    ).unwrap();
}

fn cfg() -> SemantexConfig {
    SemantexConfig {
        dense_backend: "coderank-hnsw".to_string(),
        rerank: false,
        ..SemantexConfig::default()
    }
}

fn build_and_open(project: &Path) -> Result<HybridSearcher> {
    let stats = IndexBuilder::new(&cfg())?.build(project)?;
    assert!(stats.chunks_created > 0, "fixture must create chunks");
    HybridSearcher::open(&project.join(".semantex"), &cfg())
}

/// Dense-only search returns the topically-correct chunk first.
#[test]
#[ignore]
fn coderank_hnsw_dense_search_is_relevant() -> Result<()> {
    let tmp = TempDir::new()?;
    let project = tmp.path().join("repo");
    fs::create_dir_all(&project)?;
    write_repo(&project);
    let searcher = build_and_open(&project)?;

    let q = SearchQuery::new("recursive fibonacci function").max_results(3).dense_only();
    let res = searcher.search(&q)?;
    assert!(!res.results.is_empty(), "dense search returned nothing");
    let top = res.results[0]
        .chunk
        .file_path
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    assert_eq!(top, "fib.rs", "fibonacci query should rank fib.rs first, got {top}");
    Ok(())
}

/// Determinism: two opens of the same index give identical dense rankings.
#[test]
#[ignore]
fn coderank_hnsw_dense_is_deterministic() -> Result<()> {
    let tmp = TempDir::new()?;
    let project = tmp.path().join("repo");
    fs::create_dir_all(&project)?;
    write_repo(&project);

    let s1 = build_and_open(&project)?;
    let s2 = HybridSearcher::open(&project.join(".semantex"), &cfg())?;

    let q = SearchQuery::new("open a database connection").max_results(3).dense_only();
    let sig = |s: &HybridSearcher| -> Vec<(String, u32)> {
        s.search(&q)
            .unwrap()
            .results
            .iter()
            .map(|r| {
                (
                    r.chunk.file_path.file_name().unwrap().to_string_lossy().into_owned(),
                    r.chunk.start_line,
                )
            })
            .collect()
    };
    assert_eq!(sig(&s1), sig(&s2), "dense ranking must be identical across opens");
    Ok(())
}
```

- [ ] **Step 2: Run test to verify it fails (then it should pass once the path works)**

Run: `cargo test -p semantex-core --test coderank_hnsw_backend_test -- --ignored --nocapture 2>&1 | tail -40`
Expected initially: if any wiring is off, FAIL with a concrete error (e.g. backend not loaded, empty results, wrong tensor name). This is the integration RED that surfaces any Spike-1 mismatch (tensor names / pooling / prefix).

- [ ] **Step 3: Fix any integration mismatch**

If the top result is wrong or empty, the usual causes (check against Spike-1 record): wrong `OUTPUT_NAME`/`INPUT_IDS` tensor names, wrong `POOL_CLS` choice, or a missing `token_type_ids` input the model requires. Correct `single_vector.rs`'s RECORDED constants (and add a `token_type_ids` zero tensor to the `inputs!` call if Spike 1 listed it as a third input), then re-run. No new test code.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core --test coderank_hnsw_backend_test -- --ignored 2>&1 | tail -20`
Expected: PASS — both `coderank_hnsw_dense_search_is_relevant` and `coderank_hnsw_dense_is_deterministic` pass.

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/tests/coderank_hnsw_backend_test.rs
git commit -m "test(s2): end-to-end coderank-hnsw dense search + determinism gate

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 13: workspace verification + clippy + fmt + S0 smoke

**Files:** none (verification only)

- [ ] **Step 1: Default build (zero LLM deps)**

Run: `cargo build --workspace 2>&1 | tail -15`
Expected: `Finished`. Then confirm no LLM deps leaked:

Run: `cargo tree -p semantex-core 2>/dev/null | grep -c genai`
Expected: `0`.

- [ ] **Step 2: Full default test suite**

Run: `cargo test --workspace 2>&1 | tail -25`
Expected: PASS — all non-ignored tests across the three crates (the `coderank-hnsw` end-to-end + builder model tests are `#[ignore]`d, so they don't run here; the unit tests for store/quant/kernels/preset/parse/config DO run and pass).

- [ ] **Step 3: Ignored gates (model required)**

Run: `cargo test -p semantex-core -- --ignored 2>&1 | tail -30`
Expected: PASS — S2's `coderank_hnsw_*` gates AND S1's `colbert_plaid_*` golden gate both pass (proves S2 did not perturb the colbert-plaid path).

- [ ] **Step 4: Clippy + fmt**

Run: `cargo clippy --all 2>&1 | tail -25`
Expected: no new warnings. (If `Box<dyn DenseBackend>` or the placeholder graph triggers a lint, allow it locally with a one-line `#[allow(...)]` + comment.)

Run: `cargo fmt --all`

- [ ] **Step 5: S0 harness smoke (env-selected backend)**

With a built `semantex` on PATH (or via the harness's binary arg), confirm the env-var selection path the S0 harness uses:

```bash
# Build a tiny throwaway repo, index it, search both backends.
TMP=$(mktemp -d); printf 'pub fn add(a:i32,b:i32)->i32{a+b}\n' > "$TMP/a.rs"
SEMANTEX_DENSE_BACKEND=coderank-hnsw cargo run -q -p semantex-cli -- index "$TMP" 2>&1 | tail -3
SEMANTEX_DENSE_BACKEND=coderank-hnsw cargo run -q -p semantex-cli -- "add two integers" --json --no-content -m 3 --project "$TMP" 2>&1 | tail -5
```

Expected: index completes; the `--json` array is non-empty and well-formed. This is exactly the call shape `benchmarks/relevance/src/semantex_client.py` makes for the D4 A/B.

- [ ] **Step 6: Commit fmt/clippy fixups**

```bash
git add -A
git commit -m "chore(s2): fmt + clippy cleanup for coderank-hnsw backend

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage (§4 S2 + §3 + locked interfaces):**
- ✅ "Spike: export CodeRankEmbed to ONNX int8; record dim/ctx/prefix/pooling" → Task 1 (records into research-notes; every encoder constant references it).
- ✅ "New `embedding/single_vector.rs` encoder (tokenize→forward→pool→L2→optional int8), reusing runtime_manager + CPU-EP-pin" → Tasks 4 (quant) + 6 (encoder); EP/threads/lazy mirror `colbert.rs`; `runtime_manager::ensure_onnxruntime` called defensively.
- ✅ "Spike: select pure-Rust airgap MIT/Apache HNSW crate; fallback vendor oxirs" → Task 2 (records crate + exact API + fallback note).
- ✅ "New `index/hnsw_index.rs` implementing S1 traits: int8 stored + fp32 rescore; M/ef_c/ef_search; presets; brute-force below HNSW_MIN_VECTORS; streaming encode→insert; RSS failsafe" → Tasks 7 (store/params/presets/brute-force/rescore) + 8 (traits, persist, streaming build, ANN/rescore dispatch, RSS check).
- ✅ "Wire into builder behind DenseIndexBuilder; bump schema to 11" → Task 11 (builder + hybrid arms) + Task 10 (schema 10→11).
- ✅ "Dense embeds RAW chunk (no prefix); query gets prefix; do NOT feed BM25 enrichment; NO Matryoshka" → Task 6 `encode_document`/`encode_query` (`prefix_document` = identity; `prefix_query` prepends RECORDED prefix); builder feeds `c.content` (raw), not `bm25_content`; dim is the fixed `EMBEDDING_DIM`.
- ✅ "Config/env: SEMANTEX_DENSE_BACKEND, SEMANTEX_HNSW_EF_SEARCH, SEMANTEX_HNSW_PRESET, SEMANTEX_DENSE_RESCORE_K" → `SEMANTEX_DENSE_BACKEND` is S1's; the other three added in Task 9.
- ✅ "May use S6 SIMD; start scalar, swap later" → Task 5 (scalar `search/simd.rs` with S6 swap-in note + skip-if-present check); Task 7/8 import it.
- ✅ "Acceptance: indexes benchmark repos within RSS budget; on S0 harness coderank-hnsw Recall@10/nDCG@10 ≥ colbert-plaid" → Task 12 (end-to-end + determinism, RSS via `check_rss_or_abort` in the build loop) + Task 13 Step 5 (the exact S0 env-var call shape). The numeric leaderboard A/B is S0's job (env-selected), per its plan — this stream makes the path measurable.
- ✅ LOCKED S1 interfaces: `DenseHit = ScoredChunkId` (5-field) used throughout; `DenseBackend`/`DenseIndexBuilder` signatures matched exactly; `DenseBackendKind::CoderankHnsw` added (name+parse); on-disk `.semantex/dense/coderank-hnsw/`; `positional_chunk_ids() → None` (G5) with documented subset degradation; schema → 11 (G3).

**2. Placeholder scan:** No "TBD"/"add error handling"/"similar to Task N". The only intentional substitution points are the **RECORDED-from-spike** values (Spike 1 model constants; Spike 2 HNSW API) — these are explicitly marked, the spikes are the FIRST tasks, and the brute-force path keeps the crate compiling+correct before the HNSW-API substitution. `PlaceholderGraph` is a real, named, compiling stand-in (not a hand-wave) with a one-line doc explaining its lifecycle.

**3. Type consistency:**
- `DenseHit` / `ScoredChunkId` (5-field, via `ScoredChunkId::new`) everywhere; `DenseBackend`/`DenseIndexBuilder` method shapes identical to S1 and to the `CoderankHnsw*` impls and the `builder.rs`/`hybrid.rs` call sites.
- `SingleVectorEmbedder`: `new`/`for_indexing`/`global`/`encode_document`/`encode_query`/`embedding_dim`/`is_initialized` — same names used in `hnsw_index.rs` and `builder.rs`.
- `Int8VectorStore` fields (`dim/scales/vectors/chunk_ids`) + `push`/`brute_force`/`fp32_rescore`/`clone_for_persist` — consistent across Tasks 7 & 8.
- `HnswParams` (`m/ef_construction/ef_search/rescore_k`) + `preset()` — consistent across Tasks 7, 9 (config→params), 11 (builder/hybrid).
- `CoderankHnswBackend::NAME` and `CoderankHnswIndexBuilder::name()` both `"coderank-hnsw"`, matching `DenseBackendKind::CoderankHnsw.name()` and the on-disk subdir.
- Kernel names `dot_f32`/`cosine_f32`/`dot_i8` consistent between `search/simd.rs` and `hnsw_index.rs`/`single_vector.rs`.
- Config fields `hnsw_ef_search: usize`, `hnsw_preset: String`, `dense_rescore_k: usize` set in Default + load + read in Task 11.

---

## Spec gaps / coordination notes (for the spec owner + S1/S7)

- **G3 — schema bump collision (RESOLVED here).** S1 bumps 9 → 10; S2 bumps 10 → 11. **Hard ordering dependency:** S2 Task 10 assumes S1 already moved the constant to 10 (it replaces S1's `current_schema_version_is_10` test). If S1 and S2 are merged together, do ONE bump to 11 and a single test. If S1 lands first (the intended order per §5), S2's 10→11 is clean. **S1/S2 integrators must confirm the final shipped number is 11, not two competing 10s.**
- **S1-owned file edited by S2 (`dense_backend.rs`).** Task 9 adds the `CoderankHnsw` variant to `DenseBackendKind`. Per §5, S1 lands first; S2 then edits this file. If S1/S2 run truly concurrently, assign `dense_backend.rs` to one team or serialize this one-line enum edit (it's additive and low-conflict, but it IS a shared file). Same for the `match` arms in `hybrid.rs`/`builder.rs` (S1 creates the `match`, S2 adds an arm) — these are the §5 `hybrid.rs` contention points; S2 must rebase on S1's landed shape.
- **G5 — `positional_chunk_ids() = None` (confirmed, documented).** The HNSW backend has no positional doc→chunk array, so the `hybrid.rs` `file_filter` subset prefilter degrades to unfiltered dense + result-merge filtering. S1 already designed the subset path to handle `None` (skip prefilter). S2's `search_with_subset` additionally post-filters a widened top-N if a subset is somehow passed, so correctness holds either way. **No S1 change needed** — just flagged so S1/S7 don't assume a positional array exists for the new backend.
- **`tokenizers` becomes a direct dependency.** It is already in the tree transitively (0.22.2 via fastembed/next-plaid-onnx), so this adds no new vendor surface, but the workspace/`Cargo.lock` owner should be aware. Confirm `features = ["onig"]` builds offline on the Linux CI box (the project's airgap constraint); if `onig` (C lib via `onig_sys`) violates the no-C/C++ rule, switch to the pure-Rust regex feature the spike confirms (`tokenizers` default uses `onig`; there is a `unstable_wasm`/`esaxx_fast`-free path — Spike 1 Step 1 should record which feature set compiles clean).
- **S6 SIMD seam.** Task 5 creates a scalar `search/simd.rs` if S6 hasn't. If S6 owns that path, S2 should consume S6's module and skip Task 5. The kernel signatures (`dot_f32`, `cosine_f32`, `dot_i8`) are the shared contract — S6 must keep them (its parity test guards it). Flag to whoever sequences S2 vs S6.
- **Model hosting (Task 1 Step 4) is an out-of-band artifact step.** The exported `model_int8.onnx` + `tokenizer.json` must be uploaded to a project-controlled HF repo before Task 3's downloader works end-to-end. The plan records the URL; the actual upload is a human/ops action (analogous to how `lightonai/LateOn-Code-edge` is hosted upstream). If the spec owner prefers a different host (self-hosted mirror via `SEMANTEX_*_BASE_URL`), record it and adjust `CODERANK_BASE_URL`.
- **Acceptance numeric gate lives in S0, not here.** §4 S2's "Recall@10/nDCG@10 ≥ colbert-plaid" is measured by the S0 harness driving `SEMANTEX_DENSE_BACKEND`. S2's tests prove the path is functional/deterministic; the cutover decision (D4) is Phase 3 after the S0 A/B. No CLI flag work is needed in S2 (S0 confirmed env-var-only selection).
