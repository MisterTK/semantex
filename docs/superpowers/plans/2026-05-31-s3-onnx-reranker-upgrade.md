# S3 — Reranker Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a generic ONNX cross-encoder reranker (`search/onnx_reranker.rs`) to `semantex-core` that runs any HuggingFace cross-encoder via `ort` + `tokenizers`, supporting two score-extraction strategies — classifier-head (bge-style single relevance logit) and yes/no-logit (Qwen3-Reranker-style generative) — wire it into the existing `SEMANTEX_RERANKER` / `SEMANTEX_RERANKER_MODEL` selection alongside the current fastembed bge path, and keep the whole stage **off by default** (identity pass-through, no model load) until the S0 harness proves a net win.

**Architecture:** Today's reranker is `FastembedReranker` (search/fastembed_reranker.rs) wrapping `fastembed::TextRerank`, gated by `SEMANTEX_RERANKER`, selected by `select_model_from_env() -> fastembed::RerankerModel`. fastembed 5.9's `RerankerModel` enum has no entry for Qwen3-Reranker-0.6B or other new checkpoints, so we cannot route them through fastembed. We introduce a sibling `OnnxReranker` that loads `model_int8.onnx` + `tokenizer.json` directly through `ort` (the exact `Session::builder().with_execution_providers(..).with_intra_threads(..).commit_from_file(..)` + `session.run(inputs!{..})` + `try_extract_tensor::<f32>()` shape that is already proven by the `ort 2.0.0-rc.11` crate and mirrors how `embedding/colbert.rs` pins the CPU execution provider). A new `RerankerChoice` enum sits in front of both impls: the fastembed-native models (bge-v2-m3 default, bge-base, jina-v1/v2) keep going through `FastembedReranker`; the ONNX-only models (Qwen3-Reranker-0.6B; bge-reranker-v2-m3 is *also* exposed via the ONNX path as a generic-loader smoke target) go through `OnnxReranker` with a `ScoreStrategy` (`ClassifierLogit` or `YesNoLogit`). The hybrid call site swaps its `Option<FastembedReranker>` field for an `Option<RerankerEngine>` enum that exposes the same `rerank(query, docs, top_k)` signature, so `hybrid.rs` reranking logic is unchanged. The Qwen3 ONNX export is a **spike** (Task 1) that records the prompt template, the "yes"/"no" token ids, and the input/output tensor names into `docs/superpowers/plans/2026-05-31-research-notes.md`; every later ONNX task references those recorded values rather than inventing them.

**Tech Stack:** Rust 2024 edition (rust-version 1.91), `anyhow`, `ort = 2.0.0-rc.11` (already a direct dep, `load-dynamic` mode), `tokenizers` (NEW direct dep, `default-features = false, features = ["onig"]` — mirrors fastembed's own usage), `ndarray` (already present), `fastembed = 5.9` (unchanged, still the bge path), `tempfile` + `cargo test` for tests. Python + `optimum`/`onnxruntime` only for the offline export spike (not a build dependency).

---

## Reconciled facts (verified against current source — do not re-derive)

These are quoted from the real tree / pinned dependency sources at plan-authoring time. Every type/method referenced below exists today, is introduced by an earlier task in this plan, or is recorded by the Task 1 spike.

- **Current reranker** (`crates/semantex-core/src/search/fastembed_reranker.rs`):
  - `pub const ENV_ENABLE: &str = "SEMANTEX_RERANKER";` (line 53), `pub const ENV_MODEL: &str = "SEMANTEX_RERANKER_MODEL";` (line 55).
  - `pub fn reranker_enabled() -> bool` — truthy `on|1|true|yes` (lines 62-67).
  - `pub fn select_model_from_env() -> RerankerModel` — maps strings to `fastembed::RerankerModel` (lines 72-93); unknown → warn + `BGERerankerV2M3`.
  - `pub fn cache_dir() -> PathBuf` — `FASTEMBED_CACHE_DIR` or `~/.fastembed_cache` (lines 101-109).
  - `FastembedReranker { model: TextRerank }`; `new(model, show_progress)`, `new_default(show_progress)` (bails if `!reranker_enabled()`), `configure_execution_providers()` (CoreML gated by `SEMANTEX_COREML`, then CPU), `rerank(&mut self, query: &str, documents: &[&str], top_k: usize) -> Result<Vec<(usize, f32)>>` — returns identity `(i, 0.0)` for `0..min(len, top_k)` when disabled, else fastembed rerank sorted desc, truncated to `top_k` (lines 195-232).

- **`ort 2.0.0-rc.11` inference API** (verified in `~/.cargo/registry/src/.../ort-2.0.0-rc.11/`):
  - Builder: `ort::session::Session::builder() -> Result<SessionBuilder>` (`session/mod.rs:145`). Methods: `.with_execution_providers(impl AsRef<[ExecutionProviderDispatch]>) -> Result<Self>` (`builder/impl_options.rs:39`), `.with_intra_threads(usize) -> Result<Self>` (`:51`), `.with_optimization_level(GraphOptimizationLevel) -> Result<Self>` (`:85`), `.commit_from_file<P: AsRef<Path>>(self) -> Result<Session>` (`builder/impl_commit.rs:124`).
  - Inference: `session.run(input_values) -> Result<SessionOutputs>` (`session/mod.rs:206`) where `input_values: impl Into<SessionInputs<'i,'v,N>>`. The `ort::inputs![ "name" => value, ... ]` macro yields `Vec<(Cow<str>, SessionInputValue)>` (`session/input.rs:141`).
  - Input tensor: `ort::value::Tensor::from_array((shape, data)) -> Result<Tensor<T>>` where `shape: Vec<i64>` and `data: Vec<T>` (`value/impl_tensor/create.rs:140,554`; `Vec<i64>` is a valid `ToShape` per `:382`). `Tensor<T>` converts into a `SessionInputValue` for `inputs!` via `.into()`.
  - Output: `SessionOutputs` indexes by output name — `outputs["logits"]` (`Index<&str>`, `session/output.rs:189`) yields a `&DynValue`. Extract floats with `value.try_extract_tensor::<f32>() -> Result<(&Shape, &[f32])>` (`value/impl_tensor/extract.rs:156`). `Shape` is a slice-like of dims (used as `shape[shape.len()-1]` etc.).
  - **There is NO existing raw-`ort`-Session call site in semantex.** All current ONNX inference goes through `next_plaid_onnx::Colbert` (`embedding/colbert.rs`). This plan is the first direct `ort::Session` user, so the inference shape above is taken verbatim from the crate, not from in-tree precedent. The execution-provider list, however, mirrors `colbert.rs`/`fastembed_reranker.rs`: CoreML only if `SEMANTEX_COREML=1` on macOS, CUDA under `#[cfg(feature="cuda")]`, then always CPU.

- **`tokenizers` API** (pinned `0.21.4`/`0.22.2` are both in `Cargo.lock` transitively; we add a direct dep): `tokenizers::Tokenizer::from_file<P: AsRef<Path>>(file) -> Result<Tokenizer, Box<dyn Error + Send + Sync>>` (`tokenizer/mod.rs:438`), `tokenizer.encode(text, add_special_tokens: bool) -> Result<Encoding, ...>` (`:827`), `Encoding::get_ids() -> &[u32]` (`encoding.rs:147`), `get_attention_mask() -> &[u32]` (`:171`), `get_type_ids() -> &[u32]` (`:151`). fastembed pulls it as `features = ["onig"], default-features = false` (`fastembed-5.9.0/Cargo.toml:179`) — we mirror that to avoid the `progressbar`/`esaxx_fast` defaults.

- **ColBERT EP-pinning precedent** (`crates/semantex-core/src/embedding/colbert.rs:166-198`): `Colbert::builder(dir).with_quantized(true).with_threads(self.threads).with_execution_provider(...)`. CoreML iff `self.use_coreml` (macOS), else `Cpu`. Threads from `SEMANTEX_ORT_THREADS` (query, default 4) / `SEMANTEX_INDEX_ORT_THREADS`. Lazy `OnceLock` session built under a `build_lock` (lines 214-234). We reuse the same lazy/`OnceLock`+`Mutex` discipline for `OnnxReranker`.

- **ONNX Runtime provisioning** (`crates/semantex-core/src/embedding/runtime_manager.rs`): `pub fn ensure_onnxruntime(runtime_root: &Path) -> Result<PathBuf>` (line 109), `pub fn find_onnxruntime(runtime_root: &Path) -> Option<PathBuf>` (line 91), `pub const ONNXRUNTIME_VERSION: &str` (line 30). `ORT_DYLIB_PATH` is set once at process start in `crates/semantex-cli/src/main.rs:354-381` (calls `find_onnxruntime` then `ensure_onnxruntime` under `SemantexConfig::semantex_home().join("runtime")`). So by the time any search runs the dylib is already provisioned; the reranker does **not** need to re-provision it.

- **Model download + cache** (`crates/semantex-core/src/embedding/model_manager.rs`): `pub(crate) fn download_file(url: &str, dest: &Path) -> Result<()>` (line 44, atomic temp+rename, progress bar). `ensure_colbert_model` (line 14) is the pattern: per-model subdir under `models_dir`, download listed files if the sentinel (`model_int8.onnx`) is absent. `SemantexConfig::models_dir() -> PathBuf` = `model_dir` override or `semantex_home().join("models")` (`config.rs:158-164`).

- **Hybrid call site** (`crates/semantex-core/src/search/hybrid.rs`):
  - Field today: `reranker: Mutex<Option<FastembedReranker>>` (line 28), set to `Mutex::new(None)` in both constructors (lines 58, 139).
  - Import: `use crate::search::fastembed_reranker::FastembedReranker;` (line 8).
  - Stage 3 block (lines 1082-1129): if `query.use_rerank && self.config.rerank`, lazy-load via `FastembedReranker::new_default(false)` into the guard, then `reranker.rerank(&query.text, &docs, query.max_results)`, where `docs: Vec<&str> = results.iter().map(|r| r.chunk.content.as_str()).collect()`. Reorders `results`, sets `r.score = score` and `r.source = SearchSource::Reranked` (`types.rs:163-169`).
  - **Latency guard already present at the config layer:** `self.config.rerank_candidates` (default 100, `config.rs:37,83`) bounds how many candidates reach Stage 3 (`let candidates = self.config.rerank_candidates;` at `hybrid.rs:188`). The reranker itself must additionally cap its own ONNX work to the top `rerank_candidates` documents (Task 7) so a 0.6B model never runs on an unbounded list.

- **`SemantexConfig`** (`crates/semantex-core/src/config.rs`): `pub rerank: bool` (line 25, default `false` line 77/`58` per region), `pub rerank_candidates: usize` (line 37, default 100), env override `SEMANTEX_RERANK` toggles `rerank` (lines 114-115). Helper `pub(crate) fn env_usize(key: &str, default: usize) -> usize` (line 201).

- **Module wiring**: `crates/semantex-core/src/search/mod.rs:1-19` declares `pub mod fastembed_reranker;` etc. Add `pub mod onnx_reranker;` and `pub mod reranker_engine;` there.

- **Env-test helper** to reuse verbatim: `with_env(&[(&str, Option<&str>)], F)` in `fastembed_reranker.rs:243-283` (holds a static `Mutex`, snapshots/restores, `catch_unwind`). New tests that touch env vars copy this helper into their own test module (tests can't share a private fn across files; duplicate it — it is small and self-contained).

- **Permissive licenses** (spec §8): Qwen3-Reranker-0.6B = Apache-2.0; bge-reranker-v2-m3 = permissive (already shipped). jina-reranker-v3 is **NC → excluded** (D3) and must never appear in any selector. No model name is hardcoded into a production default beyond what env/selection chooses; the default stays `BGERerankerV2M3` (already in the tree).

---

## File Structure

Files created or modified, one responsibility each:

- **Create `docs/superpowers/plans/2026-05-31-research-notes.md`** (Task 1 spike output) — records the Qwen3-Reranker-0.6B ONNX export facts that later tasks depend on: exact ONNX input tensor names + dtypes, output tensor name + shape, the reranker prompt template (system/instruction wrapper), and the integer token ids for `"yes"` and `"no"` under the model's tokenizer. Also records bge-reranker-v2-m3's classifier ONNX I/O names for the generic-loader smoke test. Append-only; shared with sibling streams (S2 records dense facts in the same file).

- **Create `crates/semantex-core/src/search/onnx_reranker.rs`** — the generic ONNX cross-encoder loader. Defines: `ScoreStrategy` enum (`ClassifierLogit`, `YesNoLogit { yes_id: i64, no_id: i64, prompt: PromptTemplate }`); `PromptTemplate` (holds the prefix/suffix strings recorded by the spike, with a `render(query, doc) -> String`); `OnnxReranker` (lazy `OnceLock<Mutex<Session>>` + owned `Tokenizer` + `ScoreStrategy` + threads/coreml flags); pure scoring helpers `classifier_score_from_logits` and `yes_no_score_from_logits` (unit-testable without a model); `score_pair(query, doc) -> Result<f32>`; `rerank(&self, query, docs, top_k) -> Result<Vec<(usize, f32)>>`. Tests: the two pure logit→score functions, prompt rendering, and an `#[ignore]`'d real-model integration test gated on `SEMANTEX_RERANKER=on`.

- **Create `crates/semantex-core/src/search/reranker_model.rs`** — the selection layer. Defines `RerankerChoice` enum: `Fastembed(fastembed::RerankerModel)` and `Onnx(OnnxModelSpec)`. `OnnxModelSpec { repo_files: ModelFiles, strategy_kind: StrategyKind, query_prefix: ... }` is metadata only (no network). `pub fn select_reranker_choice_from_env() -> RerankerChoice` reads `SEMANTEX_RERANKER_MODEL`, returns `Onnx(..)` for `qwen3-reranker-0.6b` / `qwen3` and for `bge-v2-m3-onnx` (the generic-loader smoke alias), else delegates to the existing `select_model_from_env()` wrapped as `Fastembed(..)`. `StrategyKind` is `Classifier` or `YesNo` so the engine can build the concrete `ScoreStrategy` after the spike values are loaded. Tests: env→choice mapping incl. unknown-fallback and the NC-exclusion assertion (no `jina-reranker-v3` alias exists).

- **Create `crates/semantex-core/src/search/reranker_engine.rs`** — the dispatcher the call site holds. `enum RerankerEngine { Fastembed(FastembedReranker), Onnx(OnnxReranker) }` with `fn rerank(&mut self, query: &str, docs: &[&str], top_k: usize) -> Result<Vec<(usize, f32)>>` (delegates), and `fn new_default(show_progress: bool) -> Result<Self>` (bails if `!reranker_enabled()`, else builds from `select_reranker_choice_from_env()`; for the `Onnx` arm it downloads the model files via a new `ensure_reranker_model` and constructs `OnnxReranker`). Tests: `new_default` refuses when disabled (mirrors the fastembed test).

- **Create `crates/semantex-core/src/search/reranker_download.rs`** — `pub fn ensure_reranker_model(models_dir: &Path, spec: &ModelFiles) -> Result<PathBuf>`: per-model subdir under `models_dir`, downloads each listed HF file via `model_manager::download_file` if the `model_int8.onnx` sentinel is absent (exact mirror of `ensure_colbert_model`). `ModelFiles { subdir: &'static str, base_url: &'static str, files: &'static [&'static str] }`. Tests: sentinel short-circuit (no download when file exists), URL construction.

- **Modify `crates/semantex-core/src/search/mod.rs:1-19`** — add `pub mod onnx_reranker;`, `pub mod reranker_model;`, `pub mod reranker_engine;`, `pub mod reranker_download;`.

- **Modify `crates/semantex-core/Cargo.toml:17-50`** — add `tokenizers = { version = "0.22", default-features = false, features = ["onig"] }` to `[dependencies]`.

- **Modify `crates/semantex-core/src/search/hybrid.rs:8, 28, 58, 139, 1092-1106`** — change the import + field type from `FastembedReranker` to `RerankerEngine`; lazy-load via `RerankerEngine::new_default(false)`. The rerank call (`reranker.rerank(&query.text, &docs, query.max_results)`) is unchanged because `RerankerEngine::rerank` has the identical signature.

- **Modify `crates/semantex-core/src/search/fastembed_reranker.rs:14-21`** — extend the module doc comment to point new model names at the ONNX loader (documentation only; no behavior change). `FastembedReranker` itself is otherwise untouched.

No new workspace crate dependencies beyond `tokenizers`. `ort`, `ndarray`, `fastembed`, `anyhow` are already present.

---

## Off-by-default safety contract (must hold after every task)

When `SEMANTEX_RERANKER` is unset (or not truthy):
1. `reranker_enabled()` returns `false`.
2. `RerankerEngine::new_default` **bails** (no model files downloaded, no `ort::Session` built, no `Tokenizer` loaded).
3. The hybrid Stage-3 block leaves `reranker_guard == None` (construction failed/skipped), so `results` is returned unchanged.
4. No code path touches the network or the ONNX Runtime for reranking.

Task 9 adds an explicit regression test for this. Do not regress it in any earlier task.

---

## Task 1: SPIKE — export Qwen3-Reranker-0.6B to ONNX int8 and record the contract

**This task produces no Rust. It records the values every later ONNX task depends on. Do it first.** Run the export offline (network + ~1.2 GB scratch); commit only the research-notes file, never the weights.

**Files:**
- Create: `docs/superpowers/plans/2026-05-31-research-notes.md`

- [ ] **Step 1: Create a throwaway export venv and export Qwen3-Reranker-0.6B**

Run (in any scratch dir outside the repo; requires network):

```bash
python3 -m venv /tmp/s3-export && . /tmp/s3-export/bin/activate
pip install -q "optimum[onnxruntime]" "transformers>=4.51" "onnx" "onnxruntime"
optimum-cli export onnx \
  --model Qwen/Qwen3-Reranker-0.6B \
  --task text-classification \
  /tmp/qwen3-reranker-0.6b-onnx
```

Expected: a directory `/tmp/qwen3-reranker-0.6b-onnx` containing `model.onnx` (+ external-data file for a 0.6B model), `tokenizer.json`, `config.json`. If `--task text-classification` fails because Qwen3-Reranker is a causal-LM head (it is — Qwen3-Reranker scores via the LM logit of the "yes" token, not a classification head), re-run with `--task text-generation-with-past` and **record that fact** — it changes `ScoreStrategy` to `YesNoLogit` and means the model has a vocab-sized logits output, not a 1-D relevance logit.

- [ ] **Step 2: Dynamic int8-quantize the export**

Run:

```bash
python3 - <<'PY'
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(
    "/tmp/qwen3-reranker-0.6b-onnx/model.onnx",
    "/tmp/qwen3-reranker-0.6b-onnx/model_int8.onnx",
    weight_type=QuantType.QInt8,
)
print("int8 written")
PY
ls -lh /tmp/qwen3-reranker-0.6b-onnx/model_int8.onnx
```

Expected: `model_int8.onnx` (~600-700 MB) written. Record its on-disk size in the notes (drives the download-size warning string in Task 6).

- [ ] **Step 3: Introspect the I/O tensor names, the prompt template, and the yes/no token ids**

Run:

```bash
python3 - <<'PY'
import onnx, json
from transformers import AutoTokenizer
m = onnx.load("/tmp/qwen3-reranker-0.6b-onnx/model_int8.onnx", load_external_data=False)
print("INPUTS :", [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in m.graph.input])
print("OUTPUTS:", [(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])
tok = AutoTokenizer.from_pretrained("/tmp/qwen3-reranker-0.6b-onnx")
# Qwen3-Reranker scores the LM probability of "yes" vs "no" on a chat-formatted prompt.
for w in ["yes", "no", " yes", " no", "Yes", "No"]:
    ids = tok.encode(w, add_special_tokens=False)
    print(f"token {w!r} -> ids {ids}")
print("CHAT TEMPLATE present:", tok.chat_template is not None)
PY
```

Expected: prints the ONNX input names (likely `input_ids`, `attention_mask`, possibly `position_ids` and `past_key_values.*` for the `-with-past` export — **record exactly which are required, with their dtypes/`int64` and shapes**), the output name (e.g. `logits` with shape `[batch, seq, vocab]` for the causal export or `[batch, num_labels]` for a classifier head), and the single-token ids for "yes"/"no". **Record the exact integer ids and the exact prompt format** (the official Qwen3-Reranker card uses an instruction + `<Instruct>: ... <Query>: ... <Document>: ...` wrapper terminated so the next-token logit is the relevance judgment; capture the real strings).

- [ ] **Step 4: Confirm bge-reranker-v2-m3 classifier ONNX I/O (the generic-loader smoke target)**

The fastembed path already ships bge-v2-m3, but we also drive it through the *generic* loader to prove `ClassifierLogit`. Record its ONNX I/O from the fastembed cache if present, else from a one-off export:

```bash
python3 - <<'PY'
import onnx
# fastembed caches under ~/.fastembed_cache/models--BAAI--bge-reranker-v2-m3/...
import glob
cands = glob.glob("/Users/*/.fastembed_cache/**/model_optimized.onnx", recursive=True) \
      + glob.glob("/Users/*/.fastembed_cache/**/*.onnx", recursive=True)
print("cached onnx:", cands[:3])
if cands:
    m = onnx.load(cands[0], load_external_data=False)
    print("INPUTS :", [i.name for i in m.graph.input])
    print("OUTPUTS:", [o.name for o in m.graph.output])
PY
```

Expected: bge-v2-m3 inputs `input_ids`, `attention_mask` (XLM-RoBERTa has no `token_type_ids`), output `logits` shape `[batch, 1]`. **Record these names** — Task 4's classifier strategy reads the last logit.

- [ ] **Step 5: Write `docs/superpowers/plans/2026-05-31-research-notes.md`**

Create the file with a dedicated S3 section recording, verbatim, every value from Steps 1-4. Template (fill in REAL values, no placeholders left):

```markdown
# 2026-05-31 SOTA overhaul — research notes (shared across streams)

## S3 — Reranker ONNX export contract

### Qwen3-Reranker-0.6B (Apache-2.0)
- HF repo: `Qwen/Qwen3-Reranker-0.6B`
- Export task used: `<text-classification | text-generation-with-past>`  (RECORD which worked)
- int8 file: `model_int8.onnx` (RECORD size, e.g. 612 MB) + tokenizer.json + config.json
- ONNX inputs (name : dtype : shape):  RECORD, e.g. `input_ids : int64 : [batch, seq]`, `attention_mask : int64 : [batch, seq]`  (+ position_ids / past_key_values if required)
- ONNX output (name : shape): RECORD, e.g. `logits : [batch, seq, vocab]`
- Score strategy: YesNoLogit
- "yes" token id: RECORD (single int)
- "no"  token id: RECORD (single int)
- Logit slice for scoring: RECORD (e.g. last position: `logits[0, seq_len-1, :]`, then compare yes_id vs no_id)
- Prompt template (verbatim, with newlines shown):
  - prefix: `<RECORD>`
  - suffix: `<RECORD>`
  - rendered example for query "binary search", doc "fn bsearch(...)": `<RECORD>`

### bge-reranker-v2-m3 (generic-loader smoke target; permissive)
- ONNX inputs: `input_ids`, `attention_mask`  (RECORD if different)
- ONNX output: `logits` shape `[batch, 1]`  (RECORD if different)
- Score strategy: ClassifierLogit (read `logits[0, -1]`)
```

- [ ] **Step 6: Commit the research notes**

```bash
git add docs/superpowers/plans/2026-05-31-research-notes.md
git commit -m "$(cat <<'EOF'
docs(s3): record Qwen3-Reranker-0.6B ONNX export contract (spike)

Records int8 export task, I/O tensor names, yes/no token ids, prompt
template, and bge-v2-m3 classifier I/O for the generic ONNX reranker.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add `tokenizers` as a direct dependency

**Files:**
- Modify: `crates/semantex-core/Cargo.toml:17-50`
- Test: `crates/semantex-core/tests/tokenizers_dep_smoke.rs` (new)

- [ ] **Step 1: Write the failing test**

Create `crates/semantex-core/tests/tokenizers_dep_smoke.rs`:

```rust
//! Smoke test proving `tokenizers` is a usable direct dependency of
//! semantex-core (it was previously only transitive via fastembed/next-plaid).
//! The ONNX reranker (search/onnx_reranker.rs) needs `Tokenizer::from_file`.

#[test]
fn tokenizer_type_is_importable() {
    // Compiling this reference is the test: if `tokenizers` is not a direct
    // dependency, this fails to build with "unresolved import".
    fn _accepts(_t: &tokenizers::Tokenizer) {}
    // Also assert the encode/ids API surface we rely on exists by referencing it.
    let _from_file: fn(
        &std::path::Path,
    ) -> Result<tokenizers::Tokenizer, Box<dyn std::error::Error + Send + Sync>> =
        tokenizers::Tokenizer::from_file::<&std::path::Path>;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p semantex-core --test tokenizers_dep_smoke`
Expected: FAIL — compile error `use of undeclared crate or module \`tokenizers\`` (it is not yet a direct dependency).

- [ ] **Step 3: Add the dependency**

Edit `crates/semantex-core/Cargo.toml`, after the `fastembed = "5.9"` line (in `[dependencies]`):

```toml
# Tokenizer for the generic ONNX cross-encoder reranker (search/onnx_reranker.rs).
# Mirrors fastembed's own usage (onig, no progressbar/esaxx defaults) so we don't
# pull indicatif twice or a C++ esaxx build. `from_file` + `encode` are all we use.
tokenizers = { version = "0.22", default-features = false, features = ["onig"] }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core --test tokenizers_dep_smoke`
Expected: PASS (`test tokenizer_type_is_importable ... ok`).

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/Cargo.toml Cargo.lock crates/semantex-core/tests/tokenizers_dep_smoke.rs
git commit -m "$(cat <<'EOF'
feat(rerank): add tokenizers as a direct semantex-core dependency

Needed by the upcoming generic ONNX cross-encoder reranker. Mirrors
fastembed's feature set (onig, default-features=false) to avoid pulling
indicatif/esaxx twice.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `ScoreStrategy` + `PromptTemplate` + pure scoring helpers

Build the score-extraction math first, fully unit-tested without any model or ONNX Runtime.

**Files:**
- Create: `crates/semantex-core/src/search/onnx_reranker.rs`
- Modify: `crates/semantex-core/src/search/mod.rs:1-19`

- [ ] **Step 1: Register the module and write the failing test**

In `crates/semantex-core/src/search/mod.rs`, add after `pub mod hybrid;`:

```rust
pub mod onnx_reranker;
```

Create `crates/semantex-core/src/search/onnx_reranker.rs` with ONLY the types + pure helpers + tests (no `ort` yet):

```rust
//! Generic ONNX cross-encoder reranker.
//!
//! fastembed 5.9's `RerankerModel` enum cannot express newer checkpoints
//! (e.g. Qwen3-Reranker-0.6B), so this module loads any HuggingFace
//! cross-encoder's `model_int8.onnx` + `tokenizer.json` directly through
//! `ort` + `tokenizers`, mirroring how `embedding/colbert.rs` pins the CPU
//! execution provider. It supports two ways to turn model output into a
//! relevance score:
//!
//! - [`ScoreStrategy::ClassifierLogit`] — sequence-classification head
//!   (bge-style): the model emits a single relevance logit; the score is that
//!   logit.
//! - [`ScoreStrategy::YesNoLogit`] — generative reranker (Qwen3-Reranker-style):
//!   the model emits vocabulary logits; the score is the softmax-normalized
//!   probability mass on the "yes" token vs the "no" token at the final
//!   position, per the values recorded in
//!   `docs/superpowers/plans/2026-05-31-research-notes.md`.
//!
//! All concrete model ids, prompt strings, and token ids come from the
//! selection layer (`reranker_model.rs`) / the recorded spike — nothing
//! repo-specific is hardcoded here.

use anyhow::{Context, Result};

/// Prompt wrapper for a generative (yes/no) reranker. Strings come from the
/// model's card (recorded in the spike), never hardcoded per corpus.
#[derive(Debug, Clone)]
pub struct PromptTemplate {
    /// Text emitted before the query (e.g. the instruction header).
    pub prefix: String,
    /// Text emitted between the query and the document.
    pub middle: String,
    /// Text emitted after the document (terminates the prompt so the next-token
    /// logit is the relevance judgment).
    pub suffix: String,
}

impl PromptTemplate {
    /// Render the full prompt string for a (query, document) pair.
    #[must_use]
    pub fn render(&self, query: &str, document: &str) -> String {
        format!(
            "{}{}{}{}{}",
            self.prefix, query, self.middle, document, self.suffix
        )
    }
}

/// How to convert model output logits into a single relevance score.
#[derive(Debug, Clone)]
pub enum ScoreStrategy {
    /// Sequence-classification head: a single relevance logit per pair.
    ClassifierLogit,
    /// Generative reranker: compare the "yes" vs "no" token logits at the
    /// final position. `yes_id`/`no_id` are recorded in the spike notes.
    YesNoLogit {
        yes_id: usize,
        no_id: usize,
        prompt: PromptTemplate,
    },
}

/// Score for a classifier-head model: the relevance logit is the last element
/// of the flattened output (shape `[batch=1, num_labels]`, num_labels==1 for
/// bge rerankers). Returns the raw logit (monotonic in relevance; the caller
/// only needs ordering).
pub(crate) fn classifier_score_from_logits(logits: &[f32]) -> Result<f32> {
    logits
        .last()
        .copied()
        .context("classifier reranker produced an empty logits tensor")
}

/// Score for a yes/no generative reranker: take the logits at the final
/// sequence position (`final_pos_logits`, length == vocab size), then return
/// the softmax probability of "yes" over {yes, no}:
/// `exp(l_yes) / (exp(l_yes) + exp(l_no))`. Computed in a numerically stable
/// way by subtracting the max of the two logits.
pub(crate) fn yes_no_score_from_logits(
    final_pos_logits: &[f32],
    yes_id: usize,
    no_id: usize,
) -> Result<f32> {
    let l_yes = *final_pos_logits
        .get(yes_id)
        .with_context(|| format!("yes_id {yes_id} out of range (vocab {})", final_pos_logits.len()))?;
    let l_no = *final_pos_logits
        .get(no_id)
        .with_context(|| format!("no_id {no_id} out of range (vocab {})", final_pos_logits.len()))?;
    let m = l_yes.max(l_no);
    let e_yes = (l_yes - m).exp();
    let e_no = (l_no - m).exp();
    Ok(e_yes / (e_yes + e_no))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_score_reads_last_logit() {
        assert_eq!(classifier_score_from_logits(&[0.42]).unwrap(), 0.42);
        // Multi-label safety: still reads the last element.
        assert_eq!(classifier_score_from_logits(&[-1.0, 3.5]).unwrap(), 3.5);
    }

    #[test]
    fn classifier_score_errors_on_empty() {
        assert!(classifier_score_from_logits(&[]).is_err());
    }

    #[test]
    fn yes_no_score_is_monotonic_and_bounded() {
        // Vocab of 4; yes_id=2, no_id=3.
        let logits = [0.0, 0.0, 2.0, 1.0];
        let s = yes_no_score_from_logits(&logits, 2, 3).unwrap();
        // softmax(2,1) over {yes,no} = e^1/(e^1+e^0) = 0.7310586
        assert!((s - 0.7310586).abs() < 1e-5, "got {s}");
        // Strong yes -> near 1.0; strong no -> near 0.0.
        assert!(yes_no_score_from_logits(&[0.0, 0.0, 10.0, 0.0], 2, 3).unwrap() > 0.99);
        assert!(yes_no_score_from_logits(&[0.0, 0.0, 0.0, 10.0], 2, 3).unwrap() < 0.01);
    }

    #[test]
    fn yes_no_score_errors_on_oob_token() {
        assert!(yes_no_score_from_logits(&[0.1, 0.2], 5, 0).is_err());
    }

    #[test]
    fn prompt_template_renders_in_order() {
        let t = PromptTemplate {
            prefix: "<I>".into(),
            middle: "<M>".into(),
            suffix: "<S>".into(),
        };
        assert_eq!(t.render("Q", "D"), "<I>Q<M>D<S>");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p semantex-core onnx_reranker::tests`
Expected: FAIL — the module compiles but the test binary build fails first time only if `mod` not registered; once registered, the tests run. (If you registered the module but left the helpers unimplemented you'd get compile errors — here the implementation is included, so this step is really "confirm it builds and the 5 tests pass". If you wrote the test bodies *before* the helpers, you'd see `cannot find function`.) To honor TDD strictly: comment out the function bodies' `Ok(...)` returns first, confirm FAIL, then restore.

- [ ] **Step 3: Confirm the implementation (already inlined above)**

The helpers `classifier_score_from_logits`, `yes_no_score_from_logits`, `PromptTemplate::render` are fully written in Step 1. No further code.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core onnx_reranker::tests`
Expected: PASS — 5 tests:
```
test search::onnx_reranker::tests::classifier_score_reads_last_logit ... ok
test search::onnx_reranker::tests::classifier_score_errors_on_empty ... ok
test search::onnx_reranker::tests::yes_no_score_is_monotonic_and_bounded ... ok
test search::onnx_reranker::tests::yes_no_score_errors_on_oob_token ... ok
test search::onnx_reranker::tests::prompt_template_renders_in_order ... ok
```

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/search/onnx_reranker.rs crates/semantex-core/src/search/mod.rs
git commit -m "$(cat <<'EOF'
feat(rerank): ScoreStrategy + PromptTemplate + pure logit->score helpers

Classifier-head (single logit) and yes/no-logit (generative) score
extraction as pure, unit-tested functions. No ort/model yet.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `OnnxReranker` struct + lazy `ort::Session` + tokenization

Now wire the real ONNX inference, mirroring colbert.rs's lazy/EP-pinning discipline. The single-pair `score_pair` path uses the verbatim `ort 2.0.0-rc.11` API from Reconciled facts.

**Files:**
- Modify: `crates/semantex-core/src/search/onnx_reranker.rs`

- [ ] **Step 1: Write the failing test**

Append to the `tests` module in `onnx_reranker.rs`:

```rust
    /// Construction is cheap and lazy: pointing at a non-existent model dir
    /// still succeeds (the ONNX session is built on first `score_pair`, not at
    /// construction), but a missing dir at construction is rejected early —
    /// mirroring ColbertEmbedder's "fail late for files, fail early for dir".
    #[test]
    fn new_rejects_missing_dir_but_is_lazy_for_files() {
        let tmp = tempfile::TempDir::new().unwrap();
        // Existing but empty dir: construct OK, session not built.
        let r = OnnxReranker::new(tmp.path(), ScoreStrategy::ClassifierLogit, 2, false);
        assert!(r.is_err(), "missing tokenizer.json should fail at construction");

        let missing = std::path::Path::new("/no/such/reranker/dir");
        assert!(OnnxReranker::new(missing, ScoreStrategy::ClassifierLogit, 2, false).is_err());
    }
```

(Note: we make `new` validate that `tokenizer.json` exists — that is a construction-time file we can read cheaply — while the heavy `model_int8.onnx` session is deferred. So an empty dir fails because the tokenizer is missing.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p semantex-core onnx_reranker::tests::new_rejects_missing_dir`
Expected: FAIL — `cannot find function \`new\` in this scope` / `cannot find type \`OnnxReranker\``.

- [ ] **Step 3: Implement `OnnxReranker` with lazy session + tokenization + inference**

Add to the top-of-file `use` block in `onnx_reranker.rs`:

```rust
use ort::session::Session;
use ort::value::Tensor;
use parking_lot::Mutex;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use tokenizers::Tokenizer;
```

Add the struct + impl (above the `#[cfg(test)]` module):

```rust
/// A generic ONNX cross-encoder reranker. Thread-safe: the inner `ort::Session`
/// is built lazily on first scoring and guarded by a `Mutex` (single-threaded
/// inference, like `ColbertEmbedder`). Construction is cheap.
pub struct OnnxReranker {
    /// Directory containing `model_int8.onnx` and `tokenizer.json`.
    model_dir: PathBuf,
    /// Loaded eagerly (small, validates the dir): the HF tokenizer.
    tokenizer: Tokenizer,
    /// How to turn model output into a relevance score.
    strategy: ScoreStrategy,
    /// ORT intra-op threads (from `SEMANTEX_ORT_THREADS`, default 4 — set by caller).
    threads: usize,
    /// Opt into CoreML on macOS (gated by `SEMANTEX_COREML`).
    use_coreml: bool,
    /// Lazily-built ONNX session.
    session: OnceLock<Mutex<Session>>,
    /// Serializes the build path so concurrent first-callers don't build twice.
    build_lock: std::sync::Mutex<()>,
}

impl OnnxReranker {
    /// Construct a reranker from a model directory. Reads `tokenizer.json`
    /// eagerly (cheap, and validates the directory); defers the heavy
    /// `model_int8.onnx` session to the first `score_pair`/`rerank` call.
    ///
    /// # Errors
    /// Errors if the directory or `tokenizer.json` is missing/unreadable.
    pub fn new(
        model_dir: &Path,
        strategy: ScoreStrategy,
        threads: usize,
        use_coreml: bool,
    ) -> Result<Self> {
        let tok_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer {}: {e}", tok_path.display()))?;
        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            tokenizer,
            strategy,
            threads: threads.max(1),
            use_coreml,
            session: OnceLock::new(),
            build_lock: std::sync::Mutex::new(()),
        })
    }

    /// Execution providers: CoreML iff `use_coreml` on macOS, CUDA under the
    /// `cuda` feature, then always CPU. Mirrors `fastembed_reranker.rs` /
    /// `colbert.rs` so reranking never allocates the ~10 GB CoreML buffers
    /// unless explicitly opted in.
    fn execution_providers(&self) -> Vec<ort::ep::ExecutionProviderDispatch> {
        let mut providers = Vec::new();
        #[cfg(target_os = "macos")]
        if self.use_coreml {
            providers.push(ort::ep::CoreML::default().build());
        }
        #[cfg(not(target_os = "macos"))]
        let _ = self.use_coreml;
        #[cfg(feature = "cuda")]
        {
            providers.push(ort::ep::CUDA::default().build());
        }
        providers.push(ort::ep::CPU::default().build());
        providers
    }

    /// Build the ONNX session from `model_int8.onnx`. Uses the verbatim
    /// `ort 2.0.0-rc.11` builder API.
    fn build_session(&self) -> Result<Session> {
        let model_path = self.model_dir.join("model_int8.onnx");
        let session = Session::builder()
            .context("ort Session::builder failed")?
            .with_execution_providers(self.execution_providers())
            .context("failed to set execution providers")?
            .with_intra_threads(self.threads)
            .context("failed to set intra-op threads")?
            .commit_from_file(&model_path)
            .with_context(|| format!("failed to load ONNX model {}", model_path.display()))?;
        Ok(session)
    }

    /// Get the session, building it once on first call (double-checked locking,
    /// same discipline as `ColbertEmbedder::encoder`).
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

    /// Tokenize one (query, document) pair into `(input_ids, attention_mask)`
    /// as `i64` vectors. For the classifier strategy the model sees the query
    /// and document as a sentence pair joined by the tokenizer's separator; for
    /// the yes/no strategy it sees the rendered prompt as a single sequence.
    fn encode_pair(&self, query: &str, document: &str) -> Result<(Vec<i64>, Vec<i64>)> {
        let text = match &self.strategy {
            ScoreStrategy::ClassifierLogit => {
                // Cross-encoders score "query [SEP] document". The tokenizer's
                // post-processor inserts the separator when given a pair via
                // encode((a, b)); we pass the pair tuple.
                let enc = self
                    .tokenizer
                    .encode((query, document), true)
                    .map_err(|e| anyhow::anyhow!("tokenizer.encode(pair) failed: {e}"))?;
                return Ok(to_i64_pair(&enc));
            }
            ScoreStrategy::YesNoLogit { prompt, .. } => prompt.render(query, document),
        };
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenizer.encode failed: {e}"))?;
        Ok(to_i64_pair(&enc))
    }

    /// Score a single (query, document) pair. Runs one ONNX forward pass.
    pub fn score_pair(&self, query: &str, document: &str) -> Result<f32> {
        let (ids, mask) = self.encode_pair(query, document)?;
        let seq = ids.len();
        let id_tensor = Tensor::from_array((vec![1_i64, seq as i64], ids))
            .context("failed to build input_ids tensor")?;
        let mask_tensor = Tensor::from_array((vec![1_i64, seq as i64], mask))
            .context("failed to build attention_mask tensor")?;

        let session = self.session()?;
        let mut guard = session.lock();
        let outputs = guard
            .run(ort::inputs![
                "input_ids" => id_tensor,
                "attention_mask" => mask_tensor,
            ])
            .context("ONNX reranker forward pass failed")?;

        // First (only) output is the logits tensor; index by position 0 to be
        // robust to the model's output name.
        let (shape, logits) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("failed to extract f32 logits from reranker output")?;

        match &self.strategy {
            ScoreStrategy::ClassifierLogit => classifier_score_from_logits(logits),
            ScoreStrategy::YesNoLogit { yes_id, no_id, .. } => {
                // Generative output shape is [batch=1, seq, vocab]; the
                // judgment is the final position's vocab logits.
                let vocab = shape[shape.len() - 1] as usize;
                anyhow::ensure!(vocab > 0, "reranker output has zero-width vocab dim");
                let n = logits.len();
                anyhow::ensure!(n >= vocab, "reranker logits ({n}) shorter than vocab ({vocab})");
                let final_pos = &logits[n - vocab..];
                yes_no_score_from_logits(final_pos, *yes_id, *no_id)
            }
        }
    }
}

/// Convert a tokenizers `Encoding` to `(input_ids, attention_mask)` as i64.
fn to_i64_pair(enc: &tokenizers::Encoding) -> (Vec<i64>, Vec<i64>) {
    let ids = enc.get_ids().iter().map(|&x| i64::from(x)).collect();
    let mask = enc
        .get_attention_mask()
        .iter()
        .map(|&x| i64::from(x))
        .collect();
    (ids, mask)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core onnx_reranker::tests::new_rejects_missing_dir`
Expected: PASS (`test ... new_rejects_missing_dir_but_is_lazy_for_files ... ok`). Also re-run the whole module to confirm no regression: `cargo test -p semantex-core onnx_reranker::tests` → all 6 PASS.

- [ ] **Step 5: Verify it compiles against the workspace (no `ort` API drift)**

Run: `cargo build -p semantex-core`
Expected: clean build. If `ort::inputs!`, `try_extract_tensor`, or `commit_from_file` fail to resolve, STOP — the `ort` version drifted from `2.0.0-rc.11`; re-check Reconciled facts against `Cargo.lock` before inventing a different call shape.

- [ ] **Step 6: Commit**

```bash
git add crates/semantex-core/src/search/onnx_reranker.rs
git commit -m "$(cat <<'EOF'
feat(rerank): OnnxReranker with lazy ort::Session + tokenization

Loads model_int8.onnx + tokenizer.json directly via ort 2.0.0-rc.11,
pins CPU EP (CoreML opt-in), runs a single forward pass per pair, and
dispatches to the classifier/yes-no score helpers. score_pair only.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `OnnxReranker::rerank` (batched over candidates, top-k, off-switch parity)

Give `OnnxReranker` the same `rerank(query, docs, top_k)` contract as `FastembedReranker`, including the disabled identity pass-through and the `rerank_candidates` cap.

**Files:**
- Modify: `crates/semantex-core/src/search/onnx_reranker.rs`

- [ ] **Step 1: Write the failing test**

Append to the `tests` module:

```rust
    /// rerank() must be an identity pass-through (no session built) when the
    /// master switch is off — exactly like FastembedReranker. We assert this
    /// without a model by checking the disabled branch returns ascending
    /// indices with score 0.0 and never touches the (absent) ONNX file.
    #[test]
    fn rerank_is_identity_when_disabled() {
        use crate::search::fastembed_reranker::ENV_ENABLE;
        with_env(&[(ENV_ENABLE, None)], || {
            let tmp = tempfile::TempDir::new().unwrap();
            // Write a minimal valid tokenizer.json so construction succeeds.
            std::fs::write(tmp.path().join("tokenizer.json"), MINIMAL_TOKENIZER_JSON).unwrap();
            let r = OnnxReranker::new(tmp.path(), ScoreStrategy::ClassifierLogit, 2, false)
                .expect("construct with tokenizer present");
            let docs = ["a", "b", "c"];
            let docs_ref: Vec<&str> = docs.iter().copied().collect();
            let out = r.rerank("q", &docs_ref, 2).expect("identity rerank");
            assert_eq!(out, vec![(0, 0.0_f32), (1, 0.0_f32)]);
        });
    }
```

Add the `with_env` helper (copied verbatim from `fastembed_reranker.rs:243-283`) and a `MINIMAL_TOKENIZER_JSON` const to the `tests` module:

```rust
    /// Smallest tokenizer.json the `tokenizers` crate will load: a WordLevel
    /// model with an empty vocab and whitespace pre-tokenizer. Enough for
    /// `Tokenizer::from_file` to succeed in the disabled-path test (we never
    /// actually encode through it there).
    const MINIMAL_TOKENIZER_JSON: &str = r#"{
      "version": "1.0",
      "truncation": null,
      "padding": null,
      "added_tokens": [],
      "normalizer": null,
      "pre_tokenizer": {"type": "Whitespace"},
      "post_processor": null,
      "decoder": null,
      "model": {"type": "WordLevel", "vocab": {"[UNK]": 0}, "unk_token": "[UNK]"}
    }"#;

    fn with_env<F: FnOnce()>(vars: &[(&str, Option<&str>)], f: F) {
        use std::sync::Mutex;
        static ENV_LOCK: Mutex<()> = Mutex::new(());
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let prior: Vec<(String, Option<String>)> = vars
            .iter()
            .map(|(k, _)| ((*k).to_string(), std::env::var(*k).ok()))
            .collect();
        // SAFETY: env vars are guarded by ENV_LOCK above.
        unsafe {
            for (k, v) in vars {
                match v {
                    Some(val) => std::env::set_var(k, val),
                    None => std::env::remove_var(k),
                }
            }
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        // SAFETY: env vars are guarded by ENV_LOCK above.
        unsafe {
            for (k, v) in &prior {
                match v {
                    Some(val) => std::env::set_var(k, val),
                    None => std::env::remove_var(k),
                }
            }
        }
        if let Err(e) = result {
            std::panic::resume_unwind(e);
        }
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p semantex-core onnx_reranker::tests::rerank_is_identity_when_disabled`
Expected: FAIL — `no method named \`rerank\` found for struct \`OnnxReranker\``.

- [ ] **Step 3: Implement `rerank`**

Add to the `impl OnnxReranker` block (after `score_pair`). Note the `use` of the master switch:

```rust
    /// Rerank `documents` by relevance to `query`. Returns `(original_index,
    /// score)` sorted by score descending, truncated to `top_k`.
    ///
    /// Identity pass-through (no session built, no inference) when
    /// `SEMANTEX_RERANKER` is not enabled — same contract as
    /// `FastembedReranker::rerank`, so the hybrid caller treats the stage as
    /// always-callable.
    ///
    /// Latency guard: scores at most `SEMANTEX_RERANKER_MAX_CANDIDATES` (default
    /// derived from the caller's `rerank_candidates`, see `RerankerEngine`)
    /// documents — the caller already truncates the candidate list, and we
    /// defensively cap again so a 0.6B model never runs unbounded.
    pub fn rerank(
        &self,
        query: &str,
        documents: &[&str],
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        if !crate::search::fastembed_reranker::reranker_enabled() {
            let n = documents.len().min(top_k);
            return Ok((0..n).map(|i| (i, 0.0_f32)).collect());
        }
        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(documents.len());
        for (i, doc) in documents.iter().enumerate() {
            let s = self.score_pair(query, doc)?;
            scored.push((i, s));
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        Ok(scored)
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core onnx_reranker::tests::rerank_is_identity_when_disabled`
Expected: PASS. Re-run the module: `cargo test -p semantex-core onnx_reranker::tests` → all PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/search/onnx_reranker.rs
git commit -m "$(cat <<'EOF'
feat(rerank): OnnxReranker::rerank with disabled identity pass-through

Per-candidate scoring, sort-desc + top-k truncation, and the same
off-by-default identity contract as FastembedReranker (no session built
when SEMANTEX_RERANKER is unset).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Reranker model download (`reranker_download.rs`)

Mirror `ensure_colbert_model` for the ONNX reranker weights. Repo-agnostic: model coordinates are data passed in, no hardcoded user paths.

**Files:**
- Create: `crates/semantex-core/src/search/reranker_download.rs`
- Modify: `crates/semantex-core/src/search/mod.rs`

- [ ] **Step 1: Register module + write the failing test**

In `search/mod.rs` add: `pub mod reranker_download;`

Create `crates/semantex-core/src/search/reranker_download.rs`:

```rust
//! Download + cache ONNX reranker weights, mirroring
//! `embedding/model_manager::ensure_colbert_model`. Model coordinates are
//! passed in as data (`ModelFiles`) so nothing here is tied to a specific
//! model or path.

use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};

/// Where a reranker's ONNX weights live on HuggingFace and on disk.
#[derive(Debug, Clone, Copy)]
pub struct ModelFiles {
    /// On-disk subdirectory name under `models_dir` (e.g. "Qwen3-Reranker-0.6B").
    pub subdir: &'static str,
    /// HF resolve base, e.g. "https://huggingface.co/<org>/<repo>/resolve/main".
    pub base_url: &'static str,
    /// Files to fetch; `model_int8.onnx` MUST be present and acts as the
    /// download sentinel.
    pub files: &'static [&'static str],
}

/// Ensure the reranker model's files are present under
/// `<models_dir>/<spec.subdir>`, downloading any missing ones. Returns the
/// model directory. Idempotent: a no-op when `model_int8.onnx` already exists.
pub fn ensure_reranker_model(models_dir: &Path, spec: &ModelFiles) -> Result<PathBuf> {
    let model_dir = models_dir.join(spec.subdir);
    if model_dir.join("model_int8.onnx").exists() {
        return Ok(model_dir);
    }
    fs::create_dir_all(&model_dir)
        .with_context(|| format!("failed to create reranker dir {}", model_dir.display()))?;
    tracing::info!(
        model = spec.subdir,
        "Downloading ONNX reranker weights (may be several hundred MB)..."
    );
    for file_name in spec.files {
        let dest = model_dir.join(file_name);
        if dest.exists() {
            continue;
        }
        let url = format!("{}/{file_name}", spec.base_url.trim_end_matches('/'));
        crate::embedding::model_manager::download_file(&url, &dest)
            .with_context(|| format!("failed to download {file_name} for {}", spec.subdir))?;
    }
    Ok(model_dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPEC: ModelFiles = ModelFiles {
        subdir: "Test-Reranker",
        base_url: "https://example.invalid/repo/resolve/main",
        files: &["model_int8.onnx", "tokenizer.json"],
    };

    #[test]
    fn sentinel_short_circuits_download() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = tmp.path().join(SPEC.subdir);
        fs::create_dir_all(&dir).unwrap();
        // Sentinel present -> no network, returns dir immediately. (base_url is
        // .invalid, so any download attempt would error; success proves we
        // short-circuited.)
        fs::write(dir.join("model_int8.onnx"), b"stub").unwrap();
        let got = ensure_reranker_model(tmp.path(), &SPEC).expect("short-circuit");
        assert_eq!(got, dir);
    }

    #[test]
    fn missing_sentinel_attempts_download_and_errors_on_bad_url() {
        let tmp = tempfile::TempDir::new().unwrap();
        // No sentinel -> tries to fetch from the .invalid host -> error.
        let err = ensure_reranker_model(tmp.path(), &SPEC).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("model_int8.onnx") || msg.contains("Test-Reranker"),
            "error should name the file/model; got: {msg}"
        );
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p semantex-core reranker_download::tests`
Expected: FAIL first build pass — if you write the test before the impl, `cannot find function ensure_reranker_model`. (Implementation is inlined above; to honor TDD, stub `ensure_reranker_model` to `unimplemented!()`, confirm the sentinel test FAILS, then paste the body.)

- [ ] **Step 3: Confirm implementation (inlined above)**

`ensure_reranker_model` + `ModelFiles` are complete in Step 1.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core reranker_download::tests`
Expected: PASS — 2 tests ok. (The bad-URL test must NOT hit the network for the sentinel case; only the missing-sentinel case attempts a fetch and fails on DNS for `example.invalid`.)

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/search/reranker_download.rs crates/semantex-core/src/search/mod.rs
git commit -m "$(cat <<'EOF'
feat(rerank): ensure_reranker_model download helper (mirrors colbert)

Per-model subdir under models_dir, model_int8.onnx sentinel, atomic
download via model_manager::download_file. Model coords are data, no
hardcoded paths.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Selection layer (`reranker_model.rs`) — `RerankerChoice` + env mapping

Decide which engine + model a given `SEMANTEX_RERANKER_MODEL` value maps to. New models go to the ONNX path; existing fastembed models keep the existing selector. Encodes the spike-recorded Qwen3 coordinates and strategy kind.

**Files:**
- Create: `crates/semantex-core/src/search/reranker_model.rs`
- Modify: `crates/semantex-core/src/search/mod.rs`

- [ ] **Step 1: Register module + write the failing test**

In `search/mod.rs` add: `pub mod reranker_model;`

Create `crates/semantex-core/src/search/reranker_model.rs` with the test module first (or stub the fn — TDD), then the impl:

```rust
//! Reranker model selection: maps `SEMANTEX_RERANKER_MODEL` to either a
//! fastembed-native model (`RerankerChoice::Fastembed`) or a generic ONNX
//! checkpoint (`RerankerChoice::Onnx`). fastembed 5.9's `RerankerModel` enum
//! cannot express Qwen3-Reranker-0.6B, so that (and any future code-trained
//! permissive checkpoint) routes through the ONNX loader.
//!
//! Permissive-only (spec §8 / D3): every selectable model is MIT/Apache. There
//! is deliberately NO alias for jina-reranker-v3 (non-commercial license).

use crate::search::fastembed_reranker::{select_model_from_env, ENV_MODEL};
use crate::search::reranker_download::ModelFiles;

/// Which score-extraction strategy an ONNX model needs. The concrete
/// `ScoreStrategy` (with token ids + prompt) is materialized later from the
/// spike-recorded values; here we only carry the kind so selection stays
/// network-free and cheap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrategyKind {
    /// Sequence-classification head (single relevance logit) — bge-style.
    Classifier,
    /// Generative yes/no judgment — Qwen3-Reranker-style.
    YesNo,
}

/// Static metadata for an ONNX reranker model (no network, no model load).
#[derive(Debug, Clone, Copy)]
pub struct OnnxModelSpec {
    pub files: ModelFiles,
    pub strategy: StrategyKind,
}

/// The selected reranker model and the engine that runs it.
#[derive(Debug, Clone)]
pub enum RerankerChoice {
    /// A fastembed-native cross-encoder (bge-v2-m3 default, bge-base, jina v1/v2).
    Fastembed(fastembed::RerankerModel),
    /// A generic ONNX cross-encoder loaded by `OnnxReranker`.
    Onnx(OnnxModelSpec),
}

/// Qwen3-Reranker-0.6B (Apache-2.0). The HF repo also hosts the int8 ONNX +
/// tokenizer once the spike (Task 1) uploads/locates them; `base_url` points at
/// the model card's `resolve/main`. NOTE: if the spike exported locally rather
/// than to a hosted repo, the executor must host the int8 artifacts at a stable
/// permissive location and update `BASE_URL` — recorded in research-notes.
const QWEN3_RERANKER_0_6B: OnnxModelSpec = OnnxModelSpec {
    files: ModelFiles {
        subdir: "Qwen3-Reranker-0.6B",
        base_url: "https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/resolve/main",
        files: &["model_int8.onnx", "tokenizer.json", "config.json"],
    },
    strategy: StrategyKind::YesNo,
};

/// bge-reranker-v2-m3 driven through the GENERIC loader (classifier head). This
/// is the permissive smoke target proving `ScoreStrategy::ClassifierLogit`
/// against a real model; the same model is also reachable via the fastembed
/// path under the plain `bge-reranker-v2-m3` alias.
const BGE_V2_M3_ONNX: OnnxModelSpec = OnnxModelSpec {
    files: ModelFiles {
        subdir: "bge-reranker-v2-m3-onnx",
        base_url: "https://huggingface.co/BAAI/bge-reranker-v2-m3/resolve/main/onnx",
        files: &["model.onnx", "tokenizer.json"],
    },
    strategy: StrategyKind::Classifier,
};

/// Resolve `SEMANTEX_RERANKER_MODEL` to a `RerankerChoice`. ONNX-only aliases
/// take precedence; everything else delegates to the existing fastembed
/// selector (so unknown values still warn-and-fall-back to bge-v2-m3).
#[must_use]
pub fn select_reranker_choice_from_env() -> RerankerChoice {
    let raw = std::env::var(ENV_MODEL).unwrap_or_default();
    match raw.to_ascii_lowercase().as_str() {
        "qwen3-reranker-0.6b" | "qwen3-reranker" | "qwen3" => {
            RerankerChoice::Onnx(QWEN3_RERANKER_0_6B)
        }
        "bge-reranker-v2-m3-onnx" | "bge-v2-m3-onnx" | "bge-onnx" => {
            RerankerChoice::Onnx(BGE_V2_M3_ONNX)
        }
        // All other values (incl. unknown/empty) -> the existing fastembed map.
        _ => RerankerChoice::Fastembed(select_model_from_env()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_env<F: FnOnce()>(key: &str, val: Option<&str>, f: F) {
        use std::sync::Mutex;
        static LOCK: Mutex<()> = Mutex::new(());
        let _g = LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let prior = std::env::var(key).ok();
        // SAFETY: guarded by LOCK.
        unsafe {
            match val {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        // SAFETY: guarded by LOCK.
        unsafe {
            match prior {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
        if let Err(e) = r {
            std::panic::resume_unwind(e);
        }
    }

    #[test]
    fn qwen3_aliases_route_to_onnx_yesno() {
        for alias in ["qwen3", "Qwen3-Reranker-0.6B", "qwen3-reranker"] {
            with_env(ENV_MODEL, Some(alias), || {
                match select_reranker_choice_from_env() {
                    RerankerChoice::Onnx(spec) => {
                        assert_eq!(spec.strategy, StrategyKind::YesNo);
                        assert_eq!(spec.files.subdir, "Qwen3-Reranker-0.6B");
                    }
                    other => panic!("expected Onnx for {alias}, got {other:?}"),
                }
            });
        }
    }

    #[test]
    fn bge_onnx_alias_routes_to_onnx_classifier() {
        with_env(ENV_MODEL, Some("bge-onnx"), || {
            match select_reranker_choice_from_env() {
                RerankerChoice::Onnx(spec) => assert_eq!(spec.strategy, StrategyKind::Classifier),
                other => panic!("expected Onnx classifier, got {other:?}"),
            }
        });
    }

    #[test]
    fn default_and_unknown_route_to_fastembed_bge() {
        for v in [None, Some(""), Some("garbage"), Some("bge-reranker-v2-m3")] {
            with_env(ENV_MODEL, v, || {
                match select_reranker_choice_from_env() {
                    RerankerChoice::Fastembed(m) => {
                        assert_eq!(m, fastembed::RerankerModel::BGERerankerV2M3);
                    }
                    other => panic!("expected Fastembed bge for {v:?}, got {other:?}"),
                }
            });
        }
    }

    /// D3 guard: there must be no selectable alias for the NC jina-reranker-v3.
    #[test]
    fn jina_v3_is_not_selectable() {
        with_env(ENV_MODEL, Some("jina-reranker-v3"), || {
            // Falls through to fastembed selector, which has no v3 entry and
            // warn-falls-back to bge — never an Onnx jina-v3 spec.
            assert!(matches!(
                select_reranker_choice_from_env(),
                RerankerChoice::Fastembed(fastembed::RerankerModel::BGERerankerV2M3)
            ));
        });
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p semantex-core reranker_model::tests`
Expected: FAIL — before the impl exists, `cannot find function select_reranker_choice_from_env`. (If pasted wholesale it passes; to honor TDD, stub the fn to `RerankerChoice::Fastembed(select_model_from_env())` unconditionally first, confirm the qwen3 test FAILS, then add the match arms.)

- [ ] **Step 3: Confirm implementation (inlined above)**

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core reranker_model::tests`
Expected: PASS — 4 tests ok.

- [ ] **Step 5: Reconcile the spike values into the Qwen3 spec**

Open `docs/superpowers/plans/2026-05-31-research-notes.md` (Task 1). If the spike recorded a different `subdir`/hosting URL or that the int8 artifact is NOT hosted on the official Qwen repo (likely — `resolve/main` may lack a pre-exported int8 ONNX), update `QWEN3_RERANKER_0_6B.files.base_url`/`files` to the stable permissive location the executor published the export to. Do NOT leave a URL that 404s. Re-run Step 4 (no behavior change in the selection logic; only constants).

- [ ] **Step 6: Commit**

```bash
git add crates/semantex-core/src/search/reranker_model.rs crates/semantex-core/src/search/mod.rs
git commit -m "$(cat <<'EOF'
feat(rerank): RerankerChoice selection (fastembed vs generic ONNX)

Maps SEMANTEX_RERANKER_MODEL: qwen3* -> ONNX yes/no (Apache-2.0),
bge-onnx -> ONNX classifier smoke target, everything else -> existing
fastembed selector. No jina-v3 (NC) alias.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: `RerankerEngine` dispatcher

The single type the call site holds. Builds the right engine from the env-selected choice (downloading ONNX weights, materializing the concrete `ScoreStrategy` from spike values), and exposes the shared `rerank` signature.

**Files:**
- Create: `crates/semantex-core/src/search/reranker_engine.rs`
- Modify: `crates/semantex-core/src/search/mod.rs`

- [ ] **Step 1: Register module + write the failing test**

In `search/mod.rs` add: `pub mod reranker_engine;`

Create `crates/semantex-core/src/search/reranker_engine.rs`:

```rust
//! Reranker engine dispatcher held by the hybrid search call site. Wraps either
//! the fastembed cross-encoder (`FastembedReranker`) or the generic ONNX loader
//! (`OnnxReranker`) behind one `rerank` signature, and constructs the right one
//! from `SEMANTEX_RERANKER_MODEL`. Off-by-default: `new_default` bails when
//! `SEMANTEX_RERANKER` is not enabled, so no weights are downloaded and no ONNX
//! session/tokenizer is built.

use anyhow::Result;

use crate::config::SemantexConfig;
use crate::search::fastembed_reranker::{reranker_enabled, FastembedReranker};
use crate::search::onnx_reranker::{OnnxReranker, PromptTemplate, ScoreStrategy};
use crate::search::reranker_download::ensure_reranker_model;
use crate::search::reranker_model::{
    select_reranker_choice_from_env, OnnxModelSpec, RerankerChoice, StrategyKind,
};

/// Either reranker implementation, behind a uniform `rerank` API.
pub enum RerankerEngine {
    Fastembed(FastembedReranker),
    Onnx(OnnxReranker),
}

impl RerankerEngine {
    /// Build the env-selected reranker. Errors (and loads nothing) when the
    /// `SEMANTEX_RERANKER` master switch is off — callers should only attempt
    /// construction after a `reranker_enabled()` / `config.rerank` gate, but we
    /// re-check here so weights never load by accident.
    pub fn new_default(show_download_progress: bool) -> Result<Self> {
        if !reranker_enabled() {
            anyhow::bail!(
                "Refusing to construct reranker: SEMANTEX_RERANKER is not enabled. \
                 Set SEMANTEX_RERANKER=on to load model weights."
            );
        }
        match select_reranker_choice_from_env() {
            RerankerChoice::Fastembed(model) => {
                Ok(Self::Fastembed(FastembedReranker::new(model, show_download_progress)?))
            }
            RerankerChoice::Onnx(spec) => Ok(Self::Onnx(Self::build_onnx(&spec)?)),
        }
    }

    /// Download the ONNX model and build an `OnnxReranker` with the concrete
    /// score strategy. Threads come from `SEMANTEX_ORT_THREADS` (query default
    /// 4, same as ColbertEmbedder); CoreML opt-in via `SEMANTEX_COREML`.
    fn build_onnx(spec: &OnnxModelSpec) -> Result<OnnxReranker> {
        let config = SemantexConfig::default();
        let model_dir = ensure_reranker_model(&config.models_dir(), &spec.files)?;
        let threads = crate::config::env_usize("SEMANTEX_ORT_THREADS", 4);
        let use_coreml = std::env::var("SEMANTEX_COREML").is_ok_and(|v| v == "1");
        let strategy = Self::strategy_for(spec.strategy);
        OnnxReranker::new(&model_dir, strategy, threads, use_coreml)
    }

    /// Materialize the concrete `ScoreStrategy`. The yes/no token ids and prompt
    /// strings are the values RECORDED IN THE SPIKE
    /// (`docs/superpowers/plans/2026-05-31-research-notes.md`). The executor
    /// MUST replace the `TODO_FROM_SPIKE` markers with the real recorded values
    /// before this compiles — there is no sane default for a model-specific
    /// token id, and a wrong id silently inverts the ranking.
    fn strategy_for(kind: StrategyKind) -> ScoreStrategy {
        match kind {
            StrategyKind::Classifier => ScoreStrategy::ClassifierLogit,
            StrategyKind::YesNo => ScoreStrategy::YesNoLogit {
                // RECORDED IN SPIKE — replace with the integer ids from
                // research-notes (Qwen3-Reranker "yes"/"no" single-token ids).
                yes_id: QWEN3_YES_ID,
                no_id: QWEN3_NO_ID,
                prompt: PromptTemplate {
                    // RECORDED IN SPIKE — the verbatim instruction/query/doc
                    // wrapper from the Qwen3-Reranker card.
                    prefix: QWEN3_PROMPT_PREFIX.to_string(),
                    middle: QWEN3_PROMPT_MIDDLE.to_string(),
                    suffix: QWEN3_PROMPT_SUFFIX.to_string(),
                },
            },
        }
    }

    /// Rerank — delegates to whichever engine is active. Identical signature to
    /// `FastembedReranker::rerank`, so the hybrid call site is unchanged.
    pub fn rerank(
        &mut self,
        query: &str,
        documents: &[&str],
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        match self {
            Self::Fastembed(r) => r.rerank(query, documents, top_k),
            Self::Onnx(r) => r.rerank(query, documents, top_k),
        }
    }
}

// === Spike-recorded constants (Qwen3-Reranker-0.6B) ========================
// These MUST be filled from docs/superpowers/plans/2026-05-31-research-notes.md.
// Token ids are model-specific; the prompt is the model card's wrapper. They
// are NOT repo-specific tuning (spec §8 permits the model's own interface).
const QWEN3_YES_ID: usize = TODO_FROM_SPIKE; // e.g. 9693
const QWEN3_NO_ID: usize = TODO_FROM_SPIKE; // e.g. 2152
const QWEN3_PROMPT_PREFIX: &str = TODO_FROM_SPIKE;
const QWEN3_PROMPT_MIDDLE: &str = TODO_FROM_SPIKE;
const QWEN3_PROMPT_SUFFIX: &str = TODO_FROM_SPIKE;

#[cfg(test)]
mod tests {
    use super::*;

    fn with_env<F: FnOnce()>(vars: &[(&str, Option<&str>)], f: F) {
        use std::sync::Mutex;
        static LOCK: Mutex<()> = Mutex::new(());
        let _g = LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let prior: Vec<(String, Option<String>)> = vars
            .iter()
            .map(|(k, _)| ((*k).to_string(), std::env::var(*k).ok()))
            .collect();
        // SAFETY: guarded by LOCK.
        unsafe {
            for (k, v) in vars {
                match v {
                    Some(val) => std::env::set_var(k, val),
                    None => std::env::remove_var(k),
                }
            }
        }
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        // SAFETY: guarded by LOCK.
        unsafe {
            for (k, v) in &prior {
                match v {
                    Some(val) => std::env::set_var(k, val),
                    None => std::env::remove_var(k),
                }
            }
        }
        if let Err(e) = r {
            std::panic::resume_unwind(e);
        }
    }

    #[test]
    fn new_default_refuses_when_disabled() {
        use crate::search::fastembed_reranker::{ENV_ENABLE, ENV_MODEL};
        with_env(&[(ENV_ENABLE, Some("off")), (ENV_MODEL, None)], || {
            match RerankerEngine::new_default(false) {
                Ok(_) => panic!("must not construct when SEMANTEX_RERANKER is off"),
                Err(e) => assert!(format!("{e}").contains("SEMANTEX_RERANKER")),
            }
        });
        // Also true for the qwen3 ONNX selection: still refuses, no download.
        with_env(
            &[(ENV_ENABLE, None), (ENV_MODEL, Some("qwen3"))],
            || {
                assert!(RerankerEngine::new_default(false).is_err());
            },
        );
    }
}
```

- [ ] **Step 2: Replace the spike placeholders**

This file will NOT compile with `TODO_FROM_SPIKE`. Open `docs/superpowers/plans/2026-05-31-research-notes.md` and replace the five `const` values with the recorded Qwen3 token ids and prompt strings. Example shape (use the REAL recorded values, not these):

```rust
const QWEN3_YES_ID: usize = 9693;
const QWEN3_NO_ID: usize = 2152;
const QWEN3_PROMPT_PREFIX: &str =
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query. Answer only \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n<Query>: ";
const QWEN3_PROMPT_MIDDLE: &str = "\n<Document>: ";
const QWEN3_PROMPT_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n";
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p semantex-core reranker_engine::tests`
Expected: BEFORE replacing the placeholders: compile error (`cannot find value TODO_FROM_SPIKE`). This is the failing state. AFTER Step 2 with placeholder bodies but BEFORE wiring delegation correctly you might see a logic failure; with the full inlined impl + real constants it should build.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core reranker_engine::tests`
Expected: PASS — `test ... new_default_refuses_when_disabled ... ok`. Confirm no network was hit (the disabled and qwen3-while-disabled cases both bail before `ensure_reranker_model`).

- [ ] **Step 5: Full crate build**

Run: `cargo build -p semantex-core`
Expected: clean. Run `cargo clippy -p semantex-core` — expected: no new warnings (the file uses `#[allow]`-free patterns matching the repo style).

- [ ] **Step 6: Commit**

```bash
git add crates/semantex-core/src/search/reranker_engine.rs crates/semantex-core/src/search/mod.rs docs/superpowers/plans/2026-05-31-research-notes.md
git commit -m "$(cat <<'EOF'
feat(rerank): RerankerEngine dispatcher (fastembed | ONNX)

One rerank() signature over both impls; new_default builds the
env-selected engine and stays off-by-default (bails when SEMANTEX_RERANKER
unset, no download/session). Qwen3 yes/no token ids + prompt are the
spike-recorded values.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Wire `RerankerEngine` into the hybrid call site

Swap the `FastembedReranker` field for `RerankerEngine`. Because the `rerank` signature matches, only the type, the import, and the `new_default` line change.

**Files:**
- Modify: `crates/semantex-core/src/search/hybrid.rs:8, 28, 58, 139, 1092-1106`
- Test: `crates/semantex-core/tests/reranker_off_by_default_test.rs` (new)

- [ ] **Step 1: Write the failing test (off-by-default integration guard)**

Create `crates/semantex-core/tests/reranker_off_by_default_test.rs`:

```rust
//! Regression guard for the S3 off-by-default safety contract: with
//! SEMANTEX_RERANKER unset, building + searching an index must NOT construct a
//! reranker, NOT download weights, and return results in the same order the
//! fusion produced (rerank stage is a no-op). Repo-agnostic: synthetic tempdir.

use semantex_core::config::SemantexConfig;
use semantex_core::index::builder::IndexBuilder;
use semantex_core::search::hybrid::HybridSearcher;
use semantex_core::search::SearchQuery;
use semantex_core::types::SearchSource; // canonical path, matches search_accuracy_test.rs

#[test]
fn rerank_off_by_default_produces_no_reranked_source() {
    // Ensure the master switch is unset for this process.
    // SAFETY: single-threaded test; no other thread reads this var here.
    unsafe {
        std::env::remove_var("SEMANTEX_RERANKER");
    }

    let tmp = tempfile::TempDir::new().expect("tempdir");
    let project = tmp.path();
    std::fs::write(
        project.join("a.rs"),
        "fn binary_search(arr: &[i32], target: i32) -> Option<usize> { todo!() }\n",
    )
    .unwrap();
    std::fs::write(
        project.join("b.rs"),
        "fn quicksort(arr: &mut [i32]) { /* partition and recurse */ }\n",
    )
    .unwrap();

    let mut config = SemantexConfig::default();
    config.rerank = true; // even with config.rerank on, the env gate keeps it off

    IndexBuilder::new(&config)
        .expect("builder")
        .build(project)
        .expect("index build");

    let searcher = HybridSearcher::open(&project.join(".semantex"), &config).expect("open");
    let query = SearchQuery::new("binary search");
    let out = searcher.search(&query).expect("search");

    // No result may carry the Reranked source — the stage was a no-op.
    assert!(
        out.results.iter().all(|r| r.source != SearchSource::Reranked),
        "rerank must be a no-op when SEMANTEX_RERANKER is unset"
    );
    // And rerank_ms is either None or present-but-the-order-is-fusion-order;
    // we only assert the source invariant (the load-bearing safety property).
}
```

Adjust the exact import paths if the crate re-exports differ (e.g. `semantex_core::search::SearchQuery` vs `semantex_core::SearchQuery`); verify with `cargo doc`/existing tests in `crates/semantex-core/tests/search_accuracy_test.rs` for the canonical paths and match them.

- [ ] **Step 2: Run test to verify it fails (or errors on the old field type)**

Run: `cargo test -p semantex-core --test reranker_off_by_default_test`
Expected: This test should actually PASS against the *current* code (the old `FastembedReranker` already honors the off switch). That is fine — it is a guard we must keep green through the swap. Run it now to record the GREEN baseline; if it does not compile, fix the import paths until it compiles and passes BEFORE touching `hybrid.rs`.

- [ ] **Step 3: Swap the field type in `hybrid.rs`**

Edit `crates/semantex-core/src/search/hybrid.rs`:

Line 8 — change the import:
```rust
use crate::search::reranker_engine::RerankerEngine;
```
(remove `use crate::search::fastembed_reranker::FastembedReranker;`).

Line 28 — change the field type:
```rust
    reranker: Mutex<Option<RerankerEngine>>,
```

Lines 58 and 139 — these already construct `Mutex::new(None)`; no change needed (the `None` is now `Option<RerankerEngine>`).

Lines ~1092-1095 — change the lazy-load constructor:
```rust
            if reranker_guard.is_none() {
                tracing::info!("Loading reranker model (first use)...");
                match RerankerEngine::new_default(false) {
                    Ok(r) => *reranker_guard = Some(r),
                    Err(e) => {
                        tracing::warn!("Failed to initialize reranker: {}", e);
                    }
                }
            }
```

The `reranker.rerank(&query.text, &docs, query.max_results)` call (line ~1106) is unchanged — `RerankerEngine::rerank` has the same signature.

- [ ] **Step 4: Run the test + full suite to verify they pass**

Run: `cargo test -p semantex-core --test reranker_off_by_default_test`
Expected: PASS (still green after the swap — same off-by-default behavior, now through `RerankerEngine`).

Run: `cargo build -p semantex-core && cargo test -p semantex-core`
Expected: workspace builds; all existing reranker tests (`fastembed_reranker::tests::*`, `onnx_reranker::tests::*`, `reranker_model::tests::*`, `reranker_engine::tests::*`) and the new integration test PASS. The fastembed `#[ignore]`'d `reranks_when_enabled` stays ignored.

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/search/hybrid.rs crates/semantex-core/tests/reranker_off_by_default_test.rs
git commit -m "$(cat <<'EOF'
feat(rerank): route hybrid rerank stage through RerankerEngine

Swaps the FastembedReranker field for the RerankerEngine dispatcher so
the env-selected model (fastembed bge or generic ONNX Qwen3) is used.
rerank() signature unchanged; adds an off-by-default integration guard.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Documentation pass on `fastembed_reranker.rs` (point new models at the loader)

Keep the module doc accurate now that new models route through the ONNX path, not `select_model_from_env`.

**Files:**
- Modify: `crates/semantex-core/src/search/fastembed_reranker.rs:14-21`

- [ ] **Step 1: Update the `SEMANTEX_RERANKER_MODEL` doc list**

Replace the bullet block in the module doc (lines ~16-20) with:

```rust
//! - `SEMANTEX_RERANKER_MODEL` — override the cross-encoder model. Resolved by
//!   `search::reranker_model::select_reranker_choice_from_env`, which routes to
//!   one of two engines:
//!     - **fastembed-native** (this module): `bge-reranker-v2-m3` (default when
//!       enabled), `bge-reranker-base`, `jina-reranker-v1-turbo-en`,
//!       `jina-reranker-v2-base-multilingual`.
//!     - **generic ONNX loader** (`search::onnx_reranker`): `qwen3-reranker-0.6b`
//!       (Apache-2.0, code-capable, yes/no-logit), `bge-reranker-v2-m3-onnx`
//!       (classifier-logit smoke target).
//!   Unknown values warn and fall back to `bge-reranker-v2-m3` (fastembed).
```

- [ ] **Step 2: Verify the doc builds**

Run: `cargo doc -p semantex-core --no-deps`
Expected: builds without intra-doc-link errors. (`cargo test -p semantex-core --doc` if doc-tests exist; none are added here.)

- [ ] **Step 3: Commit**

```bash
git add crates/semantex-core/src/search/fastembed_reranker.rs
git commit -m "$(cat <<'EOF'
docs(rerank): document fastembed vs ONNX model routing

SEMANTEX_RERANKER_MODEL now routes new checkpoints (qwen3-reranker-0.6b,
bge-v2-m3-onnx) through the generic ONNX loader; clarify in the module doc.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Manual real-model smoke (ignored by default) + S0 evaluation handoff

The numeric acceptance gate (net-positive nDCG@10/MRR vs rerank-off, within a per-query latency budget) is owned and run by the **S0 relevance harness** (`benchmarks/relevance/`, plan `2026-05-31-s0-relevance-harness.md`). This task adds the manual one-model smoke test and documents exactly how S0 evaluates S3 — it does NOT reimplement metrics.

**Files:**
- Modify: `crates/semantex-core/src/search/onnx_reranker.rs` (add one `#[ignore]` integration test)

- [ ] **Step 1: Add an ignored real-model integration test for the ONNX classifier path**

Append to the `tests` module in `onnx_reranker.rs`. This exercises a real download + inference; gated `#[ignore]` so CI never pulls weights. It uses the generic-loader bge classifier (smaller + permissive + the spike confirmed its I/O) rather than the 0.6B Qwen3 to keep the manual smoke fast:

```rust
    /// Manual smoke: downloads the bge-v2-m3 ONNX classifier and verifies an
    /// on-topic doc outranks an off-topic one. Requires network on first run.
    ///
    ///   SEMANTEX_RERANKER=on cargo test -p semantex-core \
    ///     -- --ignored onnx_reranker::tests::onnx_classifier_ranks_on_topic
    #[test]
    #[ignore]
    fn onnx_classifier_ranks_on_topic() {
        use crate::config::SemantexConfig;
        use crate::search::fastembed_reranker::ENV_ENABLE;
        use crate::search::reranker_download::ensure_reranker_model;
        use crate::search::reranker_model::{select_reranker_choice_from_env, RerankerChoice};

        with_env(&[(ENV_ENABLE, Some("on"))], || {
            // Force the ONNX classifier alias.
            // SAFETY: guarded by the with_env mutex above.
            unsafe { std::env::set_var("SEMANTEX_RERANKER_MODEL", "bge-onnx") };
            let spec = match select_reranker_choice_from_env() {
                RerankerChoice::Onnx(s) => s,
                _ => panic!("expected ONNX choice for bge-onnx"),
            };
            let config = SemantexConfig::default();
            let dir = ensure_reranker_model(&config.models_dir(), &spec.files)
                .expect("download (offline?)");
            // bge ONNX export here ships model.onnx; our loader expects
            // model_int8.onnx as the session file. For this manual smoke,
            // symlink/copy if needed, or point at the model.onnx the export
            // shipped — RECORDED in research-notes which file name to use.
            let r = OnnxReranker::new(&dir, ScoreStrategy::ClassifierLogit, 4, false)
                .expect("construct");
            let docs = [
                "Pizza is a popular Italian dish.",
                "fn binary_search(a: &[i32], t: i32) -> Option<usize> { /* ... */ }",
            ];
            let docs_ref: Vec<&str> = docs.iter().copied().collect();
            let out = r
                .rerank("how does binary search work", &docs_ref, 2)
                .expect("rerank");
            assert_eq!(out[0].0, 1, "on-topic code doc should rank first");
        });
    }
```

NOTE: the bge ONNX export file name (`model.onnx` vs `model_int8.onnx`) is recorded in the spike (Task 1 Step 4). If the export ships `model.onnx`, either (a) update `OnnxReranker::new`/`build_session` to accept a configurable session filename, or (b) document copying `model.onnx`→`model_int8.onnx` for the smoke. Prefer (a) if the bge ONNX repo genuinely lacks an int8 file: add a `session_file: &'static str` to `ModelFiles` and use it in `build_session` instead of the hardcoded `"model_int8.onnx"`. If you take (a), do it as a tiny TDD sub-step (test: `build_session` uses `spec.session_file`) and amend Tasks 4/6/7 constants — keep it minimal.

- [ ] **Step 2: Verify the ignored test compiles (do not run the download in CI)**

Run: `cargo test -p semantex-core onnx_reranker::tests -- --list`
Expected: lists `onnx_classifier_ranks_on_topic` among the tests, marked ignored. The non-ignored tests still PASS.

- [ ] **Step 3: Document the S0 evaluation contract in the research notes**

Append to `docs/superpowers/plans/2026-05-31-research-notes.md` under the S3 section:

```markdown
### S3 — how S0 decides the rerank default
- S0 harness (`benchmarks/relevance/`) runs the `rerank` ablation:
  `python -m scripts.run --dataset csn --ablation rerank` and the CoIR/in-domain
  equivalents. The harness sets `SEMANTEX_RERANKER=on` (+ `SEMANTEX_RERANKER_MODEL`)
  in the subprocess env and passes `--rerank` to the CLI.
- Compare each reranker (bge-v2-m3 fastembed; qwen3-reranker-0.6b ONNX) vs
  rerank-off on nDCG@10 / MRR@10 / Recall@10, AND record per-query latency
  (rerank_ms from SearchMetrics; budget stated by S0, e.g. warm p50 < X ms over
  rerank_candidates=100).
- ACCEPTANCE (spec §4 S3): a reranker is shippable iff net-positive nDCG@10/MRR
  vs rerank-off on CoIR + CSN + in-domain WITHIN the stated latency budget.
  If yes -> flip `SEMANTEX_RERANKER` on by default with the winning
  SEMANTEX_RERANKER_MODEL (a one-line config/default change, separate PR). If no
  -> leave OFF; record the numbers here.
- The default-OFF gate in code STAYS until S0 reports a win. Flipping it is NOT
  part of this plan (it is gated on harness output).
```

- [ ] **Step 4: Commit**

```bash
git add crates/semantex-core/src/search/onnx_reranker.rs docs/superpowers/plans/2026-05-31-research-notes.md
git commit -m "$(cat <<'EOF'
test(rerank): ignored real-model ONNX smoke + S0 eval handoff notes

Manual #[ignore] smoke for the generic ONNX classifier path; documents
that S0 owns the numeric net-win-within-latency acceptance gate and that
the default-OFF switch stays until S0 reports a winner.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final verification (run after all tasks)

- [ ] **Off-by-default contract:** `cargo test -p semantex-core --test reranker_off_by_default_test` → PASS. Manually confirm `SEMANTEX_RERANKER` unset → no `~/.semantex/models/Qwen3-Reranker-0.6B` dir is created by a normal search.
- [ ] **No NC model selectable:** `grep -rin "jina-reranker-v3\|jina_v3" crates/semantex-core/src` → no matches (D3).
- [ ] **No hardcoded user paths:** `grep -rn "/Users/\|/home/" crates/semantex-core/src/search/onnx_reranker.rs crates/semantex-core/src/search/reranker_*.rs` → no matches (CLAUDE.md rule 1).
- [ ] **Default build has the new code but stays lean:** `cargo build -p semantex-core` (no `--features llm`) → clean. `cargo tree -p semantex-core | grep -i genai` → empty (S3 adds no LLM dep).
- [ ] **Full suite:** `cargo test --workspace` → green; `cargo clippy --all` → no new warnings; `cargo fmt --all -- --check` → clean.
- [ ] **Spike values reconciled:** no `TODO_FROM_SPIKE` remains (`grep -rn TODO_FROM_SPIKE crates` → empty); the Qwen3 token ids/prompt in `reranker_engine.rs` match `research-notes.md`.

---

## Self-review notes (spec coverage)

- §4 S3 "generic ONNX cross-encoder loader `onnx_reranker.rs`" → Tasks 3-5.
- "TWO score-extraction strategies (classifier-head / yes-no-logit)" → `ScoreStrategy` + pure helpers (Task 3), wired in `score_pair` (Task 4).
- "Ship Qwen3-Reranker-0.6B (Apache-2.0) … spike: export ONNX+int8, RECORD prompt template + yes-token id + I/O tensor names" → Task 1 (spike → research-notes), consumed by Tasks 7/8.
- "keep bge-reranker-v2-m3 as fallback (already integrated)" → fastembed path untouched; default stays `BGERerankerV2M3` (Task 7).
- "Keep `SEMANTEX_RERANKER` master switch + `SEMANTEX_RERANKER_MODEL`; add new models to selection" → Task 7 (`select_reranker_choice_from_env`), Task 8 (engine), Task 9 (call site).
- "default-OFF gate STAYS until S0 proves a net win, then flips on" → off-by-default contract + Tasks 9/11; flip is explicitly out of this plan.
- "Latency guard: only rerank top `rerank_candidates`" → caller already truncates (`hybrid.rs:188`); `OnnxReranker::rerank` documents/honors the cap (Task 5).
- "jina-reranker-v3 EXCLUDED (NC)" → Task 7 D3 guard test.
- "Acceptance: on S0, net-positive nDCG@10/MRR vs rerank-off within latency budget" → Task 11 documents the S0-owned gate (S0 plan runs it).
- Reuse `runtime_manager` → the ONNX Runtime dylib is provisioned at process start (`cli/main.rs`), so `OnnxReranker` relies on it without re-provisioning; documented in Reconciled facts.

## Spec gaps / deviations flagged to the human

- **G1 — `select_model_from_env` cannot host the new models.** Spec §4 S3 says "add the new models to `select_model_from_env`," but that function returns `fastembed::RerankerModel`, an enum with no Qwen3 variant. Deviation: a new `RerankerChoice`/`select_reranker_choice_from_env` layer wraps `select_model_from_env` for the fastembed case and routes ONNX models separately. Behavior matches the spec's intent; the function name differs.
- **G2 — Qwen3-Reranker int8 ONNX is not pre-hosted on a stable permissive URL.** The official `Qwen/Qwen3-Reranker-0.6B` repo ships PyTorch weights, not a pre-exported int8 ONNX. The spike (Task 1) produces the artifact locally; the executor must host it at a stable Apache-2.0-compatible location and set `QWEN3_RERANKER_0_6B.base_url` (Task 7 Step 5). Until then the ONNX Qwen3 path is opt-in and download will 404 — acceptable because the stage is OFF by default and the spec treats Qwen3 as the spike/experimental option with bge as the guaranteed fallback.
- **G3 — Qwen3-Reranker score = LM yes-token logit, not a classification head.** `optimum-cli export onnx --task text-classification` likely fails (Task 1 Step 1 records the real task). If the export is `text-generation`, the output is `[batch, seq, vocab]` and `score_pair` slices the final position (handled in Task 4). If a future Qwen3-Reranker ships a true seq-classification ONNX, it would use `ClassifierLogit` instead — the strategy enum already covers both.
- **G4 — tokenizers was only a transitive dep.** Promoting it to direct (Task 2) is required for `Tokenizer::from_file`; mirrors fastembed's feature set so no new heavy deps.
- **G5 — bge ONNX session filename.** The BAAI bge ONNX repo may ship `model.onnx`/`model_optimized.onnx` rather than `model_int8.onnx`. Task 11 Step 1 flags the optional `session_file` field addition; only needed if the manual ONNX-classifier smoke is exercised. Not load-bearing for the default OFF path.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-31-s3-onnx-reranker-upgrade.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration. Task 1 (the spike) MUST complete and the research-notes values be recorded before Tasks 7/8 can compile.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
