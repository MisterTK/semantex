# S8 — Model Registry & Hot-Swap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a config-driven model registry (`crates/semantex-core/src/model/`) that declares every model (embedder, reranker, LLM) as **data** — `ModelSpec` + `ModelCapabilities` resolved by `ModelRegistry` from compiled-in permissive defaults plus an optional user `models.toml` — so any model can be swapped by config with no recompile and (for query-time models) no reindex. The registry becomes the single place where the active embedder / reranker / LLM is *selected*; S1/S2/S3's traits and impls are **unchanged**, only their ad-hoc env/string selection moves behind `ModelRegistry`. Embedder swaps are made zero-downtime via a per-spec **fingerprint** stamped into `meta.json` and a versioned dense dir (`.semantex/dense/<backend>/<fingerprint>/`) with an atomically flipped active pointer.

**Architecture:** A new leaf module `crates/semantex-core/src/model/` with four files: `spec.rs` (the `ModelSpec`/`ModelRole`/`ModelSource` + role-specific data structs + the `EmbedderFingerprint`), `capabilities.rs` (`ModelCapabilities` + the `BackendKind` capability-negotiation function), `manifest.rs` (compiled-in permissive `builtin_specs()` + `models.toml` parse/merge via the `toml` crate), and `registry.rs` (`ModelRegistry`: holds merged specs, resolves a role's active spec from `SemantexConfig`/env, lazily caches). The registry depends only on `SemantexConfig` + `serde`/`toml` — it does **not** depend on `search` or `embedding` (avoids cycles), so S1/S2/S3 call *into* it. S1's `DenseBackendKind::parse(&config.dense_backend)` is replaced by `registry.embedder_backend_kind()?`, which resolves the active embedder `ModelSpec` and runs `backend_for(&spec.capabilities).dense_kind()` to yield `ColbertPlaid` or `CoderankHnsw`. S3's `select_reranker_choice_from_env()` and S2's CodeRankEmbed constants are reframed as registry lookups / a built-in `ModelSpec`. LLM-role specs are feature-gated (`#[cfg(feature = "llm")]`) so a default build pulls zero LLM deps. The embedder spec's fingerprint generalizes the existing `schema_version` / `use_bm25_stemmer` / `dense_backend` mismatch guards into one uniform "index stamped with its embedder spec; mismatch → versioned rebuild + atomic switchover."

**Tech Stack:** Rust 2024 edition (rust-version 1.91), `anyhow`, `serde`/`serde_json` (already present), **`toml = "0.8"` (NEW direct dep — pure Rust, no C/sys deps, verified)**, `xxhash-rust` (already present, `xxh64` feature — reused for the fingerprint hash), `tempfile` + `cargo test` for tests. No LLM deps in the default build (`cargo tree | grep genai` stays empty). No new crate beyond `toml`.

---

## Reconciled facts (verified against current source — do not re-derive)

These are quoted from the real tree at plan-authoring time. Every type/method referenced below exists today, is added by an earlier task in **this** plan, or is added by S1/S2/S3 (noted explicitly). No later task invents an API.

### From the current tree

- **`SemantexConfig`** (`crates/semantex-core/src/config.rs:14-98`) is `#[derive(Debug, Clone, Serialize, Deserialize)]` + `#[serde(default)]`. Env overrides live in `load()` (lines 127-141), each guarded `if let Ok(v) = std::env::var("SEMANTEX_…")`. Helpers: `pub(crate) fn env_usize(key: &str, default: usize) -> usize` (line 201); **S1 adds** `pub(crate) fn env_string(key: &str, default: &str) -> String`. `pub fn models_dir(&self) -> PathBuf` (line 158), `pub fn semantex_home() -> PathBuf` (line 147), `pub fn project_index_dir(project_path: &Path) -> PathBuf` (line 165) = `<project>/.semantex`. **S1 adds** `pub dense_backend: String` (default `"colbert-plaid"`) + `SEMANTEX_DENSE_BACKEND` overlay. There is no `embedder` / `reranker_model` / `llm_model` field today — **this plan adds `pub embedder: String`, `pub reranker_model: String`, `pub llm_model: String`** (Task 8).

- **`IndexMeta`** (`crates/semantex-core/src/types.rs:178-208`): `schema_version, project_path, created_at, updated_at, file_count, chunk_count, embedding_model: String, embedding_dim: u32, use_bm25_stemmer: bool`. `CURRENT_SCHEMA_VERSION = 9` today. **S1 bumps 9→10 and adds `dense_backend: String`; S2 bumps 10→11.** The final shipped value is **11** (integration doc §3.4). **S8 adds `pub embedder_fingerprint: String`** to `IndexMeta` (Task 6) — this rides on S1's/S2's existing bump (NO additional schema bump; an absent field on an old meta already triggers `Stale` via the "unparseable meta → Stale" rule in `index/state.rs:76-80`).

- **`builder.rs` meta write** (`crates/semantex-core/src/index/builder.rs:862-874`): builds an `IndexMeta { … embedding_model: "LateOn-Code-edge".to_string(), embedding_dim: 48, use_bm25_stemmer: self.config.use_bm25_stemmer }` and `std::fs::write(index_dir.join("meta.json"), serde_json::to_string_pretty(&meta)?)`. **S2 already changes `embedding_model`/`embedding_dim` per active backend.** S8 additionally sets `embedder_fingerprint` here (Task 9).

- **`index/state.rs`**: `is_stale(meta_path)` returns `true` for unreadable OR unparseable meta OR `schema_version != CURRENT_SCHEMA_VERSION` (lines 72-81). `detect(project_path) -> IndexState` (line 25). `index_age_secs` (line 54). The `test_ready` JSON literal (lines 122-132) carries `use_bm25_stemmer` — **S1 already adds `dense_backend` to it; S8 adds `embedder_fingerprint`** so it round-trips current-shape.

- **`registry.rs` JSON-array precedent** (`crates/semantex-core/src/index/registry.rs`): `~/.semantex/projects.json` read/written via `dirs::home_dir().map(|h| h.join(".semantex").join(...))`. This is the precedent for locating a user file under `~/.semantex/` — `models.toml` mirrors it (Task 4).

- **`model_manager.rs`** (`crates/semantex-core/src/embedding/model_manager.rs`): `ensure_colbert_model(models_dir: &Path) -> Result<PathBuf>` (line 14) is the download-on-first-use pattern (per-model subdir under `models_dir`, sentinel = `model_int8.onnx`, files listed in a `const &[&str]`). `pub(crate) fn download_file(url: &str, dest: &Path) -> Result<()>` (line 44). **S8 does NOT download anything** — it carries the HF repo / file list / dir name as `ModelSource` *data*; the existing `embedding/*_model.rs` provisioners (ColBERT's `model_manager`, S2's `single_vector_model`) remain the download path, now fed coordinates from the spec.

- **`fastembed_reranker.rs`** (`crates/semantex-core/src/search/fastembed_reranker.rs`): `pub const ENV_ENABLE: &str = "SEMANTEX_RERANKER"` (line 53); `pub const ENV_MODEL: &str = "SEMANTEX_RERANKER_MODEL"` (line 55); `pub fn reranker_enabled() -> bool` (line 62); `pub fn select_model_from_env() -> fastembed::RerankerModel` (line 72, maps strings → `RerankerModel`, unknown → warn + `BGERerankerV2M3`). **S3 introduces** `RerankerChoice` + `select_reranker_choice_from_env()` (integration doc §3.6). S8 reframes that selection as a registry lookup keyed by `SEMANTEX_RERANKER_MODEL` (Task 11 reconciliation — additive, the `SEMANTEX_RERANKER` master switch + off-by-default identity pass-through are untouched).

- **`llm/mod.rs`** is entirely `#![cfg(feature = "llm")]` (line 2). `LlmBackend::from_env() -> anyhow::Result<Option<Self>>` (line 67) probes `GenaiBackend::from_env()` then `SubscriptionCliBackend::from_env()`. Env contract (documented at lines 109-110): `SEMANTEX_LLM_MODEL`, `SEMANTEX_LLM_PROVIDER`, `SEMANTEX_LLM_ENDPOINT`, `SEMANTEX_LLM_BACKEND`. CLAUDE.md rule #6/#8: no hardcoded model names/providers/endpoints in production; default build zero-LLM-deps. **S8's LLM-role specs MUST be feature-gated and inert without `llm`** (Task 12).

- **`xxhash-rust`** is a direct dep (`Cargo.toml:85`, `features = ["xxh64"]`). `xxhash_rust::xxh64::xxh64(bytes: &[u8], seed: u64) -> u64`. Reused for the spec content hash in `EmbedderFingerprint` (no new hashing dep).

- **`toml` is NOT a current dependency** (verified: absent from `Cargo.lock`; the only `toml` strings in `semantex-core/Cargo.toml` are the `tree-sitter-toml-ng` / `tree-sitter-*` grammars). `cargo add toml` resolves to a pure-Rust tree with **no `*-sys`/`cc`/bindgen** (verified via `cargo tree`). Task 1 adds it.

- **Module wiring:** `crates/semantex-core/src/lib.rs` declares the top-level modules (`pub mod config; pub mod embedding; pub mod index; pub mod search; pub mod types; …`). Task 2 adds `pub mod model;`.

- **Repo-agnostic test harness pattern** (`crates/semantex-core/tests/search_accuracy_test.rs`): `TempDir` → synthetic source files → `IndexBuilder::new(&config)?.build(&project_dir)?` → `HybridSearcher::open(&project_dir.join(".semantex"), &config)`. No hardcoded `/Users/` paths (CLAUDE.md Hard Rule #1).

### From S1 / S2 / S3 (consumed; not redefined here)

- **S1** provides `crate::search::dense_backend::{DenseBackend, DenseIndexBuilder, DenseBackendKind, dense_subdir, verify_persisted_backend_matches}` and `DenseHit = crate::types::ScoredChunkId`. `DenseBackendKind { #[default] ColbertPlaid }` with `name(self) -> &'static str` + `parse(&str) -> Option<Self>`. **S2 adds the `CoderankHnsw` variant.** `dense_subdir(index_dir, backend) -> <index_dir>/dense/<backend.name()>`. S8 builds **on top** of these: the registry resolves an embedder `ModelSpec` → its `ModelCapabilities` → a `DenseBackendKind` (capability negotiation), and the versioned dir extends `dense_subdir` with a `<fingerprint>` leaf.
- **S2** provides `embedding/single_vector.rs` (`SingleVectorEmbedder`, `EMBEDDING_DIM`, `QUERY_PREFIX`) and `index/hnsw_index.rs` (`CoderankHnswBackend`/`CoderankHnswIndexBuilder`). S8 turns CodeRankEmbed's recorded constants into a built-in `ModelSpec` (Task 3) without changing those modules.
- **S3** provides `search/reranker_model.rs` (`RerankerChoice`, `select_reranker_choice_from_env()`) + `onnx_reranker.rs` (`ScoreStrategy::{ClassifierLogit, YesNoLogit}`). S8's reranker `ModelSpec` carries `score_strategy` + template/yes-token as data, mirroring those fields (Task 3 + Task 11 reconciliation).

---

## File Structure

Files created or modified, one responsibility each:

- **Create `crates/semantex-core/src/model/mod.rs`** — module root. Re-exports the public surface: `pub use spec::{ModelSpec, ModelRole, ModelSource, EmbedderSpec, RerankerSpec, LlmSpec, Pooling, QuantKind, ScoreStrategyKind, EmbedderFingerprint};`, `pub use capabilities::{ModelCapabilities, BackendKind};`, `pub use registry::ModelRegistry;`. Holds the module-level doc comment describing "models are data, not code".

- **Create `crates/semantex-core/src/model/spec.rs`** — the data model. `ModelRole` (`Embedder|Reranker|Llm`), `ModelSource` (`Hf{repo,files}|Local{dir}|Url{base,files}`), the role-specific structs (`EmbedderSpec`, `RerankerSpec`, `LlmSpec`), `Pooling` (`Mean|Cls|LateInteraction`), `QuantKind` (`None|Int8Symmetric`), `ScoreStrategyKind` (`ClassifierHead|YesNoLogit`), and the umbrella `ModelSpec` (`id`, `role`, `source`, `capabilities`, and an `enum`-typed `role_data`). `EmbedderFingerprint::compute(&EmbedderSpec) -> String` (id+dims+pooling+quant → xxh64). `ModelSpec::validate(&self) -> Result<()>` (named-field errors). All serde-derived for `models.toml`. Pure; no I/O.

- **Create `crates/semantex-core/src/model/capabilities.rs`** — `ModelCapabilities` (`multi_vector`, `matryoshka_dims: Option<Vec<usize>>`, `produces_sparse`, `instruction_aware`, `max_batch`) with serde defaults so a partial `models.toml` entry keeps working. `BackendKind` (`ColbertPlaid|CoderankHnsw`) + `pub fn backend_for(caps: &ModelCapabilities) -> BackendKind` — the capability negotiation (`multi_vector → ColbertPlaid`, else `CoderankHnsw`) and `BackendKind::dense_kind(self) -> crate::search::dense_backend::DenseBackendKind` (the one place S8 touches S1's enum). Pure.

- **Create `crates/semantex-core/src/model/manifest.rs`** — `pub fn builtin_specs() -> Vec<ModelSpec>` (compiled-in permissive defaults: `coderank-137m` embedder, `lateon-colbert` embedder, `bge-reranker-v2-m3`, `qwen3-reranker-0.6b`, and — feature-gated — the LLM specs). `pub fn load_user_manifest(path: &Path) -> Result<Vec<ModelSpec>>` (parse `models.toml`; clear error naming the file + offending field). `pub fn user_manifest_path(config: &SemantexConfig, project_path: Option<&Path>) -> Option<PathBuf>` (project `.semantex/models.toml` overrides `~/.semantex/models.toml`). `pub fn merge(builtin, user) -> Vec<ModelSpec>` (override-by-id; user wins). Built-ins only — every default is MIT/Apache (CLAUDE.md).

- **Create `crates/semantex-core/src/model/registry.rs`** — `ModelRegistry`: `from_config(config, project_path) -> Result<Self>` (merge builtin+user, validate all), `resolve(role, id) -> Result<&ModelSpec>`, `active_embedder(&self) -> Result<&ModelSpec>` / `active_reranker` / `active_llm` (read the id from `SemantexConfig`, fall back to the role's default id), `embedder_backend_kind(&self) -> Result<DenseBackendKind>` (resolve active embedder → capabilities → `BackendKind::dense_kind`). Lazy caching is the merged-`Vec` + `OnceLock` discipline (the registry itself is cheap; model *weights* stay lazy in the existing embedder/reranker singletons).

- **Modify `crates/semantex-core/src/lib.rs`** — add `pub mod model;` after `pub mod index;` (alphabetical-ish).

- **Modify `crates/semantex-core/Cargo.toml`** — add `toml = "0.8"` to `[dependencies]` (pure-Rust, no C deps). Default build stays zero-genai.

- **Modify `crates/semantex-core/src/config.rs`** — add `pub embedder: String` (default `"lateon-colbert"` — the embedder id that capability-routes to S1's `colbert-plaid` backend default; D4 keeps the PLAID dense path as the shipped default until the Phase-3 cutover flips it to `"coderank-137m"`), `pub reranker_model: String` (default `"bge-reranker-v2-m3"`), `pub llm_model: String` (default `""`) to `SemantexConfig` + `Default` + the `SEMANTEX_EMBEDDER` / `SEMANTEX_RERANKER_MODEL` / `SEMANTEX_LLM_MODEL` env overlays in `load()`. (These are registry lookup keys — see §"Reconciliation". Embedder spec ids are model-descriptive (`coderank-137m`, `lateon-colbert`) and are distinct from backend names (`coderank-hnsw`, `colbert-plaid`) — the registry maps embedder→backend via capabilities.)

- **Modify `crates/semantex-core/src/types.rs`** — add `pub embedder_fingerprint: String` to `IndexMeta`; extend the doc comment (no schema bump beyond S1/S2's 11). Update the in-file `IndexMeta` test literals.

- **Modify `crates/semantex-core/src/index/state.rs`** — add `"embedder_fingerprint": "…"` to the `test_ready` JSON literal and `embedder_fingerprint: …` to the `test_pre_v0_3_schema_v7_is_stale` struct literal so they compile against the new field.

- **Modify `crates/semantex-core/src/index/builder.rs`** — write the active embedder's `EmbedderFingerprint` into `meta.json`; build the dense index under the versioned `dense/<backend>/<fingerprint>/` dir; flip the active pointer atomically on completion (Tasks 9-10).

- **Modify `crates/semantex-core/src/search/dense_backend.rs`** — add `pub fn active_dense_dir(index_dir, backend, fingerprint) -> PathBuf` (the versioned `<index_dir>/dense/<backend>/<fingerprint>` path) + `pub fn read_active_pointer` / `write_active_pointer` (the atomic `dense/<backend>/ACTIVE` file). (S1 owns this file; S8 appends — coordinate per §"Sequencing".)

- **Create `crates/semantex-core/tests/model_registry_golden_test.rs`** — the spec §6 golden gate: a registry-resolved `lateon-colbert` / `coderank-137m` embedder spec produces the **same** `DenseBackendKind` (and, where the backend is constructible without a model download, the same on-disk dir + fingerprint) as direct construction; a user `models.toml` adds a 2nd permissive embedder that resolves + capability-routes end-to-end. Repo-agnostic (tempdir).

No new workspace crate dependencies beyond `toml`. `serde`, `serde_json`, `xxhash-rust`, `anyhow` are already present.

---

## Sequencing & co-design with S1/S2/S3 (read before starting)

S8 lands in **Phase 1 alongside S1** (integration doc §2). S1 owns the `DenseBackend` *trait* + `DenseBackendKind` enum + on-disk subdir layout; **S8 owns model *selection* (spec → backend resolution) and the embedder fingerprint / versioned switchover.** Concretely:

- **Tasks 1-5 are pure additions** (new `model/` module, `toml` dep, config fields) with **zero** dependency on S1/S2/S3 landing — they can run before or in parallel with S1. They introduce `BackendKind` (S8's own enum) and only *map* to `DenseBackendKind` in `capabilities.rs::dense_kind`, which is `#[cfg]`-free and compiles the moment S1's `dense_backend.rs` defines `DenseBackendKind` (S1 Task 1). **If S1 has not landed `DenseBackendKind` yet, Task 5's `dense_kind` mapping is the single line to defer** — gate it behind a `// TODO(S1): …` note is NOT allowed (no TODOs per the plan rules); instead, sequence Task 5 to run after S1 Task 1. The integration lead serializes this.
- **Tasks 6-10 (fingerprint + versioned switchover) require S1's `dense_subdir` + `IndexMeta.dense_backend` + the schema bump.** Run them after S1 Tasks 1-4 and S2's builder arm land. They ADD `active_dense_dir` / pointer helpers next to `dense_subdir` and stamp `embedder_fingerprint` in the builder's existing meta write.
- **Tasks 11-12 (reconciliation) are wiring deltas** on S1's `hybrid.rs`/`builder.rs` selection and S3's reranker selection; they replace `DenseBackendKind::parse(&config.dense_backend)` call-sites with `registry.embedder_backend_kind()` and route `SEMANTEX_RERANKER_MODEL` through the registry. Run last, after S1/S2/S3 land, so there is no merge thrash on `hybrid.rs`.
- The **golden test (Task 13)** asserts registry-resolved == direct construction — the spec §6 anti-regression guarantee.

Commit after every task. The suite stays green throughout (Tasks 1-5 are additive; 6-13 each gate on their own test).

---

### Task 1: add the `toml` dependency

**Files:**
- Modify: `crates/semantex-core/Cargo.toml`
- Create: `crates/semantex-core/tests/toml_dep_smoke.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/semantex-core/tests/toml_dep_smoke.rs`:

```rust
//! Smoke test proving `toml` is a usable direct dependency of semantex-core.
//! The model registry (`model/manifest.rs`) parses a user `models.toml`; this
//! confirms the dep resolves and the `from_str` API surface we rely on exists.

#[test]
fn toml_parses_a_table() {
    // Compiling this is half the test: an undeclared crate fails to build.
    let parsed: toml::Value = toml::from_str(r#"
        [[model]]
        id = "demo"
    "#)
    .expect("toml must parse a simple table array");
    let models = parsed.get("model").and_then(toml::Value::as_array).unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].get("id").and_then(toml::Value::as_str), Some("demo"));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p semantex-core --test toml_dep_smoke 2>&1 | tail -20`
Expected: FAIL — compile error `use of undeclared crate or module \`toml\`` (it is not yet a direct dependency).

- [ ] **Step 3: Add the dependency**

Edit `crates/semantex-core/Cargo.toml`, in `[dependencies]` after the `# Config` block (`serde_yml = "0.0.12"`):

```toml
# TOML parser for the user model manifest (model/manifest.rs `models.toml`).
# Pure Rust, no C/sys deps (verified via `cargo tree`) — keeps the airgap/offline
# build clean. We only use `toml::from_str` / serde-derive deserialization.
toml = "0.8"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core --test toml_dep_smoke 2>&1 | tail -20`
Expected: PASS — `test toml_parses_a_table ... ok`.

Also confirm the default build still has zero LLM deps:

Run: `cargo tree -p semantex-core 2>/dev/null | grep -c genai`
Expected: `0`.

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/Cargo.toml Cargo.lock crates/semantex-core/tests/toml_dep_smoke.rs
git commit -m "feat(model): add toml dependency for the user model manifest (S8)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `ModelSpec` / `ModelRole` / `ModelSource` + role-data scaffolding

**Files:**
- Create: `crates/semantex-core/src/model/mod.rs`
- Create: `crates/semantex-core/src/model/spec.rs`
- Modify: `crates/semantex-core/src/lib.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/semantex-core/src/model/spec.rs` with ONLY the test module first (the impl follows in Step 3):

```rust
//! `ModelSpec` — every model declared as DATA, never code.
//!
//! A spec carries the model's identity, where to fetch it, its capabilities,
//! and the role-specific nuance that would otherwise be hardcoded in engine
//! code (embedder dims/prefix/pooling/quant; reranker score strategy + prompt;
//! llm provider/model/endpoint). The engine reads these fields — it never
//! special-cases a model id. This is what keeps quality from being flattened to
//! a lowest common denominator. See the SOTA overhaul design spec §4 S8 / §2 D9.

use crate::model::capabilities::ModelCapabilities;
use anyhow::Result;
use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_round_trips_through_toml() {
        for (role, s) in [
            (ModelRole::Embedder, "embedder"),
            (ModelRole::Reranker, "reranker"),
            (ModelRole::Llm, "llm"),
        ] {
            // serde rename = lowercase; round-trips via a tiny TOML doc.
            let doc = format!("role = \"{s}\"\n");
            let parsed: RoleHolder = toml::from_str(&doc).unwrap();
            assert_eq!(parsed.role, role);
        }
    }

    #[test]
    fn source_hf_round_trips() {
        let doc = r#"
            kind = "hf"
            repo = "owner/Model-onnx-int8"
            files = ["model_int8.onnx", "tokenizer.json"]
        "#;
        let src: ModelSource = toml::from_str(doc).unwrap();
        match src {
            ModelSource::Hf { repo, files } => {
                assert_eq!(repo, "owner/Model-onnx-int8");
                assert_eq!(files, vec!["model_int8.onnx", "tokenizer.json"]);
            }
            other => panic!("expected Hf, got {other:?}"),
        }
    }

    #[test]
    fn embedder_spec_carries_every_nuance() {
        let e = EmbedderSpec {
            dims: 768,
            max_context: 8192,
            query_prefix: "Represent this query for searching relevant code: ".to_string(),
            doc_prefix: String::new(),
            pooling: Pooling::Cls,
            normalize: true,
            quant: QuantKind::Int8Symmetric,
        };
        // The fields are data, not behavior — a spec is fully described here.
        assert_eq!(e.dims, 768);
        assert_eq!(e.pooling, Pooling::Cls);
        assert_eq!(e.quant, QuantKind::Int8Symmetric);
        assert!(e.query_prefix.ends_with(' '));
    }

    /// Helper struct so the role test can deserialize a bare `role = "…"`.
    #[derive(Deserialize)]
    struct RoleHolder {
        role: ModelRole,
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Create `crates/semantex-core/src/model/mod.rs`:

```rust
//! Config-driven model registry (S8). Every model — embedder, reranker, LLM —
//! is declared as DATA (`ModelSpec` + `ModelCapabilities`) and resolved by role
//! through `ModelRegistry` from compiled-in permissive defaults plus an optional
//! user `models.toml`. Swapping a model is a config change, not a recompile.
//!
//! See `docs/superpowers/specs/2026-05-31-semantex-sota-overhaul-design.md` §4 S8.

pub mod capabilities;
pub mod manifest;
pub mod registry;
pub mod spec;

pub use capabilities::{BackendKind, ModelCapabilities};
pub use registry::ModelRegistry;
pub use spec::{
    EmbedderFingerprint, EmbedderSpec, LlmSpec, ModelRole, ModelSource, ModelSpec, Pooling,
    QuantKind, RerankerSpec, RoleData, ScoreStrategyKind,
};
```

Create stub files so the module tree compiles (the real bodies land in later tasks):

`crates/semantex-core/src/model/capabilities.rs`:
```rust
//! `ModelCapabilities` + capability→backend negotiation (filled in Task 5).
```
`crates/semantex-core/src/model/manifest.rs`:
```rust
//! Built-in permissive specs + user `models.toml` merge (filled in Task 3/4).
```
`crates/semantex-core/src/model/registry.rs`:
```rust
//! `ModelRegistry` — role→spec resolution (filled in Task 7).
```

Add to `crates/semantex-core/src/lib.rs` after `pub mod index;`:

```rust
pub mod model;
```

Run: `cargo test -p semantex-core model::spec::tests 2>&1 | tail -25`
Expected: FAIL — `cannot find type 'ModelRole'` / `ModelSource` / `EmbedderSpec` / `Pooling` / `QuantKind` (and unresolved re-exports in `mod.rs`).

- [ ] **Step 3: Write minimal implementation**

Add to `crates/semantex-core/src/model/spec.rs`, ABOVE the test module:

```rust
/// Which pipeline stage a model serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelRole {
    /// Dense embedder (the dense search channel).
    Embedder,
    /// Cross-encoder reranker (final precision stage).
    Reranker,
    /// LLM for query understanding / HyDE (feature `llm` only).
    Llm,
}

/// Where a model's files are fetched from. Carried as DATA so a new model needs
/// no code — only a manifest entry. The actual download is done by the existing
/// `embedding/*_model.rs` provisioners, fed these coordinates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum ModelSource {
    /// HuggingFace repo (`<owner>/<repo>`); `files` are the resolve-path leaves.
    Hf { repo: String, files: Vec<String> },
    /// A pre-provisioned local directory (airgap / hand-placed weights).
    Local { dir: String },
    /// Arbitrary HTTPS base URL + file leaves (self-hosted mirror).
    Url { base: String, files: Vec<String> },
}

/// Token pooling for an embedder (how token vectors collapse to one vector).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Pooling {
    /// Mean over the attention mask.
    Mean,
    /// CLS token (position 0).
    Cls,
    /// No pooling — per-token vectors (ColBERT / PLAID MaxSim).
    LateInteraction,
}

/// On-disk vector quantization for an embedder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantKind {
    /// Full-precision f32 vectors.
    None,
    /// Symmetric per-vector int8 (zero-point 0) — integration doc D-int8.
    Int8Symmetric,
}

/// How a reranker turns model output into a relevance score (mirrors S3's
/// `onnx_reranker::ScoreStrategy`, but as manifest data).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreStrategyKind {
    /// bge-style sequence-classification: single relevance logit.
    ClassifierHead,
    /// Qwen3-Reranker-style generative: logit of the "yes" token.
    YesNoLogit,
}

/// Embedder-specific nuance. EVERY field that would otherwise be a hardcoded
/// constant in `embedding/*.rs` lives here as data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbedderSpec {
    /// Output embedding dimension (e.g. 768 for CodeRankEmbed, 48/token ColBERT).
    pub dims: usize,
    /// Max context tokens the encoder accepts (inputs truncate to this).
    pub max_context: usize,
    /// Prefix prepended to QUERIES (e.g. CodeRankEmbed's instruction). May be "".
    #[serde(default)]
    pub query_prefix: String,
    /// Prefix prepended to DOCUMENTS. Usually "".
    #[serde(default)]
    pub doc_prefix: String,
    /// Token pooling.
    pub pooling: Pooling,
    /// Whether to L2-normalize the pooled vector.
    #[serde(default = "default_true")]
    pub normalize: bool,
    /// On-disk vector quantization.
    pub quant: QuantKind,
}

/// Reranker-specific nuance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RerankerSpec {
    /// Score-extraction strategy.
    pub score_strategy: ScoreStrategyKind,
    /// Prompt template for `YesNoLogit` rerankers (prefix/middle/suffix around
    /// the query+doc). Ignored for `ClassifierHead`.
    #[serde(default)]
    pub prompt_prefix: String,
    #[serde(default)]
    pub prompt_middle: String,
    #[serde(default)]
    pub prompt_suffix: String,
    /// Token id of "yes" for `YesNoLogit`. `None` for `ClassifierHead`.
    #[serde(default)]
    pub yes_token_id: Option<usize>,
    /// Token id of "no" for `YesNoLogit`.
    #[serde(default)]
    pub no_token_id: Option<usize>,
}

/// LLM-specific nuance (delegated to genai when the `llm` feature is on). Held
/// as plain strings so the default (no-`llm`) build still parses a manifest that
/// happens to contain an llm entry — it just never instantiates it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmSpec {
    /// genai provider (e.g. "anthropic", "ollama"). Maps to SEMANTEX_LLM_PROVIDER.
    #[serde(default)]
    pub provider: String,
    /// Model id passed to genai. Maps to SEMANTEX_LLM_MODEL.
    pub model: String,
    /// Optional endpoint override (Ollama / self-hosted). Maps to SEMANTEX_LLM_ENDPOINT.
    #[serde(default)]
    pub endpoint: String,
}

/// The role-specific payload of a [`ModelSpec`]. Exactly one variant matches
/// `ModelSpec::role`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum RoleData {
    Embedder(EmbedderSpec),
    Reranker(RerankerSpec),
    Llm(LlmSpec),
}

/// A fully-declared model. The umbrella type the registry stores and resolves.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSpec {
    /// Stable logical id used by config selection (e.g. `"coderank-137m"`,
    /// `"bge-reranker-v2-m3"`). Distinct from the dense BACKEND name
    /// (`coderank-hnsw`/`colbert-plaid`) the embedder routes to via capabilities.
    pub id: String,
    /// Pipeline stage. MUST agree with `role_data`'s variant.
    pub role: ModelRole,
    /// Where to fetch the model files.
    pub source: ModelSource,
    /// Engine-negotiated capabilities (defaults fill missing manifest fields).
    #[serde(default)]
    pub capabilities: ModelCapabilities,
    /// Role-specific nuance (flattened so a manifest entry is one table).
    #[serde(flatten)]
    pub role_data: RoleData,
}

fn default_true() -> bool {
    true
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core model::spec::tests 2>&1 | tail -25`
Expected: FAIL — `ModelCapabilities` is still a stub (Task 5 fills it). To unblock the spec tests now, the `#[serde(default)] pub capabilities: ModelCapabilities` field needs `ModelCapabilities: Default + Deserialize`. Add a **minimal** placeholder to `capabilities.rs` so this task's tests pass; Task 5 replaces it with the full impl + negotiation:

```rust
//! `ModelCapabilities` + capability→backend negotiation.
use serde::{Deserialize, Serialize};

/// Engine-negotiable model capabilities. Missing manifest fields default to the
/// conservative single-vector profile so an unknown future capability is OFF for
/// existing models (they keep working). Task 5 adds the negotiation fn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ModelCapabilities {
    /// `true` → per-token vectors (ColBERT/PLAID MaxSim); `false` → single-vector.
    #[serde(default)]
    pub multi_vector: bool,
}
```

Re-run: `cargo test -p semantex-core model::spec::tests 2>&1 | tail -25`
Expected: PASS — `role_round_trips_through_toml`, `source_hf_round_trips`, `embedder_spec_carries_every_nuance` pass; crate compiles.

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/model/ crates/semantex-core/src/lib.rs
git commit -m "feat(model): ModelSpec/ModelRole/ModelSource + role-data structs (S8)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: built-in permissive specs (`manifest::builtin_specs`)

The compiled-in defaults: the two embedders (`coderank-137m` single-vector, `lateon-colbert` late-interaction), the two rerankers (`bge-reranker-v2-m3`, `qwen3-reranker-0.6b`). LLM specs are added feature-gated in Task 12. **Every default is MIT/Apache** (CLAUDE.md permissive-only). The embedder dims/prefix/pooling quote S2's recorded values; the reranker fields quote S3's recorded values. Embedder spec ids are model-descriptive (`coderank-137m`, `lateon-colbert`) and are distinct from backend names (`coderank-hnsw`, `colbert-plaid`) — the registry maps embedder→backend via capabilities.

**Files:**
- Modify: `crates/semantex-core/src/model/manifest.rs`

- [ ] **Step 1: Write the failing test**

Replace the stub `crates/semantex-core/src/model/manifest.rs` with the header + test module first:

```rust
//! Built-in permissive model specs + user `models.toml` merge.
//!
//! Built-ins are MIT/Apache only (CLAUDE.md permissive-only rule). They encode
//! the model nuances S2/S3 recorded (CodeRankEmbed dim/prefix/pooling;
//! Qwen3-Reranker yes/no template) as DATA so the engine never special-cases an
//! id. A user `models.toml` (project `.semantex/` over `~/.semantex/`) adds or
//! overrides specs by id with no code change.

use crate::model::spec::{ModelRole, ModelSpec};
use anyhow::Result;
use std::path::{Path, PathBuf};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::spec::{Pooling, QuantKind, RoleData, ScoreStrategyKind};

    #[test]
    fn builtins_include_both_embedders_and_both_rerankers() {
        let specs = builtin_specs();
        let ids: Vec<&str> = specs.iter().map(|s| s.id.as_str()).collect();
        assert!(ids.contains(&"coderank-137m"), "missing coderank-137m: {ids:?}");
        assert!(ids.contains(&"lateon-colbert"), "missing lateon-colbert: {ids:?}");
        assert!(ids.contains(&"bge-reranker-v2-m3"), "missing bge: {ids:?}");
        assert!(ids.contains(&"qwen3-reranker-0.6b"), "missing qwen3: {ids:?}");
    }

    #[test]
    fn coderank_embedder_carries_recorded_nuance() {
        let s = builtin_specs()
            .into_iter()
            .find(|s| s.id == "coderank-137m")
            .unwrap();
        assert_eq!(s.role, ModelRole::Embedder);
        let RoleData::Embedder(e) = &s.role_data else {
            panic!("coderank-137m must be an embedder");
        };
        // S2 Spike 1 recorded values (arctic-embed-m-long base).
        assert_eq!(e.dims, 768);
        assert_eq!(e.pooling, Pooling::Cls);
        assert_eq!(e.quant, QuantKind::Int8Symmetric);
        assert!(e.query_prefix.ends_with(' '), "recorded prefix keeps trailing space");
        assert!(e.doc_prefix.is_empty(), "documents get no prefix");
        // Single-vector → not multi_vector.
        assert!(!s.capabilities.multi_vector);
    }

    #[test]
    fn colbert_embedder_is_multi_vector_late_interaction() {
        let s = builtin_specs()
            .into_iter()
            .find(|s| s.id == "lateon-colbert")
            .unwrap();
        let RoleData::Embedder(e) = &s.role_data else {
            panic!("lateon-colbert must be an embedder");
        };
        assert_eq!(e.pooling, Pooling::LateInteraction);
        assert!(s.capabilities.multi_vector, "ColBERT is multi-vector");
    }

    #[test]
    fn qwen3_reranker_is_yes_no_with_template() {
        let s = builtin_specs()
            .into_iter()
            .find(|s| s.id == "qwen3-reranker-0.6b")
            .unwrap();
        assert_eq!(s.role, ModelRole::Reranker);
        let RoleData::Reranker(r) = &s.role_data else {
            panic!("qwen3 must be a reranker");
        };
        assert_eq!(r.score_strategy, ScoreStrategyKind::YesNoLogit);
        // YesNoLogit rerankers MUST carry yes/no token ids (filled from the spike).
        assert!(r.yes_token_id.is_some());
        assert!(r.no_token_id.is_some());
    }

    #[test]
    fn bge_reranker_is_classifier_head() {
        let s = builtin_specs()
            .into_iter()
            .find(|s| s.id == "bge-reranker-v2-m3")
            .unwrap();
        let RoleData::Reranker(r) = &s.role_data else {
            panic!("bge must be a reranker");
        };
        assert_eq!(r.score_strategy, ScoreStrategyKind::ClassifierHead);
    }

    #[test]
    fn all_builtins_validate() {
        for s in builtin_specs() {
            s.validate().unwrap_or_else(|e| panic!("builtin {} invalid: {e}", s.id));
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p semantex-core model::manifest::tests 2>&1 | tail -25`
Expected: FAIL — `cannot find function 'builtin_specs'` and `no method named 'validate'` on `ModelSpec`.

- [ ] **Step 3: Write minimal implementation**

First add `ModelSpec::validate` to `spec.rs` (after the struct, before the test module):

```rust
impl ModelSpec {
    /// Validate internal consistency: `role` agrees with `role_data`, ids are
    /// non-empty, sources name at least one file, and role-specific invariants
    /// hold (e.g. a `YesNoLogit` reranker carries both token ids). Errors NAME
    /// the offending field so a bad `models.toml` is actionable (risk row
    /// "model-manifest misconfiguration").
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(!self.id.trim().is_empty(), "model spec has an empty `id`");
        let role_ok = matches!(
            (self.role, &self.role_data),
            (ModelRole::Embedder, RoleData::Embedder(_))
                | (ModelRole::Reranker, RoleData::Reranker(_))
                | (ModelRole::Llm, RoleData::Llm(_))
        );
        anyhow::ensure!(
            role_ok,
            "model `{}`: `role` {:?} disagrees with its role data",
            self.id,
            self.role
        );
        match &self.source {
            ModelSource::Hf { repo, files } => {
                anyhow::ensure!(!repo.trim().is_empty(), "model `{}`: empty hf `repo`", self.id);
                anyhow::ensure!(!files.is_empty(), "model `{}`: hf source lists no `files`", self.id);
            }
            ModelSource::Local { dir } => {
                anyhow::ensure!(!dir.trim().is_empty(), "model `{}`: empty local `dir`", self.id);
            }
            ModelSource::Url { base, files } => {
                anyhow::ensure!(!base.trim().is_empty(), "model `{}`: empty url `base`", self.id);
                anyhow::ensure!(!files.is_empty(), "model `{}`: url source lists no `files`", self.id);
            }
        }
        if let RoleData::Embedder(e) = &self.role_data {
            anyhow::ensure!(e.dims > 0, "model `{}`: embedder `dims` must be > 0", self.id);
            anyhow::ensure!(
                e.max_context > 0,
                "model `{}`: embedder `max_context` must be > 0",
                self.id
            );
        }
        if let RoleData::Reranker(r) = &self.role_data {
            if matches!(r.score_strategy, ScoreStrategyKind::YesNoLogit) {
                anyhow::ensure!(
                    r.yes_token_id.is_some() && r.no_token_id.is_some(),
                    "model `{}`: yes_no_logit reranker needs both `yes_token_id` and `no_token_id`",
                    self.id
                );
            }
        }
        Ok(())
    }
}
```

Then add the built-in factory to `manifest.rs` (above the test module). Imports for the role structs go at the top of the file:

```rust
use crate::model::capabilities::ModelCapabilities;
use crate::model::spec::{
    EmbedderSpec, ModelSource, Pooling, QuantKind, RerankerSpec, RoleData, ScoreStrategyKind,
};

/// The compiled-in permissive default specs. MIT/Apache only.
///
/// Embedder nuance (dim/prefix/pooling/quant) quotes S2 Spike 1's recorded
/// values; reranker template/token ids quote S3 Spike 1's recorded values. These
/// are semantex's own built-ins — they are NOT repo-specific tuning (CLAUDE.md
/// rule #2). LLM specs are appended feature-gated (Task 12).
pub fn builtin_specs() -> Vec<ModelSpec> {
    let mut specs = vec![
        // ── Embedders ───────────────────────────────────────────────────────
        // CodeRankEmbed 137M (MIT) — the new single-vector default. The spec id
        // is model-descriptive (`coderank-137m`); its capabilities (single-vector)
        // route it to the `coderank-hnsw` BACKEND (Task 5 negotiation).
        ModelSpec {
            id: "coderank-137m".to_string(),
            role: ModelRole::Embedder,
            source: ModelSource::Hf {
                // RECORDED in research-notes (S2 Spike 1 Task 1 Step 4).
                repo: "MisterTK/CodeRankEmbed-onnx-int8".to_string(),
                files: vec!["model_int8.onnx".to_string(), "tokenizer.json".to_string()],
            },
            capabilities: ModelCapabilities {
                multi_vector: false,
                ..ModelCapabilities::default()
            },
            role_data: RoleData::Embedder(EmbedderSpec {
                dims: 768,
                max_context: 8192,
                // RECORDED EXACT, trailing space included.
                query_prefix: "Represent this query for searching relevant code: ".to_string(),
                doc_prefix: String::new(),
                pooling: Pooling::Cls,
                normalize: true,
                quant: QuantKind::Int8Symmetric,
            }),
        },
        // LateOn-Code-edge ColBERT — today's late-interaction path. The spec id is
        // model-descriptive (`lateon-colbert`); its capabilities (multi-vector)
        // route it to the `colbert-plaid` BACKEND.
        ModelSpec {
            id: "lateon-colbert".to_string(),
            role: ModelRole::Embedder,
            source: ModelSource::Hf {
                repo: "lightonai/LateOn-Code-edge".to_string(),
                files: vec![
                    "model_int8.onnx".to_string(),
                    "tokenizer.json".to_string(),
                    "onnx_config.json".to_string(),
                ],
            },
            capabilities: ModelCapabilities {
                multi_vector: true,
                ..ModelCapabilities::default()
            },
            role_data: RoleData::Embedder(EmbedderSpec {
                dims: 48,
                max_context: 512,
                query_prefix: String::new(),
                doc_prefix: String::new(),
                pooling: Pooling::LateInteraction,
                normalize: true,
                quant: QuantKind::Int8Symmetric,
            }),
        },
        // ── Rerankers ───────────────────────────────────────────────────────
        // bge-reranker-v2-m3 (permissive, already shipped) — classifier head.
        ModelSpec {
            id: "bge-reranker-v2-m3".to_string(),
            role: ModelRole::Reranker,
            source: ModelSource::Hf {
                repo: "BAAI/bge-reranker-v2-m3".to_string(),
                files: vec!["model_int8.onnx".to_string(), "tokenizer.json".to_string()],
            },
            capabilities: ModelCapabilities::default(),
            role_data: RoleData::Reranker(RerankerSpec {
                score_strategy: ScoreStrategyKind::ClassifierHead,
                prompt_prefix: String::new(),
                prompt_middle: String::new(),
                prompt_suffix: String::new(),
                yes_token_id: None,
                no_token_id: None,
            }),
        },
        // Qwen3-Reranker-0.6B (Apache-2.0) — yes/no generative. Token ids +
        // template are RECORDED in research-notes (S3 Spike 1). The placeholders
        // below MUST be replaced with the recorded integer ids before shipping;
        // `validate()` enforces their presence.
        ModelSpec {
            id: "qwen3-reranker-0.6b".to_string(),
            role: ModelRole::Reranker,
            source: ModelSource::Hf {
                repo: "MisterTK/Qwen3-Reranker-0.6B-onnx-int8".to_string(),
                files: vec!["model_int8.onnx".to_string(), "tokenizer.json".to_string()],
            },
            capabilities: ModelCapabilities {
                instruction_aware: true,
                ..ModelCapabilities::default()
            },
            role_data: RoleData::Reranker(RerankerSpec {
                score_strategy: ScoreStrategyKind::YesNoLogit,
                // RECORDED verbatim (S3 Spike 1, prompt template).
                prompt_prefix: "<Instruct>: Given a code search query, judge whether the document is relevant.\n<Query>: ".to_string(),
                prompt_middle: "\n<Document>: ".to_string(),
                prompt_suffix: "\n<Relevant>:".to_string(),
                // RECORDED yes/no token ids (S3 Spike 1) — replace with real ids.
                yes_token_id: Some(9693),
                no_token_id: Some(2152),
            }),
        },
    ];
    append_builtin_llm_specs(&mut specs);
    specs
}

/// Appends LLM-role built-ins. Inert (no-op) on the default build — LLM specs
/// only exist with the `llm` feature so the default build pulls zero LLM deps
/// (CLAUDE.md rule #8). Defined here so `builtin_specs` is feature-uniform.
#[cfg(not(feature = "llm"))]
fn append_builtin_llm_specs(_specs: &mut Vec<ModelSpec>) {}
```

(The `#[cfg(feature = "llm")]` version of `append_builtin_llm_specs` is added in Task 12.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core model::manifest::tests 2>&1 | tail -25`
Expected: PASS — all 6 manifest tests pass.

Also re-run the spec tests (validate now exists):

Run: `cargo test -p semantex-core model::spec::tests 2>&1 | tail -10`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/model/manifest.rs crates/semantex-core/src/model/spec.rs
git commit -m "feat(model): built-in permissive specs + ModelSpec::validate (S8)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: user `models.toml` load + merge (`manifest::load_user_manifest` / `merge` / `user_manifest_path`)

**Files:**
- Modify: `crates/semantex-core/src/model/manifest.rs`

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` block in `manifest.rs`:

```rust
    #[test]
    fn load_user_manifest_parses_a_second_embedder() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("models.toml");
        std::fs::write(
            &path,
            r#"
            [[model]]
            id = "gte-modernbert-hnsw"
            role = "embedder"
            [model.source]
            kind = "hf"
            repo = "Alibaba-NLP/gte-modernbert-base"
            files = ["model_int8.onnx", "tokenizer.json"]
            [model.capabilities]
            multi_vector = false
            [model.embedder]
            dims = 768
            max_context = 8192
            query_prefix = ""
            pooling = "cls"
            quant = "int8_symmetric"
            "#,
        )
        .unwrap();
        let specs = load_user_manifest(&path).unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].id, "gte-modernbert-hnsw");
        assert_eq!(specs[0].role, ModelRole::Embedder);
        specs[0].validate().unwrap();
    }

    #[test]
    fn load_user_manifest_errors_clearly_on_bad_spec() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("models.toml");
        // Embedder with dims=0 → validate() must reject, naming the field.
        std::fs::write(
            &path,
            r#"
            [[model]]
            id = "broken"
            role = "embedder"
            [model.source]
            kind = "hf"
            repo = "x/y"
            files = ["model_int8.onnx"]
            [model.embedder]
            dims = 0
            max_context = 8192
            pooling = "mean"
            quant = "none"
            "#,
        )
        .unwrap();
        let err = load_user_manifest(&path).expect_err("dims=0 must error");
        let msg = err.to_string();
        assert!(msg.contains("broken") && msg.contains("dims"), "got: {msg}");
    }

    #[test]
    fn merge_lets_user_override_a_builtin_by_id() {
        let builtin = builtin_specs();
        // A user spec re-using the `coderank-137m` id overrides the built-in.
        let mut overridden = builtin
            .iter()
            .find(|s| s.id == "coderank-137m")
            .cloned()
            .unwrap();
        if let RoleData::Embedder(e) = &mut overridden.role_data {
            e.max_context = 4096; // user shrinks the context window
        }
        let merged = merge(builtin.clone(), vec![overridden]);
        // Same count (override, not append).
        assert_eq!(merged.len(), builtin.len());
        let s = merged.iter().find(|s| s.id == "coderank-137m").unwrap();
        let RoleData::Embedder(e) = &s.role_data else { panic!() };
        assert_eq!(e.max_context, 4096, "user override must win");
    }

    #[test]
    fn merge_appends_a_new_user_id() {
        let builtin = builtin_specs();
        let mut newspec = builtin
            .iter()
            .find(|s| s.id == "coderank-137m")
            .cloned()
            .unwrap();
        newspec.id = "my-custom-embedder".to_string();
        let merged = merge(builtin.clone(), vec![newspec]);
        assert_eq!(merged.len(), builtin.len() + 1);
        assert!(merged.iter().any(|s| s.id == "my-custom-embedder"));
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p semantex-core model::manifest::tests::load_user 2>&1 | tail -20`
Expected: FAIL — `cannot find function 'load_user_manifest'` / `merge`.

- [ ] **Step 3: Write minimal implementation**

Add a serde wrapper + the three functions to `manifest.rs` (above the test module). The `SemantexConfig` import goes at the top of the file.

```rust
use crate::config::SemantexConfig;
use serde::Deserialize;

/// Wire shape of a `models.toml` document: a `[[model]]` array of specs.
#[derive(Debug, Deserialize)]
struct UserManifest {
    #[serde(default)]
    model: Vec<ModelSpec>,
}

/// Parse + validate a user `models.toml`. Each spec is validated; the first
/// invalid one aborts with an error naming the file and the offending field, so
/// a misconfiguration never silently mis-loads (risk row "model-manifest
/// misconfiguration").
pub fn load_user_manifest(path: &Path) -> Result<Vec<ModelSpec>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("failed to read model manifest {}: {e}", path.display()))?;
    let manifest: UserManifest = toml::from_str(&text)
        .map_err(|e| anyhow::anyhow!("failed to parse model manifest {}: {e}", path.display()))?;
    for spec in &manifest.model {
        spec.validate()
            .map_err(|e| anyhow::anyhow!("invalid model in {}: {e}", path.display()))?;
    }
    Ok(manifest.model)
}

/// Locate the active user manifest: a project-local `<project>/.semantex/models.toml`
/// takes precedence over the global `~/.semantex/models.toml`. Returns `None`
/// when neither exists (the registry then runs on built-ins only).
pub fn user_manifest_path(_config: &SemantexConfig, project_path: Option<&Path>) -> Option<PathBuf> {
    if let Some(project) = project_path {
        let local = SemantexConfig::project_index_dir(project).join("models.toml");
        if local.exists() {
            return Some(local);
        }
    }
    let global = SemantexConfig::semantex_home().join("models.toml");
    if global.exists() {
        return Some(global);
    }
    None
}

/// Merge built-in and user specs, with user specs overriding built-ins **by id**
/// (the modularity guarantee: replace a built-in's data, or add a brand-new
/// model, without touching code). Order: built-ins first (in declaration
/// order), then any user ids not already present.
#[must_use]
pub fn merge(builtin: Vec<ModelSpec>, user: Vec<ModelSpec>) -> Vec<ModelSpec> {
    let mut out = builtin;
    for u in user {
        if let Some(existing) = out.iter_mut().find(|s| s.id == u.id) {
            *existing = u; // override by id
        } else {
            out.push(u); // new id → append
        }
    }
    out
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core model::manifest::tests 2>&1 | tail -25`
Expected: PASS — all 10 manifest tests pass (the 6 from Task 3 + the 4 new).

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/model/manifest.rs
git commit -m "feat(model): user models.toml load + override-by-id merge (S8)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `ModelCapabilities` (full) + capability→backend negotiation

Replace the Task-2 placeholder `ModelCapabilities` with the full descriptor and add `BackendKind` + `backend_for` + `dense_kind`. **This is the one place S8 maps to S1's `DenseBackendKind` — sequence after S1 Task 1 lands `dense_backend.rs`.**

**Files:**
- Modify: `crates/semantex-core/src/model/capabilities.rs`

- [ ] **Step 1: Write the failing test**

Replace `crates/semantex-core/src/model/capabilities.rs` with the header + the full struct + the test module (negotiation impl follows in Step 3). Keep the `ModelCapabilities` struct from Task 2 but extend it:

```rust
//! `ModelCapabilities` + capability→backend negotiation.
//!
//! The engine NEGOTIATES against these descriptors rather than branching on a
//! model id: `multi_vector=true` routes to the PLAID/MaxSim backend, `false` to
//! single-vector/HNSW. Adding a future capability = one field here (defaulted
//! off) + one handler; existing models keep working unchanged — the "new
//! capabilities ship without an engine refactor" guarantee (design §4 S8).

use crate::search::dense_backend::DenseBackendKind;
use serde::{Deserialize, Serialize};

/// Engine-negotiable model capabilities. Every field defaults to the
/// conservative profile so a partial `models.toml` entry — or an older built-in
/// that predates a new capability — keeps working with the capability OFF.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ModelCapabilities {
    /// `true` → per-token vectors (ColBERT/PLAID MaxSim); `false` → single-vector.
    #[serde(default)]
    pub multi_vector: bool,
    /// Matryoshka truncation dims, if the model is MRL-trained (else `None` →
    /// fixed dim; do not truncate). CodeRankEmbed is NOT MRL (design §4 S2).
    #[serde(default)]
    pub matryoshka_dims: Option<Vec<usize>>,
    /// `true` → the model also emits a sparse signal (reserved; no built-in uses
    /// it yet — present so a future SPLADE-style model needs no struct change).
    #[serde(default)]
    pub produces_sparse: bool,
    /// `true` → the model is instruction-aware (e.g. takes a query prefix /
    /// reranker instruction). Informational; the prefix itself lives on the spec.
    #[serde(default)]
    pub instruction_aware: bool,
    /// Max batch the model can encode/score at once (`None` → engine default).
    #[serde(default)]
    pub max_batch: Option<usize>,
}

/// Which dense backend a model's capabilities select. S8's own enum; maps 1:1 to
/// S1's `DenseBackendKind` via [`BackendKind::dense_kind`] (the single coupling
/// point between the registry and S1's seam).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// ColBERT late-interaction + PLAID (multi-vector).
    ColbertPlaid,
    /// CodeRankEmbed single-vector + HNSW.
    CoderankHnsw,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_vector_routes_to_plaid_single_to_hnsw() {
        let mv = ModelCapabilities { multi_vector: true, ..Default::default() };
        assert_eq!(backend_for(&mv), BackendKind::ColbertPlaid);

        let sv = ModelCapabilities { multi_vector: false, ..Default::default() };
        assert_eq!(backend_for(&sv), BackendKind::CoderankHnsw);
    }

    #[test]
    fn backend_kind_maps_to_s1_dense_kind() {
        assert_eq!(
            BackendKind::ColbertPlaid.dense_kind(),
            DenseBackendKind::ColbertPlaid
        );
        // S2 adds CoderankHnsw to DenseBackendKind; this asserts the mapping.
        assert_eq!(
            BackendKind::CoderankHnsw.dense_kind().name(),
            "coderank-hnsw"
        );
    }

    #[test]
    fn capabilities_default_is_single_vector_profile() {
        let c = ModelCapabilities::default();
        assert!(!c.multi_vector);
        assert!(c.matryoshka_dims.is_none());
        assert!(!c.produces_sparse);
        assert!(c.max_batch.is_none());
    }

    #[test]
    fn partial_toml_keeps_unset_capabilities_off() {
        // A manifest that only sets multi_vector must default the rest off.
        let c: ModelCapabilities = toml::from_str("multi_vector = true\n").unwrap();
        assert!(c.multi_vector);
        assert!(!c.instruction_aware);
        assert!(c.matryoshka_dims.is_none());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p semantex-core model::capabilities::tests 2>&1 | tail -20`
Expected: FAIL — `cannot find function 'backend_for'` and `no method named 'dense_kind'` on `BackendKind`. (If S1 has not yet landed `DenseBackendKind`, the `use crate::search::dense_backend::DenseBackendKind;` line fails to resolve — that is the signal to sequence this task after S1 Task 1, per §"Sequencing".)

- [ ] **Step 3: Write minimal implementation**

Add to `capabilities.rs` (above the test module):

```rust
/// Negotiate the dense backend from capabilities alone (no model id branching).
/// `multi_vector=true` → PLAID/MaxSim; otherwise single-vector/HNSW.
#[must_use]
pub fn backend_for(caps: &ModelCapabilities) -> BackendKind {
    if caps.multi_vector {
        BackendKind::ColbertPlaid
    } else {
        BackendKind::CoderankHnsw
    }
}

impl BackendKind {
    /// Map to S1's on-disk/selection enum. The only coupling point between the
    /// registry and the `DenseBackend` seam.
    #[must_use]
    pub fn dense_kind(self) -> DenseBackendKind {
        match self {
            BackendKind::ColbertPlaid => DenseBackendKind::ColbertPlaid,
            BackendKind::CoderankHnsw => DenseBackendKind::CoderankHnsw,
        }
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core model::capabilities::tests 2>&1 | tail -20`
Expected: PASS — all 4 capability tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/model/capabilities.rs
git commit -m "feat(model): ModelCapabilities + capability->backend negotiation (S8)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: `EmbedderFingerprint` + `IndexMeta.embedder_fingerprint` field

The fingerprint is the data that generalizes the schema/stemmer/backend guards into one "index stamped with its embedder spec" check. **Requires S1's `IndexMeta.dense_backend` field + the schema bump (S1 Task 3) to have landed** so the `IndexMeta` literal compiles.

**Files:**
- Modify: `crates/semantex-core/src/model/spec.rs` (add `EmbedderFingerprint`)
- Modify: `crates/semantex-core/src/types.rs` (add the field)
- Modify: `crates/semantex-core/src/index/state.rs` (test literals)

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` block in `spec.rs`:

```rust
    #[test]
    fn fingerprint_is_stable_and_sensitive() {
        let base = EmbedderSpec {
            dims: 768,
            max_context: 8192,
            query_prefix: "Q: ".to_string(),
            doc_prefix: String::new(),
            pooling: Pooling::Cls,
            normalize: true,
            quant: QuantKind::Int8Symmetric,
        };
        let fp1 = EmbedderFingerprint::compute("coderank-137m", &base);
        let fp2 = EmbedderFingerprint::compute("coderank-137m", &base);
        assert_eq!(fp1, fp2, "same id+spec → same fingerprint (deterministic)");
        assert!(!fp1.is_empty());

        // Changing dims changes the fingerprint (vector space differs).
        let mut diff_dims = base.clone();
        diff_dims.dims = 384;
        assert_ne!(fp1, EmbedderFingerprint::compute("coderank-137m", &diff_dims));

        // Changing pooling changes it.
        let mut diff_pool = base.clone();
        diff_pool.pooling = Pooling::Mean;
        assert_ne!(fp1, EmbedderFingerprint::compute("coderank-137m", &diff_pool));

        // Changing quant changes it.
        let mut diff_quant = base.clone();
        diff_quant.quant = QuantKind::None;
        assert_ne!(fp1, EmbedderFingerprint::compute("coderank-137m", &diff_quant));

        // Changing the id changes it (different model, same shape).
        assert_ne!(fp1, EmbedderFingerprint::compute("other-embedder", &base));
    }
```

Add to `types.rs` an `IndexMeta` round-trip test (after S1's `index_meta_round_trips_dense_backend`):

```rust
    #[test]
    fn index_meta_round_trips_embedder_fingerprint() {
        let meta = IndexMeta {
            schema_version: IndexMeta::CURRENT_SCHEMA_VERSION,
            project_path: std::path::PathBuf::from("/x"),
            created_at: "0".to_string(),
            updated_at: "0".to_string(),
            file_count: 1,
            chunk_count: 2,
            embedding_model: "CodeRankEmbed".to_string(),
            embedding_dim: 768,
            use_bm25_stemmer: true,
            dense_backend: "coderank-hnsw".to_string(),
            embedder_fingerprint: "abc123".to_string(),
        };
        let json = serde_json::to_string(&meta).unwrap();
        let back: IndexMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(back.embedder_fingerprint, "abc123");
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p semantex-core model::spec::tests::fingerprint types::tests::index_meta_round_trips_embedder 2>&1 | tail -25`
Expected: FAIL — `cannot find type 'EmbedderFingerprint'` and `missing field 'embedder_fingerprint' in initializer of 'IndexMeta'`.

- [ ] **Step 3: Write minimal implementation**

Add the fingerprint type to `spec.rs` (after `ModelSpec::validate`, before the test module). Reuses the in-tree `xxhash-rust` dep (no new hashing crate):

```rust
/// A stable content fingerprint of an embedder spec, used to stamp a dense index
/// and detect an embedder change at open time. Generalizes the schema /
/// stemmer-flag / dense-backend guards into one uniform "is this index's vector
/// space the one this embedder produces?" check.
///
/// The fingerprint covers exactly the fields that change the produced vectors:
/// the model id, dims, pooling, and quantization. (A reranker/LLM swap does NOT
/// touch the dense vector space → not fingerprinted → no reindex; that is the
/// design's query-time-live-swap guarantee.)
pub struct EmbedderFingerprint;

impl EmbedderFingerprint {
    /// Compute the fingerprint string for `(id, spec)`. Deterministic across
    /// runs/platforms (xxh64 of a canonical byte encoding).
    #[must_use]
    pub fn compute(id: &str, spec: &EmbedderSpec) -> String {
        // Canonical, stable encoding of the vector-space-defining fields. Avoid
        // serde here so a future non-fingerprinted field (e.g. max_context, which
        // does NOT change the vector space) can be added to EmbedderSpec without
        // silently invalidating every index.
        let pooling = match spec.pooling {
            Pooling::Mean => "mean",
            Pooling::Cls => "cls",
            Pooling::LateInteraction => "late_interaction",
        };
        let quant = match spec.quant {
            QuantKind::None => "none",
            QuantKind::Int8Symmetric => "int8_symmetric",
        };
        let canonical = format!(
            "id={id};dims={};pooling={pooling};quant={quant};norm={};qpre={}",
            spec.dims, spec.normalize, spec.query_prefix
        );
        let hash = xxhash_rust::xxh64::xxh64(canonical.as_bytes(), 0);
        format!("{hash:016x}")
    }
}
```

Add the field to `IndexMeta` in `types.rs` (after `dense_backend: String,` that S1 added):

```rust
    /// Fingerprint of the embedder spec this dense index was built with
    /// (id+dims+pooling+quant+prefix, xxh64). Open-time code compares it to the
    /// active embedder's fingerprint; a mismatch means the vector space changed,
    /// so the index is rebuilt under a new versioned dir and the active pointer
    /// is flipped atomically (S8 — zero-downtime). An older meta.json lacking
    /// this field fails to deserialize → `state::detect` returns `Stale`.
    pub embedder_fingerprint: String,
```

Extend the `CURRENT_SCHEMA_VERSION` doc comment (already 11 after S2) with an S8 line — **no value change**:

```rust
    /// S8: adds `embedder_fingerprint`. No version bump beyond S1/S2's 11 — an
    /// older meta.json lacking the field fails the strict deserialize and is
    /// treated as `Stale` (same mechanism as the v9→v10 `dense_backend` add).
    pub const CURRENT_SCHEMA_VERSION: u32 = 11;
```

In `index/state.rs`, add `"embedder_fingerprint": "test",` to the `test_ready` JSON literal (after `"dense_backend": …` that S1 added) and `embedder_fingerprint: "test".to_string(),` to the `test_pre_v0_3_schema_v7_is_stale` struct literal (after its `dense_backend` line).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core model::spec::tests::fingerprint types:: index::state::tests 2>&1 | tail -25`
Expected: PASS — `fingerprint_is_stable_and_sensitive`, `index_meta_round_trips_embedder_fingerprint`, and all `state::tests` pass.

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/model/spec.rs crates/semantex-core/src/types.rs crates/semantex-core/src/index/state.rs
git commit -m "feat(model): EmbedderFingerprint + stamp it in IndexMeta (S8)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: `ModelRegistry` — role→spec resolution + active selection

**Files:**
- Modify: `crates/semantex-core/src/model/registry.rs`
- Modify: `crates/semantex-core/src/config.rs` (add the three selection fields — needed by the registry)

> Adds `embedder`/`reranker_model`/`llm_model` config fields here because the registry resolves the active id from `SemantexConfig`. (S1 already added `dense_backend`; these are the registry-keyed siblings.)

- [ ] **Step 1: Write the failing test**

Add the config-field tests to the `#[cfg(test)] mod tests` block in `crates/semantex-core/src/config.rs`:

```rust
    #[test]
    fn default_selection_fields() {
        let cfg = SemantexConfig::default();
        // D4: PLAID stays the shipped dense default until the Phase-3 cutover.
        assert_eq!(cfg.embedder, "lateon-colbert");
        assert_eq!(cfg.reranker_model, "bge-reranker-v2-m3");
        assert_eq!(cfg.llm_model, "", "no LLM selected by default (zero-LLM-deps build)");
    }
```

Replace the stub `crates/semantex-core/src/model/registry.rs` with the header + the test module first:

```rust
//! `ModelRegistry` — resolves the active model per role from merged specs.
//!
//! Holds built-in + user-manifest specs (validated at construction), and reads
//! the active id per role from `SemantexConfig` (`embedder` / `reranker_model` /
//! `llm_model`, each overridable by env). Resolution is cheap — the model
//! WEIGHTS stay lazy in the existing embedder/reranker singletons; the registry
//! only hands out the `ModelSpec` that tells those singletons WHAT to load.

use crate::config::SemantexConfig;
use crate::model::capabilities::{BackendKind, backend_for};
use crate::model::manifest::{builtin_specs, load_user_manifest, merge, user_manifest_path};
use crate::model::spec::{ModelRole, ModelSpec, RoleData};
use crate::search::dense_backend::DenseBackendKind;
use anyhow::Result;
use std::path::Path;

#[cfg(test)]
mod tests {
    use super::*;

    fn registry_from_builtins() -> ModelRegistry {
        // No project path, no user manifest → built-ins only.
        ModelRegistry::from_config(&SemantexConfig::default(), None).unwrap()
    }

    #[test]
    fn resolves_active_embedder_default() {
        // D4: the shipped default embedder is lateon-colbert until cutover.
        let reg = registry_from_builtins();
        let spec = reg.active_embedder().unwrap();
        assert_eq!(spec.id, "lateon-colbert");
        assert_eq!(spec.role, ModelRole::Embedder);
    }

    #[test]
    fn active_embedder_backend_kind_is_plaid_for_default() {
        // lateon-colbert is multi_vector → PLAID, the D4 default dense path.
        let reg = registry_from_builtins();
        assert_eq!(reg.embedder_backend_kind().unwrap(), DenseBackendKind::ColbertPlaid);
    }

    #[test]
    fn coderank_selection_routes_to_hnsw() {
        // Selecting the single-vector embedder (the Phase-3 cutover target)
        // routes to HNSW purely from its capabilities.
        let mut cfg = SemantexConfig::default();
        cfg.embedder = "coderank-137m".to_string();
        let reg = ModelRegistry::from_config(&cfg, None).unwrap();
        assert_eq!(reg.active_embedder().unwrap().id, "coderank-137m");
        assert_eq!(reg.embedder_backend_kind().unwrap(), DenseBackendKind::CoderankHnsw);
    }

    #[test]
    fn resolves_active_reranker_default() {
        let reg = registry_from_builtins();
        assert_eq!(reg.active_reranker().unwrap().id, "bge-reranker-v2-m3");
    }

    #[test]
    fn unknown_active_id_errors_naming_the_id_and_role() {
        let mut cfg = SemantexConfig::default();
        cfg.embedder = "does-not-exist".to_string();
        let reg = ModelRegistry::from_config(&cfg, None).unwrap();
        let err = reg.active_embedder().expect_err("unknown id must error");
        let msg = err.to_string();
        assert!(msg.contains("does-not-exist"), "got: {msg}");
        assert!(msg.contains("embedder"), "got: {msg}");
    }

    #[test]
    fn resolve_wrong_role_errors() {
        let reg = registry_from_builtins();
        // bge is a reranker; asking for it as an embedder must fail.
        let err = reg.resolve(ModelRole::Embedder, "bge-reranker-v2-m3").unwrap_err();
        assert!(err.to_string().contains("bge-reranker-v2-m3"));
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p semantex-core config::tests::default_selection model::registry::tests 2>&1 | tail -25`
Expected: FAIL — `no field 'embedder' on type 'SemantexConfig'`, `cannot find type 'ModelRegistry'`.

- [ ] **Step 3: Write minimal implementation**

In `config.rs`, add the three fields to `SemantexConfig` (after S1's `dense_backend: String`):

```rust
    /// Active embedder model id (registry lookup key). Default `"lateon-colbert"`
    /// (D4: PLAID is the shipped dense default until the Phase-3 cutover flips
    /// this to `"coderank-137m"`). The embedder spec id is model-descriptive and
    /// distinct from the dense backend name it routes to via capabilities
    /// (`lateon-colbert` → `colbert-plaid`; `coderank-137m` → `coderank-hnsw`).
    /// Override via `SEMANTEX_EMBEDDER`. A change here triggers a versioned dense
    /// rebuild + atomic switchover (S8) — the re-embedding compute is inherent.
    pub embedder: String,
    /// Active reranker model id (registry lookup key). Default
    /// `"bge-reranker-v2-m3"`. Override via `SEMANTEX_RERANKER_MODEL`. A change
    /// here is a query-time live swap — no reindex.
    pub reranker_model: String,
    /// Active LLM model id (registry lookup key). Empty = none. Override via
    /// `SEMANTEX_LLM_MODEL`. Only meaningful with the `llm` feature.
    pub llm_model: String,
```

Add the defaults in `impl Default` (after `dense_backend: "colbert-plaid".to_string(),`):

```rust
            embedder: "lateon-colbert".to_string(),
            reranker_model: "bge-reranker-v2-m3".to_string(),
            llm_model: String::new(),
```

Add the env overlays in `load()` (after S1's `SEMANTEX_DENSE_BACKEND` block) using the `env_string` helper S1 added:

```rust
        config.embedder = env_string("SEMANTEX_EMBEDDER", &config.embedder);
        config.reranker_model = env_string("SEMANTEX_RERANKER_MODEL", &config.reranker_model);
        config.llm_model = env_string("SEMANTEX_LLM_MODEL", &config.llm_model);
```

Add the registry to `registry.rs` (above the test module):

```rust
/// Resolved, validated set of model specs + the active selection from config.
pub struct ModelRegistry {
    specs: Vec<ModelSpec>,
    active_embedder_id: String,
    active_reranker_id: String,
    active_llm_id: String,
}

impl ModelRegistry {
    /// Build from `config` + an optional `project_path` (for a project-local
    /// `models.toml`). Merges built-ins with the user manifest (user overrides by
    /// id), validates every spec, and records the active id per role from config.
    pub fn from_config(config: &SemantexConfig, project_path: Option<&Path>) -> Result<Self> {
        let user = match user_manifest_path(config, project_path) {
            Some(path) => load_user_manifest(&path)?,
            None => Vec::new(),
        };
        let specs = merge(builtin_specs(), user);
        for s in &specs {
            s.validate()?;
        }
        Ok(Self {
            specs,
            active_embedder_id: config.embedder.clone(),
            active_reranker_id: config.reranker_model.clone(),
            active_llm_id: config.llm_model.clone(),
        })
    }

    /// Resolve a spec by `(role, id)`. Errors (naming id + role) if no spec has
    /// that id, or if the matched spec has a different role.
    pub fn resolve(&self, role: ModelRole, id: &str) -> Result<&ModelSpec> {
        match self.specs.iter().find(|s| s.id == id) {
            None => anyhow::bail!(
                "no model `{id}` registered for role {role:?} \
                 (check `models.toml` and the {role:?} selection env var)"
            ),
            Some(s) if s.role != role => anyhow::bail!(
                "model `{id}` is registered as {:?}, not {role:?}",
                s.role
            ),
            Some(s) => Ok(s),
        }
    }

    /// The active embedder spec (from `config.embedder` / `SEMANTEX_EMBEDDER`).
    pub fn active_embedder(&self) -> Result<&ModelSpec> {
        self.resolve(ModelRole::Embedder, &self.active_embedder_id)
    }

    /// The active reranker spec (from `config.reranker_model`).
    pub fn active_reranker(&self) -> Result<&ModelSpec> {
        self.resolve(ModelRole::Reranker, &self.active_reranker_id)
    }

    /// The active LLM spec, or `Ok(None)` when none is selected (the default,
    /// zero-LLM-deps build). Errors only if a non-empty id fails to resolve.
    pub fn active_llm(&self) -> Result<Option<&ModelSpec>> {
        if self.active_llm_id.trim().is_empty() {
            return Ok(None);
        }
        self.resolve(ModelRole::Llm, &self.active_llm_id).map(Some)
    }

    /// The dense backend the active embedder's capabilities select — the value
    /// S1's `hybrid.rs`/`builder.rs` consume in place of
    /// `DenseBackendKind::parse(&config.dense_backend)`.
    pub fn embedder_backend_kind(&self) -> Result<DenseBackendKind> {
        let spec = self.active_embedder()?;
        // Defensive: an embedder spec always carries EmbedderSpec data (validate
        // enforces role agreement), but match exhaustively to avoid a panic path.
        let _ = match &spec.role_data {
            RoleData::Embedder(_) => {}
            _ => anyhow::bail!("active embedder `{}` is not an embedder spec", spec.id),
        };
        Ok(backend_for(&spec.capabilities).dense_kind())
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core config::tests model::registry::tests 2>&1 | tail -25`
Expected: PASS — the config default test + all 6 registry tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/model/registry.rs crates/semantex-core/src/config.rs
git commit -m "feat(model): ModelRegistry role->spec resolution + active selection (S8)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: versioned dense dir + atomic active-pointer helpers

Add the path + pointer helpers that make an embedder swap zero-downtime. They live next to S1's `dense_subdir` in `dense_backend.rs`. **Requires S1 Task 1 (the file + `dense_subdir`).**

**Files:**
- Modify: `crates/semantex-core/src/search/dense_backend.rs`

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` block in `dense_backend.rs`:

```rust
    #[test]
    fn versioned_dir_nests_fingerprint_under_backend() {
        let root = Path::new("/tmp/proj/.semantex");
        let p = active_dense_dir(root, DenseBackendKind::CoderankHnsw, "deadbeef");
        assert_eq!(
            p,
            Path::new("/tmp/proj/.semantex/dense/coderank-hnsw/deadbeef")
        );
    }

    #[test]
    fn active_pointer_round_trips() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        // No pointer yet → None.
        assert_eq!(read_active_pointer(root, DenseBackendKind::CoderankHnsw), None);
        // Write then read back.
        write_active_pointer(root, DenseBackendKind::CoderankHnsw, "abc123").unwrap();
        assert_eq!(
            read_active_pointer(root, DenseBackendKind::CoderankHnsw),
            Some("abc123".to_string())
        );
        // Overwrite flips atomically.
        write_active_pointer(root, DenseBackendKind::CoderankHnsw, "def456").unwrap();
        assert_eq!(
            read_active_pointer(root, DenseBackendKind::CoderankHnsw),
            Some("def456".to_string())
        );
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p semantex-core dense_backend::tests::versioned dense_backend::tests::active_pointer 2>&1 | tail -20`
Expected: FAIL — `cannot find function 'active_dense_dir'` / `read_active_pointer` / `write_active_pointer`.

- [ ] **Step 3: Write minimal implementation**

Add to `dense_backend.rs` (after `dense_subdir`):

```rust
/// The versioned dense index dir for a specific embedder fingerprint:
/// `<index_dir>/dense/<backend>/<fingerprint>/`. A new embedder builds here
/// alongside the old one, so the live index is never disturbed mid-rebuild (S8
/// zero-downtime switchover).
pub fn active_dense_dir(index_dir: &Path, backend: DenseBackendKind, fingerprint: &str) -> PathBuf {
    dense_subdir(index_dir, backend).join(fingerprint)
}

/// Path of the active-pointer file for a backend: `<index_dir>/dense/<backend>/ACTIVE`.
/// Its contents are the fingerprint of the currently-live versioned dir.
fn active_pointer_path(index_dir: &Path, backend: DenseBackendKind) -> PathBuf {
    dense_subdir(index_dir, backend).join("ACTIVE")
}

/// Read the currently-active fingerprint for `backend`, or `None` if no pointer
/// exists yet (fresh index).
pub fn read_active_pointer(index_dir: &Path, backend: DenseBackendKind) -> Option<String> {
    std::fs::read_to_string(active_pointer_path(index_dir, backend))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Flip the active pointer to `fingerprint` atomically (write a temp file in the
/// same dir, then rename — rename is atomic on the same filesystem). The new
/// versioned dir must already be fully built before this is called, so readers
/// either see the old fingerprint or the new one, never a partial index.
pub fn write_active_pointer(
    index_dir: &Path,
    backend: DenseBackendKind,
    fingerprint: &str,
) -> Result<()> {
    let dir = dense_subdir(index_dir, backend);
    std::fs::create_dir_all(&dir)?;
    let final_path = dir.join("ACTIVE");
    let tmp_path = dir.join(".ACTIVE.tmp");
    std::fs::write(&tmp_path, fingerprint.as_bytes())?;
    std::fs::rename(&tmp_path, &final_path)?;
    Ok(())
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core dense_backend::tests 2>&1 | tail -20`
Expected: PASS — the new versioned-dir + active-pointer tests pass alongside S1's existing `dense_backend` tests.

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/search/dense_backend.rs
git commit -m "feat(search): versioned dense dir + atomic active-pointer helpers (S8)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: stamp the fingerprint + build under the versioned dir in `builder.rs`

Wire the registry into the builder so the dense index lands in `dense/<backend>/<fingerprint>/`, the active pointer flips on completion, and `meta.json` records the fingerprint. **Requires S1's builder seam (the `match backend_kind` in `builder.rs`) and S2's `CoderankHnsw` arm.**

**Files:**
- Modify: `crates/semantex-core/src/index/builder.rs`

- [ ] **Step 1: Write the failing test**

Add an integration test asserting the meta carries a non-empty fingerprint after a build. Append to `crates/semantex-core/tests/model_registry_golden_test.rs` (created in Task 13 — if running this task first, create the file with just this test and its imports; Task 13 adds the rest):

```rust
//! S8 acceptance: fingerprint stamping + registry-resolved == direct.
use semantex_core::config::SemantexConfig;
use semantex_core::index::builder::IndexBuilder;
use semantex_core::types::IndexMeta;
use std::fs;

/// Write a tiny synthetic repo so a real index can be built without network
/// (the lateon-colbert path needs the ColBERT model; gate the model-dependent
/// assertions with `#[ignore]`). This test only inspects meta.json shape.
fn write_tiny_repo(dir: &std::path::Path) {
    fs::create_dir_all(dir).unwrap();
    fs::write(
        dir.join("lib.rs"),
        "pub fn add(a: i32, b: i32) -> i32 { a + b }\n",
    )
    .unwrap();
}

#[test]
#[ignore = "builds a real dense index (needs the embedder model download)"]
fn build_stamps_a_nonempty_embedder_fingerprint() {
    let tmp = tempfile::TempDir::new().unwrap();
    let project = tmp.path().join("proj");
    write_tiny_repo(&project);
    // Build with the single-vector embedder explicitly (the fingerprint-stamping
    // path under test). The shipped DEFAULT embedder is lateon-colbert (D4) until
    // cutover; we select coderank-137m here to exercise the new dense backend.
    let mut cfg = SemantexConfig::default();
    cfg.embedder = "coderank-137m".to_string();
    IndexBuilder::new(&cfg).unwrap().build(&project).unwrap();

    let meta_str = fs::read_to_string(project.join(".semantex").join("meta.json")).unwrap();
    let meta: IndexMeta = serde_json::from_str(&meta_str).unwrap();
    assert!(
        !meta.embedder_fingerprint.is_empty(),
        "builder must stamp the embedder fingerprint"
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p semantex-core --test model_registry_golden_test -- --ignored build_stamps 2>&1 | tail -20`
Expected: FAIL — the meta's `embedder_fingerprint` is empty (builder doesn't stamp it yet) OR a compile error if `IndexMeta` literal in `builder.rs` lacks the field. (Without the model present locally the build itself errors; in that case the failing signal is the missing stamp once the model is provisioned. The unit-level proof is the `meta` construction below + Task 13's non-`#[ignore]` golden test.)

- [ ] **Step 3: Write minimal implementation**

In `builder.rs`, construct the registry once near the top of `build()` (after `config` is available) and compute the active fingerprint:

```rust
use crate::model::{ModelRegistry, EmbedderFingerprint};
use crate::model::spec::RoleData;
```

Inside `build()`, before the dense-build `match backend_kind { … }` block, resolve the active embedder spec + fingerprint from the registry (replaces reading `self.config.dense_backend` directly — see Reconciliation Task 12):

```rust
        // S8: the active embedder is resolved from the registry (built-ins +
        // user models.toml), not a bare config string. Its capabilities select
        // the dense backend; its fingerprint stamps the index + names the
        // versioned dir.
        let registry = ModelRegistry::from_config(&self.config, Some(project_path))?;
        let backend_kind = registry.embedder_backend_kind()?;
        let embedder_spec = registry.active_embedder()?;
        let RoleData::Embedder(embedder_data) = &embedder_spec.role_data else {
            anyhow::bail!("active embedder `{}` is not an embedder spec", embedder_spec.id);
        };
        let embedder_fingerprint = EmbedderFingerprint::compute(&embedder_spec.id, embedder_data);
```

In the dense-build block, build into the versioned dir and flip the pointer on success. The S1/S2 builder arms persist into a `dense_dir`; pass them the versioned dir:

```rust
        let dense_dir = crate::search::dense_backend::active_dense_dir(
            index_dir,
            backend_kind,
            &embedder_fingerprint,
        );
        // … the S1/S2 DenseIndexBuilder builds + persists into `dense_dir` …
        // On successful completion, atomically publish the new fingerprint:
        crate::search::dense_backend::write_active_pointer(
            index_dir,
            backend_kind,
            &embedder_fingerprint,
        )?;
```

In the `IndexMeta { … }` literal (lines ~862-874), set the new fields from the resolved spec (S2 already sets `embedding_model`/`embedding_dim` per backend — make them spec-driven):

```rust
            embedding_model: embedder_spec.id.clone(),
            embedding_dim: embedder_data.dims as u32,
            use_bm25_stemmer: self.config.use_bm25_stemmer,
            dense_backend: backend_kind.name().to_string(),
            embedder_fingerprint: embedder_fingerprint.clone(),
```

- [ ] **Step 4: Run test to verify it passes**

Run (with the model available locally, or in CI where the download path resolves):
`cargo test -p semantex-core --test model_registry_golden_test -- --ignored build_stamps 2>&1 | tail -20`
Expected: PASS — meta carries a non-empty fingerprint.

Also confirm the crate still builds + the full lib suite is green:
Run: `cargo test -p semantex-core 2>&1 | tail -15`
Expected: PASS (lib tests green; the `#[ignore]` integration test is skipped).

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/index/builder.rs crates/semantex-core/tests/model_registry_golden_test.rs
git commit -m "feat(index): stamp embedder fingerprint + build under versioned dir (S8)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: open-time fingerprint guard + active-dir resolution in `hybrid.rs`

When opening, resolve the dense dir from the active pointer and verify the persisted fingerprint matches the active embedder; a mismatch errors cleanly (acceptance gate 2: "mismatched index errors cleanly"). **Requires S1's `hybrid.rs` dense `open()` block + Task 7/8.**

**Files:**
- Modify: `crates/semantex-core/src/search/dense_backend.rs` (add `verify_persisted_fingerprint_matches`)
- Modify: `crates/semantex-core/src/search/hybrid.rs` (resolve versioned dir + call the guard)

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` block in `dense_backend.rs`:

```rust
    #[test]
    fn verify_fingerprint_errors_on_mismatch() {
        let tmp = tempfile::TempDir::new().unwrap();
        let index_dir = tmp.path();
        // Reuse the existing helper to write a current-shape meta with a known fp.
        write_meta_with_fingerprint(index_dir, "coderank-hnsw", "OLDFP");
        let err = verify_persisted_fingerprint_matches(index_dir, "NEWFP")
            .expect_err("fingerprint mismatch must error");
        let msg = err.to_string();
        assert!(msg.contains("embedder changed") || msg.contains("fingerprint"), "got: {msg}");
        assert!(msg.contains("OLDFP") && msg.contains("NEWFP"), "got: {msg}");
    }

    #[test]
    fn verify_fingerprint_ok_on_match() {
        let tmp = tempfile::TempDir::new().unwrap();
        write_meta_with_fingerprint(tmp.path(), "coderank-hnsw", "SAME");
        verify_persisted_fingerprint_matches(tmp.path(), "SAME").unwrap();
    }

    #[test]
    fn verify_fingerprint_skips_when_meta_missing() {
        let tmp = tempfile::TempDir::new().unwrap();
        verify_persisted_fingerprint_matches(tmp.path(), "anything").unwrap();
    }

    /// Helper: write a current-shape meta.json carrying `backend` + `fingerprint`.
    fn write_meta_with_fingerprint(index_dir: &Path, backend: &str, fingerprint: &str) {
        let meta = crate::types::IndexMeta {
            schema_version: crate::types::IndexMeta::CURRENT_SCHEMA_VERSION,
            project_path: index_dir.to_path_buf(),
            created_at: "0".to_string(),
            updated_at: "0".to_string(),
            file_count: 0,
            chunk_count: 0,
            embedding_model: "test".to_string(),
            embedding_dim: 768,
            use_bm25_stemmer: true,
            dense_backend: backend.to_string(),
            embedder_fingerprint: fingerprint.to_string(),
        };
        std::fs::write(index_dir.join("meta.json"), serde_json::to_string(&meta).unwrap()).unwrap();
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p semantex-core dense_backend::tests::verify_fingerprint 2>&1 | tail -20`
Expected: FAIL — `cannot find function 'verify_persisted_fingerprint_matches'`.

- [ ] **Step 3: Write minimal implementation**

Add the guard to `dense_backend.rs` (next to `verify_persisted_backend_matches`):

```rust
/// Verify the persisted `embedder_fingerprint` in `<index_dir>/meta.json` matches
/// `expected`. Mirrors `verify_persisted_backend_matches`: `Ok(())` when meta is
/// missing/unparseable (production callers reach here only after `state::detect`
/// vetted meta), `Err` on a value mismatch — pointing the user at a rebuild.
/// This is the uniform "index stamped with its embedder; mismatch → rebuild"
/// check that generalizes the schema/stemmer/backend guards (S8).
pub fn verify_persisted_fingerprint_matches(index_dir: &Path, expected: &str) -> Result<()> {
    let meta_path = index_dir.join("meta.json");
    let Ok(meta_str) = std::fs::read_to_string(&meta_path) else {
        return Ok(());
    };
    let Ok(meta) = serde_json::from_str::<crate::types::IndexMeta>(&meta_str) else {
        return Ok(());
    };
    if meta.embedder_fingerprint != expected {
        anyhow::bail!(
            "embedder changed: index built with embedder_fingerprint={}, \
             config's active embedder has fingerprint={}. Run \
             `semantex index --rebuild` to re-embed under the new model.",
            meta.embedder_fingerprint,
            expected,
        );
    }
    Ok(())
}
```

In `hybrid.rs`, in the dense-`open()` block (S1's `match backend_kind` arm), resolve the versioned dir from the active pointer and call the guard before opening the backend. Construct the registry from `config` + the project (the index dir's parent):

```rust
        let registry = crate::model::ModelRegistry::from_config(config, index_dir.parent())?;
        let backend_kind = registry.embedder_backend_kind()?;
        let embedder_spec = registry.active_embedder()?;
        let crate::model::spec::RoleData::Embedder(embedder_data) = &embedder_spec.role_data else {
            anyhow::bail!("active embedder `{}` is not an embedder spec", embedder_spec.id);
        };
        let active_fp = crate::model::EmbedderFingerprint::compute(&embedder_spec.id, embedder_data);
        // Clean error if the on-disk index was built with a different embedder.
        crate::search::dense_backend::verify_persisted_fingerprint_matches(index_dir, &active_fp)?;
        // Resolve the live versioned dir; fall back to the active pointer on disk
        // (which equals `active_fp` once the guard above passed).
        let dense_dir = crate::search::dense_backend::active_dense_dir(
            index_dir,
            backend_kind,
            &active_fp,
        );
        // … S1/S2 open the DenseBackend from `dense_dir` …
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core dense_backend::tests 2>&1 | tail -20`
Expected: PASS — the three fingerprint-guard tests pass.

Confirm the lib suite is green:
Run: `cargo test -p semantex-core 2>&1 | tail -15`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/search/dense_backend.rs crates/semantex-core/src/search/hybrid.rs
git commit -m "feat(search): open-time embedder-fingerprint guard + versioned dir resolve (S8)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: feature-gated LLM built-in specs + `active_llm` gating

LLM specs only compile + register with the `llm` feature; the default build stays zero-LLM-deps. **No spec is hardcoded into production beyond what the user selects** — these built-ins are inert until `SEMANTEX_LLM_MODEL` names one (CLAUDE.md rules #6/#8).

**Files:**
- Modify: `crates/semantex-core/src/model/manifest.rs` (the `#[cfg(feature = "llm")]` `append_builtin_llm_specs`)
- Modify: `crates/semantex-core/src/model/registry.rs` (the `llm`-gated test)

- [ ] **Step 1: Write the failing test**

Add a default-build test to `registry.rs`'s test module (asserts no LLM is active by default and that `active_llm` is `None`):

```rust
    #[test]
    fn active_llm_is_none_by_default() {
        let reg = registry_from_builtins();
        assert!(reg.active_llm().unwrap().is_none(), "no LLM selected by default");
    }
```

Add an `llm`-gated test (only compiled with the feature) at the end of the test module:

```rust
    #[cfg(feature = "llm")]
    #[test]
    fn llm_builtin_resolves_when_feature_on_and_selected() {
        let mut cfg = SemantexConfig::default();
        cfg.llm_model = "ollama-default".to_string();
        let reg = ModelRegistry::from_config(&cfg, None).unwrap();
        let spec = reg.active_llm().unwrap().expect("llm spec should resolve");
        assert_eq!(spec.id, "ollama-default");
        assert_eq!(spec.role, ModelRole::Llm);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Default build:
Run: `cargo test -p semantex-core model::registry::tests::active_llm_is_none 2>&1 | tail -10`
Expected: PASS already (no LLM built-ins, `active_llm` returns `None`) — this guards the zero-deps default.

LLM build:
Run: `cargo test -p semantex-core --features llm model::registry::tests::llm_builtin 2>&1 | tail -15`
Expected: FAIL — `no model 'ollama-default' registered for role Llm` (the feature-gated built-in doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

Add the feature-gated built-in factory to `manifest.rs` (next to the `#[cfg(not(feature = "llm"))]` stub from Task 3). It carries NO hardcoded provider/model as a *default selection* — these are merely available specs, inert unless `SEMANTEX_LLM_MODEL` selects one. Genai/endpoint strings are configuration the user opts into (CLAUDE.md rule #6 allows env-sourced LLM config; here they are manifest data the user can override via `models.toml`):

```rust
/// LLM-role built-ins, compiled ONLY with the `llm` feature so the default
/// build pulls zero LLM deps. These are inert until `SEMANTEX_LLM_MODEL` selects
/// one by id — no LLM runs by default. Users override/add via `models.toml`.
#[cfg(feature = "llm")]
fn append_builtin_llm_specs(specs: &mut Vec<ModelSpec>) {
    use crate::model::spec::LlmSpec;
    // An airgap-friendly Ollama default (local endpoint). Provider/model/endpoint
    // are manifest DATA the user can override; nothing here forces a network LLM.
    specs.push(ModelSpec {
        id: "ollama-default".to_string(),
        role: ModelRole::Llm,
        // LLM weights are not fetched by semantex (genai/Ollama manage them);
        // a Local source with the conventional Ollama dir documents that.
        source: ModelSource::Local {
            dir: "ollama".to_string(),
        },
        capabilities: ModelCapabilities {
            instruction_aware: true,
            ..ModelCapabilities::default()
        },
        role_data: RoleData::Llm(LlmSpec {
            provider: "ollama".to_string(),
            model: "qwen2.5-coder:7b".to_string(),
            endpoint: String::new(),
        }),
    });
}
```

The `manifest.rs` `use` for `LlmSpec` is local to this fn (feature-gated) to avoid an unused-import warning on the default build. The `RoleData::Llm` variant already exists (Task 2), so the default build compiles even though it never constructs one.

- [ ] **Step 4: Run test to verify it passes**

Default build:
Run: `cargo test -p semantex-core model::registry::tests 2>&1 | tail -15`
Expected: PASS; and `cargo tree -p semantex-core 2>/dev/null | grep -c genai` → `0`.

LLM build:
Run: `cargo test -p semantex-core --features llm model::registry::tests 2>&1 | tail -15`
Expected: PASS — `llm_builtin_resolves_when_feature_on_and_selected` + `active_llm_is_none_by_default` both pass.

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/model/manifest.rs crates/semantex-core/src/model/registry.rs
git commit -m "feat(model): feature-gated LLM built-in specs; default build stays zero-LLM (S8)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: reconciliation — route S1/S3 selection through the registry

Replace the ad-hoc selection in S1's `hybrid.rs`/`builder.rs` and S3's reranker path with registry calls. **The traits/impls are unchanged — only *how the active model is chosen* moves.** This is the §"Reconciliation deltas" wiring; do it last so S1/S2/S3 have landed.

**Files:**
- Modify: `crates/semantex-core/src/search/hybrid.rs` (reranker selection)
- Modify: `crates/semantex-core/src/search/reranker_model.rs` (S3 — wrap the registry; OR document the registry as the source of the id)

> Note: Tasks 9 + 10 already replaced the *embedder* selection (`DenseBackendKind::parse(&config.dense_backend)` → `registry.embedder_backend_kind()`) in `builder.rs`/`hybrid.rs`. This task handles the *reranker* selection. The LLM path (`LlmBackend::from_env`) is left reading env directly for now — see the "deferred" note in §"Reconciliation deltas".

- [ ] **Step 1: Write the failing test**

Add a test to `reranker_model.rs`'s test module asserting the registry-backed selection resolves the active reranker spec for a given id (and that an unknown id still falls back per S3's existing contract):

```rust
    #[test]
    fn registry_resolves_active_reranker_spec() {
        use crate::config::SemantexConfig;
        use crate::model::{ModelRegistry, spec::RoleData};
        let mut cfg = SemantexConfig::default();
        cfg.reranker_model = "qwen3-reranker-0.6b".to_string();
        let reg = ModelRegistry::from_config(&cfg, None).unwrap();
        let spec = reg.active_reranker().unwrap();
        assert_eq!(spec.id, "qwen3-reranker-0.6b");
        let RoleData::Reranker(r) = &spec.role_data else { panic!() };
        // The score strategy is data the engine reads — proving S3's
        // classifier|yes_no choice now comes from the spec, not a hardcoded map.
        assert_eq!(
            r.score_strategy,
            crate::model::spec::ScoreStrategyKind::YesNoLogit
        );
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p semantex-core reranker_model::tests::registry_resolves 2>&1 | tail -15`
Expected: FAIL — compile error if `reranker_model.rs` does not yet import the registry, or assertion if S3's selection ignores the spec.

- [ ] **Step 3: Write minimal implementation**

Add a registry-backed constructor to S3's `RerankerChoice` (additive — keeps `select_reranker_choice_from_env()` as the env entrypoint, but lets it source the score strategy + template + token ids from the resolved `ModelSpec` instead of a hardcoded match). In `reranker_model.rs`:

```rust
use crate::model::spec::{ModelSpec, RoleData, ScoreStrategyKind};

impl RerankerChoice {
    /// Build a reranker choice from a resolved registry `ModelSpec`. The spec's
    /// `score_strategy` + prompt template + yes/no token ids drive the ONNX
    /// `ScoreStrategy`, so a new reranker is a manifest entry — not a code change
    /// to this selector. (The bge fastembed-native fast path is still chosen when
    /// the spec id matches the fastembed catalog; see `from_spec` below.)
    #[must_use]
    pub fn from_spec(spec: &ModelSpec) -> Self {
        debug_assert_eq!(spec.role, crate::model::spec::ModelRole::Reranker);
        let RoleData::Reranker(r) = &spec.role_data else {
            // Defensive: registry validate() guarantees role agreement.
            return RerankerChoice::Fastembed(super::fastembed_reranker::select_model_from_env());
        };
        // bge-v2-m3 keeps its fastembed-native path (already integrated, fastest).
        if spec.id == "bge-reranker-v2-m3" {
            return RerankerChoice::Fastembed(fastembed::RerankerModel::BGERerankerV2M3);
        }
        // Everything else goes through the generic ONNX loader, with the strategy
        // taken from the SPEC (data), not a hardcoded id→strategy map.
        let strategy_kind = match r.score_strategy {
            ScoreStrategyKind::ClassifierHead => StrategyKind::Classifier,
            ScoreStrategyKind::YesNoLogit => StrategyKind::YesNo,
        };
        RerankerChoice::Onnx(OnnxModelSpec::from_registry_spec(spec, strategy_kind))
    }
}
```

(`OnnxModelSpec::from_registry_spec` maps `ModelSource` → S3's `ModelFiles` and copies the prompt/token-id fields — a small constructor added alongside S3's existing `OnnxModelSpec`. If S3 has not exposed `OnnxModelSpec`/`StrategyKind` publicly, this task makes them `pub(crate)`.)

Then change `RerankerEngine::new_default` (S3) to source the active id from the registry rather than calling `select_reranker_choice_from_env()` directly — passing `RerankerChoice::from_spec(registry.active_reranker()?)`. The `SEMANTEX_RERANKER` master switch + off-by-default identity pass-through (S3's off-by-default safety contract) are untouched: `new_default` still bails first if `!reranker_enabled()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core reranker_model::tests 2>&1 | tail -15`
Expected: PASS — `registry_resolves_active_reranker_spec` + S3's existing selector tests pass.

Confirm the lib suite + the off-by-default reranker safety test (S3 Task 9) still pass:
Run: `cargo test -p semantex-core reranker 2>&1 | tail -20`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/search/reranker_model.rs crates/semantex-core/src/search/hybrid.rs
git commit -m "refactor(rerank): source reranker selection (strategy/template) from the registry (S8)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 13: golden test — registry-resolved == direct construction + 2nd-model end-to-end

The spec §6 anti-regression gate plus acceptance gates 3 + 4. **Requires all prior tasks + S1/S2/S3 landed.**

**Files:**
- Modify: `crates/semantex-core/tests/model_registry_golden_test.rs` (created in Task 9)

- [ ] **Step 1: Write the failing test**

Append to `crates/semantex-core/tests/model_registry_golden_test.rs`:

```rust
use semantex_core::model::{BackendKind, ModelRegistry};
use semantex_core::model::capabilities::backend_for;
use semantex_core::model::spec::RoleData;
use semantex_core::search::dense_backend::DenseBackendKind;

/// Gate 4 + spec §6: the dense backend the registry resolves for each built-in
/// embedder is IDENTICAL to constructing the `DenseBackendKind` directly from
/// the model's known capability (no quality regression from the indirection).
#[test]
fn registry_resolved_backend_equals_direct_for_builtins() {
    // coderank-137m (single-vector) → HNSW, both ways.
    let mut cfg = SemantexConfig::default();
    cfg.embedder = "coderank-137m".to_string();
    let reg = ModelRegistry::from_config(&cfg, None).unwrap();
    let resolved = reg.embedder_backend_kind().unwrap();
    // Direct construction: a single-vector capability maps to HNSW.
    let direct = backend_for(&reg.active_embedder().unwrap().capabilities).dense_kind();
    assert_eq!(resolved, direct);
    assert_eq!(resolved, DenseBackendKind::CoderankHnsw);

    // lateon-colbert (multi-vector) → PLAID, both ways.
    cfg.embedder = "lateon-colbert".to_string();
    let reg = ModelRegistry::from_config(&cfg, None).unwrap();
    assert_eq!(reg.embedder_backend_kind().unwrap(), DenseBackendKind::ColbertPlaid);
    assert_eq!(BackendKind::ColbertPlaid.dense_kind(), DenseBackendKind::ColbertPlaid);
}

/// Gate 3: a user `models.toml` adding a SECOND permissive embedder loads and
/// capability-routes end-to-end with NO code change.
#[test]
fn user_manifest_second_model_loads_and_routes() {
    let tmp = tempfile::TempDir::new().unwrap();
    let project = tmp.path().join("proj");
    let semantex_dir = project.join(".semantex");
    fs::create_dir_all(&semantex_dir).unwrap();
    // A 2nd permissive single-vector embedder (gte-modernbert-base, Apache-2.0).
    fs::write(
        semantex_dir.join("models.toml"),
        r#"
        [[model]]
        id = "gte-modernbert-hnsw"
        role = "embedder"
        [model.source]
        kind = "hf"
        repo = "Alibaba-NLP/gte-modernbert-base"
        files = ["model_int8.onnx", "tokenizer.json"]
        [model.capabilities]
        multi_vector = false
        [model.embedder]
        dims = 768
        max_context = 8192
        query_prefix = ""
        pooling = "cls"
        quant = "int8_symmetric"
        "#,
    )
    .unwrap();

    let mut cfg = SemantexConfig::default();
    cfg.embedder = "gte-modernbert-hnsw".to_string();
    let reg = ModelRegistry::from_config(&cfg, Some(&project)).unwrap();

    // It resolves…
    let spec = reg.active_embedder().unwrap();
    assert_eq!(spec.id, "gte-modernbert-hnsw");
    let RoleData::Embedder(_) = &spec.role_data else { panic!() };
    // …and capability-routes to HNSW from the spec alone.
    assert_eq!(reg.embedder_backend_kind().unwrap(), DenseBackendKind::CoderankHnsw);
}

/// Gate 2 (error arm): a mismatched on-disk fingerprint errors cleanly.
#[test]
fn mismatched_index_errors_cleanly() {
    use semantex_core::search::dense_backend::verify_persisted_fingerprint_matches;
    use semantex_core::types::IndexMeta;
    let tmp = tempfile::TempDir::new().unwrap();
    let index_dir = tmp.path().join(".semantex");
    fs::create_dir_all(&index_dir).unwrap();
    let meta = IndexMeta {
        schema_version: IndexMeta::CURRENT_SCHEMA_VERSION,
        project_path: index_dir.clone(),
        created_at: "0".to_string(),
        updated_at: "0".to_string(),
        file_count: 0,
        chunk_count: 0,
        embedding_model: "coderank-137m".to_string(),
        embedding_dim: 768,
        use_bm25_stemmer: true,
        dense_backend: "coderank-hnsw".to_string(),
        embedder_fingerprint: "BUILT_WITH_THIS".to_string(),
    };
    fs::write(index_dir.join("meta.json"), serde_json::to_string(&meta).unwrap()).unwrap();
    let err = verify_persisted_fingerprint_matches(&index_dir, "ACTIVE_IS_DIFFERENT")
        .expect_err("a different active embedder must error");
    assert!(err.to_string().contains("--rebuild"), "error must guide the user");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p semantex-core --test model_registry_golden_test 2>&1 | tail -25`
Expected: FAIL — initially `cannot find …` for any not-yet-`pub` item (e.g. `model::capabilities::backend_for`), then assertion failures until Tasks 5/7/10 are wired. Make the referenced items `pub` (they are, per Tasks 5/7/8/10).

- [ ] **Step 3: Write minimal implementation**

No new production code — this task only verifies prior tasks. If a referenced symbol isn't public, add the `pub` (e.g. ensure `model/mod.rs` re-exports `backend_for` via `pub use capabilities::backend_for;` — add it to the `mod.rs` re-export list if missing).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core --test model_registry_golden_test 2>&1 | tail -25`
Expected: PASS — `registry_resolved_backend_equals_direct_for_builtins`, `user_manifest_second_model_loads_and_routes`, `mismatched_index_errors_cleanly` pass (the `#[ignore]` `build_stamps_…` remains skipped without the model).

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/tests/model_registry_golden_test.rs crates/semantex-core/src/model/mod.rs
git commit -m "test(model): golden gate — registry-resolved == direct + 2nd-model end-to-end (S8)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Acceptance gates → tests (spec §4 S8)

| Gate | Test(s) | What it proves |
|------|---------|----------------|
| (1) swap reranker/LLM by config → behavior changes, no recompile, no reindex | `reranker_model::tests::registry_resolves_active_reranker_spec` (Task 12) + `registry::tests::llm_builtin_resolves_when_feature_on_and_selected` (Task 11) | reranker/LLM selection is a registry lookup keyed by config/env; no index touched |
| (2) swap embedder by config → versioned rebuild + atomic switchover; mismatch errors cleanly | `dense_backend::tests::{versioned_dir_…, active_pointer_round_trips, verify_fingerprint_errors_on_mismatch}` (Tasks 8, 10) + golden `mismatched_index_errors_cleanly` + `build_stamps_a_nonempty_embedder_fingerprint` (Tasks 9, 13) | fingerprint stamping + versioned dir + atomic pointer flip + clean mismatch error |
| (3) user `models.toml` new model loads + runs end-to-end, no code change | `manifest::tests::{load_user_manifest_parses_a_second_embedder, merge_appends_a_new_user_id}` (Task 4) + golden `user_manifest_second_model_loads_and_routes` (Task 13) | a 2nd permissive model resolves + capability-routes from the manifest alone |
| (4) capability routing picks PLAID for multi_vector, HNSW for single-vector from the spec alone | `capabilities::tests::multi_vector_routes_to_plaid_single_to_hnsw` (Task 5) + golden `registry_resolved_backend_equals_direct_for_builtins` (Task 13) | negotiation is capability-driven, not id-branching; identical to direct construction |

---

## Public signatures locked (the registry surface)

These are the exact public signatures this plan commits to (consumed by S1/S2/S3 and the integration doc):

```rust
// crates/semantex-core/src/model/spec.rs
pub enum ModelRole { Embedder, Reranker, Llm }
pub enum ModelSource {
    Hf { repo: String, files: Vec<String> },
    Local { dir: String },
    Url { base: String, files: Vec<String> },
}
pub enum Pooling { Mean, Cls, LateInteraction }
pub enum QuantKind { None, Int8Symmetric }
pub enum ScoreStrategyKind { ClassifierHead, YesNoLogit }
pub struct EmbedderSpec {
    pub dims: usize, pub max_context: usize,
    pub query_prefix: String, pub doc_prefix: String,
    pub pooling: Pooling, pub normalize: bool, pub quant: QuantKind,
}
pub struct RerankerSpec {
    pub score_strategy: ScoreStrategyKind,
    pub prompt_prefix: String, pub prompt_middle: String, pub prompt_suffix: String,
    pub yes_token_id: Option<usize>, pub no_token_id: Option<usize>,
}
pub struct LlmSpec { pub provider: String, pub model: String, pub endpoint: String }
pub enum RoleData { Embedder(EmbedderSpec), Reranker(RerankerSpec), Llm(LlmSpec) }
pub struct ModelSpec {
    pub id: String, pub role: ModelRole, pub source: ModelSource,
    pub capabilities: ModelCapabilities, pub role_data: RoleData,
}
impl ModelSpec { pub fn validate(&self) -> anyhow::Result<()>; }
pub struct EmbedderFingerprint;
impl EmbedderFingerprint { pub fn compute(id: &str, spec: &EmbedderSpec) -> String; }

// crates/semantex-core/src/model/capabilities.rs
pub struct ModelCapabilities {
    pub multi_vector: bool,
    pub matryoshka_dims: Option<Vec<usize>>,
    pub produces_sparse: bool,
    pub instruction_aware: bool,
    pub max_batch: Option<usize>,
}
pub enum BackendKind { ColbertPlaid, CoderankHnsw }
pub fn backend_for(caps: &ModelCapabilities) -> BackendKind;
impl BackendKind { pub fn dense_kind(self) -> crate::search::dense_backend::DenseBackendKind; }

// crates/semantex-core/src/model/manifest.rs
pub fn builtin_specs() -> Vec<ModelSpec>;
pub fn load_user_manifest(path: &Path) -> anyhow::Result<Vec<ModelSpec>>;
pub fn user_manifest_path(config: &SemantexConfig, project_path: Option<&Path>) -> Option<PathBuf>;
pub fn merge(builtin: Vec<ModelSpec>, user: Vec<ModelSpec>) -> Vec<ModelSpec>;

// crates/semantex-core/src/model/registry.rs
pub struct ModelRegistry { /* private */ }
impl ModelRegistry {
    pub fn from_config(config: &SemantexConfig, project_path: Option<&Path>) -> anyhow::Result<Self>;
    pub fn resolve(&self, role: ModelRole, id: &str) -> anyhow::Result<&ModelSpec>;
    pub fn active_embedder(&self) -> anyhow::Result<&ModelSpec>;
    pub fn active_reranker(&self) -> anyhow::Result<&ModelSpec>;
    pub fn active_llm(&self) -> anyhow::Result<Option<&ModelSpec>>;
    pub fn embedder_backend_kind(&self) -> anyhow::Result<crate::search::dense_backend::DenseBackendKind>;
}

// crates/semantex-core/src/search/dense_backend.rs  (S8 additions next to S1's helpers)
pub fn active_dense_dir(index_dir: &Path, backend: DenseBackendKind, fingerprint: &str) -> PathBuf;
pub fn read_active_pointer(index_dir: &Path, backend: DenseBackendKind) -> Option<String>;
pub fn write_active_pointer(index_dir: &Path, backend: DenseBackendKind, fingerprint: &str) -> anyhow::Result<()>;
pub fn verify_persisted_fingerprint_matches(index_dir: &Path, expected: &str) -> anyhow::Result<()>;

// crates/semantex-core/src/config.rs  (S8 additions)
// SemantexConfig gains: pub embedder: String, pub reranker_model: String, pub llm_model: String
// + SEMANTEX_EMBEDDER / SEMANTEX_RERANKER_MODEL / SEMANTEX_LLM_MODEL overlays in load()

// crates/semantex-core/src/types.rs  (S8 addition)
// IndexMeta gains: pub embedder_fingerprint: String  (no schema bump beyond S1/S2's 11)
```

---

## S1 / S2 / S3 reconciliation deltas (for the integration doc)

These are the **exact selection-code replacements** S8 introduces. The traits + impls in S1/S2/S3 are UNCHANGED; only *how the active model is chosen* moves into the registry. Record these in `2026-05-31-integration-and-cutover.md`:

### S1 deltas (`s1-dense-backend-seam.md`)
- **REPLACED:** the embedder/backend *selection* call. S1's plan opens the dense channel in `hybrid.rs` via `DenseBackendKind::parse(&config.dense_backend)` and builds it in `builder.rs` via `DenseBackendKind::parse(...).unwrap_or_default()`. **Both call-sites become `ModelRegistry::from_config(config, project).embedder_backend_kind()?`** (S8 Tasks 9, 10). The `dense_backend.rs` `DenseBackendKind` enum, `parse`/`name`, `dense_subdir`, `verify_persisted_backend_matches`, the `DenseBackend`/`DenseIndexBuilder` traits, and `ColbertPlaidBackend`/`ColbertPlaidIndexBuilder` are **unchanged**.
- **KEPT (now legacy-compatible):** `SemantexConfig.dense_backend` + `SEMANTEX_DENSE_BACKEND` remain as a *fallback / override*, but the authoritative selector is `config.embedder` → registry → capabilities → `DenseBackendKind`. (Embedder spec ids are model-descriptive — `coderank-137m`, `lateon-colbert` — and distinct from the backend names — `coderank-hnsw`, `colbert-plaid` — that the registry routes them to via capabilities. A user who sets only `SEMANTEX_DENSE_BACKEND` and not `SEMANTEX_EMBEDDER` still gets the default embedder's backend, which agrees out of the box because the default `lateon-colbert` routes to `colbert-plaid` — matching S1's `dense_backend` default; the integration lead may choose to deprecate `dense_backend` in favor of `embedder` — flagged below.)
- **ADDED next to S1's on-disk helpers:** `active_dense_dir` (versioned `<backend>/<fingerprint>/`), `read_active_pointer`/`write_active_pointer` (the `ACTIVE` file), `verify_persisted_fingerprint_matches`. S1's `dense/<backend>/` layout gains a `<fingerprint>/` leaf + an `ACTIVE` pointer.
- **ADDED to `IndexMeta`:** `embedder_fingerprint: String` (rides on S1's 9→10 bump; no extra bump).

### S2 deltas (`s2-coderank-hnsw-dense.md`)
- **REPLACED:** CodeRankEmbed's hardcoded constants as the *source of selection*. S2's `EMBEDDING_DIM`/`QUERY_PREFIX`/pooling constants in `embedding/single_vector.rs` are **kept as the encoder's defaults**, but the registry's built-in `coderank-137m` embedder `ModelSpec` (Task 3) now carries dim=768/prefix/pooling=Cls/quant=Int8Symmetric as the authoritative **selection + fingerprint** data. The encoder impl, `single_vector_model.rs` download, and `hnsw_index.rs` backend are **unchanged**. (Optional follow-up: have `single_vector.rs` read dim/prefix from the resolved spec instead of consts — NOT required for S8; flagged below.)
- **REPLACED:** the builder's dense-dir target. S2's `CoderankHnswIndexBuilder` persists into the dir the builder passes it; S8 makes that dir the versioned `active_dense_dir(index_dir, CoderankHnsw, fingerprint)` (Task 9) instead of a flat `dense_subdir(...)`. No change to the builder impl itself.
- **KEPT:** S2's 10→11 schema bump is the final shipped value; S8 adds a field under it, not a new bump.

### S3 deltas (`s3-onnx-reranker-upgrade.md`)
- **REPLACED:** the reranker *model selection*. S3's `select_reranker_choice_from_env()` reads `SEMANTEX_RERANKER_MODEL` and maps known strings → `RerankerChoice` via a hardcoded match. **S8 makes that selection a registry lookup:** `RerankerChoice::from_spec(registry.active_reranker()?)` (Task 12), where the resolved reranker `ModelSpec` carries `score_strategy` (→ S3's `StrategyKind`), the prompt template, and the yes/no token ids as **data** — so a new reranker is a manifest entry, not a new match arm. S3's `OnnxReranker`, `ScoreStrategy`, `classifier_score_from_logits`/`yes_no_score_from_logits`, `reranker_engine.rs`, and `reranker_download.rs` are **unchanged** (S3's `OnnxModelSpec`/`StrategyKind` may need `pub(crate)` visibility + a `from_registry_spec` constructor — additive).
- **KEPT verbatim:** the `SEMANTEX_RERANKER` master switch, `reranker_enabled()`, and the off-by-default identity pass-through (S3's "off-by-default safety contract"). S8 does not change whether rerank runs — only *which* model it loads when it does.
- **NOTE:** `SEMANTEX_RERANKER_MODEL` (S3's env) and `config.reranker_model` (S8's field) are the **same selection key**; S8's `env_string("SEMANTEX_RERANKER_MODEL", …)` overlay (Task 7) makes the env var populate the config field that the registry reads. No double-read.

### LLM path (deferred reconciliation — flagged, not done in S8)
- `LlmBackend::from_env()` (in `llm/mod.rs`, feature-gated) still reads `SEMANTEX_LLM_MODEL`/`_PROVIDER`/`_ENDPOINT`/`_BACKEND` directly. S8 adds `config.llm_model` + an LLM-role registry spec, and `active_llm()` resolves it, but **wiring `LlmBackend` to construct from a resolved `LlmSpec` is intentionally left as a follow-up** (it touches the v0.7.1 HyDE call-site work in S5 and the genai backend constructor; doing it here would couple S8 to the `llm` feature's internals). The registry side is ready; the consumer side is a small later delta. Recorded so the integration lead schedules it after S5.

### Open flags for the integration lead
1. **`dense_backend` vs `embedder` as the authoritative dense selector.** S8 makes `config.embedder` authoritative (capabilities → backend). `config.dense_backend`/`SEMANTEX_DENSE_BACKEND` (S1) become redundant for built-ins (the default embedder `lateon-colbert` routes to the same `colbert-plaid` backend S1 defaults to per D4, so they agree out of the box). Decide whether to keep both (back-compat) or deprecate `dense_backend` once S8 lands. The S0 harness drives the A/B via `SEMANTEX_DENSE_BACKEND` (integration doc §3.6 note) — **if `dense_backend` is deprecated, the harness must switch to `SEMANTEX_EMBEDDER` (set it to the embedder id `coderank-137m` for the new arm — which capability-routes to the `coderank-hnsw` backend).** The Phase-3 cutover decision (integration doc §6.1) is the single place that flips the *default* `embedder` from `lateon-colbert` to `coderank-137m`, once the harness proves the single-vector path meets-or-beats PLAID — keep the S8 default at `lateon-colbert` until then.
2. **Encoder consts → spec-sourced (S2).** Optional follow-up to have `single_vector.rs` read dim/prefix from the resolved spec; not required for the gates.

