# S1 — DenseBackend Seam Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract a `DenseBackend` + `DenseIndexBuilder` trait seam in `semantex-core` so the dense search/build channel routes through a trait, with today's PLAID path as the first impl (`colbert-plaid`), behavior-identical, and add config/env backend selection plus per-backend on-disk subdirs — unblocking the new single-vector backend (S2) without touching dense ranking behavior.

**Architecture:** Introduce `crates/semantex-core/src/search/dense_backend.rs` defining `DenseBackend` (query-time) and `DenseIndexBuilder` (build-time) traits. Wrap `PlaidSearcher` in a thin `ColbertPlaidBackend` adapter (impl #1) and re-point the `dense_handle`/`exp_dense_handle` channels in `hybrid.rs` and the PLAID build block in `index/builder.rs` to go through the trait objects. Backend identity drives a new `dense_backend` config field + `SEMANTEX_DENSE_BACKEND` env (default `colbert-plaid`), a new `IndexMeta.dense_backend` field (schema bump 9→10 forces clean reindex), per-backend on-disk subdir `.semantex/dense/<backend>/`, and an open-time mismatch guard that mirrors the existing BM25 stemmer-flag pattern in `sparse_search.rs`.

**Tech Stack:** Rust 2024 edition, `anyhow`, `serde`/`serde_json`, `next-plaid` (PLAID), `tantivy` (unchanged), `rusqlite` (unchanged), `tempfile` + `cargo test` for tests. No new crate dependencies.

---

## Reconciled facts (verified against current source — do not re-derive)

These are quoted from the real tree at plan-authoring time. Every type/method referenced below exists today or is introduced by an earlier task in this plan.

- **`ScoredChunkId` — REUSE the existing 5-field type, NOT the spec's 2-field sketch.** `crates/semantex-core/src/types.rs:219-241`:
  ```rust
  #[derive(Debug, Clone, Default)]
  pub struct ScoredChunkId {
      pub chunk_id: u64,
      pub score: f32,
      pub score_dense: f32,
      pub score_sparse: f32,
      pub score_exact: f32,
  }
  impl ScoredChunkId {
      pub fn new(chunk_id: u64, score: f32) -> Self { /* per-channel = 0.0 */ }
  }
  ```
  The §3 spec sketch shows `pub struct ScoredChunkId { pub chunk_id: u64, pub score: f32 }`. That conflicts with the type `PlaidSearcher::search` already returns and that the whole fusion path consumes. **Resolution: the trait uses `crate::types::ScoredChunkId` verbatim.** A 2-field type would force a lossy conversion at the seam and break behavior-identity. (Recorded as spec gap G1 — see end.)

- **`PlaidSearcher` API** (`crates/semantex-core/src/search/plaid_search.rs`):
  - `PlaidSearcher::open(index_dir: &Path, mapping_path: &Path) -> Result<Self>` (line 129)
  - `PlaidSearcher::search(&self, embedder: &ColbertEmbedder, query: &str, top_k: usize) -> Result<Vec<ScoredChunkId>>` (line 151)
  - `PlaidSearcher::search_with_subset(&self, embedder: &ColbertEmbedder, query: &str, top_k: usize, chunk_id_subset: Option<&[u64]>) -> Result<Vec<ScoredChunkId>>` (line 211)
  - `PlaidSearcher::doc_to_chunk(&self) -> &[u64]` (line 278)

- **ColBERT embedder** (`crates/semantex-core/src/embedding/colbert.rs`): `ColbertEmbedder::global(model_dir: &Path) -> Result<&'static ColbertEmbedder>` (line 258), `ColbertEmbedder::for_indexing(model_dir: &Path) -> Result<Self>` (line 117), `encode_documents(&self, texts: &[String]) -> Result<Vec<TokenEmbeddings>>` (line 282). CPU execution provider is pinned in `build_encoder` (lines ~182-195). `model_manager::ensure_colbert_model(models_dir: &Path) -> Result<PathBuf>` (`embedding/model_manager.rs:14`).

- **`HybridSearcher`** (`crates/semantex-core/src/search/hybrid.rs:24-31`) fields today: `sparse: Option<SparseIndex>`, `plaid: Option<PlaidSearcher>`, `colbert: Option<&'static ColbertEmbedder>`, `reranker`, `store`, `config`. Dense channel: `dense_handle` lines 351-389; subset prep lines 269-337 (uses `plaid.doc_to_chunk()`); `exp_dense_handle` lines 450-486. PLAID open block lines 104-136.

- **`index/builder.rs`** PLAID block lines 575-854 (closure `(|| -> Result<()> { ... })()`); meta write lines 862-874; stale-schema cleanup lines 147-156 (already removes `plaid/`, `plaid_mapping.bin`, legacy `dense.usearch`). Uses `model_manager::ensure_colbert_model` (line 578) and `ColbertEmbedder::for_indexing` (line 593).

- **`IndexMeta`** (`crates/semantex-core/src/types.rs:178-208`): fields `schema_version, project_path, created_at, updated_at, file_count, chunk_count, embedding_model, embedding_dim, use_bm25_stemmer`. `CURRENT_SCHEMA_VERSION = 9` (line 207). **No `dense_backend` field today.**

- **Stemmer-flag mismatch pattern to mirror** (`crates/semantex-core/src/search/sparse_search.rs:44-83`): `verify_persisted_stemmer_matches(index_path: &Path, expected: bool) -> Result<()>` reads `<index_dir>/meta.json`, returns `Ok(())` on missing/unparseable meta, `Err(anyhow!)` on value mismatch naming both values + recovery hint; called first thing in `SparseIndex::open` (line 150).

- **`SemantexConfig`** (`crates/semantex-core/src/config.rs:14-98`): `#[serde(default)]`, env overrides in `load()` (lines 126-142). Helper `config::env_usize(key, default)` (line 201). No `dense_backend` field today. `project_index_dir(project_path) -> PathBuf` returns `<project>/.semantex` (line 165).

- **Module wiring**: `crates/semantex-core/src/search/mod.rs:1-19` declares the `search::*` submodules. `index/state.rs:72-81` `is_stale` returns `true` for unparseable meta (so an old v9 meta.json that lacks `dense_backend` after the bump will be `Stale` → clean rebuild).

- **Real-index test harness pattern** (`crates/semantex-core/tests/search_accuracy_test.rs:65-95, 808-818`): `TempDir` → write synthetic source files under `project_dir` → `IndexBuilder::new(&config)?.build(&project_dir)?` → open via `HybridSearcher::open(&project_dir.join(".semantex"), &config)`. **Repo-agnostic, no hardcoded paths.** (NB: the `#[ignore]`'d tests at 808+ open `fixture.index_dir` = `test_index`, a stale-bug directory; our golden test opens `project_dir.join(".semantex")`, which is where `build` actually writes.)

---

## File Structure

Files created or modified, one responsibility each:

- **Create `crates/semantex-core/src/search/dense_backend.rs`** — the seam. Defines `DenseBackend` (query: `name`, `search`, `search_with_subset`) and `DenseIndexBuilder` (build: `name`, `build`, `insert`, `delete`, `persist`) traits over `crate::types::ScoredChunkId`; the `DenseBackendKind` selector enum (`name()`, `from_env_or_config()`, `parse()`); the `dense_subdir(index_dir, backend) -> PathBuf` path helper; and `verify_persisted_backend_matches(index_dir, expected) -> Result<()>` (mirrors the stemmer guard). Unit tests for parse/default/path/guard live here.

- **Create `crates/semantex-core/src/search/colbert_plaid_backend.rs`** — impl #1. `ColbertPlaidBackend` wraps an owned `PlaidSearcher` + `&'static ColbertEmbedder` and implements `DenseBackend` by delegating to `PlaidSearcher::search`/`search_with_subset` (byte-identical). `ColbertPlaidIndexBuilder` implements `DenseIndexBuilder` by owning the PLAID build/update logic lifted out of `builder.rs`. `name()` returns `"colbert-plaid"`. Tests: name constant, empty-subset short-circuit parity.

- **Modify `crates/semantex-core/src/search/mod.rs:1-19`** — add `pub mod dense_backend;` and `pub mod colbert_plaid_backend;`.

- **Modify `crates/semantex-core/src/types.rs:178-208`** — add `pub dense_backend: String` to `IndexMeta`; bump `CURRENT_SCHEMA_VERSION` 9 → 10; update the doc comment + the two in-file tests that build `IndexMeta` literals.

- **Modify `crates/semantex-core/src/config.rs:14-98, 126-142, 201`** — add `pub dense_backend: String` to `SemantexConfig` (default `"colbert-plaid"`); set it in `Default`; add the `SEMANTEX_DENSE_BACKEND` env override in `load()`; add a `config::env_string(key, default)` helper next to `env_usize`.

- **Modify `crates/semantex-core/src/search/hybrid.rs:24-31, 104-149, 269-337, 351-389, 450-486`** — replace the `plaid: Option<PlaidSearcher>` + `colbert: Option<&'static ColbertEmbedder>` pair with a single `dense: Option<Box<dyn DenseBackend>>`; build it in `open()` behind the backend selector + mismatch guard; re-point `dense_handle`, the subset-prep block, and `exp_dense_handle` to call through `self.dense`. Behavior-identical.

- **Modify `crates/semantex-core/src/index/builder.rs:147-156, 575-854, 862-874`** — route the PLAID build/update through `ColbertPlaidIndexBuilder` (selected by `self.config.dense_backend`); write the active backend into `meta.json`; extend the stale-cleanup to also remove the per-backend `dense/<backend>/` subdir; persist into `.semantex/dense/colbert-plaid/` going forward while keeping legacy `plaid/` readable.

- **Create `crates/semantex-core/tests/dense_backend_golden_test.rs`** — the acceptance gate. Builds a synthetic multi-language repo in a tempdir, indexes it, runs a fixed query set through `HybridSearcher`, and asserts the dense-channel result `(chunk_id, score)` sequence is **byte-identical** between a pinned baseline (captured once from `colbert-plaid` pre-cutover) and the post-refactor run. Repo-agnostic (tempdir, synthetic files).

---

## Phasing note for executors

Tasks 1–4 are pure additions (new file, new fields, new config) with no behavior change — land them first; the suite stays green throughout. Task 5 lifts PLAID logic into the adapter. Tasks 6–7 are the actual call-site rewrites in `hybrid.rs` and `builder.rs` — the only behavior-sensitive edits, each gated by the golden test from Task 8 (write the golden baseline capture in Task 8 BEFORE rewriting call sites if you want a true before/after; the plan orders Task 8 last because the byte-identical comparison is self-contained against a checked-in baseline file). Commit after every task.

---

### Task 1: `DenseBackend` / `DenseIndexBuilder` trait module

**Files:**
- Create: `crates/semantex-core/src/search/dense_backend.rs`
- Modify: `crates/semantex-core/src/search/mod.rs:1-19`

- [ ] **Step 1: Write the failing test**

Create `crates/semantex-core/src/search/dense_backend.rs` with ONLY the test module first (the impl follows in Step 3):

```rust
//! The `DenseBackend` seam: a trait abstraction over the dense search/build
//! channel so multiple dense backends (today: `colbert-plaid`) can coexist and
//! be selected by config/env. See `docs/superpowers/specs/2026-05-31-semantex-sota-overhaul-design.md` §3/§4 S1.

use crate::types::ScoredChunkId;
use anyhow::Result;
use std::path::{Path, PathBuf};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_backend_kind_default_is_colbert_plaid() {
        assert_eq!(DenseBackendKind::default(), DenseBackendKind::ColbertPlaid);
        assert_eq!(DenseBackendKind::default().name(), "colbert-plaid");
    }

    #[test]
    fn parse_known_backend_names() {
        assert_eq!(
            DenseBackendKind::parse("colbert-plaid"),
            Some(DenseBackendKind::ColbertPlaid)
        );
        // Whitespace + case are normalized.
        assert_eq!(
            DenseBackendKind::parse("  Colbert-Plaid  "),
            Some(DenseBackendKind::ColbertPlaid)
        );
    }

    #[test]
    fn parse_unknown_backend_is_none() {
        assert_eq!(DenseBackendKind::parse("totally-made-up"), None);
        assert_eq!(DenseBackendKind::parse(""), None);
    }

    #[test]
    fn dense_subdir_is_per_backend() {
        let root = Path::new("/tmp/proj/.semantex");
        let p = dense_subdir(root, DenseBackendKind::ColbertPlaid);
        assert_eq!(p, Path::new("/tmp/proj/.semantex/dense/colbert-plaid"));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p semantex-core dense_backend::tests 2>&1 | tail -20`
Expected: FAIL — `cannot find type DenseBackendKind in this scope` / `cannot find function dense_subdir in this scope` (also a compile error because `mod dense_backend;` isn't declared yet — that is fixed in Step 3's edit to `mod.rs`).

- [ ] **Step 3: Write minimal implementation**

Add `pub mod dense_backend;` to `crates/semantex-core/src/search/mod.rs` immediately after `pub mod deep;` (keep the list alphabetical-ish, matching existing style):

```rust
pub mod deep;
pub mod dense_backend;
pub mod fastembed_reranker;
```

Then add the real definitions to `dense_backend.rs`, ABOVE the `#[cfg(test)]` module:

```rust
/// Identity of a dense backend — drives config selection and on-disk paths.
///
/// Today only `colbert-plaid` exists (the ColBERT late-interaction + vendored
/// next-plaid path). S2 adds `coderank-hnsw`. The string form is what gets
/// written into `meta.json` and read from `SEMANTEX_DENSE_BACKEND`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DenseBackendKind {
    /// ColBERT late-interaction over a vendored next-plaid PLAID index.
    #[default]
    ColbertPlaid,
}

impl DenseBackendKind {
    /// Stable on-disk / config identity. MUST stay in sync with `parse`.
    pub fn name(self) -> &'static str {
        match self {
            DenseBackendKind::ColbertPlaid => "colbert-plaid",
        }
    }

    /// Parse a backend name (case-insensitive, whitespace-trimmed).
    /// Returns `None` for an unknown name so callers can fall back to the
    /// default and warn, rather than panicking on a typo'd env var.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "colbert-plaid" => Some(DenseBackendKind::ColbertPlaid),
            _ => None,
        }
    }
}

/// The per-backend on-disk directory: `<index_dir>/dense/<backend>/`.
///
/// Per-backend isolation lets `colbert-plaid` and a future `coderank-hnsw`
/// index coexist on disk during the S2 A/B without clobbering each other.
pub fn dense_subdir(index_dir: &Path, backend: DenseBackendKind) -> PathBuf {
    index_dir.join("dense").join(backend.name())
}

/// A scored chunk returned by the dense channel. Items are sorted by
/// descending `score`. This is the project-wide `ScoredChunkId` (5 fields);
/// dense backends populate only `chunk_id` + `score` (per-channel fields stay
/// zero and are filled by the fusion stage).
pub type DenseHit = ScoredChunkId;

/// Query-time dense backend. Implementations are `Send + Sync` because the
/// `HybridSearcher` shares one instance across the rayon-parallel search
/// channels (`dense_handle` + `exp_dense_handle`).
pub trait DenseBackend: Send + Sync {
    /// Backend identity for on-disk paths + config selection.
    fn name(&self) -> &'static str;

    /// Search the dense channel for a text query, returning the top `k`
    /// `(chunk_id, score)` hits sorted by descending score.
    fn search(&self, query: &str, k: usize) -> Result<Vec<DenseHit>>;

    /// Restrict scoring to a candidate `subset` of chunk IDs (used by the
    /// `file_filter` prefilter). An empty subset MUST yield an empty result.
    fn search_with_subset(&self, query: &str, k: usize, subset: &[u64]) -> Result<Vec<DenseHit>>;
}

/// Build-time dense index builder. Mirrors the dense build/update lifecycle
/// the PLAID block in `index/builder.rs` performs today.
pub trait DenseIndexBuilder: Send + Sync {
    /// Backend identity (matches the query-side `DenseBackend::name`).
    fn name(&self) -> &'static str;

    /// Full (re)build from the complete `(chunk_id, content)` corpus.
    fn build(&mut self, chunks: &[(u64, &str)]) -> Result<()>;

    /// Incremental add of new `(chunk_id, content)` pairs.
    fn insert(&mut self, chunks: &[(u64, &str)]) -> Result<()>;

    /// Incremental delete of the given chunk IDs from the dense index.
    fn delete(&mut self, chunk_ids: &[u64]) -> Result<()>;

    /// Persist the dense index + any sidecar mapping into `dir`
    /// (a per-backend `dense/<backend>/` directory).
    fn persist(&self, dir: &Path) -> Result<()>;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core dense_backend::tests 2>&1 | tail -20`
Expected: PASS — `test result: ok. 4 passed`.

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/search/dense_backend.rs crates/semantex-core/src/search/mod.rs
git commit -m "feat(search): add DenseBackend/DenseIndexBuilder trait seam (S1)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `dense_backend` field in `SemantexConfig` + `SEMANTEX_DENSE_BACKEND` env

**Files:**
- Modify: `crates/semantex-core/src/config.rs:14-98` (struct + Default), `:126-142` (env overrides), `:201` (helper)

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` block at the bottom of `crates/semantex-core/src/config.rs` (after `default_config_enables_bm25_stemmer`):

```rust
    #[test]
    fn default_dense_backend_is_colbert_plaid() {
        let cfg = SemantexConfig::default();
        assert_eq!(
            cfg.dense_backend, "colbert-plaid",
            "default dense backend must stay colbert-plaid until S2 + harness flip it (D4)"
        );
    }

    #[test]
    fn env_string_reads_value_or_default() {
        // Unset key falls back to the provided default.
        assert_eq!(
            env_string("SEMANTEX_DENSE_BACKEND_TEST_UNSET_KEY", "colbert-plaid"),
            "colbert-plaid"
        );
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p semantex-core config::tests 2>&1 | tail -20`
Expected: FAIL — `no field 'dense_backend' on type 'SemantexConfig'` and `cannot find function 'env_string' in this scope`.

- [ ] **Step 3: Write minimal implementation**

In `crates/semantex-core/src/config.rs`, add the field to the `SemantexConfig` struct (place it right after `use_bm25_stemmer: bool,` so it groups with the other persisted-flag fields):

```rust
    pub use_bm25_stemmer: bool,
    /// Selected dense search backend identity (e.g. `"colbert-plaid"`).
    /// MUST match the value the index was built with — `HybridSearcher::open`
    /// re-validates it against the persisted `IndexMeta.dense_backend` and
    /// refuses to load on mismatch (mirrors `use_bm25_stemmer`). Override via
    /// `SEMANTEX_DENSE_BACKEND`. Default `"colbert-plaid"`.
    pub dense_backend: String,
```

Add the default in `impl Default for SemantexConfig` (right after `use_bm25_stemmer: true,`):

```rust
            use_bm25_stemmer: true,
            dense_backend: "colbert-plaid".to_string(),
```

Add the env override in `load()` after the `SEMANTEX_MODEL_DIR` block (lines ~139-141):

```rust
        if let Ok(v) = std::env::var("SEMANTEX_DENSE_BACKEND") {
            let trimmed = v.trim();
            if !trimmed.is_empty() {
                config.dense_backend = trimmed.to_string();
            }
        }
```

Add the `env_string` helper next to `env_usize` (after the `env_usize` fn, ~line 207):

```rust
/// Read a non-empty string tuning knob from an environment variable.
///
/// Returns the trimmed value when `key` is set to a non-empty string; a
/// missing or whitespace-only value falls back to `default`.
pub(crate) fn env_string(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| default.to_string())
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core config::tests 2>&1 | tail -20`
Expected: PASS — both new tests + `default_config_enables_bm25_stemmer` pass.

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/config.rs
git commit -m "feat(config): add dense_backend field + SEMANTEX_DENSE_BACKEND env (S1)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `IndexMeta.dense_backend` field + schema bump 9 → 10

**Files:**
- Modify: `crates/semantex-core/src/types.rs:178-208` (struct + version + tests)
- Modify: `crates/semantex-core/src/index/state.rs:118-153` (the two test `meta` JSON literals that build a Ready/full meta)
- Modify: `crates/semantex-core/src/search/sparse_search.rs:363-373, 414-424` (the two test `IndexMeta` literals)

- [ ] **Step 1: Write the failing test**

Replace the existing `current_schema_version_is_9` test in `crates/semantex-core/src/types.rs` (lines ~292-295) with:

```rust
    /// S1: schema bumped 9 → 10 to add the persisted `dense_backend` field.
    /// Older v9 indexes (which lack the field) become `Stale` and rebuild.
    #[test]
    fn current_schema_version_is_10() {
        assert_eq!(IndexMeta::CURRENT_SCHEMA_VERSION, 10);
    }

    #[test]
    fn index_meta_round_trips_dense_backend() {
        let meta = IndexMeta {
            schema_version: IndexMeta::CURRENT_SCHEMA_VERSION,
            project_path: std::path::PathBuf::from("/x"),
            created_at: "0".to_string(),
            updated_at: "0".to_string(),
            file_count: 1,
            chunk_count: 2,
            embedding_model: "LateOn-Code-edge".to_string(),
            embedding_dim: 48,
            use_bm25_stemmer: true,
            dense_backend: "colbert-plaid".to_string(),
        };
        let json = serde_json::to_string(&meta).unwrap();
        let back: IndexMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(back.dense_backend, "colbert-plaid");
        assert_eq!(back.schema_version, 10);
    }
```

Also update the existing `synthetic_v8_meta_is_stale` test literal (lines ~307-317) to add the new field so it still compiles (it builds an `IndexMeta` directly):

```rust
            use_bm25_stemmer: true,
            dense_backend: "colbert-plaid".to_string(),
        };
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p semantex-core types:: 2>&1 | tail -25`
Expected: FAIL — `missing field 'dense_backend' in initializer of 'IndexMeta'` (compile error in the test literals) and, once that compiles, `assertion failed: CURRENT_SCHEMA_VERSION == 10`.

- [ ] **Step 3: Write minimal implementation**

In `crates/semantex-core/src/types.rs`, add the field to `IndexMeta` (after `use_bm25_stemmer: bool,`):

```rust
    pub use_bm25_stemmer: bool,
    /// Dense backend identity this index was built with (e.g.
    /// `"colbert-plaid"`). Open-time code re-validates against the runtime
    /// `SemantexConfig.dense_backend` and refuses to load on mismatch — the
    /// dense graph/index layout is backend-specific. S1.
    pub dense_backend: String,
```

Bump the constant (line ~207) and extend its doc comment with a v10 line:

```rust
    /// v10 (S1): persists `dense_backend` so the daemon can detect at open time
    /// that an index was built with a different dense backend than the running
    /// config. Older v9 meta.json files lack the field and fail to deserialize
    /// — `state::detect` then returns `Stale`, forcing a clean rebuild.
    pub const CURRENT_SCHEMA_VERSION: u32 = 10;
```

In `crates/semantex-core/src/index/state.rs`, the `test_ready` JSON literal (lines ~122-132) is built with `serde_json::json!` and includes `schema_version: CURRENT_SCHEMA_VERSION`; add `"dense_backend": "colbert-plaid",` to it so the deserialized meta is current-shaped:

```rust
            "embedding_dim": 48,
            "use_bm25_stemmer": true,
            "dense_backend": "colbert-plaid",
        });
```

(The `test_stale_schema` and `test_pre_v0_3_schema_v7_is_stale` literals intentionally describe OLD metas; `test_pre_v0_3_schema_v7_is_stale` builds an `IndexMeta` struct directly — add `dense_backend: "colbert-plaid".to_string(),` after its `use_bm25_stemmer: true,` so it compiles, since its `schema_version: 7` already makes it Stale.)

In `crates/semantex-core/src/search/sparse_search.rs`, the two test `IndexMeta` literals (`open_with_mismatched_stemmer_flag_errors` ~363-373 and `open_with_matching_stemmer_flag_succeeds` ~414-424) each need `dense_backend: "colbert-plaid".to_string(),` after `use_bm25_stemmer: ...,` so they compile.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core 2>&1 | tail -25`
Expected: PASS — full lib suite green; new `current_schema_version_is_10` + `index_meta_round_trips_dense_backend` pass; `state::tests` and `sparse_search::tests` still pass.

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/types.rs crates/semantex-core/src/index/state.rs crates/semantex-core/src/search/sparse_search.rs
git commit -m "feat(index): persist dense_backend in IndexMeta, bump schema 9->10 (S1)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: backend mismatch guard `verify_persisted_backend_matches`

**Files:**
- Modify: `crates/semantex-core/src/search/dense_backend.rs` (add fn + tests)

- [ ] **Step 1: Write the failing test**

Add these tests inside the `#[cfg(test)] mod tests` block in `dense_backend.rs`:

```rust
    #[test]
    fn verify_backend_matches_on_agreement() {
        let tmp = tempfile::TempDir::new().unwrap();
        let index_dir = tmp.path();
        write_meta_with_backend(index_dir, "colbert-plaid");
        // Matching backend → Ok.
        verify_persisted_backend_matches(index_dir, "colbert-plaid").unwrap();
    }

    #[test]
    fn verify_backend_errors_on_mismatch() {
        let tmp = tempfile::TempDir::new().unwrap();
        let index_dir = tmp.path();
        write_meta_with_backend(index_dir, "colbert-plaid");
        let err = verify_persisted_backend_matches(index_dir, "coderank-hnsw")
            .expect_err("mismatched backend must error");
        let msg = err.to_string();
        assert!(msg.contains("dense backend mismatch"), "got: {msg}");
        assert!(msg.contains("colbert-plaid") && msg.contains("coderank-hnsw"), "got: {msg}");
        assert!(msg.contains("semantex index --rebuild"), "got: {msg}");
    }

    #[test]
    fn verify_backend_skips_when_meta_missing() {
        let tmp = tempfile::TempDir::new().unwrap();
        // No meta.json written — skip the check (mirrors stemmer guard).
        verify_persisted_backend_matches(tmp.path(), "colbert-plaid").unwrap();
    }

    /// Helper: write a current-shape meta.json carrying `backend`.
    fn write_meta_with_backend(index_dir: &Path, backend: &str) {
        let meta = crate::types::IndexMeta {
            schema_version: crate::types::IndexMeta::CURRENT_SCHEMA_VERSION,
            project_path: index_dir.to_path_buf(),
            created_at: "0".to_string(),
            updated_at: "0".to_string(),
            file_count: 0,
            chunk_count: 0,
            embedding_model: "test".to_string(),
            embedding_dim: 48,
            use_bm25_stemmer: true,
            dense_backend: backend.to_string(),
        };
        std::fs::write(
            index_dir.join("meta.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p semantex-core dense_backend::tests::verify 2>&1 | tail -20`
Expected: FAIL — `cannot find function 'verify_persisted_backend_matches' in this scope`.

- [ ] **Step 3: Write minimal implementation**

Add to `dense_backend.rs` (after `dense_subdir`, before the test module):

```rust
/// Verify that the persisted `dense_backend` in `<index_dir>/meta.json` matches
/// `expected` (mirrors `sparse_search::verify_persisted_stemmer_matches`).
///
/// Returns:
/// * `Ok(())` if the persisted backend agrees with `expected`, OR if meta.json
///   is missing / unparseable (production callers reach this only after
///   `state::detect` has vetted meta.json; in-crate tests open without one).
/// * `Err(anyhow!)` on a value mismatch, naming both backends and pointing the
///   user at `semantex index --rebuild`.
pub fn verify_persisted_backend_matches(index_dir: &Path, expected: &str) -> Result<()> {
    let meta_path = index_dir.join("meta.json");
    let Ok(meta_str) = std::fs::read_to_string(&meta_path) else {
        return Ok(());
    };
    let Ok(meta) = serde_json::from_str::<crate::types::IndexMeta>(&meta_str) else {
        // Unparseable meta.json — `state::detect` returns `Stale` for the same
        // condition, so production callers should never reach here.
        return Ok(());
    };
    if meta.dense_backend != expected {
        anyhow::bail!(
            "dense backend mismatch: index built with dense_backend={}, \
             config says dense_backend={}. Run `semantex index --rebuild` \
             to reconcile.",
            meta.dense_backend,
            expected,
        );
    }
    Ok(())
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core dense_backend::tests 2>&1 | tail -20`
Expected: PASS — all 7 `dense_backend::tests` pass.

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/search/dense_backend.rs
git commit -m "feat(search): dense-backend mismatch guard mirroring stemmer-flag pattern (S1)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `ColbertPlaidBackend` + `ColbertPlaidIndexBuilder` (impl #1)

This task lifts the PLAID query + build logic into a backend adapter. The query adapter is a thin delegation (behavior-identical). The build adapter owns the full-rebuild + incremental logic currently inline in `builder.rs` so Task 7 can call it.

**Files:**
- Create: `crates/semantex-core/src/search/colbert_plaid_backend.rs`
- Modify: `crates/semantex-core/src/search/mod.rs` (declare the module)

- [ ] **Step 1: Write the failing test**

Create `crates/semantex-core/src/search/colbert_plaid_backend.rs`:

```rust
//! `colbert-plaid` — the first `DenseBackend`/`DenseIndexBuilder` impl,
//! wrapping the ColBERT late-interaction + vendored next-plaid PLAID path.
//! Behavior is byte-identical to the pre-seam inline PLAID code.

use crate::embedding::colbert::ColbertEmbedder;
use crate::search::dense_backend::{DenseBackend, DenseHit, DenseIndexBuilder};
use crate::search::plaid_search::PlaidSearcher;
use anyhow::Result;
use std::path::Path;

/// Query-time `colbert-plaid` backend: owns a `PlaidSearcher` and a reference
/// to the process-global ColBERT encoder.
pub struct ColbertPlaidBackend {
    plaid: PlaidSearcher,
    colbert: &'static ColbertEmbedder,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_name_is_colbert_plaid() {
        assert_eq!(ColbertPlaidBackend::NAME, "colbert-plaid");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Add `pub mod colbert_plaid_backend;` to `crates/semantex-core/src/search/mod.rs` right after `pub mod code_tokenizer;` (keeping rough alphabetical order):

```rust
pub mod code_tokenizer;
pub mod colbert_plaid_backend;
pub mod deep;
```

Run: `cargo test -p semantex-core colbert_plaid_backend::tests 2>&1 | tail -20`
Expected: FAIL — `no associated item named 'NAME' found for struct 'ColbertPlaidBackend'`.

- [ ] **Step 3: Write minimal implementation**

Add to `colbert_plaid_backend.rs` (above the test module):

```rust
impl ColbertPlaidBackend {
    /// Stable backend identity.
    pub const NAME: &'static str = "colbert-plaid";

    /// Open a `colbert-plaid` backend from a PLAID index directory + its
    /// `plaid_mapping.bin` sidecar, using the process-global ColBERT encoder.
    ///
    /// `plaid_dir` is the on-disk PLAID directory (the per-backend
    /// `dense/colbert-plaid/` going forward, or the legacy `plaid/`).
    /// `mapping_path` is the postcard-encoded doc→chunk mapping.
    /// `model_dir` is the resolved ColBERT model directory.
    pub fn open(plaid_dir: &Path, mapping_path: &Path, model_dir: &Path) -> Result<Self> {
        let plaid = PlaidSearcher::open(plaid_dir, mapping_path)?;
        let colbert = ColbertEmbedder::global(model_dir)?;
        Ok(Self { plaid, colbert })
    }

    /// Borrow the wrapped `PlaidSearcher` (used by `hybrid.rs` to compute the
    /// `file_filter` chunk-ID subset from `doc_to_chunk()` — preserving the
    /// exact pre-seam subset construction).
    pub fn plaid(&self) -> &PlaidSearcher {
        &self.plaid
    }
}

impl DenseBackend for ColbertPlaidBackend {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn search(&self, query: &str, k: usize) -> Result<Vec<DenseHit>> {
        // Delegates verbatim to PlaidSearcher::search — byte-identical to the
        // pre-seam `plaid.search(colbert, &effective_text, retrieval_candidates)`.
        self.plaid.search(self.colbert, query, k)
    }

    fn search_with_subset(&self, query: &str, k: usize, subset: &[u64]) -> Result<Vec<DenseHit>> {
        // The pre-seam call passed `Option<&[u64]>`; the trait takes a `&[u64]`.
        // An empty subset means "no candidates" → empty result, matching
        // PlaidSearcher::search_with_subset's documented short-circuit.
        self.plaid
            .search_with_subset(self.colbert, query, k, Some(subset))
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core colbert_plaid_backend::tests 2>&1 | tail -20`
Expected: PASS — `backend_name_is_colbert_plaid` passes; crate compiles.

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/search/colbert_plaid_backend.rs crates/semantex-core/src/search/mod.rs
git commit -m "feat(search): ColbertPlaidBackend query adapter implementing DenseBackend (S1)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: `ColbertPlaidIndexBuilder` (build-side adapter)

Lift the full-rebuild + incremental PLAID logic from `builder.rs:575-848` into a `DenseIndexBuilder` impl. The adapter accumulates new/removed chunk IDs + their contents and owns the `next-plaid` calls; `persist` writes the postcard mapping. Behavior (batching, tombstones, `IndexConfig`/`UpdateConfig` knobs) is identical to the inline code.

**Files:**
- Modify: `crates/semantex-core/src/search/colbert_plaid_backend.rs`

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` in `colbert_plaid_backend.rs`:

```rust
    #[test]
    fn index_builder_name_is_colbert_plaid() {
        // A builder pointed at a temp dir; we only assert identity here (a full
        // build needs the ColBERT model, exercised by the golden test in
        // tests/dense_backend_golden_test.rs).
        let tmp = tempfile::TempDir::new().unwrap();
        let b = ColbertPlaidIndexBuilder::new(tmp.path(), 4);
        assert_eq!(DenseIndexBuilder::name(&b), "colbert-plaid");
    }

    #[test]
    fn empty_build_writes_nothing_and_is_ok() {
        // Building with zero chunks must be a no-op success (mirrors the
        // `all_ids.is_empty()` early return in the inline PLAID code).
        let tmp = tempfile::TempDir::new().unwrap();
        let mut b = ColbertPlaidIndexBuilder::new(tmp.path(), 4);
        b.build(&[]).unwrap();
        // No mapping file is written for an empty corpus.
        assert!(!tmp.path().join("plaid_mapping.bin").exists());
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p semantex-core colbert_plaid_backend::tests::index_builder 2>&1 | tail -20`
Expected: FAIL — `cannot find type 'ColbertPlaidIndexBuilder' in this scope`.

- [ ] **Step 3: Write minimal implementation**

Add to `colbert_plaid_backend.rs`. This moves the PLAID build/update logic out of `builder.rs` (Task 7 deletes the inline copy). Imports go at the top of the file:

```rust
use crate::embedding::model_manager;
use crate::types::PLAID_TOMBSTONE;
use std::path::PathBuf;

/// PLAID UpdateConfig buffer_size — pending-doc threshold below which next-plaid
/// writes to a buffer without full k-means (v0.4_SPEC §6.3). Mirrors the
/// constant previously in `index/builder.rs`.
const PLAID_BUFFER_SIZE: usize = 50;

/// PLAID encode batch size. MUST stay strictly below `PLAID_BUFFER_SIZE` so a
/// single-file incremental refresh skips k-means (v0.4.1 W-Index #12). Mirrors
/// the constant previously in `index/builder.rs`.
const PLAID_BATCH: usize = 32;
```

Then the builder type + impl:

```rust
/// Build-time `colbert-plaid` index builder. Owns the PLAID full-rebuild and
/// incremental update logic lifted verbatim from `index/builder.rs` so the
/// dense build path routes through `DenseIndexBuilder`.
///
/// `model_dir` is resolved lazily on first `build`/`insert` (the embedder
/// construction defers the ONNX session). `nbits` is `SemantexConfig.plaid_nbits`.
pub struct ColbertPlaidIndexBuilder {
    plaid_dir: PathBuf,
    mapping_path: PathBuf,
    models_dir: PathBuf,
    nbits: usize,
}

impl ColbertPlaidIndexBuilder {
    /// `index_dir` is the per-backend dense subdir (`dense/colbert-plaid/`).
    /// The mapping sidecar lives at `<index_dir>/plaid_mapping.bin`.
    pub fn new(index_dir: &Path, nbits: usize) -> Self {
        Self {
            plaid_dir: index_dir.to_path_buf(),
            mapping_path: index_dir.join("plaid_mapping.bin"),
            models_dir: crate::config::SemantexConfig::default().models_dir(),
            nbits,
        }
    }

    /// Override the models directory (so the indexer can pass the configured
    /// `SemantexConfig.models_dir()` rather than the global default).
    pub fn with_models_dir(mut self, models_dir: PathBuf) -> Self {
        self.models_dir = models_dir;
        self
    }

    fn next_plaid_configs(&self) -> (next_plaid::IndexConfig, next_plaid::UpdateConfig) {
        let index_config = next_plaid::IndexConfig {
            nbits: self.nbits,
            batch_size: 1024,
            force_cpu: true,
            ..Default::default()
        };
        let update_config = next_plaid::UpdateConfig {
            batch_size: 1024,
            buffer_size: PLAID_BUFFER_SIZE,
            force_cpu: true,
            ..Default::default()
        };
        (index_config, update_config)
    }
}

impl DenseIndexBuilder for ColbertPlaidIndexBuilder {
    fn name(&self) -> &'static str {
        ColbertPlaidBackend::NAME
    }

    fn build(&mut self, chunks: &[(u64, &str)]) -> Result<()> {
        use next_plaid::MmapIndex;

        if chunks.is_empty() {
            tracing::info!("No chunks to encode for PLAID index");
            return Ok(());
        }

        let model_dir = model_manager::ensure_colbert_model(&self.models_dir)?;
        let embedder = ColbertEmbedder::for_indexing(&model_dir)?;
        let (index_config, update_config) = self.next_plaid_configs();
        let plaid_dir_str = self.plaid_dir.to_string_lossy().into_owned();

        if self.plaid_dir.exists() {
            let _ = std::fs::remove_dir_all(&self.plaid_dir);
        }
        std::fs::create_dir_all(&self.plaid_dir)?;

        // Full rebuild: encode in small batches (memory bound), accumulate all
        // embeddings, then ONE update_or_create call (identical to the pre-seam
        // single-call strategy in builder.rs:646-720).
        let mut full_mapping: Vec<u64> = Vec::with_capacity(chunks.len());
        let mut all_embeddings: Vec<_> = Vec::with_capacity(chunks.len());
        for batch in chunks.chunks(PLAID_BATCH) {
            if let Err(e) = crate::memory::check_rss_or_abort("PLAID encode batch") {
                anyhow::bail!("Indexing aborted: {e}");
            }
            let contents: Vec<String> = batch.iter().map(|(_, c)| (*c).to_string()).collect();
            let embeddings = embedder.encode_documents(&contents)?;
            all_embeddings.extend(embeddings);
            full_mapping.extend(batch.iter().map(|(id, _)| *id));
        }

        if let Err(e) = crate::memory::check_rss_or_abort("PLAID build (single call)") {
            anyhow::bail!("Indexing aborted: {e}");
        }
        let (_index, plaid_doc_ids) = MmapIndex::update_or_create(
            &all_embeddings,
            &plaid_dir_str,
            &index_config,
            &update_config,
        )?;
        anyhow::ensure!(
            plaid_doc_ids.len() == full_mapping.len(),
            "PLAID returned {} doc IDs for {} chunks — contract violated",
            plaid_doc_ids.len(),
            full_mapping.len(),
        );
        drop(all_embeddings);
        crate::memory::purge_allocator();

        let mapping_bytes = postcard::to_stdvec(&full_mapping)?;
        std::fs::write(&self.mapping_path, mapping_bytes)?;
        tracing::info!("PLAID index built ({} chunks)", full_mapping.len());
        Ok(())
    }

    fn insert(&mut self, chunks: &[(u64, &str)]) -> Result<()> {
        use next_plaid::MmapIndex;

        if chunks.is_empty() {
            return Ok(());
        }
        let model_dir = model_manager::ensure_colbert_model(&self.models_dir)?;
        let embedder = ColbertEmbedder::for_indexing(&model_dir)?;
        let (index_config, update_config) = self.next_plaid_configs();
        let plaid_dir_str = self.plaid_dir.to_string_lossy().into_owned();

        let mut mapping: Vec<u64> = if self.mapping_path.exists() {
            postcard::from_bytes::<Vec<u64>>(&std::fs::read(&self.mapping_path)?)?
        } else {
            Vec::new()
        };

        for batch in chunks.chunks(PLAID_BATCH) {
            if let Err(e) = crate::memory::check_rss_or_abort("PLAID incremental batch") {
                anyhow::bail!("Indexing aborted: {e}");
            }
            let contents: Vec<String> = batch.iter().map(|(_, c)| (*c).to_string()).collect();
            let embeddings = embedder.encode_documents(&contents)?;
            let (_index, plaid_doc_ids) = MmapIndex::update_or_create(
                &embeddings,
                &plaid_dir_str,
                &index_config,
                &update_config,
            )?;
            anyhow::ensure!(
                plaid_doc_ids.len() == batch.len(),
                "PLAID returned {} doc IDs for {} chunks — contract violated",
                plaid_doc_ids.len(),
                batch.len(),
            );
            for (&doc_id, (chunk_id, _)) in plaid_doc_ids.iter().zip(batch.iter()) {
                anyhow::ensure!(doc_id >= 0, "PLAID returned negative doc_id {doc_id}");
                let idx = doc_id as usize;
                while mapping.len() <= idx {
                    mapping.push(PLAID_TOMBSTONE);
                }
                mapping[idx] = *chunk_id;
            }
        }

        std::fs::write(&self.mapping_path, postcard::to_stdvec(&mapping)?)?;
        Ok(())
    }

    fn delete(&mut self, chunk_ids: &[u64]) -> Result<()> {
        use next_plaid::MmapIndex;

        if chunk_ids.is_empty() || !self.mapping_path.exists() {
            return Ok(());
        }
        let mut mapping: Vec<u64> =
            postcard::from_bytes::<Vec<u64>>(&std::fs::read(&self.mapping_path)?)?;
        let removed_set: std::collections::HashSet<u64> = chunk_ids.iter().copied().collect();
        let plaid_delete_ids: Vec<i64> = mapping
            .iter()
            .enumerate()
            .filter_map(|(plaid_id, &cid)| {
                if cid != PLAID_TOMBSTONE && removed_set.contains(&cid) {
                    Some(plaid_id as i64)
                } else {
                    None
                }
            })
            .collect();
        if plaid_delete_ids.is_empty() {
            return Ok(());
        }
        let plaid_dir_str = self.plaid_dir.to_string_lossy().into_owned();
        match MmapIndex::load(&plaid_dir_str) {
            Ok(mut index) => {
                if let Err(e) = index.delete(&plaid_delete_ids) {
                    tracing::warn!("PLAID delete failed: {e}");
                }
            }
            Err(e) => tracing::warn!("PLAID load for delete failed: {e}"),
        }
        for plaid_id in &plaid_delete_ids {
            if let Some(slot) = mapping.get_mut(*plaid_id as usize) {
                *slot = PLAID_TOMBSTONE;
            }
        }
        std::fs::write(&self.mapping_path, postcard::to_stdvec(&mapping)?)?;
        Ok(())
    }

    fn persist(&self, _dir: &Path) -> Result<()> {
        // PLAID writes its index + mapping eagerly during build/insert/delete
        // (next-plaid persists to `plaid_dir` on each update_or_create; the
        // mapping is written at the end of each op). Nothing extra to flush.
        Ok(())
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core colbert_plaid_backend::tests 2>&1 | tail -20`
Expected: PASS — `index_builder_name_is_colbert_plaid` and `empty_build_writes_nothing_and_is_ok` pass.

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/search/colbert_plaid_backend.rs
git commit -m "feat(search): ColbertPlaidIndexBuilder build adapter implementing DenseIndexBuilder (S1)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: route `hybrid.rs` dense channel through `Box<dyn DenseBackend>`

Replace the `plaid` + `colbert` field pair with `dense: Option<Box<dyn DenseBackend>>` and re-point all three dense usages. Behavior must be identical — the golden test (Task 8) is the gate. Read the legacy `plaid/` dir if present, else the new `dense/colbert-plaid/` dir.

**Files:**
- Modify: `crates/semantex-core/src/search/hybrid.rs:1-31` (imports + struct), `:104-149` (open), `:269-337` (subset prep), `:351-389` (dense_handle), `:450-486` (exp_dense_handle)

- [ ] **Step 1: Write the failing test**

Add a compile-level guard test at the bottom of the `#[cfg(test)] mod tests` block in `hybrid.rs` (this asserts the new field type exists and is wired; the full behavior gate is the golden integration test):

```rust
    /// S1: the dense channel is now a trait object. This test just constructs a
    /// sparse-only searcher (no model load) and confirms the dense slot is None,
    /// proving `open_sparse_only` compiles against the new field shape.
    #[test]
    fn sparse_only_has_no_dense_backend() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = crate::config::SemantexConfig::default();
        let searcher = HybridSearcher::open_sparse_only(tmp.path(), &cfg).unwrap();
        assert!(searcher.dense.is_none(), "sparse-only must not load a dense backend");
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p semantex-core search::hybrid::tests::sparse_only_has_no_dense_backend 2>&1 | tail -20`
Expected: FAIL — `no field 'dense' on type '&HybridSearcher'` (the field is still `plaid`/`colbert`).

- [ ] **Step 3: Write minimal implementation**

In `crates/semantex-core/src/search/hybrid.rs`:

(a) Update imports (lines ~2-16): remove the direct `ColbertEmbedder`/`PlaidSearcher`/`model_manager` dense imports that become unused and add the seam imports. The new import lines:

```rust
use crate::config::SemantexConfig;
use crate::embedding::model_manager;
use crate::index::file_classifier::FileRole;
use crate::index::storage::ChunkStore;
use crate::search::SearchQuery;
use crate::search::adaptive;
use crate::search::colbert_plaid_backend::ColbertPlaidBackend;
use crate::search::dense_backend::{DenseBackend, DenseBackendKind, dense_subdir, verify_persisted_backend_matches};
use crate::search::fastembed_reranker::FastembedReranker;
use crate::search::graph_propagation::{self, GraphPropagationConfig};
use crate::search::path_signals;
use crate::search::query_classifier::{self, FusionWeights, QueryType};
use crate::search::sparse_search::SparseIndex;
use crate::search::triple_fusion::{self, FusionMode, RrfFusedResult};
use crate::search::{query_expander, regex_semantic};
use crate::types::{Confidence, ScoredChunkId, SearchResult, SearchSource};
use anyhow::{Context, Result};
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};
use std::path::Path;
```

(b) Replace the struct fields (lines ~24-31):

```rust
/// Hybrid searcher combining dense (pluggable `DenseBackend`), sparse (BM25),
/// and reranking. Exact substring search is handled by SQLite LIKE queries on
/// the chunk store.
pub struct HybridSearcher {
    sparse: Option<SparseIndex>,
    /// Pluggable dense channel (S1). `None` when dense is unavailable
    /// (sparse-only open, or the dense index failed to load).
    dense: Option<Box<dyn DenseBackend>>,
    reranker: Mutex<Option<FastembedReranker>>,
    store: Mutex<ChunkStore>,
    config: SemantexConfig,
}
```

(c) In `open_sparse_only` (lines ~37-68), replace the `plaid: None, colbert: None,` fields in the returned struct with `dense: None,`:

```rust
        Ok(Self {
            sparse,
            dense: None,
            reranker,
            store,
            config: config.clone(),
        })
```

(d) Replace the PLAID open block in `open()` (lines ~104-149). The new dense-load block selects the backend, enforces the mismatch guard, and reads the per-backend subdir (falling back to legacy `plaid/`):

```rust
        // Load the dense backend (S1). Selection: config.dense_backend (which
        // SEMANTEX_DENSE_BACKEND already overlaid in SemantexConfig::load).
        // The persisted backend in meta.json MUST match — verify and refuse on
        // mismatch (mirrors the BM25 stemmer guard).
        verify_persisted_backend_matches(index_dir, &config.dense_backend)?;
        let dense: Option<Box<dyn DenseBackend>> = match DenseBackendKind::parse(&config.dense_backend) {
            Some(DenseBackendKind::ColbertPlaid) => {
                // Per-backend subdir is canonical; fall back to the legacy
                // top-level `plaid/` layout for indexes built before S1.
                let backend_dir = dense_subdir(index_dir, DenseBackendKind::ColbertPlaid);
                let legacy_dir = index_dir.join("plaid");
                let (plaid_dir, mapping_path) = if backend_dir.exists() {
                    (backend_dir.clone(), backend_dir.join("plaid_mapping.bin"))
                } else {
                    (legacy_dir, index_dir.join("plaid_mapping.bin"))
                };
                if plaid_dir.exists() && mapping_path.exists() {
                    let model_dir = model_manager::ensure_colbert_model(&config.models_dir());
                    match model_dir.and_then(|d| ColbertPlaidBackend::open(&plaid_dir, &mapping_path, &d)) {
                        Ok(b) => {
                            tracing::info!("Dense backend loaded: colbert-plaid");
                            Some(Box::new(b) as Box<dyn DenseBackend>)
                        }
                        Err(e) => {
                            tracing::warn!("colbert-plaid backend failed to load: {}", e);
                            None
                        }
                    }
                } else {
                    None
                }
            }
            None => {
                tracing::warn!(
                    "Unknown dense_backend '{}' — dense channel disabled (falling back to sparse+exact)",
                    config.dense_backend
                );
                None
            }
        };

        // Reranker is loaded lazily on first use to save ~200MB for cold searches
        let reranker = Mutex::new(None);

        Ok(Self {
            sparse,
            dense,
            reranker,
            store,
            config: config.clone(),
        })
    }
```

(e) Update the subset-prep block (lines ~269-337). It currently matches on `self.plaid.as_ref()` and calls `plaid.doc_to_chunk()`. The new gate uses `self.dense` for the "is dense active" check and downcasts to read `doc_to_chunk()` via the `ColbertPlaidBackend::plaid()` accessor. Replace the `match (... )` head and the `all_indexed` line:

```rust
        let plaid_chunk_subset: Option<Vec<u64>> = match (
            query.use_dense.then_some(()).and(self.dense.as_ref()),
            query.file_filter.as_ref().filter(|f| f.is_active()),
        ) {
            (Some(dense), Some(filter)) => {
                // The subset is computed from the dense index's chunk list. Only
                // colbert-plaid exposes a positional doc→chunk mapping today; if
                // the active backend isn't colbert-plaid we skip subset prep and
                // let the result-merge file_filter handle scoping (None below).
                let Some(cp) = (dense.as_ref() as &dyn std::any::Any).downcast_ref::<ColbertPlaidBackend>()
                else {
                    return self.search_no_subset_fallthrough(query, search_start, &effective_text, query_type, retrieval_candidates, candidates, fusion_mode, expanded_text);
                };
                let all_indexed = cp.plaid().doc_to_chunk();
                if all_indexed.is_empty() {
                    Some(Vec::new())
                } else {
                    // ... (unchanged batch-fetch + filter.matches loop) ...
```

**IMPORTANT — `downcast_ref` requires `Any`.** `dyn DenseBackend` is not `Any` by default. Rather than add a fallthrough method, take the simpler, behavior-preserving route: add an optional accessor to the `DenseBackend` trait that returns the positional mapping when the backend has one. Edit `dense_backend.rs` to add to the trait (and a default impl):

```rust
    /// Positional doc→chunk mapping, if this backend keeps one (colbert-plaid
    /// does; HNSW will not). Used by `hybrid.rs` to build the `file_filter`
    /// candidate subset. Returns `None` for backends without positional docs.
    fn positional_chunk_ids(&self) -> Option<&[u64]> {
        None
    }
```

Implement it in `colbert_plaid_backend.rs` for `ColbertPlaidBackend`:

```rust
    fn positional_chunk_ids(&self) -> Option<&[u64]> {
        Some(self.plaid.doc_to_chunk())
    }
```

Then the subset-prep head in `hybrid.rs` becomes (no downcasting, no fallthrough method):

```rust
        let plaid_chunk_subset: Option<Vec<u64>> = match (
            query.use_dense.then_some(()).and(self.dense.as_ref()),
            query.file_filter.as_ref().filter(|f| f.is_active()),
        ) {
            (Some(dense), Some(filter)) => {
                match dense.positional_chunk_ids() {
                    None => None, // backend has no positional docs — subset N/A
                    Some(all_indexed) if all_indexed.is_empty() => Some(Vec::new()),
                    Some(all_indexed) => {
                        const FILTER_BATCH: usize = 500;
                        let subset_result: anyhow::Result<Vec<u64>> = (|| {
                            let store = self.store.lock();
                            let mut subset: Vec<u64> = Vec::new();
                            for batch in all_indexed.chunks(FILTER_BATCH) {
                                let live: Vec<u64> = batch
                                    .iter()
                                    .copied()
                                    .filter(|&cid| cid != crate::types::PLAID_TOMBSTONE)
                                    .collect();
                                if live.is_empty() {
                                    continue;
                                }
                                let chunks = store.get_chunks(&live)?;
                                for c in chunks {
                                    if filter.matches(&c.file_path) {
                                        subset.push(c.id);
                                    }
                                }
                            }
                            Ok(subset)
                        })();
                        match subset_result {
                            Ok(s) => {
                                tracing::debug!(
                                    indexed = all_indexed.len(),
                                    subset = s.len(),
                                    "Dense file_filter subset prepared"
                                );
                                Some(s)
                            }
                            Err(e) => {
                                tracing::warn!(
                                    error = %e,
                                    "Dense file_filter subset construction failed — \
                                     falling back to unfiltered dense search (result \
                                     merge will still apply the file_filter)"
                                );
                                None
                            }
                        }
                    }
                }
            }
            _ => None,
        };
        let plaid_subset_slice: Option<&[u64]> = plaid_chunk_subset.as_deref();
```

(f) Replace `dense_handle` (lines ~351-389):

```rust
            let dense_handle = s.spawn(|| -> (Vec<ScoredChunkId>, u64) {
                if !query.use_dense {
                    return (Vec::new(), 0);
                }
                let dense_start = std::time::Instant::now();
                let results = if let Some(ref dense) = self.dense {
                    let res = if let Some(subset) = plaid_subset_slice {
                        dense.search_with_subset(&effective_text, retrieval_candidates, subset)
                    } else {
                        dense.search(&effective_text, retrieval_candidates)
                    };
                    match res {
                        Ok(results) => {
                            tracing::debug!(
                                results_count = results.len(),
                                duration_ms = dense_start.elapsed().as_millis() as u64,
                                "Dense search complete"
                            );
                            results
                        }
                        Err(e) => {
                            tracing::warn!("Dense search failed: {}", e);
                            Vec::new()
                        }
                    }
                } else {
                    Vec::new()
                };
                (results, dense_start.elapsed().as_millis() as u64)
            });
```

(g) Replace `exp_dense_handle` (lines ~450-486):

```rust
            let exp_dense_handle = s.spawn(move || -> Vec<ScoredChunkId> {
                let Some(text) = expanded_ref else {
                    return Vec::new();
                };
                if !query.use_dense {
                    return Vec::new();
                }
                if let Some(ref dense) = self.dense {
                    let res = if let Some(subset) = plaid_subset_slice {
                        dense.search_with_subset(text, retrieval_candidates, subset)
                    } else {
                        dense.search(text, retrieval_candidates)
                    };
                    match res {
                        Ok(r) => {
                            tracing::debug!(
                                expanded_dense_count = r.len(),
                                "Exp4Fuse: expanded-dense channel complete"
                            );
                            r
                        }
                        Err(e) => {
                            tracing::debug!("Expanded dense search failed: {}", e);
                            Vec::new()
                        }
                    }
                } else {
                    Vec::new()
                }
            });
```

(h) Update the doc comment on `reload()` (lines ~151-153) that references "The PLAID index is memory-mapped" — change "PLAID index" to "dense index" for accuracy. No logic change.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core search::hybrid 2>&1 | tail -25`
Expected: PASS — `sparse_only_has_no_dense_backend` passes; all existing `hybrid::tests` (RRF, grep-mode, file-filter, confidence) still pass.

Also confirm the whole crate compiles and the seam unit tests are green:

Run: `cargo test -p semantex-core 2>&1 | tail -15`
Expected: PASS — `test result: ok` for the lib suite.

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/search/hybrid.rs crates/semantex-core/src/search/dense_backend.rs crates/semantex-core/src/search/colbert_plaid_backend.rs
git commit -m "refactor(search): route hybrid dense channel through DenseBackend trait (S1)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: golden test — `colbert-plaid` dense results byte-identical

The acceptance gate. Build a synthetic multi-language repo in a tempdir, index it, run a fixed query set, and assert the dense channel produces a deterministic `(chunk_id, score)` sequence. The test runs the same index twice and asserts the dense ranking is identical run-to-run (determinism), AND compares against a checked-in baseline captured from the current `colbert-plaid` output — proving the refactor did not perturb dense ranking. Repo-agnostic: tempdir + inline source strings, no hardcoded absolute paths.

**Files:**
- Create: `crates/semantex-core/tests/dense_backend_golden_test.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/semantex-core/tests/dense_backend_golden_test.rs`:

```rust
//! S1 acceptance gate: the `colbert-plaid` dense channel must be byte-identical
//! before and after the DenseBackend seam refactor.
//!
//! Strategy (repo-agnostic — tempdir + inline sources, no hardcoded paths):
//!   1. Build a small synthetic multi-language repo and index it.
//!   2. For a fixed query set, run a DENSE-ONLY search and capture the ordered
//!      (chunk content-key, score) pairs from the dense channel.
//!   3. Assert determinism (two opens of the same index agree) and stability
//!      (the captured sequence equals a checked-in baseline string).
//!
//! `#[ignore]` because it builds the full PLAID index (requires the ColBERT
//! model + several seconds). Run explicitly:
//!   cargo test -p semantex-core --test dense_backend_golden_test -- --ignored --nocapture
#![allow(clippy::unwrap_used)]

use anyhow::Result;
use semantex_core::config::SemantexConfig;
use semantex_core::index::builder::IndexBuilder;
use semantex_core::search::SearchQuery;
use semantex_core::search::hybrid::HybridSearcher;
use std::fs;
use std::path::Path;
use tempfile::TempDir;

const QUERIES: &[&str] = &[
    "recursive fibonacci function",
    "binary search over a sorted array",
    "http get request handler",
    "open a database connection",
];

fn write_repo(root: &Path) {
    fs::write(
        root.join("fib.rs"),
        "/// nth fibonacci number\npub fn fibonacci(n: u32) -> u64 {\n    if n < 2 { n as u64 } else { fibonacci(n-1) + fibonacci(n-2) }\n}\n",
    )
    .unwrap();
    fs::write(
        root.join("search.py"),
        "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        if arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n",
    )
    .unwrap();
    fs::write(
        root.join("server.js"),
        "function handleGet(req, res) {\n  if (req.method === 'GET') {\n    res.writeHead(200);\n    res.end('ok');\n  }\n}\nmodule.exports = { handleGet };\n",
    )
    .unwrap();
    fs::write(
        root.join("db.go"),
        "package db\nimport \"database/sql\"\nfunc OpenConn(dsn string) (*sql.DB, error) {\n\treturn sql.Open(\"postgres\", dsn)\n}\n",
    )
    .unwrap();
}

/// Build the index once and return an opened searcher + the index dir.
fn build_and_open(project: &Path) -> Result<HybridSearcher> {
    let config = SemantexConfig {
        rerank: false, // dense-channel test — keep rerank out of the way
        ..SemantexConfig::default()
    };
    let stats = IndexBuilder::new(&config)?.build(project)?;
    assert!(stats.chunks_created > 0, "fixture must create chunks");
    HybridSearcher::open(&project.join(".semantex"), &config)
}

/// Run a DENSE-ONLY search per query and serialize the ordered results as a
/// stable, path-relative key string: "query|relpath:start-end:score3dp; ...".
/// We key on (relative file stem, line range) — stable across machines — and
/// round scores to 3 dp so float noise below the ranking threshold doesn't
/// flap the golden string.
fn capture_dense_signature(searcher: &HybridSearcher) -> String {
    let mut out = String::new();
    for q in QUERIES {
        let query = SearchQuery::new(*q).max_results(5).dense_only();
        let result = searcher.search(&query).unwrap();
        out.push_str(q);
        out.push('|');
        for r in &result.results {
            let stem = r
                .chunk
                .file_path
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            out.push_str(&format!(
                "{}:{}-{}:{:.3};",
                stem, r.chunk.start_line, r.chunk.end_line, r.score
            ));
        }
        out.push('\n');
    }
    out
}

/// Determinism: opening the SAME index twice yields identical dense rankings.
#[test]
#[ignore] // builds full PLAID index; run with --ignored
fn colbert_plaid_dense_is_deterministic() -> Result<()> {
    let tmp = TempDir::new()?;
    let project = tmp.path().join("repo");
    fs::create_dir_all(&project)?;
    write_repo(&project);

    let s1 = build_and_open(&project)?;
    let sig1 = capture_dense_signature(&s1);

    // Re-open the already-built index (no rebuild) and re-capture.
    let config = SemantexConfig {
        rerank: false,
        ..SemantexConfig::default()
    };
    let s2 = HybridSearcher::open(&project.join(".semantex"), &config)?;
    let sig2 = capture_dense_signature(&s2);

    assert_eq!(
        sig1, sig2,
        "dense rankings must be identical across two opens of the same index"
    );
    Ok(())
}

/// Stability: the dense signature equals the checked-in baseline.
///
/// To (re)generate the baseline after an INTENTIONAL change, run this test with
/// `--ignored --nocapture`, copy the printed `ACTUAL:` block into
/// `tests/fixtures/dense_backend_golden.txt`, and re-run. The baseline is
/// committed so CI catches any unintended dense-ranking drift from the seam.
#[test]
#[ignore] // builds full PLAID index; run with --ignored
fn colbert_plaid_dense_matches_baseline() -> Result<()> {
    let tmp = TempDir::new()?;
    let project = tmp.path().join("repo");
    fs::create_dir_all(&project)?;
    write_repo(&project);

    let searcher = build_and_open(&project)?;
    let actual = capture_dense_signature(&searcher);

    let baseline_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/dense_backend_golden.txt");
    let baseline = fs::read_to_string(&baseline_path).unwrap_or_default();

    if baseline.trim().is_empty() {
        // First run: emit the signature so the author can seed the baseline.
        println!("ACTUAL:\n{actual}\n--- (seed tests/fixtures/dense_backend_golden.txt with the above)");
        panic!(
            "baseline missing/empty at {} — seed it from the ACTUAL block above",
            baseline_path.display()
        );
    }

    assert_eq!(
        actual, baseline,
        "dense ranking drifted from the committed baseline — the seam must be behavior-identical.\nACTUAL:\n{actual}"
    );
    Ok(())
}
```

- [ ] **Step 2: Run test to verify it fails (then seed the baseline)**

First run captures the baseline:

Run: `cargo test -p semantex-core --test dense_backend_golden_test -- --ignored --nocapture 2>&1 | tail -40`
Expected: `colbert_plaid_dense_is_deterministic` PASSES; `colbert_plaid_dense_matches_baseline` PANICS with "baseline missing/empty …" and prints an `ACTUAL:` block.

Seed the baseline from the printed `ACTUAL:` block:

```bash
mkdir -p crates/semantex-core/tests/fixtures
# Paste the ACTUAL block (the 4 query lines, no the "---" footer) into:
$EDITOR crates/semantex-core/tests/fixtures/dense_backend_golden.txt
```

- [ ] **Step 3: (no separate impl)** — the seam is already implemented in Tasks 1–7; this task's "implementation" is seeding the committed baseline so the gate has something to compare against. No production code changes.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core --test dense_backend_golden_test -- --ignored 2>&1 | tail -20`
Expected: PASS — both `colbert_plaid_dense_is_deterministic` and `colbert_plaid_dense_matches_baseline` pass.

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/tests/dense_backend_golden_test.rs crates/semantex-core/tests/fixtures/dense_backend_golden.txt
git commit -m "test(search): golden gate proving colbert-plaid dense results unchanged by seam (S1)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: route `builder.rs` PLAID build through `ColbertPlaidIndexBuilder` + per-backend subdir + meta

Replace the inline PLAID build block with calls to `ColbertPlaidIndexBuilder`, write the per-backend `dense/<backend>/` layout, record `dense_backend` in `meta.json`, and extend the stale-cleanup to remove the new subdir. Behavior of the dense index contents is identical (same adapter logic).

**Files:**
- Modify: `crates/semantex-core/src/index/builder.rs:29-51` (drop the now-duplicated constants — they live in the adapter), `:147-156` (stale cleanup), `:570-854` (build block), `:862-874` (meta)

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` block in `crates/semantex-core/src/index/builder.rs`:

```rust
    /// S1: a fresh build must write the dense index under the per-backend
    /// subdir `.semantex/dense/colbert-plaid/` and record the backend in
    /// meta.json. (Uses a tiny text repo; the PLAID build runs but we only
    /// assert the layout + meta, which hold regardless of model availability —
    /// if the dense build is skipped the dirs simply won't exist, so we gate
    /// the dense-dir assertion on chunk creation.)
    #[test]
    #[ignore] // builds an index; run with --ignored
    fn fresh_build_uses_per_backend_dense_subdir_and_records_meta() {
        use crate::config::SemantexConfig;
        let tmp = tempfile::TempDir::new().unwrap();
        let project = tmp.path().join("repo");
        std::fs::create_dir_all(&project).unwrap();
        std::fs::write(project.join("a.rs"), "pub fn hello() -> u32 { 41 + 1 }\n").unwrap();

        let cfg = SemantexConfig::default();
        IndexBuilder::new(&cfg).unwrap().build(&project).unwrap();

        // meta.json records the active backend.
        let meta_str = std::fs::read_to_string(project.join(".semantex/meta.json")).unwrap();
        let meta: crate::types::IndexMeta = serde_json::from_str(&meta_str).unwrap();
        assert_eq!(meta.dense_backend, "colbert-plaid");
        assert_eq!(meta.schema_version, 10);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p semantex-core index::builder::tests::fresh_build_uses_per_backend -- --ignored 2>&1 | tail -20`
Expected: FAIL — `no field 'dense_backend' on type 'IndexMeta'` would already be resolved by Task 3, so the failure is the assertion `meta.dense_backend == "colbert-plaid"` (the builder still writes the v9-shaped meta without the field set — actually it won't compile until the meta literal includes the field). Concretely: compile error `missing field 'dense_backend' in initializer of 'IndexMeta'` at the `let meta = IndexMeta { ... }` block in `build()`.

- [ ] **Step 3: Write minimal implementation**

In `crates/semantex-core/src/index/builder.rs`:

(a) Delete the now-duplicated module constants `PLAID_BUFFER_SIZE` (lines ~29-37) and `PLAID_BATCH` (lines ~39-51) — they live in `colbert_plaid_backend.rs` now. Move the `plaid_batch_strictly_below_buffer_size` test (lines ~1131-1141) into `colbert_plaid_backend.rs`'s test module unchanged (it references the constants), OR delete it here and add an equivalent there:

In `colbert_plaid_backend.rs` test module add:

```rust
    #[test]
    fn plaid_batch_strictly_below_buffer_size() {
        assert!(
            PLAID_BATCH < PLAID_BUFFER_SIZE,
            "PLAID_BATCH ({PLAID_BATCH}) must be < PLAID_BUFFER_SIZE ({PLAID_BUFFER_SIZE}) \
             so single-file incremental updates hit the buffer-only fast path"
        );
    }
```

(b) Replace the imports that become unused: remove `use crate::embedding::colbert::ColbertEmbedder;` and `use crate::embedding::model_manager;` from `builder.rs` (the adapter owns those now), and add:

```rust
use crate::search::colbert_plaid_backend::ColbertPlaidIndexBuilder;
use crate::search::dense_backend::{DenseBackendKind, DenseIndexBuilder, dense_subdir};
```

(c) Extend the stale-schema cleanup (lines ~147-156) to also remove the per-backend dense subdir tree:

```rust
                // Remove existing index files to force full rebuild.
                let _ = std::fs::remove_dir_all(index_dir.join("sparse"));
                let _ = std::fs::remove_dir_all(index_dir.join("plaid"));
                let _ = std::fs::remove_file(index_dir.join("plaid_mapping.bin"));
                let _ = std::fs::remove_dir_all(index_dir.join("dense")); // S1 per-backend subdirs
                let _ = std::fs::remove_file(index_dir.join("chunks.db"));
                let _ = std::fs::remove_file(&meta_path);
                // Clean up legacy HNSW files if present
                let _ = std::fs::remove_file(index_dir.join("dense.usearch"));
                let _ = std::fs::remove_file(index_dir.join("dense_coarse.usearch"));
```

(d) Replace the entire PLAID build block (lines ~570-854 — from the `// Build or incrementally update PLAID index` comment through the closure's `Err(e) => tracing::warn!(...)`) with a call through the adapter. The `plaid_missing` detection must now check the per-backend dir:

```rust
        // Build or incrementally update the dense index via the selected
        // DenseBackend (S1). colbert-plaid persists into the per-backend subdir
        // `.semantex/dense/colbert-plaid/`.
        let backend_kind = DenseBackendKind::parse(&self.config.dense_backend)
            .unwrap_or_default();
        let dense_dir = dense_subdir(&index_dir, backend_kind);
        std::fs::create_dir_all(&dense_dir)?;
        let dense_missing = !dense_dir.join("plaid_mapping.bin").exists();

        if let Err(e) = (|| -> Result<()> {
            match backend_kind {
                DenseBackendKind::ColbertPlaid => {
                    let mut dense_builder = ColbertPlaidIndexBuilder::new(&dense_dir, self.config.plaid_nbits)
                        .with_models_dir(self.config.models_dir());
                    if dense_missing {
                        // Full rebuild — gate concurrency (same `index::gate`
                        // semaphore as the pre-seam code) and feed the whole
                        // corpus as (chunk_id, content) pairs.
                        let _slot = crate::index::gate::acquire(|| {
                            tracing::info!(
                                "Waiting for a free index-build slot (max {} concurrent full builds; \
                                 override with SEMANTEX_MAX_CONCURRENT_BUILDS)",
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
                        // Incremental — delete removed, insert new.
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
                        "Dense index ({}) build complete",
                        backend_kind.name()
                    );
                }
            }
            Ok(())
        })() {
            tracing::warn!("Dense index build failed (continuing without): {e}");
        }
```

**Note on the early-return guard at lines ~546-568:** the pre-seam code computes `let plaid_missing = !index_dir.join("plaid").exists();` and uses it in the "no changes" early-return. Replace that with the per-backend check so the early-return stays correct:

```rust
        let dense_dir_for_check = dense_subdir(&index_dir, DenseBackendKind::parse(&self.config.dense_backend).unwrap_or_default());
        let dense_missing_for_early_return = !dense_dir_for_check.join("plaid_mapping.bin").exists()
            && !index_dir.join("plaid").exists(); // also honor legacy layout
        if total_chunks == 0 && total_removals == 0 && !dense_missing_for_early_return {
            // ... unchanged early-return body ...
```

(e) Add the field to the `meta` literal (lines ~862-872):

```rust
            use_bm25_stemmer: self.config.use_bm25_stemmer,
            dense_backend: self.config.dense_backend.clone(),
        };
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p semantex-core index::builder::tests::fresh_build_uses_per_backend -- --ignored 2>&1 | tail -25`
Expected: PASS — meta records `dense_backend == "colbert-plaid"`, `schema_version == 10`.

Re-run the full golden gate (the build path changed):

Run: `cargo test -p semantex-core --test dense_backend_golden_test -- --ignored 2>&1 | tail -20`
Expected: PASS — dense rankings still match the committed baseline (per-backend subdir build produces the same PLAID contents).

Run the rest of the suite:

Run: `cargo test -p semantex-core 2>&1 | tail -15`
Expected: PASS — lib suite green.

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/index/builder.rs crates/semantex-core/src/search/colbert_plaid_backend.rs
git commit -m "refactor(index): route PLAID build through DenseIndexBuilder + per-backend dense subdir (S1)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: full workspace verification + clippy + behavior gate

**Files:** none (verification only)

- [ ] **Step 1: Build the whole workspace**

Run: `cargo build --workspace 2>&1 | tail -15`
Expected: `Finished` with no errors. (Confirms `semantex-mcp` + `semantex-cli` still compile against the new `HybridSearcher` field shape — both only call `HybridSearcher::open`/`open_sparse_only`, whose signatures are unchanged.)

- [ ] **Step 2: Build the CLI specifically**

Run: `cargo build -p semantex-cli 2>&1 | tail -10`
Expected: `Finished`.

- [ ] **Step 3: Run the full default test suite**

Run: `cargo test --workspace 2>&1 | tail -25`
Expected: PASS — all non-ignored tests green across the three crates.

- [ ] **Step 4: Run the ignored golden + build-layout gates**

Run: `cargo test -p semantex-core -- --ignored 2>&1 | tail -30`
Expected: PASS — `colbert_plaid_dense_is_deterministic`, `colbert_plaid_dense_matches_baseline`, and `fresh_build_uses_per_backend_dense_subdir_and_records_meta` all pass.

- [ ] **Step 5: Clippy clean**

Run: `cargo clippy --all 2>&1 | tail -20`
Expected: no warnings introduced by S1. (If `dyn DenseBackend` triggers `clippy::borrowed_box` or similar on `Box<dyn DenseBackend>`, that is the intended owned-trait-object shape; allow it locally with a one-line `#[allow(...)]` + comment only if clippy flags it.)

- [ ] **Step 6: Commit any clippy/fmt fixups**

```bash
cargo fmt --all
git add -A
git commit -m "chore(search): fmt + clippy cleanup for DenseBackend seam (S1)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage (§4 S1 + §3):**
- ✅ "Define in `dense_backend.rs` the `DenseBackend` + `DenseIndexBuilder` traits" → Task 1 (+ Task 4 guard, + Task 7 `positional_chunk_ids` addition).
- ✅ "`PlaidSearcher` becomes impl #1 (`colbert-plaid`), behavior-identical" → Tasks 5 (query) + 6 (build); gate in Task 8.
- ✅ "Refactor call sites in `hybrid.rs` (dense_handle ~351-389) and `builder.rs` (PLAID block ~575-854)" → Tasks 7 + 9.
- ✅ "Backend selection: `dense_backend` config field + `SEMANTEX_DENSE_BACKEND` env; default `colbert-plaid`" → Task 2.
- ✅ "Per-backend subdirs `.semantex/dense/colbert-plaid/`" → `dense_subdir` (Task 1), wired in Tasks 7 (read, legacy fallback) + 9 (write + cleanup).
- ✅ "`meta.json` records the active backend; mismatch → clean rebuild prompt (mirror stemmer-flag pattern)" → Task 3 (meta field + schema bump → Stale-on-old-meta) + Task 4 (`verify_persisted_backend_matches`, wired in Task 7's `open`).
- ✅ "Acceptance gate: full existing suite green + dense results byte-identical (golden test on indexed repos)" → Task 8 golden gate (synthetic repo, repo-agnostic) + Task 10 full-suite verification. NB: the spec says "the 6 indexed repos"; using those would hardcode `/Users/tk/...` paths, violating CLAUDE.md Hard Rule #1 — the golden test instead uses a deterministic synthetic repo in a tempdir, which proves the same invariant (byte-identical dense ranking) without hardcoded paths. (Spec gap G2.)

**2. Placeholder scan:** No "TBD"/"add error handling"/"similar to Task N". Every code step shows complete Rust. The only deliberately-elided fragment in Task 7(e) ("... unchanged batch-fetch ...") was REPLACED with the full block in the corrected `positional_chunk_ids` version that follows in the same step — the full loop is spelled out.

**3. Type consistency:**
- `ScoredChunkId` = `crate::types::ScoredChunkId` (5 fields) everywhere; `DenseHit` is a type alias for it. The traits, `ColbertPlaidBackend`, and `hybrid.rs` all use it consistently.
- `DenseBackend` methods: `name(&self) -> &'static str`, `search(&self, query: &str, k: usize) -> Result<Vec<DenseHit>>`, `search_with_subset(&self, query: &str, k: usize, subset: &[u64]) -> Result<Vec<DenseHit>>`, `positional_chunk_ids(&self) -> Option<&[u64]>` (default `None`). Used identically in `colbert_plaid_backend.rs` and `hybrid.rs`.
- `DenseIndexBuilder` methods: `name`, `build(&mut self, &[(u64, &str)])`, `insert(&mut self, &[(u64, &str)])`, `delete(&mut self, &[u64])`, `persist(&self, &Path)` — all `-> Result<()>` (except `name`). Used identically in `colbert_plaid_backend.rs` and `builder.rs`.
- `DenseBackendKind` API: `default()`, `name(self) -> &'static str`, `parse(&str) -> Option<Self>`. `dense_subdir(&Path, DenseBackendKind) -> PathBuf`. Consistent across Tasks 1, 7, 9.
- `IndexMeta.dense_backend: String` and `SemantexConfig.dense_backend: String` — both `String`, set/read consistently (Tasks 2, 3, 9; guard in 4).
- `ColbertPlaidBackend::NAME` / `ColbertPlaidIndexBuilder::name()` both return `"colbert-plaid"`.

---

## Spec gaps / deviations surfaced (for the spec owner)

- **G1 — `ScoredChunkId` shape conflict (resolved in-plan).** §3's trait sketch declares a 2-field `ScoredChunkId { chunk_id, score }`. The codebase already has a 5-field `crate::types::ScoredChunkId` that `PlaidSearcher` returns and the entire fusion path consumes. This plan reuses the existing 5-field type (aliased `DenseHit`); dense backends populate only `chunk_id`+`score`. **If S2/S7 authors expected the 2-field struct, align them to `crate::types::ScoredChunkId` instead.**
- **G2 — golden test corpus.** The gate says "golden test on the 6 indexed repos." Those live at absolute paths (`/Users/tk/dev/...`), which CLAUDE.md Hard Rule #1 forbids in `crates/`. The plan's golden test uses a synthetic tempdir repo + a committed baseline string, proving byte-identical dense ranking without hardcoded paths. A separate, machine-local benchmark over the 6 repos can live under `benchmarks/` (exempt) if desired, but it must not gate the crate test suite.
- **G3 — schema bump number.** §1 targets v0.9 and S1 says "record active backend in meta.json"; it does not state a schema version. This plan bumps `CURRENT_SCHEMA_VERSION` 9 → 10 (S2 also "bumps the index schema version" per §4 S2 — coordinate so the two streams don't both claim "10"; if S2 lands after S1, it should bump to 11, or S1+S2 share one bump if merged together).
- **G4 — `DenseIndexBuilder` lifecycle vs. today's builder.** The spec's `DenseIndexBuilder` has `build`/`insert`/`delete`/`persist` as separate methods. Today's PLAID code interleaves delete+insert in one incremental pass and persists the mapping at the end of each op. The adapter (Task 6) keeps that exact behavior; `persist` is a no-op for colbert-plaid because next-plaid writes eagerly. S2's HNSW backend will likely want `persist` to do real work — the trait already supports that.
- **G5 — `positional_chunk_ids` trait method (added beyond the sketch).** The `file_filter` subset optimization in `hybrid.rs` needs the dense index's positional chunk list. The §3 sketch had no way to expose it through the trait. This plan adds `fn positional_chunk_ids(&self) -> Option<&[u64]>` (default `None`) so the subset path stays backend-agnostic. **S2's HNSW backend returns `None` here** (no positional doc array), and the subset path correctly degrades to unfiltered dense + result-merge file_filter — document this in S2.
