# LateOn-ColBERT opt-in dense backend — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add an opt-in multi-vector late-interaction dense backend (`SEMANTEX_EMBEDDER=lateon-colbert`) by resurrecting the `colbert-plaid` backend + vendored `next-plaid` crate that PR #1 removed, adapted to the post-`7a5663f` versioned-dir lifecycle, WITHOUT changing the `coderank-hnsw` default.

**Architecture:** Resurrect-and-adapt, not greenfield. PR #1 (`d996122`) removed a *working* `colbert-plaid` backend that already targeted the LateOn-Code-edge ONNX model. Commit `26e71ba` is the pre-removal state where both backends coexisted (coderank-hnsw default, colbert-plaid opt-in). The `DenseBackend`/`DenseIndexBuilder` trait seam is **byte-identical at HEAD** → the 486-LOC backend impl drops in unchanged; only its two construction sites (query: `hybrid.rs::open`; build: `index/builder.rs`) need adapting to the new `dense/<backend>/<fingerprint>/` + atomic `ACTIVE`-pointer lifecycle. The embedder spec id is `lateon-colbert` (`multi_vector=true`, `Pooling::LateInteraction`); it capability-routes to the `colbert-plaid` backend (on-disk `dense/colbert-plaid/`, sentinel `plaid_mapping.bin`).

**Tech Stack:** Rust (workspace edition 2024, rustc 1.91). Vendored `next-plaid 1.3.1` (Apache-2.0, pure-Rust PLAID over ndarray; no ort) + `next-plaid-onnx 1.3` (crates.io; ONNX ColBERT encoder, pulls `ort`). `ort 2.0.0-rc.11` (load-dynamic). Model: `lightonai/LateOn-Code-edge` (17M, dim-48, int8) cached at `~/.semantex/models/LateOn-Code-edge/`.

---

## DE-RISK — RESOLVED GREEN (do not re-litigate)

The single gating risk was the `ort` version clash (`next-plaid-onnx 1.3` pulls its own `ort`). **Proven compatible:**
- HEAD `ort` line == `26e71ba` `ort` line, byte-identical: `2.0.0-rc.11` (`crates/semantex-core/Cargo.toml:55`). `42c59f8` never touched ort; nothing bumped it since.
- At `26e71ba`, `next-plaid-onnx 1.3.1` resolved its `ort` dep to **exactly `2.0.0-rc.11`** (Cargo.lock checksum `4a5df903c0d2c07b56950f1058104ab0c8557159f2741782223704de9be73c3c` == HEAD's). They coexisted.
- `next-plaid-onnx 1.3.1` also pulls `tokenizers 0.21.4`; semantex-core uses `tokenizers 0.22`. Two minor versions coexist in the lock (built fine at `26e71ba`). Re-confirm with `cargo tree -i tokenizers` but it is additive, airgap-safe.

→ Task 1 still runs the real `cargo build` + `cargo tree -i ort` as the empirical proof, but the lock-file evidence says it WILL unify.

## Constraints (CLAUDE.md — hard rules)
- `coderank-hnsw` stays the DEFAULT. `lateon-colbert` is opt-in only. Do NOT change `config.rs` `dense_backend`/`embedder` defaults.
- Default build stays zero-LLM: `cargo tree | grep genai` empty (next-plaid/next-plaid-onnx pull no genai — verify anyway).
- `crates/` repo-agnostic, no hardcoded paths, permissive licenses (next-plaid Apache-2.0, next-plaid-onnx — verify; ort MIT/Apache).
- macOS harness/e2e: `ORT_DYLIB_PATH=/opt/homebrew/lib/libonnxruntime.dylib`.
- Full gate before the PR lands: `cargo build --workspace && cargo test --workspace && cargo clippy --all && cargo fmt --all --check` + `actually verify_change` + an e2e index/search smoke under `SEMANTEX_EMBEDDER=lateon-colbert`.

## File map (restore from `26e71ba` unless noted)
- Restore: `vendor/next-plaid/` (entire tree, ~12.4k LOC, 16 src files + 3 tests + Cargo.toml/Cargo.toml.orig/Cargo.lock/build.rs/README).
- Modify: root `Cargo.toml` (`[workspace.dependencies]` + `[patch.crates-io]`), `crates/semantex-core/Cargo.toml` (`[dependencies]`).
- Restore: `crates/semantex-core/src/embedding/colbert.rs` (442), `src/search/plaid_search.rs` (431), `src/search/colbert_plaid_backend.rs` (486).
- Modify: `src/embedding/mod.rs`, `src/search/mod.rs` (mod decls), `src/types.rs` (`PLAID_TOMBSTONE`), `src/embedding/model_manager.rs` (`ensure_colbert_model` + LATEON consts), `src/config.rs` (`plaid_nbits`).
- Modify (registry arms): `src/search/dense_backend.rs` (`DenseBackendKind::ColbertPlaid` + name/parse/sentinel), `src/model/capabilities.rs` (`BackendKind::ColbertPlaid` + `backend_for` + `dense_kind`), `src/model/manifest.rs` (`lateon-colbert` builtin spec).
- Modify (construction sites, versioned-dir adapt): `src/search/hybrid.rs::open`, `src/index/builder.rs` (build arm + meta-dim arm).

---

## Task 1: Re-vendor next-plaid + restore deps + PROVE it builds

**Files:**
- Create: `vendor/next-plaid/**` (restore tree from `26e71ba`)
- Modify: `Cargo.toml`, `crates/semantex-core/Cargo.toml`

- [ ] **Step 1: Restore the vendored crate tree**

```bash
# Restore every path under vendor/next-plaid from the pre-removal commit.
for p in $(git ls-tree -r --name-only 26e71ba -- vendor/next-plaid); do
  mkdir -p "$(dirname "$p")"
  git show "26e71ba:$p" > "$p"
done
git add -f vendor/next-plaid   # vendor/ is gitignored? verify; force-add if so
ls vendor/next-plaid/src | head
```

- [ ] **Step 2: Restore the workspace + crate deps**

Apply `git show 42c59f8 -- Cargo.toml crates/semantex-core/Cargo.toml` in reverse for the next-plaid lines only. Concretely, into root `Cargo.toml` `[workspace.dependencies]`:
```toml
next-plaid = "1.3"
next-plaid-onnx = "1.3"
```
and the patch override (own section near the bottom):
```toml
[patch.crates-io]
next-plaid = { path = "vendor/next-plaid" }
```
Into `crates/semantex-core/Cargo.toml` `[dependencies]` (after `rust-stemmers`):
```toml
next-plaid.workspace = true
next-plaid-onnx.workspace = true
```

- [ ] **Step 3: Prove unification + build (the empirical de-risk)**

```bash
cargo tree -i ort 2>&1 | head        # MUST show a single ort 2.0.0-rc.11
cargo tree -i tokenizers 2>&1 | head # expect 0.21.x (next-plaid-onnx) + 0.22 (core) — both ok
cargo tree | grep -i genai && echo "FAIL: genai leaked" || echo "OK: zero-LLM default holds"
cargo build -p next-plaid            # vendored crate builds standalone
cargo build --workspace              # whole workspace still builds with the new deps
```
Expected: `cargo tree -i ort` resolves to exactly one `ort v2.0.0-rc.11`; workspace builds. **If ort does NOT unify or the build fails → STOP, do not proceed; report.** (Lock-file evidence says it will.)

- [ ] **Step 4: Commit**

```bash
git add -f vendor/next-plaid && git add Cargo.toml crates/semantex-core/Cargo.toml Cargo.lock
git commit -m "deps: re-vendor next-plaid 1.3.1 + restore next-plaid-onnx (LateOn backend, step 1/6)"
```

---

## Task 2: Restore primitives (no behavior change yet)

**Files:**
- Modify: `crates/semantex-core/src/types.rs`, `src/embedding/model_manager.rs`, `src/config.rs`

- [ ] **Step 1: Restore `PLAID_TOMBSTONE`**

Find it in the removal: `git show 06bcf6f -- crates/semantex-core/src/types.rs | grep -A2 PLAID_TOMBSTONE` (or `git show 26e71ba:crates/semantex-core/src/types.rs | grep -n PLAID_TOMBSTONE`). Restore the `pub const PLAID_TOMBSTONE: u64 = …;` and any doc comment verbatim. (The de-PLAID rename `0952b99` introduced `DENSE_TOMBSTONE` — check whether to reuse it or keep `PLAID_TOMBSTONE`; the resurrected glue references `PLAID_TOMBSTONE`, so restore that name to minimize glue edits.)

- [ ] **Step 2: Restore `ensure_colbert_model` + LATEON consts in `model_manager.rs`**

`git show 26e71ba:crates/semantex-core/src/embedding/model_manager.rs` — restore `ensure_colbert_model(models_dir) -> Result<PathBuf>`, `is_colbert_downloaded`, and consts `LATEON_CODE_EDGE_{BASE_URL,FILES,DIR}` (`"LateOn-Code-edge"`, files `model_int8.onnx`/`tokenizer.json`/`onnx_config.json`). The shared `download_file` primitive still exists at HEAD — reuse it; only restore the colbert-specific wrappers.

- [ ] **Step 3: Restore `plaid_nbits` in `config.rs`**

`git show 26e71ba:crates/semantex-core/src/config.rs | grep -n plaid_nbits` — restore `pub plaid_nbits: usize` field + its `default` (4) + any env overlay. Do NOT restore `ColbertModelChoice`/`colbert_model` (registry-driven now). Do NOT change `dense_backend`/`embedder`/`adaptive_sizing` defaults.

- [ ] **Step 4: Build + run touched-module tests**

```bash
cargo build -p semantex-core
cargo test -p semantex-core config:: types:: model_manager:: 2>&1 | tail
```
Expected: builds; existing config/types tests still pass. (No new behavior wired yet.)

- [ ] **Step 5: Commit** — `git commit -am "feat(core): restore PLAID_TOMBSTONE + ensure_colbert_model + plaid_nbits (LateOn step 2/6)"`

---

## Task 3: Restore the backend glue verbatim (compiles, not yet selectable)

**Files:**
- Create: `crates/semantex-core/src/embedding/colbert.rs`, `src/search/plaid_search.rs`, `src/search/colbert_plaid_backend.rs`
- Modify: `src/embedding/mod.rs`, `src/search/mod.rs`

- [ ] **Step 1: Restore the three glue files verbatim**

```bash
git show 26e71ba:crates/semantex-core/src/embedding/colbert.rs            > crates/semantex-core/src/embedding/colbert.rs
git show 26e71ba:crates/semantex-core/src/search/plaid_search.rs          > crates/semantex-core/src/search/plaid_search.rs
git show 26e71ba:crates/semantex-core/src/search/colbert_plaid_backend.rs > crates/semantex-core/src/search/colbert_plaid_backend.rs
```

- [ ] **Step 2: Wire mod declarations**

In `src/embedding/mod.rs` re-add `pub mod colbert;` (match `26e71ba` ordering/visibility — `git show 26e71ba:crates/semantex-core/src/embedding/mod.rs`). In `src/search/mod.rs` re-add `pub mod plaid_search;` and `pub mod colbert_plaid_backend;` (`git show 26e71ba:crates/semantex-core/src/search/mod.rs`).

- [ ] **Step 3: Build + run the restored modules' own tests**

```bash
cargo build -p semantex-core 2>&1 | tail -30
```
Expected: compiles. The restored files reference `next_plaid::MmapIndex`/`IndexConfig`/`UpdateConfig`/`SearchParameters`/`QueryResult` (Task 1) + `next_plaid_onnx::Colbert`/`ExecutionProvider` (Task 1) + `crate::types::{PLAID_TOMBSTONE, ScoredChunkId}` (Task 2) + the unchanged `DenseBackend`/`DenseIndexBuilder` seam. Fix only compile errors caused by HEAD drift (e.g. `ScoredChunkId::new` arity, `config::env_usize` signature, `model_manager` fn names). The backend will be dead_code (no constructor yet) — acceptable warning until Task 5; if clippy-deny bites locally, add a temporary `#[allow(dead_code)]` on the structs and remove it in Task 5.

```bash
cargo test -p semantex-core colbert:: plaid_search:: colbert_plaid_backend:: 2>&1 | tail
```
Expected: the restored unit tests (colbert lazy-init/missing-dir/race; plaid_search env-knob helpers) pass.

- [ ] **Step 4: Commit** — `git commit -am "feat(search): restore colbert.rs + plaid_search.rs + colbert_plaid_backend.rs (LateOn step 3/6)"`

---

## Task 4: Restore registry/enum arms (selectable in capability routing)

**Files:**
- Modify: `src/search/dense_backend.rs`, `src/model/capabilities.rs`, `src/model/manifest.rs`

- [ ] **Step 1: `DenseBackendKind::ColbertPlaid` variant**

In `dense_backend.rs`: add `ColbertPlaid` to the enum (keep `#[default]` on `CoderankHnsw`), add `name()` arm `=> "colbert-plaid"`, `parse()` arm `"colbert-plaid" => Some(ColbertPlaid)`, and `dense_sentinel_file()` arm `ColbertPlaid => "plaid_mapping.bin"`. Update the existing test `parse_unknown_backend_is_none` (it asserts `colbert-plaid` parses to `None`) → now it must parse to `Some(ColbertPlaid)`; move that name out of the "unknown" test and add a positive parse test mirroring `parse_coderank_hnsw_backend`.

- [ ] **Step 2: `BackendKind::ColbertPlaid` + capability routing**

`git show 06bcf6f -- crates/semantex-core/src/model/capabilities.rs` shows the removed arms. Re-add `ColbertPlaid` to `enum BackendKind`; change `backend_for(caps)` so `caps.multi_vector == true` returns `Ok(BackendKind::ColbertPlaid)` (was `bail!`); add `BackendKind::ColbertPlaid` arm to `dense_kind() -> Ok(DenseBackendKind::ColbertPlaid)`. Restore/adjust the capabilities unit tests that asserted multi_vector errors.

- [ ] **Step 3: `lateon-colbert` builtin spec in `manifest.rs`**

Add to `builtin_specs()` (verbatim shape from the removed entry; `git show 06bcf6f -- crates/semantex-core/src/model/manifest.rs`):
```rust
ModelSpec {
    id: "lateon-colbert".to_string(),
    role: ModelRole::Embedder,
    source: ModelSource::Hf {
        repo: "lightonai/LateOn-Code-edge".to_string(),
        files: vec!["model_int8.onnx".to_string(), "tokenizer.json".to_string(), "onnx_config.json".to_string()],
    },
    capabilities: ModelCapabilities { multi_vector: true, ..ModelCapabilities::default() },
    role_data: RoleData::Embedder(EmbedderSpec {
        dims: 48, max_context: 512, query_prefix: String::new(), doc_prefix: String::new(),
        pooling: Pooling::LateInteraction, normalize: true, quant: QuantKind::Int8Symmetric,
    }),
},
```
(Field names/types MUST match HEAD's `ModelSpec`/`EmbedderSpec`/`ModelSource` — read them first; the above is the `26e71ba` shape and may have drifted.)

- [ ] **Step 4: Build + tests + the model-registry golden test**

```bash
cargo build -p semantex-core
cargo test -p semantex-core model:: dense_backend:: 2>&1 | tail
cargo test -p semantex-core --test model_registry_golden_test 2>&1 | tail
```
Expected: builds; `SEMANTEX_EMBEDDER=lateon-colbert` now resolves (in `registry::resolve_dense_backend`) to `DenseBackendKind::ColbertPlaid`. The golden test may need its expected snapshot updated to include `lateon-colbert` — update it to reflect the restored built-in (verify the new row is correct, don't just bless it).

- [ ] **Step 5: Commit** — `git commit -am "feat(model): register lateon-colbert → colbert-plaid backend routing (LateOn step 4/6)"`

---

## Task 5: Adapt the two construction sites to the versioned-dir lifecycle

**Files:**
- Modify: `src/search/hybrid.rs` (query-side `open`), `src/index/builder.rs` (build arm + meta-dim arm)

This is the ONLY genuinely new logic (vs `26e71ba`, which used the plain `dense_subdir`/legacy `plaid/` layout). The seam helpers (`resolve_active_dense_dir`, `active_dense_dir`, `write_active_pointer`, `dense_sentinel_file`) already exist at HEAD.

- [ ] **Step 1 (query side): add the `ColbertPlaid` arm in `hybrid.rs::open`**

At HEAD the match is `match resolved_backend { DenseBackendKind::CoderankHnsw => {…} }` (near `hybrid.rs:128`). Add:
```rust
DenseBackendKind::ColbertPlaid => {
    let plaid_dir = resolve_active_dense_dir(index_dir, DenseBackendKind::ColbertPlaid);
    let mapping_path = plaid_dir.join("plaid_mapping.bin");
    if plaid_dir.join(dense_sentinel_file(DenseBackendKind::ColbertPlaid)).exists() {
        let model_dir = crate::embedding::model_manager::ensure_colbert_model(&config.models_dir())?;
        Some(Box::new(crate::search::colbert_plaid_backend::ColbertPlaidBackend::open(
            &plaid_dir, &mapping_path, &model_dir,
        )?))
    } else { None }   // not built yet → dense channel absent, like coderank's missing-store path
}
```
Mirror exactly how the `CoderankHnsw` arm handles a missing store (it should already gate on the sentinel). Drop the `26e71ba` legacy top-level `plaid/` fallback — colbert-plaid never shipped post-versioning, so there are no legacy on-disk indexes to support.

- [ ] **Step 2 (build side): add the `ColbertPlaid` arm in `index/builder.rs`**

At HEAD the dense build match is `match backend_kind { DenseBackendKind::CoderankHnsw => {…} }` (near `builder.rs:619`); the versioned `dense_dir` is already computed (`builder.rs:551`) and the coderank arm calls `write_active_pointer` after a full build (`builder.rs:728`). Add a `ColbertPlaid` arm that:
  - constructs `ColbertPlaidIndexBuilder::new(&dense_dir, self.config.plaid_nbits).with_models_dir(self.config.models_dir())` where `dense_dir` is the versioned `active_dense_dir(index_dir, ColbertPlaid, &fingerprint)`;
  - on full rebuild calls `build_streaming_ids(chunk_ids, fetch)`, on incremental calls `insert_streaming_ids(...)` (mirror the coderank arm's full-vs-incremental branch driven by `decide_dense_build`);
  - after a COMPLETE full build calls `write_active_pointer(index_dir, ColbertPlaid, &fingerprint)` (atomic switchover — same placement as coderank).
Also add the meta-dim arm (`builder.rs:751`): `DenseBackendKind::ColbertPlaid => ("LateOn-Code-edge".to_string(), 48u32),`.

- [ ] **Step 3: Remove any temporary `#[allow(dead_code)]` from Task 3** (the backend is now constructed).

- [ ] **Step 4: Full workspace build + clippy + fmt**

```bash
cargo build --workspace
cargo clippy --all 2>&1 | tail
cargo fmt --all
cargo test -p semantex-core 2>&1 | tail
```
Expected: clean build, clippy clean, all unit tests green.

- [ ] **Step 5: Commit** — `git commit -am "feat(index): wire colbert-plaid into the versioned dense-dir build/open lifecycle (LateOn step 5/6)"`

---

## Task 6: e2e validation + full gate (the proof it actually works)

**Files:** none (validation). Uses the cached `~/.semantex/models/LateOn-Code-edge/` (confirmed present: model_int8.onnx 17MB + tokenizer.json + onnx_config.json).

- [ ] **Step 1: Build the release CLI**
```bash
cargo build -p semantex-cli --release   # → target/release/semantex
```

- [ ] **Step 2: Index a small corpus under lateon-colbert (versioned-dir + ACTIVE + auto-rebuild proof)**
```bash
TMP=$(mktemp -d); cp -r crates/semantex-core/src "$TMP/src"   # any small real code tree
export ORT_DYLIB_PATH=/opt/homebrew/lib/libonnxruntime.dylib
SEMANTEX_EMBEDDER=lateon-colbert target/release/semantex index "$TMP"
ls -R "$TMP/.semantex/dense/colbert-plaid"   # expect <fingerprint>/plaid_mapping.bin + ACTIVE pointer
cat "$TMP/.semantex/meta.json" | grep -E 'dense_backend|embedding_dim'  # colbert-plaid / 48
```
Expected: `dense/colbert-plaid/<fp>/plaid_mapping.bin` exists, `ACTIVE` points at `<fp>`, `meta.dense_backend == "colbert-plaid"`, `embedding_dim == 48`.

- [ ] **Step 3: Search and confirm non-empty dense results**
```bash
SEMANTEX_EMBEDDER=lateon-colbert target/release/semantex --json -m 5 "open the dense backend" -- 2>/dev/null
SEMANTEX_EMBEDDER=lateon-colbert target/release/semantex --dense-only --json -m 5 "hnsw index build"
```
Expected: JSON hits returned; `--dense-only` returns late-interaction-ranked results.

- [ ] **Step 4: Confirm the coderank-hnsw default is UNCHANGED**
```bash
TMP2=$(mktemp -d); cp -r crates/semantex-core/src "$TMP2/src"
target/release/semantex index "$TMP2"   # no SEMANTEX_EMBEDDER → default
grep dense_backend "$TMP2/.semantex/meta.json"   # MUST be coderank-hnsw
```

- [ ] **Step 5: Full gate**
```bash
cargo build --workspace && cargo test --workspace && cargo clippy --all && cargo fmt --all --check
cargo tree | grep -i genai && echo FAIL || echo "zero-LLM ok"
```
Then `actually verify_change`. Status must be verified/attention (no broken).

- [ ] **Step 6: Commit + land**
```bash
git commit -am "test(e2e): validate lateon-colbert index/search + coderank default intact (LateOn step 6/6)"
```
PR branch `feat/lateon-colbert-backend`; rebase-merge to main after the gate is green.

---

## Task 7 (FOLLOW-UP, separate PR): LateOn-Code 130M quality-max opt-in
Per roadmap §4: offer `LateOn-Code` (130M, dim-128) as a quality-max opt-in (not default). Needs a 2nd embedder spec (`lateon-colbert-130m`?) pointing at `lightonai/LateOn-Code` with `dims: 128`, and the `--embedding-size 128` encode path. Defer until Task 1-6 land + the Item-3 real-repo A/B picks the default arm. Not in this PR.

---

## Notes for the executor
- Restored files come WITH their tests (colbert.rs has 4 unit tests; plaid_search.rs has env-knob tests). "TDD" here = restore file → run its tests → green. Only Task 5 (construction-site adaptation) is genuinely new logic → TDD any new versioned-dir glue.
- The tree is GREEN at Task 1 (vendored crate + workspace builds) and again at Task 5 (full integration). Tasks 2-4 compile but the backend is dead_code until Task 5 — acceptable WIP; the clippy/fmt gate is enforced at Task 5/6.
- HEAD-drift fixes (Task 3 mainly): expect small signature reconciliations (`ScoredChunkId::new`, `config::env_usize`, `models_dir()`, `EmbedderSpec`/`ModelSource` field names). Read the HEAD definitions before pasting `26e71ba` shapes.
- If anything in Task 1 (ort unification / build) fails → STOP and report; the approach pivots (upstream next-plaid vs vendored, or an ort pin).
- This is Item 2 of the goal roadmap. Item 3 (chunked real-repo A/B: LateOn-edge PLAID vs coderank, both adaptive-OFF) is the promotion blocker that decides whether to flip the default — a SEPARATE effort after this lands.
