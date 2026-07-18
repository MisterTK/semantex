# Cinder — Compiled Encoder-Free Indexing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compile the doc-side lateon-colbert pipeline (embed → contextualize → assign → quantize) into precomputed artifacts + one tiny SIMD kernel so a fresh dense index builds in <5s at 159k-chunk scale, <300MB dense-increment RSS, CPU-only, at better-than-Tier-0 quality.

**Architecture:** Three offline artifacts (existing static table + frozen centroids, plus a new distilled micro-mixer and per-vocab centroid shortlists) feed an index-time pipeline of pure lookups + a ~2.8k-param depthwise-separable conv, emitting integer codes into disk segments merged once into next-plaid's exact on-disk PLAID format. Query side untouched.

**Tech Stack:** Rust workspace (semantex-core, semantex-cli, vendored next-plaid), ndarray + ndarray-npy 0.9.1, hand-rolled backprop/Adam (no new ML deps), CSN relevance harness.

**Spec:** `docs/superpowers/specs/2026-07-18-cinder-instant-indexing-design.md` (read §4–§5 before any task). Gate-3 context: `results/ember-gate3/report.md`.

## Global Constraints

- **Repo-agnostic crates/** (CLAUDE.md 1–8): no absolute paths, no repo-specific tuning; artifacts trained on generic corpora; `cargo tree -p semantex-cli | grep genai` stays empty; **no new ML dependencies** (no candle/burn/torch — backprop is hand-rolled).
- **CI pins Rust 1.91**: before ANY push run `rustup run 1.91 cargo clippy --all` (no NEW warnings in touched files; 14 pre-existing warnings in untouched files are known) and `rustup run 1.91 cargo fmt --all -- --check`. Note: the `cargo +1.91.0` toolchain name does not exist on this machine — use `rustup run 1.91 …`.
- **Never run a bare `cargo fmt --all` write-sweep.** Format only touched files: `rustfmt --edition 2024 <file>`.
- **postcard**: this plan touches NO postcard structs. If you find yourself editing one, stop (memory `postcard-non-self-describing-tradeoff`).
- **Model-gated tests carry `#[ignore = "..."]`** matching colbert.rs's existing convention; env-var tests are serialized with a file-local mutex + catch_unwind/restore pattern (see `STATIC_DOC_EMBED_ENV_LOCK` / `FROZEN_CENTROIDS_ENV_LOCK` in `colbert_plaid_backend.rs`).
- **Artifact loading**: checked size arithmetic, reject implausible headers BEFORE allocation (pattern: `static_table.rs::load` post-`c49b4cb`); writes atomic via same-dir temp + rename (pattern: `centroid_train::save_centroids_npy`).
- **Fingerprint invariance**: `SEMANTEX_CINDER` and all new artifacts stay OUT of `EmbedderFingerprint::compute` (spec §4.5); a locking test is mandatory (pattern: `aux_artifact_fields_do_not_change_the_fingerprint` in spec.rs).
- Embedding dim is **48**; window is **9** (±4); centroids **[8192, 48]**; all doc-side vectors unit-norm.
- Edge-of-document window positions **replicate the center id** — the convention shared by `static_distill.rs` and `static_token.rs::window_ids_at` since Ember A. Cinder MUST match it (train and serve).
- `next-plaid` is vendored (`vendor/next-plaid`, workspace `[patch.crates-io]` path override). To run its tests: temporarily add `"vendor/next-plaid"` to root `Cargo.toml` `workspace.members`, test, then REVERT Cargo.toml/Cargo.lock before committing (verify `git diff --stat` shows only intended files).
- Base branch: `main` @ `9aeb82f`. Work branch: `feat/cinder-instant-indexing`.
- `/docs` is gitignored — spec/plan files under `docs/` need `git add -f` (already committed; nothing to do), but `results/cinder-gate/report.md` needs a `.gitignore` exception triple (Task 8).

## File Structure (locked)

| File | Responsibility |
|---|---|
| `crates/semantex-core/src/embedding/mixer.rs` (new) | `MicroMixer`: SXCM format load/save, forward pass, window gathering. Inference only. |
| `crates/semantex-core/src/embedding/mixer_train.rs` (new) | Backprop, Adam, training loop, held-out diagnostics. Offline only. |
| `crates/semantex-core/src/embedding/shortlists.rs` (new) | `CentroidShortlists`: SXCS format, derivation from table×centroids, shortlist-union argmax, agreement diagnostic. |
| `crates/semantex-core/src/embedding/cinder.rs` (new) | `CinderEncoder`: tokenize → rows → mixer → assign → per-doc codes. Composes the three artifacts. |
| `vendor/next-plaid/src/compiled.rs` (new) | `CompiledIndexWriter`: segment spill + k-way merge → byte-compatible PLAID files. `create_index_files` (index.rs:551) is the normative reference. |
| `crates/semantex-cli/src/commands/distill_mixer.rs`, `derive_shortlists.rs` (new) | Hidden CLI subcommands (mirror `distill_centroids.rs`). |
| `crates/semantex-core/src/search/colbert_plaid_backend.rs` (modify) | `SEMANTEX_CINDER` wiring, fresh-build path switch, fallback chain. |
| `results/cinder-gate/report.md` (new) | Gate C1–C4 + floor-autopsy report. |

---

### Task 0: Branch setup

- [ ] **Step 1:**
```bash
cd /Users/tk/dev/qgrep/semantex
git status --short   # expect clean; if not, STOP and report
git checkout -b feat/cinder-instant-indexing
```

---

### Task 1: MicroMixer — format + forward pass

**Files:**
- Create: `crates/semantex-core/src/embedding/mixer.rs`
- Modify: `crates/semantex-core/src/embedding/mod.rs` (add `pub mod mixer;`)
- Test: inline `mod tests`

**Interfaces:**
- Produces (used by Tasks 2/3/4/6):
```rust
pub const MIXER_WINDOW: usize = 9;          // ±4 around center
pub const MIXER_CENTER: usize = 4;

/// f32 in memory; on disk SXCM v1 stores f32 weights (int8 quantization is a
/// measured-if-needed optimization, NOT in v1 — YAGNI until C2 profiling says so).
pub struct MicroMixer {
    pub dims: usize,                        // 48 for lateon-colbert, discovered not hardcoded
    pub dw: Vec<f32>,                       // depthwise: MIXER_WINDOW * dims, row-major [k][d]
    pub b1: Vec<f32>,                       // dims
    pub wp: Vec<f32>,                       // pointwise: dims * dims, row-major [out][in]
    pub b2: Vec<f32>,                       // dims
}

impl MicroMixer {
    pub fn zeros(dims: usize) -> Self;
    /// e = L2norm(center + Wp·GELU(Σ_k dw[k]⊙x[k] + b1) + b2); writes into `out` (len dims).
    /// `window` is MIXER_WINDOW row slices, each len dims (edge positions already replicated by caller).
    pub fn forward(&self, window: &[&[f32]; MIXER_WINDOW], center: &[f32], out: &mut [f32]);
    pub fn save(&self, path: &Path) -> anyhow::Result<()>;
    pub fn load(path: &Path) -> anyhow::Result<Self>;
}

/// Exact tanh-approx GELU used by both forward and (Task 2) backward.
pub fn gelu(x: f32) -> f32;
```
- SXCM v1 on-disk: magic `SXCM` (4B), version u32=1, dims u32, window u32 (=9), then f32 LE arrays in order dw, b1, wp, b2. Loader: checked `dims.checked_mul(window)` etc., total expected size vs actual remaining bytes BEFORE any allocation; reject version≠1, window≠MIXER_WINDOW, dims==0 or dims>4096.

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_mixer(dims: usize) -> MicroMixer {
        // Deterministic non-trivial weights.
        let mut m = MicroMixer::zeros(dims);
        for (i, v) in m.dw.iter_mut().enumerate() { *v = ((i % 7) as f32 - 3.0) * 0.05; }
        for (i, v) in m.wp.iter_mut().enumerate() { *v = ((i % 5) as f32 - 2.0) * 0.03; }
        for (i, v) in m.b1.iter_mut().enumerate() { *v = (i as f32) * 0.01; }
        for (i, v) in m.b2.iter_mut().enumerate() { *v = -(i as f32) * 0.01; }
        m
    }

    #[test]
    fn forward_matches_hand_computed_reference() {
        // dims=2 so the whole computation is checkable by hand in the test body.
        let mut m = MicroMixer::zeros(2);
        m.dw = vec![0.0; 18]; m.dw[MIXER_CENTER * 2] = 1.0; m.dw[MIXER_CENTER * 2 + 1] = 1.0; // pick center only
        m.wp = vec![1.0, 0.0, 0.0, 1.0];  // identity
        // b1=b2=0. So delta = GELU(center); e = norm(center + GELU(center)).
        let c = [0.6f32, 0.8];
        let rows: Vec<[f32; 2]> = (0..MIXER_WINDOW).map(|_| c).collect();
        let window: [&[f32]; MIXER_WINDOW] = std::array::from_fn(|k| rows[k].as_slice());
        let mut out = [0.0f32; 2];
        m.forward(&window, &c, &mut out);
        let want = [c[0] + gelu(c[0]), c[1] + gelu(c[1])];
        let n = (want[0] * want[0] + want[1] * want[1]).sqrt();
        assert!((out[0] - want[0] / n).abs() < 1e-6 && (out[1] - want[1] / n).abs() < 1e-6,
                "got {out:?}, want normalized {want:?}");
    }

    #[test]
    fn output_is_unit_norm() {
        let m = tiny_mixer(48);
        let center = vec![0.3f32; 48];
        let row = vec![0.1f32; 48];
        let window: [&[f32]; MIXER_WINDOW] = std::array::from_fn(|_| row.as_slice());
        let mut out = vec![0.0f32; 48];
        m.forward(&window, &center, &mut out);
        let n: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-5, "norm {n}");
    }

    #[test]
    fn zero_mixer_is_identity_normalize() {
        // All-zero weights => delta = Wp·GELU(b1)+b2 = 0 => e = norm(center).
        let m = MicroMixer::zeros(4);
        let center = [2.0f32, 0.0, 0.0, 0.0];
        let row = [9.0f32, 9.0, 9.0, 9.0];
        let window: [&[f32]; MIXER_WINDOW] = std::array::from_fn(|_| row.as_slice());
        let mut out = [0.0f32; 4];
        m.forward(&window, &center, &mut out);
        assert_eq!(out, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn save_load_round_trip() {
        let tmp = tempfile::TempDir::new().unwrap();
        let p = tmp.path().join("cinder_mixer.bin");
        let m = tiny_mixer(48);
        m.save(&p).unwrap();
        let l = MicroMixer::load(&p).unwrap();
        assert_eq!(l.dims, 48);
        assert_eq!(l.dw, m.dw); assert_eq!(l.b1, m.b1);
        assert_eq!(l.wp, m.wp); assert_eq!(l.b2, m.b2);
    }

    #[test]
    fn load_rejects_forged_headers() {
        let tmp = tempfile::TempDir::new().unwrap();
        let forge = |dims: u32, window: u32| {
            let mut b = Vec::new();
            b.extend_from_slice(b"SXCM");
            b.extend_from_slice(&1u32.to_le_bytes());
            b.extend_from_slice(&dims.to_le_bytes());
            b.extend_from_slice(&window.to_le_bytes());
            b
        };
        for (name, bytes) in [
            ("overflow", forge(u32::MAX, u32::MAX)),
            ("huge", forge(1_000_000, 9)),
            ("truncated", forge(48, 9)),
            ("bad-window", forge(48, 7)),
        ] {
            let p = tmp.path().join(name);
            std::fs::write(&p, bytes).unwrap();
            assert!(MicroMixer::load(&p).is_err(), "{name} must be rejected");
        }
        let p = tmp.path().join("bad-magic");
        std::fs::write(&p, b"NOPE").unwrap();
        assert!(MicroMixer::load(&p).is_err());
    }
}
```

- [ ] **Step 2:** Run `cargo test -p semantex-core mixer` — expect compile FAIL (module absent).

- [ ] **Step 3: Implement**

```rust
//! Cinder micro-mixer: a distilled contextualization operator (spec §4.1 #3).
//!
//! e_i = L2norm(t_i + Wp · GELU(DW(t_{i−4..i+4}) + b1) + b2), where DW is a
//! per-dimension 9-tap depthwise filter. ~2.8k params at d=48. This is the ONLY
//! floating-point compute in the Cinder index path; everything else is lookups.

use anyhow::{Context, Result};
use std::path::Path;

pub const MIXER_WINDOW: usize = 9;
pub const MIXER_CENTER: usize = 4;
const MAGIC: &[u8; 4] = b"SXCM";
const VERSION: u32 = 1;
const MAX_DIMS: usize = 4096; // implausibility cap for forged headers

pub struct MicroMixer {
    pub dims: usize,
    pub dw: Vec<f32>,
    pub b1: Vec<f32>,
    pub wp: Vec<f32>,
    pub b2: Vec<f32>,
}

/// tanh-approximation GELU (matches the derivative Task 2 implements).
pub fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x)).tanh())
}

impl MicroMixer {
    pub fn zeros(dims: usize) -> Self {
        Self {
            dims,
            dw: vec![0.0; MIXER_WINDOW * dims],
            b1: vec![0.0; dims],
            wp: vec![0.0; dims * dims],
            b2: vec![0.0; dims],
        }
    }

    pub fn forward(&self, window: &[&[f32]; MIXER_WINDOW], center: &[f32], out: &mut [f32]) {
        let d = self.dims;
        debug_assert_eq!(center.len(), d);
        debug_assert_eq!(out.len(), d);
        // h = Σ_k dw[k] ⊙ x[k] + b1
        let mut h = self.b1.clone();
        for (k, row) in window.iter().enumerate() {
            let w = &self.dw[k * d..(k + 1) * d];
            for i in 0..d {
                h[i] += w[i] * row[i];
            }
        }
        // g = GELU(h)
        for v in &mut h {
            *v = gelu(*v);
        }
        // out = center + Wp·g + b2, then L2 normalize
        for (o, (row, (&c, &b))) in out
            .iter_mut()
            .zip(self.wp.chunks_exact(d).zip(center.iter().zip(self.b2.iter())))
        {
            let dot: f32 = row.iter().zip(h.iter()).map(|(&w, &g)| w * g).sum();
            *o = c + dot + b;
        }
        let norm: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in out {
                *v /= norm;
            }
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let dir = path.parent().unwrap_or_else(|| Path::new("."));
        std::fs::create_dir_all(dir)?;
        let tmp = dir.join(format!(
            ".{}.{}.tmp",
            path.file_name().and_then(|n| n.to_str()).unwrap_or("mixer"),
            std::process::id()
        ));
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&(self.dims as u32).to_le_bytes());
        buf.extend_from_slice(&(MIXER_WINDOW as u32).to_le_bytes());
        for arr in [&self.dw, &self.b1, &self.wp, &self.b2] {
            for v in arr.iter() {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        std::fs::write(&tmp, &buf)?;
        std::fs::rename(&tmp, path).with_context(|| format!("saving mixer to {}", path.display()))?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let buf = std::fs::read(path)?;
        anyhow::ensure!(buf.len() >= 16, "mixer file too short for header");
        anyhow::ensure!(&buf[0..4] == MAGIC, "invalid mixer magic");
        let rd = |o: usize| u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
        anyhow::ensure!(rd(4) == VERSION, "unsupported mixer version {}", rd(4));
        let dims = rd(8) as usize;
        let window = rd(12) as usize;
        anyhow::ensure!(window == MIXER_WINDOW, "mixer window {window} != {MIXER_WINDOW}");
        anyhow::ensure!(dims > 0 && dims <= MAX_DIMS, "implausible mixer dims {dims}");
        // Checked total size BEFORE allocation.
        let n_f32 = dims
            .checked_mul(MIXER_WINDOW)
            .and_then(|n| n.checked_add(dims))                 // dw + b1
            .and_then(|n| dims.checked_mul(dims).map(|m| n + m)) // + wp
            .and_then(|n| n.checked_add(dims))                 // + b2
            .ok_or_else(|| anyhow::anyhow!("mixer size overflows"))?;
        let expected = n_f32.checked_mul(4).ok_or_else(|| anyhow::anyhow!("mixer size overflows"))?;
        anyhow::ensure!(buf.len() - 16 == expected, "mixer file wrong size: {} data bytes, want {expected}", buf.len() - 16);
        let mut cursor = 16;
        let mut read_vec = |n: usize| -> Vec<f32> {
            let out = buf[cursor..cursor + n * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            cursor += n * 4;
            out
        };
        Ok(Self {
            dims,
            dw: read_vec(MIXER_WINDOW * dims),
            b1: read_vec(dims),
            wp: read_vec(dims * dims),
            b2: read_vec(dims),
        })
    }
}
```

- [ ] **Step 4:** `cargo test -p semantex-core mixer` — all PASS.
- [ ] **Step 5:**
```bash
rustfmt --edition 2024 crates/semantex-core/src/embedding/mixer.rs crates/semantex-core/src/embedding/mod.rs
git add crates/semantex-core/src/embedding/mixer.rs crates/semantex-core/src/embedding/mod.rs
git commit -m "feat(cinder): micro-mixer format (SXCM) + forward pass"
```

---

### Task 2: Mixer training — hand-rolled backprop + Adam

**Files:**
- Create: `crates/semantex-core/src/embedding/mixer_train.rs`
- Modify: `crates/semantex-core/src/embedding/mod.rs` (add `pub mod mixer_train;`)
- Test: inline `mod tests`

**Interfaces:**
- Consumes: Task 1's `MicroMixer`, `gelu`, `MIXER_WINDOW`, `MIXER_CENTER`; existing `static_distill::DocTokenEncoder`, `StaticTokenTable`.
- Produces (used by Task 4):
```rust
pub struct MixerTrainOptions {
    pub sample_capacity: usize,  // default 2_000_000 pairs
    pub epochs: usize,           // default 3
    pub batch: usize,            // encoder batch, default 32
    pub lr: f32,                 // default 1e-3, halved each epoch
    pub holdout_frac: f32,       // default 0.05
}
impl Default for MixerTrainOptions { ... }

pub struct MixerTrainReport {
    pub pairs_trained: usize,
    pub holdout_cosine_mixer: f32,     // mean cos(student, teacher) on held-out
    pub holdout_cosine_linear: f32,    // same for the Ember 5-tap linear baseline
}

/// Streams `corpus` through the TEACHER `encoder` (contextual), reservoir-samples
/// (window_ids [u32;9], teacher_row f16[dims]) pairs, then trains `MicroMixer` by
/// gathering STUDENT inputs from `table` rows. Positions whose center-token table
/// row is all-zero are skipped (unseen tokens; miss rate ≤0.74% per Gate-1 data).
/// Deterministic: fixed seeds for reservoir, init, and shuffling.
pub fn train_mixer(
    encoder: &dyn DocTokenEncoder,
    table: &StaticTokenTable,
    corpus: impl Iterator<Item = String>,
    opts: &MixerTrainOptions,
) -> anyhow::Result<(MicroMixer, MixerTrainReport)>;
```
- Loss: `L = 1 − cos(u, y)` where `u = center + Wp·GELU(h) + b2` (pre-normalization; y is unit-norm teacher, so cos(u,y) = (u·y)/‖u‖). Gradients:
  - `dL/du = −( y/‖u‖ − (u·y)·u/‖u‖³ )`
  - `db2 = dL/du`; `dWp[o][i] = dL/du[o] · g[i]`; `dg = Wpᵀ·dL/du`; `dh = dg ⊙ gelu′(h)`; `db1 = dh`; `ddw[k][d] = dh[d] · x[k][d]`. Table rows are frozen (no input gradient needed).
  - `gelu′(x)` for the tanh approximation: with `s = SQRT_2_OVER_PI·(x + 0.044715x³)`, `t = tanh(s)`: `gelu′ = 0.5(1+t) + 0.5x(1−t²)·SQRT_2_OVER_PI·(1 + 3·0.044715x²)`.
- Init: `dw` center tap = 0 (residual path already carries center), all weights ~ uniform(−0.05, 0.05) from SplitMix64 (copy the private struct as in `centroid_train.rs`), biases 0. Adam: β1=0.9, β2=0.999, ε=1e-8.

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::mixer::{MicroMixer, MIXER_WINDOW};

    /// THE load-bearing test: analytic gradients match central finite differences.
    #[test]
    fn gradient_check_matches_finite_differences() {
        let dims = 3;
        let mut m = MicroMixer::zeros(dims);
        // Non-trivial deterministic weights.
        for (i, v) in m.dw.iter_mut().enumerate() { *v = ((i % 5) as f32 - 2.0) * 0.11; }
        for (i, v) in m.wp.iter_mut().enumerate() { *v = ((i % 3) as f32 - 1.0) * 0.17; }
        for (i, v) in m.b1.iter_mut().enumerate() { *v = 0.03 * i as f32; }
        for (i, v) in m.b2.iter_mut().enumerate() { *v = -0.02 * i as f32; }

        let window_rows: Vec<Vec<f32>> = (0..MIXER_WINDOW)
            .map(|k| (0..dims).map(|d| ((k + d) as f32 * 0.31).sin()).collect())
            .collect();
        let center: Vec<f32> = window_rows[super::super::mixer::MIXER_CENTER].clone();
        let mut y: Vec<f32> = (0..dims).map(|d| (d as f32 + 1.0).cos()).collect();
        let n: f32 = y.iter().map(|v| v * v).sum::<f32>().sqrt();
        for v in &mut y { *v /= n; }

        let grads = compute_gradients(&m, &window_rows, &center, &y);

        let eps = 1e-3f32;
        // Spot-check 12 parameters across all four tensors via finite differences.
        let mut checks: Vec<(&str, usize)> = vec![
            ("dw", 0), ("dw", 7), ("dw", MIXER_WINDOW * dims - 1),
            ("b1", 0), ("b1", dims - 1),
            ("wp", 0), ("wp", 4), ("wp", dims * dims - 1),
            ("b2", 0), ("b2", 1), ("b2", dims - 1),
            ("dw", 13),
        ];
        for (tensor, idx) in checks.drain(..) {
            let get = |m: &MicroMixer| loss(m, &window_rows, &center, &y);
            let mut mp = clone_mixer(&m); bump(&mut mp, tensor, idx, eps);
            let mut mm = clone_mixer(&m); bump(&mut mm, tensor, idx, -eps);
            let fd = (get(&mp) - get(&mm)) / (2.0 * eps);
            let an = grad_at(&grads, tensor, idx);
            assert!(
                (fd - an).abs() < 2e-3 * (1.0 + fd.abs().max(an.abs())),
                "{tensor}[{idx}]: finite-diff {fd} vs analytic {an}"
            );
        }
    }

    /// Training on a synthetic task the mixer CAN express must beat the linear
    /// baseline: teacher = normalize(center + 0.3·GELU(prev)) — nonlinear,
    /// context-dependent, inexpressible by a 5-tap linear mix.
    #[test]
    fn training_beats_linear_baseline_on_nonlinear_synthetic_task() {
        let dims = 8;
        let vocab = 16;
        let (encoder, table) = nonlinear_fake(vocab, dims); // defined below
        let corpus: Vec<String> = (0..300).map(|i| synth_doc(i, 24)).collect();
        let opts = MixerTrainOptions { sample_capacity: 50_000, epochs: 8, batch: 8, lr: 3e-3, holdout_frac: 0.1 };
        let (_m, report) = train_mixer(&encoder, &table, corpus.into_iter(), &opts).unwrap();
        assert!(
            report.holdout_cosine_mixer > report.holdout_cosine_linear + 0.01,
            "mixer {} must beat linear {} by >0.01 on a task built to require nonlinearity",
            report.holdout_cosine_mixer, report.holdout_cosine_linear
        );
        assert!(report.holdout_cosine_mixer > 0.9, "should nearly solve the synthetic task");
    }

    #[test]
    fn training_is_deterministic() {
        let dims = 4; let vocab = 8;
        let (encoder, table) = nonlinear_fake(vocab, dims);
        let corpus: Vec<String> = (0..50).map(|i| synth_doc(i, 12)).collect();
        let opts = MixerTrainOptions { sample_capacity: 5_000, epochs: 2, batch: 8, lr: 1e-3, holdout_frac: 0.1 };
        let (a, _) = train_mixer(&encoder, &table, corpus.clone().into_iter(), &opts).unwrap();
        let (b, _) = train_mixer(&encoder, &table, corpus.into_iter(), &opts).unwrap();
        assert_eq!(a.dw, b.dw); assert_eq!(a.wp, b.wp);
    }

    #[test]
    fn empty_corpus_errors() {
        let (encoder, table) = nonlinear_fake(4, 4);
        let err = train_mixer(&encoder, &table, Vec::<String>::new().into_iter(), &MixerTrainOptions::default()).unwrap_err();
        assert!(err.to_string().to_lowercase().contains("empty"));
    }
}
```

Test helpers to implement in the same module: `nonlinear_fake(vocab, dims)` returns a `FakeEncoder` (implements `DocTokenEncoder`: token id = byte − b'a'; teacher row = L2norm(onehot(id) + 0.3·GELU-elementwise(onehot(prev_id))) as `TokenEmbeddings`) plus a `StaticTokenTable` whose row for each id = onehot(id) (set via `set_row`) and mix_weights = `[0,0,1,0,0]`; `synth_doc(seed, len)` generates a deterministic lowercase string via SplitMix64. `compute_gradients`, `loss`, `clone_mixer`, `bump`, `grad_at` are small `#[cfg(test)]`-visible wrappers over the module's internal backprop (expose the internal `fn backward(...) -> Gradients` as `pub(crate)`).

- [ ] **Step 2:** `cargo test -p semantex-core mixer_train` — compile FAIL.
- [ ] **Step 3: Implement.** Module layout: `struct Gradients { dw, b1, wp, b2: Vec<f32> }`; `fn forward_raw(m, window_rows, center) -> (h_pre_gelu, g, u)` (un-normalized u, reused by loss/backward); `pub(crate) fn backward(m, window_rows, center, y) -> (f32 /*loss*/, Gradients)` implementing exactly the formulas in Interfaces; `struct Adam { m, v: Gradients-shaped, t: u64 }` with the standard bias-corrected update; `train_mixer` = reservoir-fill pass (windows via the same edge-replication as `static_distill::ingest_document`, storing `([u32; 9], Vec<half::f16>)` pairs) → deterministic shuffle → split holdout → epoch loop (minibatch accumulate gradients, Adam step, lr halves each epoch) → compute both holdout cosines (mixer via `MicroMixer::forward`; linear baseline via re-implementing the 5-tap mix from `table.mix_weights` over the same rows — mirror `static_token.rs::mix_token`'s formula) → return. Skip positions with all-zero center rows. Error on zero usable pairs (message contains "empty").
- [ ] **Step 4:** `cargo test -p semantex-core mixer_train` — all PASS (gradient check is the gate; if it fails, fix backward, do not loosen tolerances).
- [ ] **Step 5:**
```bash
rustfmt --edition 2024 crates/semantex-core/src/embedding/mixer_train.rs crates/semantex-core/src/embedding/mod.rs
git add crates/semantex-core/src/embedding/mixer_train.rs crates/semantex-core/src/embedding/mod.rs
git commit -m "feat(cinder): hand-rolled mixer training with gradient-checked backprop"
```

---

### Task 3: Centroid shortlists — derivation, format, union-argmax

**Files:**
- Create: `crates/semantex-core/src/embedding/shortlists.rs`
- Modify: `crates/semantex-core/src/embedding/mod.rs`; `crates/semantex-core/src/embedding/model_manager.rs` (add `pub const CINDER_MIXER_FILE: &str = "cinder_mixer.bin"; pub const CINDER_SHORTLISTS_FILE: &str = "cinder_shortlists.bin";` + `cinder_mixer_path(&Path)`/`cinder_shortlists_path(&Path)` helpers mirroring `frozen_centroids_path`)
- Test: inline `mod tests`

**Interfaces:**
- Consumes: `StaticTokenTable` (rows), `ndarray::Array2<f32>` centroids (via Task 3's own loader arg — callers use `centroid_train::load_centroids_npy`).
- Produces (used by Tasks 4/6):
```rust
pub struct CentroidShortlists {
    pub m: usize,                 // entries per token
    pub vocab_size: usize,
    data: Vec<u16>,               // (vocab_size + 1) * m; last row = global fallback
}
impl CentroidShortlists {
    /// Exact top-m centroids per nonzero table row; zero rows get the global
    /// fallback row (top-m of the mean of all nonzero rows). m capped at
    /// centroids.nrows() and at u16 id range (centroids.nrows() must fit u16).
    pub fn derive(table: &StaticTokenTable, centroids: &ndarray::ArrayView2<f32>, m: usize) -> anyhow::Result<Self>;
    pub fn for_token(&self, token_id: u32) -> &[u16];   // zero-row/OOV ids -> fallback row
    pub fn save(&self, path: &Path) -> anyhow::Result<()>;   // SXCS v1, atomic
    pub fn load(path: &Path) -> anyhow::Result<Self>;        // checked header
}

/// argmax_{c ∈ union of window shortlists} (e · centroid_c); ties broken by lowest id.
/// `scratch` is a reusable Vec<u16> to avoid per-token allocation.
pub fn shortlist_argmax(
    e: &[f32],
    window_ids: &[u32],
    shortlists: &CentroidShortlists,
    centroids: &ndarray::ArrayView2<f32>,
    scratch: &mut Vec<u16>,
) -> usize;

/// Diagnostic for gate C4: fraction of `samples` whose shortlist_argmax equals
/// exhaustive argmax over all centroids.
pub fn shortlist_agreement(
    samples: &[(Vec<f32>, Vec<u32>)],   // (embedding, window_ids)
    shortlists: &CentroidShortlists,
    centroids: &ndarray::ArrayView2<f32>,
) -> f64;
```
- SXCS v1 on-disk: magic `SXCS`, version u32=1, vocab_size u32, m u32, then `(vocab_size+1)·m` u16 LE. Loader: checked `(vocab_size+1).checked_mul(m).checked_mul(2)`, exact-size match, vocab_size ≤ 10_000_000, m ∈ 1..=1024.

- [ ] **Step 1: Write the failing tests** — cover: (a) `derive` on a hand-built 4-token/6-centroid setup puts each token's true nearest centroid first in its shortlist; (b) zero-row token returns the fallback row and the fallback row equals top-m of the mean of nonzero rows (hand-computed); (c) `shortlist_argmax` equals exhaustive argmax when the shortlist contains it, on 20 deterministic pseudo-random embeddings; (d) `shortlist_agreement` returns 1.0 on those samples and <1.0 when shortlists are truncated to m=1 with adversarial centroids (construct: token A's row nearest centroid 0, but query embedding nearest centroid 1 which is only in token B's shortlist; window = [A] only); (e) save/load round-trip; (f) forged-header rejection (overflow vocab×m, truncated, m=0, m>1024). Write them concretely following Task 1's test style (full Rust, deterministic constructions with SplitMix64 where randomness is needed).
- [ ] **Step 2:** `cargo test -p semantex-core shortlists` — compile FAIL.
- [ ] **Step 3: Implement.** `derive`: for each nonzero row compute all-centroid dot products (this is offline; exhaustive is fine), partial-select top-m by value (then sort selected by descending dot, ties by ascending id — deterministic); reject `centroids.nrows() > u16::MAX as usize + 1`. `shortlist_argmax`: collect unique candidate ids into `scratch` (clear, extend from each window token's `for_token`, sort_unstable, dedup), then scan dots. Note window_ids may contain the same id repeatedly (edge replication) — dedup handles it.
- [ ] **Step 4:** `cargo test -p semantex-core shortlists` — PASS.
- [ ] **Step 5:**
```bash
rustfmt --edition 2024 crates/semantex-core/src/embedding/shortlists.rs crates/semantex-core/src/embedding/model_manager.rs crates/semantex-core/src/embedding/mod.rs
git add crates/semantex-core/src/embedding/shortlists.rs crates/semantex-core/src/embedding/model_manager.rs crates/semantex-core/src/embedding/mod.rs
git commit -m "feat(cinder): per-vocab centroid shortlists (SXCS) + union argmax"
```

---

### Task 4: CLI — `distill-mixer` and `derive-shortlists`

**Files:**
- Create: `crates/semantex-cli/src/commands/distill_mixer.rs`, `crates/semantex-cli/src/commands/derive_shortlists.rs`
- Modify: `crates/semantex-cli/src/commands/mod.rs`, `crates/semantex-cli/src/main.rs`

**Interfaces:**
- Consumes: Tasks 1–3 (`train_mixer`, `MixerTrainOptions`, `MicroMixer::save`, `CentroidShortlists::derive/save`), existing `distill_corpus::corpus_chunk_texts`, `ColbertEmbedder::for_indexing`, `model_manager::ensure_colbert_model`, `StaticTokenTable::load`, `centroid_train::load_centroids_npy`, `model_manager::{static_token_table_path, frozen_centroids_path, cinder_mixer_path, cinder_shortlists_path}`.
- Produces:
  - `semantex distill-mixer --corpus <dir>... --out <file> [--sample 2000000] [--epochs 3] [--verify]` (hidden). Loads the static table + teacher encoder from the resolved model dir, streams corpus via `corpus_chunk_texts`, calls `train_mixer`, prints `holdout cosine: mixer=X linear=Y pairs=N`, saves, `--verify` reloads and prints dims.
  - `semantex derive-shortlists --out <file> [--m 32] [--verify]` (hidden). No corpus needed: loads table + frozen centroids from the model dir, derives, saves, prints `derived shortlists m=<m> vocab=<v>`.
  - Register both EXACTLY parallel to `distill-centroids` in main.rs (`#[command(hide = true)]`, loaded `SemantexConfig` threaded through — the config-parity precedent `94fa51a`).

- [ ] **Step 1:** Implement both command modules (mirror `distill_centroids.rs`'s structure: fail-fast artifact resolution with `.context(...)`, then core call, then save + optional verify) and register. `distill-mixer` errors clearly if `static_token_table.bin` is missing ("run distill-static-table first"); `derive-shortlists` errors if either the table or `frozen_centroids.npy` is missing ("run distill-centroids first").
- [ ] **Step 2:** 
```bash
cargo build -p semantex-cli
target/debug/semantex distill-mixer --help && target/debug/semantex derive-shortlists --help
cargo test -p semantex-cli
cargo clippy -p semantex-cli -p semantex-core -- -D warnings
```
Expected: help shows the flags above; tests pass; clippy clean on touched crates (default toolchain OK here; 1.91 gate runs at push time).
- [ ] **Step 3:**
```bash
rustfmt --edition 2024 crates/semantex-cli/src/commands/distill_mixer.rs crates/semantex-cli/src/commands/derive_shortlists.rs crates/semantex-cli/src/commands/mod.rs crates/semantex-cli/src/main.rs
git add crates/semantex-cli/src/commands/ crates/semantex-cli/src/main.rs
git commit -m "feat(cinder): distill-mixer + derive-shortlists hidden CLI subcommands"
```

---

### Task 5: `CompiledIndexWriter` — single-pass segmented PLAID construction (vendored next-plaid)

**Files:**
- Create: `vendor/next-plaid/src/compiled.rs`
- Modify: `vendor/next-plaid/src/lib.rs` (add `pub mod compiled;` + re-export `CompiledIndexWriter`)
- Test: inline `mod tests` (the differential byte-equivalence test IS gate C4's second criterion)

**Interfaces:**
- Consumes: `create_index_files` (index.rs:551) as the NORMATIVE REFERENCE — read it fully first; it writes `centroids.npy`, `bucket_cutoffs.npy`, `bucket_weights.npy`, `avg_residual.npy`, `cluster_threshold.npy`, `plan.json`, per-chunk codes/residuals + `doclens.{i}.json` + chunk metadata, `ivf.npy`, `ivf_lengths.npy`, `metadata.json` (see index.rs:400–911). Reuse its exact helpers: `utils::{atomic_write_file, quantile, quantiles}`, `ResidualCodec::{new, compress_into_codes_cpu, quantize_residuals}`.
- Produces:
```rust
/// Builds a PLAID index in ONE pass over per-document embeddings, with frozen
/// centroids, spilling packed codes to disk segments instead of holding all
/// embeddings in memory, and NEVER rewriting the IVF incrementally.
///
/// Contract (gate C4): given the same documents, centroids, and config, the
/// resulting index directory is BYTE-IDENTICAL (excluding embeddings.npy /
/// buffer files, which this writer never creates) to
/// `create_index_files(embeddings, centroids, path, config)`.
pub struct CompiledIndexWriter { /* private */ }

impl CompiledIndexWriter {
    /// `sample_docs`: the residual-statistics sample (cutoffs/weights/avg_residual/
    /// cluster_threshold are computed from these EXACTLY as create_index_files
    /// computes them from its full input — extract that logic into a shared
    /// private fn used by both, so the math cannot drift).
    pub fn new(
        index_path: &str,
        centroids: Array2<f32>,
        config: &IndexConfig,
        sample_docs: &[Array2<f32>],
    ) -> Result<Self>;

    /// Append one document's token embeddings (unit-norm, [n_tokens, dim]).
    /// Assigns via codec.compress_into_codes_cpu, quantizes, buffers; spills a
    /// segment to `<index_path>/.cinder_seg_<n>` every `segment_tokens` tokens
    /// (default 524_288).
    pub fn add_document(&mut self, embeddings: &Array2<f32>) -> Result<()>;

    /// K-way merge segments by doc order into the chunked on-disk layout,
    /// build IVF once, write all files, delete segments. Returns Metadata.
    pub fn finalize(self) -> Result<Metadata>;
}
```
- Design notes (binding): documents are chunked into on-disk "chunks" of `config.batch_size` docs exactly as `create_index_files` does — the writer tracks doc counts and flushes chunk files in order, so byte-identity holds without buffering everything. The IVF is built from an in-memory `Vec<(u32 centroid, i64 doc_id)>` accumulated during add (16B × n_tokens — 152MB at 9.5M tokens; acceptable, it is the ONLY O(corpus) memory and is far under R3's 300MB) — sort at finalize (`sort_unstable`), dedup per centroid, write. If chunk files can be written incrementally (they can — they only need that chunk's docs), segments on disk hold packed residual codes awaiting their chunk flush only when a chunk spans an in-memory boundary; if after reading `create_index_files` the implementer finds chunk-incremental writing makes segments unnecessary, dropping the segment files in favor of per-chunk flushing is an APPROVED simplification (record it) — byte-identity + bounded memory are the requirements, not the mechanism.

- [ ] **Step 1: Write the failing differential test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{create_index_files, IndexConfig};
    use ndarray::Array2;

    fn synth_docs(n_docs: usize, dim: usize) -> Vec<Array2<f32>> {
        (0..n_docs)
            .map(|d| {
                let toks = 3 + (d % 5);
                let mut a = Array2::from_shape_fn((toks, dim), |(i, j)| {
                    ((d * 31 + i * 7 + j) as f32 * 0.37).sin()
                });
                for mut row in a.rows_mut() {
                    let n = row.dot(&row).sqrt();
                    row.mapv_inplace(|v| v / n);
                }
                a
            })
            .collect()
    }

    fn synth_centroids(k: usize, dim: usize) -> Array2<f32> {
        let mut c = Array2::from_shape_fn((k, dim), |(i, j)| ((i * 13 + j * 3) as f32 * 0.29).cos());
        for mut row in c.rows_mut() {
            let n = row.dot(&row).sqrt();
            row.mapv_inplace(|v| v / n);
        }
        c
    }

    #[test]
    fn output_is_byte_identical_to_create_index_files() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dim = 8;
        let docs = synth_docs(57, dim);            // >1 on-disk chunk with batch_size=16
        let centroids = synth_centroids(32, dim);
        let config = IndexConfig { nbits: 4, batch_size: 16, force_cpu: true, ..Default::default() };

        let ref_dir = tmp.path().join("reference");
        create_index_files(&docs, centroids.clone(), ref_dir.to_str().unwrap(), &config).unwrap();

        let out_dir = tmp.path().join("compiled");
        let mut w = CompiledIndexWriter::new(out_dir.to_str().unwrap(), centroids, &config, &docs).unwrap();
        for d in &docs { w.add_document(d).unwrap(); }
        w.finalize().unwrap();

        // Every file create_index_files wrote must exist byte-identical
        // (embeddings.npy excluded: the compiled writer never persists raw
        // embeddings; delete it from the reference before comparing).
        let _ = std::fs::remove_file(ref_dir.join("embeddings.npy"));
        let mut ref_files: Vec<_> = std::fs::read_dir(&ref_dir).unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
        ref_files.sort();
        let mut out_files: Vec<_> = std::fs::read_dir(&out_dir).unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
        out_files.sort();
        assert_eq!(ref_files, out_files, "file inventories differ");
        for f in &ref_files {
            let a = std::fs::read(ref_dir.join(f)).unwrap();
            let b = std::fs::read(out_dir.join(f)).unwrap();
            assert_eq!(a, b, "file {f} differs");
        }
    }

    #[test]
    fn compiled_index_loads_and_searches() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dim = 8;
        let docs = synth_docs(30, dim);
        let config = IndexConfig { nbits: 4, batch_size: 16, force_cpu: true, ..Default::default() };
        let out = tmp.path().join("idx");
        let mut w = CompiledIndexWriter::new(out.to_str().unwrap(), synth_centroids(16, dim), &config, &docs).unwrap();
        for d in &docs { w.add_document(d).unwrap(); }
        let meta = w.finalize().unwrap();
        assert_eq!(meta.num_documents, 30);
        let idx = crate::MmapIndex::load(out.to_str().unwrap()).unwrap();
        assert_eq!(idx.metadata.num_documents, 30);
    }

    #[test]
    fn memory_stays_bounded_via_segment_spill() {
        // Not an RSS assertion (flaky) — asserts the MECHANISM: with a tiny
        // segment_tokens override, add_document must produce segment files
        // (or per-chunk flushes) on disk before finalize is called.
        let tmp = tempfile::TempDir::new().unwrap();
        let dim = 8;
        let docs = synth_docs(64, dim);
        let config = IndexConfig { nbits: 4, batch_size: 8, force_cpu: true, ..Default::default() };
        let out = tmp.path().join("idx");
        let mut w = CompiledIndexWriter::new(out.to_str().unwrap(), synth_centroids(16, dim), &config, &docs).unwrap();
        w.set_segment_tokens_for_test(64);
        for d in &docs { w.add_document(d).unwrap(); }
        let files_before_finalize = std::fs::read_dir(&out).unwrap().count();
        assert!(files_before_finalize > 0, "expected on-disk spill before finalize");
        w.finalize().unwrap();
        // No leftover temp/segment files after finalize.
        for e in std::fs::read_dir(&out).unwrap() {
            let name = e.unwrap().file_name().into_string().unwrap();
            assert!(!name.starts_with(".cinder_seg"), "leftover segment {name}");
        }
    }
}
```

(Add `pub(crate) fn set_segment_tokens_for_test(&mut self, n: usize)` — or make the threshold a `new` parameter with a default; implementer's choice, note it.)

- [ ] **Step 2:** Run (with the temporary workspace-member dance from Global Constraints): `cargo test -p next-plaid compiled` — compile FAIL.
- [ ] **Step 3: Implement.** Read `create_index_files` end to end FIRST. Extract its residual-statistics computation (the code that produces bucket_cutoffs/bucket_weights/avg_residual/cluster_threshold from embeddings — around index.rs:400–433 and whatever precedes it) into a shared private `fn residual_stats(embeddings: &[Array2<f32>], centroids: &ArrayView2<f32>, nbits: usize) -> (…)` called by BOTH `create_index_files` and `CompiledIndexWriter::new` — shared code is what makes byte-identity durable, not luck. Then implement add/finalize per the design notes. All file writes through `utils::atomic_write_file` exactly as the reference does.
- [ ] **Step 4:** `cargo test -p next-plaid` — ALL pass (the crate's whole suite is the regression net: `create_index_files`'s refactor must not change its behavior). Then revert Cargo.toml/Cargo.lock; `git diff --stat` must show only `vendor/next-plaid/src/{compiled.rs,index.rs,lib.rs}` (+ possibly utils.rs if visibility changes were needed).
- [ ] **Step 5:**
```bash
rustfmt --edition 2024 vendor/next-plaid/src/compiled.rs vendor/next-plaid/src/index.rs vendor/next-plaid/src/lib.rs
git add vendor/next-plaid/src/
git commit -m "feat(cinder): CompiledIndexWriter — single-pass byte-compatible PLAID construction"
```

---

### Task 6: `CinderEncoder` + builder wiring + fallback chain

**Files:**
- Create: `crates/semantex-core/src/embedding/cinder.rs`
- Modify: `crates/semantex-core/src/embedding/mod.rs`; `crates/semantex-core/src/search/colbert_plaid_backend.rs`; `crates/semantex-core/src/model/registry.rs` (extend the fingerprint-exclusion doc comment to name `SEMANTEX_CINDER`)
- Test: inline + model-gated integration test

**Interfaces:**
- Consumes: Tasks 1/3/5 outputs; existing `StaticTokenTable`, `static_token.rs`'s tokenization plumbing (`load_doc_id_alignment` is `pub(crate)` in colbert.rs since Ember A; `window_ids_at` in static_token.rs — promote it to `pub(crate)` rather than duplicating), `centroid_train::load_centroids_npy`, `model_manager::{cinder_mixer_path, cinder_shortlists_path, frozen_centroids_path, static_token_table_path}`, `crate::config::env_bool`.
- Produces:
```rust
/// Composes table + mixer + shortlists + centroids into per-document
/// UNIT-NORM embeddings ready for CompiledIndexWriter. (v1 hands f32
/// embeddings to the writer — the writer assigns/quantizes so byte-identity
/// with the reference path is trivially preserved. The "integer codes end to
/// end" micro-optimization is deferred until C2 profiling demands it.)
pub struct CinderEncoder { /* table, mixer, shortlists, alignment */ }
impl CinderEncoder {
    /// Errors if ANY artifact is missing/corrupt (caller decides fallback).
    pub fn new(model_dir: &Path) -> anyhow::Result<Self>;
    pub fn encode_documents(&self, texts: &[String]) -> anyhow::Result<Vec<TokenEmbeddings>>;
}
```
  Wait — design decision recorded: `shortlist_argmax` lives in the WRITER path only if the writer accepts precomputed assignments; v1 keeps the writer's own `compress_into_codes_cpu` (exhaustive over 8192) for byte-identity. That would reintroduce the assignment bottleneck — so `CompiledIndexWriter` gains an optional `with_assigner(Box<dyn Fn(&Array2<f32>) -> Array1<usize>>)` hook (Task 6 modifies vendor `compiled.rs` minimally to add it), and the differential test runs BOTH with the default exhaustive assigner (byte-identity gate) AND with a shortlist assigner asserting ≥99% same-code agreement on the synthetic data (mechanism gate). Cinder passes a closure wrapping `shortlist_argmax`.
- Builder wiring in `colbert_plaid_backend.rs`:
  - `fn cinder_for_build(model_dir: &Path) -> Option<CinderEncoder>`: `env_bool("SEMANTEX_CINDER")` gate → try `CinderEncoder::new` → `Some` + `tracing::info!("using cinder compiled indexing")`; on error `tracing::warn!(...)` naming the failed artifact → `None` (build proceeds on the existing tier chain: static-embed flag, frozen-centroids flag, contextual — untouched).
  - In `build_streaming_ids` ONLY (fresh build): when cinder is Some, replace the Phase-A/Phase-B `update_or_create`/`update_append` loop with: stream batches → `cinder.encode_documents` → `writer.add_document` per doc → `finalize()` → write mapping (same `write_mapping_atomic`, mapping = doc order fed). Frozen centroids loaded via `load_centroids_npy(frozen_centroids_path(model_dir))` — Cinder REQUIRES them (no per-repo k-means in this path; missing → fallback chain above). Residual-stats sample = the first `INITIAL_BUILD_CHUNKS.min(total)` chunks' embeddings (bounded; document this constant reuse). `insert_streaming_ids` untouched.
- Env-test serialization: add `CINDER_ENV_LOCK` mutex following the existing two.

- [ ] **Step 1: Write failing unit tests** (in `colbert_plaid_backend.rs` tests + `cinder.rs` tests): (a) flag off → `cinder_for_build` None even with artifacts present; (b) flag on + missing mixer → None (and the error path is exercised — assert via `CinderEncoder::new` error message naming `cinder_mixer.bin`); (c) `CinderEncoder::new` with all four artifacts hand-built in a temp dir (tiny table via `StaticTokenTable::save`, zero mixer via `MicroMixer::save`, shortlists via `derive`+`save`, centroids via `save_centroids_npy`) + the test-fixture tokenizer used by `static_token.rs::end_to_end_with_real_tokenizer_and_hand_built_table` (reuse its `test_tokenizer_dir` helper pattern) → `encode_documents` returns unit-norm rows; with the ZERO mixer, rows must exactly equal `StaticTokenEmbedder`'s output for `mix_weights=[0,0,1,0,0]` given the same table (the identity-mixer ≡ pure-center-lookup equivalence — a strong cross-implementation consistency check). (d) Fingerprint invariance test in registry/spec tests asserting `SEMANTEX_CINDER` never enters `EmbedderFingerprint::compute` (compute fingerprint with env set and unset under the lock — equal).
- [ ] **Step 2:** compile-FAIL run.
- [ ] **Step 3: Implement** cinder.rs + the vendor `with_assigner` hook (+ its two added tests in compiled.rs per the Interfaces note) + builder wiring + registry doc-comment extension.
- [ ] **Step 4: Model-gated end-to-end test** in `colbert_plaid_backend.rs` (fully written, `#[ignore = "requires the downloaded LateOn-Code-edge model and real Cinder artifacts"]`): overlay temp model dir (symlink real model files — the Task-6/Plan-B overlay pattern); train a REAL tiny mixer via `train_mixer` over ~50 synthetic code strings with the real ColbertEmbedder teacher + real static table; derive shortlists from the real frozen centroids; build a small index with `SEMANTEX_CINDER=1`; assert (i) build OK + confirmation log, (ii) `MmapIndex::load` works and `num_documents` matches, (iii) a search for a distinctive token via the normal query path returns the chunk containing it top-1.
- [ ] **Step 5:** `cargo test --workspace` + model-gated run (`cargo test -p semantex-core --release -- --ignored cinder`) + vendor suite via the member dance. All green.
- [ ] **Step 6:**
```bash
rustfmt --edition 2024 crates/semantex-core/src/embedding/cinder.rs crates/semantex-core/src/search/colbert_plaid_backend.rs crates/semantex-core/src/model/registry.rs crates/semantex-core/src/embedding/mod.rs vendor/next-plaid/src/compiled.rs
git add crates/semantex-core/src/ vendor/next-plaid/src/compiled.rs
git commit -m "feat(cinder): CinderEncoder + SEMANTEX_CINDER fresh-build wiring with fallback chain"
```

---

### Task 7: Memory-floor autopsy (ungated workstream)

**Files:**
- Create: findings section prepared for Task 8's report (write to `/tmp/cinder-floor-autopsy.md`, merged in Task 8)
- Possibly modify: `crates/semantex-core` ORT-provisioning call path IF (and only if) the "skip ORT on encoder-free builds" win is confirmed obvious (<20-line change) — otherwise document only.

- [ ] **Step 1: Stage-boundary RSS instrumentation run.** Build release, then on `/Users/tk/dev/gin` (preserve/restore `.semantex` via the `mv .semantex .semantex.pre-cinder` protocol): run `SEMANTEX_STATIC_DOC_EMBED=1 SEMANTEX_FROZEN_CENTROIDS=1 RUST_LOG=semantex_core=info /usr/bin/time -l target/release/semantex index .` and record the `rss_mb` values already logged at PLAID batch boundaries plus `vmmap --summary <pid>` snapshots (run `semantex index` with a `& sleep 0.5; vmmap` harness, three snapshots). Attribute the ~1.06GB across: ORT dylib + session, tantivy writer heaps, sqlite page cache, allocator arenas (MALLOC regions in vmmap), static artifacts.
- [ ] **Step 2: ORT-skip experiment.** Find where ORT/the runtime manager is provisioned during an encoder-free build (start from `runtime_manager.rs` and `ColbertEmbedder::for_indexing`'s lazy session — memory `ort-load-dynamic-runtime-provisioning`). If ORT is being loaded despite `DocEncoderKind::Static`/Cinder never running a session, gate it off for encoder-free builds and measure the delta on gin. If ORT is NOT the floor, record what is; change nothing.
- [ ] **Step 3:** Write `/tmp/cinder-floor-autopsy.md` with the attribution table + deltas; commit any code change separately (`fix(cinder): skip ORT provisioning on encoder-free builds` — only if taken).

---

### Task 8: Train real artifacts, run gates C1–C4, write the report

**Files:**
- Create: `results/cinder-gate/report.md`
- Modify: `.gitignore` (exception triple mirroring `ember-gate3`: `!/results/cinder-gate/`, `/results/cinder-gate/*`, `!/results/cinder-gate/report.md`)
- Raw data (gitignored): `benchmarks/relevance/results/cinder-gate-*`

**Interfaces:** consumes everything; Gate-1 full-hybrid baselines (python 0.89696 / javascript 0.55657 / go 0.75934 nDCG@10) and Gate-1 tier0-hybrid (0.9120 / 0.5092 / 0.7324); Gate-3 report conventions.

- [ ] **Step 1: Train artifacts** (release binary; Gate-3 corpus recipe; record SHAs; `tee` logs to `benchmarks/relevance/results/cinder-gate-train/`):
```bash
target/release/semantex distill-mixer \
  --corpus /Users/tk/dev/CopilotKit/packages --corpus /Users/tk/dev/platform \
  --corpus /Users/tk/dev/pub --corpus /Users/tk/dev/gin \
  --corpus /Users/tk/dev/adk-python --corpus /Users/tk/dev/qgrep/semantex \
  --out ~/.semantex/models/LateOn-Code-edge/cinder_mixer.bin --verify
target/release/semantex derive-shortlists \
  --out ~/.semantex/models/LateOn-Code-edge/cinder_shortlists.bin --m 32 --verify
```
Record the printed holdout cosines (mixer vs linear) — the first C1 leading indicator. Run each under `/usr/bin/time -l`; run ONE training process at a time (Gate-3 lesson: no liveness-test duplicates).
- [ ] **Step 2: C4 shortlist agreement** on real data: add a `--agreement <corpus-dir>` option to `derive-shortlists` (small addition, commit with Task 8) that samples 100k mixed embeddings from a corpus dir via `CinderEncoder` internals and prints `shortlist_agreement`. Gate: ≥99%; if missed, retrain shortlists with `--m 64` and re-measure (document).
- [ ] **Step 3: C2/C3 speed+memory** on CopilotKit (159k chunks) and platform, `.semantex` preserve/restore protocol, three arms each under `/usr/bin/time -l` with `RUST_LOG=semantex_core=info`: (a) sparse-only control (`SEMANTEX_DENSE=0` if such a switch exists — otherwise measure by diffing against arm (b)'s dense-stage log timestamps AND run the tier0+frozen arm as the reference; check `semantex index --help`/config for the dense-disable switch first and record which method was used), (b) Cinder (`SEMANTEX_CINDER=1`), (c) tier0+frozen (Gate-3 config, for the comparison table). Gates: dense increment <5s CopilotKit, <1s platform; peak-RSS increment <300MB. Confirm the cinder info line + zero fallback warns in (b).
- [ ] **Step 4: C1 quality** — CSN hybrid, 4 arms for the ablation matrix (run-ids `cinder-gate-*`): (1) Cinder full (`SEMANTEX_CINDER=1`), (2) mixer+exact (env `SEMANTEX_CINDER_EXACT_ASSIGN=1` — add this diagnostic env read in `cinder_for_build`'s writer construction, choosing the default exhaustive assigner; ~5-line addition, commit with Task 8), (3) tier0+frozen (= linear baseline, Gate-3 recording reusable if comparability holds — verify no query-path diffs since `fd136f7`, else rerun), (4) full-hybrid baseline (Gate-1 recording, reuse justified same way). Gate per language on arm (1): js ≥0.55657×0.9575=0.5329, go ≥0.75934×0.982=0.7457, py ≥0.89696×1.00=0.8970 nDCG@10. If missed: attribute via arms (2)/(3) (assignment vs mixer), apply the ONE contingency (hashed bigram tables — only if the mixer is the shortfall AND the shortfall is ≤2pts; otherwise report FAIL honestly with attribution).
- [ ] **Step 5: Fallback checks** — mixer file moved aside + `SEMANTEX_CINDER=1` on gin → warn + tier-chain build OK; all artifacts present + flag off → byte-identical-to-today behavior (no cinder log lines). Restore everything; verify all repos' `.semantex` restored.
- [ ] **Step 6: Report** `results/cinder-gate/report.md` per Gate-3 conventions: TL;DR verdicts per gate (C1–C4), training costs + holdout cosines, ablation matrix table, floor-autopsy section (from Task 7), corpus SHAs, judgment calls, honest FAIL handling if any, recommendation (ship default-off / iterate / promote). Commit with the `.gitignore` triple:
```bash
git add .gitignore results/cinder-gate/report.md
git commit -m "docs(cinder): gate evaluation report — compiled encoder-free indexing"
```
- [ ] **Step 7: Final gates**
```bash
cargo test --workspace
rustup run 1.91 cargo clippy --all      # no NEW warnings in touched files
rustup run 1.91 cargo fmt --all -- --check
```

---

## Self-Review (done at plan-writing time)

- **Spec coverage:** §4.1 artifacts → Tasks 1/3 (+ existing); §4.2 training → Tasks 2/4/8; §4.3 pipeline → Tasks 5/6; §4.4 scope boundaries → Task 6 (fresh-build only), Task 7 (autopsy); §4.5 fallback/fingerprint → Task 6; §5 gates C1–C4 → Task 8 (+C4 differential in Task 5, agreement in Tasks 3/8); §6 risks: gradient-check (Task 2), byte-compat differential (Task 5), shortlist gate before quality (Task 8 step 2 precedes step 4), tokenizer-throughput checkpoint (Task 8 step 3 timestamps expose it), codec reuse (Task 5 shared `residual_stats`).
- **Type consistency:** `MicroMixer::forward(&[&[f32]; MIXER_WINDOW], &[f32], &mut [f32])` used in Tasks 1/2/6; `CentroidShortlists::derive(table, centroids_view, m)` in 3/4/8; `CompiledIndexWriter::{new, add_document, finalize}` in 5/6; `train_mixer(encoder, table, corpus, opts) -> (MicroMixer, MixerTrainReport)` in 2/4/8. `with_assigner` is introduced in Task 6's Interfaces and modifies Task 5's file — flagged explicitly there so neither implementer is surprised.
- **Known deviations from spec prose:** spec §4.1 says int8 mixer weights; v1 ships f32 (2.8k params ≈ 11KB — int8 saves nothing that matters) with int8 named as a non-goal unless C2 misses. Spec §4.3's "integer codes into segments" is realized as f32-embedding handoff + writer-side assign/quantize with an injectable assigner — same complexity bounds, strictly better byte-identity story. Both recorded here as approved plan-level decisions.
