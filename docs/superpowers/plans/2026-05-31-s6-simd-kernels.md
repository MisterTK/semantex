# S6 — SIMD Distance Kernels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a portable, dependency-free SIMD distance-kernel module `crates/semantex-core/src/search/simd.rs` exposing `dot_f32`, `cosine_f32`, `l2_f32` (fp32) plus an int8 dot/cosine path, runtime-dispatched to AVX2 (x86_64) / NEON (aarch64) with a scalar fallback and a `len & !N` + scalar tail, so the new single-vector dense backend (S2) and its brute-force / rescore paths have a fast, audited kernel.

**Architecture:** One public module `simd.rs` with safe public entry points (`dot_f32`, `cosine_f32`, `l2_f32`, `dot_i8`, `cosine_i8`) that own the runtime dispatch. Each entry point: below `SIMD_MIN_LEN` (batch-size gate, oxirs idea) calls the always-correct scalar reference in a private `scalar` submodule; at/above the gate it calls an `unsafe`, `#[target_feature]`-gated, `#[cfg(target_arch=…)]` arch kernel (`avx2` submodule on x86_64, `neon` submodule on aarch64) guarded by runtime CPU detection (`is_x86_feature_detected!` / `is_aarch64_feature_detected!`). Kernels are **reimplemented from** the oxirs-core `simd/{scalar,x86_simd,arm_simd}.rs` reference pattern (Apache-2.0/MIT) — not copied — to keep the tree dependency-free (`std::arch` only).

**Tech Stack:** Rust 2024 edition, `std::arch::x86_64` / `std::arch::aarch64` intrinsics only, no new crate dependencies. Tests via `cargo test -p semantex-core`. Benchmark via `criterion 0.8` (already a `semantex-core` dev-dependency) wired as a new `[[bench]]` target. `cargo clippy` for the unsafe audit.

---

## Reconciled facts (verified against the current tree — do not re-derive)

These are quoted from the real tree / authoritative inputs at plan-authoring time. Every type, path, and command below depends on them.

- **Module home & wiring.** `crates/semantex-core/src/search/mod.rs:1-19` declares the `search::*` submodules with plain `pub mod NAME;` lines, alphabetically ordered (`adaptive` … `triple_fusion`). The spec (§4 S6) names the module `crates/semantex-core/src/search/simd.rs`. We add `pub mod simd;` to that list (between `regex_semantic` and `ripgrep_fallback` to keep alpha order).

- **`ScoredChunkId` is the 5-field type** at `crates/semantex-core/src/types.rs:219-241` (`chunk_id, score, score_dense, score_sparse, score_exact`). S6 does **not** touch it — the kernels operate on raw `&[f32]` / `&[i8]` slices. S2 maps kernel outputs into `ScoredChunkId { chunk_id, score }`. (Listed so executors don't invent a 2-field type.)

- **Workspace lints (`/Users/tk/dev/qgrep/semantex/Cargo.toml:44-51`):** `unsafe_op_in_unsafe_fn = "deny"` and `unused_must_use = "deny"` under `[workspace.lints.rust]`; clippy `pedantic = warn`. **Consequence:** inside an `unsafe fn`, every intrinsic call (itself `unsafe`) MUST be wrapped in an explicit inner `unsafe { … }` block — bare intrinsic calls in an `unsafe fn` body will fail the build. The oxirs reference does NOT do this (it predates `unsafe_op_in_unsafe_fn`); our reimplementation MUST. This is the single most common way executors will break the build — every arch kernel below already shows the inner `unsafe { … }` blocks.

- **`edition = "2024"`** (`Cargo.toml:7`). `is_x86_feature_detected!` / `is_aarch64_feature_detected!` are in the prelude (no import). `std::arch::x86_64::*` / `std::arch::aarch64::*` must be imported under the matching `#[cfg(target_arch)]`.

- **criterion is already available** to `semantex-core`: `crates/semantex-core/Cargo.toml:136-137` has `[dev-dependencies]` → `criterion = { version = "0.8", features = ["html_reports"] }`. **But there are NO `[[bench]]` targets registered in any Cargo.toml in the repo** (verified: only `semantex-core/Cargo.toml:137` mentions criterion; no `[[bench]]` anywhere). The five files under the workspace-root `benches/` (`embedding_bench.rs` etc.) are **orphaned** — the workspace root is a virtual manifest (no `[package]`), so its `benches/` is not auto-discovered and `cargo bench` builds nothing there. **Therefore S6 must register its own `[[bench]]` target** in `semantex-core/Cargo.toml` pointing at a file under `crates/semantex-core/benches/`, which Cargo auto-associates with the `semantex-core` package. Criterion 0.8 benches use `criterion_group!` + `criterion_main!` and the target needs `harness = false`.

- **CPU-only identity / env-thread knobs** are an embedding/session concern (`embedding/colbert.rs:182-195` pins the CPU execution provider; threads via `SEMANTEX_ORT_THREADS`). **S6 is pure arithmetic on slices — it spawns no threads, allocates nothing on the hot path, and reads no env at call time.** The only env it reads is the optional `SEMANTEX_SIMD_MIN_LEN` override for the batch-size gate, read once via a `OnceLock` (mirrors the `config::env_usize` style at `config.rs:201`, but self-contained so the module has zero deps beyond `std`).

- **Float reordering is expected.** FMA (`_mm256_fmadd_ps` / `vfmaq_f32`) and SIMD-lane partial sums reorder the additions vs the strictly-sequential scalar reference, so results differ in the last ULPs. The spec (§4 S6 acceptance) mandates **parity within `1e-6`**. Tests use an **absolute-or-relative** tolerance `1e-6` (absolute for small magnitudes, relative for large) so a dim-768 dot product of unit-ish vectors (magnitudes up to ~few hundred) still passes — a pure absolute `1e-6` would be too strict at large magnitudes. Helper `assert_close(a, b)` defined once in Task 1's tests and reused.

- **No `#![allow(unsafe_code)]` needed.** The workspace does not `forbid(unsafe_code)`; it only denies `unsafe_op_in_unsafe_fn`. `unsafe fn` + inner `unsafe {}` blocks compile. (The existing `next-plaid` vendor already uses `unsafe`.)

- **Repo-agnostic / pure-std (CLAUDE.md Hard Rules):** no hardcoded paths, no external deps, generalizes to any codebase. SIMD math is inherently domain-neutral. The one env var (`SEMANTEX_SIMD_MIN_LEN`) is a generic performance knob, not repo metadata.

---

## File Structure

Files created or modified, one responsibility each:

- **Create `crates/semantex-core/src/search/simd.rs`** — the entire kernel module. Public safe API (`dot_f32`, `cosine_f32`, `l2_f32`, `dot_i8`, `cosine_i8`) + the `SIMD_MIN_LEN` gate (`simd_min_len()` reading the optional env once) + runtime dispatch. Private `scalar` submodule (always-correct reference: `dot_f32`, `cosine_f32`, `l2_f32`, `dot_i8`, `cosine_i8`). Private `avx2` submodule (`#[cfg(target_arch = "x86_64")]`, `#[target_feature(enable = "avx2")]` `unsafe fn`s + `horizontal_sum_avx2` helper). Private `neon` submodule (`#[cfg(target_arch = "aarch64")]`, `#[target_feature(enable = "neon")]` `unsafe fn`s). `#[cfg(test)]` unit + parity tests live at the bottom of this file.

- **Modify `crates/semantex-core/src/search/mod.rs:1-19`** — add `pub mod simd;` (alpha order, after `regex_semantic`).

- **Create `crates/semantex-core/benches/simd_bench.rs`** — criterion benchmark comparing scalar vs dispatched `dot_f32` / `cosine_f32` at representative dims (the spec calls out **768**); proves speedup. Self-contained (generates random vectors with a tiny LCG, no deps beyond criterion + the crate).

- **Modify `crates/semantex-core/Cargo.toml:136-137`** — register the `[[bench]]` target (`name = "simd_bench"`, `harness = false`) so `cargo bench -p semantex-core --bench simd_bench` actually builds and runs.

---

## Public API the consumer (S2) will call

S2 (`hnsw_index.rs`, brute-force + fp32 rescore) and S7 (MMR / semantic-cache cosine) call **only** these safe free functions from `semantex_core::search::simd`. All take equal-length slices; on length mismatch they panic with a clear message (debug-assert + runtime check) — callers must pass equal-length vectors (HNSW vectors are fixed-dim, so this always holds).

```rust
/// Dot product ⟨a, b⟩. a.len() == b.len() required.
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32;

/// Cosine SIMILARITY in [-1, 1]: ⟨a,b⟩ / (‖a‖·‖b‖). Returns 0.0 if either norm is 0.
/// (NB: similarity, not distance — distance = 1.0 - cosine_f32(a,b). Chosen because
///  the dense channel ranks by similarity; documented to avoid the oxirs "distance" confusion.)
pub fn cosine_f32(a: &[f32], b: &[f32]) -> f32;

/// Squared? No — Euclidean (L2) DISTANCE: sqrt(Σ (aᵢ-bᵢ)²). a.len() == b.len() required.
pub fn l2_f32(a: &[f32], b: &[f32]) -> f32;

/// Dot product of two int8 vectors, accumulated in i32 then returned as f32 (exact
/// for dim ≤ ~131072 since |i8·i8| ≤ 16384 and i32 holds ±2.1e9). For scoring
/// scalar-quantized vectors before fp32 rescore (S2). a.len() == b.len() required.
pub fn dot_i8(a: &[i8], b: &[i8]) -> f32;

/// Cosine similarity of two int8 vectors (i32 accumulators → f32). Returns 0.0 if
/// either norm is 0. a.len() == b.len() required.
pub fn cosine_i8(a: &[i8], b: &[i8]) -> f32;
```

**S2 wiring note (for the S2 team, not implemented here):** the int8 path scores `cosine_i8` / `dot_i8` over scale-quantized candidate vectors for the approximate prefilter, then re-ranks the top `rescore_k` with `cosine_f32` / `dot_f32` over the dequantized (or stored fp32) vectors. S2 owns the scale/zero-point quant recipe; S6 only provides the i8 arithmetic assuming a **symmetric** quantization (zero-point 0) — i.e. the i8 values already encode `round(x / scale)`. Cosine over symmetric-quantized i8 equals cosine over the reals up to quant error, so `cosine_i8` is meaningful directly. If S2 chooses asymmetric (non-zero zero-point) quant, it must dequantize before calling these — record that as the contract.

---

## Phasing note for executors

Strict TDD, scalar-first. Task 1 lands the **scalar reference + its parity-anchor tests** (the scalar fns are the ground truth all later parity tests compare against). Tasks 2–3 add the AVX2 fp32 kernels behind the dispatcher with parity tests (`scalar == dispatched` within `1e-6`). Tasks 4–5 add NEON fp32. Task 6 adds the int8 paths (scalar + both arches, since the i8 kernel is small). Task 7 wires the criterion benchmark + `[[bench]]` registration and runs it. Task 8 is the clippy unsafe-audit pass. Commit after every task. The dispatcher in Task 1 already routes to arch kernels via `cfg!`-gated calls that are **stubbed to scalar** until the arch submodule exists — so the suite stays green from Task 1 onward, and each arch task swaps the stub for the real kernel.

A note on testing arch kernels: on the dev machine (Apple Silicon, `aarch64`) the AVX2 path is `#[cfg]`-compiled-out and cannot run — its parity test is `#[cfg(target_arch = "x86_64")]`-gated and only executes in x86_64 CI. Symmetrically the NEON test is `aarch64`-gated. The **dispatch test** (calling the public fn) runs on every arch and exercises whichever kernel is live. CI must run on both arches to validate both kernels; the plan notes the gate so executors don't think a kernel is "untested" when it's merely not their host arch.

---

### Task 1: Scalar reference + module skeleton + dispatcher

**Files:**
- Create: `crates/semantex-core/src/search/simd.rs`
- Modify: `crates/semantex-core/src/search/mod.rs:1-19`

- [ ] **Step 1: Register the module**

Edit `crates/semantex-core/src/search/mod.rs` — add the line in alpha order after `pub mod regex_semantic;`:

```rust
pub mod regex_semantic;
pub mod ripgrep_fallback;
pub mod simd;
pub mod sparse_search;
```

(Final ordering: `regex_semantic`, `ripgrep_fallback`, `simd`, `sparse_search`. Insert `pub mod simd;` between `ripgrep_fallback` and `sparse_search`.)

- [ ] **Step 2: Write the module with scalar reference + gate + dispatcher (arch kernels stubbed to scalar) and the failing tests**

Create `crates/semantex-core/src/search/simd.rs`:

```rust
//! Portable SIMD distance kernels (dot / cosine / L2) for the single-vector dense
//! backend's hot path (ANN distance, brute-force fallback, fp32 rescore).
//!
//! Public safe functions own the runtime dispatch:
//!   * below [`SIMD_MIN_LEN`] (a batch-size gate; SIMD setup + horizontal reduction
//!     dominate on tiny inputs) we use the always-correct [`scalar`] reference;
//!   * at/above the gate we dispatch to an arch kernel via runtime CPU detection
//!     (`is_x86_feature_detected!` on x86_64, `is_aarch64_feature_detected!` on
//!     aarch64), falling back to scalar when the feature is absent.
//!
//! Kernels are **reimplemented from** the oxirs-core `simd/{scalar,x86_simd,arm_simd}.rs`
//! reference pattern (Apache-2.0/MIT); no code is copied. Zero external deps — `std::arch`
//! only. Each `unsafe fn` wraps every intrinsic in an inner `unsafe {}` block to satisfy
//! the workspace `unsafe_op_in_unsafe_fn = "deny"` lint.
//!
//! All public fns require `a.len() == b.len()`; they panic otherwise (HNSW vectors are
//! fixed-dim, so this always holds in practice).

use std::sync::OnceLock;

/// Default batch-size gate: below this many elements, skip SIMD and use scalar.
/// 32 matches the oxirs default and covers ColBERT-class dims (≥768) firmly in the
/// SIMD regime while keeping tiny ad-hoc vectors on the cheaper scalar path.
const DEFAULT_SIMD_MIN_LEN: usize = 32;

/// Resolve the SIMD batch-size gate, honoring an optional `SEMANTEX_SIMD_MIN_LEN`
/// override (read once). Generic perf knob — not repo metadata.
fn simd_min_len() -> usize {
    static CACHE: OnceLock<usize> = OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("SEMANTEX_SIMD_MIN_LEN")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(DEFAULT_SIMD_MIN_LEN)
    })
}

#[inline]
fn assert_same_len(a_len: usize, b_len: usize) {
    assert!(
        a_len == b_len,
        "simd kernel requires equal-length slices (got {a_len} and {b_len})"
    );
}

// ----------------------------------------------------------------------------
// Public safe API — runtime dispatch lives here.
// ----------------------------------------------------------------------------

/// Dot product ⟨a, b⟩. Requires `a.len() == b.len()`.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_same_len(a.len(), b.len());
    if a.len() < simd_min_len() {
        return scalar::dot_f32(a, b);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 confirmed present at runtime; slices are equal length.
            return unsafe { avx2::dot_f32(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("neon") {
            // SAFETY: neon confirmed present at runtime; slices are equal length.
            return unsafe { neon::dot_f32(a, b) };
        }
    }
    scalar::dot_f32(a, b)
}

/// Cosine **similarity** in [-1, 1]: ⟨a,b⟩ / (‖a‖·‖b‖). Returns `0.0` if either
/// norm is zero. Requires `a.len() == b.len()`.
#[inline]
pub fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_same_len(a.len(), b.len());
    if a.len() < simd_min_len() {
        return scalar::cosine_f32(a, b);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 confirmed present at runtime; slices are equal length.
            return unsafe { avx2::cosine_f32(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("neon") {
            // SAFETY: neon confirmed present at runtime; slices are equal length.
            return unsafe { neon::cosine_f32(a, b) };
        }
    }
    scalar::cosine_f32(a, b)
}

/// Euclidean (L2) **distance**: `sqrt(Σ (aᵢ-bᵢ)²)`. Requires `a.len() == b.len()`.
#[inline]
pub fn l2_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_same_len(a.len(), b.len());
    if a.len() < simd_min_len() {
        return scalar::l2_f32(a, b);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 confirmed present at runtime; slices are equal length.
            return unsafe { avx2::l2_f32(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("neon") {
            // SAFETY: neon confirmed present at runtime; slices are equal length.
            return unsafe { neon::l2_f32(a, b) };
        }
    }
    scalar::l2_f32(a, b)
}

/// Int8 dot product (i32 accumulator → f32). Requires `a.len() == b.len()`.
#[inline]
pub fn dot_i8(a: &[i8], b: &[i8]) -> f32 {
    assert_same_len(a.len(), b.len());
    if a.len() < simd_min_len() {
        return scalar::dot_i8(a, b);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 confirmed present at runtime; slices are equal length.
            return unsafe { avx2::dot_i8(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("neon") {
            // SAFETY: neon confirmed present at runtime; slices are equal length.
            return unsafe { neon::dot_i8(a, b) };
        }
    }
    scalar::dot_i8(a, b)
}

/// Int8 cosine similarity (i32 accumulators → f32). Returns `0.0` if either norm is
/// zero. Requires `a.len() == b.len()`.
#[inline]
pub fn cosine_i8(a: &[i8], b: &[i8]) -> f32 {
    assert_same_len(a.len(), b.len());
    if a.len() < simd_min_len() {
        return scalar::cosine_i8(a, b);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 confirmed present at runtime; slices are equal length.
            return unsafe { avx2::cosine_i8(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("neon") {
            // SAFETY: neon confirmed present at runtime; slices are equal length.
            return unsafe { neon::cosine_i8(a, b) };
        }
    }
    scalar::cosine_i8(a, b)
}

// ----------------------------------------------------------------------------
// Scalar reference — always correct; the parity ground truth.
// ----------------------------------------------------------------------------

mod scalar {
    #[inline]
    pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[inline]
    pub fn l2_f32(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum::<f32>()
            .sqrt()
    }

    #[inline]
    pub fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f32;
        let mut na = 0.0f32;
        let mut nb = 0.0f32;
        for (&x, &y) in a.iter().zip(b) {
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        let denom = na.sqrt() * nb.sqrt();
        if denom == 0.0 { 0.0 } else { dot / denom }
    }

    #[inline]
    pub fn dot_i8(a: &[i8], b: &[i8]) -> f32 {
        let mut acc: i32 = 0;
        for (&x, &y) in a.iter().zip(b) {
            acc += i32::from(x) * i32::from(y);
        }
        acc as f32
    }

    #[inline]
    pub fn cosine_i8(a: &[i8], b: &[i8]) -> f32 {
        let mut dot: i32 = 0;
        let mut na: i32 = 0;
        let mut nb: i32 = 0;
        for (&x, &y) in a.iter().zip(b) {
            let (xi, yi) = (i32::from(x), i32::from(y));
            dot += xi * yi;
            na += xi * xi;
            nb += yi * yi;
        }
        let denom = (na as f32).sqrt() * (nb as f32).sqrt();
        if denom == 0.0 { 0.0 } else { dot as f32 / denom }
    }
}

// ----------------------------------------------------------------------------
// Arch kernels — STUBBED to scalar in Task 1; real intrinsics land in Tasks 2-6.
// The stubs keep the dispatcher compiling and the suite green from Task 1.
// ----------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod avx2 {
    /// # Safety
    /// Caller must ensure AVX2 is available and `a.len() == b.len()`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        super::scalar::dot_f32(a, b)
    }
    /// # Safety
    /// Caller must ensure AVX2 is available and `a.len() == b.len()`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
        super::scalar::cosine_f32(a, b)
    }
    /// # Safety
    /// Caller must ensure AVX2 is available and `a.len() == b.len()`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn l2_f32(a: &[f32], b: &[f32]) -> f32 {
        super::scalar::l2_f32(a, b)
    }
    /// # Safety
    /// Caller must ensure AVX2 is available and `a.len() == b.len()`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_i8(a: &[i8], b: &[i8]) -> f32 {
        super::scalar::dot_i8(a, b)
    }
    /// # Safety
    /// Caller must ensure AVX2 is available and `a.len() == b.len()`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn cosine_i8(a: &[i8], b: &[i8]) -> f32 {
        super::scalar::cosine_i8(a, b)
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    /// # Safety
    /// Caller must ensure NEON is available and `a.len() == b.len()`.
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        super::scalar::dot_f32(a, b)
    }
    /// # Safety
    /// Caller must ensure NEON is available and `a.len() == b.len()`.
    #[target_feature(enable = "neon")]
    pub unsafe fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
        super::scalar::cosine_f32(a, b)
    }
    /// # Safety
    /// Caller must ensure NEON is available and `a.len() == b.len()`.
    #[target_feature(enable = "neon")]
    pub unsafe fn l2_f32(a: &[f32], b: &[f32]) -> f32 {
        super::scalar::l2_f32(a, b)
    }
    /// # Safety
    /// Caller must ensure NEON is available and `a.len() == b.len()`.
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_i8(a: &[i8], b: &[i8]) -> f32 {
        super::scalar::dot_i8(a, b)
    }
    /// # Safety
    /// Caller must ensure NEON is available and `a.len() == b.len()`.
    #[target_feature(enable = "neon")]
    pub unsafe fn cosine_i8(a: &[i8], b: &[i8]) -> f32 {
        super::scalar::cosine_i8(a, b)
    }
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute-or-relative tolerance: FMA / lane partial-sums reorder additions,
    /// so the SIMD result differs from the strictly-sequential scalar in the last
    /// ULPs. `1e-6` absolute for small magnitudes; relative for large (a dim-768
    /// dot of unit-ish vectors can reach magnitudes in the hundreds).
    fn assert_close(a: f32, b: f32) {
        let diff = (a - b).abs();
        let tol = 1e-6_f32 * a.abs().max(b.abs()).max(1.0);
        assert!(
            diff <= tol,
            "values differ beyond 1e-6 tolerance: a={a}, b={b}, diff={diff}, tol={tol}"
        );
    }

    /// Tiny deterministic LCG → reproducible test vectors (no rand dep).
    fn make_vec(len: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                // map high bits to [-1, 1)
                ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
            })
            .collect()
    }

    #[test]
    fn scalar_dot_matches_hand_computation() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        // 1*4 + 2*5 + 3*6 = 32
        assert_close(scalar::dot_f32(&a, &b), 32.0);
    }

    #[test]
    fn scalar_l2_matches_hand_computation() {
        let a = [0.0f32, 0.0];
        let b = [3.0f32, 4.0];
        // sqrt(9 + 16) = 5
        assert_close(scalar::l2_f32(&a, &b), 5.0);
    }

    #[test]
    fn scalar_cosine_orthogonal_is_zero_and_parallel_is_one() {
        assert_close(scalar::cosine_f32(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
        assert_close(scalar::cosine_f32(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]), 1.0);
    }

    #[test]
    fn cosine_zero_norm_returns_zero() {
        assert_close(cosine_f32(&[0.0; 768], &make_vec(768, 1)), 0.0);
        assert_close(scalar::cosine_i8(&[0i8; 768], &[1i8; 768]), 0.0);
    }

    #[test]
    fn scalar_i8_dot_matches_hand_computation() {
        let a = [1i8, -2, 3];
        let b = [4i8, 5, -6];
        // 1*4 + (-2)*5 + 3*(-6) = 4 - 10 - 18 = -24
        assert_close(scalar::dot_i8(&a, &b), -24.0);
    }

    /// Dispatch parity: the PUBLIC fn (whatever kernel is live on this arch) must
    /// match the scalar reference within tolerance, across sizes that exercise the
    /// gate (below `DEFAULT_SIMD_MIN_LEN`), an exact multiple of the SIMD width, and
    /// a non-multiple (forces the scalar-tail path). Runs on every architecture.
    #[test]
    fn dispatch_matches_scalar_across_sizes() {
        for &len in &[1usize, 3, 7, 8, 16, 31, 32, 33, 100, 768, 769] {
            let a = make_vec(len, 0xABCD ^ len as u64);
            let b = make_vec(len, 0x1234 ^ len as u64);
            assert_close(dot_f32(&a, &b), scalar::dot_f32(&a, &b));
            assert_close(cosine_f32(&a, &b), scalar::cosine_f32(&a, &b));
            assert_close(l2_f32(&a, &b), scalar::l2_f32(&a, &b));

            let ai: Vec<i8> = a.iter().map(|x| (x * 100.0) as i8).collect();
            let bi: Vec<i8> = b.iter().map(|x| (x * 100.0) as i8).collect();
            assert_close(dot_i8(&ai, &bi), scalar::dot_i8(&ai, &bi));
            assert_close(cosine_i8(&ai, &bi), scalar::cosine_i8(&ai, &bi));
        }
    }

    #[test]
    #[should_panic(expected = "equal-length")]
    fn mismatched_lengths_panic() {
        let _ = dot_f32(&[1.0, 2.0], &[1.0]);
    }
}
```

- [ ] **Step 3: Run the tests to verify they pass (scalar is the ground truth; stubs equal scalar)**

Run: `cargo test -p semantex-core simd::tests -- --nocapture`
Expected: PASS — all of `scalar_dot_matches_hand_computation`, `scalar_l2_matches_hand_computation`, `scalar_cosine_orthogonal_is_zero_and_parallel_is_one`, `cosine_zero_norm_returns_zero`, `scalar_i8_dot_matches_hand_computation`, `dispatch_matches_scalar_across_sizes`, `mismatched_lengths_panic`. (Arch kernels are scalar stubs, so dispatch == scalar trivially. This anchors the parity harness BEFORE real intrinsics exist — the failing→passing flips happen in Tasks 2–6 when a stub is replaced and the same parity test must still hold.)

- [ ] **Step 4: Verify the crate builds clean**

Run: `cargo build -p semantex-core`
Expected: compiles with no errors (the `#[target_feature]` stub `unsafe fn`s are valid; on aarch64 the `avx2` mod is `#[cfg]`-stripped and vice-versa — no dead-code warnings because they're behind `cfg`).

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/search/simd.rs crates/semantex-core/src/search/mod.rs
git commit -m "$(cat <<'EOF'
feat(simd): scalar distance kernels + runtime dispatch skeleton (S6)

Adds search::simd with safe dot_f32/cosine_f32/l2_f32/dot_i8/cosine_i8,
the SIMD_MIN_LEN batch-size gate, and the AVX2/NEON runtime dispatcher.
Arch kernels are scalar stubs for now; real intrinsics follow. Parity
harness (assert_close within 1e-6, sizes across the gate + tail) lands
green against the scalar ground truth.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: AVX2 fp32 dot + L2 kernels

**Files:**
- Modify: `crates/semantex-core/src/search/simd.rs` (the `avx2` submodule)

- [ ] **Step 1: Replace the `dot_f32` and `l2_f32` AVX2 stubs with real intrinsics + add the horizontal-sum helper**

In `crates/semantex-core/src/search/simd.rs`, replace the `#[cfg(target_arch = "x86_64")] mod avx2 { … }` block's `dot_f32` and `l2_f32` stub bodies (leave `cosine_f32`, `dot_i8`, `cosine_i8` stubs untouched for now) and add the import + helper. The full `dot_f32` / `l2_f32` plus helper:

```rust
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    /// Horizontal sum of an 8-lane f32 AVX register → scalar f32.
    /// # Safety
    /// Caller must ensure AVX2 is available (the `__m256` argument already implies it).
    #[target_feature(enable = "avx2")]
    unsafe fn hsum256_ps(v: __m256) -> f32 {
        unsafe {
            // Fold the high 128 lanes onto the low 128.
            let low = _mm256_castps256_ps128(v);
            let high = _mm256_extractf128_ps(v, 1);
            let sum128 = _mm_add_ps(low, high);
            // Fold 4 → 2 → 1.
            let shuf = _mm_movehdup_ps(sum128); // [a1,a1,a3,a3]
            let sums = _mm_add_ps(sum128, shuf); // [a0+a1, _, a2+a3, _]
            let hi64 = _mm_movehl_ps(shuf, sums); // bring a2+a3 to lane 0
            let final_sum = _mm_add_ss(sums, hi64);
            _mm_cvtss_f32(final_sum)
        }
    }

    /// # Safety
    /// Caller must ensure AVX2 is available and `a.len() == b.len()`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let len = a.len();
            let chunks = len & !7; // process 8 f32 at a time
            let mut acc = _mm256_setzero_ps();
            let mut i = 0;
            while i < chunks {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                // fused multiply-add: acc += va * vb
                acc = _mm256_fmadd_ps(va, vb, acc);
                i += 8;
            }
            let mut result = hsum256_ps(acc);
            // scalar tail
            while i < len {
                result += a[i] * b[i];
                i += 1;
            }
            result
        }
    }

    /// # Safety
    /// Caller must ensure AVX2 is available and `a.len() == b.len()`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn l2_f32(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let len = a.len();
            let chunks = len & !7;
            let mut acc = _mm256_setzero_ps();
            let mut i = 0;
            while i < chunks {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                let diff = _mm256_sub_ps(va, vb);
                acc = _mm256_fmadd_ps(diff, diff, acc); // acc += diff²
                i += 8;
            }
            let mut sumsq = hsum256_ps(acc);
            while i < len {
                let d = a[i] - b[i];
                sumsq += d * d;
                i += 1;
            }
            sumsq.sqrt()
        }
    }

    // --- still stubs until later tasks ---

    /// # Safety
    /// Caller must ensure AVX2 is available and `a.len() == b.len()`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
        super::scalar::cosine_f32(a, b)
    }
    /// # Safety
    /// Caller must ensure AVX2 is available and `a.len() == b.len()`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_i8(a: &[i8], b: &[i8]) -> f32 {
        super::scalar::dot_i8(a, b)
    }
    /// # Safety
    /// Caller must ensure AVX2 is available and `a.len() == b.len()`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn cosine_i8(a: &[i8], b: &[i8]) -> f32 {
        super::scalar::cosine_i8(a, b)
    }
}
```

- [ ] **Step 2: Add an x86_64-gated AVX2 parity test for `dot_f32` / `l2_f32`**

Append to the `#[cfg(test)] mod tests` block in `simd.rs`:

```rust
    /// x86_64-only: directly exercise the AVX2 dot/L2 kernels and assert parity with
    /// scalar. `#[cfg]`-gated so it compiles only where the kernel exists; on aarch64
    /// hosts this test is absent (the dispatch test covers NEON there).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_dot_l2_match_scalar() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("skipping avx2 parity: AVX2 not present on this host");
            return;
        }
        for &len in &[8usize, 15, 16, 17, 64, 768, 769] {
            let a = make_vec(len, 0x55 ^ len as u64);
            let b = make_vec(len, 0xAA ^ len as u64);
            // SAFETY: AVX2 detected above; equal-length slices.
            let (d, l) = unsafe { (avx2::dot_f32(&a, &b), avx2::l2_f32(&a, &b)) };
            assert_close(d, scalar::dot_f32(&a, &b));
            assert_close(l, scalar::l2_f32(&a, &b));
        }
    }
```

- [ ] **Step 3: Run the parity tests**

On an **x86_64** host (or x86_64 CI):
Run: `cargo test -p semantex-core simd::tests -- --nocapture`
Expected: PASS, including `avx2_dot_l2_match_scalar` and the still-green `dispatch_matches_scalar_across_sizes` (which now routes `dot_f32`/`l2_f32` through real AVX2 on this host).

On the **aarch64** dev machine: `avx2_dot_l2_match_scalar` is `#[cfg]`-absent; the suite still builds and passes (`dispatch` runs the NEON stub == scalar). To validate AVX2 from an aarch64 host you need x86_64 CI — note in the PR which arch ran.
Run: `cargo test -p semantex-core simd::tests`
Expected: PASS (AVX2 test compiled out).

- [ ] **Step 4: Clippy the unsafe code**

Run: `cargo clippy -p semantex-core --all-targets -- -D warnings`
Expected: no warnings. (Pedantic is `warn` workspace-wide but `-D warnings` promotes them; if clippy flags `cast_precision_loss` on `acc as f32`-style casts elsewhere it's in i8 code added later — fp32 kernels here have no lossy casts.)

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/search/simd.rs
git commit -m "$(cat <<'EOF'
feat(simd): AVX2 dot_f32 + l2_f32 kernels with FMA + horizontal sum (S6)

Real _mm256_fmadd_ps accumulation, len & !7 SIMD body + scalar tail,
hsum256_ps fold. Inner unsafe{} blocks satisfy unsafe_op_in_unsafe_fn.
x86_64-gated parity test holds within 1e-6 vs the scalar reference.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: AVX2 fp32 cosine kernel

**Files:**
- Modify: `crates/semantex-core/src/search/simd.rs` (the `avx2::cosine_f32` fn)

- [ ] **Step 1: Replace the `avx2::cosine_f32` stub with real intrinsics**

In `simd.rs`, replace the `avx2::cosine_f32` body (three parallel FMA accumulators: dot, ‖a‖², ‖b‖²):

```rust
    /// # Safety
    /// Caller must ensure AVX2 is available and `a.len() == b.len()`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let len = a.len();
            let chunks = len & !7;
            let mut dot = _mm256_setzero_ps();
            let mut na = _mm256_setzero_ps();
            let mut nb = _mm256_setzero_ps();
            let mut i = 0;
            while i < chunks {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                dot = _mm256_fmadd_ps(va, vb, dot);
                na = _mm256_fmadd_ps(va, va, na);
                nb = _mm256_fmadd_ps(vb, vb, nb);
                i += 8;
            }
            let mut dot_s = hsum256_ps(dot);
            let mut na_s = hsum256_ps(na);
            let mut nb_s = hsum256_ps(nb);
            while i < len {
                let (x, y) = (a[i], b[i]);
                dot_s += x * y;
                na_s += x * x;
                nb_s += y * y;
                i += 1;
            }
            let denom = na_s.sqrt() * nb_s.sqrt();
            if denom == 0.0 { 0.0 } else { dot_s / denom }
        }
    }
```

- [ ] **Step 2: Extend the AVX2 parity test to cover cosine**

In `simd.rs` tests, replace the body of `avx2_dot_l2_match_scalar` with the cosine-inclusive version (rename to reflect coverage):

```rust
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_dot_l2_cosine_match_scalar() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("skipping avx2 parity: AVX2 not present on this host");
            return;
        }
        for &len in &[8usize, 15, 16, 17, 64, 768, 769] {
            let a = make_vec(len, 0x55 ^ len as u64);
            let b = make_vec(len, 0xAA ^ len as u64);
            // SAFETY: AVX2 detected above; equal-length slices.
            let (d, l, c) =
                unsafe { (avx2::dot_f32(&a, &b), avx2::l2_f32(&a, &b), avx2::cosine_f32(&a, &b)) };
            assert_close(d, scalar::dot_f32(&a, &b));
            assert_close(l, scalar::l2_f32(&a, &b));
            assert_close(c, scalar::cosine_f32(&a, &b));
        }
    }
```

Delete the old `avx2_dot_l2_match_scalar` fn (replaced by `avx2_dot_l2_cosine_match_scalar`).

- [ ] **Step 3: Run the parity tests**

On x86_64:
Run: `cargo test -p semantex-core simd::tests -- --nocapture`
Expected: PASS — `avx2_dot_l2_cosine_match_scalar` plus `dispatch_matches_scalar_across_sizes` (cosine now routes through AVX2 here).

- [ ] **Step 4: Clippy**

Run: `cargo clippy -p semantex-core --all-targets -- -D warnings`
Expected: no warnings.

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/search/simd.rs
git commit -m "$(cat <<'EOF'
feat(simd): AVX2 cosine_f32 kernel (three FMA accumulators) (S6)

Parallel dot / ‖a‖² / ‖b‖² accumulation, zero-norm → 0.0, scalar tail.
x86_64 parity test extended to cosine; within 1e-6 vs scalar.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: NEON fp32 dot + L2 kernels

**Files:**
- Modify: `crates/semantex-core/src/search/simd.rs` (the `neon` submodule)

- [ ] **Step 1: Replace the `neon::dot_f32` and `neon::l2_f32` stubs with real intrinsics**

In `simd.rs`, add the import and replace the `neon` submodule's `dot_f32` / `l2_f32` stub bodies (leave `cosine_f32`, `dot_i8`, `cosine_i8` stubs):

```rust
#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// # Safety
    /// Caller must ensure NEON is available and `a.len() == b.len()`.
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let len = a.len();
            let chunks = len & !3; // 4 f32 per NEON q-register
            let mut acc = vdupq_n_f32(0.0);
            let mut i = 0;
            while i < chunks {
                let va = vld1q_f32(a.as_ptr().add(i));
                let vb = vld1q_f32(b.as_ptr().add(i));
                acc = vfmaq_f32(acc, va, vb); // acc += va * vb (fused)
                i += 4;
            }
            let mut result = vaddvq_f32(acc); // horizontal add of 4 lanes
            while i < len {
                result += a[i] * b[i];
                i += 1;
            }
            result
        }
    }

    /// # Safety
    /// Caller must ensure NEON is available and `a.len() == b.len()`.
    #[target_feature(enable = "neon")]
    pub unsafe fn l2_f32(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let len = a.len();
            let chunks = len & !3;
            let mut acc = vdupq_n_f32(0.0);
            let mut i = 0;
            while i < chunks {
                let va = vld1q_f32(a.as_ptr().add(i));
                let vb = vld1q_f32(b.as_ptr().add(i));
                let diff = vsubq_f32(va, vb);
                acc = vfmaq_f32(acc, diff, diff); // acc += diff²
                i += 4;
            }
            let mut sumsq = vaddvq_f32(acc);
            while i < len {
                let d = a[i] - b[i];
                sumsq += d * d;
                i += 1;
            }
            sumsq.sqrt()
        }
    }

    // --- still stubs until later tasks ---

    /// # Safety
    /// Caller must ensure NEON is available and `a.len() == b.len()`.
    #[target_feature(enable = "neon")]
    pub unsafe fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
        super::scalar::cosine_f32(a, b)
    }
    /// # Safety
    /// Caller must ensure NEON is available and `a.len() == b.len()`.
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_i8(a: &[i8], b: &[i8]) -> f32 {
        super::scalar::dot_i8(a, b)
    }
    /// # Safety
    /// Caller must ensure NEON is available and `a.len() == b.len()`.
    #[target_feature(enable = "neon")]
    pub unsafe fn cosine_i8(a: &[i8], b: &[i8]) -> f32 {
        super::scalar::cosine_i8(a, b)
    }
}
```

- [ ] **Step 2: Add an aarch64-gated NEON parity test**

Append to the `#[cfg(test)] mod tests` block:

```rust
    /// aarch64-only: directly exercise the NEON dot/L2 kernels vs scalar. NEON is
    /// mandatory on aarch64, so no runtime skip is needed. `#[cfg]`-gated so it
    /// compiles only on aarch64; on x86_64 hosts this test is absent.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_dot_l2_match_scalar() {
        for &len in &[4usize, 7, 8, 9, 64, 768, 769] {
            let a = make_vec(len, 0x55 ^ len as u64);
            let b = make_vec(len, 0xAA ^ len as u64);
            // SAFETY: NEON is mandatory on aarch64; equal-length slices.
            let (d, l) = unsafe { (neon::dot_f32(&a, &b), neon::l2_f32(&a, &b)) };
            assert_close(d, scalar::dot_f32(&a, &b));
            assert_close(l, scalar::l2_f32(&a, &b));
        }
    }
```

- [ ] **Step 3: Run the parity tests (this is the dev machine's native arch — runs locally)**

Run: `cargo test -p semantex-core simd::tests -- --nocapture`
Expected: PASS — `neon_dot_l2_match_scalar` plus `dispatch_matches_scalar_across_sizes` (dot/L2 now route through real NEON on aarch64). This is the FAIL→PASS flip relative to the Task-1 NEON stub.

- [ ] **Step 4: Clippy**

Run: `cargo clippy -p semantex-core --all-targets -- -D warnings`
Expected: no warnings.

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/search/simd.rs
git commit -m "$(cat <<'EOF'
feat(simd): NEON dot_f32 + l2_f32 kernels (vfmaq_f32 + vaddvq_f32) (S6)

len & !3 body, vfmaq_f32 fused accumulate, vaddvq_f32 horizontal sum,
scalar tail. aarch64 parity test holds within 1e-6 vs scalar.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: NEON fp32 cosine kernel

**Files:**
- Modify: `crates/semantex-core/src/search/simd.rs` (the `neon::cosine_f32` fn)

- [ ] **Step 1: Replace the `neon::cosine_f32` stub with real intrinsics**

In `simd.rs`, replace the `neon::cosine_f32` body (three NEON FMA accumulators):

```rust
    /// # Safety
    /// Caller must ensure NEON is available and `a.len() == b.len()`.
    #[target_feature(enable = "neon")]
    pub unsafe fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let len = a.len();
            let chunks = len & !3;
            let mut dot = vdupq_n_f32(0.0);
            let mut na = vdupq_n_f32(0.0);
            let mut nb = vdupq_n_f32(0.0);
            let mut i = 0;
            while i < chunks {
                let va = vld1q_f32(a.as_ptr().add(i));
                let vb = vld1q_f32(b.as_ptr().add(i));
                dot = vfmaq_f32(dot, va, vb);
                na = vfmaq_f32(na, va, va);
                nb = vfmaq_f32(nb, vb, vb);
                i += 4;
            }
            let mut dot_s = vaddvq_f32(dot);
            let mut na_s = vaddvq_f32(na);
            let mut nb_s = vaddvq_f32(nb);
            while i < len {
                let (x, y) = (a[i], b[i]);
                dot_s += x * y;
                na_s += x * x;
                nb_s += y * y;
                i += 1;
            }
            let denom = na_s.sqrt() * nb_s.sqrt();
            if denom == 0.0 { 0.0 } else { dot_s / denom }
        }
    }
```

- [ ] **Step 2: Extend the NEON parity test to cover cosine**

In `simd.rs` tests, replace `neon_dot_l2_match_scalar` with the cosine-inclusive version:

```rust
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_dot_l2_cosine_match_scalar() {
        for &len in &[4usize, 7, 8, 9, 64, 768, 769] {
            let a = make_vec(len, 0x55 ^ len as u64);
            let b = make_vec(len, 0xAA ^ len as u64);
            // SAFETY: NEON is mandatory on aarch64; equal-length slices.
            let (d, l, c) =
                unsafe { (neon::dot_f32(&a, &b), neon::l2_f32(&a, &b), neon::cosine_f32(&a, &b)) };
            assert_close(d, scalar::dot_f32(&a, &b));
            assert_close(l, scalar::l2_f32(&a, &b));
            assert_close(c, scalar::cosine_f32(&a, &b));
        }
    }
```

Delete the old `neon_dot_l2_match_scalar`.

- [ ] **Step 3: Run the parity tests**

Run: `cargo test -p semantex-core simd::tests -- --nocapture`
Expected: PASS — `neon_dot_l2_cosine_match_scalar` and `dispatch_matches_scalar_across_sizes` (cosine now via NEON on aarch64).

- [ ] **Step 4: Clippy**

Run: `cargo clippy -p semantex-core --all-targets -- -D warnings`
Expected: no warnings.

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/src/search/simd.rs
git commit -m "$(cat <<'EOF'
feat(simd): NEON cosine_f32 kernel (three FMA accumulators) (S6)

Parallel dot / ‖a‖² / ‖b‖² via vfmaq_f32, zero-norm → 0.0, scalar tail.
aarch64 parity test extended to cosine; within 1e-6 vs scalar.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: int8 dot + cosine kernels (AVX2 + NEON)

**Files:**
- Modify: `crates/semantex-core/src/search/simd.rs` (the `avx2` and `neon` i8 fns)

The int8 path scores scalar-quantized vectors before fp32 rescore (spec §4 S6, §2 D7). Accumulate products in i32 (exact: `|i8·i8| ≤ 16384`, and for dim ≤ ~131072 the i32 sum cannot overflow ±2.1e9), then convert to f32. Both arch kernels widen i8→i16→i32; the AVX2 path uses `_mm256_madd_epi16` (multiply i16 pairs, horizontally add adjacent → i32), the NEON path uses `vmull_s8` + pairwise widening adds.

- [ ] **Step 1: Replace the AVX2 i8 stubs with real intrinsics**

In `simd.rs`, replace `avx2::dot_i8` and `avx2::cosine_i8`. Add an i32-horizontal-sum helper inside `mod avx2`:

```rust
    /// Horizontal sum of an 8-lane i32 AVX register → scalar i32.
    /// # Safety
    /// Caller must ensure AVX2 is available.
    #[target_feature(enable = "avx2")]
    unsafe fn hsum256_epi32(v: __m256i) -> i32 {
        unsafe {
            let low = _mm256_castsi256_si128(v);
            let high = _mm256_extracti128_si256(v, 1);
            let sum128 = _mm_add_epi32(low, high);
            let hi64 = _mm_unpackhi_epi64(sum128, sum128);
            let sum64 = _mm_add_epi32(sum128, hi64);
            let hi32 = _mm_shuffle_epi32(sum64, 0b01);
            let sum32 = _mm_add_epi32(sum64, hi32);
            _mm_cvtsi128_si32(sum32)
        }
    }

    /// Widen 16 i8 (in the low 128 bits of a loaded register) to two i16x8 halves,
    /// returned as one i16x16 (`__m256i`). Caller loads via `_mm_loadu_si128`.
    /// # Safety
    /// Caller must ensure AVX2 is available.
    #[target_feature(enable = "avx2")]
    unsafe fn widen_i8_to_i16(v: __m128i) -> __m256i {
        unsafe { _mm256_cvtepi8_epi16(v) } // sign-extend 16×i8 → 16×i16
    }

    /// # Safety
    /// Caller must ensure AVX2 is available and `a.len() == b.len()`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_i8(a: &[i8], b: &[i8]) -> f32 {
        unsafe {
            let len = a.len();
            let chunks = len & !15; // 16 i8 per iteration
            let mut acc = _mm256_setzero_si256();
            let mut i = 0;
            while i < chunks {
                let va = widen_i8_to_i16(_mm_loadu_si128(a.as_ptr().add(i).cast::<__m128i>()));
                let vb = widen_i8_to_i16(_mm_loadu_si128(b.as_ptr().add(i).cast::<__m128i>()));
                // madd: (va0*vb0 + va1*vb1), ... → 8×i32
                let prod = _mm256_madd_epi16(va, vb);
                acc = _mm256_add_epi32(acc, prod);
                i += 16;
            }
            let mut total = hsum256_epi32(acc);
            while i < len {
                total += i32::from(a[i]) * i32::from(b[i]);
                i += 1;
            }
            total as f32
        }
    }

    /// # Safety
    /// Caller must ensure AVX2 is available and `a.len() == b.len()`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn cosine_i8(a: &[i8], b: &[i8]) -> f32 {
        unsafe {
            let len = a.len();
            let chunks = len & !15;
            let mut dot = _mm256_setzero_si256();
            let mut na = _mm256_setzero_si256();
            let mut nb = _mm256_setzero_si256();
            let mut i = 0;
            while i < chunks {
                let va = widen_i8_to_i16(_mm_loadu_si128(a.as_ptr().add(i).cast::<__m128i>()));
                let vb = widen_i8_to_i16(_mm_loadu_si128(b.as_ptr().add(i).cast::<__m128i>()));
                dot = _mm256_add_epi32(dot, _mm256_madd_epi16(va, vb));
                na = _mm256_add_epi32(na, _mm256_madd_epi16(va, va));
                nb = _mm256_add_epi32(nb, _mm256_madd_epi16(vb, vb));
                i += 16;
            }
            let mut dot_s = hsum256_epi32(dot);
            let mut na_s = hsum256_epi32(na);
            let mut nb_s = hsum256_epi32(nb);
            while i < len {
                let (x, y) = (i32::from(a[i]), i32::from(b[i]));
                dot_s += x * y;
                na_s += x * x;
                nb_s += y * y;
                i += 1;
            }
            let denom = (na_s as f32).sqrt() * (nb_s as f32).sqrt();
            if denom == 0.0 { 0.0 } else { dot_s as f32 / denom }
        }
    }
```

- [ ] **Step 2: Replace the NEON i8 stubs with real intrinsics**

In `simd.rs`, replace `neon::dot_i8` and `neon::cosine_i8`:

```rust
    /// # Safety
    /// Caller must ensure NEON is available and `a.len() == b.len()`.
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_i8(a: &[i8], b: &[i8]) -> f32 {
        unsafe {
            let len = a.len();
            let chunks = len & !7; // 8 i8 per iteration (vld1_s8 loads 8)
            let mut acc = vdupq_n_s32(0);
            let mut i = 0;
            while i < chunks {
                let va = vld1_s8(a.as_ptr().add(i)); // int8x8
                let vb = vld1_s8(b.as_ptr().add(i));
                let prod = vmull_s8(va, vb); // int16x8 = va * vb (widened)
                // widen-add int16x8 into int32x4 accumulator
                acc = vaddq_s32(acc, vpaddlq_s16(prod)); // pairwise widen i16→i32 then add
                i += 8;
            }
            let mut total = vaddvq_s32(acc); // horizontal add 4×i32
            while i < len {
                total += i32::from(a[i]) * i32::from(b[i]);
                i += 1;
            }
            total as f32
        }
    }

    /// # Safety
    /// Caller must ensure NEON is available and `a.len() == b.len()`.
    #[target_feature(enable = "neon")]
    pub unsafe fn cosine_i8(a: &[i8], b: &[i8]) -> f32 {
        unsafe {
            let len = a.len();
            let chunks = len & !7;
            let mut dot = vdupq_n_s32(0);
            let mut na = vdupq_n_s32(0);
            let mut nb = vdupq_n_s32(0);
            let mut i = 0;
            while i < chunks {
                let va = vld1_s8(a.as_ptr().add(i));
                let vb = vld1_s8(b.as_ptr().add(i));
                dot = vaddq_s32(dot, vpaddlq_s16(vmull_s8(va, vb)));
                na = vaddq_s32(na, vpaddlq_s16(vmull_s8(va, va)));
                nb = vaddq_s32(nb, vpaddlq_s16(vmull_s8(vb, vb)));
                i += 8;
            }
            let mut dot_s = vaddvq_s32(dot);
            let mut na_s = vaddvq_s32(na);
            let mut nb_s = vaddvq_s32(nb);
            while i < len {
                let (x, y) = (i32::from(a[i]), i32::from(b[i]));
                dot_s += x * y;
                na_s += x * x;
                nb_s += y * y;
                i += 1;
            }
            let denom = (na_s as f32).sqrt() * (nb_s as f32).sqrt();
            if denom == 0.0 { 0.0 } else { dot_s as f32 / denom }
        }
    }
```

- [ ] **Step 3: Add arch-gated i8 parity tests + an exactness check (i8 dot is integer-exact, so equality, not tolerance)**

Append to the `#[cfg(test)] mod tests` block:

```rust
    /// i8 dot is an exact integer computation regardless of lane ordering, so the
    /// SIMD result must EQUAL the scalar i32 sum (cast to f32) — no tolerance.
    /// Cosine involves a sqrt, so compare with tolerance.
    fn i8_vecs(len: usize, seed: u64) -> (Vec<i8>, Vec<i8>) {
        let af = make_vec(len, seed);
        let bf = make_vec(len, seed ^ 0xFFFF);
        (
            af.iter().map(|x| (x * 127.0) as i8).collect(),
            bf.iter().map(|x| (x * 127.0) as i8).collect(),
        )
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_i8_match_scalar() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("skipping avx2 i8 parity: AVX2 not present");
            return;
        }
        for &len in &[16usize, 17, 31, 32, 768, 769] {
            let (a, b) = i8_vecs(len, 0x9 ^ len as u64);
            // SAFETY: AVX2 detected; equal-length slices.
            let (d, c) = unsafe { (avx2::dot_i8(&a, &b), avx2::cosine_i8(&a, &b)) };
            assert_eq!(d, scalar::dot_i8(&a, &b), "i8 dot must be integer-exact");
            assert_close(c, scalar::cosine_i8(&a, &b));
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_i8_match_scalar() {
        for &len in &[8usize, 9, 15, 16, 768, 769] {
            let (a, b) = i8_vecs(len, 0x9 ^ len as u64);
            // SAFETY: NEON mandatory on aarch64; equal-length slices.
            let (d, c) = unsafe { (neon::dot_i8(&a, &b), neon::cosine_i8(&a, &b)) };
            assert_eq!(d, scalar::dot_i8(&a, &b), "i8 dot must be integer-exact");
            assert_close(c, scalar::cosine_i8(&a, &b));
        }
    }
```

- [ ] **Step 4: Run the parity tests**

On aarch64 (dev machine):
Run: `cargo test -p semantex-core simd::tests -- --nocapture`
Expected: PASS — `neon_i8_match_scalar` (exact integer dot, cosine within tolerance) + the i8 branch of `dispatch_matches_scalar_across_sizes` now routing through real NEON.

On x86_64 CI:
Run: `cargo test -p semantex-core simd::tests`
Expected: PASS — `avx2_i8_match_scalar` plus the i8 dispatch branch.

- [ ] **Step 5: Clippy (i8 casts are the likely flag site)**

Run: `cargo clippy -p semantex-core --all-targets -- -D warnings`
Expected: no warnings. If clippy `pedantic` flags `cast_precision_loss` on `total as f32` / `na_s as f32`, add a scoped `#[allow(clippy::cast_precision_loss)]` on the i8 fns with the comment `// i32→f32: magnitudes ≤ dim·16384 ≪ 2^24, exactly representable`. (These casts are intentional and bounded; the allow is local, not module-wide.)

- [ ] **Step 6: Commit**

```bash
git add crates/semantex-core/src/search/simd.rs
git commit -m "$(cat <<'EOF'
feat(simd): int8 dot + cosine kernels (AVX2 madd / NEON vmull) (S6)

i32 accumulation (exact for dim ≤ 131072): AVX2 _mm256_madd_epi16 over
sign-extended i16; NEON vmull_s8 + vpaddlq_s16. i8 dot parity is exact
equality; cosine within 1e-6. For scoring scalar-quantized vectors
before fp32 rescore (S2).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Criterion benchmark + `[[bench]]` registration (proves the speedup)

**Files:**
- Create: `crates/semantex-core/benches/simd_bench.rs`
- Modify: `crates/semantex-core/Cargo.toml:136-137`

The spec acceptance requires "criterion benchmark shows speedup at dim 768". criterion is already a `semantex-core` dev-dependency, but no `[[bench]]` target exists in the repo — register one so it actually builds/runs.

- [ ] **Step 1: Register the bench target in `semantex-core/Cargo.toml`**

Edit `crates/semantex-core/Cargo.toml` — append after the existing `[dev-dependencies]` block (lines 136-137):

```toml
[dev-dependencies]
criterion = { version = "0.8", features = ["html_reports"] }

[[bench]]
name = "simd_bench"
harness = false
```

- [ ] **Step 2: Write the benchmark**

Create `crates/semantex-core/benches/simd_bench.rs`. It benchmarks the dispatched public fn vs the (re-implemented inline) scalar baseline at dim 768 (and a couple of others) so the criterion report shows the SIMD speedup. The scalar baseline is re-implemented locally because the module's `scalar` submodule is private — that's fine; it's a one-line reference and keeps the bench self-contained.

```rust
#![allow(clippy::unwrap_used)]

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use semantex_core::search::simd;
use std::hint::black_box;

/// Tiny deterministic LCG → reproducible vectors without a rand dependency.
fn make_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

/// Local scalar baseline (the module's `scalar` submodule is private).
fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn scalar_cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (&x, &y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 { 0.0 } else { dot / denom }
}

fn bench_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_f32");
    for &dim in &[128usize, 768, 1536] {
        let a = make_vec(dim, 1);
        let b = make_vec(dim, 2);
        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::new("scalar", dim), &dim, |bn, _| {
            bn.iter(|| black_box(scalar_dot(black_box(&a), black_box(&b))));
        });
        group.bench_with_input(BenchmarkId::new("simd", dim), &dim, |bn, _| {
            bn.iter(|| black_box(simd::dot_f32(black_box(&a), black_box(&b))));
        });
    }
    group.finish();
}

fn bench_cosine(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_f32");
    for &dim in &[128usize, 768, 1536] {
        let a = make_vec(dim, 3);
        let b = make_vec(dim, 4);
        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::new("scalar", dim), &dim, |bn, _| {
            bn.iter(|| black_box(scalar_cosine(black_box(&a), black_box(&b))));
        });
        group.bench_with_input(BenchmarkId::new("simd", dim), &dim, |bn, _| {
            bn.iter(|| black_box(simd::cosine_f32(black_box(&a), black_box(&b))));
        });
    }
    group.finish();
}

fn bench_dot_i8(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_i8");
    for &dim in &[768usize, 1536] {
        let a: Vec<i8> = make_vec(dim, 5).iter().map(|x| (x * 127.0) as i8).collect();
        let b: Vec<i8> = make_vec(dim, 6).iter().map(|x| (x * 127.0) as i8).collect();
        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::new("simd", dim), &dim, |bn, _| {
            bn.iter(|| black_box(simd::dot_i8(black_box(&a), black_box(&b))));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_dot, bench_cosine, bench_dot_i8);
criterion_main!(benches);
```

- [ ] **Step 3: Build the bench (compile-only first, fast feedback)**

Run: `cargo bench -p semantex-core --bench simd_bench --no-run`
Expected: compiles; prints an `Executable …/simd_bench-<hash>` line. (Confirms the `[[bench]]` registration + `harness = false` wiring is correct.)

- [ ] **Step 4: Run the benchmark and confirm a dim-768 speedup**

Run: `cargo bench -p semantex-core --bench simd_bench -- dot_f32`
Expected: criterion prints `dot_f32/scalar/768` and `dot_f32/simd/768` timings; the `simd/768` time is **measurably lower** than `scalar/768` (on AVX2/NEON hardware, typically 3–6× for dot at dim 768). Then run the full set:
Run: `cargo bench -p semantex-core --bench simd_bench`
Expected: all three groups (`dot_f32`, `cosine_f32`, `dot_i8`) complete; `simd` < `scalar` at dim 768 for dot and cosine. (On a host lacking AVX2/NEON the dispatcher falls to scalar and the two lines converge — that is correct behavior, not a failure; note the host arch in the PR.)

- [ ] **Step 5: Commit**

```bash
git add crates/semantex-core/benches/simd_bench.rs crates/semantex-core/Cargo.toml
git commit -m "$(cat <<'EOF'
bench(simd): criterion scalar-vs-SIMD benchmark at dim 768 (S6)

Registers the first [[bench]] target in the repo (harness=false) so
cargo bench actually builds it. Compares scalar vs dispatched dot_f32 /
cosine_f32 / dot_i8 at dims 128/768/1536; shows the SIMD speedup at 768.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Unsafe audit + full-suite verification

**Files:**
- Modify: `crates/semantex-core/src/search/simd.rs` (only if the audit finds a gap; otherwise no code change — this task is verification + the `# Safety` doc completeness pass)

- [ ] **Step 1: Confirm every `unsafe fn` documents its safety contract and every intrinsic call sits in an inner `unsafe {}` block**

Visually audit `simd.rs`: each `pub unsafe fn` in `avx2`/`neon` and each private helper (`hsum256_ps`, `hsum256_epi32`, `widen_i8_to_i16`) MUST carry a `/// # Safety` doc and a body wrapped in `unsafe { … }`. Each call site in the public dispatcher MUST have a `// SAFETY:` comment justifying the runtime-detection precondition. (All of these are present as written in Tasks 1–6; this step verifies nothing regressed during edits.)

Quick grep to confirm no bare (undocumented) `unsafe fn`:
Run: `grep -n "unsafe fn" crates/semantex-core/src/search/simd.rs`
Expected: every match is immediately preceded (in the file) by a `/// # Safety` line. Eyeball the list (should be: 5 avx2 pub fns + 3 avx2 helpers, 5 neon pub fns — guarded by `#[cfg]`).

- [ ] **Step 2: Run the focused module test suite once more (all arches' shared tests + the host arch's kernel tests)**

Run: `cargo test -p semantex-core simd -- --nocapture`
Expected: PASS. On aarch64: `dispatch_matches_scalar_across_sizes`, `neon_dot_l2_cosine_match_scalar`, `neon_i8_match_scalar`, the scalar unit tests, `mismatched_lengths_panic`. On x86_64: the `avx2_*` tests instead of the `neon_*` ones.

- [ ] **Step 3: Run the FULL crate test suite to confirm no regression elsewhere**

Run: `cargo test -p semantex-core`
Expected: PASS — the existing ~780+ lib tests plus the new `simd` tests, no failures, no new warnings.

- [ ] **Step 4: Clippy the whole crate including benches, deny warnings (the unsafe-audit gate)**

Run: `cargo clippy -p semantex-core --all-targets -- -D warnings`
Expected: zero warnings. This is the spec's "`unsafe` audited" gate — pedantic + the unsafe lints all pass.

- [ ] **Step 5: Format**

Run: `cargo fmt -p semantex-core`
Expected: no diff (or only whitespace it auto-fixes). If it reformats `simd.rs`, that's fine — stage it.

- [ ] **Step 6: Commit (audit pass; may be docs/format-only)**

```bash
git add crates/semantex-core/src/search/simd.rs
git commit -m "$(cat <<'EOF'
chore(simd): unsafe-audit pass — Safety docs + clippy -D warnings green (S6)

Every unsafe fn carries a # Safety contract; every intrinsic call is in an
inner unsafe{} block (unsafe_op_in_unsafe_fn); every dispatch call site has
a SAFETY justification. Full crate test suite + clippy --all-targets green.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

(If Step 6 has nothing to commit because the audit found no gaps and fmt produced no diff, skip the commit — the work is already captured in Tasks 1–7.)

---

## Self-Review (run against the spec §4 S6 + §2 D7 + §5)

**1. Spec coverage** — every S6 requirement maps to a task:

| Spec §4 S6 requirement | Task(s) |
|---|---|
| New module `crates/semantex-core/src/search/simd.rs` | Task 1 |
| `dot_f32`, `cosine_f32`, `l2_f32` | Tasks 1 (scalar) → 2,3 (AVX2) → 4,5 (NEON) |
| int8 dot/cosine path for quantized scoring | Task 6 |
| AVX2 (x86_64) + NEON (aarch64) + scalar fallback | Tasks 1–6 |
| Runtime-dispatched (`is_x86_feature_detected!`) | Task 1 dispatcher |
| `len & !N` + scalar tail | Tasks 2–6 (`& !7` AVX2 fp32, `& !3` NEON fp32, `& !15` AVX2 i8, `& !7` NEON i8) |
| Reimplement from oxirs reference, zero external deps (`std::arch` only) | Tasks 2–6 (own code; no copy; `use std::arch::*` only) |
| Batch-size gating below `SIMD_MIN_LEN` → scalar | Task 1 (`simd_min_len()` + per-fn gate) |
| Consumed by S2 (start scalar, swap in) | "Public API" section + scalar-first phasing |
| Parity within `1e-6` (FMA reorders) | `assert_close` (Task 1), every parity test |
| Criterion benchmark, speedup at dim 768 | Task 7 |
| `unsafe` audited + `cfg`-gated per arch | Tasks 1–6 (`#[cfg(target_arch)]` + `#[target_feature]` + inner `unsafe{}`), Task 8 audit |
| §2 D7: int8 vectors + fp32 rescore | `dot_i8`/`cosine_i8` (Task 6) for the int8 prefilter; `dot_f32`/`cosine_f32` for fp32 rescore |
| §5: "S6 has no dependencies; consumed by S2" | No edits outside `search/simd.rs`, `search/mod.rs`, `benches/`, `Cargo.toml`; does not touch `hybrid.rs` — zero contention with S1/S2/S7 |

**2. Placeholder scan** — no "TBD/TODO/similar to Task N"; every code step is complete real Rust with real intrinsics (`_mm256_loadu_ps`, `_mm256_fmadd_ps`, `_mm256_castps256_ps128`/`_mm256_extractf128_ps` horizontal sum, `_mm256_madd_epi16`, `_mm256_cvtepi8_epi16`; `vld1q_f32`, `vfmaq_f32`, `vaddvq_f32`, `vmull_s8`, `vpaddlq_s16`, `vaddvq_s32`). All intrinsics are real `std::arch::{x86_64,aarch64}` names.

**3. Type/signature consistency** — the public fns (`dot_f32`, `cosine_f32`, `l2_f32`, `dot_i8`, `cosine_i8`) keep identical signatures across the dispatcher, scalar, avx2, and neon submodules in every task. `assert_close` / `make_vec` defined once (Task 1) and reused. The cosine convention (similarity in [-1,1], not oxirs's `1 - cos` distance) is fixed in Task 1's scalar and matched by every arch kernel and test.

---

## Spec gaps surfaced (for the controller / S2 team)

- **G1 — public fn signatures S2 should call (the spec only says "exposing `dot_f32`, `cosine_f32`, `l2_f32`").** This plan pins them as free functions in `semantex_core::search::simd`:
  - `pub fn dot_f32(a: &[f32], b: &[f32]) -> f32`
  - `pub fn cosine_f32(a: &[f32], b: &[f32]) -> f32`  ← **similarity** in [-1,1] (NOT distance; distance = `1.0 - cosine_f32`). The oxirs reference returns `1 - cos` ("cosine_distance"); we deliberately return similarity because the dense channel ranks by similarity. S2 must use `1.0 - cosine_f32(..)` if it wants a distance.
  - `pub fn l2_f32(a: &[f32], b: &[f32]) -> f32`  ← Euclidean **distance** (`sqrt(Σ d²)`), not squared.
  - `pub fn dot_i8(a: &[i8], b: &[i8]) -> f32`, `pub fn cosine_i8(a: &[i8], b: &[i8]) -> f32`.
  All require `a.len() == b.len()` and **panic** otherwise.

- **G2 — int8 quantization symmetry contract.** S6's `dot_i8`/`cosine_i8` assume **symmetric** quantization (zero-point 0, i.e. the i8 already encodes `round(x/scale)`). Spec §2 D7 / §4 S2 says "scale+zero-point per the standard scalar-quant recipe" — if S2 picks **asymmetric** quant (non-zero zero-point), raw i8 cosine is biased and S2 must dequantize before calling the f32 path (or subtract the zero-point first). Recommend S2 use symmetric quant for embeddings (they're ~zero-centered after L2-norm), which makes `cosine_i8` directly meaningful. **Decision needed by S2.**

- **G3 — module path.** Spec §4 S6 offers "`search/simd.rs` (or `embedding/simd.rs`)". This plan commits to `crates/semantex-core/src/search/simd.rs` (consumers — HNSW search, MMR, rescore — all live under `search/` and `index/`; `search` is the natural home and matches the S1/S2 seam location). No action needed unless S2 prefers `embedding/`.

- **G4 — no `[[bench]]` infrastructure existed.** The repo's `benches/*.rs` at the workspace root are orphaned (virtual manifest, no `[package]`, criterion only in `semantex-core` dev-deps, no `[[bench]]` registration anywhere). Task 7 introduces the **first** working `[[bench]]` target (`crates/semantex-core/benches/simd_bench.rs`). If the team later wants the existing root benches runnable, they need the same treatment (separate follow-up; out of S6 scope).

- **G5 — AVX2 i8 load width / remainder.** The AVX2 i8 kernel processes 16 i8/iter (`& !15`) via `_mm_loadu_si128` + `_mm256_cvtepi8_epi16`; the NEON i8 kernel processes 8/iter (`& !7`) via `vld1_s8`. Both have correct scalar tails for non-multiple dims (tested at 17/31/769 etc.). No SSE4.1-vs-AVX2 ambiguity: `_mm256_cvtepi8_epi16` and `_mm256_madd_epi16` are both AVX2, covered by the single `avx2` `#[target_feature]`. (Recorded so reviewers don't flag the mixed `_mm_`/`_mm256_` intrinsics — the `_mm_loadu_si128` load is a 128-bit load feeding the 256-bit widen, which is the standard idiom.)
