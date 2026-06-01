//! Portable SIMD distance kernels (dot / cosine / L2) for the single-vector dense
//! backend's hot path (ANN distance, brute-force fallback, fp32 rescore).
//!
//! Public safe functions own the runtime dispatch:
//!   * below [`simd_min_len`] (a batch-size gate; SIMD setup + horizontal reduction
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

// `is_aarch64_feature_detected!` is not in the prelude on all toolchains; import it
// under the matching arch cfg. (`is_x86_feature_detected!` is in the std prelude.)
#[cfg(target_arch = "aarch64")]
use std::arch::is_aarch64_feature_detected;

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
