//! Distance kernels for the single-vector dense path (S2 hot path + brute-force
//! fallback + fp32 rescore).
//!
//! **SCALAR SHIM — S6 replaces the bodies.** S6 owns `search::simd` and will
//! swap these scalar bodies for runtime-dispatched AVX2/NEON behind these EXACT
//! locked signatures (integration §3 item 5 / §6). The five locked kernels are
//! `dot_f32`, `cosine_f32` (cosine SIMILARITY in [-1, 1]), `l2_f32`, `dot_i8`,
//! and `cosine_i8`. S2 needs `dot_f32`/`cosine_f32`/`dot_i8`; the other two come
//! along so the signature surface matches S6's contract. If S6 lands first, S6
//! MODIFIES this file (swaps bodies, keeps the signatures) — it does NOT
//! re-create it. The parity tests below are S6's contract; keep them.

/// Dot product of two equal-length f32 slices. Panics on length mismatch.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot_f32 length mismatch");
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Cosine SIMILARITY in `[-1, 1]`. Inputs need NOT be normalized.
#[inline]
pub fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_f32(a, b);
    let na = dot_f32(a, a).sqrt();
    let nb = dot_f32(b, b).sqrt();
    let denom = (na * nb).max(1e-12);
    dot / denom
}

/// Squared Euclidean (L2) distance between two equal-length f32 slices.
/// (S6-locked signature; returns the SQUARED distance — callers take `.sqrt()`
/// when a true L2 norm is needed.)
#[inline]
pub fn l2_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "l2_f32 length mismatch");
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Dot product of two equal-length i8 slices (for scoring quantized vectors
/// before fp32 rescore). Accumulates in i32 then returns `as f32` — for 768-dim
/// int8 the max magnitude is 768 * 127 * 127 ≈ 1.2e7, within i32 AND within
/// f32's exact-integer range (2^24 ≈ 1.7e7), so the `as f32` cast is lossless.
/// Returns `f32` to match the S6-locked kernel signature (integration §3 item 5).
#[inline]
pub fn dot_i8(a: &[i8], b: &[i8]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot_i8 length mismatch");
    let acc: i32 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| i32::from(x) * i32::from(y))
        .sum();
    acc as f32
}

/// Cosine SIMILARITY in `[-1, 1]` over two equal-length i8 slices (symmetric
/// quantization, zero-point 0 — integration §4 D-int8). S6-locked signature.
#[inline]
pub fn cosine_i8(a: &[i8], b: &[i8]) -> f32 {
    let dot = dot_i8(a, b);
    let na = dot_i8(a, a).sqrt();
    let nb = dot_i8(b, b).sqrt();
    let denom = (na * nb).max(1e-12);
    dot / denom
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
    fn l2_f32_basic() {
        // squared distance: (1-4)^2 + (2-6)^2 = 9 + 16 = 25
        assert!((l2_f32(&[1.0, 2.0], &[4.0, 6.0]) - 25.0).abs() < 1e-6);
        assert_eq!(l2_f32(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 0.0);
    }

    #[test]
    fn dot_i8_basic() {
        assert!((dot_i8(&[1, 2, 3], &[4, 5, 6]) - 32.0_f32).abs() < 1e-6);
        assert_eq!(dot_i8(&[127, -127], &[127, 127]), 0.0_f32); // 127*127 - 127*127
    }

    #[test]
    fn cosine_i8_parallel_is_one() {
        assert!((cosine_i8(&[10, 20, 30], &[10, 20, 30]) - 1.0).abs() < 1e-6);
        assert!(cosine_i8(&[1, 0], &[0, 1]).abs() < 1e-6);
    }
}
