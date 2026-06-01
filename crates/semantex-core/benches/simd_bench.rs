#![allow(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use semantex_core::search::simd;
use std::hint::black_box;

/// Tiny deterministic LCG → reproducible vectors without a rand dependency.
fn make_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(2_862_933_555_777_941_757)
        .wrapping_add(3_037_000_493);
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
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
