//! Contains benchmarks of the functions stored in neural_network::functions.
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use neat_gru::neural_network::functions::*;

extern crate neat_gru;

fn bench_sigmoid(c: &mut Criterion) {
    let size: f32 = 0.3518392;
    let mut group = c.benchmark_group("Sigmoid Function");
    for size in [
        size * 0.0,
        size,
        size * 2.0,
        size * 4.0,
        size * 6.0,
        size * 8.0,
        size * 10.0,
        size * 12.0,
        size * 14.0,
    ]
    .iter()
    {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, size| {
            b.iter(|| fast_sigmoid(*size))
        });
    }
    group.finish();
}

fn bench_tanh(c: &mut Criterion) {
    let size: f32 = 0.3518392;
    let mut group = c.benchmark_group("tanh Function");
    for size in [
        size * 0.0,
        size,
        size * 2.0,
        size * 4.0,
        size * 6.0,
        size * 8.0,
        size * 10.0,
        size * 12.0,
        size * 14.0,
    ]
    .iter()
    {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, size| {
            b.iter(|| fast_tanh(*size))
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bench_tanh, bench_sigmoid,
}
criterion_main!(benches);
