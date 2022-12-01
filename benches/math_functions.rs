//! Contains benchmarks of the functions stored in neural_network::functions.
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use neat_gru::neural_network::functions::*;

extern crate neat_gru;
fn input_data() -> impl IntoIterator<Item = f32>{
    let size: f32 = 0.3518392;
    (0..20).map(move |i| size * i as f32)
}
fn bench_sigmoid(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sigmoid Function");
    input_data().into_iter().for_each(|s|{
        group.bench_with_input(BenchmarkId::from_parameter(s), &s, |b, size| {
            b.iter(|| fast_sigmoid(*size))
        });
    });
    group.finish();
}

fn bench_tanh(c: &mut Criterion) {
    let mut group = c.benchmark_group("Tanh Function");
    input_data().into_iter().for_each(|s|{
        group.bench_with_input(BenchmarkId::from_parameter(s), &s, |b, s| {
            b.iter(|| fast_tanh(*s))
        });
    });
    group.finish();
}

fn bench_relu(c: &mut Criterion) {
    let mut group = c.benchmark_group("Relu Function");
    input_data().into_iter().for_each(|s|{
        group.bench_with_input(BenchmarkId::from_parameter(s), &s, |b, s| {
            b.iter(|| re_lu(*s))
        });
    });
    group.finish();
}

fn comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Relu vs Sigmoid");
    input_data().into_iter().for_each(|s|
    {
        group.bench_with_input(BenchmarkId::new("Sigmoid", s), &s,
        |b, size| b.iter(|| fast_sigmoid(*size)));
        group.bench_with_input(BenchmarkId::new("Relu", s), &s,
                               |b, s| b.iter(|| re_lu(*s)));
        group.bench_with_input(BenchmarkId::new("Tanh", s), &s,
                               |b, s| b.iter(|| fast_tanh(*s)));
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bench_tanh, bench_sigmoid, bench_relu, comparison
}
criterion_main!(benches);
