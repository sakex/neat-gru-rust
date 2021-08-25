use criterion::{black_box, criterion_group, criterion_main, Criterion};
extern crate neat_gru;
use std::fs::File;
use neat_gru::neural_network::nn;
use std::io::Read;

fn benchmark(c: &mut Criterion) {
    let mut file = File::open("XOR").expect("Can't open XOR file. Run 'Example' first.");
    let mut file_string = &mut "".to_string();
    file.read_to_string(file_string);
    let mut network =  unsafe { nn::NeuralNetwork::from_string(file_string) };

    c.bench_function("nn::compute", |b| b.iter(|| network.compute(black_box(&[0.0, 0.0]))));
}
criterion_group!(benches, benchmark);
criterion_main!(benches);