use criterion::{black_box, criterion_group, criterion_main, Criterion};
extern crate neat_gru;
use neat_gru::neural_network::nn::NeuralNetwork;
use neat_gru::topology::topology::Topology;
use std::fs::File;
use std::io::Read;

fn benchmark(c: &mut Criterion) {
    let mut file = File::open("snakes_benchmark.json").expect("Can't open snakes_benchmark.json");
    let file_string = &mut "".to_string();
    file.read_to_string(file_string).unwrap();
    let topology = Topology::from_string(file_string);
    let mut network = unsafe { NeuralNetwork::new(&topology) };
    c.bench_function("nn::compute", |b| {
        b.iter(|| network.compute(black_box(&[0.0, 0.0])))
    });
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
