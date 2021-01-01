use crate::game::Game;
use crate::neural_network::nn::NeuralNetwork;
use crate::topology::mutation_probabilities::MutationProbabilities;
use crate::topology::topology::Topology;
use crate::train::train::Train;
use rand::{thread_rng, Rng};
use std::fs;

#[test]
pub fn test_import_network() {
    let serialized: String =
        fs::read_to_string("topology_test.json").expect("Something went wrong reading the file");

    let mut net = NeuralNetwork::from_string(&serialized);
    let input_1: Vec<f64> = vec![0.5, 0.5, 0.1, -0.2];
    let input_2: Vec<f64> = vec![-0.5, -0.5, -0.1, 0.2];

    let output_1 = net.compute(&input_1);
    let output_2 = net.compute(&input_2);
    let output_3 = net.compute(&input_1);

    // 1 and 2 should by definition be different
    assert_ne!(output_1, output_2);
    assert_ne!(output_1, output_3);
    //Because of GRU gates, giving the same input twice won't yield the same output
    assert_ne!(output_2, output_3);

    // Reset
    net.reset_state();
    let output_4 = net.compute(&input_1);

    // After resetting, giving the same input sequence should yield the same results
    assert_eq!(output_1, output_4);
}

struct TestGame {
    nets: Vec<NeuralNetwork<f64>>,
}

impl TestGame {
    pub fn new() -> TestGame {
        TestGame { nets: Vec::new() }
    }
}

impl Game<f64> for TestGame {
    fn run_generation(&mut self) -> Vec<f64> {
        self.nets
            .iter_mut()
            .map(|network| {
                let inputs = vec![0.1, 0.2, 0.3, 0.4, 0.5];
                let out = network.compute(&*inputs);
                let mut diff = 0f64;
                inputs.iter().zip(out.iter()).for_each(|(a, b)| {
                    diff -= (a - b).abs();
                });
                diff
            })
            .collect()
    }

    fn reset_players(&mut self, nets: Vec<NeuralNetwork<f64>>) {
        self.nets = nets;
    }

    fn post_training(&mut self, _history: &[Topology<f64>]) {}
}

#[test]
pub fn test_train() {
    let mut game = TestGame::new();
    let mut runner = Train::new(&mut game);
    runner
        .iterations(100)
        .max_individuals(50)
        .max_species(5)
        .inputs(5)
        .outputs(5);
    runner.start();
}

struct MemoryCount {
    nets: Vec<NeuralNetwork<f64>>,
}

impl MemoryCount {
    pub fn new() -> MemoryCount {
        MemoryCount { nets: Vec::new() }
    }
}

impl Game<f64> for MemoryCount {
    fn run_generation(&mut self) -> Vec<f64> {
        let mut values: Vec<usize> = Vec::new();
        let mut rng = thread_rng();
        for _ in 0..100 {
            values.push(rng.gen_range(0..4));
        }
        self.nets
            .iter_mut()
            .map(|net| {
                let mut current_counts = [0usize; 4];
                values
                    .iter()
                    .map(|&v| {
                        current_counts[v] += 1;
                        let mut inputs = [0.0; 4];
                        inputs[v] = 1.0;
                        let outputs = net.compute(&inputs);
                        let (index_output, _) = outputs
                            .iter()
                            .enumerate()
                            .fold((0usize, &outputs[0]), |a, b| if a.1 > b.1 { a } else { b });
                        let (real_index, _) = current_counts.iter().enumerate().fold(
                            (0, &current_counts[0]),
                            |a, b| if a.1 > b.1 { a } else { b },
                        );
                        (index_output == real_index) as usize as f64
                    })
                    .sum()
            })
            .collect()
    }

    fn reset_players(&mut self, nets: Vec<NeuralNetwork<f64>>) {
        self.nets = nets;
    }

    fn post_training(&mut self, history: &[Topology<f64>]) {
        println!("TRAINING DONE");
    }
}

#[test]
pub fn test_train_memory() {
    let mut game = MemoryCount::new();
    let proba = MutationProbabilities::new(0.4, 0.2).unwrap();
    let mut runner = Train::new(&mut game);
    runner
        .mutation_probabilities(proba)
        .iterations(100)
        .max_individuals(50)
        .max_species(5)
        .inputs(5)
        .outputs(5);
    runner.start();
}
