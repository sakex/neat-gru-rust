use crate::game::Game;
use crate::neural_network::nn::NeuralNetwork;
use crate::topology::topology::Topology;
use crate::train::train::Train;
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
