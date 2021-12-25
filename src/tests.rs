use crate::neural_network::NeuralNetwork;
use crate::topology::mutation_probabilities::MutationProbabilities;
use crate::topology::Topology;
use crate::train::Train;
use crate::{game::Game, section};
use rand::{thread_rng, Rng};
use std::fs;

macro_rules! check_output {
    ($output: expr, $as_str: expr, $index: expr) => {
        if !($output[0] - 1e-8 < $output[1] && $output[0] + 1e-8 > $output[1]) {
            println!("{}", $as_str);
            panic!("{}: {} != {}", $index, $output[0], $output[1])
        }
    };
}

#[test]
pub fn test_import_network() {
    let serialized: String = fs::read_to_string("topology_test.json")
        .expect("Something went wrong reading the topology_test.json");

    let top = Topology::from_string(&serialized);
    let cloned: NeuralNetwork<f64> = unsafe { NeuralNetwork::new(&top) };
    let mut net = NeuralNetwork::from_string(&serialized);
    assert_eq!(net, cloned);

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

#[test]
pub fn test_clone_network() {
    let serialized: String =
        fs::read_to_string("topology_test.json").expect("Something went wrong reading the file");

    let mut net = NeuralNetwork::<f64>::from_string(&serialized);
    let mut cloned = net.clone();

    assert_eq!(net, cloned);

    let input: Vec<f64> = vec![0.5, 0.5, 0.1, -0.2];
    let input_2: Vec<f64> = vec![-0.5, -0.5, -0.1, 0.2];

    let output = net.compute(&input);
    let output_cloned = cloned.compute(&input);

    assert_eq!(output, output_cloned);

    let mut cloned_2 = cloned.clone();

    let output = net.compute(&input_2);
    let output_cloned = cloned.compute(&input_2);
    let output_cloned_2 = cloned_2.compute(&input_2);
    assert_eq!(output, output_cloned);
    assert_eq!(output, output_cloned_2);
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

    fn post_training(&mut self, history: &[Topology<f64>]) {
        for (index, top) in history.iter().enumerate() {
            let top_cp = top.clone();
            assert_eq!(*top, top_cp);

            let as_str = top.to_string();
            let network = unsafe { NeuralNetwork::new(top) };
            let top2 = Topology::from_string(&*as_str);
            let network_from_string: NeuralNetwork<f64> = unsafe { NeuralNetwork::new(&top2) };
            if network != network_from_string {
                println!("{:?}, {:?}", top.layers_sizes, top2.layers_sizes);
                println!("{}", as_str);
                section!();
                println!("{}", top2.to_string());
                panic!("Network != Network from string");
            }
            self.nets = vec![network, network_from_string];
            let output = self.run_generation();
            check_output!(output, as_str, index);
        }
    }
}

#[test]
pub fn test_train() {
    let mut game = TestGame::new();
    let mut runner: Train<TestGame, f64> = Train::new(&mut game);
    runner
        .max_layers(5)
        .max_per_layers(10)
        .iterations(300)
        .max_individuals(50)
        .inputs(5)
        .outputs(5);
    runner.start().expect("Could not start the test");
}

struct MemoryCount {
    values: Vec<usize>,
    nets: Vec<NeuralNetwork<f64>>,
}

impl MemoryCount {
    pub fn new() -> MemoryCount {
        let mut values: Vec<usize> = Vec::new();
        let mut rng = thread_rng();
        for _ in 0..100 {
            values.push(rng.gen_range(0..4));
        }
        MemoryCount {
            values,
            nets: Vec::new(),
        }
    }
}

impl Game<f64> for MemoryCount {
    fn run_generation(&mut self) -> Vec<f64> {
        let values = self.values.clone();
        self.nets
            .iter_mut()
            .map(|net| {
                let mut current_counts = [0usize; 4];
                values
                    .iter()
                    .map(|&v| {
                        current_counts[v] += 1;
                        let mut inputs = [0.; 4];
                        inputs[v] = 1.;
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
        for (index, top) in history.iter().enumerate() {
            let top_cp = top.clone();
            assert_eq!(*top, top_cp);
            let as_str = top.to_string();
            let network = unsafe { NeuralNetwork::new(top) };
            let top2 = Topology::from_string(&*as_str);
            let network_from_string: NeuralNetwork<f64> = unsafe { NeuralNetwork::new(&top2) };
            if network != network_from_string {
                println!("{}", as_str);
                section!();
                println!("{}", top2.to_string());
                panic!("Network != Network from string");
            }
            self.nets = vec![network, network_from_string];
            let output = self.run_generation();
            check_output!(output, as_str, index);
        }
    }
}

#[test]
pub fn test_train_memory() {
    let mut game = MemoryCount::new();
    let proba = MutationProbabilities::new(0.8, 0.2).unwrap();
    let mut runner = Train::new(&mut game);
    runner
        .max_layers(5)
        .max_per_layers(3)
        .mutation_probabilities(proba)
        .iterations(50)
        .max_individuals(50)
        .inputs(5)
        .outputs(5);
    runner.start().unwrap();
}
