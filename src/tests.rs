use crate::game::GameAsync;
use crate::neural_network::NeuralNetwork;
use crate::topology::mutation_probabilities::MutationProbabilities;
use crate::topology::Topology;
use crate::train::evolution_number::EvNumber;
use crate::train::species::Species;
use crate::train::{HistoricTopologyLazy, Train};
use crate::{game::Game, section};
use async_trait::async_trait;
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
    // Because of GRU gates, giving the same input twice won't yield the same output
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
                let out = network.compute(&inputs);
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

    fn post_training(&mut self, history: Vec<HistoricTopologyLazy<f64>>) {
        for (index, top) in history.into_iter().enumerate() {
            let top = top.into_historic().unwrap().topology;
            let top_cp = top.clone();
            assert_eq!(top, top_cp);

            let as_str = top.to_string();
            let network = unsafe { NeuralNetwork::new(&top) };
            let top2 = Topology::from_string(&as_str);
            let network_from_string: NeuralNetwork<f64> = unsafe { NeuralNetwork::new(&top2) };
            if network != network_from_string {
                println!("{:?}, {:?}", top.layers_sizes, top2.layers_sizes);
                println!("{}", as_str);
                section!();
                println!("{}", top2);
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

    fn post_training(&mut self, history: Vec<HistoricTopologyLazy<f64>>) {
        for (index, top) in history.into_iter().enumerate() {
            let top = &top.into_historic().unwrap().topology;
            let top_cp = top.clone();
            assert_eq!(*top, top_cp);
            let as_str = top.to_string();
            let network = unsafe { NeuralNetwork::new(top) };
            let top2 = Topology::from_string(&as_str);
            let network_from_string: NeuralNetwork<f64> = unsafe { NeuralNetwork::new(&top2) };
            if network != network_from_string {
                println!("{}", as_str);
                section!();
                println!("{}", top2);
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

fn single_connection_json(output: (u8, u8), connection_type: i32) -> String {
    format!(
        r#"{{"biases":[
            {{"neuron":[0,0],"bias":{{"bias_input":0.0,"bias_update":0.0,"bias_reset":0.0}}}},
            {{"neuron":[1,0],"bias":{{"bias_input":0.0,"bias_update":0.0,"bias_reset":0.0}}}}],
        "genes":[{{"connection_type":{},"disabled":false,"input":[0,0],"input_weight":1.0,
            "memory_weight":0.0,"output":[{},{}],"reset_input_weight":0.0,
            "reset_memory_weight":0.0,"update_input_weight":0.0,"update_memory_weight":0.0,
            "ev_number":1.0}}]}}"#,
        connection_type, output.0, output.1
    )
}

#[test]
pub fn test_input_is_not_scaled() {
    // One input linearly connected to one output with a weight of 1
    let mut net = NeuralNetwork::<f64>::from_string(&single_connection_json((1, 0), 0));
    let x = 0.1f64;
    // The output neuron's update gate is at 0.5 with a zero bias
    let expected = (0.5 * x.tanh()).tanh();
    let output = net.compute(&[x]);
    assert!(
        (output[0] - expected).abs() < 1e-3,
        "{} != {}",
        output[0],
        expected
    );
    // Inputs in excess of the first layer are ignored
    net.reset_state();
    assert_eq!(net.compute(&[x, 12., 13., 14.]), output);
}

#[test]
pub fn test_invalid_topologies_are_rejected() {
    assert!(Topology::<f64>::is_valid_topology_json(
        &single_connection_json((1, 0), 0)
    ));
    // Output neuron doesn't exist
    assert!(!Topology::<f64>::is_valid_topology_json(
        &single_connection_json((1, 7), 0)
    ));
    assert!(!Topology::<f64>::is_valid_topology_json(
        &single_connection_json((9, 0), 0)
    ));
    // Connection going backward
    assert!(!Topology::<f64>::is_valid_topology_json(
        &single_connection_json((0, 0), 0)
    ));
    // Unknown connection type
    assert!(!Topology::<f64>::is_valid_topology_json(
        &single_connection_json((1, 0), 5)
    ));
    assert!(!Topology::<f64>::is_valid_topology_json("{}"));
}

#[test]
pub fn test_mutation_probabilities_bounds() {
    assert!(MutationProbabilities::new(1.0, 0.0).is_ok());
    assert!(MutationProbabilities::new(1.1, 0.0).is_err());
    assert!(MutationProbabilities::new(0.5, -0.1).is_err());
}

/// Records what the trainer does to check that the limits are respected
struct LimitsGame {
    nets: Vec<NeuralNetwork<f64>>,
    generations: usize,
    max_population: usize,
    max_layers: usize,
    max_per_layers: usize,
    outputs: usize,
    negative_scores: bool,
}

impl LimitsGame {
    pub fn new(negative_scores: bool) -> LimitsGame {
        LimitsGame {
            nets: Vec::new(),
            generations: 0,
            max_population: 0,
            max_layers: 0,
            max_per_layers: 0,
            outputs: 0,
            negative_scores,
        }
    }
}

impl Game<f64> for LimitsGame {
    fn run_generation(&mut self) -> Vec<f64> {
        self.generations += 1;
        let generation = self.generations as f64;
        let mut rng = thread_rng();
        let negative_scores = self.negative_scores;
        let mut outputs = 0;
        let results = self
            .nets
            .iter_mut()
            .map(|net| {
                let out = net.compute(&[0.1, 0.2, 0.3, 0.4, 0.5]);
                outputs = out.len();
                if negative_scores {
                    // Always improving, but always negative
                    -1000. + generation + rng.gen_range(0.0..0.5)
                } else {
                    let diff: f64 = out.iter().map(|v| (v - 0.3).abs()).sum();
                    1. / (0.01 + diff) + rng.gen_range(0.0..0.1)
                }
            })
            .collect();
        self.outputs = outputs;
        results
    }

    fn reset_players(&mut self, nets: Vec<NeuralNetwork<f64>>) {
        self.max_population = self.max_population.max(nets.len());
        self.nets = nets;
    }

    fn post_training(&mut self, history: Vec<HistoricTopologyLazy<f64>>) {
        for top in history {
            let top = top.into_historic().unwrap();
            self.max_layers = self.max_layers.max(top.layers_sizes.len());
            // The input and output layers are set by the user
            let hidden = &top.layers_sizes[1..top.layers_sizes.len() - 1];
            let biggest = hidden.iter().max().copied().unwrap_or(0) as usize;
            self.max_per_layers = self.max_per_layers.max(biggest);
        }
    }
}

#[async_trait]
impl GameAsync<f64> for LimitsGame {
    async fn run_generation_async(&mut self) -> Vec<f64> {
        self.run_generation()
    }
}

#[test]
pub fn test_limits_are_respected() {
    for delta_threshold in [0.5, 3.] {
        let mut game = LimitsGame::new(false);
        {
            let mut runner = Train::new(&mut game);
            runner
                .max_layers(4)
                .max_per_layers(3)
                .iterations(150)
                .max_individuals(60)
                .delta_threshold(delta_threshold)
                .inputs(5)
                .outputs(2);
            runner.start().unwrap();
        }
        assert_eq!(game.generations, 150);
        assert!(
            game.max_population <= 60,
            "{} networks",
            game.max_population
        );
        assert!(game.max_layers <= 4, "{} layers", game.max_layers);
        assert!(
            game.max_per_layers <= 3,
            "{} per layer",
            game.max_per_layers
        );
    }
}

#[test]
pub fn test_negative_scores_dont_stop_training() {
    let mut game = LimitsGame::new(true);
    {
        let mut runner = Train::new(&mut game);
        runner
            .iterations(60)
            .max_individuals(30)
            // Everything in a single species
            .delta_threshold(1000.)
            .inputs(5)
            .outputs(2);
        runner.start().unwrap();
    }
    assert_eq!(game.generations, 60);
}

#[test]
pub fn test_start_async_uses_outputs() {
    use std::future::Future;
    use std::pin::Pin;
    use std::task::{Context, Poll, Waker};

    let mut game = LimitsGame::new(false);
    {
        let mut runner = Train::new(&mut game);
        runner
            .iterations(3)
            .max_individuals(10)
            .inputs(5)
            .outputs(2);
        let mut context = Context::from_waker(Waker::noop());
        let mut future: Pin<Box<dyn Future<Output = _> + '_>> = Box::pin(runner.start_async());
        match future.as_mut().poll(&mut context) {
            Poll::Ready(result) => result.unwrap(),
            Poll::Pending => panic!("The training never awaits anything pending"),
        };
    }
    assert_eq!(game.outputs, 2);
}

#[test]
pub fn test_species_selection_doesnt_deadlock() {
    use std::sync::mpsc::channel;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    let (sender, receiver) = channel();
    std::thread::spawn(move || {
        let ev_number = Arc::new(EvNumber::new());
        let mut rng = thread_rng();
        let new_topology = |result: f64| {
            let mut topology = Topology::<f64>::new_uniform(2, 1, 4, 4, &ev_number);
            topology.set_last_result(result);
            Arc::new(Mutex::new(topology))
        };
        let mut species = Species::new(new_topology(0.));
        // The same topology can be in a species twice, the sort must not lock it twice
        for _ in 0..200 {
            let topology = new_topology(rng.gen_range(0.0..1.));
            species.push(topology.clone());
            species.push(topology);
        }
        species.max_topologies = 100;
        let proba = MutationProbabilities::new(0.8, 0.2).unwrap();
        species.natural_selection(ev_number.clone(), proba, true);
        sender.send(species.topologies.len()).unwrap();
    });
    let size = receiver
        .recv_timeout(Duration::from_secs(30))
        .expect("Natural selection deadlocked");
    assert_eq!(size, 100);
}
