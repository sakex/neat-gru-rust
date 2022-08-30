extern crate neat_gru;

use std::fs::File;
use std::io::Write;

use neat_gru::game::Game;
use neat_gru::neural_network::NeuralNetwork;
use neat_gru::train::{Train, HistoricTopology};
struct Player {
    pub net: NeuralNetwork<f64>,
}

impl Player {
    pub fn new(net: NeuralNetwork<f64>) -> Player {
        Player { net }
    }
    /// Runs all the inputs and calculates the outputs
    fn run(&mut self) -> f64 {
        // Get the inputs
        let inputs = Xor::get_inputs();
        // Calculate a score for every input
        let outputs: Vec<f64> = inputs.iter().map(|i| self.net.compute(i)[0]).collect();
        let mut scores: Vec<f64> = vec![];
        for (input, output) in inputs.iter().zip(outputs.iter()) {
            scores.push(compute_score(input, *output));
        }
        // And return the sum of the scores
        scores.iter().sum()
    }
}

struct Simulation {
    players: Vec<Player>,
}

impl Simulation {
    pub fn new() -> Simulation {
        Simulation {
            players: Vec::new(),
        }
    }
}

struct Xor {}

impl Xor {
    fn get_inputs<'a>() -> &'a [[f64; 2]; 4] {
        &[[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    }
}

/// Computes the score with given inputs and one output
fn compute_score(inputs: &[f64], output: f64) -> f64 {
    // https://en.wikipedia.org/wiki/XOR_gate
    // Returns 1.0 for a wrong output and 0.0 for a right output. Should be used as a score
    // We first need to round the numbers to booleans
    let inputs: Vec<bool> = inputs.iter().map(|f| round(*f)).collect();
    let output = round(output);
    if inputs[0] ^ inputs[1] == output {
        return 1.0;
    }
    0.0
}

/// Rounds a float to a bool
fn round(float: f64) -> bool {
    float >= 0.1
}

impl Game<f64> for Simulation {
    /// Loss function
    fn run_generation(&mut self) -> Vec<f64> {
        self.players.iter_mut().map(|p| p.run()).collect()
    }

    /// Reset networks
    fn reset_players(&mut self, nets: Vec<NeuralNetwork<f64>>) {
        self.players.clear();
        self.players = nets.into_iter().map(Player::new).collect();
    }

    /// Called at the end of training
    fn post_training(&mut self, history: &[HistoricTopology<f64>]) {
        // Iter on best topologies and upload the best one
        let best = &history.last().unwrap().topology;
        let mut output = File::create("XOR").expect("Could not create output file");
        write!(output, "{}", best).unwrap();
    }
}

const INPUT_COUNT: usize = 2;
const OUTPUT_COUNT: usize = 1;
const NB_GENERATIONS: usize = 60;
const HIDDEN_LAYERS: usize = 2;
const MAX_INDIVIDUALS: usize = 200;
fn run_sim() {
    let mut sim = Simulation::new();

    let mut runner = Train::new(&mut sim);
    runner
        .inputs(INPUT_COUNT)
        .outputs(OUTPUT_COUNT)
        .iterations(NB_GENERATIONS)
        .max_layers(HIDDEN_LAYERS + 2)
        .max_per_layers(HIDDEN_LAYERS)
        .max_individuals(MAX_INDIVIDUALS)
        .delta_threshold(2.) // Delta parameter from NEAT paper
        .formula(0.8, 0.8, 0.3) // c1, c2 and c3 from NEAT paper
        .access_train_object(Box::new(|train| {
            let species_count = train.species_count();
            println!("Species count: {}", species_count);
        })) // Callback called after `reset_players` that gives you access to the train object during training
        .start()
        .unwrap(); // .start_async().await for async version
}
fn main() {
    run_sim();
}
