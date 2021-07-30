extern crate neat_gru;

use neat_gru::game::Game;
use neat_gru::neural_network::nn::NeuralNetwork;
use neat_gru::topology::topology::Topology;
use neat_gru::train::train::Train;
struct Player {
    pub net: NeuralNetwork<f64>,
}

impl Player {
    pub fn new(net: NeuralNetwork<f64>) -> Player {
        Player { net }
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

impl Game<f64> for Simulation {
    /// Loss function
    fn run_generation(&mut self) -> Vec<f64> {
        let inputs = get_inputs();
        self.players
            .iter()
            .map(|p| {
                let output = p.net.compute(inputs);
                let scores = compute_score(output, target);
                scores
            })
            .collect()
    }

    /// Reset networks
    fn reset_players(&mut self, nets: Vec<NeuralNetwork<f64>>) {
        self.players.clear();
        self.players = nets.into_iter().map(Player::new).collect();
    }

    /// Called at the end of training
    fn post_training(&mut self, history: &[Topology<f64>]) {
        // Iter on best topologies and upload the best one
    }
}

const INPUT_COUNT: usize = 2;
const OUTPUT_COUNT: usize = 1;
const NB_GENERATIONS: usize = 5;
const ITERATIONS: usize = 1000;
const HIDDEN_LAYERS: usize = 2;
const MAX_INDIVIDUALS: usize = 100;
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
        .start(); // .start_async().await for async version
}
fn main() {
    run_sim();
}
