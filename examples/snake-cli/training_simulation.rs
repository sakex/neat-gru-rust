use std::fs::File;
use std::io::Write;



use crate::defs::GUI;
use crate::game::Game;
use neat_gru::neural_network::nn::NeuralNetwork;


pub struct TrainingSimulation {
    networks: Option<Vec<NeuralNetwork<f64>>>,
    generation: usize,
    pub species_count: usize,
}

impl TrainingSimulation {
    pub fn new() -> Self {
        Self {
            networks: None,
            generation: 0,
            species_count: 0,
        }
    }
}

impl neat_gru::game::Game<f64> for TrainingSimulation {
    fn run_generation(&mut self) -> Vec<f64> {
        let generation = self.generation;
        self.generation += 1;
        let species_count = self.species_count;
        let networks = self.networks.take().unwrap();
        let mut game = Game::new(networks);

        game.run_game();
        game.get_scores()
    }

    fn reset_players(&mut self, nets: Vec<NeuralNetwork<f64>>) {
        self.networks = Some(nets);
    }

    fn post_training(&mut self, history: &[neat_gru::topology::topology::Topology<f64>]) {
        let history: Vec<String> = history.iter().map(|t| t.to_string()).collect();
        let mut output = File::create("snakes.json").expect("Could not create output file");
        write!(output, "{}", history.join(" ")).unwrap();
    }
}
