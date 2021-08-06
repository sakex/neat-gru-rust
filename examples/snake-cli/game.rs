use neat_gru::neural_network::nn::NeuralNetwork;

use crate::{
    apple::Apple,
    snake::Snake,
    utils::{distance_to_apple_x, distance_to_apple_y, distance_to_wall_x, distance_to_wall_y},
};

#[derive(Debug)]
pub struct Game {
    snakes: Vec<Snake>,
    apple: Apple,
    scores: Vec<f64>,
    tick: usize,
}

impl Game {
    pub fn get_scores(&self) -> Vec<f64> {
        self.scores.clone()
    }

    pub fn new(neural_networks: Vec<NeuralNetwork<f64>>) -> Game {
        let snakes: Vec<Snake> = neural_networks
            .iter()
            .map(|nn| Snake::new(nn.clone()))
            .collect();
        let mut scores = vec![];
        neural_networks.iter().for_each(|_| scores.push(0.0));
        Game {
            snakes,
            apple: Apple::generate_apple(),
            scores,
            tick: 0,
        }
    }

    /// Runs the game and if it finishes returns the game at the end
    pub fn run_game(&mut self) {
        while !self.game_over() {
            self.update();
        }
    }

    /// Make the snakes make their decision
    pub fn make_decision(&mut self) {
        let mut inputs: [f64; 5] = [0., 0., 0., 0., 0.];
        let cloned_apple = self.apple.clone();
        // Let each snake make a decision
        self.snakes.iter_mut().for_each(|s| {
            // First inputs are the distance to the apple from -1 to 1
            inputs[0] = distance_to_apple_x(s, cloned_apple);
            inputs[1] = distance_to_apple_y(s, cloned_apple);
            inputs[2] = distance_to_wall_x(s);
            inputs[3] = distance_to_wall_y(s);
            s.make_decision(&inputs)
        });
    }

    /// Updates the game. Should be called every tick
    pub fn update(&mut self) {
        let apple_coordinate = self.apple.get_coordinate();
        self.remove_if_dead();
        // And then update it
        let replace_apple = self
            .snakes
            .iter_mut()
            .map(|s| -> bool { s.update(apple_coordinate) })
            .any(|b| b);
        self.tick += 1;
        if replace_apple {
            self.apple = Apple::generate_apple();
        }
        // Let each snake make a decision
        self.make_decision();
    }

    fn game_over(&self) -> bool {
        self.snakes.is_empty()
    }

    fn remove_if_dead(&mut self) {
        for idx in 0..self.snakes.len() {
            if self.snakes[idx].is_colliding() {
                self.scores[idx] = self.snakes[idx].size() as f64 + (self.tick as f64 * 0.01);
            }
        }
        self.snakes.retain(|s| !s.is_colliding());
    }
}
