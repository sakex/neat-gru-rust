use neat_gru::neural_network::NeuralNetwork;

use crate::utils::distance_to_apple;
use crate::{
    apple::Apple,
    snake::Snake,
    utils::{distance_to_apple_x, distance_to_apple_y, distance_to_wall_x, distance_to_wall_y},
};
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelRefMutIterator;

#[derive(Debug)]
pub struct Game {
    snakes: Vec<Snake>,
    apple: Apple,
    scores: Vec<f64>,
    tick: usize,
    ticks_since_eaten: usize,
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
            ticks_since_eaten: 0,
        }
    }

    /// Runs the game until every snake is dead
    pub fn run_game(&mut self) {
        while !self.game_over() {
            self.tick();
        }
    }

    /// Make the snakes make their decision
    pub fn make_decision(&mut self) {
        let cloned_apple = self.apple;
        // Let each snake make a decision
        self.snakes.par_iter_mut().for_each(|s| {
            s.make_decision(&[
                // First inputs are the distance to the apple from -1 to 1
                distance_to_apple_x(s, cloned_apple),
                distance_to_apple_y(s, cloned_apple),
                // The other inputs are the position in the game from -1 to 1
                distance_to_wall_x(s),
                distance_to_wall_y(s),
            ])
        });
    }

    /// Updates the game. Should be called every tick
    pub fn tick(&mut self) {
        let apple_coordinate = self.apple.get_coordinate();

        // Let each snake make a decision
        self.make_decision();

        // Remove the dead snake so that they can't produce invalid coordinates
        self.remove_if_dead();

        // Update every snake
        let replace_apple = self
            .snakes
            .par_iter_mut()
            .map(|s| -> bool { s.update(apple_coordinate) })
            .any(|b| b);
        // Increase all ticks and increase ticks_since_eaten if no apple was replaced
        self.tick += 1;
        if replace_apple {
            self.apple = Apple::generate_apple();
            self.ticks_since_eaten = 0;
        } else {
            self.ticks_since_eaten += 1;
        }
    }

    fn game_over(&self) -> bool {
        // When the snakes don't eat obviously starve
        self.snakes.is_empty() || self.ticks_since_eaten >= 150
    }

    fn remove_if_dead(&mut self) {
        for idx in 0..self.snakes.len() {
            if self.snakes[idx].is_colliding() {
                self.scores[idx] = self.snakes[idx].size() as f64
                    - 0.1 * (distance_to_apple(&self.snakes[idx], self.apple));
            }
        }

        // Snakes that are idiots get removed
        self.snakes.retain(|s| !s.is_colliding());
    }
}
