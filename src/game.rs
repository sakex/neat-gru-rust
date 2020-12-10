use crate::neural_network::nn::NeuralNetwork;
use crate::topology::topology::Topology;
use num::Float;

/// Trait to implement in order to use Train
pub trait Game<T>
    where T: Float {
    /// Run a game round
    fn run_generation(&mut self) -> Vec<T>;

    /// Resets the neural networks
    ///
    /// # Arguments
    ///
    /// `nets` - A vector containing the last generation of neural networks
    fn reset_players(&mut self, nets: &[NeuralNetwork<T>]);

    /// Function to be run at the end of the training
    ///
    /// # Arguments
    ///
    /// `net` - The best historical network
    fn post_training(&mut self, history: &[Topology<T>]);
}