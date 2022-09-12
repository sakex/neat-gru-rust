use crate::neural_network::NeuralNetwork;
use crate::train::HistoricTopology;
use async_trait::async_trait;
use num::Float;
use serde::{de, Serialize};
use std::fmt::Display;

/// Trait to implement in order to use Train
pub trait Game<T>
where
    T: Float + std::ops::AddAssign + Display + Serialize + de::DeserializeOwned + Send,
{
    /// Run a game round
    fn run_generation(&mut self) -> Vec<T>;

    /// Resets the neural networks
    ///
    /// # Arguments
    ///
    /// `nets` - A vector containing the last generation of neural networks
    fn reset_players(&mut self, nets: Vec<NeuralNetwork<T>>);

    /// Function to be run at the end of the training
    ///
    /// # Arguments
    ///
    /// `net` - The best historical network
    fn post_training(&mut self, history: &[HistoricTopology<T>]);
}

#[async_trait]
pub trait GameAsync<T>: Game<T>
where
    T: Float + std::ops::AddAssign + Display + Serialize + de::DeserializeOwned + Send,
{
    async fn run_generation_async(&mut self) -> Vec<T>;
}
