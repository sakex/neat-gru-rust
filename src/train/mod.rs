#[cfg(feature = "snn")]
use crate::neural_network::spiking::SpikingNeuralNetwork;
use crate::neural_network::NeuralNetwork;

pub mod error;
pub mod evolution_number;
mod species;
mod training;

pub type Train<'a, T, F> = training::Train<'a, T, F, NeuralNetwork<F>>;
#[cfg(feature = "snn")]
pub type TrainSnn<'a, T, F> = training::Train<'a, T, F, SpikingNeuralNetwork<F>>;
