mod connection_gru;
mod connection_relu;
mod connection_sigmoid;
mod functions;
mod neuron;
pub mod nn;
pub mod nn_trait;

pub use nn::*;
#[cfg(feature = "snn")]
pub mod spiking;
