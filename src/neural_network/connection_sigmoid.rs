use num::Float;
use crate::neural_network::neuron::Neuron;

pub struct ConnectionSigmoid<T>
    where T: Float {
    weight: T,
    output: *const Neuron<T>
}

impl<T> ConnectionSigmoid<T> where T: Float {
    pub fn new(weight: T, output: *const Neuron<T>) -> ConnectionSigmoid<T> {
        ConnectionSigmoid {
            weight,
            output
        }
    }
}