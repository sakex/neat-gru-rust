use crate::topology::topology::Topology;
use num::Float;

pub struct NeuralNetwork<T>
    where T: Float {
    x: T
}

impl<T> NeuralNetwork<T>
    where T: Float {
    pub fn new(_topology: &Topology<T>) -> NeuralNetwork<T> {
        NeuralNetwork {
            x: T::from(0).unwrap()
        }
    }
}