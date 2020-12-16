use crate::neural_network::neuron::Neuron;
use num::Float;

pub struct ConnectionSigmoid<T>
where
    T: Float,
{
    weight: T,
    output: *mut Neuron<T>,
}

impl<T> ConnectionSigmoid<T>
where
    T: Float,
{
    pub fn new(weight: T, output: *mut Neuron<T>) -> ConnectionSigmoid<T> {
        ConnectionSigmoid { weight, output }
    }

    #[inline]
    pub fn activate(&mut self, value: T) {
        unsafe {
            (*self.output).increment_value(value * self.weight);
        };
    }
}
