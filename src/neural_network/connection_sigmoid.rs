use crate::neural_network::neuron::Neuron;
use crate::utils::floats_almost_equal;
use num::Float;
use std::ops::AddAssign;

pub struct ConnectionSigmoid<T>
where
    T: Float + std::cmp::PartialEq + std::cmp::PartialEq + AddAssign,
{
    weight: T,
    output: *mut Neuron<T>,
}

impl<T> ConnectionSigmoid<T>
where
    T: Float + std::cmp::PartialEq + std::cmp::PartialEq + AddAssign,
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

impl<T> PartialEq for ConnectionSigmoid<T>
where
    T: Float + std::cmp::PartialEq + AddAssign,
{
    fn eq(&self, other: &Self) -> bool {
        floats_almost_equal(self.weight, other.weight)
    }
}

impl<T> Eq for ConnectionSigmoid<T> where T: Float + std::cmp::PartialEq + AddAssign {}
