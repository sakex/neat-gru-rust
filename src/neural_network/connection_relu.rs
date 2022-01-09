use crate::neural_network::neuron::Neuron;
use crate::utils::floats_almost_equal;
use num::Float;
use std::ops::AddAssign;

#[derive(Debug)]
pub struct ConnectionRelu<T>
where
    T: Float + std::cmp::PartialEq + std::cmp::PartialEq + AddAssign + Send,
{
    weight: T,
    output: *mut Neuron<T>,
}

unsafe impl<T> Send for ConnectionRelu<T> where
    T: Float + std::cmp::PartialEq + std::cmp::PartialEq + AddAssign + Send
{
}

impl<T> ConnectionRelu<T>
where
    T: Float + std::cmp::PartialEq + std::cmp::PartialEq + AddAssign + Send,
{
    pub(crate) fn new(weight: T, output: *mut Neuron<T>) -> ConnectionRelu<T> {
        ConnectionRelu { weight, output }
    }

    pub(crate) fn clone_with_old_pointer(&self) -> ConnectionRelu<T> {
        ConnectionRelu {
            weight: self.weight,
            output: self.output,
        }
    }

    pub(crate) fn increment_pointer(&mut self, diff: isize) {
        self.output = (self.output as isize + diff) as *mut Neuron<T>;
    }

    #[inline]
    pub(crate) fn activate(&mut self, value: T) {
        unsafe {
            (*self.output).increment_value((value * self.weight).max(T::zero()));
        };
    }
}

impl<T> PartialEq for ConnectionRelu<T>
where
    T: Float + std::cmp::PartialEq + AddAssign + Send,
{
    fn eq(&self, other: &Self) -> bool {
        floats_almost_equal(self.weight, other.weight)
    }
}

impl<T> Eq for ConnectionRelu<T> where T: Float + std::cmp::PartialEq + AddAssign + Send {}
