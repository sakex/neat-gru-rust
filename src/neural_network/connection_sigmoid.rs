use crate::neural_network::neuron::Neuron;
use crate::utils::floats_almost_equal;
use num::Float;
use std::ops::AddAssign;
use rand_distr::{Distribution, Uniform};
use rand::{thread_rng};

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

    #[inline]
    pub fn random_weights_sigmoid(&mut self) {
        println!("sigmoid");
        let mut rng = thread_rng();
        let percent_change = 0.01;

        if self.weight.to_f64().unwrap() != 0. {
            let min: f64 = (1. - percent_change * self.weight.to_f64().unwrap().signum()) * self.weight.to_f64().unwrap();
            let max: f64 = (1. + percent_change * self.weight.to_f64().unwrap().signum()) * self.weight.to_f64().unwrap();
            let uniform = Uniform::from(min..max);
            self.weight = T::from(uniform.sample(&mut rng)).unwrap();
        }
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
