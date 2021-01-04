use crate::utils::floats_almost_equal;
use num::traits::Float;
use rand::distributions::{Distribution, Uniform};
use rand::prelude::ThreadRng;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Clone)]
pub struct Bias<T>
where
    T: Float,
{
    pub bias_input: T,
    pub bias_update: T,
    pub bias_reset: T,
}

impl Bias<f64> {
    pub fn cast<T>(&self) -> Bias<T>
    where
        T: Float,
    {
        Bias {
            bias_input: num::cast(self.bias_input).unwrap(),
            bias_update: num::cast(self.bias_update).unwrap(),
            bias_reset: num::cast(self.bias_reset).unwrap(),
        }
    }
}

impl<T> Bias<T>
where
    T: Float,
{
    pub fn new_random(rng: &mut ThreadRng) -> Bias<T> {
        let min: f64 = -1.;
        let max: f64 = 1.;
        let uniform = Uniform::from(min..max);
        Bias {
            bias_input: T::from(uniform.sample(rng)).unwrap(),
            bias_update: T::from(uniform.sample(rng)).unwrap(),
            bias_reset: T::from(uniform.sample(rng)).unwrap(),
        }
    }

    pub fn new_zero() -> Bias<T> {
        Bias {
            bias_input: T::zero(),
            bias_update: T::zero(),
            bias_reset: T::zero(),
        }
    }

    pub fn new(bias_input: T, bias_update: T, bias_reset: T) -> Bias<T> {
        Bias {
            bias_input,
            bias_update,
            bias_reset,
        }
    }
}

impl<T> PartialEq for Bias<T>
where
    T: Float,
{
    fn eq(&self, other: &Self) -> bool {
        floats_almost_equal(self.bias_input, other.bias_input)
            && floats_almost_equal(self.bias_update, other.bias_update)
            && floats_almost_equal(self.bias_reset, other.bias_reset)
    }
}

impl<T> Eq for Bias<T> where T: Float {}
