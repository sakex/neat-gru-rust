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

impl<T> Bias<T>
where
    T: Float,
{
    pub fn new_random(rng: &mut ThreadRng) -> Bias<T> {
        let min: f64 = -1.0;
        let max: f64 = 1.0;
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
