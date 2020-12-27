use num::traits::Float;
use numeric_literals::replace_numeric_literals;
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

    #[replace_numeric_literals(T::from(literal).unwrap())]
    pub fn new_uniform() -> Bias<T> {
        Bias {
            bias_input: 0,
            bias_update: 0,
            bias_reset: 0,
        }
    }

    pub fn new_zero() -> Bias<T> {
        Bias {
            bias_input: T::from(0).unwrap(),
            bias_update: T::from(0).unwrap(),
            bias_reset: T::from(0).unwrap(),
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
