use num::traits::Float;
use rand::prelude::ThreadRng;
use rand::distributions::{Uniform, Distribution};

#[derive(Clone)]
pub struct Bias<T>
    where T: Float {
    pub bias_input: T,
    pub bias_update: T,
    pub bias_reset: T,
}

impl<T> Bias<T> where T: Float {
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

    pub fn new() -> Bias<T> {
        Bias {
            bias_input: T::from(0).unwrap(),
            bias_update: T::from(0).unwrap(),
            bias_reset: T::from(0).unwrap(),
        }
    }
}