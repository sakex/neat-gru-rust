use num::traits::Float;
use rand::prelude::ThreadRng;
use rand::distributions::{Uniform, Distribution};

#[derive(Clone)]
pub struct Bias<T>
    where T: Float {
    bias_input: T,
    bias_update: T,
    bias_reset: T,
}

impl<T> Bias<T> where T: Float {
    pub fn new_random(rng: &mut ThreadRng) -> Bias<T> {
        let min: f64 = -1.0;
        let max: f64 = 1.0;
        let uniform = Uniform::from(min..max);
        Bias {
            bias_input: T::from(uniform.sample(rng)).unwrap(),
            bias_update:  T::from(uniform.sample(rng)).unwrap(),
            bias_reset:  T::from(uniform.sample(rng)).unwrap(),
        }
    }
}