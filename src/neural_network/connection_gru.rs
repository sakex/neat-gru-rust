use crate::neural_network::functions::fast_tanh;
use crate::neural_network::neuron::Neuron;
use crate::utils::floats_almost_equal;
use num::Float;
use rand_distr::{Distribution, Uniform};
use rand::{thread_rng};

pub struct ConnectionGru<T>
where
    T: Float + std::ops::AddAssign + std::cmp::PartialEq,
{
    memory: T,
    prev_input: T,
    input_weight: T,
    memory_weight: T,
    reset_input_weight: T,
    update_input_weight: T,
    reset_memory_weight: T,
    update_memory_weight: T,
    output: *mut Neuron<T>,
}

impl<T> ConnectionGru<T>
where
    T: Float + std::ops::AddAssign + std::cmp::PartialEq,
{
    pub fn new(
        input_weight: T,
        memory_weight: T,
        reset_input_weight: T,
        update_input_weight: T,
        reset_memory_weight: T,
        update_memory_weight: T,
        output: *mut Neuron<T>,
    ) -> ConnectionGru<T> {
        ConnectionGru {
            memory: T::zero(),
            prev_input: T::zero(),
            input_weight,
            memory_weight,
            reset_input_weight,
            update_input_weight,
            reset_memory_weight,
            update_memory_weight,
            output,
        }
    }

    #[inline]
    pub fn activate(&mut self, value: T) {
        let prev_reset = unsafe { (*self.output).prev_reset };
        self.memory = fast_tanh(
            self.prev_input * self.input_weight + self.memory_weight * prev_reset * self.memory,
        );
        self.prev_input = value;

        let update_mem = self.memory * self.memory_weight;
        unsafe {
            (*self.output).increment_state(
                update_mem,
                value * self.input_weight,
                value * self.reset_input_weight + self.memory * self.reset_memory_weight,
                value * self.update_input_weight + self.memory * self.update_memory_weight,
            );
        }
    }

    #[inline]
    pub fn reset_state(&mut self) {
        self.memory = T::zero();
        self.prev_input = T::zero();
    }

    #[inline]
    pub fn random_weights(&mut self) {
        println!("gru");
        let mut rng = thread_rng();
        let percent_change = 0.01;
        /*
        let min: f64 = -1.;
        let max: f64 = 1.;
        println!("{} <= {} for input_weight", min, max);
        let uniform = Uniform::from(min..max);
        self.input_weight = T::from(uniform.sample(&mut rng)).unwrap();

        println!("{} <= {} for memory_weight", min, max);
        let uniform = Uniform::from(min..max);
        self.memory_weight = T::from(uniform.sample(&mut rng)).unwrap();

        println!("{} <= {} for reset_input_weight", min, max);
        let uniform = Uniform::from(min..max);
        self.reset_input_weight = T::from(uniform.sample(&mut rng)).unwrap();

        println!("{} <= {} for update_input_weight", min, max);
        let uniform = Uniform::from(min..max);
        self.update_input_weight = T::from(uniform.sample(&mut rng)).unwrap();

        println!("{} <= {} for reset_memory_weight", min, max);
        let uniform = Uniform::from(min..max);
        self.reset_memory_weight = T::from(uniform.sample(&mut rng)).unwrap();

        println!("{} <= {} for update_memory_weight", min, max);
        let uniform = Uniform::from(min..max);
        self.update_memory_weight = T::from(uniform.sample(&mut rng)).unwrap();
         */

        println!("{}", self.input_weight.to_f64().unwrap());
        if self.input_weight.to_f64().unwrap() != 0. {
            let min: f64 = (1. - percent_change*self.input_weight.to_f64().unwrap().signum()) * self.input_weight.to_f64().unwrap();
            let max: f64 = (1. + percent_change*self.input_weight.to_f64().unwrap().signum()) * self.input_weight.to_f64().unwrap();
            println!("{} <= {} for input_weight", min, max);
            let uniform = Uniform::from(min..max);
            self.input_weight = T::from(uniform.sample(&mut rng)).unwrap();
        }

        println!("{}", self.memory_weight.to_f64().unwrap());
        if self.memory_weight.to_f64().unwrap() != 0. {
            let min: f64 = (1. - percent_change*self.memory_weight.to_f64().unwrap().signum()) * self.memory_weight.to_f64().unwrap();
            let max: f64 = (1. + percent_change*self.memory_weight.to_f64().unwrap().signum()) * self.memory_weight.to_f64().unwrap();
            println!("{} <= {} for memory_weight", min, max);
            let uniform = Uniform::from(min..max);
            self.memory_weight = T::from(uniform.sample(&mut rng)).unwrap();
        }

        println!("{}", self.reset_input_weight.to_f64().unwrap());
        if self.reset_input_weight.to_f64().unwrap() != 0. {
            let min: f64 = (1. - percent_change*self.reset_input_weight.to_f64().unwrap().signum()) * self.reset_input_weight.to_f64().unwrap();
            let max: f64 = (1. + percent_change*self.reset_input_weight.to_f64().unwrap().signum()) * self.reset_input_weight.to_f64().unwrap();
            println!("{} <= {} for reset_input_weight", min, max);
            let uniform = Uniform::from(min..max);
            self.reset_input_weight = T::from(uniform.sample(&mut rng)).unwrap();
        }

        println!("{}", self.update_input_weight.to_f64().unwrap());
        if self.update_input_weight.to_f64().unwrap() != 0. {
            let min: f64 = (1. - percent_change*self.update_input_weight.to_f64().unwrap().signum()) * self.update_input_weight.to_f64().unwrap();
            let max: f64 = (1. + percent_change*self.update_input_weight.to_f64().unwrap().signum()) * self.update_input_weight.to_f64().unwrap();
            println!("{} <= {} for update_input_weight", min, max);
            let uniform = Uniform::from(min..max);
            self.update_input_weight = T::from(uniform.sample(&mut rng)).unwrap();
        }

        println!("{}", self.reset_memory_weight.to_f64().unwrap());
        if self.reset_memory_weight.to_f64().unwrap() != 0. {
            let min: f64 = (1. - percent_change*self.reset_memory_weight.to_f64().unwrap().signum()) * self.reset_memory_weight.to_f64().unwrap();
            let max: f64 = (1. + percent_change*self.reset_memory_weight.to_f64().unwrap().signum()) * self.reset_memory_weight.to_f64().unwrap();
            println!("{} <= {} for reset_memory_weight", min, max);
            let uniform = Uniform::from(min..max);
            self.reset_memory_weight = T::from(uniform.sample(&mut rng)).unwrap();
        }

        println!("{}", self.update_memory_weight.to_f64().unwrap());
        if self.update_memory_weight.to_f64().unwrap() != 0. {
            let min: f64 = (1. - percent_change*self.update_memory_weight.to_f64().unwrap().signum()) * self.update_memory_weight.to_f64().unwrap();
            let max: f64 = (1. + percent_change*self.update_memory_weight.to_f64().unwrap().signum()) * self.update_memory_weight.to_f64().unwrap();
            println!("{} <= {} for update_memory_weight", min, max);
            let uniform = Uniform::from(min..max);
            self.update_memory_weight = T::from(uniform.sample(&mut rng)).unwrap();
        }


    }
}

impl<T> PartialEq for ConnectionGru<T>
where
    T: Float + std::ops::AddAssign + std::cmp::PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        floats_almost_equal(self.memory, other.memory)
            && floats_almost_equal(self.prev_input, other.prev_input)
            && floats_almost_equal(self.input_weight, other.input_weight)
            && floats_almost_equal(self.reset_input_weight, other.reset_input_weight)
            && floats_almost_equal(self.update_input_weight, other.update_input_weight)
            && floats_almost_equal(self.reset_memory_weight, other.reset_memory_weight)
            && floats_almost_equal(self.update_memory_weight, other.update_memory_weight)
    }
}

impl<T> Eq for ConnectionGru<T> where T: Float + std::ops::AddAssign + std::cmp::PartialEq {}
