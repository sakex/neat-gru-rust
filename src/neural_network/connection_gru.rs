use num::Float;
use crate::neural_network::neuron::Neuron;

pub struct ConnectionGru<T>
    where T: Float {
    memory: T,
    prev_input: T,
    input_weight: T,
    memory_weight: T,
    reset_input_weight: T,
    update_input_weight: T,
    reset_memory_weight: T,
    update_memory_weight: T,
    output: *const Neuron<T>,
}

impl<T> ConnectionGru<T> where T: Float {
    pub fn new(input_weight: T,
               memory_weight: T,
               reset_input_weight: T,
               update_input_weight: T,
               reset_memory_weight: T,
               update_memory_weight: T,
               output: *const Neuron<T>) -> ConnectionGru<T> {
        ConnectionGru {
            memory: T::from(0).unwrap(),
            prev_input: T::from(0).unwrap(),
            input_weight,
            memory_weight,
            reset_input_weight,
            update_input_weight,
            reset_memory_weight,
            update_memory_weight,
            output,
        }
    }
}