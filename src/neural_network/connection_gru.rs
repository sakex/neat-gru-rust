use crate::neural_network::functions::fast_tanh;
use crate::neural_network::neuron::Neuron;
use num::Float;
use numeric_literals::replace_numeric_literals;

pub struct ConnectionGru<T>
where
    T: Float,
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
    T: Float,
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

    #[replace_numeric_literals(T::from(literal).unwrap())]
    #[inline]
    pub fn reset_state(&mut self) {
        self.memory = 0;
        self.prev_input = 0;
    }
}
