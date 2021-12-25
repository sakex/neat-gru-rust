use crate::neural_network::functions::fast_tanh;
use crate::neural_network::neuron::Neuron;
use crate::utils::floats_almost_equal;
use num::Float;
#[derive(Debug)]
pub struct ConnectionGru<T>
where
    T: Float + std::ops::AddAssign + std::cmp::PartialEq + Send,
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

unsafe impl<T> Send for ConnectionGru<T> where
    T: Float + std::cmp::PartialEq + std::cmp::PartialEq + std::ops::AddAssign + Send
{
}

impl<T> ConnectionGru<T>
where
    T: Float + std::ops::AddAssign + std::cmp::PartialEq + Send,
{
    pub(crate) fn new(
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

    pub(crate) fn clone_with_old_pointer(&self) -> ConnectionGru<T> {
        ConnectionGru {
            memory: self.memory,
            prev_input: self.prev_input,
            input_weight: self.input_weight,
            memory_weight: self.memory_weight,
            reset_input_weight: self.reset_input_weight,
            update_input_weight: self.update_input_weight,
            reset_memory_weight: self.reset_memory_weight,
            update_memory_weight: self.update_memory_weight,
            output: self.output,
        }
    }

    pub(crate) fn increment_pointer(&mut self, diff: isize) {
        self.output = (self.output as isize + diff) as *mut Neuron<T>;
    }

    #[inline]
    pub(crate) fn activate(&mut self, value: T) {
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
    pub(crate) fn reset_state(&mut self) {
        self.memory = T::zero();
        self.prev_input = T::zero();
    }
}

impl<T> PartialEq for ConnectionGru<T>
where
    T: Float + std::ops::AddAssign + std::cmp::PartialEq + Send,
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

impl<T> Eq for ConnectionGru<T> where T: Float + std::ops::AddAssign + std::cmp::PartialEq + Send {}
