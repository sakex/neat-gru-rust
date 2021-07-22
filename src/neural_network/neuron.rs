use crate::neural_network::connection_gru::ConnectionGru;
use crate::neural_network::connection_sigmoid::ConnectionSigmoid;
use crate::neural_network::functions::{fast_sigmoid, fast_tanh};
use crate::topology::bias::Bias;
use crate::utils::floats_almost_equal;
use num::Float;
use numeric_literals::replace_numeric_literals;
use std::ops::AddAssign;

pub struct Neuron<T>
where
    T: Float + std::cmp::PartialEq + std::cmp::PartialEq + AddAssign,
{
    input: T,
    memory: T,
    update: T,
    reset: T,
    pub(crate) prev_reset: T,
    pub(crate) connections_gru: Vec<ConnectionGru<T>>,
    pub(crate) connections_sigmoid: Vec<ConnectionSigmoid<T>>,
}

impl<T> Neuron<T>
where
    T: Float + std::cmp::PartialEq + std::cmp::PartialEq + AddAssign,
{
    pub fn new() -> Neuron<T> {
        Neuron {
            input: T::zero(),
            memory: T::zero(),
            update: T::zero(),
            reset: T::zero(),
            prev_reset: T::zero(),
            connections_gru: vec![],
            connections_sigmoid: vec![],
        }
    }

    pub fn clone_with_old_pointer(&self) -> Neuron<T> {
        Neuron {
            input: self.input,
            memory: self.memory,
            update: self.update,
            reset: self.reset,
            prev_reset: self.prev_reset,
            connections_gru: self
                .connections_gru
                .iter()
                .map(ConnectionGru::clone_with_old_pointer)
                .collect(),
            connections_sigmoid: self
                .connections_sigmoid
                .iter()
                .map(ConnectionSigmoid::<T>::clone_with_old_pointer)
                .collect(),
        }
    }

    pub fn increment_connections_pointers(&mut self, diff: isize) {
        for connection in &mut self.connections_gru {
            connection.increment_pointer(diff);
        }
        for connection in &mut self.connections_sigmoid {
            connection.increment_pointer(diff);
        }
    }

    #[replace_numeric_literals(T::from(literal).unwrap())]
    #[inline]
    pub fn set_input_value(&mut self, input: T) {
        self.input = input;
        self.update = -1000;
        self.reset = -1000;
        self.memory = 0;
        self.prev_reset = 0;
    }

    #[replace_numeric_literals(T::from(literal).unwrap())]
    #[inline]
    pub fn get_value(&mut self) -> T {
        let update_gate = fast_sigmoid(self.update);
        let reset_gate = fast_sigmoid(self.reset);
        let current_memory = fast_tanh(self.input + self.memory * reset_gate);
        let value = update_gate * self.memory + (1 - update_gate) * current_memory;

        self.prev_reset = reset_gate;
        fast_tanh(value)
    }

    #[replace_numeric_literals(T::from(literal).unwrap())]
    #[inline]
    pub fn feed_forward(&mut self) {
        let update_gate = fast_sigmoid(self.update);
        let reset_gate = fast_sigmoid(self.reset);
        let current_memory = self.input + self.memory * reset_gate;
        let value = update_gate * self.memory + (1 - update_gate) * current_memory;
        for connection in self.connections_gru.iter_mut() {
            connection.activate(value);
        }

        for connection in self.connections_sigmoid.iter_mut() {
            connection.activate(value);
        }

        self.prev_reset = reset_gate;
    }

    #[inline]
    pub fn reset_value(&mut self, bias: &Bias<T>) {
        self.input = bias.bias_input;
        self.update = bias.bias_update;
        self.reset = bias.bias_reset;
        self.memory = T::zero();
    }

    #[inline]
    pub fn reset_state(&mut self) {
        self.prev_reset = T::zero();
        for connection in self.connections_gru.iter_mut() {
            connection.reset_state();
        }
    }

    #[inline]
    pub fn increment_state(&mut self, mem: T, inp: T, res: T, upd: T) {
        self.memory += mem;
        self.input += inp;
        self.reset += res;
        self.update += upd;
    }

    #[inline]
    pub fn increment_value(&mut self, value: T) {
        self.input += value;
    }
}

impl<T> PartialEq for Neuron<T>
where
    T: Float + std::cmp::PartialEq + std::cmp::PartialEq + AddAssign,
{
    fn eq(&self, other: &Neuron<T>) -> bool {
        if self.connections_sigmoid.len() != other.connections_sigmoid.len()
            || self.connections_gru.len() != other.connections_gru.len()
        {
            return false;
        }

        if !(floats_almost_equal(self.input, other.input)
            && floats_almost_equal(self.memory, other.memory)
            && floats_almost_equal(self.update, other.update)
            && floats_almost_equal(self.reset, other.reset)
            && floats_almost_equal(self.prev_reset, other.prev_reset))
        {
            return false;
        }
        if !self
            .connections_gru
            .iter()
            .zip(other.connections_gru.iter())
            .all(|(c1, c2)| *c1 == *c2)
        {
            return false;
        }

        self.connections_sigmoid
            .iter()
            .zip(other.connections_sigmoid.iter())
            .all(|(c1, c2)| *c1 == *c2)
    }
}
