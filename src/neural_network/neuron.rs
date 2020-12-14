use crate::neural_network::connection_gru::ConnectionGru;
use crate::neural_network::connection_sigmoid::ConnectionSigmoid;
use crate::neural_network::functions::{fast_sigmoid, fast_tanh};
use crate::topology::bias::Bias;
use num::Float;
use numeric_literals::replace_numeric_literals;

pub struct Neuron<T>
where
    T: Float,
{
    input: T,
    memory: T,
    update: T,
    reset: T,
    pub(crate) prev_reset: T,
    pub(crate) bias: Bias<T>,
    pub(crate) connections_gru: Vec<ConnectionGru<T>>,
    pub(crate) connections_sigmoid: Vec<ConnectionSigmoid<T>>,
}

impl<T> Neuron<T>
where
    T: Float,
{
    #[replace_numeric_literals(T::from(literal).unwrap())]
    pub fn new() -> Neuron<T> {
        Neuron {
            input: 0,
            memory: 0,
            update: 0,
            reset: 0,
            prev_reset: 0,
            bias: Bias::new_zero(),
            connections_gru: vec![],
            connections_sigmoid: vec![],
        }
    }

    #[inline]
    pub fn set_input_value(&mut self, input: T) {
        self.input = input + self.bias.bias_input;
        self.update = self.bias.bias_update;
        self.reset = self.bias.bias_reset;
    }

    #[replace_numeric_literals(T::from(literal).unwrap())]
    #[inline]
    pub fn get_value(&mut self) -> T {
        let update_gate = fast_sigmoid(self.update);
        let reset_gate = fast_sigmoid(self.reset);
        let current_memory = fast_tanh(self.input + self.memory * reset_gate);
        let value = update_gate * self.memory + (1 - update_gate) * current_memory;

        self.prev_reset = reset_gate;
        self.reset_value();
        fast_tanh(value)
    }

    #[replace_numeric_literals(T::from(literal).unwrap())]
    #[inline]
    pub fn feed_forward(&mut self) {
        let update_gate = fast_sigmoid(self.update);
        let reset_gate = fast_sigmoid(self.reset);
        let current_memory = fast_tanh(self.input + self.memory * reset_gate);
        let value = update_gate * self.memory + (1 - update_gate) * current_memory;
        for connection in self.connections_gru.iter_mut() {
            connection.activate(value);
        }

        for connection in self.connections_sigmoid.iter_mut() {
            connection.activate(value);
        }

        self.prev_reset = reset_gate;
        self.reset_value();
    }

    #[replace_numeric_literals(T::from(literal).unwrap())]
    #[inline]
    pub fn reset_value(&mut self) {
        self.input = self.bias.bias_input;
        self.update = self.bias.bias_update;
        self.reset = self.bias.bias_reset;
        self.memory = 0;
    }

    #[replace_numeric_literals(T::from(literal).unwrap())]
    #[inline]
    pub fn reset_state(&mut self) {
        self.reset_value();
        self.prev_reset = 0;
        for connection in self.connections_gru.iter_mut() {
            connection.reset_state();
        }
    }

    #[inline(always)]
    pub fn increment_state(&mut self, mem: T, inp: T, res: T, upd: T) {
        self.memory = self.memory + mem;
        self.input = self.input + inp;
        self.reset = self.reset + res;
        self.update = self.update + upd;
    }

    #[inline(always)]
    pub fn increment_value(&mut self, value: T) {
        self.input = self.input + value;
    }
}
