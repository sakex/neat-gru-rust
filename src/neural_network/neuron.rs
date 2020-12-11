use num::Float;
use crate::topology::bias::Bias;
use crate::neural_network::connection_gru::ConnectionGru;
use crate::neural_network::connection_sigmoid::ConnectionSigmoid;

pub struct Neuron<T>
    where T: Float {
    pub input: T,
    pub memory: T,
    pub update: T,
    pub reset: T,
    pub prev_reset: T,
    pub bias: Bias<T>,
    pub activated: bool,
    pub connections_gru: Vec<ConnectionGru<T>>,
    pub connections_sigmoid: Vec<ConnectionSigmoid<T>>
}

impl<T> Neuron<T> where T: Float {
    pub fn new() -> Neuron<T> {
        Neuron {
            input: T::from(0).unwrap(),
            memory: T::from(0).unwrap(),
            update: T::from(0).unwrap(),
            reset: T::from(0).unwrap(),
            prev_reset: T::from(0).unwrap(),
            bias: Bias::new(),
            activated: false,
            connections_gru: vec![],
            connections_sigmoid: vec![]
        }
    }

    #[inline(always)]
    pub fn set_input_value(&mut self, input: T) {
        self.input = input + self.bias.bias_input;
        self.activated = true;
        self.update = self.bias.bias_update;
        self.reset = self.bias.bias_reset;
    }

    #[inline(always)]
    pub fn feed_forward(&mut self) {
        // TODO
    }

    #[inline(always)]
    pub fn get_value(&mut self) -> T {
        // TODO
    }

    #[inline(always)]
    pub fn reset_state(&mut self) {
        // TODO
    }
}