use crate::neural_network::connection_gru::ConnectionGru;
use crate::neural_network::connection_sigmoid::ConnectionSigmoid;
use crate::neural_network::neuron::Neuron;
use crate::topology::bias::Bias;
use crate::topology::connection_type::ConnectionType;
use crate::topology::Topology;
use num::Float;
use std::collections::HashMap;
use std::fmt::Display;

use super::connection_relu::ConnectionRelu;

#[derive(Debug)]
pub struct NeuralNetwork<T>
where
    T: Float + std::ops::AddAssign + Display + Send,
{
    input_size: usize,
    output_size: usize,
    neurons: Vec<Neuron<T>>,
    biases: Vec<Bias<T>>,
}

unsafe impl<T> Send for NeuralNetwork<T> where T: Float + std::ops::AddAssign + Display + Send {}
unsafe impl<T> Sync for NeuralNetwork<T> where T: Float + std::ops::AddAssign + Display + Send {}

impl<T> NeuralNetwork<T>
where
    T: Float + std::ops::AddAssign + Display + Send,
{
    /// Instantiates a new Neural Network from a `Topology`
    ///
    /// The neurons are laid out with the inputs first, then the hidden neurons in an order
    /// where each one comes after the neurons it depends on, then the outputs.
    pub fn new(topology: &Topology<T>) -> NeuralNetwork<T> {
        let input_size = topology.inputs();
        let output_size = topology.outputs();
        let hidden_order = topology.hidden_evaluation_order();
        let first_output = input_size + hidden_order.len();
        let neurons_count = first_output + output_size;

        // Position of each neuron of the topology in the network
        let mut positions: HashMap<usize, usize> = HashMap::with_capacity(neurons_count);
        for input in 0..input_size {
            positions.insert(input, input);
        }
        for output in 0..output_size {
            positions.insert(input_size + output, first_output + output);
        }
        for (index, neuron) in hidden_order.iter().enumerate() {
            positions.insert(*neuron, input_size + index);
        }

        let mut neurons: Vec<Neuron<T>> = (0..neurons_count).map(|_| Neuron::new()).collect();
        let mut biases: Vec<Bias<T>> = vec![Bias::new_zero(); neurons_count];
        for (neuron, bias) in topology.hidden_bias() {
            biases[positions[neuron]] = bias.clone();
        }
        for (index, bias) in topology.output_bias().iter().enumerate() {
            biases[first_output + index] = bias.clone();
        }

        let neurons_ptr = neurons.as_mut_ptr();
        // Sorted so that the same topology always gives the same network
        for gene in topology.sorted_genes() {
            if gene.disabled {
                continue;
            }
            let (input, output) = match (positions.get(&gene.input), positions.get(&gene.output)) {
                (Some(&input), Some(&output)) => (input, output),
                _ => continue,
            };
            // Connections only go forward, anything else would be an ill-formed topology
            if output <= input {
                continue;
            }
            // Safety: both positions are within the neurons, which are never reallocated
            unsafe {
                let input_neuron: *mut Neuron<T> = neurons_ptr.add(input);
                let output_neuron: *mut Neuron<T> = neurons_ptr.add(output);
                match gene.connection_type {
                    ConnectionType::Sigmoid => {
                        let connection = ConnectionSigmoid::new(gene.input_weight, output_neuron);
                        (*input_neuron).connections_sigmoid.push(connection);
                    }
                    ConnectionType::Relu => {
                        let connection = ConnectionRelu::new(gene.input_weight, output_neuron);
                        (*input_neuron).connections_relu.push(connection);
                    }
                    ConnectionType::GRU => {
                        let connection = ConnectionGru::new(
                            gene.input_weight,
                            gene.memory_weight,
                            gene.reset_input_weight,
                            gene.update_input_weight,
                            gene.reset_memory_weight,
                            gene.update_memory_weight,
                            output_neuron,
                        );
                        (*input_neuron).connections_gru.push(connection);
                    }
                }
            }
        }

        let mut net = NeuralNetwork {
            input_size,
            output_size,
            neurons,
            biases,
        };

        net.reset_neurons_value();
        net
    }

    #[inline]
    fn reset_neurons_value(&mut self) {
        for (neuron, bias) in self.neurons.iter_mut().zip(self.biases.iter()) {
            neuron.reset_value(bias);
        }
    }

    /// Assigns the inputs to the first layer. Inputs in excess of the first layer are ignored
    #[inline]
    fn set_inputs(&mut self, inputs: &[T]) {
        for (neuron, input) in self
            .neurons
            .iter_mut()
            .take(self.input_size)
            .zip(inputs.iter())
        {
            neuron.set_input_value(*input);
        }
    }

    /// Computes the Neural Network based on the inputs and returns the output into a Vec
    #[inline]
    pub fn compute(&mut self, inputs: &[T]) -> Vec<T> {
        self.set_inputs(inputs);
        let take_amount = self.neurons.len() - self.output_size;
        for neuron in self.neurons.iter_mut().take(take_amount) {
            neuron.feed_forward();
        }
        let ret = self
            .neurons
            .iter_mut()
            .skip(take_amount)
            .map(|neuron| neuron.get_value())
            .collect();
        self.reset_neurons_value();
        ret
    }

    /// Computes the output into a buffer to prevent allocation
    ///
    /// # Safety
    ///
    /// The output buffer must be of the same size as the output layer
    #[inline]
    pub unsafe fn compute_buffer(&mut self, inputs: &[T], output_buffer: &mut [T]) {
        self.set_inputs(inputs);
        let take_amount = self.neurons.len() - self.output_size;
        for neuron in self.neurons.iter_mut().take(take_amount) {
            neuron.feed_forward();
        }
        for (idx, neuron) in self.neurons.iter_mut().skip(take_amount).enumerate() {
            *output_buffer.get_unchecked_mut(idx) = neuron.get_value();
        }
        self.reset_neurons_value();
    }

    #[inline]
    pub fn reset_state(&mut self) {
        for neuron in self.neurons.iter_mut() {
            neuron.reset_state();
        }
    }

    /// Loads a network from the JSON representation of a `Topology`
    ///
    /// # Panics
    ///
    /// Panics if the JSON is malformed or describes an ill-formed topology
    pub fn from_string(serialized: &str) -> NeuralNetwork<T> {
        let top = Topology::from_string(serialized);
        NeuralNetwork::new(&top)
    }
}

impl<T> PartialEq for NeuralNetwork<T>
where
    T: Float + std::ops::AddAssign + Display + Send,
{
    fn eq(&self, other: &Self) -> bool {
        if self.output_size != other.output_size {
            return false;
        }
        self.neurons
            .iter()
            .zip(other.neurons.iter())
            .all(|(first, second)| *first == *second)
    }
}

impl<T> Clone for NeuralNetwork<T>
where
    T: Float + std::ops::AddAssign + Display + Send,
{
    fn clone(&self) -> NeuralNetwork<T> {
        let mut ret = NeuralNetwork {
            input_size: self.input_size,
            output_size: self.output_size,
            neurons: self
                .neurons
                .iter()
                .map(Neuron::clone_with_old_pointer)
                .collect(),
            biases: self.biases.clone(),
        };
        if !self.neurons.is_empty() {
            let original_address = &self.neurons[0] as *const Neuron<T>;
            let new_address = &ret.neurons[0] as *const Neuron<T>;
            let diff = new_address as isize - original_address as isize;
            for neuron in &mut ret.neurons {
                neuron.increment_connections_pointers(diff);
            }
        }
        ret
    }
}
