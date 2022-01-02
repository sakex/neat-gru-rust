use crate::neural_network::connection_gru::ConnectionGru;
use crate::neural_network::connection_sigmoid::ConnectionSigmoid;
use crate::neural_network::neuron::Neuron;
use crate::topology::bias::Bias;
use crate::topology::connection_type::ConnectionType;
use crate::topology::Topology;
use num::Float;
use std::fmt::Display;

use super::connection_relu::ConnectionRelu;
use super::nn_trait::NN;

#[derive(Debug)]
pub struct NeuralNetwork<T>
where
    T: Float + std::ops::AddAssign + Display + Send,
{
    output_size: usize,
    neurons: Vec<Neuron<T>>,
    biases: Vec<Bias<T>>,
}

unsafe impl<T> Send for NeuralNetwork<T> where T: Float + std::ops::AddAssign + Display + Send {}
unsafe impl<T> Sync for NeuralNetwork<T> where T: Float + std::ops::AddAssign + Display + Send {}

impl<T> NN<T> for NeuralNetwork<T>
where
    T: Float + std::ops::AddAssign + Display + Send,
{
    unsafe fn from_topology(topology: &Topology<T>) -> NeuralNetwork<T> {
        let layer_count = topology.layers_sizes.len();
        let sizes = &topology.layers_sizes;
        let mut layer_addresses = vec![0; layer_count];
        let mut neurons_count: usize = 0;
        for i in 0..layer_count {
            layer_addresses[i] = neurons_count;
            neurons_count += sizes[i] as usize;
        }
        let output_size = *sizes.last().unwrap() as usize;
        let mut neurons: Vec<Neuron<T>> = Vec::with_capacity(neurons_count);
        let mut biases: Vec<Bias<T>> = Vec::with_capacity(neurons_count);
        for _ in 0..neurons_count {
            neurons.push(Neuron::new());
        }
        for _ in 0..neurons_count {
            biases.push(Bias::new_zero());
        }

        let neurons_ptr = neurons.as_mut_ptr();

        for (point, gene_and_bias) in topology.genes_point.iter() {
            if gene_and_bias.genes.is_empty()
                || gene_and_bias
                    .genes
                    .iter()
                    .all(|gene| gene.borrow().disabled)
            {
                continue;
            }
            let neuron_index = layer_addresses[point.layer as usize] + point.index as usize;
            let input_neuron: *mut Neuron<T> = neurons_ptr.add(neuron_index);
            biases[neuron_index] = gene_and_bias.bias.clone();
            for gene_rc in &gene_and_bias.genes {
                let gene = gene_rc.borrow();
                if gene.disabled {
                    continue;
                }
                let output = &gene.output;
                let index = layer_addresses[output.layer as usize] + output.index as usize;
                let output_neuron: *mut Neuron<T> = neurons_ptr.add(index);
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

        let base = output_size as isize - neurons_count as isize;

        for it in (neurons_count - output_size) as isize..neurons_count as isize {
            biases[it as usize] = topology.output_bias[(it + base) as usize].clone();
        }

        let mut net = NeuralNetwork {
            output_size,
            neurons,
            biases,
        };

        net.reset_neurons_value();
        net
    }
}

impl<T> NeuralNetwork<T>
where
    T: Float + std::ops::AddAssign + Display + Send,
{
    #[inline]
    fn reset_neurons_value(&mut self) {
        for (neuron, bias) in self.neurons.iter_mut().zip(self.biases.iter()) {
            neuron.reset_value(bias);
        }
    }

    #[inline]
    pub fn compute(&mut self, inputs: &[T]) -> Vec<T> {
        let len = inputs.len();
        unsafe {
            for i in 0..len {
                self.neurons
                    .get_unchecked_mut(i)
                    .set_input_value(*inputs.get_unchecked(i));
            }
        }
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

    #[inline]
    pub fn reset_state(&mut self) {
        for neuron in self.neurons.iter_mut() {
            neuron.reset_state();
        }
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
