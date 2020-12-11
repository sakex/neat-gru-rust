use crate::topology::topology::Topology;
use num::Float;
use crate::neural_network::neuron::Neuron;
use crate::neural_network::connection_sigmoid::ConnectionSigmoid;
use crate::neural_network::connection_gru::ConnectionGru;
use crate::topology::connection_type::ConnectionType;

pub struct NeuralNetwork<T>
    where T: Float {
    input_size: usize,
    output_size: usize,
    neurons: Vec<Neuron<T>>,
}

impl<T> NeuralNetwork<T>
    where T: Float {
    pub unsafe fn new(topology: &Topology<T>) -> NeuralNetwork<T> {
        let layer_count = topology.layers;
        let sizes = &topology.layers_sizes;
        let mut layer_addresses = vec![0; layer_count];
        let mut neurons_count: usize = 0;
        for i in 0..layer_count {
            layer_addresses[i] = neurons_count;
            neurons_count += sizes[i] as usize;
        }
        let input_size = layer_addresses[0];
        let output_size = *layer_addresses.last().unwrap();
        let mut neurons: Vec<Neuron<T>> = Vec::with_capacity(neurons_count);
        for _ in 0..neurons_count {
            neurons.push(Neuron::new());
        }

        let neurons_ptr = neurons.as_mut_ptr();

        for (point, gene_and_bias) in topology.genes_point.iter() {
            let neuron_index = layer_addresses[point.layer as usize] + point.index as usize;
            let mut input_neuron: *mut Neuron<T> = neurons_ptr.offset(neuron_index as isize);
            let bias = &gene_and_bias.bias;
            (*input_neuron).bias = bias.clone();
            for gene_rc in &gene_and_bias.genes {
                let gene = gene_rc.borrow();
                if gene.disabled {
                    continue;
                }
                let output = &gene.output;
                let index = layer_addresses[output.layer as usize] + output.index as usize;
                let output_neuron: *const Neuron<T> = neurons_ptr.offset(index as isize);
                match gene.connection_type {
                    ConnectionType::Sigmoid => {
                        let connection = ConnectionSigmoid::new(gene.input_weight, output_neuron);
                        (*input_neuron).connections_sigmoid.push(connection);
                    }
                    ConnectionType::GRU => {
                        let connection = ConnectionGru::new(
                            gene.input_weight,
                            gene.memory_weight,
                            gene.reset_input_weight,
                            gene.update_input_weight,
                            gene.reset_memory_weight,
                            gene.update_memory_weight,
                            output_neuron);
                        (*input_neuron).connections_gru.push(connection);
                    }
                }
            }
        }


        NeuralNetwork {
            input_size,
            output_size,
            neurons,
        }
    }

    #[inline(always)]
    pub fn compute(&mut self, inputs: &[T]) -> Vec<T> {
        let len = inputs.len();
        unsafe {
            for i in 0..len {
                self.neurons.get_unchecked_mut(i).set_input_value(inputs.get_unchecked(i));
            }
        }
        let take_amount = self.neurons.len() - self.output_size;
        for neuron in self.neurons.iter_mut().take(take_amount) {
            neuron.feed_forward();
        }
        self.neurons.iter_mut().skip(take_amount).map(|neuron| neuron.get_value()).collect()
    }

    #[inline(always)]
    pub fn reset_state(&mut self) {
        for neuron in self.neurons.iter_mut() {
            neuron.reset_state();
        }
    }
}