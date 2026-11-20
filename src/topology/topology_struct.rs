use crate::topology::bias::Bias;
use crate::topology::connection_type::ConnectionType;
use crate::topology::gene::Gene;
use crate::topology::mutation_probabilities::MutationProbabilities;
use crate::topology::serialization::{
    SerializationBias, SerializationGene, SerializationNeuron, SerializationTopology,
};
use crate::train::evolution_number::EvNumber;
use num::traits::Float;
use numeric_literals::replace_numeric_literals;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::{Display, Formatter};
use std::sync::{Arc, Mutex};

const NORMAL_STDDEV: f64 = 0.04;
/// Number of random pairs of neurons tried when adding a connection
const ADD_CONNECTION_ATTEMPTS: usize = 20;

/// The genome of a neural network: a graph of neurons joined by connections (genes).
///
/// Neurons are identified by an id:
/// - `0..inputs` for the input neurons
/// - `inputs..inputs + outputs` for the output neurons
/// - anything above for the hidden neurons
///
/// Connections go from an input or hidden neuron to a hidden or output neuron, and never form
/// a cycle.
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Topology<T>
where
    T: Float + std::ops::AddAssign + Display + Send,
{
    max_hidden: usize,
    last_result: T,
    inputs: usize,
    outputs: usize,
    /// Biases of the hidden neurons, by neuron id
    hidden_bias: HashMap<usize, Bias<T>>,
    output_bias: Vec<Bias<T>>,
    /// Connections, by evolution number
    genes: HashMap<usize, Gene<T>>,
}

pub type TopologySmrtPtr<T> = Arc<Mutex<Topology<T>>>;

impl<T> Topology<T>
where
    T: Float + std::ops::AddAssign + Display + Send,
{
    /// Creates a topology without any connection
    ///
    /// # Arguments
    ///
    /// `inputs` - The number of input neurons
    /// `outputs` - The number of output neurons
    /// `max_hidden` - The maximum number of hidden neurons mutations can create
    pub fn new(inputs: usize, outputs: usize, max_hidden: usize) -> Topology<T> {
        Topology {
            max_hidden,
            last_result: T::zero(),
            inputs,
            outputs,
            hidden_bias: HashMap::new(),
            output_bias: vec![Bias::new_zero(); outputs],
            genes: HashMap::new(),
        }
    }

    /// The number of input neurons
    pub fn inputs(&self) -> usize {
        self.inputs
    }

    /// The number of output neurons
    pub fn outputs(&self) -> usize {
        self.outputs
    }

    /// The number of hidden neurons
    pub fn hidden_count(&self) -> usize {
        self.hidden_bias.len()
    }

    /// The connections, in no particular order
    pub fn genes(&self) -> impl Iterator<Item = &Gene<T>> {
        self.genes.values()
    }

    /// The connections sorted by evolution number
    pub(crate) fn sorted_genes(&self) -> Vec<&Gene<T>> {
        let mut genes: Vec<&Gene<T>> = self.genes.values().collect();
        genes.sort_by_key(|gene| gene.evolution_number);
        genes
    }

    /// The biases of the hidden neurons by neuron id
    pub fn hidden_bias(&self) -> &HashMap<usize, Bias<T>> {
        &self.hidden_bias
    }

    /// The biases of the output neurons
    pub fn output_bias(&self) -> &[Bias<T>] {
        &self.output_bias
    }

    fn first_hidden(&self) -> usize {
        self.inputs + self.outputs
    }

    fn is_hidden(&self, neuron: usize) -> bool {
        neuron >= self.first_hidden()
    }

    fn neuron_exists(&self, neuron: usize) -> bool {
        neuron < self.first_hidden() || self.hidden_bias.contains_key(&neuron)
    }

    #[replace_numeric_literals(T::from(literal).unwrap())]
    pub fn delta_compatibility(top1: &Topology<T>, top2: &Topology<T>, c1: T, c2: T, c3: T) -> T {
        // Disjoints = present in Gene1 but not Gene2
        let mut disjoints = T::zero();
        let mut common = T::zero();
        let mut w = T::zero();

        let one = T::one();
        for (ev_number, gene1) in top1.genes.iter() {
            match top2.genes.get(ev_number) {
                Some(gene2) => {
                    common += one;
                    w += ((gene1.input_weight - gene2.input_weight).abs()
                        + (gene1.memory_weight - gene2.memory_weight).abs()
                        + (gene1.reset_input_weight - gene2.reset_input_weight).abs()
                        + (gene1.update_input_weight - gene2.update_input_weight).abs()
                        + (gene1.reset_memory_weight - gene2.reset_memory_weight).abs()
                        + (gene1.update_memory_weight - gene2.update_memory_weight).abs())
                        / 6;
                }
                None => {
                    disjoints += one;
                }
            }
        }
        if common > T::zero() {
            w = w / common;
        }
        let size_1 = T::from(top1.genes.len()).unwrap();
        let size_2 = T::from(top2.genes.len()).unwrap();
        let larger = size_1.max(size_2);
        let n = if larger > 20 { larger } else { 1 };
        // Excess = present in gene2 but not gene1
        let excess = size_2 - common;
        let v = c1 * disjoints + c2 * excess;
        v / n + w * c3
    }

    /// Creates a topology where each input is connected to some outputs with random weights
    pub fn new_random(
        rng: &mut ThreadRng,
        input_count: usize,
        output_count: usize,
        max_hidden: usize,
        ev_number: &EvNumber,
    ) -> Topology<T> {
        let connections_per_input = (output_count as f64 / input_count as f64).ceil() as usize;
        let mut new_topology = Topology::new(input_count, output_count, max_hidden);
        let mut output_index: usize = 0;
        for input in 0..input_count {
            for _ in 0..connections_per_input {
                let output = input_count + output_index;
                output_index = (output_index + 1) % output_count;
                new_topology.insert_gene(Gene::new_random(rng, input, output, -1., 1., ev_number));
            }
        }
        new_topology.output_bias = (0..output_count).map(|_| Bias::new_random(rng)).collect();
        new_topology
    }

    /// Creates a topology where each input is connected to each output with weights of 1
    pub fn new_uniform(
        input_count: usize,
        output_count: usize,
        max_hidden: usize,
        ev_number: &EvNumber,
    ) -> Topology<T> {
        let mut new_topology = Topology::new(input_count, output_count, max_hidden);
        for input in 0..input_count {
            for output in 0..output_count {
                new_topology.insert_gene(Gene::new_one(input, input_count + output, ev_number));
            }
        }
        new_topology
    }

    pub fn set_last_result(&mut self, result: T) {
        self.last_result = result;
    }

    pub fn get_last_result(&self) -> T {
        self.last_result
    }

    pub fn new_generation(
        &self,
        new_topologies: &mut Vec<Arc<Mutex<Topology<T>>>>,
        ev_number: &EvNumber,
        reproduction_count: usize,
        proba: &MutationProbabilities,
    ) {
        for _ in 0..reproduction_count {
            let mut cp = self.clone();
            cp.mutate(ev_number, proba);
            new_topologies.push(Arc::new(Mutex::new(cp)));
        }
    }

    pub fn change_weights(&mut self, rng: &mut ThreadRng) {
        let normal = Normal::new(0.0, NORMAL_STDDEV).unwrap();
        for gene in self.genes.values_mut() {
            let change_weights = rng.gen_range(0.0..1.);
            if change_weights < 0.995 {
                gene.input_weight += T::from(normal.sample(rng)).unwrap();
                gene.memory_weight += T::from(normal.sample(rng)).unwrap();
                gene.reset_input_weight += T::from(normal.sample(rng)).unwrap();
                gene.update_input_weight += T::from(normal.sample(rng)).unwrap();
                gene.reset_memory_weight += T::from(normal.sample(rng)).unwrap();
                gene.update_memory_weight += T::from(normal.sample(rng)).unwrap();
            } else {
                gene.random_reassign(rng);
            }
        }
        for bias in self.hidden_bias.values_mut() {
            let change_bias = rng.gen_range(0.0..1.);
            if change_bias < 0.995 {
                bias.bias_input += T::from(normal.sample(rng)).unwrap();
                bias.bias_update += T::from(normal.sample(rng)).unwrap();
                bias.bias_reset += T::from(normal.sample(rng)).unwrap();
            } else {
                *bias = Bias::new_random(rng);
            }
        }
        for bias in &mut self.output_bias {
            let change_bias = rng.gen_range(0.0..1.);
            if change_bias < 0.9 {
                bias.bias_input += T::from(normal.sample(rng)).unwrap();
                bias.bias_update += T::from(normal.sample(rng)).unwrap();
                bias.bias_reset += T::from(normal.sample(rng)).unwrap();
            }
        }
    }

    /// Adds a connection, replacing the one with the same evolution number if there is one.
    ///
    /// Hidden neurons the connection refers to are created if needed
    #[inline]
    pub fn insert_gene(&mut self, gene: Gene<T>) {
        for neuron in [gene.input, gene.output] {
            if self.is_hidden(neuron) {
                self.hidden_bias
                    .entry(neuron)
                    .or_insert_with(Bias::new_zero);
            }
        }
        self.genes.insert(gene.evolution_number, gene);
    }

    /// Picks a random connection and splits it by adding a neuron in the middle
    ///
    /// Returns false if no neuron could be added without exceeding the maximum of hidden neurons
    fn add_node(&mut self, ev_number: &EvNumber, rng: &mut ThreadRng) -> bool {
        if self.hidden_bias.len() >= self.max_hidden {
            return false;
        }
        let enabled: Vec<usize> = self
            .genes
            .values()
            .filter(|gene| !gene.disabled)
            .map(|gene| gene.evolution_number)
            .collect();
        if enabled.is_empty() {
            return false;
        }
        let to_split = enabled[rng.gen_range(0..enabled.len())];
        let original_gene = {
            let gene = self.genes.get_mut(&to_split).unwrap();
            gene.disabled = true;
            gene.clone()
        };
        // Splitting the same connection gives the same neuron in every topology, unless this
        // topology already split it before
        let mut middle = self.first_hidden() + ev_number.split_neuron(to_split);
        if self.hidden_bias.contains_key(&middle) {
            middle = self.first_hidden() + ev_number.new_neuron();
        }
        let (first_gene, second_gene) = original_gene.split(middle, ev_number);
        self.insert_gene(first_gene);
        self.insert_gene(second_gene);
        true
    }

    /// Connects two random neurons that aren't connected yet
    ///
    /// Returns false if no connection could be added
    fn add_connection(&mut self, ev_number: &EvNumber, rng: &mut ThreadRng) -> bool {
        let hidden: Vec<usize> = self.hidden_bias.keys().copied().collect();
        // Sources are inputs and hidden neurons, targets are outputs and hidden neurons
        let pick = |rng: &mut ThreadRng, others: std::ops::Range<usize>| {
            let index = rng.gen_range(0..others.len() + hidden.len());
            if index < others.len() {
                others.start + index
            } else {
                hidden[index - others.len()]
            }
        };
        for _ in 0..ADD_CONNECTION_ATTEMPTS {
            let input = pick(rng, 0..self.inputs);
            let output = pick(rng, self.inputs..self.first_hidden());
            if input == output || self.has_path(output, input) {
                continue;
            }
            let number = ev_number.number(input, output);
            match self.genes.get_mut(&number) {
                Some(gene) if !gene.disabled => continue,
                Some(gene) => {
                    // The connection existed in the past, bring it back
                    gene.disabled = false;
                    gene.random_reassign(rng);
                }
                None => {
                    self.insert_gene(Gene::new_random(rng, input, output, -1., 1., ev_number));
                }
            }
            return true;
        }
        false
    }

    /// Returns true if `to` can be reached from `from` by following enabled connections
    fn has_path(&self, from: usize, to: usize) -> bool {
        if !self.is_hidden(from) || !self.is_hidden(to) {
            // Inputs have no incoming connection and outputs have no outgoing one
            return false;
        }
        let mut outgoing: HashMap<usize, Vec<usize>> = HashMap::new();
        for gene in self.genes.values().filter(|gene| !gene.disabled) {
            outgoing.entry(gene.input).or_default().push(gene.output);
        }
        let mut visited: HashSet<usize> = HashSet::new();
        let mut stack = vec![from];
        while let Some(neuron) = stack.pop() {
            if neuron == to {
                return true;
            }
            if visited.insert(neuron) {
                if let Some(next) = outgoing.get(&neuron) {
                    stack.extend(next);
                }
            }
        }
        false
    }

    pub fn mutate(&mut self, ev_number: &EvNumber, proba: &MutationProbabilities) {
        let mut rng = thread_rng();
        let change_weights = rng.gen_range(0.0..1.);
        if change_weights < proba.change_weights {
            self.change_weights(&mut rng);
        } else {
            let change_topology = rng.gen_range(0.0..1.);
            // Fall back to a new connection if no neuron can be added within the size limit
            if change_topology >= proba.guaranteed_new_neuron || !self.add_node(ev_number, &mut rng)
            {
                self.add_connection(ev_number, &mut rng);
            }
        }
        self.remove_dangling_neurons();
    }

    /// Removes the hidden neurons that lost all their inputs or all their outputs
    fn remove_dangling_neurons(&mut self) {
        loop {
            let mut has_input: HashSet<usize> = HashSet::new();
            let mut has_output: HashSet<usize> = HashSet::new();
            for gene in self.genes.values().filter(|gene| !gene.disabled) {
                has_output.insert(gene.input);
                has_input.insert(gene.output);
            }
            let before = self.hidden_bias.len();
            self.hidden_bias
                .retain(|neuron, _| has_input.contains(neuron) && has_output.contains(neuron));
            if self.hidden_bias.len() == before {
                break;
            }
            // Disable the connections of the neurons that were removed
            let first_hidden = self.first_hidden();
            let hidden_bias = &self.hidden_bias;
            let exists = |neuron: usize| neuron < first_hidden || hidden_bias.contains_key(&neuron);
            for gene in self.genes.values_mut() {
                if !exists(gene.input) || !exists(gene.output) {
                    gene.disabled = true;
                }
            }
        }
    }

    /// Returns the hidden neurons in an order where each neuron comes after all the neurons
    /// it receives a connection from
    pub(crate) fn hidden_evaluation_order(&self) -> Vec<usize> {
        let mut incoming: HashMap<usize, usize> =
            self.hidden_bias.keys().map(|&neuron| (neuron, 0)).collect();
        let mut outgoing: HashMap<usize, Vec<usize>> = HashMap::new();
        for gene in self.genes.values().filter(|gene| !gene.disabled) {
            if self.hidden_bias.contains_key(&gene.input) {
                if let Some(count) = incoming.get_mut(&gene.output) {
                    *count += 1;
                    outgoing.entry(gene.input).or_default().push(gene.output);
                }
            }
        }
        // Sorted so that the same topology always gives the same order
        let mut ready: BTreeSet<usize> = incoming
            .iter()
            .filter(|(_, &count)| count == 0)
            .map(|(&neuron, _)| neuron)
            .collect();
        let mut order = Vec::with_capacity(incoming.len());
        while let Some(neuron) = ready.pop_first() {
            order.push(neuron);
            for next in outgoing.get(&neuron).into_iter().flatten() {
                let count = incoming.get_mut(next).unwrap();
                *count -= 1;
                if *count == 0 {
                    ready.insert(*next);
                }
            }
        }
        if order.len() < incoming.len() {
            // Only happens on ill-formed topologies with cycles
            let ordered: HashSet<usize> = order.iter().copied().collect();
            let mut remaining: Vec<usize> = incoming
                .keys()
                .copied()
                .filter(|neuron| !ordered.contains(neuron))
                .collect();
            remaining.sort_unstable();
            order.append(&mut remaining);
        }
        order
    }

    /// Returns true if the string can be loaded with `from_string`
    pub fn is_valid_topology_json(serialized: &str) -> bool {
        Self::try_from_string(serialized).is_ok()
    }

    /// Loads a topology from its JSON representation
    ///
    /// # Panics
    ///
    /// Panics if the JSON is malformed or describes an ill-formed topology.
    /// Use `try_from_string` to handle the error instead.
    pub fn from_string(serialized: &str) -> Topology<T> {
        match Self::try_from_string(serialized) {
            Ok(topology) => topology,
            Err(error) => panic!("Invalid topology: {}", error),
        }
    }

    /// Loads a topology from its JSON representation, checking that it is well-formed.
    ///
    /// Files saved when topologies were organised in layers can still be loaded.
    pub fn try_from_string(serialized: &str) -> Result<Topology<T>, String> {
        let mut serialization: SerializationTopology =
            SerializationTopology::from_string(serialized).map_err(|err| err.to_string())?;
        let (inputs, outputs) = match (serialization.inputs, serialization.outputs) {
            (Some(inputs), Some(outputs)) => (inputs, outputs),
            (None, None) => Self::convert_layers_to_ids(&mut serialization)?,
            _ => return Err("Only one of inputs and outputs is given".to_string()),
        };
        if inputs == 0 || outputs == 0 {
            return Err("The topology needs at least one input and one output".to_string());
        }
        let id = |neuron: SerializationNeuron| match neuron {
            SerializationNeuron::Id(id) => Ok(id),
            SerializationNeuron::Layered(layer, index) => Err(format!(
                "Neuron [{}, {}] should be an id as the topology isn't layered",
                layer, index
            )),
        };
        let mut topology = Topology::new(inputs, outputs, 0);
        for ser_bias in &serialization.biases {
            let neuron = id(ser_bias.neuron)?;
            let bias = Bias {
                bias_input: Self::cast_value(ser_bias.bias.bias_input)?,
                bias_update: Self::cast_value(ser_bias.bias.bias_update)?,
                bias_reset: Self::cast_value(ser_bias.bias.bias_reset)?,
            };
            if topology.is_hidden(neuron) {
                topology.hidden_bias.insert(neuron, bias);
            } else if neuron >= inputs {
                topology.output_bias[neuron - inputs] = bias;
            }
        }

        let mut used_numbers: HashSet<usize> = HashSet::new();
        let mut genes = Vec::new();
        for gene in &serialization.genes {
            let (input, output) = (id(gene.input)?, id(gene.output)?);
            if !(0..3).contains(&gene.connection_type) {
                return Err(format!("Invalid connection type {}", gene.connection_type));
            }
            if !gene.disabled {
                let is_output = |neuron: usize| neuron >= inputs && neuron < inputs + outputs;
                let valid = topology.neuron_exists(input)
                    && topology.neuron_exists(output)
                    && !is_output(input)
                    && output >= inputs;
                if !valid {
                    return Err(format!(
                        "Gene {} -> {} doesn't go from an input or hidden neuron to a hidden or output neuron",
                        input, output
                    ));
                }
            }
            let evolution_number = match gene.ev_number {
                Some(ev_number) => Some(
                    num::cast::<f64, usize>(ev_number)
                        .ok_or_else(|| format!("Invalid evolution number {}", ev_number))?,
                ),
                None => None,
            };
            // Genes without a usable evolution number get one below
            let evolution_number = evolution_number.filter(|number| used_numbers.insert(*number));
            genes.push((
                evolution_number,
                Gene {
                    input,
                    output,
                    input_weight: Self::cast_value(gene.input_weight)?,
                    memory_weight: Self::cast_value(gene.memory_weight)?,
                    reset_input_weight: Self::cast_value(gene.reset_input_weight)?,
                    update_input_weight: Self::cast_value(gene.update_input_weight)?,
                    reset_memory_weight: Self::cast_value(gene.reset_memory_weight)?,
                    update_memory_weight: Self::cast_value(gene.update_memory_weight)?,
                    evolution_number: 0,
                    connection_type: ConnectionType::from_int(gene.connection_type),
                    disabled: gene.disabled,
                },
            ));
        }
        let mut next_number = used_numbers.iter().max().map_or(1, |max| max + 1);
        for (evolution_number, mut gene) in genes {
            gene.evolution_number = evolution_number.unwrap_or_else(|| {
                next_number += 1;
                next_number - 1
            });
            topology.genes.insert(gene.evolution_number, gene);
        }

        if topology.has_cycle() {
            return Err("The connections form a cycle".to_string());
        }
        topology.max_hidden = topology.hidden_bias.len();
        Ok(topology)
    }

    /// Replaces the `[layer, index]` neurons of a layered file by ids.
    ///
    /// Returns the number of inputs and outputs
    fn convert_layers_to_ids(
        serialization: &mut SerializationTopology,
    ) -> Result<(usize, usize), String> {
        let layered = |neuron: SerializationNeuron| match neuron {
            SerializationNeuron::Layered(layer, index) => Ok((layer as usize, index as usize)),
            SerializationNeuron::Id(id) => Err(format!(
                "Neuron {} is an id but the topology has no inputs and outputs counts",
                id
            )),
        };
        let mut layers_sizes: Vec<usize> = Vec::new();
        for ser_bias in &serialization.biases {
            let (layer, index) = layered(ser_bias.neuron)?;
            if layer >= layers_sizes.len() {
                layers_sizes.resize(layer + 1, 0);
            }
            layers_sizes[layer] = layers_sizes[layer].max(index + 1);
        }
        if layers_sizes.len() < 2 || layers_sizes.contains(&0) {
            return Err("The topology needs at least two layers, none of them empty".to_string());
        }
        let last_layer = layers_sizes.len() - 1;
        let (inputs, outputs) = (layers_sizes[0], layers_sizes[last_layer]);
        // Hidden neurons come after the inputs and outputs, layer after layer
        let mut layer_start = vec![0; layers_sizes.len()];
        layer_start[last_layer] = inputs;
        let mut next = inputs + outputs;
        for layer in 1..last_layer {
            layer_start[layer] = next;
            next += layers_sizes[layer];
        }
        let convert = |neuron: SerializationNeuron| {
            let (layer, index) = layered(neuron)?;
            if layer > last_layer || index >= layers_sizes[layer] {
                return Err(format!("Neuron [{}, {}] doesn't exist", layer, index));
            }
            Ok(SerializationNeuron::Id(layer_start[layer] + index))
        };

        for ser_bias in serialization.biases.iter_mut() {
            ser_bias.neuron = convert(ser_bias.neuron)?;
        }
        // Every neuron of a hidden layer exists, even the ones without a bias
        for neuron in (inputs + outputs)..next {
            let neuron = SerializationNeuron::Id(neuron);
            if !serialization
                .biases
                .iter()
                .any(|bias| bias.neuron == neuron)
            {
                serialization.biases.push(SerializationBias {
                    neuron,
                    bias: Bias::new_zero(),
                });
            }
        }
        // Disabled genes may refer to neurons that were removed, they are of no use
        serialization.genes.retain(|gene| !gene.disabled);
        for gene in serialization.genes.iter_mut() {
            let (input_layer, _) = layered(gene.input)?;
            let (output_layer, _) = layered(gene.output)?;
            if output_layer <= input_layer {
                return Err(format!(
                    "Gene from layer {} to layer {} doesn't go to a further layer",
                    input_layer, output_layer
                ));
            }
            gene.input = convert(gene.input)?;
            gene.output = convert(gene.output)?;
        }
        Ok((inputs, outputs))
    }

    /// Returns true if the enabled connections form a cycle
    fn has_cycle(&self) -> bool {
        self.genes
            .values()
            .filter(|gene| !gene.disabled)
            .any(|gene| gene.input == gene.output || self.has_path(gene.output, gene.input))
    }

    fn cast_value(value: f64) -> Result<T, String> {
        num::cast(value).ok_or_else(|| format!("Value {} can't be represented", value))
    }

    /// Creates a child from two topologies.
    ///
    /// The child has the structure of the best one. The weights of the connections that exist
    /// in both are taken randomly from one or the other
    #[inline]
    pub fn crossover(best: &Topology<T>, worst: &Topology<T>) -> TopologySmrtPtr<T> {
        let mut rng = thread_rng();
        let mut new_topology = best.clone();
        for (ev_number, gene) in new_topology.genes.iter_mut() {
            if let Some(worst_gene) = worst.genes.get(ev_number) {
                if rng.gen_bool(0.5) {
                    gene.assign_values(worst_gene);
                }
            }
        }
        Arc::new(Mutex::new(new_topology))
    }
}

impl<T> Display for Topology<T>
where
    T: Float + std::ops::AddAssign + Display + Send,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut hidden: Vec<(&usize, &Bias<T>)> = self.hidden_bias.iter().collect();
        hidden.sort_by_key(|(neuron, _)| **neuron);
        let biases: Vec<SerializationBias> = self
            .output_bias
            .iter()
            .enumerate()
            .map(|(index, bias)| SerializationBias::new(self.inputs + index, bias))
            .chain(
                hidden
                    .into_iter()
                    .map(|(neuron, bias)| SerializationBias::new(*neuron, bias)),
            )
            .collect();
        let genes = self
            .sorted_genes()
            .into_iter()
            .map(|gene| SerializationGene {
                connection_type: gene.connection_type.to_int(),
                disabled: gene.disabled,
                input: SerializationNeuron::Id(gene.input),
                input_weight: num::cast(gene.input_weight).unwrap(),
                memory_weight: num::cast(gene.memory_weight).unwrap(),
                output: SerializationNeuron::Id(gene.output),
                reset_input_weight: num::cast(gene.reset_input_weight).unwrap(),
                reset_memory_weight: num::cast(gene.reset_memory_weight).unwrap(),
                update_input_weight: num::cast(gene.update_input_weight).unwrap(),
                update_memory_weight: num::cast(gene.update_memory_weight).unwrap(),
                ev_number: num::cast(gene.evolution_number),
            })
            .collect();
        let serialization = SerializationTopology {
            inputs: Some(self.inputs),
            outputs: Some(self.outputs),
            biases,
            genes,
        };
        write!(
            f,
            "{}",
            serde_json::to_string_pretty(&serialization).unwrap()
        )
    }
}

impl<T> PartialEq for Topology<T>
where
    T: Float + std::ops::AddAssign + Display + Send,
{
    fn eq(&self, other: &Self) -> bool {
        self.inputs == other.inputs
            && self.outputs == other.outputs
            && self.output_bias == other.output_bias
            && self.hidden_bias == other.hidden_bias
            && self.genes == other.genes
    }
}
