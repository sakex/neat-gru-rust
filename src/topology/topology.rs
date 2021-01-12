use crate::topology::bias::Bias;
use crate::topology::bias_and_genes::BiasAndGenes;
use crate::topology::connection_type::ConnectionType;
use crate::topology::gene::{Gene, Point};
use crate::topology::mutation_probabilities::MutationProbabilities;
use crate::topology::serialization::{SerializationBias, SerializationGene, SerializationTopology};
use crate::train::evolution_number::EvNumber;
use num::traits::Float;
use numeric_literals::replace_numeric_literals;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Display;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

pub type GeneSmrtPtr<T> = Rc<RefCell<Gene<T>>>;

#[derive(Deserialize, Serialize)]
pub struct Topology<T>
where
    T: Float + std::ops::AddAssign + Display,
{
    max_layers: usize,
    max_per_layers: usize,
    last_result: T,
    result_before_mutation: T,
    pub layers_sizes: Vec<u8>,
    pub output_bias: Vec<Bias<T>>,
    pub genes_point: HashMap<Point, BiasAndGenes<T>>,
    genes_ev_number: HashMap<usize, GeneSmrtPtr<T>>,
}

pub type TopologySmrtPtr<T> = Arc<Mutex<Topology<T>>>;

unsafe impl<T> Send for Topology<T> where T: Float + std::ops::AddAssign + Display {}
unsafe impl<T> Sync for Topology<T> where T: Float + std::ops::AddAssign + Display {}

impl<'a, T> Clone for Topology<T>
where
    T: Float + std::ops::AddAssign + Display,
{
    fn clone(&self) -> Topology<T> {
        let genes_ev_number: HashMap<usize, GeneSmrtPtr<T>> = self
            .genes_ev_number
            .iter()
            .map(|(&ev_number, rc)| {
                let cell = &**rc;
                let ref_cell = &*cell.borrow();
                let cp = ref_cell.clone();
                (ev_number, Rc::new(RefCell::new(cp)))
            })
            .collect();

        let genes_point: HashMap<Point, BiasAndGenes<T>> = self
            .genes_point
            .iter()
            .map(|(point, bias_and_genes)| {
                let mut new_bg = BiasAndGenes::new(bias_and_genes.bias.clone());
                new_bg.genes = bias_and_genes
                    .genes
                    .iter()
                    .filter_map(|rc| {
                        let cell = &**rc;
                        let Gene {
                            evolution_number,
                            disabled,
                            ..
                        } = &*cell.borrow();
                        if !disabled {
                            let smart_ptr_cp =
                                genes_ev_number.get(evolution_number).unwrap().clone();
                            Some(smart_ptr_cp)
                        } else {
                            None
                        }
                    })
                    .collect();
                (point.clone(), new_bg)
            })
            .collect();

        Topology {
            max_layers: self.max_layers,
            max_per_layers: self.max_per_layers,
            last_result: self.last_result,
            result_before_mutation: self.result_before_mutation,
            layers_sizes: self.layers_sizes.clone(),
            output_bias: self.output_bias.clone(),
            genes_point,
            genes_ev_number,
        }
    }
}

impl<'a, T> Topology<T>
where
    T: Float + std::ops::AddAssign + Display,
{
    pub fn new(max_layers: usize, max_per_layers: usize) -> Topology<T> {
        Topology {
            max_layers,
            max_per_layers,
            last_result: T::zero(),
            result_before_mutation: T::zero(),
            layers_sizes: Vec::new(),
            output_bias: Vec::new(),
            genes_point: HashMap::new(),
            genes_ev_number: HashMap::new(),
        }
    }

    #[replace_numeric_literals(T::from(literal).unwrap())]
    pub fn delta_compatibility(top1: &Topology<T>, top2: &Topology<T>) -> T {
        // Disjoints = present in Gene1 but not Gene2
        let mut disjoints = T::zero();
        let mut common = T::zero();
        let mut w = T::zero();

        let one = T::one();
        for (ev_number, gene1) in top1.genes_ev_number.iter() {
            let cell = &**gene1;
            let gene1 = &*cell.borrow();
            match top2.genes_ev_number.get(ev_number) {
                Some(gene2) => {
                    let cell2 = &**gene2;
                    let gene2 = &*cell2.borrow();
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
                    disjoints = disjoints + one;
                }
            }
        }
        w = w / common;
        let size_1 = T::from(top1.genes_ev_number.len()).unwrap();
        let size_2 = T::from(top2.genes_ev_number.len()).unwrap();
        let larger = size_1.max(size_2);
        let n = if larger > 20 { larger - 20 } else { one };
        // Excess = present in gene2 but not gene1
        let excess = size_2 - common;
        let v = disjoints + excess;
        let ret = 12 * v / n + 3 * w;
        ret
    }

    pub fn new_random(
        rng: &mut ThreadRng,
        input_count: usize,
        output_count: usize,
        max_layers: usize,
        max_per_layers: usize,
        ev_number: &EvNumber,
    ) -> Topology<T> {
        let connections_per_input = (output_count as f64 / input_count as f64).ceil() as u32;
        let mut not_added: Vec<usize> = Vec::new();
        let mut output_index: usize = 0;
        for _i in 0..input_count {
            for _o in 0..connections_per_input {
                not_added.push(output_index);
                if output_index < output_count - 1 {
                    output_index += 1;
                } else {
                    output_index = 0;
                }
            }
        }
        let mut new_topology = Topology::new(max_layers, max_per_layers);
        new_topology.layers_sizes = vec![input_count as u8, output_count as u8];
        let mut not_added_it = 0;
        for i in 0..input_count {
            let input = Point::new(0, i as u8);
            for _j in 0..connections_per_input {
                let index = not_added[not_added_it];
                not_added_it += 1;
                let output = Point::new(1, index as u8);
                let gene = Rc::new(RefCell::new(Gene::new_random(
                    rng,
                    input.clone(),
                    output,
                    -1.,
                    1.,
                    &ev_number,
                )));
                new_topology.insert_gene(gene);
            }
        }
        new_topology.generate_output_bias(rng);
        new_topology
    }

    pub fn new_uniform(
        input_count: usize,
        output_count: usize,
        max_layers: usize,
        max_per_layers: usize,
        ev_number: &EvNumber,
    ) -> Topology<T> {
        let mut new_topology = Topology::new(max_layers, max_per_layers);
        new_topology.layers_sizes = vec![input_count as u8, output_count as u8];
        for i in 0..input_count {
            for j in 0..output_count {
                let input = Point::new(0u8, i as u8);
                let output = Point::new(1u8, j as u8);
                let gene = Rc::new(RefCell::new(Gene::new_uniform(input, output, &ev_number)));
                new_topology.insert_gene(gene);
            }
        }
        new_topology.uniform_output_bias();
        new_topology
    }

    fn generate_output_bias(&mut self, rng: &mut ThreadRng) {
        let last_layer_size = self.layers_sizes.last().unwrap();
        self.output_bias = (0..*last_layer_size)
            .into_iter()
            .map(|_| Bias::new_random(rng))
            .collect();
    }

    fn uniform_output_bias(&mut self) {
        let last_layer_size = self.layers_sizes.last().unwrap();
        self.output_bias = (0..*last_layer_size)
            .into_iter()
            .map(|_| Bias::new_zero())
            .collect();
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
            cp.mutate(&ev_number, &proba);
            new_topologies.push(Arc::new(Mutex::new(cp)));
        }
    }

    fn change_weights(&mut self, rng: &mut ThreadRng) {
        let normal = Normal::new(0.0, 0.1).unwrap();
        for (_ev_number, gene) in &self.genes_ev_number {
            let mut gene_cp = gene.borrow_mut();
            let change_weights = rng.gen_range(0.0..1.);
            if change_weights < 0.9 {
                gene_cp.input_weight += T::from(normal.sample(rng)).unwrap();
                gene_cp.memory_weight += T::from(normal.sample(rng)).unwrap();
                gene_cp.reset_input_weight += T::from(normal.sample(rng)).unwrap();
                gene_cp.update_input_weight += T::from(normal.sample(rng)).unwrap();
                gene_cp.reset_memory_weight += T::from(normal.sample(rng)).unwrap();
                gene_cp.update_memory_weight += T::from(normal.sample(rng)).unwrap();
            } else {
                gene_cp.random_reassign(rng);
            }
        }
        for (_point, gene_and_bias) in &mut self.genes_point {
            let change_bias = rng.gen_range(0.0..1.);
            if change_bias < 0.9 {
                gene_and_bias.bias.bias_input += T::from(normal.sample(rng)).unwrap();
                gene_and_bias.bias.bias_update += T::from(normal.sample(rng)).unwrap();
                gene_and_bias.bias.bias_reset += T::from(normal.sample(rng)).unwrap();
            } else {
                gene_and_bias.bias = Bias::new_random(rng);
            }
        }
        for bias in &mut self.output_bias {
            let change_bias = rng.gen_range(0.0..1.);
            let normal = Normal::new(0.0, 0.1).unwrap();
            if change_bias < 0.8 {
                bias.bias_input += T::from(normal.sample(rng)).unwrap();
                bias.bias_update += T::from(normal.sample(rng)).unwrap();
                bias.bias_reset += T::from(normal.sample(rng)).unwrap();
            }
        }
    }

    #[inline]
    pub fn insert_gene(&mut self, gene: GeneSmrtPtr<T>) {
        let (input, ev_number) = {
            let gene = &*gene.borrow();
            (gene.input.clone(), gene.evolution_number)
        };
        match self.genes_point.get_mut(&input) {
            Some(found) => {
                found.genes.push(gene.clone());
                found.genes.sort();
            }
            None => {
                let bias = Bias::new_zero();
                let mut bias_and_genes: BiasAndGenes<T> = BiasAndGenes::new(bias);
                bias_and_genes.genes = vec![gene.clone()];
                self.genes_point.insert(input.clone(), bias_and_genes);
            }
        }
        self.genes_ev_number.insert(ev_number, gene);
    }

    #[inline]
    fn shift_right_one_layer(&mut self, layer: u8) {
        self.layers_sizes.insert(layer as usize, 1);

        for (_ev, gene_rc) in &mut self.genes_ev_number {
            let mut gene = &mut *gene_rc.borrow_mut();
            if gene.input.layer >= layer {
                gene.input.layer += 1;
            }
            if gene.output.layer >= layer {
                gene.output.layer += 1;
            }
        }

        self.genes_point = self
            .genes_point
            .iter_mut()
            .map(|(point, b_and_g)| {
                if point.layer >= layer {
                    let point_cp = Point::new(point.layer + 1, point.index);
                    (point_cp, b_and_g.clone())
                } else {
                    (point.clone(), b_and_g.clone())
                }
            })
            .collect();
    }

    /// Find random input and random output and adds a connection in the middle
    #[inline]
    fn add_node(&mut self, ev_number: &EvNumber, rng: &mut ThreadRng) {
        let non_disabled_connections = self
            .genes_ev_number
            .iter()
            .filter_map(|(_ev, gene_rc)| {
                if !gene_rc.borrow().disabled {
                    Some(gene_rc.clone())
                } else {
                    None
                }
            })
            .collect::<Vec<GeneSmrtPtr<T>>>();

        let gene_to_split_index = rng.gen_range(0..non_disabled_connections.len());
        let gene_to_split = &non_disabled_connections[gene_to_split_index];
        let mut original_gene = {
            let mut gene = &mut *gene_to_split.borrow_mut();
            gene.disabled = true;
            gene.clone()
        };

        // If there is a layer between input and output, just add new node to the last index of the
        // layer just after input
        // Otherwise, we create a gene in the middle
        let output_of_input = if original_gene.output.layer - original_gene.input.layer >= 2 {
            let output_of_input = Point::new(
                original_gene.input.layer + 1,
                self.layers_sizes[original_gene.input.layer as usize + 1],
            );
            self.layers_sizes[original_gene.input.layer as usize + 1] += 1;
            output_of_input
        } else {
            let output_of_input = Point::new(original_gene.input.layer + 1, 0);
            self.shift_right_one_layer(original_gene.input.layer + 1);
            original_gene.output.layer += 1;
            output_of_input
        };
        let (middle_gene, end_gene) = original_gene.split(output_of_input, &ev_number);
        self.insert_gene(Rc::new(RefCell::new(middle_gene)));
        self.insert_gene(Rc::new(RefCell::new(end_gene)));
    }

    #[inline]
    fn add_connection(&mut self, ev_number: &EvNumber, mut rng: &mut ThreadRng) {
        let max_layer = self.layers_sizes.len().min(self.max_layers);
        let input_layer = if self.layers_sizes.len() > 2 {
            rng.gen_range(0..(max_layer - 2)) as u8
        } else {
            0
        };
        let input_index: u8 = rng.gen_range(0..self.layers_sizes[input_layer as usize]);
        let output_layer: u8 = rng.gen_range((input_layer + 1)..max_layer as u8);

        let output_index = if (output_layer as usize) < self.layers_sizes.len() - 1 {
            rng.gen_range(
                0..((self.layers_sizes[output_layer as usize]).min(self.max_per_layers as u8)),
            )
        } else {
            // If on output layer
            rng.gen_range(0..self.layers_sizes[output_layer as usize])
        };
        let input = Point::new(input_layer, input_index);
        let output = Point::new(output_layer, output_index);
        let just_created = self.new_gene(&mut rng, input.clone(), output.clone(), &ev_number);
        self.disable_genes(input.clone(), output.clone(), just_created);
    }

    /*#[allow(dead_code)]
    fn delete_neuron(&mut self, rng: &mut ThreadRng) {
        let input_layer = rng.gen_range(1..self.layers_sizes.len() - 2) as u8;
        let input_index: u8 = rng.gen_range(0..self.layers_sizes[input_layer as usize]);
        let input = Point::new(input_layer, input_index);
        self.remove_neuron(&input);
    }*/

    pub fn mutate(&mut self, ev_number: &EvNumber, proba: &MutationProbabilities) {
        let mut rng = thread_rng();
        let change_weights = rng.gen_range(0.0..1.);
        if change_weights < proba.change_weights {
            self.change_weights(&mut rng);
        } else if change_weights < 1. - proba.guaranteed_new_neuron - proba.delete_neuron {
            self.add_connection(&ev_number, &mut rng);
        } else if change_weights < 1. - proba.guaranteed_new_neuron || self.layers_sizes.len() <= 3
        {
            self.add_node(&ev_number, &mut rng);
        } else {
            self.add_connection(&ev_number, &mut rng);
            // self.delete_neuron(&mut rng);
        }
        loop {
            let dont_have_outputs: Vec<GeneSmrtPtr<T>> = self
                .genes_point
                .iter()
                .filter_map(|(input, gene_and_bias)| {
                    if input.layer != 0
                        && (gene_and_bias.genes.is_empty()
                            || gene_and_bias
                                .genes
                                .iter()
                                .all(|gene_rc| gene_rc.borrow().disabled))
                    {
                        Some(gene_and_bias.genes.first().unwrap().clone())
                    } else {
                        None
                    }
                })
                .collect();
            let dont_have_inputs: Vec<GeneSmrtPtr<T>> = self
                .genes_point
                .iter()
                .map(|(input, gene_and_bias)| {
                    if input.layer != 0 {
                        gene_and_bias
                            .genes
                            .iter()
                            .filter(|gene| {
                                let borrow = gene.borrow();
                                let input = borrow.input.clone();
                                !borrow.disabled && !self.check_has_inputs(&input)
                            })
                            .collect()
                    } else {
                        Vec::new()
                    }
                })
                .flatten()
                .cloned()
                .collect();
            if dont_have_outputs.is_empty() && dont_have_inputs.is_empty() {
                break;
            }
            for gene in &dont_have_outputs {
                if !gene.borrow().disabled {
                    self.remove_neuron(&gene);
                }
            }
            for gene in &dont_have_inputs {
                if !gene.borrow().disabled {
                    self.remove_neuron(&gene);
                }
            }
        }
    }

    fn new_gene(
        &mut self,
        rng: &mut ThreadRng,
        input: Point,
        output: Point,
        ev_number: &EvNumber,
    ) -> GeneSmrtPtr<T> {
        let new_gene = Rc::new(RefCell::new(Gene::new_random(
            rng, input, output, -1., 1., &ev_number,
        )));
        self.insert_gene(new_gene.clone());
        new_gene
    }

    fn remove_neuron(&mut self, gene_rc: &GeneSmrtPtr<T>) {
        // get bias_and_gene
        let input = { gene_rc.borrow().input.clone() };
        let bias_and_gene = match self.genes_point.remove(&input) {
            None => {
                return;
            }
            Some(v) => v,
        };
        // disable all outputs from this neuron
        for gene_rc in &bias_and_gene.genes {
            let mut gene = gene_rc.borrow_mut();
            gene.disabled = true;
        }
        // Lower layer size
        self.layers_sizes[input.layer as usize] -= 1;
        let is_removed_layer = self.layers_sizes[input.layer as usize] == 0;
        if is_removed_layer {
            self.layers_sizes.retain(|&v| v != 0);
        };
        // Adjust all genes with inputs or outputs on the layer after the index
        // If the layer only had one neuron, do the same on further layers
        self.genes_point = self
            .genes_point
            .iter_mut()
            .map(|(point, bias_and_gene)| {
                let mut point = point.clone();
                if point.layer == input.layer && point.index > input.index {
                    point.index -= 1;
                }
                if is_removed_layer && point.layer > input.layer {
                    point.layer -= 1;
                }
                for gene_rc in &bias_and_gene.genes {
                    let mut gene = gene_rc.borrow_mut();
                    if gene.output == input {
                        gene.disabled = true;
                        continue;
                    }
                    if gene.input.layer == input.layer && gene.input.index > input.index {
                        gene.input.index -= 1;
                    }
                    if gene.output.layer == input.layer && gene.output.index > input.index {
                        gene.output.index -= 1;
                    }
                    if is_removed_layer {
                        if gene.input.layer > input.layer {
                            gene.input.layer -= 1;
                        }
                        if gene.output.layer > input.layer {
                            gene.output.layer -= 1;
                        }
                    }
                }
                (point, bias_and_gene.clone())
            })
            .collect();
    }

    /// Returns true if no Gene has the given output
    ///
    /// # Argument
    ///
    /// `output` - The point to check if any gene points to it
    fn check_has_inputs(&self, input: &Point) -> bool {
        self.genes_point.iter().any(|(_point, b_and_c)| {
            b_and_c.genes.iter().any(|gene_rc| {
                let gene = gene_rc.borrow();
                !gene.disabled && gene.output == *input
            })
        })
    }

    fn disable_genes(&mut self, input: Point, output: Point, last: GeneSmrtPtr<T>) {
        match self.genes_point.get(&input) {
            Some(found) => {
                let genes = &found.genes;
                for gene_rc in genes {
                    if Rc::ptr_eq(gene_rc, &last) {
                        continue;
                    }
                    let cell = &**gene_rc;
                    let compared_output = {
                        let gene = cell.borrow();
                        if gene.disabled {
                            continue;
                        }
                        gene.output.clone()
                    };
                    if output == compared_output
                        || self.path_overrides(&input, &compared_output, &last, 1)
                        || self.path_overrides(&output, &compared_output, &last, 1)
                    {
                        let mut gene = cell.borrow_mut();
                        gene.disabled = true;
                    }
                }
            }
            None => {}
        }
    }

    fn path_overrides(
        &self,
        input: &Point,
        output: &Point,
        last: &GeneSmrtPtr<T>,
        recursion: i8,
    ) -> bool {
        if recursion <= 0 {
            return false;
        }
        match self.genes_point.get(&input) {
            Some(found) => {
                let genes = &found.genes;
                for gene_rc in genes {
                    if Rc::ptr_eq(gene_rc, last) {
                        continue;
                    }
                    let cell = &**gene_rc;
                    let gene = &*cell.borrow();
                    if gene.disabled {
                        continue;
                    }
                    let compared_output = &gene.output;
                    if *compared_output == *output {
                        return true;
                    } else if compared_output.layer < output.layer {
                        if self.path_overrides(&compared_output, &output, &last, recursion - 1) {
                            return true;
                        }
                    }
                }
                return false;
            }
            None => false,
        }
    }

    pub fn from_string(serialized: &str) -> Topology<T> {
        let serialization: SerializationTopology =
            SerializationTopology::from_string(serialized).unwrap();
        let mut layers_sizes = Vec::new();
        let mut output_bias: Vec<Bias<T>> = Vec::new();
        let mut genes_point = HashMap::new();
        let genes_ev_number = HashMap::new();

        for ser_bias in &serialization.biases {
            let input = Point::new(ser_bias.neuron.0, ser_bias.neuron.1);
            if input.layer >= layers_sizes.len() as u8 {
                layers_sizes.resize((input.layer + 1) as usize, 0);
                layers_sizes[input.layer as usize] = input.index + 1;
            } else {
                layers_sizes[input.layer as usize] =
                    layers_sizes[input.layer as usize].max(input.index + 1);
            }
        }

        output_bias.resize(*layers_sizes.last().unwrap() as usize, Bias::new_zero());
        for ser_bias in &serialization.biases {
            let input = Point::new(ser_bias.neuron.0, ser_bias.neuron.1);
            if input.layer < (layers_sizes.len() - 1) as u8 {
                let bias_and_gene = BiasAndGenes::new(ser_bias.bias.cast());
                genes_point.insert(input, bias_and_gene);
            } else {
                output_bias[input.index as usize] = ser_bias.bias.cast();
            }
        }

        for gene in &serialization.genes {
            let input = Point::new(gene.input.0, gene.input.1);
            let output = Point::new(gene.output.0, gene.output.1);
            let new_gene = Rc::new(RefCell::new(Gene::new(
                input.clone(),
                output,
                num::cast(gene.input_weight).unwrap(),
                num::cast(gene.memory_weight).unwrap(),
                num::cast(gene.reset_input_weight).unwrap(),
                num::cast(gene.update_input_weight).unwrap(),
                num::cast(gene.reset_memory_weight).unwrap(),
                num::cast(gene.update_memory_weight).unwrap(),
                0,
                ConnectionType::from_int(gene.connection_type),
                gene.disabled,
            )));

            if gene.disabled {
                continue;
            }

            match genes_point.get_mut(&input) {
                Some(b_and_g) => b_and_g.genes.push(new_gene),
                None => panic!("Gene doesn't have a neuron"),
            }
        }

        Topology {
            max_layers: layers_sizes.len(),
            max_per_layers: *layers_sizes.iter().max().unwrap() as usize,
            last_result: T::zero(),
            result_before_mutation: T::zero(),
            layers_sizes,
            output_bias,
            genes_point,
            genes_ev_number,
        }
    }

    pub fn to_string(&self) -> String {
        let mut biases: Vec<SerializationBias> = self
            .genes_point
            .iter()
            .map(|(point, b_and_g)| SerializationBias::new(point.clone(), b_and_g.bias.clone()))
            .collect();
        let last_layer = self.layers_sizes.len() - 1;
        let mut output_biases: Vec<SerializationBias> = self
            .output_bias
            .iter()
            .enumerate()
            .map(|(index, bias)| {
                SerializationBias::new(Point::new(last_layer as u8, index as u8), bias.clone())
            })
            .collect();
        biases.append(&mut output_biases);
        let genes = self
            .genes_point
            .iter()
            .map(|(_point, gene)| {
                gene.genes
                    .iter()
                    .map(|gene| {
                        let cell = &**gene;
                        let gene = &*cell.borrow();
                        SerializationGene::new(
                            gene.connection_type.to_int(),
                            gene.disabled,
                            (gene.input.layer, gene.input.index),
                            num::cast(gene.input_weight).unwrap(),
                            num::cast(gene.memory_weight).unwrap(),
                            (gene.output.layer, gene.output.index),
                            num::cast(gene.reset_input_weight).unwrap(),
                            num::cast(gene.reset_memory_weight).unwrap(),
                            num::cast(gene.update_input_weight).unwrap(),
                            num::cast(gene.update_memory_weight).unwrap(),
                        )
                    })
                    .collect::<Vec<SerializationGene>>()
            })
            .flatten()
            .collect();
        let serialization = SerializationTopology::new(biases, genes);
        serde_json::to_string(&serialization).unwrap()
    }
}

impl<'a, T> PartialEq for Topology<T>
where
    T: Float + std::ops::AddAssign + Display,
{
    fn eq(&self, other: &Self) -> bool {
        if self.layers_sizes.len() != other.layers_sizes.len()
            || self.genes_ev_number.len() != other.genes_ev_number.len()
            || self.genes_point.len() != other.genes_point.len()
        {
            return false;
        }
        if self
            .layers_sizes
            .iter()
            .zip(other.layers_sizes.iter())
            .any(|(&a, &b)| a != b)
        {
            return false;
        }
        for (ev_number, gene) in &self.genes_ev_number {
            match other.genes_ev_number.get(ev_number) {
                Some(gene2) => {
                    let gene = gene.borrow();
                    let gene2 = gene2.borrow();
                    if *gene != *gene2 {
                        return false;
                    }
                }
                None => {
                    return false;
                }
            }
        }
        for (point, b_and_g) in &self.genes_point {
            match other.genes_point.get(point) {
                Some(b_and_g2) => {
                    if b_and_g.bias != b_and_g2.bias {
                        return false;
                    }
                    if b_and_g
                        .genes
                        .iter()
                        .zip(b_and_g2.genes.iter())
                        .any(|(gene1, gene2)| {
                            let gene1 = &*gene1.borrow();
                            let gene2 = &*gene2.borrow();
                            *gene1 != *gene2
                        })
                    {
                        return false;
                    }
                }
                None => {
                    return false;
                }
            }
        }
        true
    }
}

impl<'a, T> Topology<T>
where
    T: Float + std::ops::AddAssign + Deserialize<'a> + Serialize + Display,
{
    pub fn from_serde_string(serialized: &'a str) -> Topology<T> {
        let new_top: Topology<T> = serde_json::from_str(serialized).unwrap();
        new_top
    }

    pub fn to_serde_string(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
}
