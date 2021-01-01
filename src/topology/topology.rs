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
use serde::export::fmt::Display;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

pub struct Topology<T>
where
    T: Float + std::ops::AddAssign,
{
    max_layers: usize,
    max_per_layers: usize,
    last_result: T,
    result_before_mutation: T,
    pub layers_sizes: Vec<u8>,
    pub output_bias: Vec<Bias<T>>,
    pub genes_point: HashMap<Point, BiasAndGenes<T>>,
    genes_ev_number: HashMap<usize, Rc<RefCell<Gene<T>>>>,
}

impl<T> Clone for Topology<T>
where
    T: Float + std::ops::AddAssign + Display,
{
    fn clone(&self) -> Topology<T> {
        let genes_point: HashMap<Point, BiasAndGenes<T>> = self
            .genes_point
            .iter()
            .map(|(point, bias_and_genes)| {
                let mut new_bg = BiasAndGenes::new(bias_and_genes.bias.clone());
                new_bg.genes = bias_and_genes
                    .genes
                    .iter()
                    .map(|rc| {
                        let cell = &**rc;
                        let ref_cell = &*cell.borrow();
                        let cp = ref_cell.clone();
                        Rc::new(RefCell::new(cp))
                    })
                    .collect();
                (point.clone(), new_bg)
            })
            .collect();

        let genes_ev_number: HashMap<usize, Rc<RefCell<Gene<T>>>> = self
            .genes_ev_number
            .iter()
            .map(|(&ev_number, rc)| {
                let cell = &**rc;
                let ref_cell = &*cell.borrow();
                let cp = ref_cell.clone();
                (ev_number, Rc::new(RefCell::new(cp)))
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

impl<T> Topology<T>
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
        let n = if size_1 >= 20 || size_2 >= 20 {
            size_1.max(size_2)
        } else {
            one
        };
        disjoints = disjoints + size_1 - common;
        12 * disjoints / n + w * 3
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
        new_topology.set_layers(2);
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
                    -1.0,
                    1.0,
                    &ev_number,
                )));
                new_topology.add_relationship(gene, true);
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
        new_topology.set_layers(2);
        for i in 0..input_count {
            for j in 0..output_count {
                let input = Point::new(0u8, i as u8);
                let output = Point::new(1u8, j as u8);
                let gene = Rc::new(RefCell::new(Gene::new_uniform(input, output, &ev_number)));
                new_topology.add_relationship(gene, true);
            }
        }
        new_topology.uniform_output_bias();
        new_topology
    }

    pub fn set_layers(&mut self, layers: usize) {
        self.layers_sizes.resize(layers, 1);
        self.layers_sizes[layers - 1] = self.layers_sizes[layers - 2];
        self.layers_sizes[layers - 2] = 1;
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
            .map(|_| Bias::new_uniform())
            .collect();
    }

    pub fn add_relationship(&mut self, gene: Rc<RefCell<Gene<T>>>, init: bool) {
        let gene_cp = gene.clone();
        // Drop refcell
        let (input, ev_number) = {
            let cell = &*gene;
            let gene = &mut *cell.borrow_mut();
            let input = gene.input.clone();
            let output = gene.output.clone();
            let ev_number = gene.evolution_number;
            if input.index + 1 > self.layers_sizes[input.layer as usize] {
                self.layers_sizes[input.layer as usize] = input.index + 1;
            }
            if !init && output.layer as usize == self.layers_sizes.len() {
                self.resize(output.layer as usize);
                gene.decrement_output();
            } else if output.index + 1 > self.layers_sizes[output.layer as usize] {
                self.layers_sizes[output.layer as usize] = output.index + 1;
            }
            (input, ev_number)
        };
        let mut rng = thread_rng();
        match self.genes_point.get_mut(&input) {
            Some(found) => {
                found.genes.push(gene_cp.clone());
                found.genes.sort();
            }
            None => {
                let bias = Bias::new_random(&mut rng);
                let mut bias_and_genes: BiasAndGenes<T> = BiasAndGenes::new(bias);
                bias_and_genes.genes = vec![gene_cp.clone()];
                self.genes_point.insert(input.clone(), bias_and_genes);
            }
        }
        self.genes_ev_number.insert(ev_number, gene_cp);
    }

    fn resize(&mut self, layers: usize) {
        for (_point, bias_and_gene) in self.genes_point.iter() {
            for gene_rc in &bias_and_gene.genes {
                let mut gene = (&**gene_rc).borrow_mut();
                if gene.output.layer == (layers - 1) as u8 {
                    gene.output.layer = layers as u8
                }
            }
        }
        self.set_layers(layers + 1);
    }

    pub fn set_last_result(&mut self, result: T) {
        self.last_result = result;
    }

    pub fn get_last_result(&self) -> T {
        self.last_result
    }

    pub fn new_generation(
        &self,
        new_topologies: &mut Vec<Rc<RefCell<Topology<T>>>>,
        ev_number: &EvNumber,
        reproduction_count: usize,
        proba: &MutationProbabilities,
    ) {
        for _ in 0..reproduction_count {
            let mut cp = self.clone();
            cp.mutate(&ev_number, &proba);
            new_topologies.push(Rc::new(RefCell::new(cp)));
        }
    }

    fn change_weights(&mut self, rng: &mut ThreadRng) {
        for (_point, gene_and_bias) in self.genes_point.iter_mut() {
            let change_bias = rng.gen_range(0.0..1.0);
            let normal = Normal::new(0.0, 0.1).unwrap();
            if change_bias < 0.8 {
                gene_and_bias.bias.bias_input += T::from(normal.sample(rng)).unwrap();
                gene_and_bias.bias.bias_update += T::from(normal.sample(rng)).unwrap();
                gene_and_bias.bias.bias_reset += T::from(normal.sample(rng)).unwrap();
            }
            for gene in gene_and_bias.genes.iter_mut() {
                let mut gene_cp = gene.borrow_mut();
                let change_weights = rng.gen_range(0.0..1.0);
                if change_weights < 0.8 {
                    gene_cp.input_weight += T::from(normal.sample(rng)).unwrap();
                    gene_cp.memory_weight += T::from(normal.sample(rng)).unwrap();
                    gene_cp.reset_input_weight += T::from(normal.sample(rng)).unwrap();
                    gene_cp.update_input_weight += T::from(normal.sample(rng)).unwrap();
                    gene_cp.reset_memory_weight += T::from(normal.sample(rng)).unwrap();
                    gene_cp.update_memory_weight += T::from(normal.sample(rng)).unwrap();
                }
            }
        }
    }

    fn change_topology(
        &mut self,
        ev_number: &EvNumber,
        mut rng: &mut ThreadRng,
        guaranteed_new_neuron: bool,
    ) {
        let mut new_output = false;
        let max_layer = self.layers_sizes.len().min(self.max_layers);
        let input_layer = if self.layers_sizes.len() > 2 {
            rng.gen_range(0..(max_layer - 2)) as u8
        } else {
            0
        };
        let input_index: u8 = rng.gen_range(0..self.layers_sizes[input_layer as usize]);
        let output_layer: u8 = rng.gen_range((input_layer + 1)..(max_layer + 1) as u8);
        let mut output_index: u8 = 0;

        if (output_layer as usize) < self.layers_sizes.len() - 1 {
            output_index = if guaranteed_new_neuron {
                (self.layers_sizes[output_layer as usize]).min(self.max_per_layers as u8)
            } else {
                rng.gen_range(
                    0..(self.layers_sizes[output_layer as usize].min(self.max_per_layers as u8)
                        + 1),
                )
            };
            if output_index >= self.layers_sizes[output_layer as usize] {
                new_output = true
            }
        } else if (output_layer as usize) == self.layers_sizes.len() - 1 {
            output_index = rng.gen_range(0..self.layers_sizes[output_layer as usize]);
        } else {
            // if output_index == layers
            new_output = true;
        }
        let input = Point::new(input_layer, input_index);
        let output = Point::new(output_layer, output_index);
        if !new_output {
            let just_created = self.new_gene(&mut rng, input.clone(), output.clone(), &ev_number);
            self.disable_genes(input.clone(), output.clone(), just_created);
        } else {
            let mut output_cp = output.clone();
            output_cp.layer -= 1;
            let just_created = self.new_gene(&mut rng, input.clone(), output.clone(), &ev_number);
            self.disable_genes(input.clone(), output_cp.clone(), just_created);
            let last_layer_size = self.layers_sizes.last().unwrap();
            let index = rng.gen_range(0..*last_layer_size);
            let output_of_output = Point::new((self.layers_sizes.len() - 1) as u8, index);
            self.new_gene(&mut rng, output_cp.clone(), output_of_output, &ev_number);
        }
    }

    pub fn mutate(&mut self, ev_number: &EvNumber, proba: &MutationProbabilities) {
        let mut rng = thread_rng();
        let change_weights = rng.gen_range(0.0..1.0);
        if change_weights < proba.change_weights {
            self.change_weights(&mut rng);
        } else if change_weights < 1.0 - proba.guaranteed_new_neuron {
            self.change_topology(&ev_number, &mut rng, false);
        } else {
            self.change_topology(&ev_number, &mut rng, true);
        }
    }

    fn new_gene(
        &mut self,
        rng: &mut ThreadRng,
        input: Point,
        output: Point,
        ev_number: &EvNumber,
    ) -> Rc<RefCell<Gene<T>>> {
        let new_gene = Rc::new(RefCell::new(Gene::new_random(
            rng, input, output, -1.0, 1.0, &ev_number,
        )));
        self.add_relationship(new_gene.clone(), false);
        new_gene
    }

    fn remove_no_inputs(&mut self, gene: Rc<RefCell<Gene<T>>>) {
        let output = { gene.borrow().output.clone() };
        let mut vec_check_disabled = Vec::new();
        if !self.check_no_inputs(&output) {
            let bias_and_gene = match self.genes_point.get(&output) {
                None => {
                    return;
                }
                Some(v) => v,
            };
            for gene_rc in &bias_and_gene.genes {
                vec_check_disabled.push(gene_rc.clone());
                let gene = gene_rc.borrow();
                self.genes_ev_number.remove(&gene.evolution_number);
            }
            self.genes_point.remove(&output);
            self.layers_sizes[output.layer as usize] -= 1;
            let is_removed_layer = self.layers_sizes[output.layer as usize] == 0;
            if is_removed_layer {
                self.layers_sizes.retain(|&v| v != 0);
            };
            self.genes_point = self
                .genes_point
                .iter_mut()
                .map(|(point, bias_and_gene)| {
                    let mut point = point.clone();
                    if point.layer == output.layer && point.index > output.index {
                        point.index -= 1;
                    }
                    if is_removed_layer && point.layer > output.layer {
                        point.layer -= 1;
                    }
                    for gene_rc in &bias_and_gene.genes {
                        if Rc::ptr_eq(gene_rc, &gene) {
                            continue;
                        }
                        let mut gene = gene_rc.borrow_mut();
                        if gene.input.layer == output.layer && gene.input.index > output.index {
                            gene.input.index -= 1;
                        }
                        if gene.output.layer == output.layer && gene.output.index > output.index {
                            gene.output.index -= 1;
                        }
                        if is_removed_layer {
                            if gene.input.layer > output.layer {
                                gene.input.layer -= 1;
                            }
                            if gene.output.layer > output.layer {
                                gene.output.layer -= 1;
                            }
                        }
                    }
                    (point, bias_and_gene.clone())
                })
                .collect();
        }
        for gene_rc in &vec_check_disabled {
            self.remove_no_inputs(gene_rc.clone());
        }
    }

    fn check_no_inputs(&self, output: &Point) -> bool {
        self.genes_point.iter().any(|(_point, gene_and_bias)| {
            gene_and_bias.genes.iter().any(|gene_rc| {
                let gene = gene_rc.borrow();
                !gene.disabled && gene.output == *output
            })
        })
    }

    fn disable_genes(&mut self, input: Point, output: Point, last: Rc<RefCell<Gene<T>>>) {
        let mut disabled_genes = Vec::new();
        match self.genes_point.get(&input) {
            Some(found) => {
                let genes = &found.genes;
                for gene_rc in genes {
                    if Rc::ptr_eq(gene_rc, &last) {
                        continue;
                    }
                    let cloned_rc = gene_rc.clone();
                    let cell = &**gene_rc;
                    let compared_output = {
                        let gene = cell.borrow();
                        if gene.disabled {
                            continue;
                        }
                        gene.output.clone()
                    };
                    if output == compared_output
                        || self.path_overrides(&compared_output, &output)
                        || self.path_overrides(&output, &compared_output)
                    {
                        let mut gene = cell.borrow_mut();
                        gene.disabled = true;
                        disabled_genes.push(cloned_rc);
                    }
                }
            }
            None => {}
        }
        for gene_rc in disabled_genes {
            self.remove_no_inputs(gene_rc.clone());
        }
    }

    fn path_overrides(&self, input: &Point, output: &Point) -> bool {
        match self.genes_point.get(&input) {
            Some(found) => {
                let genes = &found.genes;
                for gene_rc in genes {
                    let cell = &**gene_rc;
                    let gene = &*cell.borrow();
                    if gene.disabled {
                        continue;
                    }
                    let compared_output = &gene.output;
                    if *compared_output == *output {
                        return true;
                    } else if compared_output.layer < output.layer {
                        if self.path_overrides(&compared_output, &output) {
                            return true;
                        }
                    }
                }
                return false;
            }
            None => false,
        }
    }

    fn set_bias(&mut self, neuron: Point, bias: Bias<T>) {
        if neuron.layer as usize != self.layers_sizes.len() - 1 {
            match self.genes_point.get_mut(&neuron) {
                Some(found) => found.bias = bias,
                None => panic!(
                    "Error in serialization, Neuron {} {} doesn't exist",
                    neuron.layer, neuron.index
                ),
            }
        } else {
            if self.output_bias.len() != *self.layers_sizes.last().unwrap() as usize {
                self.output_bias.resize(
                    *self.layers_sizes.last().unwrap() as usize,
                    Bias::new_zero(),
                );
            }
            self.output_bias[neuron.index as usize] = bias;
        }
    }

    pub fn from_string(serialized: &str) -> Topology<T> {
        let serialization: SerializationTopology =
            SerializationTopology::from_string(serialized).unwrap();
        let mut max_layers = 0u8;
        let gene_vec: Vec<Rc<RefCell<Gene<T>>>> = serialization
            .genes
            .iter()
            .map(|ser_gene| {
                let input = Point::new(ser_gene.input.0, ser_gene.input.1);
                let output = Point::new(ser_gene.output.0, ser_gene.output.1);
                if output.layer > max_layers {
                    max_layers = output.layer;
                }
                let gene = Rc::new(RefCell::new(Gene::new(
                    input,
                    output,
                    T::from(ser_gene.input_weight).unwrap(),
                    T::from(ser_gene.memory_weight).unwrap(),
                    T::from(ser_gene.reset_input_weight).unwrap(),
                    T::from(ser_gene.update_input_weight).unwrap(),
                    T::from(ser_gene.reset_memory_weight).unwrap(),
                    T::from(ser_gene.update_memory_weight).unwrap(),
                    0,
                    ConnectionType::from_int(ser_gene.connection_type),
                    ser_gene.disabled,
                )));
                gene
            })
            .collect();
        let mut new_top = Topology::new(max_layers as usize, max_layers as usize);
        new_top.set_layers((max_layers + 1).into());
        for gene in gene_vec {
            new_top.add_relationship(gene.clone(), true);
        }
        for bias in serialization.biases {
            let neuron = Point::new(bias.neuron.0, bias.neuron.1);
            new_top.set_bias(
                neuron,
                Bias::new(
                    T::from(bias.bias.bias_input).unwrap(),
                    T::from(bias.bias.bias_update).unwrap(),
                    T::from(bias.bias.bias_reset).unwrap(),
                ),
            );
        }
        new_top
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
