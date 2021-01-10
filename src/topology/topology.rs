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
use std::rc::Rc;

pub type GeneSmrtPtr<T> = Rc<RefCell<Gene<T>>>;

#[derive(Deserialize, Serialize)]
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
    genes_ev_number: HashMap<usize, GeneSmrtPtr<T>>,
}

impl<'a, T> Clone for Topology<T>
where
    T: Float + std::ops::AddAssign,
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
    T: Float + std::ops::AddAssign,
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
        2 * disjoints / n + w * 3
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
                    -1.,
                    1.,
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
            .map(|_| Bias::new_zero())
            .collect();
    }

    pub fn add_relationship(&mut self, gene_rc: GeneSmrtPtr<T>, init: bool) {
        let gene_cp = gene_rc.clone();
        // Drop refcell
        let (input, ev_number) = {
            let gene = &mut *gene_cp.borrow_mut();
            let input = gene.input.clone();
            let output = gene.output.clone();
            let ev_number = gene.evolution_number;
            if !gene.disabled && input.index + 1 > self.layers_sizes[input.layer as usize] {
                self.layers_sizes[input.layer as usize] = input.index + 1;
            }
            if !gene.disabled && !init && output.layer as usize == self.layers_sizes.len() {
                self.resize(output.layer as usize);
                gene.decrement_output();
            } else if !gene.disabled && output.index + 1 > self.layers_sizes[output.layer as usize]
            {
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
        self.genes_ev_number.insert(ev_number, gene_rc);
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
            let change_bias = rng.gen_range(0.0..1.);
            let normal = Normal::new(0.0, 0.1).unwrap();
            if change_bias < 0.9 {
                gene_and_bias.bias.bias_input += T::from(normal.sample(rng)).unwrap();
                gene_and_bias.bias.bias_update += T::from(normal.sample(rng)).unwrap();
                gene_and_bias.bias.bias_reset += T::from(normal.sample(rng)).unwrap();
            } else {
                gene_and_bias.bias = Bias::new_random(rng);
            }
            for gene in gene_and_bias.genes.iter_mut() {
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
                    0..((self.layers_sizes[output_layer as usize] + 1)
                        .min(self.max_per_layers as u8 + 1)),
                )
            };
            if output_index >= self.layers_sizes[output_layer as usize] {
                new_output = true
            }
        } else if (output_layer as usize) == self.layers_sizes.len() - 1 {
            output_index = rng.gen_range(0..self.layers_sizes[output_layer as usize]);
        } else {
            // Case where we create a new layer, output_index is 0
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
            self.disable_genes(input.clone(), output_cp.clone(), just_created.clone());
            let last_layer_size = self.layers_sizes.last().unwrap();
            let index = rng.gen_range(0..*last_layer_size);
            let output_of_output = Point::new((self.layers_sizes.len() - 1) as u8, index);
            if just_created.borrow().disabled {
                return;
            }
            let extra_gene = self.new_gene(
                &mut rng,
                just_created.borrow().output.clone(),
                output_of_output.clone(),
                &ev_number,
            );
            self.disable_genes(
                just_created.borrow().output.clone(),
                output_of_output,
                extra_gene,
            );
        }
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
            self.change_topology(&ev_number, &mut rng, false);
        } else if change_weights < 1. - proba.guaranteed_new_neuron || self.layers_sizes.len() <= 3
        {
            self.change_topology(&ev_number, &mut rng, true);
        } else {
            self.change_topology(&ev_number, &mut rng, false);
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
        self.add_relationship(new_gene.clone(), false);
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
                        || self.path_overrides(&input, &compared_output, 1)
                        || self.path_overrides(&output, &compared_output, 1)
                    {
                        let mut gene = cell.borrow_mut();
                        gene.disabled = true;
                    }
                }
            }
            None => {}
        }
    }

    fn path_overrides(&self, input: &Point, output: &Point, recursion: i8) -> bool {
        if recursion <= 0 {
            return false;
        }
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
                        if self.path_overrides(&compared_output, &output, recursion - 1) {
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
    T: Float + std::ops::AddAssign,
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
    T: Float + std::ops::AddAssign + Deserialize<'a> + Serialize,
{
    pub fn from_serde_string(serialized: &'a str) -> Topology<T> {
        let new_top: Topology<T> = serde_json::from_str(serialized).unwrap();
        new_top
    }

    pub fn to_serde_string(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
}
