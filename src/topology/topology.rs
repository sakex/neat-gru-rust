use crate::topology::bias::Bias;
use crate::topology::bias_and_genes::BiasAndGenes;
use crate::topology::connection_type::ConnectionType;
use crate::topology::gene::{Gene, Point};
use crate::topology::serialization::{SerializationBias, SerializationGene, SerializationTopology};
use crate::train::evolution_number::EvNumber;
use num::traits::Float;
use rand::prelude::ThreadRng;
use rand::{thread_rng, Rng};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

pub struct Topology<T>
    where
        T: Float,
{
    pub layers: usize,
    max_layers: usize,
    max_per_layers: usize,
    last_result: T,
    best_historical_result: T,
    result_before_mutation: T,
    pub layers_sizes: Vec<u8>,
    pub output_bias: Vec<Bias<T>>,
    pub genes_point: HashMap<Point, BiasAndGenes<T>>,
    genes_ev_number: HashMap<usize, Rc<RefCell<Gene<T>>>>,
}

impl<T> Clone for Topology<T>
    where
        T: Float,
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
            layers: self.layers,
            max_layers: self.max_layers,
            max_per_layers: self.max_per_layers,
            last_result: self.last_result,
            best_historical_result: self.best_historical_result,
            result_before_mutation: self.result_before_mutation,
            layers_sizes: self.layers_sizes.clone(),
            output_bias: Vec::new(),
            genes_point,
            genes_ev_number,
        }
    }
}

impl<T> Topology<T>
    where
        T: Float,
{
    pub fn new(max_layers: usize, max_per_layers: usize) -> Topology<T> {
        Topology {
            layers: 0,
            max_layers,
            max_per_layers,
            last_result: T::from(0).unwrap(),
            best_historical_result: T::from(0).unwrap(),
            result_before_mutation: T::from(0).unwrap(),
            layers_sizes: Vec::new(),
            output_bias: Vec::new(),
            genes_point: HashMap::new(),
            genes_ev_number: HashMap::new(),
        }
    }

    pub fn delta_compatibility(top1: &Topology<T>, top2: &Topology<T>) -> T {
        let mut disjoints = T::from(0).unwrap();
        let mut common = T::from(0).unwrap();
        let mut w = T::from(0).unwrap();

        let one = T::from(1).unwrap();
        for (ev_number, gene1) in top1.genes_ev_number.iter() {
            let cell = &**gene1;
            let gene1 = &*cell.borrow();
            match top2.genes_ev_number.get(ev_number) {
                Some(gene2) => {
                    let cell2 = &**gene2;
                    let gene2 = &*cell2.borrow();
                    common = common + one;
                    w = w
                        + (gene1.input_weight - gene2.input_weight).abs()
                        + (gene1.memory_weight - gene2.memory_weight).abs()
                        + (gene1.reset_input_weight - gene2.reset_input_weight).abs()
                        + (gene1.update_input_weight - gene2.update_input_weight).abs()
                        + (gene1.reset_memory_weight - gene2.reset_memory_weight).abs()
                        + (gene1.update_memory_weight - gene2.update_memory_weight).abs();
                }
                None => {
                    disjoints = disjoints + one;
                }
            }
        }
        let size_1 = T::from(top1.genes_ev_number.len()).unwrap();
        let size_2 = T::from(top2.genes_ev_number.len()).unwrap();
        disjoints = disjoints + size_1 - common;
        let n = if size_1 + size_2 <= T::from(60).unwrap() {
            one
        } else {
            T::from(size_1 + size_2).unwrap() / T::from(60).unwrap()
        };
        T::from(2).unwrap() * disjoints / n + w / (common * T::from(3).unwrap())
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

    fn generate_output_bias(&mut self, rng: &mut ThreadRng) {
        let last_layer_size = self.layers_sizes.last().unwrap();
        self.output_bias = (0..*last_layer_size)
            .into_iter()
            .map(|_| Bias::new_random(rng))
            .collect();
    }

    fn add_to_relationship_map(&mut self, gene: Rc<RefCell<Gene<T>>>) {
        let mut rng = thread_rng();
        let gene_cp = gene.clone();
        let cell = &*gene_cp;
        let gene_borrow = &*cell.borrow();
        let input = &gene_borrow.input;
        let ev_number = gene_borrow.evolution_number;
        match self.genes_point.get_mut(&input) {
            Some(found) => {
                found.genes.push(gene.clone());
                found.genes.sort();
            }
            None => {
                let bias = Bias::new_random(&mut rng);
                let mut bias_and_genes: BiasAndGenes<T> = BiasAndGenes::new(bias);
                bias_and_genes.genes = vec![gene.clone()];
                self.genes_point.insert(input.clone(), bias_and_genes);
            }
        }
        self.genes_ev_number.insert(ev_number, gene);
    }

    pub fn add_relationship(&mut self, gene: Rc<RefCell<Gene<T>>>, init: bool) {
        let gene_cp = gene.clone();
        let cell = &*gene;
        let gene = &*cell.borrow();
        let input = &gene.input;
        let output = &gene.output;
        if input.index + 1 > self.layers_sizes[input.layer as usize] {
            self.layers_sizes[input.layer as usize] = input.index + 1;
        }
        if !init && output.index as usize == self.layers {
            self.resize(output.layer as usize);
            gene_cp.borrow_mut().decrement_output();
        } else {
            self.layers_sizes[output.layer as usize] = output.index + 1;
        }
        self.add_to_relationship_map(gene_cp);
    }

    fn resize(&mut self, layers: usize) {
        for (_point, bias_and_gene) in self.genes_point.iter() {
            for gene_rc in &bias_and_gene.genes {
                let mut gene = gene_rc.borrow_mut();
                gene.resize(layers - 1, layers);
            }
        }
        self.layers = layers;
        self.layers_sizes.resize(layers, 1);
        self.layers_sizes[layers - 1] = self.layers_sizes[layers - 2];
        self.layers_sizes[layers - 2] = 1;
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
        count: usize,
        ev_number: &EvNumber,
    ) {
        for _ in 0..count {
            let mut cp = self.clone();
            cp.mutate(&ev_number);
            new_topologies.push(Rc::new(RefCell::new(cp)));
        }
    }

    pub fn mutate(&mut self, ev_number: &EvNumber) {
        let mut rng = thread_rng();

        let mut new_output = false;
        let max_layer = self.layers.max(self.max_layers);
        let input_layer = if self.layers >= 2 {
            rng.gen_range(0, self.max_layers - 2) as u8
        } else {
            0
        };
        let input_index: u8 = rng.gen_range(0, self.layers_sizes[input_layer as usize]);
        let output_layer: u8 = rng.gen_range(input_index, max_layer as u8);
        let mut output_index: u8 = 0;

        if (output_layer as usize) < self.layers - 1 {
            output_index = rng.gen_range(
                0,
                (self.layers_sizes[output_layer as usize]).min(self.max_per_layers as u8),
            );
            if output_index >= self.layers_sizes[output_layer as usize] {
                new_output = true
            }
        } else if (output_layer as usize) == self.layers - 1 {
            output_index = rng.gen_range(0, self.layers_sizes[output_layer as usize]);
        } else {
            // if output_index == layers
            new_output = true;
        }
        let input = Point::new(input_layer, input_index);
        let output = Point::new(output_layer, output_index);
        self.new_gene(&mut rng, input.clone(), output.clone(), &ev_number);
        if new_output {
            let last_layer_size = self.layers_sizes.last().unwrap();
            let index = rng.gen_range(0, last_layer_size);
            let output_of_output = Point::new((self.layers - 1) as u8, index);
            self.new_gene(&mut rng, output.clone(), output_of_output, &ev_number);
        }
        self.disable_genes(input, output);
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

    fn disable_genes(&mut self, input: Point, output: Point) {
        match self.genes_point.get(&input) {
            Some(found) => {
                let genes = &found.genes;
                let last = genes.last().unwrap();
                for gene_rc in genes {
                    if gene_rc == last {
                        continue;
                    }
                    let mut gene = gene_rc.borrow_mut();
                    if gene.disabled {
                        continue;
                    }
                    let compared_output = &gene.output;
                    if output == *compared_output
                        || self.path_overrides(&compared_output, &output)
                        || self.path_overrides(&output, &compared_output)
                    {
                        gene.disabled = true;
                    }
                }
            }
            None => {}
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
                    if compared_output == output {
                        return true;
                    } else if compared_output.layer <= output.layer {
                        if self.path_overrides(&compared_output, &output) {
                            return true;
                        }
                    }
                }
                false
            }
            None => false,
        }
    }

    fn set_bias(&mut self, neuron: Point, bias: Bias<T>) {
        if neuron.layer as usize != self.layers - 1 {
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
            .map(|(point, b_and_g)| {
                SerializationBias::new(
                    point.clone(),
                    b_and_g.bias.clone(),
                )
            })
            .collect();
        let last_layer = self.layers - 1;
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
