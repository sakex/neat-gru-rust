use crate::topology::connection_type::ConnectionType;
use crate::train::evolution_number::EvNumber;
use crate::utils::floats_almost_equal;
use num::traits::Float;
use rand::distributions::{Distribution, Uniform};
use rand::prelude::ThreadRng;
use serde::{Deserialize, Serialize};

/// A connection between two neurons, identified by their ids
#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct Gene<T>
where
    T: Float,
{
    pub input: usize,
    pub output: usize,
    pub input_weight: T,
    pub memory_weight: T,
    pub reset_input_weight: T,
    pub update_input_weight: T,
    pub reset_memory_weight: T,
    pub update_memory_weight: T,
    pub evolution_number: usize,
    pub connection_type: ConnectionType,
    pub disabled: bool,
}

impl<T> Gene<T>
where
    T: Float,
{
    pub fn new_random(
        rng: &mut ThreadRng,
        input: usize,
        output: usize,
        min: f64,
        max: f64,
        ev_number: &EvNumber,
    ) -> Gene<T> {
        let connection_type_picker = Uniform::from(0..3);
        let connection_type = connection_type_picker.sample(rng);
        let mut gene = Gene::new_one(input, output, ev_number);
        gene.connection_type = ConnectionType::from_int(connection_type);
        gene.random_reassign_in(rng, min, max);
        gene
    }

    pub fn new_one(input: usize, output: usize, ev_number: &EvNumber) -> Gene<T> {
        Gene {
            input,
            output,
            input_weight: T::one(),
            memory_weight: T::one(),
            reset_input_weight: T::one(),
            update_input_weight: T::one(),
            reset_memory_weight: T::one(),
            update_memory_weight: T::one(),
            evolution_number: ev_number.number(input, output),
            connection_type: ConnectionType::GRU,
            disabled: false,
        }
    }

    /// Copies the weights and the connection type of another gene
    pub fn assign_values(&mut self, values: &Self) {
        self.input_weight = values.input_weight;
        self.memory_weight = values.memory_weight;
        self.reset_input_weight = values.reset_input_weight;
        self.update_input_weight = values.update_input_weight;
        self.reset_memory_weight = values.reset_memory_weight;
        self.update_memory_weight = values.update_memory_weight;
        self.connection_type = values.connection_type;
    }

    /// Splits the connection in two by adding a neuron in the middle.
    ///
    /// The first half is neutral and the second one keeps the weights, to stay as close as
    /// possible to the original connection
    pub fn split(&self, middle: usize, ev_number: &EvNumber) -> (Gene<T>, Gene<T>) {
        let first_gene = Gene::new_one(self.input, middle, ev_number);

        let mut second_gene = self.clone();
        second_gene.input = middle;
        second_gene.evolution_number = ev_number.number(middle, self.output);
        second_gene.disabled = false;

        (first_gene, second_gene)
    }

    #[inline]
    pub fn random_reassign(&mut self, rng: &mut ThreadRng) {
        self.random_reassign_in(rng, -1., 1.);
    }

    fn random_reassign_in(&mut self, rng: &mut ThreadRng, min: f64, max: f64) {
        let unif = Uniform::from(min..max);
        self.input_weight = T::from(unif.sample(rng)).unwrap();
        self.memory_weight = T::from(unif.sample(rng)).unwrap();
        self.reset_input_weight = T::from(unif.sample(rng)).unwrap();
        self.update_input_weight = T::from(unif.sample(rng)).unwrap();
        self.reset_memory_weight = T::from(unif.sample(rng)).unwrap();
        self.update_memory_weight = T::from(unif.sample(rng)).unwrap();
    }
}

impl<T> PartialEq for Gene<T>
where
    T: Float,
{
    fn eq(&self, other: &Self) -> bool {
        self.input == other.input
            && self.output == other.output
            && self.evolution_number == other.evolution_number
            && self.connection_type == other.connection_type
            && self.disabled == other.disabled
            && floats_almost_equal(self.input_weight, other.input_weight)
            && floats_almost_equal(self.memory_weight, other.memory_weight)
            && floats_almost_equal(self.reset_input_weight, other.reset_input_weight)
            && floats_almost_equal(self.update_input_weight, other.update_input_weight)
            && floats_almost_equal(self.reset_memory_weight, other.reset_memory_weight)
            && floats_almost_equal(self.update_memory_weight, other.update_memory_weight)
    }
}
