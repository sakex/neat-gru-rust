use crate::topology::connection_type::ConnectionType;
use crate::train::evolution_number::EvNumber;
use core::cmp::Ordering;
use num::traits::Float;
use numeric_literals::replace_numeric_literals;
use rand::distributions::{Distribution, Uniform};
use rand::prelude::ThreadRng;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::hash::Hash;
use std::ops::{Add, AddAssign};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub struct Point {
    pub layer: u8,
    pub index: u8,
}

impl Point {
    pub fn new(layer: u8, index: u8) -> Point {
        Point { layer, index }
    }
}

#[derive(Clone, PartialEq, Hash, Eq)]
pub struct Coordinate {
    input: Point,
    output: Point,
}

impl Coordinate {
    pub fn new(input: Point, output: Point) -> Coordinate {
        Coordinate { input, output }
    }
}

#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct Gene<T>
where
    T: Float,
{
    pub input: Point,
    pub output: Point,
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

impl<T> Add for Gene<T>
where
    T: Float,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            input: self.input,
            output: self.input,
            input_weight: self.input_weight + rhs.input_weight,
            memory_weight: self.memory_weight + rhs.memory_weight,
            reset_input_weight: self.reset_input_weight + rhs.reset_input_weight,
            update_input_weight: self.update_input_weight + rhs.update_input_weight,
            reset_memory_weight: self.reset_memory_weight + rhs.reset_memory_weight,
            update_memory_weight: self.update_memory_weight + rhs.update_memory_weight,
            evolution_number: self.evolution_number,
            connection_type: if matches!(self.connection_type, ConnectionType::GRU) {
                self.connection_type
            } else {
                rhs.connection_type
            },
            disabled: self.disabled,
        }
    }
}

impl<T> AddAssign for Gene<T>
where
    T: Float,
{
    fn add_assign(&mut self, rhs: Self) {
        self.input_weight = self.input_weight + rhs.input_weight;
        self.memory_weight = self.memory_weight + rhs.memory_weight;
        self.reset_input_weight = self.reset_input_weight + rhs.reset_input_weight;
        self.update_input_weight = self.update_input_weight + rhs.update_input_weight;
        self.reset_memory_weight = self.reset_memory_weight + rhs.reset_memory_weight;
        self.update_memory_weight = self.update_memory_weight + rhs.update_memory_weight;
        self.connection_type = if matches!(self.connection_type, ConnectionType::GRU) {
            self.connection_type
        } else {
            rhs.connection_type
        };
    }
}

impl<T> Gene<T>
where
    T: Float,
{
    pub fn new_random(
        rng: &mut ThreadRng,
        input: Point,
        output: Point,
        min: f64,
        max: f64,
        ev_number: &EvNumber,
    ) -> Gene<T> {
        let unif = Uniform::from(min..max);
        let connection_type_picker = Uniform::from(0..3);
        let connection_type = connection_type_picker.sample(rng);
        let coordinate = Coordinate::new(input, output);
        Gene {
            input,
            output,
            input_weight: T::from(unif.sample(rng)).unwrap(),
            memory_weight: T::from(unif.sample(rng)).unwrap(),
            reset_input_weight: T::from(unif.sample(rng)).unwrap(),
            update_input_weight: T::from(unif.sample(rng)).unwrap(),
            reset_memory_weight: T::from(unif.sample(rng)).unwrap(),
            update_memory_weight: T::from(unif.sample(rng)).unwrap(),
            evolution_number: ev_number.number(coordinate),
            connection_type: ConnectionType::from_int(connection_type),
            disabled: false,
        }
    }

    pub fn new_one(input: Point, output: Point, ev_number: &EvNumber) -> Gene<T> {
        let coordinate = Coordinate::new(input, output);
        Gene {
            input,
            output,
            input_weight: T::one(),
            memory_weight: T::one(),
            reset_input_weight: T::one(),
            update_input_weight: T::one(),
            reset_memory_weight: T::one(),
            update_memory_weight: T::one(),
            evolution_number: ev_number.number(coordinate),
            connection_type: ConnectionType::GRU,
            disabled: false,
        }
    }

    #[replace_numeric_literals(T::from(literal).unwrap())]
    pub fn new_zero(input: Point, output: Point, ev_number: &EvNumber) -> Gene<T> {
        let coordinate = Coordinate::new(input, output);
        Gene {
            input,
            output,
            input_weight: 0,
            memory_weight: 0,
            reset_input_weight: 0,
            update_input_weight: 0,
            reset_memory_weight: 0,
            update_memory_weight: 0,
            evolution_number: ev_number.number(coordinate),
            connection_type: ConnectionType::GRU,
            disabled: false,
        }
    }

    pub fn new_zero_random_type(
        input: Point,
        output: Point,
        ev_number: &EvNumber,
        rng: &mut ThreadRng,
    ) -> Gene<T> {
        let mut new_gene = Gene::new_zero(input, output, ev_number);
        let connection_type: i32 = rng.gen_range(0..3);
        new_gene.connection_type = ConnectionType::from_int(connection_type);
        new_gene
    }

    pub fn assign_values(&mut self, values: Self) {
        self.input_weight = values.input_weight;
        self.memory_weight = values.memory_weight;
        self.reset_input_weight = values.reset_input_weight;
        self.update_input_weight = values.update_input_weight;
        self.reset_memory_weight = values.reset_memory_weight;
        self.update_memory_weight = values.update_memory_weight;
        self.evolution_number = values.evolution_number;
        self.connection_type = values.connection_type;
    }

    pub fn split(&self, middle_point: Point, ev_number: &EvNumber) -> (Gene<T>, Gene<T>) {
        let first_gene = Gene::new_one(self.input, middle_point, ev_number);

        let coordinate = Coordinate::new(middle_point, self.output);
        let mut second_gene = self.clone();
        second_gene.input = middle_point;
        second_gene.evolution_number = ev_number.number(coordinate);
        second_gene.disabled = false;

        (first_gene, second_gene)
    }

    #[inline]
    pub fn random_reassign(&mut self, rng: &mut ThreadRng) {
        let unif = Uniform::from(-1.0..1.);
        self.input_weight = T::from(unif.sample(rng)).unwrap();
        self.memory_weight = T::from(unif.sample(rng)).unwrap();
        self.reset_input_weight = T::from(unif.sample(rng)).unwrap();
        self.update_input_weight = T::from(unif.sample(rng)).unwrap();
        self.reset_memory_weight = T::from(unif.sample(rng)).unwrap();
        self.update_memory_weight = T::from(unif.sample(rng)).unwrap();
    }

    pub fn decrement_output(&mut self) {
        self.output.layer -= 1;
    }

    #[inline]
    #[replace_numeric_literals(T::from(literal).unwrap())]
    pub fn average_weights(&mut self, other: &Gene<T>) {
        self.input_weight = (other.input_weight + self.input_weight) / 2.0;
        self.memory_weight = (other.memory_weight + self.memory_weight) / 2.0;
        self.reset_input_weight = (other.reset_input_weight + self.reset_input_weight) / 2.0;
        self.update_input_weight = (other.update_input_weight + self.update_input_weight) / 2.0;
        self.reset_memory_weight = (other.reset_memory_weight + self.reset_memory_weight) / 2.0;
        self.update_memory_weight = (other.update_memory_weight + self.update_memory_weight) / 2.0;
    }
}

impl<T> PartialEq for Gene<T>
where
    T: Float,
{
    fn eq(&self, other: &Self) -> bool {
        self.output.layer == other.output.layer && self.output.index == other.output.index
    }
}

impl<T> Eq for Gene<T> where T: Float {}

impl<T> Ord for Gene<T>
where
    T: Float,
{
    fn cmp(&self, other: &Self) -> Ordering {
        if self.output.layer == other.output.layer && self.output.index == other.output.index {
            Ordering::Equal
        } else if self.output.layer < other.output.layer
            || (self.output.layer == other.output.layer && self.output.index < other.output.index)
        {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    }
}

impl<T> PartialOrd for Gene<T>
where
    T: Float,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }

    fn lt(&self, other: &Self) -> bool {
        (self.output.layer == other.output.layer && self.output.index < other.output.index)
            || self.output.layer < other.output.layer
    }

    fn le(&self, other: &Self) -> bool {
        (self.output.layer == other.output.layer && self.output.index <= other.output.index)
            || self.output.layer <= other.output.layer
    }

    fn gt(&self, other: &Self) -> bool {
        (self.output.layer == other.output.layer && self.output.index > other.output.index)
            || self.output.layer > other.output.layer
    }

    fn ge(&self, other: &Self) -> bool {
        (self.output.layer == other.output.layer && self.output.index >= other.output.index)
            || self.output.layer >= other.output.layer
    }
}
