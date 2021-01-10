use crate::topology::connection_type::ConnectionType;
use crate::train::evolution_number::EvNumber;
use core::cmp::Ordering;
use num::traits::Float;
use numeric_literals::replace_numeric_literals;
use rand::distributions::{Distribution, Uniform};
use rand::prelude::ThreadRng;
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Point {
    pub layer: u8,
    pub index: u8,
}

impl Point {
    pub fn new(layer: u8, index: u8) -> Point {
        Point { layer, index }
    }
}

impl Hash for Point {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.layer.hash(state);
        self.index.hash(state)
    }

    fn hash_slice<H: Hasher>(data: &[Point], state: &mut H)
    where
        Self: Sized,
    {
        for point in data.iter() {
            point.hash(state);
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct Coordinate {
    input: Point,
    output: Point,
}

impl Coordinate {
    pub fn new(input: Point, output: Point) -> Coordinate {
        Coordinate { input, output }
    }
}

impl Hash for Coordinate {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.input.layer.hash(state);
        self.input.index.hash(state);
        self.output.layer.hash(state);
        self.output.index.hash(state)
    }

    fn hash_slice<H: Hasher>(data: &[Coordinate], state: &mut H)
    where
        Self: Sized,
    {
        for coordinate in data.iter() {
            coordinate.hash(state);
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
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

impl<T> Gene<T>
where
    T: Float,
{
    pub fn new(
        input: Point,
        output: Point,
        input_weight: T,
        memory_weight: T,
        reset_input_weight: T,
        update_input_weight: T,
        reset_memory_weight: T,
        update_memory_weight: T,
        evolution_number: usize,
        connection_type: ConnectionType,
        disabled: bool,
    ) -> Gene<T> {
        Gene {
            input,
            output,
            input_weight,
            memory_weight,
            reset_input_weight,
            update_input_weight,
            reset_memory_weight,
            update_memory_weight,
            evolution_number,
            connection_type,
            disabled,
        }
    }

    pub fn new_random(
        rng: &mut ThreadRng,
        input: Point,
        output: Point,
        min: f64,
        max: f64,
        ev_number: &EvNumber,
    ) -> Gene<T> {
        let unif = Uniform::from(min..max);
        let connection_type_picker = Uniform::from(0..2);
        let connection_type = connection_type_picker.sample(rng);
        let coordinate = Coordinate::new(input.clone(), output.clone());
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

    #[replace_numeric_literals(T::from(literal).unwrap())]
    pub fn new_uniform(input: Point, output: Point, ev_number: &EvNumber) -> Gene<T> {
        let coordinate = Coordinate::new(input.clone(), output.clone());
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

    pub fn decrement_output(&mut self) {
        self.output.layer -= 1;
    }
}

impl<T> PartialEq for Gene<T>
where
    T: Float,
{
    fn eq(&self, other: &Self) -> bool {
        self.output.layer == other.output.layer && self.output.index == other.output.index
    }

    fn ne(&self, other: &Self) -> bool {
        self.output.layer != other.output.layer || self.output.index != other.output.index
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
        } else if self.output.layer < other.output.layer && self.output.layer < other.output.layer {
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
