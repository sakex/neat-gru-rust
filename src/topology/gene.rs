use num::traits::Float;
use rand::distributions::{Uniform, Distribution};
use rand::prelude::ThreadRng;
use crate::topology::connection_type::ConnectionType;
use core::cmp::Ordering;
use std::hash::{Hash, Hasher};

#[derive(Clone, PartialEq, Eq)]
pub struct Point {
    pub layer: u8,
    pub index: u8,
}

impl Point {
    pub fn new(layer: u8, index: u8) -> Point {
        Point {
            layer,
            index,
        }
    }
}


impl Hash for Point {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.layer.hash(state);
        self.index.hash(state)
    }

    fn hash_slice<H: Hasher>(data: &[Self], state: &mut H) where
        Self: Sized, {
        for point in data.iter() {
            point.hash(state);
        }
    }
}

#[derive(Clone)]
pub struct Gene<T>
    where T: Float {
    pub input: Point,
    pub output: Point,
    pub input_weight: T,
    pub memory_weight: T,
    pub reset_input_weight: T,
    pub update_input_weight: T,
    pub reset_memory_weight: T,
    pub update_memory_weight: T,
    pub evolution_number: u64,
    pub connection_type: ConnectionType,
    pub disabled: bool,
}

impl<T> Gene<T> where T: Float {
    pub fn new(input: Point,
               output: Point,
               input_weight: T,
               memory_weight: T,
               reset_input_weight: T,
               update_input_weight: T,
               reset_memory_weight: T,
               update_memory_weight: T,
               evolution_number: u64,
               connection_type: ConnectionType,
               disabled: bool) -> Gene<T> {
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

    pub fn new_random(rng: &mut ThreadRng, input: Point, output: Point, min: f64, max: f64) -> Gene<T> {
        let unif = Uniform::from(min..max);
        let connection_type_picker = Uniform::from(0..2);
        let connection_type = connection_type_picker.sample(rng);
        Gene {
            input,
            output,
            input_weight: T::from(unif.sample(rng)).unwrap(),
            memory_weight: T::from(unif.sample(rng)).unwrap(),
            reset_input_weight: T::from(unif.sample(rng)).unwrap(),
            update_input_weight: T::from(unif.sample(rng)).unwrap(),
            reset_memory_weight: T::from(unif.sample(rng)).unwrap(),
            update_memory_weight: T::from(unif.sample(rng)).unwrap(),
            evolution_number: 0,
            connection_type: ConnectionType::from_int(connection_type),
            disabled: false,
        }
    }

    pub fn decrement_output(&mut self) {
        self.output.layer -= 1;
    }

    pub fn resize(&mut self, former_size: usize, new_size: usize) {
        if self.input.layer == former_size as u8 {
            self.input.layer = new_size as u8
        }
    }
}

impl<T> PartialEq for Gene<T>
    where T: Float {
    fn eq(&self, other: &Self) -> bool {
        self.output.layer == other.output.layer && self.output.index == other.output.index
    }

    fn ne(&self, other: &Self) -> bool {
        self.output.layer != other.output.layer || self.output.index != other.output.index
    }
}

impl<T> Eq for Gene<T> where T: Float {}

impl<T> Ord for Gene<T>
    where T: Float {
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
    where T: Float {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }

    fn lt(&self, other: &Self) -> bool {
        (self.output.layer == other.output.layer && self.output.index < other.output.index) || self.output.layer < other.output.layer
    }

    fn le(&self, other: &Self) -> bool {
        (self.output.layer == other.output.layer && self.output.index <= other.output.index) || self.output.layer <= other.output.layer
    }

    fn gt(&self, other: &Self) -> bool {
        (self.output.layer == other.output.layer && self.output.index > other.output.index) || self.output.layer > other.output.layer
    }

    fn ge(&self, other: &Self) -> bool {
        (self.output.layer == other.output.layer && self.output.index >= other.output.index) || self.output.layer >= other.output.layer
    }
}