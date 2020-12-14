use crate::topology::bias::Bias;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
pub struct SerializationBias {
    pub neuron: (u8, u8),
    pub bias: Bias<f64>,
}

#[derive(Deserialize, Serialize)]
pub struct SerializationGene {
    pub connection_type: i32,
    pub disabled: bool,
    pub input: (u8, u8),
    pub input_weight: f64,
    pub memory_weight: f64,
    pub output: (u8, u8),
    pub reset_input_weight: f64,
    pub reset_memory_weight: f64,
    pub update_input_weight: f64,
    pub update_memory_weight: f64,
}

#[derive(Deserialize, Serialize)]
pub struct SerializationTopology {
    pub biases: Vec<SerializationBias>,
    pub genes: Vec<SerializationGene>,
}

impl SerializationTopology {
    pub fn from_string(serialized: &str) -> Result<SerializationTopology, serde_json::Error> {
        serde_json::from_str(serialized)
    }
}
