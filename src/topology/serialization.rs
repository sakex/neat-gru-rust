use crate::topology::bias::Bias;
use crate::topology::gene::Point;
use num::Float;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
pub struct SerializationBias {
    pub neuron: (u8, u8),
    pub bias: Bias<f64>,
}

impl SerializationBias {
    pub fn new<T>(neuron: Point, bias: Bias<T>) -> SerializationBias
    where
        T: Float,
    {
        let bias_f64: Bias<f64> = Bias::new(
            num::cast(bias.bias_input).unwrap(),
            num::cast(bias.bias_update).unwrap(),
            num::cast(bias.bias_reset).unwrap()
        );
        SerializationBias {
            neuron: (neuron.layer, neuron.index),
            bias: bias_f64,
        }
    }
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

impl SerializationGene {
    pub fn new(
        connection_type: i32,
        disabled: bool,
        input: (u8, u8),
        input_weight: f64,
        memory_weight: f64,
        output: (u8, u8),
        reset_input_weight: f64,
        reset_memory_weight: f64,
        update_input_weight: f64,
        update_memory_weight: f64,
    ) -> SerializationGene {
        SerializationGene {
            connection_type,
            disabled,
            input,
            input_weight,
            memory_weight,
            output,
            reset_input_weight,
            reset_memory_weight,
            update_input_weight,
            update_memory_weight,
        }
    }
}

#[derive(Deserialize, Serialize)]
pub struct SerializationTopology {
    pub biases: Vec<SerializationBias>,
    pub genes: Vec<SerializationGene>,
}

impl SerializationTopology {
    pub fn new(
        biases: Vec<SerializationBias>,
        genes: Vec<SerializationGene>,
    ) -> SerializationTopology {
        SerializationTopology { biases, genes }
    }

    pub fn from_string(serialized: &str) -> Result<SerializationTopology, serde_json::Error> {
        serde_json::from_str(serialized)
    }
}
