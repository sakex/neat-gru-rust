use crate::topology::bias::Bias;
use num::Float;
use serde::{Deserialize, Serialize};

/// A neuron is identified by its id.
///
/// Files written before neurons had ids identify them by `[layer, index]` instead
#[derive(Deserialize, Serialize, Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[serde(untagged)]
pub enum SerializationNeuron {
    Id(usize),
    Layered(u8, u8),
}

#[derive(Deserialize, Serialize)]
pub struct SerializationBias {
    pub neuron: SerializationNeuron,
    pub bias: Bias<f64>,
}

impl SerializationBias {
    pub fn new<T>(neuron: usize, bias: &Bias<T>) -> SerializationBias
    where
        T: Float + Send,
    {
        let bias_f64: Bias<f64> = Bias::new(
            num::cast(bias.bias_input).unwrap(),
            num::cast(bias.bias_update).unwrap(),
            num::cast(bias.bias_reset).unwrap(),
        );
        SerializationBias {
            neuron: SerializationNeuron::Id(neuron),
            bias: bias_f64,
        }
    }
}

#[derive(Deserialize, Serialize)]
pub struct SerializationGene {
    pub connection_type: i32,
    pub disabled: bool,
    pub input: SerializationNeuron,
    pub input_weight: f64,
    pub memory_weight: f64,
    pub output: SerializationNeuron,
    pub reset_input_weight: f64,
    pub reset_memory_weight: f64,
    pub update_input_weight: f64,
    pub update_memory_weight: f64,
    pub ev_number: Option<f64>,
}

#[derive(Deserialize, Serialize)]
pub struct SerializationTopology {
    /// Number of input neurons, their ids are `0..inputs`. Absent from layered files
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inputs: Option<usize>,
    /// Number of output neurons, their ids are `inputs..inputs + outputs`. Absent from layered
    /// files
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outputs: Option<usize>,
    pub biases: Vec<SerializationBias>,
    pub genes: Vec<SerializationGene>,
}

impl SerializationTopology {
    pub fn from_string(serialized: &str) -> Result<SerializationTopology, serde_json::Error> {
        serde_json::from_str(serialized)
    }
}
