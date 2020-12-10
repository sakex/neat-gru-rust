use serde::{Deserialize, Serialize};
use num::traits::Float;

#[derive(Serialize, Deserialize)]
pub struct Topology<T>
where T: Float {
    layers: u8,
    last_result: T,
    best_historical_result: T,
    result_before_mutation: T,
    layers_size: Vec<u8>,

}