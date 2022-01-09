use crate::topology::bias::Bias;
use crate::topology::GeneSmrtPtr;
use num::Float;
use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct BiasAndGenes<T>
where
    T: Float + Send,
{
    pub bias: Bias<T>,
    pub genes: Vec<GeneSmrtPtr<T>>,
}

impl<T> BiasAndGenes<T>
where
    T: Float + Send,
{
    pub fn new(bias: Bias<T>) -> BiasAndGenes<T> {
        BiasAndGenes {
            bias,
            genes: Vec::new(),
        }
    }
}
