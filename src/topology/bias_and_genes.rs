use crate::topology::bias::Bias;
use crate::topology::topology::GeneSmrtPtr;
use num::Float;
use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize, Serialize)]
pub struct BiasAndGenes<T>
where
    T: Float,
{
    pub bias: Bias<T>,
    pub genes: Vec<GeneSmrtPtr<T>>,
}

impl<T> BiasAndGenes<T>
where
    T: Float,
{
    pub fn new(bias: Bias<T>) -> BiasAndGenes<T> {
        BiasAndGenes {
            bias,
            genes: Vec::new(),
        }
    }
}
