use crate::topology::bias::Bias;
use crate::topology::gene::Gene;
use num::Float;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone, Deserialize, Serialize)]
pub struct BiasAndGenes<T>
where
    T: Float,
{
    pub bias: Bias<T>,
    pub genes: Vec<Rc<RefCell<Gene<T>>>>,
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
