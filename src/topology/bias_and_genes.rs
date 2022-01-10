use std::{cell::RefCell, rc::Rc};

use crate::topology::bias::Bias;
use num::Float;
use serde::{Deserialize, Serialize};

use super::gene::Gene;

pub type GeneSmrtPtr<T> = Rc<RefCell<Gene<T>>>;

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
