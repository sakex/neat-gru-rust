use crate::topology::bias::Bias;
use num::Float;
use crate::topology::gene::Gene;
use std::rc::Rc;
use std::cell::RefCell;

pub struct BiasAndGenes<T>
    where T: Float {
    pub bias: Bias<T>,
    pub genes: Vec<Rc<RefCell<Gene<T>>>>,
}

impl<T> BiasAndGenes<T>
    where T: Float {
    pub fn new(bias: Bias<T>) -> BiasAndGenes<T> {
        BiasAndGenes {
            bias,
            genes: Vec::new(),
        }
    }
}