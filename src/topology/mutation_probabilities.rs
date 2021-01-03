use std::fmt;

#[derive(Debug, Clone)]
pub struct ProbabilitiesError {
    sum: f64,
}

impl fmt::Display for ProbabilitiesError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "change_weights + guaranteed_new_neuron = {}. Out of bounds, should be between 0 and 1",
            self.sum
        )
    }
}

/// Probabilities of a mutation to happen  
///
/// The default mutation generates a new gene randomly  
///
/// `change_weights`: Every weights will be added a value following a normal distribution ~N(0, 0.1)  
///
/// `guaranteed_new_neuron`: Creates a new neuron randomly if within bounds of max_layers and max_per_layers  
#[derive(Clone)]
pub struct MutationProbabilities {
    pub(crate) change_weights: f64,
    pub(crate) guaranteed_new_neuron: f64,
    pub(crate) delete_neuron: f64,
}

impl MutationProbabilities {
    /// # Arguments
    ///
    /// `change_weights`: Every weights will be added a value following a normal distribution ~N(0, 0.1)  
    ///
    /// `guaranteed_new_neuron`: Creates a new neuron randomly if within bounds of max_layers and max_per_layers  
    pub fn new(
        change_weights: f64,
        guaranteed_new_neuron: f64,
        delete_neuron: f64,
    ) -> Result<MutationProbabilities, ProbabilitiesError> {
        if change_weights + guaranteed_new_neuron + delete_neuron > 1.0
            || change_weights + guaranteed_new_neuron + delete_neuron < 0.0
        {
            return Err(ProbabilitiesError {
                sum: change_weights + guaranteed_new_neuron,
            });
        }
        Ok(MutationProbabilities {
            change_weights,
            guaranteed_new_neuron,
            delete_neuron,
        })
    }
}

unsafe impl Send for MutationProbabilities {}
unsafe impl Sync for MutationProbabilities {}
