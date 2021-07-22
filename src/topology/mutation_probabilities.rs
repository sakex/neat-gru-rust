use std::fmt;

#[derive(Debug, Clone)]
pub struct ProbabilitiesError {}

impl fmt::Display for ProbabilitiesError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "change_weights or guaranteed_new_neuron not in interval [0.0, 1.0]",
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
    ) -> Result<MutationProbabilities, ProbabilitiesError> {
        let range = 0.0..1.0;
        if !range.contains(&guaranteed_new_neuron) || !range.contains(&change_weights) {
            return Err(ProbabilitiesError {});
        } else {
            Ok(MutationProbabilities {
                change_weights,
                guaranteed_new_neuron,
            })
        }
    }
}

unsafe impl Send for MutationProbabilities {}
unsafe impl Sync for MutationProbabilities {}
