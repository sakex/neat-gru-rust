use std::fmt::Display;

use num::Float;

use crate::topology::Topology;

pub trait NN<T>: Sized
where
    T: Float + std::ops::AddAssign + Display + Send,
{
    /// Instantiates a new Neural Network from a `Topology`
    ///
    /// # Safety
    ///
    /// If the Topology is ill-formed, it will result in pointer overflow.
    /// Topologies generated by this crate are guaranteed to be safe.
    unsafe fn from_topology(topology: &Topology<T>) -> Self;

    /// Deserializes a serde serialized Topology into a neural network
    fn from_string(serialized: &str) -> Self {
        let top = Topology::from_string(serialized);
        unsafe { Self::from_topology(&top) }
    }
}
