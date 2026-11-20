use std::collections::HashMap;
use std::sync::Mutex;

#[derive(Default)]
struct EvNumberData {
    counter: usize,
    connections: HashMap<(usize, usize), usize>,
    hidden_counter: usize,
    split_neurons: HashMap<usize, usize>,
}

/// Hands out the historical markings shared by all the topologies of a training:
/// the evolution number of each connection and the index of each hidden neuron
#[derive(Default)]
pub struct EvNumber {
    mutex: Mutex<EvNumberData>,
}

impl EvNumber {
    pub fn new() -> EvNumber {
        Self::default()
    }

    /// Returns the evolution number of the connection between two neurons.
    ///
    /// The same connection always gets the same number
    pub fn number(&self, input: usize, output: usize) -> usize {
        let mut lock = self.mutex.lock().unwrap();
        if let Some(found) = lock.connections.get(&(input, output)) {
            return *found;
        }
        lock.counter += 1;
        let counter = lock.counter;
        lock.connections.insert((input, output), counter);
        counter
    }

    /// Returns the index of the hidden neuron created by splitting a connection.
    ///
    /// Splitting the same connection always gives the same neuron
    pub fn split_neuron(&self, evolution_number: usize) -> usize {
        let mut lock = self.mutex.lock().unwrap();
        if let Some(found) = lock.split_neurons.get(&evolution_number) {
            return *found;
        }
        let index = lock.hidden_counter;
        lock.hidden_counter += 1;
        lock.split_neurons.insert(evolution_number, index);
        index
    }

    /// Returns the index of a hidden neuron that was never used
    pub fn new_neuron(&self) -> usize {
        let mut lock = self.mutex.lock().unwrap();
        let index = lock.hidden_counter;
        lock.hidden_counter += 1;
        index
    }
}
