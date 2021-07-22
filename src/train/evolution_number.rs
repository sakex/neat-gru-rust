use crate::topology::gene::Coordinate;
use std::collections::HashMap;
use std::sync::Mutex;

struct EvNumberData {
    counter: usize,
    current_pairs: HashMap<Coordinate, usize>,
}

pub struct EvNumber {
    mutex: Mutex<EvNumberData>,
}

impl EvNumber {
    pub fn new() -> EvNumber {
        Self::default()
    }

    pub fn reset(&self) {
        let mut lock = self.mutex.lock().unwrap();
        lock.current_pairs.clear();
    }

    pub fn number(&self, coordinate: Coordinate) -> usize {
        let mut lock = self.mutex.lock().unwrap();
        match lock.current_pairs.get_mut(&coordinate) {
            Some(found) => *found,
            None => {
                lock.counter += 1;
                let counter = lock.counter;
                lock.current_pairs.insert(coordinate, counter);
                counter
            }
        }
    }
}
impl Default for EvNumber {
    fn default() -> Self {
        EvNumber {
            mutex: Mutex::new(EvNumberData {
                counter: 0,
                current_pairs: HashMap::new(),
            }),
        }
    }
}
