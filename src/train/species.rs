use std::rc::Rc;
use crate::topology::topology::Topology;
use num::Float;
use rand::prelude::ThreadRng;
use std::cell::RefCell;

pub struct Species<T>
    where T: Float {
    max_individuals: usize,
    pub topologies: Vec<Rc<RefCell<Topology<T>>>>,
    best_topology: Rc<RefCell<Topology<T>>>,
}

impl<T> Species<T>
    where T: Float {
    pub fn new(first_topology: Rc<RefCell<Topology<T>>>, max_individuals: usize) -> Species<T> {
        Species {
            max_individuals,
            topologies: vec![first_topology.clone()],
            best_topology: first_topology,
        }
    }

    pub fn new_random(max_individuals: usize, input_count: usize, output_count: usize, max_layers: usize, max_per_layers: usize) -> Species<T> {
        let mut rng: ThreadRng = rand::thread_rng();
        let topologies: Vec<Rc<RefCell<Topology<T>>>> = (0..max_individuals)
            .map(|_| {
                Rc::new(RefCell::new(Topology::<T>::new_random(&mut rng, input_count, output_count, max_layers, max_per_layers)))
            })
            .collect();
        let best_topology = topologies.last().unwrap().clone();
        Species {
            max_individuals,
            topologies,
            best_topology,
        }
    }

    pub fn natural_selection(&mut self) {
        self.topologies.sort_by(|top1, top2| {
            let top1_borrow = &**top1;
            let top1 = top1_borrow.borrow();
            let top2_borrow = &**top2;
            let top2 = top2_borrow.borrow();
            top1.get_last_result().partial_cmp(&top2.get_last_result()).unwrap()
        });
        let best_topology = self.topologies.last().unwrap();
        self.best_topology = best_topology.clone();
        self.do_selection();
    }

    fn do_selection(&mut self) {
        let size = self.topologies.len();
        if size == 0 {
            return;
        }
        // Kill half
        let mut surviving_topologies: Vec<Rc<RefCell<Topology<T>>>> =
            self.topologies.iter().skip(size / 2).cloned().collect();

        surviving_topologies.reserve(self.max_individuals as usize);
        self.evolve(&mut surviving_topologies);
    }

    fn evolve(&mut self, surviving_topologies: &mut Vec<Rc<RefCell<Topology<T>>>>) {
        let mut new_topologies: Vec<Rc<RefCell<Topology<T>>>> = Vec::new();
        for topology in surviving_topologies.iter().rev() {
            let top = topology.borrow_mut();
            top.new_generation(&mut new_topologies, 2);
        }
    }
}