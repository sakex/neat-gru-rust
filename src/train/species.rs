use crate::topology::topology::Topology;
use crate::train::evolution_number::EvNumber;
use num::Float;
use numeric_literals::replace_numeric_literals;
use rand::prelude::ThreadRng;
use std::cell::RefCell;
use std::rc::Rc;

pub struct Species<T>
where
    T: Float,
{
    max_individuals: usize,
    pub topologies: Vec<Rc<RefCell<Topology<T>>>>,
    pub best_topology: Rc<RefCell<Topology<T>>>,
    best_historical_score: T,
    pub stagnation_counter: u8,
}

impl<T> Species<T>
where
    T: Float,
{
    pub fn new(first_topology: Rc<RefCell<Topology<T>>>, max_individuals: usize) -> Species<T> {
        Species {
            max_individuals,
            topologies: vec![first_topology.clone()],
            best_topology: first_topology,
            best_historical_score: T::from(0).unwrap(),
            stagnation_counter: 0,
        }
    }

    pub fn new_random(
        max_individuals: usize,
        input_count: usize,
        output_count: usize,
        max_layers: usize,
        max_per_layers: usize,
        ev_number: &EvNumber,
    ) -> Species<T> {
        let mut rng: ThreadRng = rand::thread_rng();
        let topologies: Vec<Rc<RefCell<Topology<T>>>> = (0..max_individuals)
            .map(|_| {
                Rc::new(RefCell::new(Topology::<T>::new_random(
                    &mut rng,
                    input_count,
                    output_count,
                    max_layers,
                    max_per_layers,
                    &ev_number,
                )))
            })
            .collect();
        let best_topology = topologies.last().unwrap().clone();
        Species {
            max_individuals,
            topologies,
            best_topology,
            best_historical_score: T::from(0).unwrap(),
            stagnation_counter: 0,
        }
    }

    pub fn natural_selection(&mut self, ev_number: &EvNumber, cutoff: T) {
        self.topologies.sort_by(|top1, top2| {
            let top1_borrow = &**top1;
            let top1 = top1_borrow.borrow();
            let top2_borrow = &**top2;
            let top2 = top2_borrow.borrow();
            top1.get_last_result()
                .partial_cmp(&top2.get_last_result())
                .unwrap()
        });
        let best_topology = self.topologies.last().unwrap();
        let last_result = {
            let top_borrow = &**best_topology;
            let best_top = top_borrow.borrow();
            best_top.get_last_result()
        };
        if last_result > self.best_historical_score {
            self.best_historical_score = last_result;
            self.stagnation_counter = 0;
        } else {
            self.stagnation_counter += 1;
        }
        self.best_topology = best_topology.clone();
        self.do_selection(&ev_number, cutoff);
    }

    fn do_selection(&mut self, ev_number: &EvNumber, cutoff: T) {
        let size = self.topologies.len();
        if size == 0 {
            return;
        }
        // Kill half
        let mut surviving_topologies: Vec<Rc<RefCell<Topology<T>>>> = self
            .topologies
            .iter()
            .filter(|&top| {
                let borrow = top.borrow();
                borrow.get_last_result() >= cutoff
            })
            .cloned()
            .collect();

        surviving_topologies.reserve(self.max_individuals as usize);
        self.topologies = self.evolve(&mut surviving_topologies, &ev_number);
        self.topologies.push(self.best_topology.clone());
    }

    fn evolve(
        &mut self,
        surviving_topologies: &Vec<Rc<RefCell<Topology<T>>>>,
        ev_number: &EvNumber,
    ) -> Vec<Rc<RefCell<Topology<T>>>> {
        let mut new_topologies: Vec<Rc<RefCell<Topology<T>>>> = Vec::new();
        let reproduction_count = self.max_individuals / surviving_topologies.len();
        for topology in surviving_topologies.iter().rev() {
            let top = topology.borrow_mut();
            top.new_generation(&mut new_topologies, reproduction_count, &ev_number);
        }
        new_topologies
    }

    pub fn push(&mut self, top: Rc<RefCell<Topology<T>>>) {
        self.update_best(&top);
        self.topologies.push(top);
        self.max_individuals += 1;
    }

    fn update_best(&mut self, top: &Rc<RefCell<Topology<T>>>) {
        if top.borrow().get_last_result() >= self.best_topology.borrow().get_last_result() {
            self.best_topology = top.clone();
            let top_borrow = &*self.best_topology;
            let best_top = top_borrow.borrow();
            self.best_historical_score = best_top.get_last_result();
        }
    }

    #[replace_numeric_literals(T::from(literal).unwrap())]
    pub fn outsiders(&mut self) -> Vec<Rc<RefCell<Topology<T>>>> {
        let mut outsiders = Vec::new();
        let best = self.best_topology.borrow();
        self.topologies = self
            .topologies
            .iter()
            .filter(|&top| {
                let top_cp = &*top.clone();
                let top_borrow = top_cp.borrow();
                if Topology::delta_compatibility(&best, &top_borrow) >= 2 {
                    outsiders.push(top.clone());
                    false
                } else {
                    true
                }
            })
            .cloned()
            .collect();
        outsiders
    }

    pub fn score(&self) -> T {
        self.best_topology.borrow().get_last_result()
    }

    pub fn set_max_individuals(&mut self, new_max: usize) {
        self.max_individuals = new_max;
    }

    pub fn get_last_result(&self) -> T {
        let best_top_cell = &*self.best_topology;
        let best_topology = best_top_cell.borrow();
        best_topology.get_last_result()
    }

    pub fn get_best(&self) -> Topology<T> {
        let best_top_cell = &*self.best_topology;
        let best_topology = best_top_cell.borrow();
        best_topology.clone()
    }
}
