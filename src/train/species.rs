use crate::topology::mutation_probabilities::MutationProbabilities;
use crate::topology::topology::{Topology, TopologySmrtPtr};
use crate::train::evolution_number::EvNumber;
use num::Float;
use rand::prelude::ThreadRng;
use std::fmt::Display;
use std::iter::Sum;
use std::sync::{Arc, Mutex};

pub struct Species<T>
where
    T: Float + Sum + std::ops::AddAssign + Display,
{
    pub topologies: Vec<TopologySmrtPtr<T>>,
    pub best_topology: Topology<T>,
    best_historical_score: T,
    pub stagnation_counter: u8,
    pub adjusted_fitness: T,
    pub max_topologies: usize,
}

impl<T> Species<T>
where
    T: Float + Sum + std::ops::AddAssign + Display,
{
    pub fn new(first_topology: TopologySmrtPtr<T>) -> Species<T> {
        Species {
            topologies: vec![first_topology.clone()],
            best_topology: first_topology.lock().unwrap().clone(),
            best_historical_score: T::zero(),
            stagnation_counter: 0,
            adjusted_fitness: T::zero(),
            max_topologies: 0,
        }
    }

    #[allow(dead_code)]
    pub fn new_random(
        max_individuals: usize,
        input_count: usize,
        output_count: usize,
        max_layers: usize,
        max_per_layers: usize,
        ev_number: &EvNumber,
    ) -> Species<T> {
        let mut rng: ThreadRng = rand::thread_rng();
        let topologies: Vec<TopologySmrtPtr<T>> = (0..max_individuals)
            .map(|_| {
                Arc::new(Mutex::new(Topology::<T>::new_random(
                    &mut rng,
                    input_count,
                    output_count,
                    max_layers,
                    max_per_layers,
                    ev_number,
                )))
            })
            .collect();
        let best_topology = topologies.last().unwrap().lock().unwrap().clone();
        Species {
            topologies,
            best_topology,
            best_historical_score: T::zero(),
            stagnation_counter: 0,
            adjusted_fitness: T::zero(),
            max_topologies: 0,
        }
    }

    pub fn new_uniform(
        input_count: usize,
        output_count: usize,
        max_layers: usize,
        max_per_layers: usize,
        ev_number: &EvNumber,
    ) -> Species<T> {
        let topologies: Vec<TopologySmrtPtr<T>> = (0..1)
            .map(|_| {
                Arc::new(Mutex::new(Topology::<T>::new_uniform(
                    input_count,
                    output_count,
                    max_layers,
                    max_per_layers,
                    ev_number,
                )))
            })
            .collect();
        let best_topology = topologies.last().unwrap().lock().unwrap().clone();
        Species {
            topologies,
            best_topology,
            best_historical_score: T::zero(),
            stagnation_counter: 0,
            adjusted_fitness: T::zero(),
            max_topologies: 0,
        }
    }

    pub fn natural_selection(&mut self, ev_number: Arc<EvNumber>, proba: MutationProbabilities) {
        self.topologies.sort_by(|top1, top2| {
            let top1_borrow = &**top1;
            let top1 = &*top1_borrow.lock().unwrap();
            let top2_borrow = &**top2;
            let top2 = &*top2_borrow.lock().unwrap();
            top1.get_last_result()
                .partial_cmp(&top2.get_last_result())
                .unwrap()
        });
        let best_topology = self.topologies.last().unwrap();
        let last_result = {
            let top_borrow = &**best_topology;
            let best_top = &*top_borrow.lock().unwrap();
            self.best_topology = best_top.clone();
            best_top.get_last_result()
        };
        if last_result > self.best_historical_score {
            self.best_historical_score = last_result;
            self.stagnation_counter = 0;
        } else {
            self.stagnation_counter += 1;
        }
        self.do_selection(ev_number, proba);
    }

    fn do_selection(&mut self, ev_number: Arc<EvNumber>, proba: MutationProbabilities) {
        let size = self.topologies.len();
        let will_copy_best = size >= 5;
        if size == 0 || self.max_topologies == 0 {
            self.topologies.clear();
            return;
        }

        let surviving_topologies: Vec<TopologySmrtPtr<T>> = self
            .topologies
            .iter()
            .skip(self.topologies.len() / 2)
            .cloned()
            .collect();

        self.topologies = self.evolve(&surviving_topologies, ev_number, proba);
        if will_copy_best {
            self.topologies
                .push(Arc::new(Mutex::new(self.best_topology.clone())));
        }
    }

    fn evolve(
        &mut self,
        surviving_topologies: &[TopologySmrtPtr<T>],
        ev_number: Arc<EvNumber>,
        proba: MutationProbabilities,
    ) -> Vec<TopologySmrtPtr<T>> {
        let mut new_topologies: Vec<TopologySmrtPtr<T>> = Vec::new();
        new_topologies.reserve_exact(self.max_topologies);
        loop {
            for topology in surviving_topologies.iter().rev() {
                let top = &mut *topology.lock().unwrap();
                top.new_generation(&mut new_topologies, &ev_number, 1, &proba);
                let full = new_topologies.len() >= self.max_topologies;
                if full {
                    return new_topologies;
                }
            }
        }
    }

    pub fn push(&mut self, top: TopologySmrtPtr<T>) {
        self.topologies.push(top);
    }

    pub fn score(&self) -> T {
        self.best_topology.get_last_result()
    }

    pub fn compute_adjusted_fitness(&mut self) {
        /*
            self.adjusted_fitness = self.best_topology.borrow().get_last_result();
        }*/
        let top_len = T::from(self.topologies.len()).unwrap();
        if top_len == T::zero() {
            self.adjusted_fitness = T::zero();
            return;
        }
        self.adjusted_fitness = self
            .topologies
            .iter()
            .map(|top| {
                let borrowed = top.lock().unwrap();
                borrowed.get_last_result()
            })
            .sum::<T>()
            / top_len;
    }
}
