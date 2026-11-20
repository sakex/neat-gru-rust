use crate::topology::mutation_probabilities::MutationProbabilities;
use crate::topology::{Topology, TopologySmrtPtr};
use crate::train::evolution_number::EvNumber;
use num::Float;
use rand::prelude::ThreadRng;
use rand::{thread_rng, Rng};
use std::fmt::Display;
use std::iter::Sum;
use std::sync::{Arc, Mutex};

pub struct Species<T>
where
    T: Float + Sum + std::ops::AddAssign + Display + Send,
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
    T: Float + Sum + std::ops::AddAssign + Display + Send,
{
    pub fn new(first_topology: TopologySmrtPtr<T>) -> Species<T> {
        Species {
            topologies: vec![first_topology.clone()],
            best_topology: first_topology.lock().unwrap().clone(),
            best_historical_score: T::neg_infinity(),
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
            best_historical_score: T::neg_infinity(),
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
            best_historical_score: T::neg_infinity(),
            stagnation_counter: 0,
            adjusted_fitness: T::zero(),
            max_topologies: 0,
        }
    }

    pub fn natural_selection(
        &mut self,
        ev_number: Arc<EvNumber>,
        proba: MutationProbabilities,
        run_crossovers: bool,
    ) {
        // The results are read one at a time: locking two topologies in the comparator
        // deadlocks when the sort compares a topology with itself
        self.topologies.sort_by(|top1, top2| {
            let result1 = top1.lock().unwrap().get_last_result();
            let result2 = top2.lock().unwrap().get_last_result();
            result1.partial_cmp(&result2).unwrap()
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
        self.do_selection(ev_number, proba, run_crossovers);
    }

    fn do_selection(
        &mut self,
        ev_number: Arc<EvNumber>,
        proba: MutationProbabilities,
        run_crossovers: bool,
    ) {
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

        // Keep a slot for the unchanged copy of the best topology
        let offspring_count = self.max_topologies - will_copy_best as usize;
        self.topologies = self.evolve(
            &surviving_topologies,
            offspring_count,
            ev_number,
            proba,
            run_crossovers,
        );
        if will_copy_best {
            self.topologies
                .push(Arc::new(Mutex::new(self.best_topology.clone())));
        }
    }

    /// Creates `offspring_count` new topologies from the surviving ones
    fn evolve(
        &mut self,
        surviving_topologies: &[TopologySmrtPtr<T>],
        offspring_count: usize,
        ev_number: Arc<EvNumber>,
        proba: MutationProbabilities,
        run_crossovers: bool,
    ) -> Vec<TopologySmrtPtr<T>> {
        let mut new_topologies: Vec<TopologySmrtPtr<T>> = Vec::new();
        new_topologies.reserve_exact(offspring_count);
        let mutations_count = if run_crossovers && surviving_topologies.len() >= 3 {
            // 90% mutation, 10% crossover
            (offspring_count * 9).div_ceil(10)
        } else {
            offspring_count
        };
        while new_topologies.len() < mutations_count {
            for topology in surviving_topologies.iter().rev() {
                let top = &mut *topology.lock().unwrap();
                top.new_generation(&mut new_topologies, &ev_number, 1, &proba);
                if new_topologies.len() >= mutations_count {
                    break;
                }
            }
        }
        let mut rng = thread_rng();
        while new_topologies.len() < offspring_count {
            for (index, best) in surviving_topologies
                .iter()
                .enumerate()
                .rev()
                .take(surviving_topologies.len() - 2)
            {
                let mated_index = rng.gen_range(0..index);
                let worst = &*surviving_topologies[mated_index].lock().unwrap();
                let best = &*best.lock().unwrap();
                new_topologies.push(Topology::crossover(best, worst));
                if new_topologies.len() >= offspring_count {
                    return new_topologies;
                }
            }
        }
        new_topologies
    }

    pub fn push(&mut self, top: TopologySmrtPtr<T>) {
        self.topologies.push(top);
    }

    pub fn score(&self) -> T {
        self.best_topology.get_last_result()
    }

    pub fn compute_adjusted_fitness(&mut self) {
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
