use crate::game::{Game, GameAsync};
#[cfg(target_arch = "wasm32")]
use crate::instant_wasm_replacement::Instant;
use crate::neural_network::nn::NeuralNetwork;
use crate::section;
use crate::topology::mutation_probabilities::MutationProbabilities;
use crate::topology::topology::{Topology, TopologySmrtPtr};
use crate::train::error::TrainingError;
use crate::train::evolution_number::EvNumber;
use crate::train::species::Species;
use itertools::Itertools;
use num::Float;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::fmt::Display;
use std::iter::Sum;
use std::sync::{Arc, Mutex};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

macro_rules! cond_iter {
    ($collection: expr) => {{
        #[cfg(not(target_arch = "wasm32"))]
        {
            $collection.par_iter()
        }
        #[cfg(target_arch = "wasm32")]
        {
            $collection.iter()
        }
    }};
}

macro_rules! cond_iter_mut {
    ($collection: expr) => {{
        #[cfg(not(target_arch = "wasm32"))]
        {
            $collection.par_iter_mut()
        }
        #[cfg(target_arch = "wasm32")]
        {
            $collection.iter_mut()
        }
    }};
}

/// The train struct is used to train a Neural Network on a simulation with the NEAT algorithm
pub struct Train<'a, T, F>
where
    F: 'a + Float + Sum + Display + std::ops::AddAssign + std::ops::SubAssign + Send + Sync,
    T: Game<F>,
    &'a [F]: rayon::iter::IntoParallelIterator,
{
    pub simulation: &'a mut T,
    iterations_: usize,
    max_individuals_: usize,
    max_layers_: usize,
    max_per_layers_: usize,
    delta_threshold_: F,
    c1_: F,
    c2_: F,
    c3_: F,
    inputs_: Option<usize>,
    outputs_: Option<usize>,
    topologies_: Vec<TopologySmrtPtr<F>>,
    species_: Vec<Mutex<Species<F>>>,
    history_: Vec<Topology<F>>,
    ev_number_: Arc<EvNumber>,
    best_historical_score: F,
    no_progress_counter: usize,
    proba: MutationProbabilities,
    access_train_object_fn: Option<Box<dyn FnMut(&mut Train<'a, T, F>)>>,
}

impl<'a, T, F> Train<'a, T, F>
where
    T: Game<F>,
    F: 'a + Float + Sum + Display + std::ops::AddAssign + std::ops::SubAssign + Send + Sync,
    &'a [F]: rayon::iter::IntoParallelIterator,
{
    /// Creates a Train<T: Game> instance
    ///
    /// Default values are:
    /// - iterations -> the number of generations to be run: 1000
    /// - max_individuals -> number of networks per generation: 100
    /// - max_layers -> maximum number of layers
    /// - max_per_layers -> maximum number of neurons per layer
    ///
    /// Mandatory fields (use setters):
    /// - inputs -> the number of neurons on the first layer
    /// - outputs -> the number of neurons on the last layer
    ///
    /// # Example  
    ///
    /// ```
    /// use neat_gru::neural_network::nn::NeuralNetwork;;
    /// use neat_gru::topology::topology::Topology;
    /// use neat_gru::game::Game;
    ///
    /// struct TestGame {
    ///     nets: Vec<NeuralNetwork<f64>>,
    /// }
    ///
    /// impl TestGame {
    ///     pub fn new() -> TestGame {
    ///         TestGame { nets: Vec::new() }
    ///     }
    /// }
    ///
    /// impl Game<f64> for TestGame {
    ///     fn run_generation(&mut self) -> Vec<f64> {
    ///         self.nets
    ///             .iter_mut()
    ///             .map(|network| {
    ///                 let inputs = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    ///                 let out = network.compute(&*inputs);
    ///                 let mut diff = 0f64;
    ///                 inputs.iter().zip(out.iter()).for_each(|(a, b)| {
    ///                     diff -= (a - b).abs();
    ///                 });
    ///                 diff
    ///             })
    ///             .collect()
    ///     }
    ///
    ///     fn reset_players(&mut self, nets: Vec<NeuralNetwork<f64>>) {
    ///         self.nets = nets;
    ///     }
    ///
    ///     fn post_training(&mut self, _history: &[Topology<f64>]) {}
    /// }
    /// ```
    #[inline]
    pub fn new(simulation: &'a mut T) -> Train<'a, T, F> {
        let iterations_: usize = 1000;
        let max_individuals_: usize = 100;
        let inputs_ = None;
        let outputs_ = None;

        Train {
            simulation,
            iterations_,
            max_individuals_,
            max_layers_: 4,
            max_per_layers_: 20,
            delta_threshold_: F::from(3).unwrap(),
            c1_: F::one(),
            c2_: F::one(),
            c3_: F::one(),
            inputs_,
            outputs_,
            topologies_: Vec::new(),
            species_: Vec::new(),
            history_: Vec::new(),
            ev_number_: Arc::new(EvNumber::new()),
            best_historical_score: F::zero(),
            no_progress_counter: 0,
            access_train_object_fn: None,
            proba: MutationProbabilities {
                change_weights: 0.95,
                guaranteed_new_neuron: 0.2,
            },
        }
    }

    /// Sets the number of iterations
    ///
    /// Iterations is the maximum number of generations to be run, optional and defaults to 1000
    ///
    /// # Arguments
    ///
    /// `it` - The number of generations to be run
    pub fn iterations(&mut self, it: usize) -> &mut Self {
        self.iterations_ = it;
        self
    }

    /// Sets the number of networks per generation
    ///
    /// This function is optional as the number of max individuals defaults to 100
    ///
    /// # Arguments
    ///
    /// `network_number` - The number of networks per generation
    #[inline]
    pub fn max_individuals(&mut self, network_number: usize) -> &mut Self {
        self.max_individuals_ = network_number;
        self
    }

    /// Sets the delta threshold at which two topologies don't belong to the same species
    ///
    /// This function is optional as the number of max individuals defaults to 100
    ///
    /// # Arguments
    ///
    /// `threshold` - The new delta threshold
    #[inline]
    pub fn delta_threshold(&mut self, threshold: F) -> &mut Self {
        self.delta_threshold_ = threshold;
        self
    }

    /// Sets the delta threshold formula parameter  
    ///
    /// The formula is:  
    ///
    /// `delta = (c1 * disjoints + c2 * excess) / (larger_topology_length - initial_size) + (mean(weight_distances) * c3)`
    ///
    /// # Arguments
    ///
    /// `c1` - Defaults to 1
    /// `c2` - Defaults to 1
    /// `c3` - Defaults to 1
    #[inline]
    pub fn formula(&mut self, c1: F, c2: F, c3: F) -> &mut Self {
        self.c1_ = c1;
        self.c2_ = c2;
        self.c3_ = c3;
        self
    }

    /// Sets the probabilities of different mutations
    ///
    /// # Arguments
    ///
    /// `proba` - The new probabilities
    #[inline]
    pub fn mutation_probabilities(&mut self, proba: MutationProbabilities) -> &mut Self {
        self.proba = proba;
        self
    }

    /// Sets the number of neurons on the first layer
    ///
    /// This function has to be called in order to start training
    ///
    /// # Arguments
    ///
    /// `i` - The number of neurons on the first layer
    #[inline]
    pub fn inputs(&mut self, i: usize) -> &mut Self {
        self.inputs_ = Some(i);
        self
    }

    /// Sets the number of neurons on the last layer
    ///
    /// This function has to be called in order to start training
    ///
    /// # Arguments
    ///
    /// `o` - The number of neurons on the last layer
    #[inline]
    pub fn outputs(&mut self, o: usize) -> &mut Self {
        self.outputs_ = Some(o);
        self
    }

    /// Sets the maximum number of layers for the networks
    ///
    /// This function is optional as the max number of layers defaults to 4
    ///
    /// # Arguments
    ///
    /// `layers` - The maximum number of layers
    #[inline]
    pub fn max_layers(&mut self, layers: usize) -> &mut Self {
        self.max_layers_ = layers;
        self
    }

    /// Sets the maximum number of neurons per layers for the networks
    ///
    /// This function is optional as the max neurons per layer defaults to 50
    ///
    /// # Arguments
    ///
    /// `n` - The maximum number of neurons per layers
    #[inline]
    pub fn max_per_layers(&mut self, n: usize) -> &mut Self {
        self.max_per_layers_ = n;
        self
    }

    /// Returns the number of species
    #[inline]
    pub fn species_count(&self) -> usize {
        self.species_.len()
    }

    /// Access train object after `reset_players`
    ///
    /// # Arguments
    ///
    /// `callback` - Callback called after `reset_players`
    #[inline]
    pub fn access_train_object(
        &mut self,
        callback: Box<dyn FnMut(&mut Train<'a, T, F>)>,
    ) -> &mut Self {
        self.access_train_object_fn = Some(callback);
        self
    }

    /// Starts the training.
    ///
    /// May return a NoInput Error if no input or output is given
    #[inline]
    pub fn start(&mut self) -> Result<(), TrainingError> {
        let inputs = self.inputs_.ok_or(TrainingError::NoInput)?;

        let outputs = self.outputs_.ok_or(TrainingError::NoInput)?;

        self.species_.push(Mutex::new(Species::new_uniform(
            inputs,
            outputs,
            self.max_layers_,
            self.max_per_layers_,
            &self.ev_number_,
        )));

        self.reset_players();
        // Run generations
        for i in 0..self.iterations_ {
            section!();
            println!("Generation {}", i);
            let now = Instant::now();
            let results = self.simulation.run_generation();
            println!("RUN GENERATION: {}ms", now.elapsed().as_millis());
            self.set_last_results(&results);
            let now = Instant::now();
            self.natural_selection();
            self.push_to_history();
            self.reset_species();
            if self.species_.is_empty() {
                break;
            }
            println!("NATURAL SELECTION: {}ms", now.elapsed().as_millis());
            let now = Instant::now();
            self.reset_players();
            println!("RESET PLAYERS: {}ms", now.elapsed().as_millis());
            let mut cb_option = self.access_train_object_fn.take();
            let cb_option_borrow = &mut cb_option;
            if let Some(cb) = cb_option_borrow {
                (*cb)(self);
                self.access_train_object_fn = cb_option;
            }
        }
        section!();
        println!("POST TRAINING");
        self.simulation.post_training(&*self.history_);
        Ok(())
    }

    fn collect_topologies(&mut self) {
        self.topologies_ = cond_iter!(self.species_)
            .map(|mutex| {
                let lock = mutex.lock().unwrap();
                let species = &*lock;
                species.topologies.to_vec()
            })
            .flatten()
            .collect();
    }

    fn reset_players(&mut self) {
        self.collect_topologies();

        let networks: Vec<NeuralNetwork<F>> = cond_iter!(self.topologies_)
            .map(|top_rc| {
                let lock = top_rc.lock().unwrap();
                let top = &*lock;
                unsafe { NeuralNetwork::new(top) }
            })
            .collect();
        println!(
            "TOPOLOGIES: {}, SPECIES: {}",
            networks.len(),
            self.species_.len()
        );
        self.simulation.reset_players(networks);
    }

    fn set_last_results(&mut self, results: &Vec<F>) {
        cond_iter_mut!(self.topologies_)
            .zip(cond_iter!(results))
            .for_each(|(topology, result)| {
                if result.is_nan() {
                    panic!("NaN result");
                }
                topology.lock().unwrap().set_last_result(*result);
            })
    }

    fn natural_selection(&mut self) {
        self.species_
            .retain(|spec| spec.lock().unwrap().stagnation_counter < 20);
        if self.species_.len() == 1 {
            let first_spec = &mut *self.species_[0].lock().unwrap();
            first_spec.max_topologies = self.max_individuals_;
            self.ev_number_.reset();
            let ev_number = self.ev_number_.clone();
            first_spec.natural_selection(ev_number, self.proba.clone());
            return;
        }
        if self.species_.is_empty() {
            return;
        }
        self.species_.iter_mut().for_each(|spec| {
            spec.get_mut().unwrap().compute_adjusted_fitness();
        });
        let mean = cond_iter!(self.species_)
            .clone()
            .map(|spec| spec.lock().unwrap().adjusted_fitness)
            .sum::<F>()
            / F::from(self.species_.len()).unwrap();
        let variance = cond_iter!(self.species_)
            .clone()
            .map(|spec| (spec.lock().unwrap().adjusted_fitness - mean).powf(F::from(2.).unwrap()))
            .sum::<F>()
            / F::from(self.species_.len() - 1).unwrap();
        if variance >= F::from(0.00001).unwrap() {
            let volatility = variance.sqrt();
            self.species_.iter_mut().for_each(|spec| {
                let spec = &mut *spec.get_mut().unwrap();
                spec.adjusted_fitness = F::from(1.3)
                    .unwrap()
                    .powf((spec.adjusted_fitness - mean) / volatility);
            });
        } else {
            self.species_.iter_mut().for_each(|spec| {
                spec.get_mut().unwrap().adjusted_fitness = F::one();
            });
        }
        self.species_.sort_by(|spec1, spec2| {
            let spec1 = &*spec1.lock().unwrap();
            let spec2 = &*spec2.lock().unwrap();
            if spec1.adjusted_fitness == spec2.adjusted_fitness {
                spec1
                    .topologies
                    .len()
                    .partial_cmp(&spec2.topologies.len())
                    .unwrap()
            } else {
                spec1
                    .adjusted_fitness
                    .partial_cmp(&spec2.adjusted_fitness)
                    .expect(&*format!(
                        "First: {}, second: {}, variance {}",
                        spec1.adjusted_fitness, spec2.adjusted_fitness, variance
                    ))
            }
        });
        let sum: F = cond_iter!(self.species_)
            .map(|spec| spec.lock().unwrap().adjusted_fitness)
            .sum();
        let multiplier: F = F::from(self.max_individuals_).unwrap() / sum;
        let mut assigned_count: usize = 0;
        for spec in self.species_.iter_mut().rev() {
            let spec = &mut *spec.get_mut().unwrap();
            let to_assign = if assigned_count < self.max_individuals_ {
                (spec.adjusted_fitness * multiplier)
                    .max(F::zero())
                    .round()
                    .to_usize()
                    .unwrap()
                    .min(self.max_individuals_ - assigned_count)
            } else {
                0
            };
            assigned_count += to_assign;
            spec.max_topologies = to_assign;
        }
        self.ev_number_.reset();
        let ev_number = self.ev_number_.clone();
        let proba = self.proba.clone();
        #[cfg(any(debug_assertions, target_arch = "wasm32"))]
        {
            self.species_.iter_mut().for_each(|species| {
                species
                    .get_mut()
                    .unwrap()
                    .natural_selection(ev_number.clone(), proba.clone());
            });
        }
        #[cfg(all(not(debug_assertions), not(target_arch = "wasm32")))]
        {
            self.species_.par_iter_mut().for_each(|species| {
                species
                    .lock()
                    .unwrap()
                    .natural_selection(ev_number.clone(), proba.clone());
            });
        }

        let mut species_sizes_vec: Vec<(usize, usize)> = Vec::new();
        let mut current_count: (usize, usize) = (0, 0);
        for mutex in &self.species_ {
            let Species { topologies, .. } = &*mutex.lock().unwrap();
            if topologies.len() == current_count.0 {
                current_count.1 += 1;
            } else {
                if current_count.1 != 0 {
                    species_sizes_vec.push(current_count);
                }
                current_count = (topologies.len(), 1);
            }
        }
        species_sizes_vec.push(current_count);
        let lengths_str = species_sizes_vec
            .iter()
            .map(|(value, count)| format!("{} x {}", count, value))
            .join(" | ");
        println!("SPECIES LENGTHS: {}", lengths_str);
    }

    fn push_to_history(&mut self) {
        if self.species_.is_empty() {
            return;
        }
        self.species_.sort_by(|s1, s2| {
            s1.lock()
                .unwrap()
                .score()
                .partial_cmp(&s2.lock().unwrap().score())
                .unwrap()
        });

        let best = { self.species_.last().unwrap().lock().unwrap().score() };

        {
            println!(
                "BEST OF WORST: {} BEST: {}",
                self.species_[0].lock().unwrap().score(),
                best
            );
        }
        if best > self.best_historical_score {
            self.best_historical_score = best;
            self.no_progress_counter = 0;
        } else {
            self.no_progress_counter += 1;
            if self.no_progress_counter >= self.iterations_ / 10 && self.iterations_ > 500 {
                println!("=========================RESET TO TWO SPECIES=========================");
                self.best_historical_score = F::zero();
                self.no_progress_counter = 0;
                if self.species_.len() > 2 {
                    self.species_ = self.species_.split_off(self.species_.len() - 2);
                }
            }
        }

        for species in self.species_.iter() {
            self.history_
                .push(species.lock().unwrap().best_topology.clone())
        }
    }

    fn reset_species(&mut self) {
        self.collect_topologies();
        cond_iter_mut!(self.species_).for_each(|spec| {
            spec.get_mut().unwrap().topologies.clear();
        });
        let mut species = self.species_.split_off(0);
        let topologies = self.topologies_.clone();
        let delta_t = self.delta_threshold_;
        let (c1, c2, c3) = (self.c1_, self.c2_, self.c3_);
        let mut new_species = cond_iter!(topologies)
            .filter_map(|topology_rc| {
                let top_cp = topology_rc.clone();
                // We could have the same topology in a species twice if it was one of the best
                let mut assigned = false;
                {
                    let top1 = top_cp.lock().unwrap();
                    for spec in &species {
                        let top2 = {
                            let spec = &*spec.lock().unwrap();
                            spec.best_topology.clone()
                        };
                        let delta = Topology::delta_compatibility(&*top1, &top2, c1, c2, c3);
                        if delta <= delta_t {
                            let spec = &mut *spec.lock().unwrap();
                            spec.push(topology_rc.clone());
                            assigned = true;
                            break;
                        }
                    }
                }
                if !assigned {
                    let new_species = Species::new(topology_rc.clone());
                    Some(Mutex::new(new_species))
                } else {
                    None
                }
            })
            .collect::<Vec<Mutex<Species<F>>>>();
        species.append(&mut new_species);
        self.species_ = species;
        self.species_
            .retain(|spec| !spec.lock().unwrap().topologies.is_empty());
        let biggest_species = cond_iter!(self.species_)
            .map(|spec| spec.lock().unwrap().topologies.len())
            .max()
            .unwrap_or(0);
        println!("BIGGEST SPECIES: {}", biggest_species);
    }
}

impl<'a, T, F> Train<'a, T, F>
where
    T: GameAsync<F>,
    F: 'a + Float + Sum + Display + std::ops::AddAssign + std::ops::SubAssign + Send + Sync,
    &'a [F]: rayon::iter::IntoParallelIterator,
{
    pub async fn start_async(&mut self) -> Result<(), TrainingError> {
        let inputs = self.inputs_.ok_or(TrainingError::NoInput)?;

        let outputs = self.inputs_.ok_or(TrainingError::NoInput)?;

        self.species_.push(Mutex::new(Species::new_uniform(
            inputs,
            outputs,
            self.max_layers_,
            self.max_per_layers_,
            &self.ev_number_,
        )));

        self.reset_players();
        for i in 0..self.iterations_ {
            section!();
            println!("Generation {}", i);
            let now = Instant::now();
            let results = self.simulation.run_generation_async().await;
            println!("RUN GENERATION: {}ms", now.elapsed().as_millis());
            self.set_last_results(&results);
            let now = Instant::now();
            self.natural_selection();
            self.push_to_history();
            self.reset_species();
            if self.species_.is_empty() {
                break;
            }
            println!("NATURAL SELECTION: {}ms", now.elapsed().as_millis());
            let now = Instant::now();
            self.reset_players();
            println!("RESET PLAYERS: {}ms", now.elapsed().as_millis());
            let mut cb_option = self.access_train_object_fn.take();
            let cb_option_borrow = &mut cb_option;
            if let Some(cb) = cb_option_borrow {
                (*cb)(self);
                self.access_train_object_fn = cb_option;
            }
        }
        println!("POST TRAINING");
        self.simulation.post_training(&*self.history_);
        Ok(())
    }
}
