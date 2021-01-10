use crate::game::Game;
use crate::neural_network::nn::NeuralNetwork;
use crate::topology::mutation_probabilities::MutationProbabilities;
use crate::topology::topology::Topology;
use crate::train::evolution_number::EvNumber;
use crate::train::species::Species;
use num::Float;
use rayon::prelude::*;
use std::cell::RefCell;
use std::fmt::Display;
use std::iter::Sum;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

/// The train struct is used to train a Neural Network on a simulation with the NEAT algorithm
pub struct Train<'a, T, F>
where
    F: Float + Sum + Display + std::ops::AddAssign + std::ops::SubAssign + Send + Sync,
    T: Game<F>,
{
    simulation: &'a mut T,
    iterations_: usize,
    max_individuals_: usize,
    max_species_: usize,
    max_layers_: usize,
    max_per_layers_: usize,
    delta_threshold_: F,
    inputs_: Option<usize>,
    outputs_: Option<usize>,
    topologies_: Vec<Rc<RefCell<Topology<F>>>>,
    species_: Vec<Species<F>>,
    history_: Vec<Topology<F>>,
    ev_number_: Arc<EvNumber>,
    best_historical_score: F,
    no_progress_counter: usize,
    proba: MutationProbabilities,
}

impl<'a, T, F> Train<'a, T, F>
where
    T: Game<F>,
    F: Float + Sum + Display + std::ops::AddAssign + std::ops::SubAssign + Send + Sync,
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
        let max_species_: usize = 100;
        let inputs_ = None;
        let outputs_ = None;

        Train {
            simulation,
            iterations_,
            max_individuals_,
            max_species_,
            max_layers_: 4,
            max_per_layers_: 20,
            delta_threshold_: F::from(3).unwrap(),
            inputs_,
            outputs_,
            topologies_: Vec::new(),
            species_: Vec::new(),
            history_: Vec::new(),
            ev_number_: Arc::new(EvNumber::new()),
            best_historical_score: F::zero(),
            no_progress_counter: 0,
            proba: MutationProbabilities {
                change_weights: 0.4,
                guaranteed_new_neuron: 0.1,
                delete_neuron: 0.1,
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
    /// `v` - The number of networks per generation
    #[inline]
    pub fn max_individuals(&mut self, v: usize) -> &mut Self {
        self.max_individuals_ = v;
        self
    }

    /// Sets the number of maximum species per generation
    ///
    /// This function is optional as the number of max species defaults to 100
    ///
    /// # Arguments
    ///
    /// `v` - The number of maximum species per generation
    #[inline]
    pub fn max_species(&mut self, v: usize) -> &mut Self {
        self.max_species_ = v;
        self
    }

    /// Sets the delta threshold at which two topologies don't belong to the same species
    ///
    /// This function is optional as the number of max individuals defaults to 100
    ///
    /// # Arguments
    ///
    /// `v` - The new delta threshold
    #[inline]
    pub fn delta_threshold(&mut self, v: F) -> &mut Self {
        self.delta_threshold_ = v;
        self
    }

    /// Sets the probabilities of different mutations
    ///
    /// # Arguments
    ///
    /// `v` - The new probabilities
    #[inline]
    pub fn mutation_probabilities(&mut self, v: MutationProbabilities) -> &mut Self {
        self.proba = v;
        self
    }

    /// Sets the number of neurons on the first layer
    ///
    /// This function has to be called in order to start training
    ///
    /// # Arguments
    ///
    /// `v` - The number of neurons on the first layer
    #[inline]
    pub fn inputs(&mut self, v: usize) -> &mut Self {
        self.inputs_ = Some(v);
        self
    }

    /// Sets the number of neurons on the last layer
    ///
    /// This function has to be called in order to start training
    ///
    /// # Arguments
    ///
    /// `v` - The number of neurons on the last layer
    #[inline]
    pub fn outputs(&mut self, v: usize) -> &mut Self {
        self.outputs_ = Some(v);
        self
    }

    /// Sets the maximum number of layers for the networks
    ///
    /// This function is optional as the max number of layers defaults to 4
    ///
    /// # Arguments
    ///
    /// `v` - The maximum number of layers
    #[inline]
    pub fn max_layers(&mut self, v: usize) -> &mut Self {
        self.max_layers_ = v;
        self
    }

    /// Sets the maximum number of neurons per layers for the networks
    ///
    /// This function is optional as the max neurons per layer defaults to 50
    ///
    /// # Arguments
    ///
    /// `v` - The maximum number of neurons per layers
    #[inline]
    pub fn max_per_layers(&mut self, v: usize) -> &mut Self {
        self.max_per_layers_ = v;
        self
    }

    #[inline]
    pub fn start(&mut self) {
        let inputs = match self.inputs_ {
            Some(v) => v,
            None => panic!("Didn't provide a number of inputs"),
        };

        let outputs = match self.outputs_ {
            Some(v) => v,
            None => panic!("Didn't provide a number of inputs"),
        };

        self.species_.push(Species::new_uniform(
            inputs,
            outputs,
            self.max_layers_,
            self.max_per_layers_,
            &self.ev_number_,
        ));

        self.reset_players();
        for i in 0..self.iterations_ {
            println!("\n=========================\n");
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
        }
        println!("\n=========================\n");
        println!("POST TRAINING");
        self.simulation.post_training(&*self.history_);
    }

    fn get_topologies(&mut self) {
        self.topologies_ = self
            .species_
            .iter()
            .map(|species| {
                species
                    .topologies
                    .iter()
                    .map(|top| top.clone())
                    .collect::<Vec<Rc<RefCell<Topology<F>>>>>()
            })
            .flatten()
            .collect();
    }

    fn reset_players(&mut self) {
        self.get_topologies();

        let networks: Vec<NeuralNetwork<F>> = self
            .topologies_
            .iter()
            .map(|top_rc| {
                let top = &*top_rc.borrow();
                unsafe { NeuralNetwork::new(&top) }
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
        for (topology, result) in self.topologies_.iter_mut().zip(results.iter()) {
            topology.borrow_mut().set_last_result(*result);
        }
    }

    fn natural_selection(&mut self) {
        self.species_.retain(|spec| spec.stagnation_counter < 15);
        if self.species_.len() == 1 {
            self.species_[0].max_topologies = self.max_individuals_;
            self.ev_number_.reset();
            let ev_number = self.ev_number_.clone();
            self.species_[0].natural_selection(ev_number.clone(), self.proba.clone());
            return;
        }
        if self.species_.is_empty() {
            return;
        }
        self.species_.iter_mut().for_each(|spec| {
            spec.compute_adjusted_fitness();
        });
        let mean = self
            .species_
            .par_iter()
            .map(|spec| spec.adjusted_fitness)
            .sum::<F>()
            / F::from(self.species_.len()).unwrap();
        let variance = self
            .species_
            .par_iter()
            .map(|spec| (spec.adjusted_fitness - mean).powf(F::from(2.).unwrap()))
            .sum::<F>()
            / F::from(self.species_.len() - 1).unwrap();
        let volatility = variance.sqrt();
        self.species_.iter_mut().for_each(|spec| {
            spec.adjusted_fitness = F::from(1.1)
                .unwrap()
                .powf((spec.adjusted_fitness - mean) / volatility);
        });

        self.species_.sort_by(|spec1, spec2| {
            spec1
                .adjusted_fitness
                .partial_cmp(&spec2.adjusted_fitness)
                .expect(&*format!(
                    "First: {}, second: {}",
                    spec1.adjusted_fitness, spec2.adjusted_fitness
                ))
        });
        let sum: F = self
            .species_
            .iter()
            .map(|spec| spec.adjusted_fitness.clone())
            .sum();
        let multiplier: F = F::from(self.max_individuals_).unwrap() / sum.clone();
        let mut assigned_count: usize = 0;
        for spec in self.species_.iter_mut() {
            let to_assign = (spec.adjusted_fitness * multiplier)
                .max(F::zero())
                .round()
                .to_usize()
                .unwrap()
                .min(self.max_individuals_ - assigned_count);
            assigned_count += to_assign;
            spec.max_topologies = to_assign;
        }
        self.ev_number_.reset();
        let ev_number = self.ev_number_.clone();
        let proba = self.proba.clone();
        #[cfg(debug_assertions)]
        {
            self.species_.iter_mut().for_each(|species| {
                species.natural_selection(ev_number.clone(), proba.clone());
            });
        }
        #[cfg(not(debug_assertions))]
        {
            self.species_.par_iter_mut().for_each(|species| {
                species.natural_selection(ev_number.clone(), proba.clone());
            });
        }
    }

    fn push_to_history(&mut self) {
        if self.species_.is_empty() {
            return;
        }
        self.species_
            .sort_by(|s1, s2| s1.score().partial_cmp(&s2.score()).unwrap());

        let best = self.species_.last().unwrap().score();

        println!("BEST OF WORST: {} BEST: {}", self.species_[0].score(), best);

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
            self.history_.push(species.get_best())
        }
    }

    fn reset_species(&mut self) {
        self.get_topologies();
        for spec in &mut self.species_ {
            spec.topologies.clear();
        }
        for topology_rc in self.topologies_.iter() {
            let top_cp = topology_rc.clone();
            // We could have the same topology in a species twice if it was one of the best
            let mut is_one_of_best = false;
            for spec in &self.species_ {
                let best_top_rc = &spec.best_topology;
                if Rc::ptr_eq(best_top_rc, &top_cp) {
                    is_one_of_best = true;
                    break;
                }
            }
            if is_one_of_best {
                continue;
            }
            let top_borrow = top_cp.borrow();
            let mut assigned = false;
            for spec in self.species_.iter_mut() {
                let top2 = spec.get_best();
                let delta = Topology::delta_compatibility(&top_borrow, &top2);
                if delta <= self.delta_threshold_ {
                    spec.push(topology_rc.clone());
                    assigned = true;
                    break;
                }
            }
            if !assigned {
                let new_species = Species::new(topology_rc.clone());
                self.species_.push(new_species);
            }
        }
        self.species_.retain(|spec| spec.topologies.len() > 0);
        let biggest_species = self
            .species_
            .iter()
            .map(|spec| spec.topologies.len())
            .max()
            .unwrap_or(0);
        println!("BIGGEST SPECIES: {}", biggest_species);
    }
}
