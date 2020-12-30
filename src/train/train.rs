use crate::game::Game;
use crate::neural_network::nn::NeuralNetwork;
use crate::topology::topology::Topology;
use crate::train::evolution_number::EvNumber;
use crate::train::species::Species;
use num::Float;
use std::cell::RefCell;
use std::fmt::Display;
use std::iter::Sum;
use std::rc::Rc;
use std::time::Instant;

/// The train struct is used to train a Neural Network on a simulation with the NEAT algorithm
pub struct Train<'a, T, F>
where
    F: Float + Sum + Display,
    T: Game<F>,
{
    simulation: &'a mut T,
    iterations_: usize,
    max_individuals_: usize,
    max_species_: usize,
    max_layers_: usize,
    max_per_layers_: usize,
    inputs_: Option<usize>,
    outputs_: Option<usize>,
    topologies_: Vec<Rc<RefCell<Topology<F>>>>,
    species_: Vec<Species<F>>,
    history_: Vec<Topology<F>>,
    ev_number_: EvNumber,
    best_historical_score: F,
    no_progress_counter: usize,
}

impl<'a, T, F> Train<'a, T, F>
where
    T: Game<F>,
    F: Float + Sum + Display,
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
            inputs_,
            outputs_,
            topologies_: Vec::new(),
            species_: Vec::new(),
            history_: Vec::new(),
            ev_number_: EvNumber::new(),
            best_historical_score: F::zero(),
            no_progress_counter: 0,
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
    /// This function is optional as the max number of layers defaults to 10
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

    /// Starts the training with the given parameters
    ///
    /// # Example
    ///
    /// ```
    /// use neat_gru::train::train::Train;
    /// use neat_gru::game::Game;
    /// use neat_gru::topology::topology::Topology;
    /// use neat_gru::neural_network::nn::NeuralNetwork;
    /// struct Simulation {
    /// }
    ///
    /// impl Game<f64> for Simulation {
    ///     fn run_generation(&mut self) -> Vec<f64> {
    ///         vec![1.; 5]
    ///     }
    ///
    /// fn reset_players(&mut self,nets: Vec<NeuralNetwork<f64>>) { }
    ///
    /// fn post_training(&mut self,history: &[Topology<f64>]) { }
    ///
    /// }
    ///
    /// let mut sim = Simulation {}; // Has to implement trait Game
    /// let mut runner: Train<Simulation, f64> = Train::new(&mut sim);
    /// runner.max_individuals(5).inputs(5).outputs(1);
    /// runner.start();
    /// ```
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
        self.species_
            .sort_by(|s1, s2| s1.score().partial_cmp(&s2.score()).unwrap());

        let best = self.species_.last().unwrap().score();

        println!(
            "BEST OF WORST: {} BEST: {}",
            num::cast::<F, f32>(self.species_[0].score()).unwrap(),
            best
        );

        if best > self.best_historical_score {
            self.best_historical_score = best;
            self.no_progress_counter = 0;
        } else {
            self.no_progress_counter += 1;
            if self.no_progress_counter >= 20 {
                self.best_historical_score = F::zero();
                self.no_progress_counter = 0;
                self.species_ = self.species_.split_off(self.species_.len() - 2);
            }
        }

        for species in self.species_.iter() {
            self.history_.push(species.get_best())
        }

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
        self.species_.iter_mut().for_each(|spec| {
            spec.compute_adjusted_fitness();
        });
        self.species_.sort_by(|spec1, spec2| {
            spec1
                .adjusted_fitness
                .partial_cmp(spec2.adjusted_fitness)
                .unwrap()
        });
        if self.species_.len() >= self.max_species_ {
            self.species_ = self.species_.split_off(self.species_.len() / 2);
        }
        let sum: F = self
            .species_
            .iter()
            .map(|spec| spec.adjusted_fitness.clone())
            .sum();
        let multiplier: F = F::from(self.max_individuals_).unwrap() / sum.clone();
        for spec in self.species_.iter_mut() {
            spec.max_topologies = (spec.adjusted_fitness * multiplier)
                .round()
                .to_usize()
                .unwrap();
        }
        self.ev_number_.reset();
        for species in self.species_.iter_mut() {
            species.natural_selection(&self.ev_number_);
        }
    }

    fn reset_species(&mut self) {
        self.species_.retain(|spec| spec.stagnation_counter < 15);
        self.get_topologies();
        for topology_rc in self.topologies_.iter() {
            let top_cp = topology_rc.clone();
            let top_borrow = top_cp.borrow();
            let mut assigned = false;
            for spec in self.species_.iter_mut() {
                let top2 = spec.get_best();
                let delta = Topology::delta_compatibility(&top_borrow, &top2);
                if delta <= F::from(3).unwrap() {
                    spec.push(spec.best_topology.clone());
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
    }
}
