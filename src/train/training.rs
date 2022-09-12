use crate::game::{Game, GameAsync};
#[cfg(target_arch = "wasm32")]
use crate::instant_wasm_replacement::Instant;
use crate::neural_network::NeuralNetwork;
use crate::section;
use crate::topology::mutation_probabilities::MutationProbabilities;
use crate::topology::{Topology, TopologySmrtPtr};
use crate::train::error::TrainingError;
use crate::train::evolution_number::EvNumber;
use crate::train::species::Species;
use itertools::Itertools;
use num::Float;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt::Display;
use std::fs::File;
use std::io::{self, BufReader, Write};
use std::iter::Sum;
use std::sync::{Arc, Mutex};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
use tempdir::TempDir;

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

const TEMP_DIR: &str = "temp_history";

pub type TrainAccessCallback<'a, T, F> = Box<dyn FnMut(&mut Train<'a, T, F>)>;

pub struct HistoricTopology<F>
where
    F: Float + std::ops::AddAssign + Display + Send,
{
    pub topology: Topology<F>,
    pub generation: usize,
}

impl<F> std::ops::Deref for HistoricTopology<F>
where
    F: Float + std::ops::AddAssign + Display + Send,
{
    type Target = Topology<F>;

    fn deref(&self) -> &Self::Target {
        &self.topology
    }
}

#[derive(Serialize, Deserialize)]
struct HistoricTopologyDisk {
    topology: String,
    generation: usize,
}

impl<F> From<HistoricTopology<F>> for HistoricTopologyDisk
where
    F: Float + std::ops::AddAssign + Display + Send,
{
    fn from(history: HistoricTopology<F>) -> HistoricTopologyDisk {
        HistoricTopologyDisk {
            topology: history.topology.to_string(),
            generation: history.generation,
        }
    }
}

impl<F> From<HistoricTopologyDisk> for HistoricTopology<F>
where
    F: Float + std::ops::AddAssign + Display + Send,
{
    fn from(disk: HistoricTopologyDisk) -> HistoricTopology<F> {
        HistoricTopology {
            topology: Topology::from_string(&disk.topology),
            generation: disk.generation,
        }
    }
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
    crossovers_: bool,
    inputs_: Option<usize>,
    outputs_: Option<usize>,
    topologies_: Vec<TopologySmrtPtr<F>>,
    species_: Vec<Mutex<Species<F>>>,
    history_: Vec<HistoricTopology<F>>,
    ev_number_: Arc<EvNumber>,
    save_history_to_disk_: bool,
    best_historical_score_: F,
    no_progress_counter_: usize,
    proba_: MutationProbabilities,
    access_train_object_fn_: Option<TrainAccessCallback<'a, T, F>>,
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
    /// use neat_gru::neural_network::NeuralNetwork;
    /// use neat_gru::topology::Topology;
    /// use neat_gru::train::HistoricTopology;
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
    ///     fn post_training(&mut self, _history: &[HistoricTopology<f64>]) {}
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
            crossovers_: true,
            inputs_,
            outputs_,
            topologies_: Vec::new(),
            species_: Vec::new(),
            history_: Vec::new(),
            ev_number_: Arc::new(EvNumber::new()),
            best_historical_score_: F::zero(),
            no_progress_counter_: 0,
            access_train_object_fn_: None,
            proba_: MutationProbabilities {
                change_weights: 0.95,
                guaranteed_new_neuron: 0.2,
            },
            save_history_to_disk_: false,
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
        self.proba_ = proba;
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

    /// Specifies if we should run crossovers or not
    ///
    /// This function is optional as the crossovers are run by default
    ///
    /// # Arguments
    ///
    /// `should_run` - Whether crossover should be run or not
    #[inline]
    pub fn crossovers(&mut self, should_run: bool) -> &mut Self {
        self.crossovers_ = should_run;
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
        self.access_train_object_fn_ = Some(callback);
        self
    }

    fn run_iterations(&mut self, tempdir: &Option<TempDir>) -> Result<(), TrainingError> {
        for i in 0..self.iterations_ {
            section!();
            log::info!("Generation {}", i);
            let now = Instant::now();
            let results = self.simulation.run_generation();
            log::info!("RUN GENERATION: {}ms", now.elapsed().as_millis());
            self.set_last_results(results);
            let now = Instant::now();
            self.natural_selection();
            self.push_to_history(i, tempdir)?;
            self.reset_species();
            if self.species_.is_empty() {
                break;
            }
            log::info!("NATURAL SELECTION: {}ms", now.elapsed().as_millis());
            let now = Instant::now();
            self.reset_players();
            log::info!("RESET PLAYERS: {}ms", now.elapsed().as_millis());
            let mut cb_option = self.access_train_object_fn_.take();
            let cb_option_borrow = &mut cb_option;
            if let Some(cb) = cb_option_borrow {
                (*cb)(self);
                self.access_train_object_fn_ = cb_option;
            }
        }

        Ok(())
    }

    /// If set to true, saves the history in the disk instead of keeping in RAM to prevent memory leak.
    /// The files are saved in a TempFile
    ///
    /// Defaults to false
    ///
    /// # Arguments
    ///
    /// `should_use_disk` - Whether to save history on disk
    #[inline]
    pub fn save_history_to_disk(&mut self, should_use_disk: bool) -> &mut Self {
        self.save_history_to_disk_ = should_use_disk;
        self
    }

    fn create_temp_dir() -> Result<TempDir, io::Error> {
        let tmp_dir = TempDir::new(TEMP_DIR)?;
        Ok(tmp_dir)
    }

    fn load_post_training_from_temp(
        tempdir: &TempDir,
    ) -> Result<Vec<HistoricTopology<F>>, io::Error> {
        let mut saved_topologies: Vec<HistoricTopology<F>> = Vec::new();
        let path = tempdir.path();
        let files = std::fs::read_dir(path)?;
        for file_path in files {
            let file_path = file_path?;
            let is_file = file_path.file_type()?.is_file();
            if is_file {
                let file = std::fs::File::open(&file_path.path())?;
                let reader = BufReader::new(file);

                let topology =
                    if let Ok(top) = serde_json::from_reader::<_, HistoricTopologyDisk>(reader) {
                        top.into()
                    } else {
                        return Err(io::ErrorKind::InvalidData.into());
                    };
                saved_topologies.push(topology);
            }
        }

        Ok(saved_topologies)
    }

    /// Starts the training.
    ///
    /// May return a NoInput Error if no input or output is given
    #[inline]
    pub fn start(&mut self) -> Result<(), TrainingError> {
        let topologies_tmp_dir = if self.save_history_to_disk_ {
            Some(Self::create_temp_dir().map_err(TrainingError::from)?)
        } else {
            None
        };
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
        self.run_iterations(&topologies_tmp_dir)?;
        section!();
        log::info!("POST TRAINING");
        if let Some(topologies_tmp_dir) = topologies_tmp_dir {
            let topologies_on_disk = Self::load_post_training_from_temp(&topologies_tmp_dir)?;
            for topology in topologies_on_disk {
                self.history_.push(topology);
            }
        }
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
        log::info!(
            "TOPOLOGIES: {}, SPECIES: {}",
            networks.len(),
            self.species_.len()
        );
        self.simulation.reset_players(networks);
    }

    fn set_last_results(&mut self, results: Vec<F>) {
        cond_iter_mut!(self.topologies_)
            .zip(cond_iter!(results))
            .for_each(|(topology, result)| {
                if result.is_nan() {
                    panic!("NaN result");
                }
                topology.lock().unwrap().set_last_result(*result);
            })
    }

    /// Calculates the mean
    fn calculate_mean(&self) -> F {
        cond_iter!(self.species_)
            .clone()
            .map(|spec| spec.lock().unwrap().adjusted_fitness)
            .sum::<F>()
            / F::from(self.species_.len()).unwrap()
    }

    /// Calculates the variance
    fn calculate_variance(&self, mean: F) -> F {
        cond_iter!(self.species_)
            .clone()
            .map(|spec| (spec.lock().unwrap().adjusted_fitness - mean).powf(F::from(2.).unwrap()))
            .sum::<F>()
            / F::from(self.species_.len() - 1).unwrap()
    }

    /// Gets the species lengths as a string
    fn get_species_lengths(&self, species_sizes_vec: Vec<(usize, usize)>) -> String {
        species_sizes_vec
            .iter()
            .map(|(value, count)| format!("{} x {}", count, value))
            .join(" | ")
    }

    fn natural_selection(&mut self) {
        self.species_
            .retain(|spec| spec.lock().unwrap().stagnation_counter < 20);
        match self.species_.len() {
            0 => return,
            1 => {
                let first_spec = &mut *self.species_[0].lock().unwrap();
                first_spec.max_topologies = self.max_individuals_;
                self.ev_number_.reset();
                let ev_number = self.ev_number_.clone();
                first_spec.natural_selection(ev_number, self.proba_.clone(), self.crossovers_);
                return;
            }
            _ => {}
        };
        cond_iter_mut!(self.species_).for_each(|spec| {
            spec.get_mut().unwrap().compute_adjusted_fitness();
        });

        let mean = self.calculate_mean();
        let variance: F = self.calculate_variance(mean);

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
        self.sort_species(variance);
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
        let proba = self.proba_.clone();
        let run_crossovers = self.crossovers_;
        cond_iter_mut!(self.species_).for_each(|species| {
            species.get_mut().unwrap().natural_selection(
                ev_number.clone(),
                proba.clone(),
                run_crossovers,
            );
        });

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
        log::info!(
            "SPECIES LENGTHS: {}",
            self.get_species_lengths(species_sizes_vec)
        );
    }

    fn push_to_history(
        &mut self,
        generation: usize,
        tempdir: &Option<TempDir>,
    ) -> Result<(), io::Error> {
        if self.species_.is_empty() {
            return Ok(());
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
            log::info!(
                "BEST OF WORST: {} BEST: {}",
                self.species_[0].lock().unwrap().score(),
                best
            );
        }
        if best > self.best_historical_score_ {
            self.best_historical_score_ = best;
            self.no_progress_counter_ = 0;
        } else {
            self.no_progress_counter_ += 1;
            if self.no_progress_counter_ >= self.iterations_ / 10 && self.iterations_ > 500 {
                log::info!(
                    "=========================RESET TO TWO SPECIES========================="
                );
                self.best_historical_score_ = F::zero();
                self.no_progress_counter_ = 0;
                if self.species_.len() > 2 {
                    self.species_ = self.species_.split_off(self.species_.len() - 2);
                }
            }
        }

        for (idx, species) in self.species_.iter().enumerate() {
            let topology_history = HistoricTopology {
                topology: species.lock().unwrap().best_topology.clone(),
                generation,
            };
            if let Some(tempdir) = tempdir {
                let file_path = tempdir
                    .path()
                    .join(format!("generation-{}-species-{}.json", generation, idx));
                let mut tmp_file = File::create(file_path)?;
                let disk_topology_history: HistoricTopologyDisk = topology_history.into();
                match serde_json::to_string(&disk_topology_history) {
                    Ok(serialized) => tmp_file.write_all(serialized.as_bytes())?,
                    Err(e) => log::error!("Failed to serialize with error: {:?}", e),
                }
            } else {
                self.history_.push(topology_history);
            }
        }
        Ok(())
    }

    /// Sorts the species according to fitness
    fn sort_species(&mut self, variance: F) {
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
        log::info!("BIGGEST SPECIES: {}", self.get_biggest_species_len());
    }

    /// Gets the length of the biggest species
    fn get_biggest_species_len(&self) -> usize {
        cond_iter!(self.species_)
            .map(|spec| spec.lock().unwrap().topologies.len())
            .max()
            .unwrap_or(0)
    }
}

impl<'a, T, F> Train<'a, T, F>
where
    T: GameAsync<F>,
    F: 'a + Float + Sum + Display + std::ops::AddAssign + std::ops::SubAssign + Send + Sync,
    &'a [F]: rayon::iter::IntoParallelIterator,
{
    pub async fn start_async(&mut self) -> Result<(), TrainingError> {
        let topologies_tmp_dir = if self.save_history_to_disk_ {
            Some(Self::create_temp_dir().map_err(TrainingError::from)?)
        } else {
            None
        };
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
            log::info!("Generation {}", i);
            let now = Instant::now();
            let results = self.simulation.run_generation_async().await;
            log::info!("RUN GENERATION: {}ms", now.elapsed().as_millis());
            self.set_last_results(results);
            let now = Instant::now();
            self.natural_selection();
            self.push_to_history(i, &topologies_tmp_dir)?;
            self.reset_species();
            if self.species_.is_empty() {
                break;
            }
            log::info!("NATURAL SELECTION: {}ms", now.elapsed().as_millis());
            let now = Instant::now();
            self.reset_players();
            log::info!("RESET PLAYERS: {}ms", now.elapsed().as_millis());
            let mut cb_option = self.access_train_object_fn_.take();
            let cb_option_borrow = &mut cb_option;
            if let Some(cb) = cb_option_borrow {
                (*cb)(self);
                self.access_train_object_fn_ = cb_option;
            }
        }
        log::info!("POST TRAINING");
        if let Some(topologies_tmp_dir) = topologies_tmp_dir {
            let topologies_on_disk = Self::load_post_training_from_temp(&topologies_tmp_dir)?;
            for topology in topologies_on_disk {
                self.history_.push(topology);
            }
        }
        self.simulation.post_training(&*self.history_);
        Ok(())
    }
}
