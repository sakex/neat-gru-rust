use crate::game::Game;
use num::Float;
use std::rc::Rc;
use crate::topology::topology::Topology;
use crate::train::species::Species;
use crate::neural_network::nn::NeuralNetwork;
use std::cell::RefCell;

/// The train struct is used to train a Neural Network on a simulation with the NEAT algorithm
pub struct Train<'a, T, F>
    where F: Float, T: Game<F> {
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
}

impl<'a, T, F> Train<'a, T, F>
    where T: Game<F>, F: Float {
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
    /// fn reset_players(&mut self,nets: &[NeuralNetwork<f64>]) { }
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
    pub fn start(&mut self) {
        let inputs = match self.inputs_ {
            Some(v) => v,
            None => { panic!("Didn't provide a number of inputs") }
        };

        let outputs = match self.outputs_ {
            Some(v) => v,
            None => { panic!("Didn't provide a number of inputs") }
        };

        self.species_ = vec![Species::new_random(self.max_individuals_, inputs, outputs, self.max_layers_, self.max_per_layers_)];

        self.reset_players();
        for _i in 0..self.iterations_ {
            let results = self.simulation.run_generation();
            self.set_last_results(&results);
            self.natural_selection();
            self.reset_players();
        }
    }

    fn reset_players(&mut self) {
        self.topologies_.clear();
        self.topologies_ = self.species_.iter()
            .map(|species| species.topologies.iter()
                .map(|top| top.clone())
                .collect::<Vec<Rc<RefCell<Topology<F>>>>>()
            ).flatten().collect();
        let networks: Vec<NeuralNetwork<F>> = self.topologies_.iter().map(|top_rc| {
            let top = &*top_rc.borrow();
            NeuralNetwork::new(&top)
        }).collect();
        self.simulation.reset_players(&networks);
    }

    fn set_last_results(&mut self, results: &Vec<F>) {
        for (topology, result) in self.topologies_.iter_mut().zip(results.iter()) {
            topology.borrow_mut().set_last_result(*result);
        }
    }

    fn natural_selection(&mut self) {
        for species in self.species_.iter_mut() {
            species.natural_selection();
        }
    }
}