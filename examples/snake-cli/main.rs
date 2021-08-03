use defs::{OUTPUTS, SNAKE_COUNT};
use neat_gru::train::train::Train;
use training_simulation::TrainingSimulation;

mod apple;
mod coordinate;
pub mod defs;
mod direction;
mod game;
mod snake;
mod training_simulation;
mod utils;

fn main() {
    run_training();
}

fn run_training() {
    let mut sim = TrainingSimulation::new();
    let mut runner: Train<TrainingSimulation, f64> = Train::new(&mut sim);
    runner
        .inputs(5)
        .outputs(OUTPUTS)
        .iterations(500)
        .delta_threshold(2.)
        .formula(0.8, 0.8, 0.3)
        .max_layers(10)
        .max_individuals(SNAKE_COUNT)
        .access_train_object(Box::new(|train| {
            let species_count = train.species_count();
            train.simulation.species_count = species_count;
        }))
        .start()
        .unwrap();
}
