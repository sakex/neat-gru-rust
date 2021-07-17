# neat-gru-rust

## Documentation
[Crates.io doc](https://docs.rs/neat-gru/)

## How to use
In `Cargo.toml`:
```
[dependencies]
neat-gru = "0.6.4"
```
Create a struct that implements the `Game` trait
```rust
use neat_gru::game::Game;
use neat_gru::neural_network::nn::NeuralNetwork;
use neat_gru::topology::topology::Topology;
struct Player {
    net: NeuralNetwork<f64>,
    score: f64,
}

impl Player {
    pub fn new(net: NeuralNetwork<f64>) -> Player {
        Player {
            net: net,
            score: 0f64,
        }
    }
}

struct Simulation {
    players: Vec<Player>,
}

impl Simulation {
    pub fn new() -> Simulation {
        Simulation {
            players: Vec::new(),
        }
    }
}

impl Game<f64> for Simulation {
    // Loss function
    fn run_generation(&mut self) -> Vec<f64> {
        self.players.iter().map(|p| p.score).collect()
    }

    // Reset networks
    fn reset_players(&mut self, nets: Vec<NeuralNetwork<f64>>) {
        self.players.clear();
        self.players.reserve(nets.len());
        self.players = nets
            .into_iter()
            .map(|net| Player::new(net.clone()))
            .collect();
    }

    // Called at the end of training
    fn post_training(&mut self, history: &[Topology<f64>]) {
        // Iter on best topologies and upload the best one
    }
}

```
Launch a training
```rust
let mut sim = Simulation::new();
         
let mut runner = Train::new(&mut sim);
runner
     .inputs(input_count)
     .outputs(output_count as i32)
     .iterations(nb_generations as i32)
     .max_layers((hidden_layers + 2) as i32)
     .max_per_layers(hidden_layers as i32)
     .max_species(max_species as i32)
     .max_individuals(max_individuals as i32)
     .start();
```
