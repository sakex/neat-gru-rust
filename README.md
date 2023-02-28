# neat-gru-rust

[![CICD](https://github.com/Nereuxofficial/neat-gru-rust/actions/workflows/test.yml/badge.svg)](https://github.com/Nereuxofficial/neat-gru-rust/actions/workflows/test.yml)
![Crates.io](https://img.shields.io/crates/v/neat-gru?style=for-the-badge)
![Downloads](https://img.shields.io/crates/d/neat-gru?style=for-the-badge)
![License](https://img.shields.io/github/license/sakex/neat-gru-rust?style=for-the-badge)

## Documentation

[Crates.io doc](https://docs.rs/neat-gru/)

## Install & Build for C / C++

Build the dynamic library and install it in `/usr/local/bin`, copy headers in `/usr/local/include/neat_gru_lib`.
This command will ask for your root password as it is needed to do the installation.

```bash
./bootstrap.sh
```

## Use in C / C++

Link `neat_gru` dynamic library.


```
#include "neat_gru_lib/neat_gru.h"
```

C

```c
const char path[] = "path.json";

const NeatGruResult result = load_network_from_file_f64(path);
assert(result.status == NeatGruStatus::Sucess);

const long INPUT_SIZE = 4;
const double input[INPUT_SIZE] = {0.5, 0.5, 0.1, -0.2};
double output[2];
compute_network_f64(result.network, INPUT_SIZE, input, output);
// Do something with output buffer.
// ...
// Free memory.
free_network_f64(result.network);
```

C++

```cpp
using namespace NeatGru;

const std::string path = "path.json";
NeuralNetwork<double> network = NeuralNetwork<double>::FromFile(path);

const long INPUT_SIZE = 4;
const double input[INPUT_SIZE] = {0.5, 0.5, 0.1, -0.2};
double output[2];

// We use raw C pointers for performance.
network.Compute(INPUT_SIZE, input, output);
// Do something with output buffer.
// ...
network.Reset();
```

## Examples

[XOR](examples/example.rs)

[Snake](examples/snake-cli)

Right now this is the only working example. You can run it via:

```
cargo run --example example
```

## How to use

In `Cargo.toml`:

```
[dependencies]
neat-gru = "1.4.0"
```

Create a struct that implements the `Game` trait

```rust
use neat_gru::game::Game;
use neat_gru::neural_network::NeuralNetwork;
use neat_gru::train::{Train, HistoricTopology};
struct Player {
    pub net: NeuralNetwork<f64>,
}

impl Player {
    pub fn new(net: NeuralNetwork<f64>) -> Player {
        Player {
            net,
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
        let inputs = get_inputs();
        self.players.iter().map(|p| {
            let output = p.net.compute(inputs);
            let scores = compute_score(output, target);
            scores
        }).collect()
    }

    // Reset networks
    fn reset_players(&mut self, nets: Vec<NeuralNetwork<f64>>) {
        self.players.clear();
        self.players = nets
            .into_iter()
            .map(Player::new)
            .collect();
    }

    // Called at the end of training
    fn post_training(&mut self, history: &[HistoricTopology<f64>]) {
        // Iter on best topologies and upload the best one
    }
}

```

Async run_generation (has to be run inside an async runtime like Tokio)

```rust

#[async_trait]
impl GameAsync<f64> for Simulation {
    // Loss function
    async fn run_generation(&mut self) -> Vec<f64> {
        let inputs = get_inputs().await;
        self.players.iter().map(|p| {
            let output = p.net.compute(inputs);
            let scores = compute_score(output, target);
            scores
        }).collect()
    }
}
```

Launch a training

```rust
fn run_sim() {
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
        .delta_threshold(2.) // Delta parameter from NEAT paper
        .formula(0.8, 0.8, 0.3) // c1, c2 and c3 from NEAT paper
        .access_train_object(Box::new(|train| {
            let species_count = train.species_count();
            println!("Species count: {}", species_count);
        })) // Callback called after `reset_players` that gives you access to the train object during training
        .start(); // .start_async().await for async version
}
```
