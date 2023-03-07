#[cfg(not(target_arch = "wasm32"))]
pub mod ffi;
pub mod game;
#[cfg(target_arch = "wasm32")]
mod instant_wasm_replacement;
pub mod neural_network;
#[cfg(test)]
mod tests;
pub mod topology;
pub mod train;
mod utils;

#[cfg(not(target_arch = "wasm32"))]
pub use ffi::*;
