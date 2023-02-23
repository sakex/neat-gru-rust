use std::{ffi::CStr, fmt::Display, io::Read};

use libc::c_char;
use num::Float;
use std::fs::File;

use crate::{neural_network::NeuralNetwork, topology::Topology};

#[repr(C)]
pub enum NeatGruStatus {
    Sucess,
    InvalidString,
    MissingFile,
    FailedToReadFile,
    InvalidFile,
}

#[repr(C)]
pub struct NeuralNetworkErased {}

#[repr(C)]
pub struct NeatGruResult {
    status: NeatGruStatus,
    network: *mut NeuralNetworkErased,
}

impl From<NeatGruStatus> for NeatGruResult {
    fn from(status: NeatGruStatus) -> Self {
        Self {
            status,
            network: std::ptr::null_mut(),
        }
    }
}

fn load_network_from_file_impl<T>(file_path: *const c_char) -> NeatGruResult
where
    T: Float + std::ops::AddAssign + Display + Send,
{
    let file_path_cstr: &CStr = unsafe { CStr::from_ptr(file_path) };
    let file_path: &str = match file_path_cstr.to_str() {
        Ok(s) => s,
        Err(_) => return NeatGruResult::from(NeatGruStatus::InvalidString),
    };

    let mut file = match File::open(file_path) {
        Ok(f) => f,
        Err(_) => return NeatGruResult::from(NeatGruStatus::MissingFile),
    };

    let mut file_string = String::new();
    if let Err(_) = file.read_to_string(&mut file_string) {
        return NeatGruResult::from(NeatGruStatus::FailedToReadFile);
    }

    if !Topology::<T>::is_valid_topology_json(&file_string) {
        return NeatGruResult::from(NeatGruStatus::InvalidFile);
    }
    let topology = Topology::<T>::from_string(&file_string);
    let network = unsafe { Box::new(NeuralNetwork::new(&topology)) };
    let network_ptr = Box::leak(network) as *mut NeuralNetwork<T>;
    NeatGruResult {
        status: NeatGruStatus::Sucess,
        network: network_ptr as *mut NeuralNetworkErased,
    }
}

#[no_mangle]
pub extern "C" fn load_network_from_file_f32(file_path: *const c_char) -> NeatGruResult {
    load_network_from_file_impl::<f32>(file_path)
}

#[no_mangle]
pub extern "C" fn load_network_from_file_f64(file_path: *const c_char) -> NeatGruResult {
    load_network_from_file_impl::<f64>(file_path)
}

unsafe fn compute_network_impl<T>(
    network: &mut NeuralNetwork<T>,
    input_size: usize,
    inputs: *const T,
    outputs: *mut T,
) where
    T: Float + std::ops::AddAssign + Display + Send,
{
    let input_slice = std::slice::from_raw_parts(inputs, input_size);
    // Super unsafe, but we don't actually rely on bound checking in `compute_buffer` so the size will be ignored.
    let output_slice = std::slice::from_raw_parts_mut(outputs, 1);
    network.compute_buffer(input_slice, output_slice);
}

#[no_mangle]
pub extern "C" fn compute_network_f32(
    network: *mut NeuralNetworkErased,
    input_size: std::ffi::c_long,
    inputs: *const f32,
    outputs: *mut f32,
) {
    let network_f32 = network as *mut NeuralNetwork<f32>;
    assert!(!network_f32.is_null());
    unsafe {
        compute_network_impl(&mut *network_f32, input_size as usize, inputs, outputs);
    }
}

#[no_mangle]
pub extern "C" fn compute_network_f64(
    network: *mut NeuralNetworkErased,
    input_size: std::ffi::c_long,
    inputs: *const f64,
    outputs: *mut f64,
) {
    let network_f64 = network as *mut NeuralNetwork<f64>;
    assert!(!network_f64.is_null());
    unsafe {
        compute_network_impl(&mut *network_f64, input_size as usize, inputs, outputs);
    }
}

#[no_mangle]
pub extern "C" fn reset_network_f32(network: *mut NeuralNetworkErased) {
    let network_f32 = network as *mut NeuralNetwork<f32>;
    assert!(!network_f32.is_null());
    unsafe {
        (*network_f32).reset_state();
    }
}

#[no_mangle]
pub extern "C" fn reset_network_f64(network: *mut NeuralNetworkErased) {
    let network_f64 = network as *mut NeuralNetwork<f64>;
    assert!(!network_f64.is_null());
    unsafe {
        (*network_f64).reset_state();
    }
}
