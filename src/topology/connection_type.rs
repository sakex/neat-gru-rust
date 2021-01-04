use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize, Serialize)]
pub enum ConnectionType {
    Sigmoid,
    GRU,
}

impl ConnectionType {
    pub fn from_int(i: i32) -> ConnectionType {
        match i {
            0 => ConnectionType::Sigmoid,
            1 => ConnectionType::GRU,
            _ => panic!("Invalid value {}", i),
        }
    }

    pub fn to_int(&self) -> i32 {
        match self {
            ConnectionType::Sigmoid => 0,
            ConnectionType::GRU => 1,
        }
    }
}
