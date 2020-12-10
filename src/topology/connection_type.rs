#[derive(Clone)]
pub enum ConnectionType {
    Sigmoid = 0,
    GRU = 1
}

impl ConnectionType {
    pub fn from_int(i: i32) -> ConnectionType{
        match i {
            0 => ConnectionType::Sigmoid,
            1 => ConnectionType::GRU,
            _ => panic!("Invalid value {}", i)
        }
    }
}