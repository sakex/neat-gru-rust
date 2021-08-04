use std::fmt;

#[derive(Debug)]
pub enum CoordinateError {
    OutOfBoundsError,
}

impl std::error::Error for CoordinateError {}

impl fmt::Display for CoordinateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoordinateError::OutOfBoundsError => write!(f, "Coordinate out of bounds"),
        }
    }
}
