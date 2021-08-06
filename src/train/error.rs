use std::fmt;

#[derive(Debug)]
pub enum TrainingError {
    NoInput,
}

impl std::error::Error for TrainingError {}

impl fmt::Display for TrainingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrainingError::NoInput => write!(f, "No inputs were provided"),
        }
    }
}
