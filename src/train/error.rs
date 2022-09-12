use std::{fmt, io};

#[derive(Debug)]
pub enum TrainingError {
    NoInput,
    TempDirError(io::Error),
}

impl std::error::Error for TrainingError {}

impl fmt::Display for TrainingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrainingError::NoInput => write!(f, "No inputs were provided"),
            TrainingError::TempDirError(err) => write!(f, "Failed to create Temp Dir: {:?}", err),
        }
    }
}

impl From<io::Error> for TrainingError {
    fn from(err: io::Error) -> Self {
        Self::TempDirError(err)
    }
}
