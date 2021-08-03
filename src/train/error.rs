use std::fmt;

#[derive(Debug)]
pub enum InputError {
    NoInput,
}

impl std::error::Error for InputError {}

impl fmt::Display for InputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InputError::NoInput => write!(f, "No inputs were provided"),
        }
    }
}
