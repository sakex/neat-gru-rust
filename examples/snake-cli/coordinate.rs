use crate::{direction::Direction, error::CoordinateError};
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Coordinate {
    pub x: usize,
    pub y: usize,
}
impl Coordinate {
    /// Transforms a coordinate with a given direction and step value into that direction
    pub fn transform(&mut self, direction: Direction, step: usize) -> Result<(), CoordinateError> {
        match direction {
            Direction::Up => {
                if let Some(x) = usize::checked_sub(self.y, step) {
                    self.y = x;
                    return Ok(());
                }
            }
            Direction::Right => {
                if let Some(x) = usize::checked_add(self.x, step) {
                    self.x = x;
                    return Ok(());
                }
            }
            Direction::Down => {
                if let Some(x) = usize::checked_add(self.y, step) {
                    self.y = x;
                    return Ok(());
                }
            }
            Direction::Left => {
                if let Some(x) = usize::checked_sub(self.x, step) {
                    self.x = x;
                    return Ok(());
                }
            }
        }
        Err(CoordinateError::OutOfBoundsError)
    }
    pub fn new_transform(
        coordinate: Coordinate,
        direction: Direction,
        step: usize,
    ) -> Result<Self, CoordinateError> {
        let mut clone = coordinate;
        let overflow = clone.transform(direction, step);
        return match overflow {
            Ok(_) => Ok(clone),
            Err(e) => Err(e),
        };
    }
}
impl From<(usize, usize)> for Coordinate {
    fn from(i: (usize, usize)) -> Self {
        Self { x: i.0, y: i.1 }
    }
}
