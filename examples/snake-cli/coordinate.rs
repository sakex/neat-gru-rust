use crate::direction::Direction;
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Coordinate {
    pub x: usize,
    pub y: usize,
}
impl Coordinate {
    /// This function returns true if it causes an over-/underflow upon which the snake should die
    // TODO: Make this return Result<(), OutOfBoundsError>
    pub fn transform(&mut self, direction: Direction, step: usize) -> bool {
        match direction {
            // Since this could cause an over-/underflow it's safer to use checked_sub
            Direction::Up => {
                if let Some(x) = usize::checked_sub(self.y, step) {
                    self.y = x;
                    false
                } else {
                    true
                }
            }
            Direction::Right => {
                if let Some(x) = usize::checked_add(self.x, step) {
                    self.x = x;
                    false
                } else {
                    true
                }
            }
            Direction::Down => {
                if let Some(x) = usize::checked_add(self.y, step) {
                    self.y = x;
                    false
                } else {
                    true
                }
            }
            Direction::Left => {
                if let Some(x) = usize::checked_sub(self.x, step) {
                    self.x = x;
                    false
                } else {
                    true
                }
            }
        }
    }
    pub fn new_transform(
        coordinate: Coordinate,
        direction: Direction,
        step: usize,
    ) -> (Self, bool) {
        let mut clone = coordinate;
        let overflow = clone.transform(direction, step);
        (clone, overflow)
    }
}
impl From<(usize, usize)> for Coordinate {
    fn from(i: (usize, usize)) -> Self {
        Self { x: i.0, y: i.1 }
    }
}
