#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Direction {
    Up,
    Right,
    Down,
    Left,
}

impl Direction {
    pub fn opposite(&self) -> Direction {
        match self {
            Direction::Up => Direction::Down,
            Direction::Right => Direction::Left,
            Direction::Down => Direction::Up,
            Direction::Left => Direction::Right,
        }
    } /*
      pub fn from_keycode(key: KeyCode) -> Option<Direction> {
          match key {
              KeyCode::Up | KeyCode::W => Some(Direction::Up),
              KeyCode::Down | KeyCode::S => Some(Direction::Down),
              KeyCode::Left | KeyCode::A => Some(Direction::Left),
              KeyCode::Right | KeyCode::D => Some(Direction::Right),
              _ => None,
          }
      }*/
}
