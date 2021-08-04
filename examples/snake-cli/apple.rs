use crate::{coordinate::Coordinate, defs::RESOLUTION};
use rand::Rng;

#[derive(Copy, Clone, Debug)]
pub struct Apple {
    cord: Coordinate,
}
impl Apple {
    pub fn generate_apple() -> Self {
        let mut rng = rand::thread_rng();
        Apple {
            cord: (
                rng.gen_range(0..RESOLUTION as usize),
                rng.gen_range(0..RESOLUTION as usize),
            )
                .into(),
        }
    }
    pub fn get_coordinate(&self) -> Coordinate {
        self.cord
    }
}
