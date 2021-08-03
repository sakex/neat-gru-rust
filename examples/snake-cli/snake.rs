use std::collections::LinkedList;

use crate::coordinate::Coordinate;
use crate::defs::RESOLUTION;
use crate::direction::Direction;
use neat_gru::neural_network::nn::NeuralNetwork;
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Block {
    cord: Coordinate,
}

impl Block {}
#[derive(Clone, PartialEq, Debug)]
pub struct Snake {
    blocks: LinkedList<Block>,
    moving_direction: Direction,
    dir_changed: bool,
    net: Option<NeuralNetwork<f64>>,
}

impl Snake {
    pub fn new(net: NeuralNetwork<f64>) -> Self {
        let mut blocks = LinkedList::new();
        blocks.push_front((5, 5).into());
        blocks.push_front((6, 5).into());
        blocks.push_front((7, 5).into());
        Snake {
            blocks,
            moving_direction: Direction::Right,
            dir_changed: false,
            net: Some(net),
        }
    }

    /// Returns the size of the snake
    pub fn size(&self) -> usize {
        self.blocks.len()
    }
    /// Changes the direction of the snake if it hasn't already been changed this tick
    pub fn change_direction(&mut self, dir: Direction) {
        // We don't want to change the direction multiple times per tick and also don't want to move into the snakes body
        if !self.dir_changed && dir.opposite() != self.moving_direction {
            self.moving_direction = dir;
            self.dir_changed = true;
        }
    }
    fn is_eating(&self, apple: Coordinate) -> bool {
        self.get_head_position() == apple
    }
    /// Updates the snake's body and returns whether it produced an over-/underflow
    fn move_body(&mut self) -> bool {
        // Add a block in front
        let block_in_front = Coordinate::new_transform(
            self.blocks.front().expect("Could not find a snake!").cord,
            self.moving_direction,
            1,
        );
        match block_in_front {
            Ok(cord) => self.blocks.push_front(Block{cord}),
            Err(_) => return true,
        }
        self.dir_changed = false;
        false
    }
    /// Updates the snake. Eating determines whether the snake is eating and returns
    /// whether it crashed and whether it ate an apple
    pub fn update(&mut self, apple: Coordinate) -> bool {
        let mut eating = false;
        // If the snake is eating it gets longer
        if !self.is_eating(apple) {
            self.blocks.pop_back();
            eating = true;
        }
        // Move the body
        if self.move_body() {
            // And if it tries to move out of the bounds of a usize(e.g. less than zero we panic)
            println!("{:?}", self.get_head_position());
            panic!("Invalid coordinates!")
        }
        eating
    }
    /// Returns if the snake is overlapping "Biting it's own tail"
    fn is_overlapping(&self) -> bool {
        // We have to see if any of the blocks matches the first one
        let mut cloned_snake = self.clone();
        // Remove the head block
        let future_head_block = cloned_snake.blocks.pop_front().unwrap();
        // And check if the head block is the same as any other block
        let result = cloned_snake.blocks.contains(&future_head_block);
        if result {
            println!("Bit it's own tail");
        };
        result
    }
    pub fn is_colliding(&self) -> bool {
        self.is_overlapping() || self.is_colliding_with_wall()
    }
    fn is_colliding_with_wall(&self) -> bool {
        let head_position = self.get_head_position();
        match self.moving_direction {
            Direction::Up => {
                if head_position.y <= 0 {
                    return true;
                }
            }
            Direction::Right => {
                if head_position.x >= RESOLUTION - 1 {
                    return true;
                }
            }
            Direction::Down => {
                if head_position.y >= RESOLUTION - 1 {
                    return true;
                }
            }
            Direction::Left => {
                if head_position.x <= 0 {
                    return true;
                }
            }
        }
        false
    }
    /// Returns the head position of the snake.
    pub fn get_head_position(&self) -> Coordinate {
        self.blocks.front().unwrap().cord
    }

    /// Executes a decision based on given input
    pub fn make_decision(&mut self, inputs: &[f64]) {
        let output = self.net.as_mut().unwrap().compute(inputs);
        // Since the direction can only be changed once per tick once we have a direction we don't have to check for any other direction
        if output[0] >= 0.0 {
            self.change_direction(Direction::Up)
        } else if output[1] >= 0.0 {
            self.change_direction(Direction::Right)
        } else if output[2] >= 0.0 {
            self.change_direction(Direction::Left)
        } else if output[3] >= 0.0 {
            self.change_direction(Direction::Down)
        }
    }
}
impl From<(usize, usize)> for Block {
    fn from(i: (usize, usize)) -> Self {
        Self {
            cord: (i.0, i.1).into(),
        }
    }
}
impl Default for Snake {
    fn default() -> Self {
        let mut blocks = LinkedList::new();
        blocks.push_front((5, 5).into());
        blocks.push_front((6, 5).into());
        blocks.push_front((7, 5).into());
        Snake {
            blocks,
            moving_direction: Direction::Right,
            dir_changed: false,
            net: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::apple::Apple;

    use super::*;

    #[test]
    fn test_is_colliding_with_wall() {
        let mut snake = Snake::default();
        let apple = Apple::generate_apple().get_coordinate();
        snake.change_direction(Direction::Up);
        snake.update(apple);
        assert_eq!(
            &Block {
                cord: (7, 4).into()
            },
            snake.blocks.front().unwrap()
        );
        snake.update(apple);
        assert_eq!(
            &Block {
                cord: (7, 3).into()
            },
            snake.blocks.front().unwrap()
        );
        snake.update(apple);
        assert_eq!(
            &Block {
                cord: (7, 2).into()
            },
            snake.blocks.front().unwrap()
        );
        snake.update(apple);
        assert_eq!(
            &Block {
                cord: (7, 1).into()
            },
            snake.blocks.front().unwrap()
        );
        snake.update(apple);
        assert_eq!(
            &Block {
                cord: (7, 0).into()
            },
            snake.blocks.front().unwrap()
        );
        assert!(snake.is_colliding());

        let mut snake = Snake::default();
        let apple = Apple::generate_apple().get_coordinate();
        snake.update(apple);
        snake.change_direction(Direction::Up);
        assert_eq!(
            &Block {
                cord: (8, 5).into()
            },
            snake.blocks.front().unwrap()
        );
        snake.update(apple);
        assert_eq!(
            &Block {
                cord: (8, 4).into()
            },
            snake.blocks.front().unwrap()
        );
        snake.update(apple);
        assert_eq!(
            &Block {
                cord: (8, 3).into()
            },
            snake.blocks.front().unwrap()
        );
        snake.update(apple);
        assert_eq!(
            &Block {
                cord: (8, 2).into()
            },
            snake.blocks.front().unwrap()
        );
        snake.update(apple);
        assert_eq!(
            &Block {
                cord: (8, 1).into()
            },
            snake.blocks.front().unwrap()
        );
        snake.update(apple);
        assert_eq!(
            &Block {
                cord: (8, 0).into()
            },
            snake.blocks.front().unwrap()
        );
        assert!(snake.is_colliding());
    }
}
