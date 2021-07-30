use crate::{apple::Apple, defs::RESOLUTION, snake::Snake};

pub fn distance_to_apple_x(snake: &Snake, apple: Apple) -> f64 {
    let snake_coordinate = snake.get_head_position().x;
    let apple_coordinate = apple.get_coordinate().x;
    (snake_coordinate - apple_coordinate / RESOLUTION / 2) as f64 - 1.
}

pub fn distance_to_apple_y(snake: &Snake, apple: Apple) -> f64 {
    let snake_coordinate = snake.get_head_position().y;
    let apple_coordinate = apple.get_coordinate().y;
    (snake_coordinate - apple_coordinate / RESOLUTION / 2) as f64 - 1.
}
