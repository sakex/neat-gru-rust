use crate::{apple::Apple, defs::RESOLUTION, snake::Snake};

pub fn distance_to_apple_x(snake: &Snake, apple: Apple) -> f64 {
    let snake_coordinate = snake.get_head_position().x as f64;
    let apple_coordinate = apple.get_coordinate().x as f64;
    (snake_coordinate - apple_coordinate) / RESOLUTION as f64
}

pub fn distance_to_apple_y(snake: &Snake, apple: Apple) -> f64 {
    let snake_coordinate = snake.get_head_position().y as f64;
    let apple_coordinate = apple.get_coordinate().y as f64;
    (snake_coordinate - apple_coordinate) / RESOLUTION as f64
}

pub fn distance_to_apple(snake: &Snake, apple: Apple) -> f64 {
    // In this case we can directly add the distances since the snake can't go diagonally
    distance_to_apple_x(snake, apple) + distance_to_apple_y(snake, apple)
}

pub fn distance_to_wall_x(snake: &Snake) -> f64 {
    let pos = snake.get_head_position();
    let distance = (RESOLUTION - pos.x) as f64;
    (distance / RESOLUTION.checked_div(2).unwrap() as f64) - 1.
}

pub fn distance_to_wall_y(snake: &Snake) -> f64 {
    let pos = snake.get_head_position();
    let distance = (RESOLUTION - pos.y) as f64;
    (distance / RESOLUTION.checked_div(2).unwrap() as f64) - 1.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_to_wall() {
        // These are specific to the resolution but they do work with Resolution 100, others may differ
        let snake = Snake::default();
        assert!(distance_to_wall_x(&snake) - 0.8600 < f64::EPSILON);
        assert!(distance_to_wall_y(&snake) - 0.8999999999999999 < f64::EPSILON);
    }
    #[test]
    fn test_distance_to_apple() {
        let snake = Snake::default();
        let apple = Apple::generate_apple();
        println!("{}", distance_to_apple_x(&snake, apple));
    }
}
