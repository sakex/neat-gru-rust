use num::Float;
use numeric_literals::replace_numeric_literals;

#[replace_numeric_literals(T::from(literal).unwrap())]
#[inline]
pub fn fast_sigmoid<T: Float>(value: T) -> T {
    value / (1 + value.abs())
}

#[replace_numeric_literals(T::from(literal).unwrap())]
#[inline]
pub fn fast_tanh<T: Float>(x: T) -> T {
    if x.abs() >= 4.97 {
        let values = [-1, 1];
        return unsafe { *values.get_unchecked((x >= 0) as usize) };
    }
    let x2 = x * x;
    let a = x * (135135 + x2 * (17325 + x2 * (378 + x2)));
    let b = 135135 + x2 * (62370 + x2 * (3150 + x2 * 28));
    a / b
}

#[inline]
pub fn re_lu<T: Float>(x: T) -> T{
    x.max(T::zero())
}