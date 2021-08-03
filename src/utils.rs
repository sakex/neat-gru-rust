use num::Float;
use numeric_literals::replace_numeric_literals;

#[inline(always)]
#[replace_numeric_literals(T::from(literal).unwrap())]
pub fn floats_almost_equal<T: Float>(f1: T, f2: T) -> bool {
    f1 - f2 < 1e-7
}
