/// Arithmetic no-op.
///
/// Think of it as a black hole.
#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone, Copy, Default, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct ArithmeticNop;

impl<T> std::ops::Add<T> for ArithmeticNop {
    type Output = Self;

    fn add(self, _other: T) -> Self::Output {
        self
    }
}

// const NOP: ArithmeticNoOp = ArithmeticNoOp;
