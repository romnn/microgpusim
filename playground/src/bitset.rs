use playground_sys::types::bitset::{bitset, new_bitset};

#[derive()]
pub struct Bitset(cxx::UniquePtr<bitset>);

impl Default for Bitset {
    fn default() -> Self {
        Self(new_bitset())
    }
}

impl std::fmt::Debug for Bitset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

impl std::fmt::Display for Bitset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.to_string())
    }
}

impl Bitset {
    // #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    // #[inline]
    #[must_use]
    pub fn size(&self) -> usize {
        self.0.size()
    }

    // #[inline]
    pub fn reset(&mut self) {
        self.0.pin_mut().reset();
    }

    // #[inline]
    pub fn shift_right(&mut self, n: usize) {
        self.0.pin_mut().shift_right(n);
    }

    // #[inline]
    pub fn shift_left(&mut self, n: usize) {
        self.0.pin_mut().shift_left(n);
    }

    // #[inline]
    pub fn set(&mut self, pos: usize, set: bool) {
        self.0.pin_mut().set(pos, set);
    }
}
