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
        write!(f, "{}", self.to_string())
    }
}

impl std::fmt::Display for Bitset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.to_string().to_string())
    }
}

impl Bitset {
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn size(&self) -> usize {
        self.0.size()
    }

    pub fn reset(&mut self) {
        self.0.pin_mut().reset();
    }

    pub fn shift_right(&mut self, n: usize) {
        self.0.pin_mut().shift_right(n);
    }

    pub fn shift_left(&mut self, n: usize) {
        self.0.pin_mut().shift_left(n);
    }

    pub fn set(&mut self, pos: usize, set: bool) {
        self.0.pin_mut().set(pos, set);
    }
}
