use crate::ported::{interconn as ic, mem_fetch};

#[derive(Debug)]
pub struct TextureL1 {
    id: usize,
    interconn: ic::Interconnect,
}

impl TextureL1 {
    pub fn new(id: usize, interconn: ic::Interconnect) -> Self {
        Self { id, interconn }
    }

    pub fn cycle(&mut self) {}

    pub fn fill(&self, fetch: &mem_fetch::MemFetch) {}

    pub fn has_free_fill_port(&self) -> bool {
        false
    }
}
