use crate::ported::mem_fetch;

#[derive(Debug, Default)]
pub struct ConstL1 {}

impl ConstL1 {
    pub fn cycle(&mut self) {}

    pub fn fill(&self, fetch: &mem_fetch::MemFetch) {}

    pub fn has_free_fill_port(&self) -> bool {
        false
    }
}
