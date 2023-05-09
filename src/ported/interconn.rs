use super::MemFetch;

pub trait MemFetchInterconnect {
    fn full(&self, size: usize, write: bool) -> bool;
    fn push(&mut self, fetch: MemFetch);
}

#[derive(Debug, Clone, Default)]
pub struct Interconnect {}

impl Interconnect {
    pub fn push(&mut self, fetch: MemFetch) {
        println!("interconnect: pushed fetch {:?}", fetch);
    }

    pub fn full(&self, size: usize, write: bool) -> bool {
        false
    }
}
