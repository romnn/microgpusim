use super::MemFetch;

pub trait MemPort {
    fn full(&self, size: u32, write: bool) -> bool;
    fn push(&mut self, fetch: MemFetch);
}

#[derive(Debug, Clone, Default)]
pub struct Interconnect {}

impl MemPort for Interconnect {
    fn push(&mut self, fetch: MemFetch) {
        println!("interconnect: pushed fetch {:?}", fetch);
    }

    fn full(&self, size: u32, write: bool) -> bool {
        false
    }
}
