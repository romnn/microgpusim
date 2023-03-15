#![allow(warnings)]

pub mod cache;
pub use cache::{Cache, Config as CacheConfig, MainMemory, Simulation};

#[cfg(test)]
mod tests {}
