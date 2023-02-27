#![allow(warnings)]

pub mod cache;
pub use cache::{Cache, Config as CacheConfig, MainMemory, Simulation};

#[cfg(test)]
mod tests {
    // use super::*;

    // #[test]
    // fn it_works() {
    //     let result = add(2, 2);
    //     assert_eq!(result, 4);
    // }
}
