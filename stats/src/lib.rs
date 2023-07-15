#![allow(non_camel_case_types, clippy::upper_case_acronyms)]

pub mod cache;
pub mod dram;
pub mod instructions;
pub mod mem;
pub mod sim;

pub use cache::{Cache, PerCache};
pub use dram::DRAM;
pub use instructions::InstructionCounts;
pub use mem::Accesses;
pub use sim::Sim;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stats {
    pub accesses: Accesses,
    pub instructions: InstructionCounts,
    pub sim: Sim,
    pub dram: DRAM,
    pub l1i_stats: PerCache,
    pub l1c_stats: PerCache,
    pub l1t_stats: PerCache,
    pub l1d_stats: PerCache,
    pub l2d_stats: PerCache,
}

impl Stats {
    #[must_use]
    pub fn new(num_total_cores: usize, num_mem_units: usize, num_dram_banks: usize) -> Self {
        Self {
            accesses: Accesses::default(),
            instructions: InstructionCounts::default(),
            sim: Sim::default(),
            dram: DRAM::new(num_total_cores, num_mem_units, num_dram_banks),
            l1i_stats: PerCache::default(),
            l1c_stats: PerCache::default(),
            l1t_stats: PerCache::default(),
            l1d_stats: PerCache::default(),
            l2d_stats: PerCache::default(),
        }
    }
}

pub trait ConvertHashMap<K, V, IK, IV, S>
where
    IK: Into<K>,
    IV: Into<V>,
    S: std::hash::BuildHasher,
{
    fn convert(self) -> HashMap<K, V, S>;
}

impl<K, V, IK, IV, S> ConvertHashMap<K, V, IK, IV, S> for HashMap<IK, IV, S>
where
    IK: Into<K>,
    IV: Into<V>,
    K: Eq + std::hash::Hash,
    S: std::hash::BuildHasher + std::default::Default,
{
    fn convert(self) -> HashMap<K, V, S> {
        self.into_iter()
            .map(|(k, v)| (k.into(), v.into()))
            .collect()
    }
}
