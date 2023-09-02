#![allow(
    non_camel_case_types,
    clippy::upper_case_acronyms,
    clippy::missing_panics_doc
)]

pub mod cache;
pub mod dram;
pub mod instructions;
pub mod mem;
pub mod scheduler;
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
    /// Number of memory fetches sent from SMs to the interconnect.
    pub accesses: Accesses,
    /// Instruction count breakdown per memory space and kind.
    pub instructions: InstructionCounts,
    /// High-level simulation metrics.
    pub sim: Sim,
    /// DRAM access stats.
    pub dram: DRAM,
    /// L1 instruction cache stats.
    pub l1i_stats: PerCache,
    /// L1 const cache stats.
    pub l1c_stats: PerCache,
    /// L1 texture cache stats.
    pub l1t_stats: PerCache,
    /// L1 data cache stats.
    pub l1d_stats: PerCache,
    /// L2 data cache stats.
    pub l2d_stats: PerCache,
    // where should those go? stall reasons? per core?
    pub stall_dram_full: u64,
}

impl Stats {
    #[must_use]
    pub fn new(
        num_total_cores: usize,
        num_mem_units: usize,
        num_sub_partitions: usize,
        num_dram_banks: usize,
    ) -> Self {
        Self {
            accesses: Accesses::default(),
            instructions: InstructionCounts::default(),
            sim: Sim::default(),
            dram: DRAM::new(num_total_cores, num_mem_units, num_dram_banks),
            l1i_stats: PerCache::new(num_total_cores),
            l1c_stats: PerCache::new(num_total_cores),
            l1t_stats: PerCache::new(num_total_cores),
            l1d_stats: PerCache::new(num_total_cores),
            l2d_stats: PerCache::new(num_sub_partitions),
            stall_dram_full: 0,
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
