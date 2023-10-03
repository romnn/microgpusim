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
pub use utils::box_slice;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Statistics configuration.
#[derive(Clone, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelInfo {
    pub name: String,
    pub mangled_name: String,
    pub launch_id: usize,
}

/// Statistics configuration.
#[derive(Clone, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Config {
    pub num_total_cores: usize,
    pub num_mem_units: usize,
    pub num_sub_partitions: usize,
    pub num_dram_banks: usize,
}

/// Per kernel statistics.
///
/// Stats at index `i` correspond to the kernel with launch id `i`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerKernel {
    pub inner: Vec<Stats>,
    pub config: Config,
}

impl AsRef<Vec<Stats>> for PerKernel {
    fn as_ref(&self) -> &Vec<Stats> {
        &self.inner
    }
}

impl AsMut<Vec<Stats>> for PerKernel {
    fn as_mut(&mut self) -> &mut Vec<Stats> {
        &mut self.inner
    }
}

impl PerKernel {
    #[must_use]
    pub fn new(config: Config) -> Self {
        Self {
            config,
            inner: Vec::new(),
        }
    }

    // #[inline]
    pub fn get_mut(&mut self, idx: usize) -> &mut Stats {
        if idx >= self.inner.len() {
            self.inner.resize_with(idx + 1, || Stats::new(&self.config));
        }
        &mut self.inner[idx]
    }

    // #[inline]
    #[must_use]
    pub fn reduce(self) -> Stats {
        let mut reduced = Stats::new(&self.config);
        for per_kernel_stats in self.inner {
            reduced += per_kernel_stats;
        }
        reduced
    }
}

impl std::ops::AddAssign for Stats {
    fn add_assign(&mut self, other: Self) {
        self.accesses += other.accesses;
        self.instructions += other.instructions;
        self.sim += other.sim;
        self.dram += other.dram;
        self.l1i_stats += other.l1i_stats;
        self.l1c_stats += other.l1c_stats;
        self.l1t_stats += other.l1t_stats;
        self.l1d_stats += other.l1d_stats;
        self.l2d_stats += other.l2d_stats;
        self.stall_dram_full += other.stall_dram_full;
    }
}

// todo: implement index for kernel references too
impl std::ops::Index<usize> for PerKernel {
    type Output = Stats;
    fn index(&self, idx: usize) -> &Self::Output {
        &self.inner[idx]
    }
}

impl std::ops::IndexMut<usize> for PerKernel {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.inner[idx]
    }
}

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
    pub fn new(config: &Config) -> Self {
        Self {
            accesses: Accesses::default(),
            instructions: InstructionCounts::default(),
            sim: Sim::default(),
            dram: DRAM::new(
                config.num_total_cores,
                config.num_mem_units,
                config.num_dram_banks,
            ),
            l1i_stats: PerCache::new(config.num_total_cores),
            l1c_stats: PerCache::new(config.num_total_cores),
            l1t_stats: PerCache::new(config.num_total_cores),
            l1d_stats: PerCache::new(config.num_total_cores),
            l2d_stats: PerCache::new(config.num_sub_partitions),
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
