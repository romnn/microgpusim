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
pub use scheduler::Scheduler;
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
    pub kernel_stats: Vec<Stats>,
    pub no_kernel: Stats,
    pub config: Config,
}

impl IntoIterator for PerKernel {
    type Item = Stats;
    type IntoIter = std::vec::IntoIter<Stats>;

    fn into_iter(self) -> Self::IntoIter {
        self.kernel_stats.into_iter()
    }
}

impl std::ops::AddAssign for PerKernel {
    fn add_assign(&mut self, mut other: Self) {
        let num_kernel = self.kernel_stats.len().max(other.kernel_stats.len());
        self.kernel_stats
            .resize(num_kernel, Stats::new(&self.config));

        for (i, s) in other.kernel_stats.drain(..).enumerate() {
            self.kernel_stats[i] += s;
        }

        self.no_kernel += other.no_kernel;
    }
}

impl std::ops::Index<usize> for PerKernel {
    type Output = Stats;
    fn index(&self, idx: usize) -> &Self::Output {
        &self.kernel_stats[idx]
    }
}

impl std::ops::IndexMut<usize> for PerKernel {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.kernel_stats[idx]
    }
}

impl std::ops::Index<Option<usize>> for PerKernel {
    type Output = Stats;
    fn index(&self, idx: Option<usize>) -> &Self::Output {
        match idx {
            None => &self.no_kernel,
            Some(idx) => &self[idx],
        }
    }
}

impl std::ops::IndexMut<Option<usize>> for PerKernel {
    fn index_mut(&mut self, idx: Option<usize>) -> &mut Self::Output {
        match idx {
            None => &mut self.no_kernel,
            Some(idx) => &mut self[idx],
        }
    }
}

impl std::ops::Index<&KernelInfo> for PerKernel {
    type Output = Stats;
    fn index(&self, kernel: &KernelInfo) -> &Self::Output {
        &self[kernel.launch_id]
    }
}

impl std::ops::IndexMut<&KernelInfo> for PerKernel {
    fn index_mut(&mut self, kernel: &KernelInfo) -> &mut Self::Output {
        &mut self[kernel.launch_id]
    }
}

impl PerKernel {
    #[must_use]
    pub fn new(config: Config) -> Self {
        let no_kernel = Stats::new(&config);
        Self {
            config,
            no_kernel,
            kernel_stats: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.kernel_stats.len()
    }

    pub fn num_kernels(&self) -> usize {
        self.kernel_stats.len()
    }

    pub fn iter(&self) -> std::slice::Iter<Stats> {
        self.kernel_stats.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<Stats> {
        self.kernel_stats.iter_mut()
    }

    pub fn get_mut(&mut self, idx: Option<usize>) -> &mut Stats {
        match idx {
            None => &mut self.no_kernel,
            Some(idx) => {
                if idx >= self.kernel_stats.len() {
                    self.kernel_stats
                        .resize_with(idx + 1, || Stats::new(&self.config));
                }
                &mut self.kernel_stats[idx]
            }
        }
    }

    #[must_use]
    pub fn reduce(self) -> Stats {
        let mut reduced = Stats::new(&self.config);
        for per_kernel_stats in self.kernel_stats {
            reduced += per_kernel_stats;
        }
        reduced
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stats {
    /// Number of memory fetches sent from SMs to the interconnect.
    pub accesses: Accesses,
    /// Instruction count breakdown per memory space and kind.
    pub instructions: InstructionCounts,
    /// Scheduler stats
    pub scheduler: Scheduler,
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

impl std::ops::AddAssign for Stats {
    fn add_assign(&mut self, other: Self) {
        self.accesses += other.accesses;
        self.instructions += other.instructions;
        self.scheduler += other.scheduler;
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

impl Stats {
    #[must_use]
    pub fn empty() -> Self {
        let num_total_cores = 1;
        let num_mem_units = 1;
        let num_dram_banks = 1;
        let num_sub_partitions = 1;
        Self {
            accesses: Accesses::default(),
            instructions: InstructionCounts::default(),
            scheduler: Scheduler::default(),
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

    #[must_use]
    pub fn new(config: &Config) -> Self {
        Self {
            accesses: Accesses::default(),
            instructions: InstructionCounts::default(),
            scheduler: Scheduler::default(),
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
