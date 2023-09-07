use crate::config;

use serde::{Deserialize, Serialize};

/// Cache write-allocate policy.
///
/// For more details about difference between `FETCH_ON_WRITE` and WRITE
/// VALIDAE policies Read: Jouppi, Norman P. "Cache write policies and
/// performance". ISCA 93. `WRITE_ALLOCATE` is the old write policy in
/// GPGPU-sim 3.x, that send WRITE and READ for every write request
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub enum WriteAllocatePolicy {
    NO_WRITE_ALLOCATE,  // N
    WRITE_ALLOCATE,     // W
    FETCH_ON_WRITE,     // F
    LAZY_FETCH_ON_READ, // L
}

/// A cache write policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub enum WritePolicy {
    READ_ONLY,          // R
    WRITE_BACK,         // B
    WRITE_THROUGH,      // T
    WRITE_EVICT,        // E
    LOCAL_WB_GLOBAL_WT, // L
}

/// A cache allocate policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AllocatePolicy {
    ON_MISS,   // M
    ON_FILL,   // F
    STREAMING, // S
}

/// A cache replacement policy
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ReplacementPolicy {
    LRU,  // L
    FIFO, // F
}

// #[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
#[derive(Clone, Debug)]
pub struct Config {
    /// Cache set index function.
    // pub set_index_function: Arc<dyn crate::set_index::SetIndexer>,

    /// Cache allocate policy.
    pub allocate_policy: AllocatePolicy,

    /// Cache replacement policy.
    pub replacement_policy: ReplacementPolicy,

    /// Cache write allocate policy.
    pub write_allocate_policy: WriteAllocatePolicy,

    /// Cache write policy.
    pub write_policy: WritePolicy,

    /// Cache line size.
    pub line_size: u32,

    /// Cache associativity.
    pub associativity: usize,

    /// Number of sets.
    pub num_sets: usize,

    /// Cache atom size.
    pub atom_size: u32,

    /// Cache miss queue size.
    pub miss_queue_size: usize,

    /// Cache miss queue size.
    pub mshr_kind: crate::mshr::Kind,

    /// Number of lines.
    ///
    /// NOTE: CAN BE COMPUTED from sets and associativity.
    pub total_lines: usize,
    pub line_size_log2: u32,
    pub num_sets_log2: u32,
}

impl From<&config::Cache> for Config {
    fn from(config: &config::Cache) -> Self {
        Self {
            // set_index_function: Arc::<crate::set_index::linear::SetIndex>::default(),
            write_policy: config.write_policy,
            write_allocate_policy: config.write_allocate_policy,
            allocate_policy: config.allocate_policy,
            replacement_policy: config.replacement_policy,
            associativity: config.associativity,
            num_sets: config.num_sets,
            atom_size: config.atom_size(),
            miss_queue_size: config.miss_queue_size,
            mshr_kind: config.mshr_kind,
            total_lines: config.num_sets * config.associativity,
            line_size: config.line_size,
            line_size_log2: config.line_size_log2(),
            num_sets_log2: config.num_sets_log2(),
        }
    }
}

// impl CacheConfig {
//     #[inline]
//     #[must_use]
//     fn total_lines(&self) -> usize {
//         self.num_sets * self.associativity
//     }
// }

// pub trait CacheConfig: std::fmt::Debug + Sync + Send + 'static {
//     #[must_use]
//     fn associativity(&self) -> usize;
//
//     #[must_use]
//     fn num_sets(&self) -> usize;
//
//     #[must_use]
//     fn total_lines(&self) -> usize {
//         self.num_sets() * self.associativity()
//     }
// }

pub mod sector {
    #[allow(dead_code)]
    pub struct Config {
        pub sector_size: usize,
        pub sector_size_log2: usize,
    }

    // impl Config {
    //     #[inline]
    //     #[must_use]
    //     pub fn sector_size(&self) -> u32 {
    //         mem_sub_partition::SECTOR_SIZE
    //     }
    //
    //     #[inline]
    //     #[must_use]
    //     pub fn sector_size_log2(&self) -> u32 {
    //         addrdec::logb2(self.sector_size())
    //     }
    // }
}
