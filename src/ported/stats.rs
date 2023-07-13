use super::{cache, mem_fetch};
use crate::config;
use std::collections::HashMap;
use std::sync::Mutex;

#[derive(Clone, Default, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DRAM {
    /// bank writes [shader id][dram chip id][bank id]
    pub bank_writes: Vec<Vec<Vec<u64>>>,
    /// bank reads [shader id][dram chip id][bank id]
    pub bank_reads: Vec<Vec<Vec<u64>>>,
    /// bank writes [dram chip id][bank id]
    pub total_bank_writes: Vec<Vec<u64>>,
    /// bank reads [dram chip id][bank id]
    pub total_bank_reads: Vec<Vec<u64>>,
}

impl DRAM {
    pub fn new(config: &config::GPUConfig) -> Self {
        let num_dram_banks = config.dram_timing_options.num_banks;
        let total_bank_writes = vec![vec![0; num_dram_banks]; config.num_mem_units];
        let total_bank_reads = total_bank_writes.clone();
        let bank_reads = vec![total_bank_reads.clone(); config.total_cores()];
        let bank_writes = bank_reads.clone();
        Self {
            total_bank_reads,
            total_bank_writes,
            bank_reads,
            bank_writes,
        }
    }

    pub fn total_reads(&self) -> u64 {
        self.total_bank_reads.iter().flatten().sum()
    }

    pub fn total_writes(&self) -> u64 {
        self.total_bank_writes.iter().flatten().sum()
    }
}

#[derive(Clone, Default, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Sim {
    pub cycles: usize,
    pub instructions: usize,
}

#[derive(Clone, Default, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PerCacheStats(pub HashMap<usize, CacheStats>);

impl PerCacheStats {
    pub fn shave(&mut self) {
        for stats in self.values_mut() {
            stats.shave();
        }
    }

    pub fn total_accesses(&self) -> usize {
        self.reduce().total_accesses()
    }

    pub fn reduce(&self) -> CacheStats {
        let mut out = CacheStats::default();
        for stats in self.0.values() {
            out += stats.clone();
        }
        out
    }
}

impl std::ops::Deref for PerCacheStats {
    type Target = HashMap<usize, CacheStats>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for PerCacheStats {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MemAccesses(pub HashMap<mem_fetch::AccessKind, usize>);

impl MemAccesses {
    pub fn num_writes(&self) -> usize {
        self.0
            .iter()
            .filter(|(kind, _)| kind.is_write())
            .map(|(_, count)| count)
            .sum()
    }

    pub fn num_reads(&self) -> usize {
        self.0
            .iter()
            .filter(|(kind, _)| !kind.is_write())
            .map(|(_, count)| count)
            .sum()
    }

    pub fn inc(&mut self, kind: mem_fetch::AccessKind, count: usize) {
        *self.0.entry(kind).or_insert(0) += count;
    }
}

impl std::fmt::Debug for MemAccesses {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut accesses: Vec<_> = self
            .0
            .iter()
            .filter(|(kind, &count)| count > 0)
            .map(|(kind, count)| (format!("{:?}", kind), count))
            .collect();
        accesses.sort_by_key(|(key, _)| key.clone());

        let mut out = f.debug_struct("CacheStats");
        for (key, count) in accesses {
            out.field(&key, count);
        }
        out.finish()
    }
}

impl std::ops::Deref for MemAccesses {
    type Target = HashMap<mem_fetch::AccessKind, usize>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for MemAccesses {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct InstructionCounts(pub HashMap<(super::instruction::MemorySpace, bool), usize>);

impl InstructionCounts {
    pub fn get_total(&self, space: super::instruction::MemorySpace) -> usize {
        let stores = self.0.get(&(space, true)).unwrap_or(&0);
        let loads = self.0.get(&(space, false)).unwrap_or(&0);
        stores + loads
    }

    pub fn inc(&mut self, space: super::instruction::MemorySpace, is_store: bool, count: usize) {
        *self.0.entry((space, is_store)).or_insert(0) += count;
    }
}

impl std::fmt::Debug for InstructionCounts {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut instructions: Vec<_> = self
            .0
            .iter()
            .filter(|(_, &count)| count > 0)
            .map(|((space, is_store), count)| {
                (
                    format!("{:?}[{}]", space, if *is_store { "STORE" } else { "LOAD" }),
                    count,
                )
            })
            .collect();
        instructions.sort_by_key(|(key, _)| key.clone());

        let mut out = f.debug_struct("InstructionCounts");
        for (key, count) in instructions {
            out.field(&key, count);
        }
        out.finish()
    }
}

impl std::ops::Deref for InstructionCounts {
    type Target = HashMap<(super::instruction::MemorySpace, bool), usize>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for InstructionCounts {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Stats {
    pub accesses: MemAccesses,
    pub instructions: InstructionCounts,
    pub sim: Sim,
    pub dram: DRAM,
    pub l1i_stats: PerCacheStats,
    pub l1c_stats: PerCacheStats,
    pub l1t_stats: PerCacheStats,
    pub l1d_stats: PerCacheStats,
    pub l2d_stats: PerCacheStats,
}

impl Stats {
    pub fn new(config: &config::GPUConfig) -> Self {
        Self {
            accesses: MemAccesses::default(),
            instructions: InstructionCounts::default(),
            sim: Sim::default(),
            dram: DRAM::new(config),
            l1i_stats: PerCacheStats::default(),
            l1c_stats: PerCacheStats::default(),
            l1t_stats: PerCacheStats::default(),
            l1d_stats: PerCacheStats::default(),
            l2d_stats: PerCacheStats::default(),
        }
    }
}

pub type CacheRequestStatusCounters = HashMap<(mem_fetch::AccessKind, cache::AccessStat), usize>;

#[derive(Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CacheStats {
    pub accesses: CacheRequestStatusCounters,
}

impl Default for CacheStats {
    fn default() -> Self {
        use strum::IntoEnumIterator;
        let mut accesses = HashMap::new();
        for access_kind in mem_fetch::AccessKind::iter() {
            for status in cache::RequestStatus::iter() {
                accesses.insert((access_kind, cache::AccessStat::Status(status)), 0);
            }
            for failure in cache::ReservationFailure::iter() {
                accesses.insert(
                    (access_kind, cache::AccessStat::ReservationFailure(failure)),
                    0,
                );
            }
        }
        Self { accesses }
    }
}

impl std::fmt::Debug for CacheStats {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut accesses: Vec<_> = self
            .accesses
            .iter()
            .filter(|(_, &count)| count > 0)
            .map(|((access_kind, access_stat), count)| {
                let key = match access_stat {
                    cache::AccessStat::Status(status) => {
                        // format!("{:?}[{:?}]={}", access_kind, status, count)
                        format!("{:?}[{:?}]", access_kind, status)
                    }
                    cache::AccessStat::ReservationFailure(failure) => {
                        // format!("{:?}[{:?}]={}", access_kind, failure, count)
                        format!("{:?}[{:?}]", access_kind, failure)
                    }
                };
                (key, count)
            })
            .collect();
        accesses.sort_by_key(|(key, _)| key.clone());

        let mut out = f.debug_struct("CacheStats");
        for (key, count) in accesses {
            out.field(&key, count);
        }
        out.finish()
    }
}

impl CacheStats {
    pub fn shave(&mut self) {
        self.accesses.retain(|_, v| *v > 0);
    }

    pub fn total_accesses(&self) -> usize {
        self.accesses.values().sum()
    }

    #[deprecated]
    pub fn sub_stats(&self) {
        use cache::{AccessStat, RequestStatus};
        let mut total_accesses = 0;
        let mut total_misses = 0;
        let mut total_pending_hits = 0;
        let mut total_reservation_fails = 0;
        for ((access_kind, status), accesses) in &self.accesses {
            if let AccessStat::Status(
                RequestStatus::HIT
                | RequestStatus::MISS
                | RequestStatus::SECTOR_MISS
                | RequestStatus::HIT_RESERVED,
            ) = status
            {
                total_accesses += accesses;
            }

            match status {
                AccessStat::Status(RequestStatus::MISS | RequestStatus::SECTOR_MISS) => {
                    total_misses += accesses;
                }
                AccessStat::Status(RequestStatus::HIT_RESERVED) => {
                    total_pending_hits += accesses;
                }
                AccessStat::Status(RequestStatus::RESERVATION_FAIL) => {
                    total_reservation_fails += accesses;
                }
                _ => {}
            }
        }
    }
}

impl std::ops::AddAssign for CacheStats {
    fn add_assign(&mut self, other: Self) {
        for (k, v) in other.accesses.into_iter() {
            *self.accesses.entry(k).or_insert(0) += v;
        }
    }
}

impl CacheStats {
    #[inline]
    pub fn inc(&mut self, kind: mem_fetch::AccessKind, access: cache::AccessStat, count: usize) {
        *self.accesses.entry((kind, access)).or_insert(0) += count;
    }

    /// This function selects how the cache access outcome should be counted.
    ///
    /// HIT_RESERVED is considered as a MISS in the cores, however, it should be
    /// counted as a HIT_RESERVED in the caches.
    pub fn select_status(
        probe: cache::RequestStatus,
        access: cache::RequestStatus,
    ) -> cache::RequestStatus {
        if probe == cache::RequestStatus::HIT_RESERVED
            && access != cache::RequestStatus::RESERVATION_FAIL
        {
            probe
        } else if probe == cache::RequestStatus::SECTOR_MISS && access == cache::RequestStatus::MISS
        {
            probe
        } else {
            access
        }
    }
}
