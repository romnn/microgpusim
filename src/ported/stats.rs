use super::{cache, mem_fetch};
use crate::config;
use std::collections::HashMap;
use std::sync::Mutex;

pub type CacheRequestStatusCounters = HashMap<(mem_fetch::AccessKind, cache::AccessStat), usize>;

#[derive(Clone, Default, Debug, serde::Serialize, serde::Deserialize)]
pub struct DRAMStats {
    /// bank writes [shader id][dram chip id][bank id]
    pub bank_writes: Vec<Vec<Vec<u64>>>,
    /// bank reads [shader id][dram chip id][bank id]
    pub bank_reads: Vec<Vec<Vec<u64>>>,
    /// bank writes [dram chip id][bank id]
    pub total_bank_writes: Vec<Vec<u64>>,
    /// bank reads [dram chip id][bank id]
    pub total_bank_reads: Vec<Vec<u64>>,
    /// bank accesses [dram chip id][bank id]
    pub total_bank_accesses: Vec<Vec<u64>>,
}

impl DRAMStats {
    pub fn new(config: &config::GPUConfig) -> Self {
        let mut total_bank_reads = Vec::new();
        let num_dram_banks = 0;
        for chip_id in 0..config.num_mem_units {
            total_bank_reads.push(vec![0; num_dram_banks]);
        }
        Self {
            total_bank_reads,
            ..Self::default()
        }
    }
}

#[derive(Clone, Default, Debug, serde::Serialize, serde::Deserialize)]
pub struct Stats {
    pub num_mem_write: usize,
    pub num_mem_read: usize,
    pub num_mem_const: usize,
    pub num_mem_texture: usize,
    pub num_mem_read_global: usize,
    pub num_mem_write_global: usize,
    pub num_mem_read_local: usize,
    pub num_mem_write_local: usize,
    pub num_mem_read_inst: usize,
    pub num_mem_l2_writeback: usize,
    pub num_mem_l1_write_allocate: usize,
    pub num_mem_l2_write_allocate: usize,
}

#[derive(Clone, Default, Debug, serde::Serialize, serde::Deserialize)]
pub struct CacheStats {
    pub accesses: CacheRequestStatusCounters,
}

impl std::ops::AddAssign for CacheStats {
    // type Output = Self;

    fn add_assign(&mut self, other: Self) {
        for (k, v) in other.accesses.into_iter() {
            *self.accesses.entry(k).or_insert(0) += v;
        }
        // self
    }
}

impl CacheStats {
    // pub fn add(&mut self, other: AccessKind, access: cache::AccessStat) {
    pub fn inc_access(&mut self, kind: mem_fetch::AccessKind, access: cache::AccessStat) {
        self.accesses
            .entry((kind, access))
            .and_modify(|s| *s += 1)
            .or_insert(1);
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

    // for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
    //     for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
    //       fprintf(fout, "\t%s[%s][%s] = %llu\n", m_cache_name.c_str(),
    //               mem_access_type_str((enum mem_access_type)type),
    //               cache_request_status_str((enum cache_request_status)status),
    //               m_stats[type][status]);
    //
    //       if (status != RESERVATION_FAIL && status != MSHR_HIT)
    //         // MSHR_HIT is a special type of SECTOR_MISS
    //         // so its already included in the SECTOR_MISS
    //         total_access[type] += m_stats[type][status];
    //     }
    //   }
    //   for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
    //     if (total_access[type] > 0)
    //       fprintf(fout, "\t%s[%s][%s] = %u\n", m_cache_name.c_str(),
    //               mem_access_type_str((enum mem_access_type)type), "TOTAL_ACCESS",
    //               total_access[type]);
    //   }
}
