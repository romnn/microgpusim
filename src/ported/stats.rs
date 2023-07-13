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

#[derive(Clone, Default, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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

    pub l1_data: CacheStats,
}

#[derive(Clone, Default, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CacheStats {
    pub accesses: CacheRequestStatusCounters,
}

impl CacheStats {
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
            // if let AccessStat::Status(RequestStatus::MISS | RequestStatus::SECTOR_MISS) = status {
            //     total_misses += accesses;
            // }
            //
            // if let AccessStat::Status(RequestStatus::HIT_RESERVED) = status {
            //     total_pending_hits += accesses;
            // }
            //
            // if let AccessStat::Status(RequestStatus::RESERVATION_FAIL) = status {
            //     total_reservation_fails += accesses;
            // }
        }
    }
    // for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
    //     for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
    //       if (status == HIT || status == MISS || status == SECTOR_MISS ||
    //           status == HIT_RESERVED)
    //         t_css.accesses += m_stats[type][status];
    //
    //       if (status == MISS || status == SECTOR_MISS)
    //         t_css.misses += m_stats[type][status];
    //
    //       if (status == HIT_RESERVED) t_css.pending_hits += m_stats[type][status];
    //
    //       if (status == RESERVATION_FAIL) t_css.res_fails += m_stats[type][status];
    //     }
    //   }
}

impl std::ops::AddAssign for CacheStats {
    fn add_assign(&mut self, other: Self) {
        for (k, v) in other.accesses.into_iter() {
            *self.accesses.entry(k).or_insert(0) += v;
        }
    }
}

impl CacheStats {
    // pub fn add(&mut self, other: AccessKind, access: cache::AccessStat) {
    pub fn inc_access(&mut self, kind: mem_fetch::AccessKind, access: cache::AccessStat) {
        *self.accesses.entry((kind, access)).or_insert(0) += 1;

        // self.accesses
        //     .entry((kind, access))
        //     .and_modify(|s| *s += 1)
        //     .or_insert(1);
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
