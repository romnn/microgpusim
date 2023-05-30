use super::{cache, mem_fetch};
use std::collections::HashMap;
use std::sync::Mutex;

pub type CacheRequestStatusCounters = HashMap<(mem_fetch::AccessKind, cache::AccessStat), usize>;

#[derive(Default, Debug)]
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
    pub accesses: CacheRequestStatusCounters,
}

impl Stats {
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
}

// avoid issues with running test / and or parallel instances
// lazy_static::lazy_static! {
//     pub static ref STATS: Mutex<Stats> = Mutex::new(Stats::default());
// }
