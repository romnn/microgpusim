use super::{address, cache, mem_fetch};
use crate::{config, mem_sub_partition::SECTOR_CHUNK_SIZE};

use std::collections::HashMap;

pub type LineTable = HashMap<address, u64>;

#[derive(Debug, Clone, Default, Hash, PartialEq, Eq)]
pub struct EvictedBlockInfo {
    pub writeback: bool,
    pub block_addr: address,
    pub allocation: Option<crate::allocation::Allocation>,
    pub modified_size: u32,
    pub byte_mask: mem_fetch::ByteMask,
    pub sector_mask: mem_fetch::SectorMask,
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct AccessStatus {
    pub cache_index: Option<usize>,
    // pub writeback: bool,
    pub evicted: Option<EvictedBlockInfo>,
    pub status: cache::RequestStatus,
}

/// Tag array configuration.
// #[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
// pub struct Config {
//     allocate_policy: config::CacheAllocatePolicy,
//     replacement_policy: config::CacheReplacementPolicy,
//     max_num_lines: usize,
//     addr_translation: Box<dyn CacheAddressTranslation>,
// }

/// Tag array.
#[derive(Debug)]
pub struct TagArray<B, CC> {
    /// Cache lines
    ///
    /// The number of lines is nbanks x nset x assoc lines
    pub lines: Vec<B>,
    is_used: bool,
    num_access: usize,
    num_miss: usize,
    num_pending_hit: usize,
    num_reservation_fail: usize,
    pub num_dirty: usize,
    max_dirty_cache_lines_percent: usize,
    cache_controller: CC,
    cache_config: cache::Config,
    pending_lines: LineTable,
}

impl<B, CC> TagArray<B, CC>
where
    B: Default,
{
    #[must_use]
    pub fn new(config: &config::Cache, cache_controller: CC, accelsim_compat: bool) -> Self {
        let num_cache_lines = config.max_num_lines();
        let lines = (0..num_cache_lines).map(|_| B::default()).collect();

        let cache_config = cache::Config::new(config, accelsim_compat);

        Self {
            lines,
            is_used: false,
            num_access: 0,
            num_miss: 0,
            num_pending_hit: 0,
            num_reservation_fail: 0,
            num_dirty: 0,
            max_dirty_cache_lines_percent: config.l1_cache_write_ratio_percent,
            cache_config: cache_config.clone(),
            cache_controller,
            pending_lines: LineTable::new(),
        }
    }
}

pub trait Access<B> {
    /// Accesses the tag array.
    #[must_use]
    fn access(&mut self, addr: address, fetch: &mem_fetch::MemFetch, time: u64) -> AccessStatus;

    /// Flushes all dirty (modified) lines to the upper level cache.
    ///
    /// # Returns
    /// The number of dirty lines flushed.
    fn flush(&mut self) -> usize;

    /// Invalidates all tags stored in this array.
    ///
    /// This effectively resets the tag array.
    fn invalidate(&mut self);

    /// The maximum number of tags this array can hold.
    #[must_use]
    fn size(&self) -> usize;

    /// Get a mutable reference to a block.
    #[must_use]
    fn get_block_mut(&mut self, idx: usize) -> &mut B;

    /// Get a reference to a block
    #[must_use]
    fn get_block(&self, idx: usize) -> &B;

    /// Add a new pending line for a fetch request.
    fn add_pending_line(&mut self, fetch: &mem_fetch::MemFetch);

    /// Remove pending line for a fetch request.
    fn remove_pending_line(&mut self, fetch: &mem_fetch::MemFetch);
}

impl<B, T> Access<B> for TagArray<B, T>
where
    B: cache::block::Block,
    T: cache::CacheController,
{
    // #[inline]
    fn access(&mut self, addr: address, fetch: &mem_fetch::MemFetch, time: u64) -> AccessStatus {
        log::trace!("tag_array::access({}, time={})", fetch, time);
        self.num_access += 1;
        self.is_used = true;

        // let mut writeback = false;
        let mut evicted = None;

        let Some((cache_index, status)) = self.probe(addr, fetch, fetch.is_write(), false) else {
            self.num_reservation_fail += 1;
            return AccessStatus {
                cache_index: None,
                // writeback,
                evicted,
                status: cache::RequestStatus::RESERVATION_FAIL,
            };
        };

        match status {
            cache::RequestStatus::RESERVATION_FAIL => {
                self.num_reservation_fail += 1;
            }
            cache::RequestStatus::HIT | cache::RequestStatus::HIT_RESERVED => {
                if status == cache::RequestStatus::HIT_RESERVED {
                    self.num_pending_hit += 1;
                }

                let line = &mut self.lines[cache_index];
                line.set_last_access_time(time, &fetch.access.sector_mask);
            }
            cache::RequestStatus::MISS => {
                self.num_miss += 1;
                let line = &mut self.lines[cache_index];

                log::trace!(
                    "tag_array::access({}, time={}) => {:?} line[{}]={} allocate policy={:?}",
                    fetch,
                    time,
                    status,
                    cache_index,
                    line,
                    self.cache_config.allocate_policy
                );

                if self.cache_config.allocate_policy == cache::config::AllocatePolicy::ON_MISS {
                    if line.is_modified() {
                        // writeback = true;
                        evicted = Some(EvictedBlockInfo {
                            writeback: true,
                            allocation: fetch.access.allocation.clone(),
                            block_addr: line.block_addr(),
                            modified_size: line.modified_size(),
                            byte_mask: line.dirty_byte_mask(),
                            sector_mask: line.dirty_sector_mask(),
                        });
                        self.num_dirty -= 1;
                    }
                    log::trace!(
                        "tag_array::allocate(cache={}, tag={}, modified={}, time={})",
                        cache_index,
                        self.cache_controller.tag(addr),
                        line.is_modified(),
                        time,
                    );
                    line.allocate(
                        self.cache_controller.tag(addr),
                        self.cache_controller.block_addr(addr),
                        &fetch.access.sector_mask,
                        time,
                    );
                }
            }
            cache::RequestStatus::SECTOR_MISS => {
                // debug_assert_eq!(self.cache_config.kind, config::CacheKind::Sector);
                // self.num_sector_miss += 1;
                if self.cache_config.allocate_policy == cache::config::AllocatePolicy::ON_MISS {
                    let line = &mut self.lines[cache_index];
                    let was_modified_before = line.is_modified();
                    line.allocate_sector(&fetch.access.sector_mask, time);
                    if was_modified_before && !line.is_modified() {
                        self.num_dirty -= 1;
                    }
                }
            }
            cache::RequestStatus::MSHR_HIT => {
                panic!("tag_array access: status {status:?} should never be returned");
            }
        }
        AccessStatus {
            cache_index: Some(cache_index),
            // writeback,
            evicted,
            status,
        }
    }

    // #[inline]
    fn flush(&mut self) -> usize {
        let mut flushed = 0;
        for line in &mut self.lines {
            if line.is_modified() {
                for i in 0..SECTOR_CHUNK_SIZE {
                    let mut sector_mask = mem_fetch::SectorMask::ZERO;
                    sector_mask.set(i as usize, true);
                    line.set_status(cache::block::Status::INVALID, &sector_mask);
                }
                flushed += 1;
            }
        }
        self.num_dirty = 0;
        flushed
    }

    // #[inline]
    fn invalidate(&mut self) {
        for line in &mut self.lines {
            for i in 0..SECTOR_CHUNK_SIZE {
                let mut sector_mask = mem_fetch::SectorMask::ZERO;
                sector_mask.set(i as usize, true);
                line.set_status(cache::block::Status::INVALID, &sector_mask);
            }
        }
        self.num_dirty = 0;
    }

    // #[inline]
    fn size(&self) -> usize {
        self.lines.len()
    }

    // #[inline]
    fn get_block_mut(&mut self, idx: usize) -> &mut B {
        &mut self.lines[idx]
    }

    // #[inline]
    fn get_block(&self, idx: usize) -> &B {
        &self.lines[idx]
    }

    // #[inline]
    fn add_pending_line(&mut self, fetch: &mem_fetch::MemFetch) {
        let addr = self.cache_controller.block_addr(fetch.addr());
        let instr = fetch.instr.as_ref().unwrap();
        if self.pending_lines.contains_key(&addr) {
            self.pending_lines.insert(addr, instr.uid);
        }
    }

    // #[inline]
    fn remove_pending_line(&mut self, fetch: &mem_fetch::MemFetch) {
        let addr = self.cache_controller.block_addr(fetch.addr());
        self.pending_lines.remove(&addr);
    }
}

impl<B, T> TagArray<B, T>
where
    B: cache::block::Block,
    T: cache::CacheController,
{
    /// Probes the tag array
    ///
    /// # Returns
    /// A tuple with the cache index `Option<usize>` and cache request status.
    #[must_use]
    pub fn probe(
        &self,
        block_addr: address,
        fetch: &mem_fetch::MemFetch,
        is_write: bool,
        is_probe: bool,
    ) -> Option<(usize, cache::RequestStatus)> {
        self.probe_masked(
            block_addr,
            &fetch.access.sector_mask,
            is_write,
            is_probe,
            Some(fetch),
        )
    }

    pub fn probe_masked(
        &self,
        block_addr: address,
        sector_mask: &mem_fetch::SectorMask,
        is_write: bool,
        _is_probe: bool,
        fetch: Option<&mem_fetch::MemFetch>,
    ) -> Option<(usize, cache::RequestStatus)> {
        let set_index = self.cache_controller.set_index(block_addr) as usize;
        let tag = self.cache_controller.tag(block_addr);

        let mut invalid_line = None;
        let mut valid_line = None;
        let mut valid_time = u64::MAX;

        let mut all_reserved = true;

        // percentage of dirty lines in the cache
        // number of dirty lines / total lines in the cache
        let dirty_line_percent = self.num_dirty as f64 / self.cache_config.total_lines as f64;
        let dirty_line_percent = (dirty_line_percent * 100f64) as usize;

        log::trace!(
            "tag_array::probe({}) set_idx = {}, tag = {}, assoc = {} dirty lines = {}%",
            crate::Optional(fetch),
            set_index,
            tag,
            self.cache_config.associativity,
            dirty_line_percent,
        );

        // check for hit or pending hit
        for way in 0..self.cache_config.associativity {
            let idx = set_index * self.cache_config.associativity + way;
            let line = &self.lines[idx];
            log::trace!(
                "tag_array::probe({}) => checking cache index {} (tag={}, status={:?}, last_access={})",
                crate::Optional(fetch),
                idx,
                line.tag(),
                line.status(sector_mask),
                line.last_access_time()
            );
            if line.tag() == tag {
                // if (line->get_status(mask) == RESERVED) {
                //     idx = index;
                //     return HIT_RESERVED;
                //   } else if (line->get_status(mask) == VALID) {
                //     idx = index;
                //     return HIT;
                //   } else if (line->get_status(mask) == MODIFIED) {
                //     if ((!is_write && line->is_readable(mask)) || is_write) {
                //       idx = index;
                //       return HIT;
                //     } else {
                //       idx = index;
                //       return SECTOR_MISS;
                //     }
                //
                //   } else if (line->is_valid_line() && line->get_status(mask) == INVALID) {
                //     idx = index;
                //     return SECTOR_MISS;
                //   } else {
                //     assert(line->get_status(mask) == INVALID);
                //   }
                match line.status(sector_mask) {
                    cache::block::Status::RESERVED => {
                        return Some((idx, cache::RequestStatus::HIT_RESERVED));
                    }
                    cache::block::Status::VALID => {
                        return Some((idx, cache::RequestStatus::HIT));
                    }
                    cache::block::Status::MODIFIED => {
                        let status = if is_write || (!is_write && line.is_readable(sector_mask)) {
                            cache::RequestStatus::HIT
                        } else {
                            cache::RequestStatus::SECTOR_MISS
                        };
                        return Some((idx, status));
                    }
                    cache::block::Status::INVALID => {
                        if line.is_valid() {
                            return Some((idx, cache::RequestStatus::SECTOR_MISS));
                        }
                    }
                }
            }
            if !line.is_reserved() {
                // If the cache line is from a load op (not modified),
                // or the number of total dirty cache lines is above a specific value,
                // the cache line is eligible to be considered for replacement candidate
                //
                // i.e. only evict clean cache lines until total dirty cache lines reach the limit.
                if !line.is_modified() || dirty_line_percent >= self.max_dirty_cache_lines_percent {
                    all_reserved = false;
                    if line.is_invalid() {
                        invalid_line = Some(idx);
                    } else {
                        // valid line: keep track of most appropriate replacement candidate
                        if self.cache_config.replacement_policy
                            == cache::config::ReplacementPolicy::LRU
                        {
                            if line.last_access_time() < valid_time {
                                valid_time = line.last_access_time();
                                valid_line = Some(idx);
                            }
                        } else if self.cache_config.replacement_policy
                            == cache::config::ReplacementPolicy::FIFO
                            && line.alloc_time() < valid_time
                        {
                            valid_time = line.alloc_time();
                            valid_line = Some(idx);
                        }
                    }
                }
            }
        }

        log::trace!(
            "tag_array::probe({}) => all reserved={} invalid_line={:?} valid_line={:?} ({:?} policy)",
            crate::Optional(fetch),
            all_reserved,
            invalid_line,
            valid_line,
            self.cache_config.replacement_policy,
        );

        if all_reserved {
            debug_assert_eq!(
                self.cache_config.allocate_policy,
                cache::config::AllocatePolicy::ON_MISS
            );
            // miss and not enough space in cache to allocate on miss
            return None;
            // return cache::RequestStatus::RESERVATION_FAIL;
        }

        let cache_idx = match (valid_line, invalid_line) {
            (_, Some(invalid)) => invalid,
            (Some(valid), None) => valid,
            (None, None) => {
                // if an unreserved block exists,
                // it is either invalid or replaceable
                panic!("found neither a valid nor invalid cache line");
            }
        };
        Some((cache_idx, cache::RequestStatus::MISS))
    }

    pub fn fill_on_miss(
        &mut self,
        cache_index: usize,
        addr: address,
        sector_mask: &mem_fetch::SectorMask,
        byte_mask: &mem_fetch::ByteMask,
        time: u64,
    ) {
        debug_assert!(self.cache_config.allocate_policy == cache::config::AllocatePolicy::ON_MISS);

        log::trace!(
            "tag_array::fill(cache={}, tag={}, addr={}) (on miss)",
            cache_index,
            self.cache_controller.tag(addr),
            addr,
        );

        let was_modified_before = self.lines[cache_index].is_modified();
        self.lines[cache_index].fill(sector_mask, byte_mask, time);
        if self.lines[cache_index].is_modified() && !was_modified_before {
            self.num_dirty += 1;
        }
    }

    pub fn fill_on_fill(
        &mut self,
        addr: address,
        sector_mask: &mem_fetch::SectorMask,
        byte_mask: &mem_fetch::ByteMask,
        is_write: bool,
        time: u64,
    ) {
        let is_probe = false;
        let probe = self.probe_masked(addr, sector_mask, is_write, is_probe, None);
        let Some((cache_index, probe_status)) = probe else {
            return;
        };

        if probe_status == cache::RequestStatus::RESERVATION_FAIL {
            return;
        }

        log::trace!(
            "tag_array::fill(cache={}, tag={}, addr={}, time={}) (on fill) status={:?}",
            cache_index,
            self.cache_controller.tag(addr),
            addr,
            time,
            probe_status,
        );

        let line = &mut self.lines[cache_index];
        let mut was_modified_before = line.is_modified();

        if probe_status == cache::RequestStatus::MISS {
            log::trace!(
                "tag_array::allocate(cache={}, tag={}, time={})",
                cache_index,
                self.cache_controller.tag(addr),
                time,
            );
            line.allocate(
                self.cache_controller.tag(addr),
                self.cache_controller.block_addr(addr),
                sector_mask,
                time,
            );
        } else if probe_status == cache::RequestStatus::SECTOR_MISS {
            line.allocate_sector(&sector_mask, time);
        }
        if was_modified_before && !line.is_modified() {
            self.num_dirty -= 1;
        }
        was_modified_before = line.is_modified();
        line.fill(sector_mask, byte_mask, time);
        if line.is_modified() && !was_modified_before {
            self.num_dirty += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    // use super::TagArray;
    // use crate::config;
    // use std::sync::Arc;

    #[ignore = "todo"]
    #[test]
    fn test_tag_array() {
        // let config = config::GPU::default().data_cache_l1.unwrap();
        // let _tag_array: TagArray<usize, T> = TagArray::new(Arc::clone(&config.inner));
    }
}
