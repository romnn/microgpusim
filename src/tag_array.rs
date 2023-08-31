use super::{address, cache, mem_fetch};
use crate::config;

use std::collections::HashMap;
use std::sync::Arc;

pub type LineTable = HashMap<address, u64>;

#[derive(Debug, Clone, Default, Hash, PartialEq, Eq)]
pub struct EvictedBlockInfo {
    pub block_addr: address,
    pub allocation: Option<crate::allocation::Allocation>,
    pub modified_size: u32,
    pub byte_mask: mem_fetch::ByteMask,
    pub sector_mask: mem_fetch::SectorMask,
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct AccessStatus {
    pub index: Option<usize>,
    pub writeback: bool,
    pub evicted: Option<EvictedBlockInfo>,
    pub status: cache::RequestStatus,
}

#[derive(Debug, Clone)]
pub struct TagArray<B> {
    /// nbanks x nset x assoc lines in total
    pub lines: Vec<B>,
    // pub lines: Vec<cache::block::Line>,
    // phantom: std::marker::PhantomData<B>,
    is_used: bool,
    num_access: usize,
    num_miss: usize,
    num_pending_hit: usize,
    num_reservation_fail: usize,
    num_sector_miss: usize,
    pub num_dirty: usize,
    config: Arc<config::Cache>,
    pending_lines: LineTable,
}

impl<B> TagArray<B>
where
    B: Default,
{
    #[must_use]
    pub fn new(config: Arc<config::Cache>) -> Self {
        let num_cache_lines = config.max_num_lines();
        let lines = (0..num_cache_lines)
            .map(|_| B::default())
            // .map(|_| cache::block::Line::default())
            .collect();

        Self {
            lines,
            // phantom: std::marker::PhantomData,
            is_used: false,
            num_access: 0,
            num_miss: 0,
            num_pending_hit: 0,
            num_reservation_fail: 0,
            num_sector_miss: 0,
            num_dirty: 0,
            config,
            pending_lines: LineTable::new(),
        }
    }
}

pub trait Access<B> {
    /// Accesses the tag array
    #[must_use]
    fn access(&mut self, addr: address, fetch: &mem_fetch::MemFetch, time: u64) -> AccessStatus;

    fn flush(&mut self);

    fn invalidate(&mut self);

    #[must_use]
    fn size(&self) -> usize;

    #[must_use]
    fn get_block_mut(&mut self, idx: usize) -> &mut B;

    #[must_use]
    fn get_block(&self, idx: usize) -> &B;

    fn add_pending_line(&mut self, fetch: &mem_fetch::MemFetch);

    fn remove_pending_line(&mut self, fetch: &mem_fetch::MemFetch);
}

impl<B> Access<B> for TagArray<B>
where
    B: cache::block::Block,
{
    #[inline]
    fn access(&mut self, addr: address, fetch: &mem_fetch::MemFetch, time: u64) -> AccessStatus {
        log::trace!("tag_array::access({}, time={})", fetch, time);
        self.num_access += 1;
        self.is_used = true;

        let mut writeback = false;
        let mut evicted = None;

        let (index, status) = self.probe(addr, fetch, fetch.is_write(), false);
        match status {
            cache::RequestStatus::HIT | cache::RequestStatus::HIT_RESERVED => {
                if status == cache::RequestStatus::HIT_RESERVED {
                    self.num_pending_hit += 1;
                }

                // TODO: use an enum like either here
                let index = index.expect("hit has idx");
                let line = &mut self.lines[index];
                line.set_last_access_time(time, fetch.access_sector_mask());
            }
            cache::RequestStatus::MISS => {
                self.num_miss += 1;
                let index = index.expect("hit has idx");
                let line = &mut self.lines[index];

                log::trace!(
                    "tag_array::access({}, time={}) => {:?} line[{}]={} allocate policy={:?}",
                    fetch,
                    time,
                    status,
                    index,
                    line,
                    self.config.allocate_policy
                );

                if self.config.allocate_policy == config::CacheAllocatePolicy::ON_MISS {
                    if line.is_modified() {
                        writeback = true;
                        evicted = Some(EvictedBlockInfo {
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
                        index,
                        self.config.tag(addr),
                        line.is_modified(),
                        time,
                    );
                    line.allocate(
                        self.config.tag(addr),
                        self.config.block_addr(addr),
                        fetch.access_sector_mask(),
                        time,
                    );
                }
            }
            cache::RequestStatus::SECTOR_MISS => {
                debug_assert!(self.config.kind == config::CacheKind::Sector);
                self.num_sector_miss += 1;
                if self.config.allocate_policy == config::CacheAllocatePolicy::ON_MISS {
                    let index = index.expect("hit has idx");
                    let line = &mut self.lines[index];
                    let was_modified_before = line.is_modified();
                    line.allocate_sector(fetch.access_sector_mask(), time);
                    if was_modified_before && !line.is_modified() {
                        self.num_dirty -= 1;
                    }
                }
                unimplemented!("sector miss");
            }
            cache::RequestStatus::RESERVATION_FAIL => {
                self.num_reservation_fail += 1;
            }
            status @ cache::RequestStatus::MSHR_HIT => {
                panic!("tag_array access: status {status:?} should never be returned");
            }
        }
        AccessStatus {
            index,
            writeback,
            evicted,
            status,
        }
    }

    fn flush(&mut self) {
        todo!("flush tag array");
    }

    fn invalidate(&mut self) {
        todo!("invalidate tag array");
    }

    #[must_use]
    fn size(&self) -> usize {
        self.config.max_num_lines()
    }

    fn get_block_mut(&mut self, idx: usize) -> &mut B {
        &mut self.lines[idx]
    }

    #[must_use]
    fn get_block(&self, idx: usize) -> &B {
        &self.lines[idx]
    }

    fn add_pending_line(&mut self, fetch: &mem_fetch::MemFetch) {
        let addr = self.config.block_addr(fetch.addr());
        let instr = fetch.instr.as_ref().unwrap();
        if self.pending_lines.contains_key(&addr) {
            self.pending_lines.insert(addr, instr.uid);
        }
    }

    fn remove_pending_line(&mut self, fetch: &mem_fetch::MemFetch) {
        let addr = self.config.block_addr(fetch.addr());
        self.pending_lines.remove(&addr);
    }
}

impl<B> TagArray<B>
where
    B: cache::block::Block,
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
    ) -> (Option<usize>, cache::RequestStatus) {
        self.probe_masked(
            block_addr,
            fetch.access_sector_mask(),
            is_write,
            is_probe,
            Some(fetch),
        )
    }

    pub fn probe_masked(
        &self,
        block_addr: address,
        mask: &mem_fetch::SectorMask,
        is_write: bool,
        _is_probe: bool,
        fetch: Option<&mem_fetch::MemFetch>,
    ) -> (Option<usize>, cache::RequestStatus) {
        let set_index = self.config.set_index(block_addr) as usize;
        let tag = self.config.tag(block_addr);

        let mut invalid_line = None;
        let mut valid_line = None;
        let mut valid_time = u64::MAX;

        let mut all_reserved = true;

        // percentage of dirty lines in the cache
        // number of dirty lines / total lines in the cache
        let dirty_line_percent = self.num_dirty as f64 / self.config.total_lines() as f64;
        let dirty_line_percent = (dirty_line_percent * 100f64) as usize;

        log::trace!(
            "tag_array::probe({:?}) set_idx = {}, tag = {}, assoc = {} dirty lines = {}%",
            fetch.map(ToString::to_string),
            set_index,
            tag,
            self.config.associativity,
            dirty_line_percent,
        );

        // check for hit or pending hit
        for way in 0..self.config.associativity {
            let idx = set_index * self.config.associativity + way;
            let line = &self.lines[idx];
            log::trace!(
                "tag_array::probe({:?}) => checking cache index {} (tag={}, status={:?}, last_access={})",
                fetch.map(ToString::to_string),
                idx,
                line.tag(),
                line.status(mask),
                line.last_access_time()
            );
            if line.tag() == tag {
                match line.status(mask) {
                    cache::block::Status::RESERVED => {
                        return (Some(idx), cache::RequestStatus::HIT_RESERVED);
                    }
                    cache::block::Status::VALID => {
                        return (Some(idx), cache::RequestStatus::HIT);
                    }
                    cache::block::Status::MODIFIED => {
                        let status = if is_write || line.is_readable(mask) {
                            cache::RequestStatus::HIT
                        } else {
                            cache::RequestStatus::SECTOR_MISS
                        };
                        // let status = match is_write {
                        //     true => cache::RequestStatus::HIT,
                        //     false if line.is_readable(mask) => cache::RequestStatus::HIT,
                        //     _ => cache::RequestStatus::SECTOR_MISS,
                        // };
                        return (Some(idx), status);
                    }
                    cache::block::Status::INVALID if line.is_valid() => {
                        return (Some(idx), cache::RequestStatus::SECTOR_MISS);
                    }
                    cache::block::Status::INVALID => {}
                }
            }
            if !line.is_reserved() {
                // If the cacheline is from a load op (not modified),
                // or the total dirty cacheline is above a specific value,
                // Then this cacheline is eligible to be considered for replacement candidate
                // i.e. Only evict clean cachelines until total dirty cachelines reach the limit.
                if !line.is_modified()
                    || dirty_line_percent >= self.config.l1_cache_write_ratio_percent
                {
                    all_reserved = false;
                    if line.is_invalid() {
                        invalid_line = Some(idx);
                    } else {
                        // valid line: keep track of most appropriate replacement candidate
                        if self.config.replacement_policy == config::CacheReplacementPolicy::LRU {
                            if line.last_access_time() < valid_time {
                                valid_time = line.last_access_time();
                                valid_line = Some(idx);
                            }
                        } else if self.config.replacement_policy
                            == config::CacheReplacementPolicy::FIFO
                            && line.alloc_time() < valid_time
                        {
                            valid_time = line.alloc_time();
                            valid_line = Some(idx);
                        }
                    }
                }
            }
        }

        log::trace!("tag_array::probe({:?}) => all reserved={} invalid_line={:?} valid_line={:?} ({:?} policy)", fetch.map(ToString::to_string), all_reserved, invalid_line, valid_line, self.config.replacement_policy);

        if all_reserved {
            debug_assert_eq!(
                self.config.allocate_policy,
                config::CacheAllocatePolicy::ON_MISS
            );
            // miss and not enough space in cache to allocate on miss
            return (None, cache::RequestStatus::RESERVATION_FAIL);
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
        (Some(cache_idx), cache::RequestStatus::MISS)
    }

    pub fn fill_on_miss(&mut self, cache_index: usize, fetch: &mem_fetch::MemFetch, time: u64) {
        debug_assert!(self.config.allocate_policy == config::CacheAllocatePolicy::ON_MISS);

        log::trace!(
            "tag_array::fill(cache={}, tag={}, addr={}) (on miss)",
            cache_index,
            self.config.tag(fetch.addr()),
            fetch.addr(),
        );

        let was_modified_before = self.lines[cache_index].is_modified();
        self.lines[cache_index].fill(fetch.access_sector_mask(), fetch.access_byte_mask(), time);
        if self.lines[cache_index].is_modified() && !was_modified_before {
            self.num_dirty += 1;
        }
    }

    /// TODO: consolidate with fill on fill
    pub fn populate_memcopy(
        &mut self,
        addr: address,
        sector_mask: mem_fetch::SectorMask,
        byte_mask: mem_fetch::ByteMask,
        is_write: bool,
        time: u64,
    ) {
        let is_probe = false;
        let (cache_index, probe_status) =
            self.probe_masked(addr, &sector_mask, is_write, is_probe, None);

        log::trace!(
            "tag_array::fill(cache={}, tag={}, addr={}) (on fill) status={:?}",
            cache_index.map_or(-1, |i| i as i64),
            self.config.tag(addr),
            addr,
            probe_status,
        );

        if probe_status == cache::RequestStatus::RESERVATION_FAIL {
            return;
        }
        let cache_index = cache_index.unwrap();

        let line = self.lines.get_mut(cache_index).unwrap();
        let mut was_modified_before = line.is_modified();

        if probe_status == cache::RequestStatus::MISS {
            log::trace!(
                "tag_array::allocate(cache={}, tag={}, time={})",
                cache_index,
                self.config.tag(addr),
                time,
            );
            line.allocate(
                self.config.tag(addr),
                self.config.block_addr(addr),
                &sector_mask,
                time,
            );
        } else if probe_status == cache::RequestStatus::SECTOR_MISS {
            debug_assert_eq!(self.config.kind, config::CacheKind::Sector);
            line.allocate_sector(&sector_mask, time);
        }
        if was_modified_before && !line.is_modified() {
            self.num_dirty -= 1;
        }
        was_modified_before = line.is_modified();
        line.fill(&sector_mask, &byte_mask, time);
        if line.is_modified() && !was_modified_before {
            self.num_dirty += 1;
        }
    }

    /// TODO: consolidate with populate memcopy
    pub fn fill_on_fill(&mut self, addr: address, fetch: &mem_fetch::MemFetch, time: u64) {
        debug_assert!(self.config.allocate_policy == config::CacheAllocatePolicy::ON_FILL);

        // probe tag array
        let is_probe = false;
        let (cache_index, probe_status) = self.probe(addr, fetch, fetch.is_write(), is_probe);

        log::trace!(
            "tag_array::fill(cache={}, tag={}, addr={}) (on fill) status={:?}",
            cache_index.map_or(-1, |i| i as i64),
            self.config.tag(fetch.addr()),
            fetch.addr(),
            probe_status,
        );

        if probe_status == cache::RequestStatus::RESERVATION_FAIL {
            return;
        }
        let cache_index = cache_index.unwrap();

        let line = self.lines.get_mut(cache_index).unwrap();
        let mut was_modified_before = line.is_modified();

        if probe_status == cache::RequestStatus::MISS {
            log::trace!(
                "tag_array::allocate(cache={}, tag={}, time={})",
                cache_index,
                self.config.tag(addr),
                time,
            );
            line.allocate(
                self.config.tag(addr),
                self.config.block_addr(addr),
                fetch.access_sector_mask(),
                time,
            );
        } else if probe_status == cache::RequestStatus::SECTOR_MISS {
            debug_assert_eq!(self.config.kind, config::CacheKind::Sector);
            line.allocate_sector(fetch.access_sector_mask(), time);
        }
        if was_modified_before && !line.is_modified() {
            self.num_dirty -= 1;
        }
        was_modified_before = line.is_modified();
        line.fill(fetch.access_sector_mask(), fetch.access_byte_mask(), time);
        if line.is_modified() && !was_modified_before {
            self.num_dirty += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TagArray;
    use crate::config;
    use std::sync::Arc;

    #[ignore = "todo"]
    #[test]
    fn test_tag_array() {
        let config = config::GPU::default().data_cache_l1.unwrap();
        let _tag_array: TagArray<usize> = TagArray::new(Arc::clone(&config.inner));
    }
}
