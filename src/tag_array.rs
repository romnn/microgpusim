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

/// Tag array configuration.
// #[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
// pub struct Config {
//     allocate_policy: config::CacheAllocatePolicy,
//     replacement_policy: config::CacheReplacementPolicy,
//     max_num_lines: usize,
//     addr_translation: Box<dyn CacheAddressTranslation>,
// }

pub trait CacheAddressTranslation: std::fmt::Debug + Sync + Send + 'static {
    /// Compute cache line tag for an address.
    #[must_use]
    fn tag(&self, addr: address) -> address;

    /// Compute block address for an address.
    #[must_use]
    fn block_addr(&self, addr: address) -> address;

    /// Compute set index for an address.
    #[must_use]
    fn set_index(&self, addr: address) -> u64;

    /// Compute miss status handling register address.
    ///
    /// The default implementation uses the block address.
    #[must_use]
    fn mshr_addr(&self, addr: address) -> address {
        self.block_addr(addr)
    }
}

#[derive(Debug, Clone)]
pub struct Pascal {
    set_index_function: crate::set_index::linear::SetIndex,
    config: cache::Config,
}

impl Pascal {
    pub fn new(config: cache::Config) -> Self {
        Self {
            config,
            set_index_function: crate::set_index::linear::SetIndex::default(),
        }
    }
}

// TODO: make a new address translation unit to get the set indexing function out of the config
// impl CacheAddressTranslation for cache::Config {
impl CacheAddressTranslation for Pascal {
    // impl<I> CacheAddressTranslation for CacheConfig<I>
    // where
    //     I: crate::set_index::SetIndexer,
    // {
    #[inline]
    fn tag(&self, addr: address) -> address {
        // For generality, the tag includes both index and tag.
        // This allows for more complex set index calculations that
        // can result in different indexes mapping to the same set,
        // thus the full tag + index is required to check for hit/miss.
        // Tag is now identical to the block address.

        // return addr >> (m_line_sz_log2+m_nset_log2);
        // return addr & ~(new_addr_type)(m_line_sz - 1);
        addr & !u64::from(self.config.line_size - 1)
    }

    #[inline]
    fn block_addr(&self, addr: address) -> address {
        addr & !u64::from(self.config.line_size - 1)
    }

    #[inline]
    fn set_index(&self, addr: address) -> u64 {
        use crate::set_index::SetIndexer;
        self.set_index_function.compute_set_index(
            addr,
            self.config.num_sets,
            self.config.line_size_log2,
            self.config.num_sets_log2,
        )
    }

    #[inline]
    fn mshr_addr(&self, addr: address) -> address {
        addr & !u64::from(self.config.line_size - 1)
    }
}

/// Tag array.
#[derive(Debug)]
pub struct TagArray<B, T> {
    /// nbanks x nset x assoc lines in total
    pub lines: Vec<B>,
    is_used: bool,
    num_access: usize,
    num_miss: usize,
    num_pending_hit: usize,
    num_reservation_fail: usize,
    // num_sector_miss: usize,
    pub num_dirty: usize,
    // config: Arc<config::Cache>,
    // config: CacheConfig,
    // l1_cache_write_ratio_percent: usize,
    max_dirty_cache_lines_percent: usize,
    // addr_translation: Box<dyn CacheAddressTranslation>,
    addr_translation: T,
    cache_config: cache::Config,
    pending_lines: LineTable,
}

// CacheAddressTranslation

// impl<B> TagArray<B>
impl<B> TagArray<B, Pascal>
where
    B: Default,
{
    #[must_use]
    pub fn new(config: Arc<config::Cache>) -> Self {
        let num_cache_lines = config.max_num_lines();
        let lines = (0..num_cache_lines).map(|_| B::default()).collect();

        let cache_config = cache::Config::from(&*config);
        // let cache_config = cache::Config {
        //     set_index_function: Arc::<crate::set_index::linear::SetIndex>::default(),
        //     allocate_policy: config.allocate_policy,
        //     replacement_policy: config.replacement_policy,
        //     associativity: config.associativity,
        //     num_sets: config.num_sets,
        //     total_lines: config.num_sets * config.associativity,
        //     line_size: config.line_size,
        //     line_size_log2: config.line_size_log2(),
        //     num_sets_log2: config.num_sets_log2(),
        // };

        Self {
            lines,
            is_used: false,
            num_access: 0,
            num_miss: 0,
            num_pending_hit: 0,
            num_reservation_fail: 0,
            // num_sector_miss: 0,
            num_dirty: 0,
            // config: Config {
            //     allocate_policy: config.allocate_policy,
            //     replacement_policy: config.replacement_policy,
            //     max_num_lines: config.max_num_lines(),
            //     addr_translation: todo!(),
            // },
            // l1_cache_write_ratio_percent: config.l1_cache_write_ratio_percent,
            // max_dirty_cache_lines_threshold: config.l1_cache_write_ratio_percent,
            max_dirty_cache_lines_percent: config.l1_cache_write_ratio_percent,
            cache_config: cache_config.clone(),
            addr_translation: Pascal::new(cache_config),
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
// impl<B> Access<B> for TagArray<B>
where
    B: cache::block::Block,
    T: CacheAddressTranslation,
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
                line.set_last_access_time(time, &fetch.access.sector_mask);
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
                    self.cache_config.allocate_policy
                );

                if self.cache_config.allocate_policy == cache::config::AllocatePolicy::ON_MISS {
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
                        self.addr_translation.tag(addr),
                        line.is_modified(),
                        time,
                    );
                    line.allocate(
                        self.addr_translation.tag(addr),
                        self.addr_translation.block_addr(addr),
                        &fetch.access.sector_mask,
                        time,
                    );
                }
            }
            cache::RequestStatus::SECTOR_MISS => {
                // debug_assert!(self.config.kind == config::CacheKind::Sector);
                // self.num_sector_miss += 1;
                // if self.config.allocate_policy == config::CacheAllocatePolicy::ON_MISS {
                //     let index = index.expect("hit has idx");
                //     let line = &mut self.lines[index];
                //     let was_modified_before = line.is_modified();
                //     line.allocate_sector(fetch.access_sector_mask(), time);
                //     if was_modified_before && !line.is_modified() {
                //         self.num_dirty -= 1;
                //     }
                // }
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

    #[inline]
    fn flush(&mut self) -> usize {
        use crate::mem_sub_partition::SECTOR_CHUNCK_SIZE;
        let mut flushed = 0;
        for line in self.lines.iter_mut() {
            if line.is_modified() {
                for i in 0..SECTOR_CHUNCK_SIZE {
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

    #[inline]
    fn invalidate(&mut self) {
        use crate::mem_sub_partition::SECTOR_CHUNCK_SIZE;
        for line in self.lines.iter_mut() {
            for i in 0..SECTOR_CHUNCK_SIZE {
                let mut sector_mask = mem_fetch::SectorMask::ZERO;
                sector_mask.set(i as usize, true);
                line.set_status(cache::block::Status::INVALID, &sector_mask);
            }
        }
        self.num_dirty = 0;
    }

    #[inline]
    fn size(&self) -> usize {
        self.lines.len()
    }

    #[inline]
    fn get_block_mut(&mut self, idx: usize) -> &mut B {
        &mut self.lines[idx]
    }

    #[inline]
    fn get_block(&self, idx: usize) -> &B {
        &self.lines[idx]
    }

    #[inline]
    fn add_pending_line(&mut self, fetch: &mem_fetch::MemFetch) {
        let addr = self.addr_translation.block_addr(fetch.addr());
        let instr = fetch.instr.as_ref().unwrap();
        if self.pending_lines.contains_key(&addr) {
            self.pending_lines.insert(addr, instr.uid);
        }
    }

    #[inline]
    fn remove_pending_line(&mut self, fetch: &mem_fetch::MemFetch) {
        let addr = self.addr_translation.block_addr(fetch.addr());
        self.pending_lines.remove(&addr);
    }
}

// impl<B> TagArray<B>
impl<B, T> TagArray<B, T>
where
    B: cache::block::Block,
    T: CacheAddressTranslation,
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
            &fetch.access.sector_mask,
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
        let set_index = self.addr_translation.set_index(block_addr) as usize;
        let tag = self.addr_translation.tag(block_addr);

        let mut invalid_line = None;
        let mut valid_line = None;
        let mut valid_time = u64::MAX;

        let mut all_reserved = true;

        // percentage of dirty lines in the cache
        // number of dirty lines / total lines in the cache
        let dirty_line_percent = self.num_dirty as f64 / self.cache_config.total_lines as f64;
        let dirty_line_percent = (dirty_line_percent * 100f64) as usize;

        log::trace!(
            "tag_array::probe({:?}) set_idx = {}, tag = {}, assoc = {} dirty lines = {}%",
            fetch.map(ToString::to_string),
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

        log::trace!("tag_array::probe({:?}) => all reserved={} invalid_line={:?} valid_line={:?} ({:?} policy)", fetch.map(ToString::to_string), all_reserved, invalid_line, valid_line, self.cache_config.replacement_policy);

        if all_reserved {
            debug_assert_eq!(
                self.cache_config.allocate_policy,
                cache::config::AllocatePolicy::ON_MISS
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
            self.addr_translation.tag(addr),
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
        let (cache_index, probe_status) =
            self.probe_masked(addr, sector_mask, is_write, is_probe, None);

        log::trace!(
            "tag_array::fill(cache={}, tag={}, addr={}, time={}) (on fill) status={:?}",
            cache_index.map_or(-1, |i| i as i64),
            self.addr_translation.tag(addr),
            addr,
            time,
            probe_status,
        );

        if probe_status == cache::RequestStatus::RESERVATION_FAIL {
            return;
        }
        let cache_index = cache_index.unwrap();

        let line = &mut self.lines[cache_index];
        let mut was_modified_before = line.is_modified();

        if probe_status == cache::RequestStatus::MISS {
            log::trace!(
                "tag_array::allocate(cache={}, tag={}, time={})",
                cache_index,
                self.addr_translation.tag(addr),
                time,
            );
            line.allocate(
                self.addr_translation.tag(addr),
                self.addr_translation.block_addr(addr),
                sector_mask,
                time,
            );
        } else if probe_status == cache::RequestStatus::SECTOR_MISS {
            // debug_assert_eq!(self.cache_config.kind, config::CacheKind::Sector);
            // line.allocate_sector(&sector_mask, time);
            unimplemented!("sectored cache");
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
