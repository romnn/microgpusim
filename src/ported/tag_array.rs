use super::{address, cache, cache_block, mem_fetch};
use crate::config;
use bitvec::{array::BitArray, BitArr};
use std::collections::HashMap;
use std::sync::Arc;

pub type LineTable = HashMap<address, usize>;

#[derive(Debug, Clone, Default, Hash, PartialEq, Eq)]
pub struct EvictedBlockInfo {
    pub block_addr: address,
    pub allocation: Option<super::Allocation>,
    pub modified_size: u32,
    pub byte_mask: mem_fetch::MemAccessByteMask,
    pub sector_mask: mem_fetch::MemAccessSectorMask,
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
    pub lines: Vec<cache_block::LineCacheBlock>,
    phantom: std::marker::PhantomData<B>,
    access: usize,
    miss: usize,
    pending_hit: usize,
    res_fail: usize,
    sector_miss: usize,
    core_id: usize,
    type_id: usize,
    is_used: bool,
    num_access: usize,
    num_miss: usize,
    num_pending_hit: usize,
    num_reservation_fail: usize,
    num_sector_miss: usize,
    pub num_dirty: usize,
    config: Arc<config::CacheConfig>,
    pending_lines: LineTable,
}

impl<B> TagArray<B> {
    #[must_use]
    pub fn new(core_id: usize, type_id: usize, config: Arc<config::CacheConfig>) -> Self {
        let num_cache_lines = config.max_num_lines();
        let lines = (0..num_cache_lines)
            .map(|_| cache_block::LineCacheBlock::new())
            .collect();
        // if (config.m_cache_type == NORMAL) {
        //   for (unsigned i = 0; i < cache_lines_num; ++i)
        //     m_lines[i] = new line_cache_block();
        // } else if (config.m_cache_type == SECTOR) {
        //   for (unsigned i = 0; i < cache_lines_num; ++i)
        //     m_lines[i] = new sector_cache_block();
        // } else
        //   assert(0);
        //
        // init(core_id, type_id);

        Self {
            lines,
            phantom: std::marker::PhantomData,
            access: 0,
            miss: 0,
            pending_hit: 0,
            res_fail: 0,
            sector_miss: 0,
            core_id,
            type_id,
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

    /// Accesses the tag array
    ///
    /// # Returns
    /// Index, writeback, evicted
    pub fn access(
        &mut self,
        addr: address,
        fetch: &mem_fetch::MemFetch,
        time: u64,
        // index: &mut usize,
        // writeback: &mut bool,
        // evicted: &mut EvictedBlockInfo,
        // ) -> (Option<usize>, cache::RequestStatus) {
    ) -> AccessStatus {
        log::trace!("tag_array::access({}, time={})", fetch, time);
        self.num_access += 1;
        self.is_used = true;

        let mut writeback = false;
        let mut evicted = None;

        // shader_cache_access_log(m_core_id, m_type_id, 0);
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
                if self.config.allocate_policy == config::CacheAllocatePolicy::ON_MISS {
                    if line.is_modified() {
                        writeback = true;
                        evicted = Some(EvictedBlockInfo {
                            block_addr: addr,
                            allocation: fetch.access.allocation.clone(),
                            modified_size: line.modified_size(),
                            byte_mask: line.dirty_byte_mask(),
                            sector_mask: line.dirty_sector_mask(),
                        });
                        self.num_dirty -= 1;
                    }
                    log::trace!(
                        "tag_array::allocate(cache={}, tag={})",
                        index,
                        self.config.tag(addr)
                    );
                    line.allocate(
                        self.config.tag(addr),
                        self.config.block_addr(addr),
                        time,
                        fetch.access_sector_mask(),
                    );
                }
            }
            cache::RequestStatus::SECTOR_MISS => {
                unimplemented!("no sector miss");
                debug_assert!(self.config.kind == config::CacheKind::Sector);
                self.num_sector_miss += 1;
                // shader_cache_access_log(m_core_id, m_type_id, 1);
                if self.config.allocate_policy == config::CacheAllocatePolicy::ON_MISS {
                    let index = index.expect("hit has idx");
                    let line = &mut self.lines[index];
                    let was_modified_before = line.is_modified();
                    line.allocate_sector(fetch.access_sector_mask(), time);
                    if was_modified_before && !line.is_modified() {
                        self.num_dirty -= 1;
                    }
                }
            }
            cache::RequestStatus::RESERVATION_FAIL => {
                self.num_reservation_fail += 1;
            }
            status => {
                panic!("tag_array access: unknown cache request status {status:?}");
            }
        }
        AccessStatus {
            index,
            writeback,
            evicted,
            status,
        }
    }

    /// Probes the tag array
    ///
    /// # Returns
    /// A tuple with the cache index `Option<usize>` and cache request status.
    pub fn probe(
        &self,
        block_addr: address,
        // cache_idx: Option<usize>,
        fetch: &mem_fetch::MemFetch,
        is_write: bool,
        is_probe: bool,
    ) -> (Option<usize>, cache::RequestStatus) {
        self.probe_masked(
            block_addr,
            fetch.access_sector_mask(),
            is_write,
            is_probe,
            fetch,
        )
    }

    fn probe_masked(
        &self,
        block_addr: address,
        // cache_idx: Option<usize>,
        mask: &mem_fetch::MemAccessSectorMask,
        is_write: bool,
        is_probe: bool,
        fetch: &mem_fetch::MemFetch,
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
            "tag_array::probe({}) set_idx = {}, tag = {}, assoc = {} dirty lines = {}%",
            fetch,
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
                "tag_array::probe({}) => checking cache index {} (tag={}, status={:?}, last_access={})",
                fetch,
                idx,
                line.tag,
                line.status(&mask),
                line.last_access_time()
            );
            if line.tag == tag {
                match line.status(&mask) {
                    cache_block::Status::RESERVED => {
                        return (Some(idx), cache::RequestStatus::HIT_RESERVED);
                    }
                    cache_block::Status::VALID => {
                        return (Some(idx), cache::RequestStatus::HIT);
                    }
                    cache_block::Status::MODIFIED => {
                        if (!is_write && line.is_readable(mask)) || is_write {
                            return (Some(idx), cache::RequestStatus::HIT);
                        } else {
                            return (Some(idx), cache::RequestStatus::SECTOR_MISS);
                        }
                    }
                    cache_block::Status::INVALID if line.is_valid() => {
                        return (Some(idx), cache::RequestStatus::SECTOR_MISS);
                    }
                    cache_block::Status::INVALID => {}
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
                        {
                            if line.alloc_time() < valid_time {
                                valid_time = line.alloc_time();
                                valid_line = Some(idx);
                            }
                        }
                    }
                }
            }
        }

        log::trace!("tag_array::probe({}) => all reserved={} invalid_line={:?} valid_line={:?} ({:?} policy)", fetch, all_reserved, invalid_line, valid_line, self.config.replacement_policy);

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
        // let cache_idx = if invalid_line.is_some() {
        //     invalid_line
        // } else if valid_line.is_some() {
        //     valid_line
        // } else {
        //     // if an unreserved block exists,
        //     // it is either invalid or replaceable
        //     panic!("found neither a valid nor invalid cache line");
        // };

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
        self.lines[cache_index].fill(time, fetch.access_sector_mask(), fetch.access_byte_mask());
        if self.lines[cache_index].is_modified() && !was_modified_before {
            self.num_dirty += 1;
        }
    }

    pub fn fill_on_fill(&mut self, addr: address, fetch: &mem_fetch::MemFetch, time: u64) {
        debug_assert!(self.config.allocate_policy == config::CacheAllocatePolicy::ON_FILL);

        // probe tag array
        let is_probe = false;
        let (cache_index, probe_status) = self.probe(addr, fetch, fetch.is_write(), is_probe);

        log::trace!(
            "tag_array::fill(cache={}, tag={}, addr={}) (on fill) status={:?}",
            cache_index.map(|i| i as i64).unwrap_or(-1),
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
                "tag_array::allocate(cache={}, tag={})",
                cache_index,
                self.config.tag(addr)
            );
            line.allocate(
                self.config.tag(addr),
                self.config.block_addr(addr),
                time,
                fetch.access_sector_mask(),
            );
        } else if probe_status == cache::RequestStatus::SECTOR_MISS {
            debug_assert_eq!(self.config.kind, config::CacheKind::Sector);
            line.allocate_sector(fetch.access_sector_mask(), time);
        }
        if was_modified_before && !line.is_modified() {
            self.num_dirty -= 1;
        }
        was_modified_before = line.is_modified();
        line.fill(time, fetch.access_sector_mask(), fetch.access_byte_mask());
        if line.is_modified() && !was_modified_before {
            self.num_dirty += 1;
        }
    }

    pub fn flush(&mut self) {
        todo!("flush tag array");
    }

    pub fn invalidate(&mut self) {
        todo!("invalidate tag array");
    }

    pub fn size(&self) -> usize {
        self.config.max_num_lines()
    }

    pub fn get_block_mut(&mut self, idx: usize) -> &mut cache_block::LineCacheBlock {
        &mut self.lines[idx]
    }

    pub fn get_block(&self, idx: usize) -> &cache_block::LineCacheBlock {
        &self.lines[idx]
    }

    pub fn add_pending_line(&mut self, fetch: &mem_fetch::MemFetch) {
        // log::debug!("tag_array::add_pending_line({})", fetch.addr());
        let addr = self.config.block_addr(fetch.addr());
        let instr = fetch.instr.as_ref().unwrap();
        if self.pending_lines.contains_key(&addr) {
            self.pending_lines.insert(addr, instr.uid);
        }
    }

    pub fn remove_pending_line(&mut self, fetch: &mem_fetch::MemFetch) {
        // log::debug!("tag_array::remove_pending_line({})", fetch.addr());
        let addr = self.config.block_addr(fetch.addr());
        self.pending_lines.remove(&addr);
    }

    // pub fn from_block(
    //     config: GenericCacheConfig,
    //     core_id: usize,
    //     type_id: usize,
    //     block: CacheBlock,
    // ) -> Self {
    //     Self {
    //         // config,
    //         lines: Vec::new(),
    //     }
    // }

    // pub fn from_config(config: GenericCacheConfig, core_id: usize, type_id: usize) -> Self {
    //     config.max_lines;
    //     let lines =
    //     Self {
    //         // config,
    //         lines: Vec::new(),
    //     }
    //     // unsigned cache_lines_num = config.get_max_num_lines();
    //     //   m_lines = new cache_block_t *[cache_lines_num];
    //     //   if (config.m_cache_type == NORMAL) {
    //     //     for (unsigned i = 0; i < cache_lines_num; ++i)
    //     //       m_lines[i] = new line_cache_block();
    //     //   } else if (config.m_cache_type == SECTOR) {
    //     //     for (unsigned i = 0; i < cache_lines_num; ++i)
    //     //       m_lines[i] = new sector_cache_block();
    //     //   } else
    //     //     assert(0);
    // }
    // todo: update config (GenericCacheConfig)
}

#[cfg(test)]
mod tests {
    use super::TagArray;
    use crate::config::GPUConfig;
    use std::sync::Arc;

    #[ignore = "todo"]
    #[test]
    fn test_tag_array() {
        let config = GPUConfig::default().data_cache_l1.unwrap();
        let tag_array: TagArray<usize> = TagArray::new(0, 0, Arc::clone(&config.inner));
        assert!(false);
    }
}
