use super::{address, cache, mem_fetch};
use crate::config;
use bitvec::{array::BitArray, BitArr};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug)]
pub struct LineCacheBlock {
    tag: u64,
    block_addr: address,

    status: cache::CacheBlockState,
    is_readable: bool,

    alloc_time: usize,
    fill_time: usize,
    last_access_time: usize,

    ignore_on_fill_status: bool,
    set_byte_mask_on_fill: bool,
    set_modified_on_fill: bool,
    set_readable_on_fill: bool,

    dirty_byte_mask: mem_fetch::MemAccessByteMask,
}

impl std::fmt::Display for LineCacheBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("LineCacheBlock")
            .field("addr", &self.block_addr)
            .field("status", &self.status)
            .finish()
    }
}

impl Default for LineCacheBlock {
    fn default() -> Self {
        Self {
            tag: 0,
            block_addr: 0,
            status: cache::CacheBlockState::INVALID,
            alloc_time: 0,
            fill_time: 0,
            last_access_time: 0,
            ignore_on_fill_status: false,
            set_byte_mask_on_fill: false,
            set_modified_on_fill: false,
            set_readable_on_fill: false,
            is_readable: true,
            dirty_byte_mask: BitArray::ZERO,
        }
    }
}

impl LineCacheBlock {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn allocate_sector(&mut self, time: usize, sector_mask: &mem_fetch::MemAccessSectorMask) {
        unimplemented!()
    }

    pub fn allocate(
        &mut self,
        tag: address,
        block_addr: address,
        time: usize,
        sector_mask: &mem_fetch::MemAccessSectorMask,
    ) {
        self.tag = tag;
        self.block_addr = block_addr;
        self.alloc_time = time;
        self.last_access_time = time;
        self.fill_time = 0;
        self.status = cache::CacheBlockState::RESERVED;
        self.ignore_on_fill_status = false;
        self.set_modified_on_fill = false;
        self.set_readable_on_fill = false;
        self.set_byte_mask_on_fill = false;
    }

    // pub fn allocate(
    //     tag: address,
    //     block_addr: address,
    //     time: usize,
    //     sector_mask: mem_fetch::MemAccessSectorMask,
    // ) -> Self {
    //     Self {
    //         tag,
    //         block_addr,
    //         alloc_time: time,
    //         last_access_time: time,
    //         fill_time: 0,
    //         status: cache::CacheBlockState::RESERVED,
    //         ..Self::default()
    //     }
    // }

    pub fn fill(
        &mut self,
        time: usize,
        sector_mask: &mem_fetch::MemAccessSectorMask,
        byte_mask: &mem_fetch::MemAccessByteMask,
    ) {
        self.status = if self.set_modified_on_fill {
            cache::CacheBlockState::MODIFIED
        } else {
            cache::CacheBlockState::VALID
        };

        if self.set_readable_on_fill {
            self.is_readable = true;
        }
        if self.set_byte_mask_on_fill {
            self.set_byte_mask(&byte_mask)
        }

        self.fill_time = time;
    }

    #[inline]
    pub fn set_last_access_time(&mut self, time: usize, _mask: &mem_fetch::MemAccessSectorMask) {
        self.last_access_time = time;
    }

    #[inline]
    pub fn set_byte_mask(&mut self, mask: &mem_fetch::MemAccessByteMask) {
        self.dirty_byte_mask |= mask;
    }

    #[inline]
    pub fn set_status(
        &mut self,
        status: cache::CacheBlockState,
        _mask: &mem_fetch::MemAccessSectorMask,
    ) {
        self.status = status;
    }

    #[inline]
    pub fn status(&self, mask: &mem_fetch::MemAccessSectorMask) -> cache::CacheBlockState {
        self.status
    }

    #[inline]
    pub fn is_valid(&self) -> bool {
        self.status == cache::CacheBlockState::VALID
    }

    #[inline]
    pub fn is_modified(&self) -> bool {
        self.status == cache::CacheBlockState::MODIFIED
    }

    #[inline]
    pub fn is_invalid(&self) -> bool {
        self.status == cache::CacheBlockState::INVALID
    }

    #[inline]
    pub fn is_reserved(&self) -> bool {
        self.status == cache::CacheBlockState::RESERVED
    }

    #[inline]
    pub fn is_readable(&self, _mask: &mem_fetch::MemAccessSectorMask) -> bool {
        self.is_readable
    }

    #[inline]
    pub fn alloc_time(&self) -> usize {
        self.alloc_time
    }

    #[inline]
    pub fn last_access_time(&self) -> usize {
        self.last_access_time
    }

    #[inline]
    pub fn modified_size(&self) -> usize {
        super::SECTOR_CHUNCK_SIZE * super::SECTOR_SIZE // cache line size
    }

    #[inline]
    pub fn dirty_sector_mask(&self) -> mem_fetch::MemAccessSectorMask {
        if self.is_modified() {
            !BitArray::ZERO
        } else {
            BitArray::ZERO
        }
    }
}

pub type LineTable = HashMap<address, usize>;

#[derive(Debug, Clone, Default, Hash, PartialEq, Eq)]
pub struct EvictedBlockInfo {
    block_addr: address,
    modified_size: usize,
    byte_mask: mem_fetch::MemAccessByteMask,
    sector_mask: mem_fetch::MemAccessSectorMask,
}

#[derive(Debug)]
pub struct TagArrayAccessStatus {
    pub index: Option<usize>,
    pub writeback: bool,
    pub evicted: Option<EvictedBlockInfo>,
    pub status: cache::CacheRequestStatus,
}

#[derive(Debug)]
pub struct TagArray<B> {
    /// nbanks x nset x assoc lines in total
    pub lines: Vec<LineCacheBlock>,
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
            .map(|_| LineCacheBlock::new())
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
        time: usize,
        fetch: &mem_fetch::MemFetch,
        // index: &mut usize,
        // writeback: &mut bool,
        // evicted: &mut EvictedBlockInfo,
        // ) -> (Option<usize>, cache::CacheRequestStatus) {
    ) -> TagArrayAccessStatus {
        // ) {
        println!("tag_array::access({})", addr);
        self.num_access += 1;
        self.is_used = true;

        let mut writeback = false;
        let mut evicted = None;

        // shader_cache_access_log(m_core_id, m_type_id, 0);
        let (index, status) = self.probe(addr, fetch, fetch.is_write(), false);
        match status {
            cache::CacheRequestStatus::HIT_RESERVED => {
                self.num_pending_hit += 1;
            }
            cache::CacheRequestStatus::HIT => {
                // TODO: use an enum like either here
                let index = index.expect("hit has idx");
                let line = &mut self.lines[index];
                line.set_last_access_time(time, fetch.access_sector_mask());
            }
            cache::CacheRequestStatus::MISS => {
                self.num_miss += 1;
                let index = index.expect("hit has idx");
                let line = &mut self.lines[index];
                // shader_cache_access_log(m_core_id, m_type_id, 1);
                if self.config.allocate_policy == config::CacheAllocatePolicy::ON_MISS {
                    if line.is_modified() {
                        writeback = true;
                        evicted = Some(EvictedBlockInfo {
                            block_addr: addr,
                            modified_size: line.modified_size(),
                            byte_mask: line.dirty_byte_mask,
                            sector_mask: line.dirty_sector_mask(),
                        });
                        self.num_dirty -= 1;
                    }
                    line.allocate(
                        self.config.tag(addr),
                        self.config.block_addr(addr),
                        time,
                        fetch.access_sector_mask(),
                    );
                }
            }
            cache::CacheRequestStatus::SECTOR_MISS => {
                debug_assert!(self.config.kind == config::CacheKind::Sector);
                self.num_sector_miss += 1;
                // shader_cache_access_log(m_core_id, m_type_id, 1);
                if self.config.allocate_policy == config::CacheAllocatePolicy::ON_MISS {
                    let index = index.expect("hit has idx");
                    let line = &mut self.lines[index];
                    let before = line.is_modified();
                    line.allocate_sector(time, fetch.access_sector_mask());
                    // ((sector_cache_block *)m_lines[idx]) ->allocate_sector(time, mf->get_access_sector_mask());
                    if before && !line.is_modified() {
                        self.num_dirty -= 1;
                    }
                }
            }
            cache::CacheRequestStatus::RESERVATION_FAIL => {
                self.num_reservation_fail += 1;
                // shader_cache_access_log(m_core_id, m_type_id, 1);
            }
            status => {
                panic!("tag_array access: unknown cache request status {status:?}");
            }
        }
        TagArrayAccessStatus {
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
    ) -> (Option<usize>, cache::CacheRequestStatus) {
        let mask = fetch.access_sector_mask();
        self.probe_masked(block_addr, &mask, is_write, is_probe, fetch)
    }

    fn probe_masked(
        &self,
        block_addr: address,
        // cache_idx: Option<usize>,
        mask: &mem_fetch::MemAccessSectorMask,
        is_write: bool,
        is_probe: bool,
        fetch: &mem_fetch::MemFetch,
    ) -> (Option<usize>, cache::CacheRequestStatus) {
        println!("tag_array::probe({block_addr})");
        let set_index = self.config.set_index(block_addr) as usize;
        let tag = self.config.tag(block_addr);
        // dbg(set_index,

        let mut invalid_line = None;
        let mut valid_line = None;
        let mut valid_timestamp = usize::MAX;

        let mut all_reserved = true;

        // check for hit or pending hit
        for way in 0..self.config.associativity {
            let idx = set_index * self.config.associativity + way;
            let line = &self.lines[idx];
            if line.tag == tag {
                match line.status(&mask) {
                    cache::CacheBlockState::RESERVED => {
                        return (Some(idx), cache::CacheRequestStatus::HIT_RESERVED);
                    }
                    cache::CacheBlockState::VALID => {
                        return (Some(idx), cache::CacheRequestStatus::HIT);
                    }
                    cache::CacheBlockState::MODIFIED => {
                        if (!is_write && line.is_readable(mask)) || is_write {
                            return (Some(idx), cache::CacheRequestStatus::HIT);
                        } else {
                            return (Some(idx), cache::CacheRequestStatus::SECTOR_MISS);
                        }
                    }
                    cache::CacheBlockState::INVALID if line.is_valid() => {
                        return (Some(idx), cache::CacheRequestStatus::SECTOR_MISS);
                    }
                    cache::CacheBlockState::INVALID => {}
                }
            }
            if !line.is_reserved() {
                // percentage of dirty lines in the cache
                // number of dirty lines / total lines in the cache
                let dirty_line_percent =
                    (self.num_dirty as f64 / self.config.total_lines() as f64) * 100f64;
                // If the cacheline is from a load op (not modified),
                // or the total dirty cacheline is above a specific value,
                // Then this cacheline is eligible to be considered for replacement candidate
                // i.e. Only evict clean cachelines until total dirty cachelines reach the limit.
                if !line.is_modified() {
                    // || dirty_line_percent >= self.config.wr_percent {
                    all_reserved = false;
                    if line.is_invalid() {
                        invalid_line = Some(idx);
                    } else {
                        // valid line: keep track of most appropriate replacement candidate
                        if self.config.replacement_policy == config::CacheReplacementPolicy::LRU {
                            if line.last_access_time() < valid_timestamp {
                                valid_timestamp = line.last_access_time();
                                valid_line = Some(idx);
                            }
                        } else if self.config.replacement_policy
                            == config::CacheReplacementPolicy::FIFO
                        {
                            if line.alloc_time() < valid_timestamp {
                                valid_timestamp = line.alloc_time();
                                valid_line = Some(idx);
                            }
                        }
                    }
                }
            }
        }

        if all_reserved {
            debug_assert_eq!(
                self.config.allocate_policy,
                config::CacheAllocatePolicy::ON_MISS
            );
            // miss and not enough space in cache to allocate on miss
            return (None, cache::CacheRequestStatus::RESERVATION_FAIL);
        }

        let cache_idx = if invalid_line.is_some() {
            invalid_line
        } else if valid_line.is_some() {
            valid_line
        } else {
            // if an unreserved block exists,
            // it is either invalid or replaceable
            panic!("found neither a valid nor invalid cache line");
        };

        (cache_idx, cache::CacheRequestStatus::MISS)
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

    pub fn get_block_mut(&mut self, idx: usize) -> &mut LineCacheBlock {
        &mut self.lines[idx]
    }

    pub fn get_block(&self, idx: usize) -> &LineCacheBlock {
        &self.lines[idx]
    }

    pub fn add_pending_line(&mut self, fetch: &mem_fetch::MemFetch) {
        println!("tag_array::add_pending_line({})", fetch.addr());
        let addr = self.config.block_addr(fetch.addr());
        if self.pending_lines.contains_key(&addr) {
            self.pending_lines.insert(addr, fetch.instr.uid);
        }
    }

    pub fn remove_pending_line(&mut self, fetch: &mem_fetch::MemFetch) {
        println!("tag_array::remove_pending_line({})", fetch.addr());
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

    struct Interconnect {}

    impl Interconnect {}

    #[test]
    fn test_tag_array() {
        let config = GPUConfig::default().data_cache_l1.unwrap();
        let tag_array: TagArray<usize> = TagArray::new(0, 0, config);
        assert!(false);
    }
}
