#![allow(warnings)]

use anyhow::Result;

pub enum StaticCacheRequestStatus {
    HIT,
    HIT_RESERVED,
    MISS,
    RESERVATION_FAIL,
    SECTOR_MISS,
}

pub type address = u64;

pub trait SetIndexFunction {
    /// compute set index
    fn compute_set_index(
        addr: address,
        num_sets: u64,
        line_size_log2: u64,
        num_sets_log2: u64,
    ) -> Result<usize>;
}

// see src/gpgpu-sim/gpu-cache.cc
pub struct FermiSetIndexFunction {}
pub struct BitwiseXORSetIndexFunction {}
pub struct IpolyHashSetIndexFunction {}
pub struct LinearHashSetIndexFunction {}

impl SetIndexFunction for BitwiseXORSetIndexFunction {
    fn compute_set_index(
        addr: address,
        num_sets: u64,
        line_size_log2: u64,
        num_sets_log2: u64,
    ) -> Result<usize> {
        let higher_bits = addr >> (line_size_log2 + num_sets_log2);
        let index = (addr >> line_size_log2) & (num_sets - 1);
        // let set_index = bitwise_hash_function(higher_bits, index, m_nset);
        todo!();
        let set_index = 0;
        Ok(set_index)
    }
}

impl SetIndexFunction for FermiSetIndexFunction {
    /// Set Indexing function from "A Detailed GPU Cache Model Based on Reuse
    /// Distance Theory" Cedric Nugteren et al. HPCA 2014
    fn compute_set_index(
        addr: address,
        num_sets: u64,
        line_size_log2: u64,
        num_sets_log2: u64,
    ) -> Result<usize> {
        if num_sets != 32 && num_sets != 64 {
            panic!("cache config error: number of sets should be 32 or 64");
        }
        let set_index = 0;
        let lower_xor = 0;
        let upper_xor = 0;
        todo!();

        //   // Lower xor value is bits 7-11
        //   lower_xor = (addr >> m_line_sz_log2) & 0x1F;
        //
        //   // Upper xor value is bits 13, 14, 15, 17, and 19
        //   upper_xor = (addr & 0xE000) >> 13;    // Bits 13, 14, 15
        //   upper_xor |= (addr & 0x20000) >> 14;  // Bit 17
        //   upper_xor |= (addr & 0x80000) >> 15;  // Bit 19
        //
        //   set_index = (lower_xor ^ upper_xor);
        //
        //   // 48KB cache prepends the set_index with bit 12
        //   if (m_nset == 64) set_index |= (addr & 0x1000) >> 7;
        //
        Ok(set_index)
    }
}

// todo: replace
pub struct GenericCacheConfig {
    pub max_num_lines: usize,
}

// 2d matrix?
pub struct LineCacheBlock {}
pub struct SectorCacheBlock {}

pub struct TagArray<B> {
    // pub config: GenericCacheConfig,
    pub lines: Vec<B>,
    access: usize,
    miss: usize,
    pending_hit: usize,
    res_fail: usize,
    sector_miss: usize,
    // initialize snapshot counters for visualizer
    // prev_snapshot_access = 0;
    // prev_snapshot_miss = 0;
    // prev_snapshot_pending_hit = 0;
    core_id: usize,
    type_id: usize,
    is_used: bool,
}

impl<B> TagArray<B> {
    pub fn new(
        core_id: usize,
        type_id: usize,
    ) -> Self {
        Self {
            // config,
            lines: Vec::new(),
    access: 0,
    miss: 0,
    pending_hit: 0,
    res_fail: 0,
    sector_miss: 0,
    // initialize snapshot counters for visualizer
    // prev_snapshot_access = 0;
    // prev_snapshot_miss = 0;
    // prev_snapshot_pending_hit = 0;
    core_id,
    type_id,
    is_used: false,
        }
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

pub struct L1DCacheConfig {}
pub struct L2CacheConfig {}

impl L2CacheConfig {
    pub fn new() -> Self {
        Self {}
    }

    pub fn set_index(addr: address) {
        //   new_addr_type part_addr = addr;
        //
        //   if (m_address_mapping) {
        //       // Calculate set index without memory partition bits to reduce set camping
        //       part_addr = m_address_mapping->partition_address(addr);
        //   }
        //
        // return cache_config::set_index(part_addr);
    }
}

impl L1DCacheConfig {
    pub fn new() -> Self {
        Self {}
    }

    pub fn set_bank(addr: address) {}
    // unsigned set_index = 0;
}
