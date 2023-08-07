#![allow(warnings)]
#![allow(clippy::all, clippy::pedantic)]

use anyhow::Result;

// todo: replace
pub struct GenericCacheConfig {
    pub max_num_lines: usize,
}

// 2d matrix?
pub struct LineCacheBlock {}
pub struct SectorCacheBlock {}

// pub struct L1DCacheConfig {}
// pub struct L2CacheConfig {}
//
// impl L2CacheConfig {
//     #[must_use] pub fn new() -> Self {
//         Self {}
//     }
//
//     pub fn set_index(addr: address) {
//         //   new_addr_type part_addr = addr;
//         //
//         //   if (m_address_mapping) {
//         //       // Calculate set index without memory partition bits to reduce set camping
//         //       part_addr = m_address_mapping->partition_address(addr);
//         //   }
//         //
//         // return cache_config::set_index(part_addr);
//     }
// }
//
// impl L1DCacheConfig {
//     #[must_use] pub fn new() -> Self {
//         Self {}
//     }
//
//     pub fn set_bank(addr: super::address) {}
//     // unsigned set_index = 0;
// }
