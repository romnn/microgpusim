use super::{cache, mem_fetch};
use std::collections::HashMap;
use std::sync::Mutex;

pub type CacheRequestStatusCounters =
    HashMap<(mem_fetch::AccessKind, cache::CacheRequestStatus), usize>;

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

lazy_static::lazy_static! {
    pub static ref STATS: Mutex<Stats> = Mutex::new(Stats::default());
}
