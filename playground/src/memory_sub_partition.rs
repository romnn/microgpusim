use super::mem_fetch::{get_mem_fetches, MemFetch};
use playground_sys::memory_partition_unit::memory_sub_partition_bridge;

#[derive(Clone)]
pub struct MemorySubPartition<'a>(pub(crate) &'a memory_sub_partition_bridge);

impl<'a> MemorySubPartition<'a> {
    #[must_use]
    pub fn interconn_to_l2_queue(&self) -> Vec<MemFetch<'a>> {
        get_mem_fetches(&self.0.get_icnt_L2_queue())
    }

    #[must_use]
    pub fn l2_to_interconn_queue(&self) -> Vec<MemFetch<'a>> {
        get_mem_fetches(&self.0.get_L2_icnt_queue())
    }

    #[must_use]
    pub fn dram_to_l2_queue(&self) -> Vec<MemFetch<'a>> {
        get_mem_fetches(&self.0.get_dram_L2_queue())
    }

    #[must_use]
    pub fn l2_to_dram_queue(&self) -> Vec<MemFetch<'a>> {
        get_mem_fetches(&self.0.get_L2_dram_queue())
    }

    #[must_use]
    pub fn l2_cache(&self) -> super::cache::L2<'a> {
        super::cache::L2::new(self.0.get_l2_cache())
    }
}
