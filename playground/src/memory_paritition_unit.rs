use playground_sys::memory_partition_unit::memory_partition_unit_bridge;

#[derive(Clone)]
pub struct MemoryPartitionUnit<'a>(pub(crate) &'a memory_partition_unit_bridge);

impl<'a> MemoryPartitionUnit<'a> {
    #[must_use]
    pub fn dram_latency_queue(&self) -> Vec<super::mem_fetch::MemFetch<'a>> {
        super::mem_fetch::get_mem_fetches(&self.0.get_dram_latency_queue())
    }
}
