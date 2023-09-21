mem_fetch *n_mf = m_mf_allocator->alloc(
mf->get_addr() + SECTOR_SIZE * i, // diff
mf->get_access_type(),
mf->get_access_warp_mask(),
mf->get_access_byte_mask() & mask,
std::bitset<SECTOR_CHUNCK_SIZE>().set(i),
SECTOR_SIZE,
mf->is_write(),
m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle,
mf->get_wid(),
mf->get_sid(), mf->get_tpc(), mf);

mem_fetch *n_mf = m_mf_allocator->alloc(
mf->get_addr(), // diff
mf->get_access_type(),
mf->get_access_warp_mask(),
mf->get_access_byte_mask() & mask,
std::bitset<SECTOR_CHUNCK_SIZE>().set(i),
SECTOR_SIZE,
mf->is_write(),
m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle,
mf->get_wid(),
mf->get_sid(), mf->get_tpc(), mf);

mem_fetch *n_mf = m_mf_allocator->alloc(
mf->get_addr() + SECTOR_SIZE * i, // diff
mf->get_access_type(),
mf->get_access_warp_mask(),
mf->get_access_byte_mask() & mask,
std::bitset<SECTOR_CHUNCK_SIZE>().set(i),
SECTOR_SIZE,
mf->is_write(),
m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle,
mf->get_wid(), mf->get_sid(), mf->get_tpc(), mf);


// #[must_use]
// pub fn full(&self, _size: u32) -> bool {
//     self.interconn_to_l2_queue.full()
// }
//
// #[must_use]
// pub fn interconn_to_l2_can_fit(&self, size: usize) -> bool {
//     self.interconn_to_l2_queue.can_fit(size)
// }

//// pub fn dram_l2_queue_push(&mut self, _fetch: &mem_fetch::MemFetch) {
//     todo!("mem sub partition: dram l2 queue push");
// }
//
// #[must_use]
// pub fn dram_l2_queue_full(&self) -> bool {
//     todo!("mem sub partition: dram l2 queue full");
// }
