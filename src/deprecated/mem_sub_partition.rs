fn breakdown_request_to_sector_requests(
    &self,
    _fetch: &mem_fetch::MemFetch,
) -> Vec<mem_fetch::MemFetch> {
    todo!("breakdown request to sector");

    // let mut result = Vec::new();
    // let sector_mask = fetch.access_sector_mask().clone();
    // if fetch.data_size == SECTOR_SIZE && fetch.access_sector_mask().count_ones() == 1 {
    //     result.push(fetch);
    //     return result;
    // }
    //
    // // create new fetch requests
    // let control_size = if fetch.is_write() {
    //     super::WRITE_PACKET_SIZE
    // } else {
    //     super::READ_PACKET_SIZE
    // } as u32;
    // // let mut sector_mask = mem_fetch::MemAccessSectorMask::ZERO;
    // // sector_mask.set(i, true);
    // let old_access = fetch.access.clone();
    // let new_fetch = mem_fetch::MemFetch {
    //     control_size,
    //     instr: None,
    //     access: mem_fetch::MemAccess {
    //         // addr: fetch.addr() + SECTOR_SIZE * i,
    //         // byte_mask: fetch.access_byte_mask() & byte_mask,
    //         sector_mask: mem_fetch::MemAccessSectorMask::ZERO,
    //         req_size_bytes: SECTOR_SIZE,
    //         ..fetch.access
    //     },
    //     // ..fetch.clone()
    //     // consume fetch
    //     ..fetch.clone()
    // };
    //
    // if fetch.data_size == MAX_MEMORY_ACCESS_SIZE {
    //     // break down every sector
    //     let mut byte_mask = mem_fetch::MemAccessByteMask::ZERO;
    //     for i in 0..SECTOR_CHUNCK_SIZE {
    //         for k in (i * SECTOR_SIZE)..((i + 1) * SECTOR_SIZE) {
    //             byte_mask.set(k as usize, true);
    //         }
    //
    //         let mut new_fetch = new_fetch.clone();
    //         // new_fetch.access.addr = fetch.addr() + SECTOR_SIZE as u64 * i as u64;
    //         new_fetch.access.addr += SECTOR_SIZE as u64 * i as u64;
    //         // new_fetch.access.byte_mask = *fetch.access_byte_mask() & byte_mask;
    //         new_fetch.access.byte_mask &= byte_mask;
    //         new_fetch.access.sector_mask.set(i as usize, true);
    //         // mf->get_addr() + SECTOR_SIZE * i, mf->get_access_type(),
    //         // mf->get_access_warp_mask(), mf->get_access_byte_mask() & mask,
    //         // std::bitset<SECTOR_CHUNCK_SIZE>().set(i), SECTOR_SIZE, mf->is_write(),
    //         // m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, mf->get_wid(),
    //         // mf->get_sid(), mf->get_tpc(), mf);
    //
    //         result.push(new_fetch);
    //     }
    //     // This is for constant cache
    // } else if fetch.data_size == 64
    //     && (fetch.access_sector_mask().all() || fetch.access_sector_mask().not_any())
    // {
    //     let start = if fetch.addr() % MAX_MEMORY_ACCESS_SIZE as u64 == 0 {
    //         0
    //     } else {
    //         2
    //     };
    //     let mut byte_mask = mem_fetch::MemAccessByteMask::ZERO;
    //     for i in start..(start + 2) {
    //         for k in i * SECTOR_SIZE..((i + 1) * SECTOR_SIZE) {
    //             byte_mask.set(k as usize, true);
    //         }
    //         let mut new_fetch = new_fetch.clone();
    //         // address is the same
    //         // new_fetch.access.byte_mask = *fetch.access_byte_mask() & byte_mask;
    //         new_fetch.access.byte_mask &= byte_mask;
    //         new_fetch.access.sector_mask.set(i as usize, true);
    //
    //         // mf->get_addr(), mf->get_access_type(), mf->get_access_warp_mask(),
    //         // mf->get_access_byte_mask() & mask,
    //         // std::bitset<SECTOR_CHUNCK_SIZE>().set(i), SECTOR_SIZE, mf->is_write(),
    //         // m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, mf->get_wid(),
    //         // mf->get_sid(), mf->get_tpc(), mf);
    //
    //         result.push(new_fetch);
    //     }
    // } else {
    //     for i in 0..SECTOR_CHUNCK_SIZE {
    //         if sector_mask[i as usize] {
    //             let mut byte_mask = mem_fetch::MemAccessByteMask::ZERO;
    //
    //             for k in (i * SECTOR_SIZE)..((i + 1) * SECTOR_SIZE) {
    //                 byte_mask.set(k as usize, true);
    //             }
    //             let mut new_fetch = new_fetch.clone();
    //             // new_fetch.access.addr = fetch.addr() + SECTOR_SIZE as u64 * i as u64;
    //             new_fetch.access.addr += SECTOR_SIZE as u64 * i as u64;
    //             // new_fetch.access.byte_mask = *fetch.access_byte_mask() & byte_mask;
    //             new_fetch.access.byte_mask &= byte_mask;
    //             new_fetch.access.sector_mask.set(i as usize, true);
    //             // different addr
    //             // mf->get_addr() + SECTOR_SIZE * i, mf->get_access_type(),
    //             // mf->get_access_warp_mask(), mf->get_access_byte_mask() & mask,
    //             // std::bitset<SECTOR_CHUNCK_SIZE>().set(i), SECTOR_SIZE,
    //             // mf->is_write(), m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle,
    //             // mf->get_wid(), mf->get_sid(), mf->get_tpc(), mf);
    //
    //             result.push(new_fetch);
    //         }
    //     }
    // }
    // debug_assert!(!result.is_empty(), "no fetch sent");
    // result
}

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
