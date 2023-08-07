fn write_miss_write_allocate_fetch_on_write(
    &mut self,
    addr: address,
    cache_index: Option<usize>,
    fetch: mem_fetch::MemFetch,
    time: u64,
    events: &mut Vec<cache::Event>,
    probe_status: cache::RequestStatus,
) -> cache::RequestStatus {
    // let super::base::Base { ref cache_config, ref mut tag_array, .. } = self.inner;
    todo!("write_miss_write_allocate_fetch_on_write");
    let super::base::Base {
        ref cache_config, ..
    } = self.inner;
    let block_addr = cache_config.block_addr(addr);
    let mshr_addr = cache_config.mshr_addr(fetch.addr());

    if fetch.access_byte_mask().count_ones() == cache_config.atom_size() as usize {
        // if the request writes to the whole cache line/sector,
        // then write and set cache line modified.
        //
        // no need to send read request to memory or reserve mshr
        if self.inner.miss_queue_full() {
            let stats = self.inner.stats.lock().unwrap();
            stats.inc(
                *fetch.access_kind(),
                cache::AccessStat::ReservationFailure(cache::ReservationFailure::MISS_QUEUE_FULL),
                1,
            );
            // cannot handle request this cycle
            return cache::RequestStatus::RESERVATION_FAIL;
        }

        // bool wb = false;
        // evicted_block_info evicted;
        let tag_array::AccessStatus {
            status,
            index,
            writeback,
            evicted,
            ..
        } = self.inner.tag_array.access(block_addr, &fetch, time);
        // , cache_index);
        // , wb, evicted, mf);
        debug_assert_ne!(status, cache::RequestStatus::HIT);
        let block = self.inner.tag_array.get_block_mut(index.unwrap());
        let was_modified_before = block.is_modified();
        block.set_status(cache_block::Status::MODIFIED, fetch.access_sector_mask());
        block.set_byte_mask(fetch.access_byte_mask());
        if status == cache::RequestStatus::HIT_RESERVED {
            block.set_ignore_on_fill(true, fetch.access_sector_mask());
        }
        if !was_modified_before {
            self.inner.tag_array.num_dirty += 1;
            // self.tag_array.inc_dirty();
        }

        if (status != cache::RequestStatus::RESERVATION_FAIL) {
            // If evicted block is modified and not a write-through
            // (already modified lower level)

            if writeback && cache_config.write_policy != config::CacheWritePolicy::WRITE_THROUGH {
                // let writeback_fetch = mem_fetch::MemFetch::new(
                //     fetch.instr,
                //     access,
                //     &*self.config,
                //     if wr {
                //         super::WRITE_PACKET_SIZE
                //     } else {
                //         super::READ_PACKET_SIZE
                //     }
                //     .into(),
                //     0,
                //     0,
                //     0,
                // );

                //     evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
                //     evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
                //     true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
                // NULL);

                // the evicted block may have wrong chip id when
                // advanced L2 hashing  is used,
                // so set the right chip address from the original mf
                // writeback_fetch.set_chip(mf->get_tlx_addr().chip);
                // writeback_fetch.set_parition(mf->get_tlx_addr().sub_partition);
                // self.send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                // time, events);
            }
            todo!("write_miss_write_allocate_fetch_on_write");
            return cache::RequestStatus::MISS;
        }
        return cache::RequestStatus::RESERVATION_FAIL;
    } else {
        todo!("write_miss_write_allocate_fetch_on_write");
        return cache::RequestStatus::RESERVATION_FAIL;
    }
}

fn write_miss_write_allocate_lazy_fetch_on_read(
    &mut self,
    addr: address,
    cache_index: Option<usize>,
    fetch: mem_fetch::MemFetch,
    time: u64,
    events: &mut Vec<cache::Event>,
    probe_status: cache::RequestStatus,
) -> cache::RequestStatus {
    todo!("write_miss_write_allocate_lazy_fetch_on_read");
    cache::RequestStatus::MISS
}
