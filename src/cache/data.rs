use crate::sync::Arc;
use crate::DEBUG_PRINT;
use crate::{
    address, cache, config, interconn as ic, mem_fetch,
    mem_sub_partition::{NUM_SECTORS, SECTOR_SIZE},
    mshr::MSHR,
    tag_array,
};
use cache::block::Block;
use color_eyre::eyre;
use mem_fetch::access::Kind as AccessKind;
use std::collections::VecDeque;
use tag_array::Access;

#[derive(Clone)]
pub struct Builder<MC, CC, S> {
    // pub struct Builder<CC, S> {
    pub name: String,
    /// SM ID or subpartition ID depending on cache type
    pub id: usize,
    pub kind: cache::base::Kind,
    // pub stats: Arc<Mutex<S>>,
    pub stats: S,
    // pub mem_controller: Arc<dyn mcu::MemoryController>,
    pub mem_controller: Arc<MC>,
    pub cache_controller: CC,
    pub config: Arc<config::GPU>,
    pub cache_config: Arc<config::Cache>,
    pub write_alloc_type: AccessKind,
    pub write_back_type: AccessKind,
}

/// First level data cache in Fermi.
///
/// The cache uses a write-evict (global) or write-back (local) policy
/// at the granularity of individual blocks.
/// (the policy used in fermi according to the CUDA manual)
pub struct Data<MC, CC, S> {
    // pub struct Data<CC, S> {
    pub inner: cache::base::Base<cache::block::sector::Block<NUM_SECTORS>, CC, S>,

    /// Memory controller
    // pub mem_controller: Arc<dyn mcu::MemoryController>,
    pub mem_controller: Arc<MC>,
    /// Specifies type of write allocate request (e.g., L1 or L2)
    write_alloc_type: AccessKind,
    /// Specifies type of writeback request (e.g., L1 or L2)
    write_back_type: AccessKind,
}

impl<MC, CC, S> Builder<MC, CC, S>
where
    CC: Clone,
{
    pub fn build(self) -> Data<MC, CC, S> {
        let inner = super::base::Builder {
            name: self.name,
            id: self.id,
            kind: self.kind,
            stats: self.stats,
            cache_controller: self.cache_controller,
            cache_config: self.cache_config,
            accelsim_compat: self.config.accelsim_compat,
        }
        .build();
        Data {
            inner,
            mem_controller: self.mem_controller,
            write_alloc_type: self.write_alloc_type,
            write_back_type: self.write_back_type,
        }
    }
}

// kind: cache::data::Kind::OffChip,

impl<MC, CC, S> Data<MC, CC, S> {
    // impl<CC, S> Data<CC, S> {
    // #[inline]
    pub fn set_top_port(&mut self, port: ic::Port<mem_fetch::MemFetch>) {
        self.inner.set_top_port(port);
    }
}

impl<MC, CC> Data<MC, CC, stats::cache::PerKernel>
// impl<CC> Data<CC, stats::cache::PerKernel>
where
    MC: crate::mcu::MemoryController,
    CC: cache::CacheController,
{
    #[allow(clippy::needless_pass_by_value)]
    fn write_miss_write_allocate_naive(
        &mut self,
        addr: address,
        cache_index: Option<usize>,
        fetch: mem_fetch::MemFetch,
        time: u64,
        events: &mut Vec<cache::Event>,
        probe_status: cache::RequestStatus,
    ) -> cache::RequestStatus {
        debug_assert_eq!(addr, fetch.addr());

        let block_addr = self.inner.cache_controller.block_addr(addr);
        let mshr_addr = self.inner.cache_controller.mshr_addr(fetch.addr());

        // Write allocate, maximum 3 requests:
        //  (write miss, read request, write back request)
        //
        //  Conservatively ensure the worst-case request can be handled this cycle
        let mshr_hit = self.inner.mshrs.get(mshr_addr).is_some();
        let mshr_free = !self.inner.mshrs.full(mshr_addr);
        let mshr_full = !self.inner.miss_queue_can_fit(2);
        let mshr_miss_but_free = !mshr_hit && mshr_free && !self.inner.miss_queue_full();

        log::debug!("handling write miss for {} (block addr={}, mshr addr={}, mshr hit={} mshr avail={}, miss queue full={})", &fetch, block_addr, mshr_addr, mshr_hit, mshr_free, self.inner.miss_queue_can_fit(2));

        if mshr_full || !(mshr_miss_but_free || mshr_hit && mshr_free) {
            // check what is the exact failure reason
            let failure = if mshr_full {
                cache::ReservationFailure::MISS_QUEUE_FULL
            } else if mshr_hit && !mshr_free {
                cache::ReservationFailure::MSHR_MERGE_ENTRY_FAIL
            } else if !mshr_hit && !mshr_free {
                cache::ReservationFailure::MSHR_ENTRY_FAIL
            } else {
                panic!("write_miss_write_allocate_naive bad reason");
            };
            // let mut stats = self.inner.stats.lock();
            let num_accesses = self.inner.num_accesses_stat(&fetch);
            let kernel_stats = self.inner.stats.get_mut(fetch.kernel_launch_id());
            kernel_stats.inc(
                fetch.allocation_id(),
                fetch.access_kind(),
                cache::AccessStat::ReservationFailure(failure),
                num_accesses,
                // if self.inner.cache_config.accelsim_compat {
                //     1
                // } else {
                //     fetch.access.num_transactions()
                // },
            );
            log::debug!("handling write miss for {}: RESERVATION FAIL", &fetch);
            return cache::RequestStatus::RESERVATION_FAIL;
        }

        let event = cache::Event::WriteRequestSent;
        self.send_write_request(fetch.clone(), event, time, events);

        let new_access = mem_fetch::access::Builder {
            kind: self.write_alloc_type,
            addr: fetch.addr(),
            kernel_launch_id: fetch.kernel_launch_id(),
            allocation: fetch.access.allocation.clone(),
            req_size_bytes: self.inner.cache_config.atom_size,
            is_write: false, // Now performing a read
            warp_active_mask: fetch.access.warp_active_mask,
            byte_mask: fetch.access.byte_mask,
            sector_mask: fetch.access.sector_mask,
        }
        .build();

        let physical_addr = self.mem_controller.to_physical_address(new_access.addr);

        let new_fetch = mem_fetch::Builder {
            instr: None,
            access: new_access,
            warp_id: fetch.warp_id,
            core_id: fetch.core_id,
            cluster_id: fetch.cluster_id,
            physical_addr: physical_addr.clone(),
        }
        .build();
        debug_assert_eq!(new_fetch.addr(), fetch.addr());
        debug_assert_eq!(new_fetch.sub_partition_id(), fetch.sub_partition_id());
        debug_assert_eq!(new_fetch.sub_partition_id(), self.inner.id);

        let Some(cache_index) = cache_index else {
            return cache::RequestStatus::RESERVATION_FAIL;
        };

        // Send read request resulting from write miss
        let is_read_only = false;
        let is_write_allocate = true;
        let (should_miss, evicted) = self.inner.send_read_request(
            addr,
            block_addr,
            cache_index,
            new_fetch,
            time,
            events,
            is_read_only,
            is_write_allocate,
        );

        events.push(cache::Event::WriteAllocateSent);

        // does not hold, but we could return another status from send_read_request
        // debug_assert_eq!(should_miss, evicted.is_some());

        if should_miss {
            // If evicted block is modified and not a write-through
            // (already modified lower level)
            // log::debug!(
            //     "evicted block: {:?}",
            //     evicted.as_ref().map(|e| e.block_addr)
            // );

            let write_policy = self.inner.cache_config.write_policy;
            // let not_write_through =
            //     self.inner.cache_config.write_policy != cache::config::WritePolicy::WRITE_THROUGH;

            if let Some(evicted) = evicted {
                if evicted.writeback && !write_policy.is_write_through() {
                    log::debug!("evicted block: {:?}", evicted.block_addr);

                    // SECTOR_MISS and HIT_RESERVED should not send write back
                    debug_assert_eq!(probe_status, cache::RequestStatus::MISS);

                    let writeback_access = mem_fetch::access::Builder {
                        kind: self.write_back_type,
                        addr: evicted.block_addr,
                        kernel_launch_id: fetch.kernel_launch_id(),
                        allocation: evicted.allocation.clone(),
                        req_size_bytes: evicted.modified_size,
                        is_write: true,
                        warp_active_mask: fetch.access.warp_active_mask,
                        byte_mask: evicted.byte_mask,
                        sector_mask: evicted.sector_mask,
                    }
                    .build();

                    // the evicted block may have wrong chip id when advanced L2 hashing
                    // is used, so set the right chip address from the original mf
                    let mut physical_address = self
                        .mem_controller
                        .to_physical_address(writeback_access.addr);
                    // if self.inner.cache_config.accelsim_compat {
                    physical_address.chip = fetch.physical_addr.chip;
                    physical_address.sub_partition = fetch.physical_addr.sub_partition;
                    // }

                    let writeback_fetch = mem_fetch::Builder {
                        instr: None,
                        access: writeback_access,
                        warp_id: 0,
                        core_id: None,
                        cluster_id: None,
                        physical_addr,
                    }
                    .build();

                    let event = cache::Event::WriteBackRequestSent {
                        evicted_block: Some(evicted),
                    };

                    self.send_write_request(writeback_fetch, event, time, events);
                }
            }
            return cache::RequestStatus::MISS;
        }

        cache::RequestStatus::RESERVATION_FAIL
    }

    /// Baseline read miss
    ///
    /// Send read request to lower level memory and perform
    /// write-back as necessary.
    fn read_miss(
        &mut self,
        addr: address,
        cache_index: usize,
        fetch: &mem_fetch::MemFetch,
        time: u64,
        events: &mut Vec<cache::Event>,
        _probe_status: cache::RequestStatus,
    ) -> cache::RequestStatus {
        if self.inner.kind == cache::base::Kind::OffChip {
            debug_assert_eq!(fetch.sub_partition_id(), self.inner.id);
        }

        log::debug!(
            "READ MISS: miss queue={:?}",
            self.inner
                .miss_queue
                .iter()
                .map(|fetch| fetch.to_string())
                .collect::<Vec<_>>()
        );
        if !self.inner.miss_queue_can_fit(1) {
            // cannot handle request this cycle, might need to generate two requests
            // let mut stats = self.inner.stats.lock();
            // if let Some(kernel_launch_id) = fetch.kernel_launch_id() {
            let num_accesses = self.inner.num_accesses_stat(&fetch);
            let kernel_stats = self.inner.stats.get_mut(fetch.kernel_launch_id());
            kernel_stats.inc(
                fetch.allocation_id(),
                fetch.access_kind(),
                cache::AccessStat::ReservationFailure(cache::ReservationFailure::MISS_QUEUE_FULL),
                num_accesses,
                // if self.inner.cache_config.accelsim_compat {
                //     1
                // } else {
                //     fetch.access.num_transactions()
                // },
            );
            // }
            return cache::RequestStatus::RESERVATION_FAIL;
        }

        let block_addr = self.inner.cache_controller.block_addr(addr);
        let (should_miss, evicted) = self.inner.send_read_request(
            addr,
            block_addr,
            cache_index,
            fetch.clone(),
            time,
            events,
            false,
            false,
        );

        let write_policy = self.inner.cache_config.write_policy;
        log::debug!(
            "handling READ MISS for {} (should miss={})",
            fetch,
            should_miss,
        );

        // must not hold
        // debug_assert_eq!(should_miss, evicted.is_some());

        if should_miss {
            // If evicted block is modified and not a write-through
            // (already modified lower level)
            if let Some(evicted) = evicted {
                if evicted.writeback && write_policy != cache::config::WritePolicy::WRITE_THROUGH {
                    let is_write = true;
                    let writeback_access = mem_fetch::access::Builder {
                        kind: self.write_back_type,
                        addr: evicted.block_addr,
                        kernel_launch_id: fetch.kernel_launch_id(),
                        allocation: evicted.allocation.clone(),
                        req_size_bytes: evicted.modified_size,
                        is_write,
                        warp_active_mask: fetch.access.warp_active_mask,
                        byte_mask: evicted.byte_mask,
                        sector_mask: evicted.sector_mask,
                    }
                    .build();

                    let mut physical_addr = self
                        .mem_controller
                        .to_physical_address(writeback_access.addr);

                    // the evicted block may have wrong chip id when
                    // advanced L2 hashing is used, so set the right chip
                    // address from the original mf
                    // if self.inner.cache_config.accelsim_compat {
                    physical_addr.chip = fetch.physical_addr.chip;
                    physical_addr.sub_partition = fetch.physical_addr.sub_partition;
                    // }

                    let writeback_fetch = mem_fetch::Builder {
                        instr: fetch.instr.clone(),
                        access: writeback_access,
                        warp_id: 0,
                        core_id: None,
                        cluster_id: None,
                        physical_addr,
                    }
                    .build();

                    let event = cache::Event::WriteBackRequestSent {
                        evicted_block: None,
                    };

                    log::trace!(
                        "handling READ MISS for {}: => sending writeback {}",
                        fetch,
                        writeback_fetch
                    );

                    self.send_write_request(writeback_fetch, event, time, events);
                }
            }
            return cache::RequestStatus::MISS;
        }

        // read miss on a non-miss line
        cache::RequestStatus::RESERVATION_FAIL
    }

    #[allow(clippy::needless_pass_by_value)]
    fn write_miss_write_allocate_lazy_fetch_on_read(
        &mut self,
        addr: address,
        _cache_index: Option<usize>,
        fetch: mem_fetch::MemFetch,
        time: u64,
        events: &mut Vec<cache::Event>,
        _probe_status: cache::RequestStatus,
    ) -> cache::RequestStatus {
        let block_addr = self.inner.cache_controller.block_addr(addr);
        // panic!("lazy");

        // if the request writes to the whole cache line/sector, then, write and set
        // cache line Modified. and no need to send read request to memory or reserve
        // mshr

        if self.inner.miss_queue_full() {
            // cannot handle request this cycle
            // let mut stats = self.inner.stats.lock();
            let kernel_stats = self.inner.stats.get_mut(fetch.kernel_launch_id());

            kernel_stats.inc(
                fetch.allocation_id(),
                fetch.access_kind(),
                cache::AccessStat::ReservationFailure(cache::ReservationFailure::MISS_QUEUE_FULL),
                1,
            );
            return cache::RequestStatus::RESERVATION_FAIL;
        }

        let write_policy = self.inner.cache_config.write_policy;
        if write_policy.is_write_through() {
            let event = cache::Event::WriteRequestSent {};
            self.send_write_request(fetch.clone(), event, time, events);
        }

        // bool wb = false;
        // evicted_block_info evicted;
        // cache_request_status m_status =
        //     m_tag_array->access(block_addr, time, cache_index, wb, evicted, mf);
        // assert(m_status != HIT);
        // cache_block_t *block = m_tag_array->get_block(cache_index);
        // if (!block->is_modified_line()) {
        //   m_tag_array->inc_dirty();
        // }
        // block->set_status(MODIFIED, mf->get_access_sector_mask());
        // block->set_byte_mask(mf);
        // if (m_status == HIT_RESERVED) {
        //   block->set_ignore_on_fill(true, mf->get_access_sector_mask());
        //   block->set_modified_on_fill(true, mf->get_access_sector_mask());
        //   block->set_byte_mask_on_fill(true);
        // }

        // if (mf->get_access_byte_mask().count() == m_config.get_atom_sz()) {
        //   block->set_m_readable(true, mf->get_access_sector_mask());
        // } else {
        //   block->set_m_readable(false, mf->get_access_sector_mask());
        //   if (m_status == HIT_RESERVED)
        //     block->set_readable_on_fill(true, mf->get_access_sector_mask());
        // }
        // update_m_readable(mf, cache_index);

        let tag_array::AccessStatus {
            cache_index,
            evicted,
            status,
            ..
        } = self.inner.tag_array.access(block_addr, &fetch, time);
        debug_assert_ne!(status, cache::RequestStatus::HIT);
        let block = self.inner.tag_array.get_block_mut(cache_index.unwrap());
        let sector_idx = fetch.access.sector_mask.first_one().unwrap();

        let was_modified_before = block.is_modified();
        block.set_status(cache::block::Status::MODIFIED, sector_idx);
        block.set_byte_mask(&fetch.access.byte_mask);

        if status == cache::RequestStatus::HIT_RESERVED {
            block.set_ignore_on_fill(true, sector_idx);
            block.set_modified_on_fill(true, sector_idx);
            block.set_byte_mask_on_fill(true);
        }

        let atom_size = self.inner.cache_config.atom_size as usize;
        if fetch.access.byte_mask.count_ones() == atom_size {
            block.set_readable(true, sector_idx);
        } else {
            block.set_readable(false, sector_idx);
            if status == cache::RequestStatus::HIT_RESERVED {
                block.set_readable_on_fill(true, sector_idx);
            }
        }
        self.update_readable(&fetch, cache_index.unwrap());

        if !was_modified_before {
            self.inner.tag_array.num_dirty += 1;
        }

        // if (m_status != RESERVATION_FAIL) {
        //   // If evicted block is modified and not a write-through
        //   // (already modified lower level)
        //   if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
        //     mem_fetch *wb = m_memfetch_creator->alloc(
        //         evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
        //         evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
        //         true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
        //         NULL);
        //     wb->set_alloc_id(evicted.m_allocation_id);
        //     wb->set_alloc_start_addr(evicted.m_allocation_start_addr);
        //
        //     // the evicted block may have wrong chip id when advanced L2 hashing  is
        //     // used, so set the right chip address from the original mf
        //     wb->set_chip(mf->get_tlx_addr().chip);
        //     wb->set_partition(mf->get_tlx_addr().sub_partition);
        //     send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
        //                        time, events);
        //   }
        //   return cache::RequestStatus::MISS;
        // }

        if !status.is_reservation_fail() {
            // If evicted block is modified and not a write-through
            // (already modified lower level)
            if let Some(evicted) = evicted {
                if evicted.writeback && !write_policy.is_write_through() {
                    let writeback_access = mem_fetch::access::Builder {
                        kind: self.write_back_type,
                        addr: evicted.block_addr,
                        kernel_launch_id: fetch.kernel_launch_id(),
                        allocation: evicted.allocation.clone(),
                        req_size_bytes: evicted.modified_size,
                        is_write: true,
                        warp_active_mask: fetch.access.warp_active_mask,
                        byte_mask: evicted.byte_mask,
                        sector_mask: evicted.sector_mask,
                    }
                    .build();

                    // the evicted block may have wrong chip id when advanced L2 hashing
                    // is used, so set the right chip address from the original mf
                    let mut physical_addr = self
                        .mem_controller
                        .to_physical_address(writeback_access.addr);
                    physical_addr.chip = fetch.physical_addr.chip;
                    physical_addr.sub_partition = fetch.physical_addr.sub_partition;

                    let writeback_fetch = mem_fetch::Builder {
                        instr: None,
                        access: writeback_access,
                        warp_id: 0,
                        core_id: None,
                        cluster_id: None,
                        physical_addr,
                    }
                    .build();

                    let event = cache::Event::WriteBackRequestSent {
                        evicted_block: Some(evicted),
                    };

                    self.send_write_request(writeback_fetch, event, time, events);
                }
            }
            return cache::RequestStatus::MISS;
        }

        cache::RequestStatus::RESERVATION_FAIL
    }

    fn write_miss(
        &mut self,
        addr: address,
        cache_index: Option<usize>,
        fetch: mem_fetch::MemFetch,
        time: u64,
        events: &mut Vec<cache::Event>,
        probe_status: cache::RequestStatus,
    ) -> cache::RequestStatus {
        use cache::config::WriteAllocatePolicy;
        let func = match self.inner.cache_config.write_allocate_policy {
            WriteAllocatePolicy::NO_WRITE_ALLOCATE => Self::write_miss_no_write_allocate,
            WriteAllocatePolicy::WRITE_ALLOCATE => Self::write_miss_write_allocate_naive,
            WriteAllocatePolicy::FETCH_ON_WRITE => {
                // Self::write_miss_write_allocate_fetch_on_write
                unimplemented!("fetch on write")
            }
            WriteAllocatePolicy::LAZY_FETCH_ON_READ => {
                Self::write_miss_write_allocate_lazy_fetch_on_read
                // unimplemented!("fetch on read")
            }
        };
        (func)(self, addr, cache_index, fetch, time, events, probe_status)
    }

    // A general function that takes the result of a tag_array probe.
    //
    // It performs the correspding functions based on the cache configuration.
    fn process_tag_probe(
        &mut self,
        is_write: bool,
        probe: Option<(usize, cache::RequestStatus)>,
        addr: address,
        fetch: mem_fetch::MemFetch,
        events: &mut Vec<cache::Event>,
        time: u64,
    ) -> cache::RequestStatus {
        debug_assert!(
            !matches!(probe, Some((_, cache::RequestStatus::RESERVATION_FAIL))),
            "reservation fail should not be returned as a status"
        );

        let probe_status = probe.map_or(cache::RequestStatus::RESERVATION_FAIL, |(_, s)| s);
        let mut access_status = probe_status;
        let data_size = fetch.data_size();

        if is_write {
            let no_allocate_on_write = self.inner.cache_config.write_allocate_policy
                == cache::config::WriteAllocatePolicy::NO_WRITE_ALLOCATE;
            match probe {
                Some((cache_index, cache::RequestStatus::HIT)) => {
                    access_status = self.write_hit(
                        addr,
                        cache_index,
                        &fetch,
                        time,
                        events,
                        probe_status,
                        // cache::RequestStatus::RESERVATION_FAIL,
                    );
                }
                Some((cache_index, probe_status)) => {
                    access_status =
                        self.write_miss(addr, Some(cache_index), fetch, time, events, probe_status);
                }
                None if no_allocate_on_write => {
                    // this almost never happens
                    // todo!("have no probe status");
                    access_status = self.write_miss(
                        addr,
                        None,
                        fetch,
                        time,
                        events,
                        probe_status,
                        // cache::RequestStatus::RESERVATION_FAIL,
                    );
                }
                None => {
                    // this almost never happens
                    // the only reason for reservation fail here is LINE_ALLOC_FAIL
                    // (i.e all lines are reserved)
                    // let mut stats = self.inner.stats.lock();

                    let num_accesses = self.inner.num_accesses_stat(&fetch);
                    let kernel_stats = self.inner.stats.get_mut(fetch.kernel_launch_id());
                    // let count = if self.inner.cache_config.accelsim_compat ||  {
                    //         1
                    //     } else {
                    //         fetch.access.num_transactions()
                    //     },
                    kernel_stats.inc(
                        fetch.allocation_id(),
                        fetch.access_kind(),
                        cache::AccessStat::ReservationFailure(
                            cache::ReservationFailure::LINE_ALLOC_FAIL,
                        ),
                        num_accesses,
                    );
                }
            }
        } else {
            match probe {
                None => {
                    // the only reason for reservation fail here is LINE_ALLOC_FAIL
                    // (i.e all lines are reserved)
                    // let mut stats = self.inner.stats.lock();

                    let num_accsesses = self.inner.num_accesses_stat(&fetch);
                    let kernel_stats = self.inner.stats.get_mut(fetch.kernel_launch_id());
                    kernel_stats.inc(
                        fetch.allocation_id(),
                        fetch.access_kind(),
                        cache::AccessStat::ReservationFailure(
                            cache::ReservationFailure::LINE_ALLOC_FAIL,
                        ),
                        num_accsesses,
                        // if self.inner.cache_config.accelsim_compat {
                        //     1
                        // } else {
                        //     fetch.access.num_transactions()
                        // },
                    );
                }
                Some((_cache_index, cache::RequestStatus::HIT)) => {
                    access_status = self.read_hit(addr, &fetch, time, events);
                }
                Some((cache_index, probe_status)) => {
                    access_status =
                        self.read_miss(addr, cache_index, &fetch, time, events, probe_status);
                }
            }
        }

        self.inner
            .bandwidth
            .use_data_port(data_size, access_status, events);

        access_status
    }
}

impl<MC, CC> Data<MC, CC, stats::cache::PerKernel>
where
    CC: cache::CacheController,
{
    /// Write-back hit: mark block as modified.
    fn write_hit_write_back(
        &mut self,
        addr: address,
        cache_index: usize,
        fetch: &mem_fetch::MemFetch,
        time: u64,
        _events: &mut Vec<cache::Event>,
        _probe_status: cache::RequestStatus,
    ) -> cache::RequestStatus {
        debug_assert_eq!(addr, fetch.addr());

        let block_addr = self.inner.cache_controller.block_addr(addr);
        log::debug!(
            "handling WRITE HIT WRITE BACK for {} (block_addr={}, cache_idx={:?})",
            fetch,
            block_addr,
            cache_index,
        );

        // update LRU state
        let old_cache_index = cache_index;
        let tag_array::AccessStatus { cache_index, .. } =
            self.inner.tag_array.access(block_addr, fetch, time);
        let cache_index = cache_index.expect("write hit write back");
        debug_assert_eq!(old_cache_index, cache_index);

        let block = self.inner.tag_array.get_block_mut(cache_index);
        let was_modified_before = block.is_modified();
        block.set_status(
            cache::block::Status::MODIFIED,
            fetch.access.sector_mask.first_one().unwrap(),
        );
        block.set_byte_mask(&fetch.access.byte_mask);
        if !was_modified_before {
            self.inner.tag_array.num_dirty += 1;
        }
        self.update_readable(fetch, cache_index);

        cache::RequestStatus::HIT
    }

    /// Write-evict hit.
    /// Send request to lower level memory and invalidate corresponding block
    #[allow(dead_code)]
    fn write_hit_write_evict(
        &mut self,
        _addr: address,
        cache_index: usize,
        fetch: &mem_fetch::MemFetch,
        time: u64,
        events: &mut Vec<cache::Event>,
        _probe_status: cache::RequestStatus,
    ) -> cache::RequestStatus {
        if self.inner.miss_queue_full() {
            // let mut stats = self.inner.stats.lock();
            let kernel_stats = self.inner.stats.get_mut(fetch.kernel_launch_id());
            kernel_stats.inc(
                fetch.allocation_id(),
                fetch.access_kind(),
                cache::AccessStat::ReservationFailure(cache::ReservationFailure::MISS_QUEUE_FULL),
                1,
            );
            // cannot handle request this cycle
            return cache::RequestStatus::RESERVATION_FAIL;
        }

        // generate a write-through/evict
        let block = self.inner.tag_array.get_block_mut(cache_index);

        // Invalidate block
        block.set_status(
            cache::block::Status::INVALID,
            fetch.access.sector_mask.first_one().unwrap(),
        );

        let event = cache::Event::WriteRequestSent {};
        self.send_write_request(fetch.clone(), event, time, events);

        cache::RequestStatus::HIT
    }

    #[allow(dead_code, clippy::needless_pass_by_value)]
    fn write_hit_global_write_evict_local_write_back(
        &mut self,
        addr: address,
        cache_index: usize,
        fetch: &mem_fetch::MemFetch,
        time: u64,
        events: &mut Vec<cache::Event>,
        probe_status: cache::RequestStatus,
    ) -> cache::RequestStatus {
        // evict a line that hits on global memory write
        if let mem_fetch::access::Kind::GLOBAL_ACC_W = fetch.access_kind() {
            // write-evict
            self.write_hit_write_evict(addr, cache_index, fetch, time, events, probe_status)
        } else {
            // write-back
            self.write_hit_write_back(addr, cache_index, fetch, time, events, probe_status)
        }
    }

    fn update_readable(&mut self, fetch: &mem_fetch::MemFetch, cache_index: usize) {
        let block = self.inner.tag_array.get_block_mut(cache_index);
        for sector in 0..NUM_SECTORS as usize {
            let sector_mask = &fetch.access.sector_mask;
            if sector_mask[sector] {
                let dirty_byte_mask = &block.dirty_byte_mask();
                let bytes = &dirty_byte_mask
                    [sector * SECTOR_SIZE as usize..(sector + 1) * SECTOR_SIZE as usize];

                // TODO: test if this is equal
                // let mut all_set = true;
                // for byte in (sector * SECTOR_SIZE as usize)..((sector + 1) * SECTOR_SIZE as usize) {
                //     // If any bit in the byte mask (within the sector) is not set,
                //     // the sector is unreadble
                //     if !dirty_byte_mask[byte] {
                //         all_set = false;
                //         break;
                //     }
                // }
                // if all_set {
                if bytes.all() {
                    block.set_readable(true, sector);
                }
            }
        }
    }

    fn read_hit(
        &mut self,
        addr: address,
        fetch: &mem_fetch::MemFetch,
        time: u64,
        _events: &mut [cache::event::Event],
    ) -> cache::RequestStatus {
        let super::base::Base {
            ref mut tag_array,
            ref cache_controller,
            ..
        } = self.inner;
        let block_addr = cache_controller.block_addr(addr);
        let access_status = tag_array.access(block_addr, fetch, time);
        let cache_index = access_status.cache_index.expect("read hit has cache index");

        // Atomics treated as global read/write requests:
        // Perform read, mark line as MODIFIED
        if fetch.is_atomic() {
            debug_assert_eq!(fetch.access_kind(), AccessKind::GLOBAL_ACC_R);
            let block = tag_array.get_block_mut(cache_index);
            let was_modified_before = block.is_modified();
            block.set_status(
                cache::block::Status::MODIFIED,
                fetch.access.sector_mask.first_one().unwrap(),
            );
            block.set_byte_mask(&fetch.access.byte_mask);
            if !was_modified_before {
                tag_array.num_dirty += 1;
            }
        }
        cache::RequestStatus::HIT
    }

    /// Sends write request to lower level memory (write or writeback)
    pub fn send_write_request(
        &mut self,
        mut fetch: mem_fetch::MemFetch,
        request: cache::Event,
        time: u64,
        events: &mut Vec<cache::Event>,
    ) {
        log::debug!("data_cache::send_write_request({})", fetch);
        events.push(request);
        fetch.set_status(self.inner.miss_queue_status, time);
        self.inner.miss_queue.push_back(fetch);
    }

    fn write_miss_no_write_allocate(
        &mut self,
        addr: address,
        _cache_index: Option<usize>,
        fetch: mem_fetch::MemFetch,
        time: u64,
        events: &mut Vec<cache::Event>,
        _probe_status: cache::RequestStatus,
    ) -> cache::RequestStatus {
        debug_assert_eq!(addr, fetch.addr());
        log::debug!(
            "handling WRITE MISS NO WRITE ALLOCATE for {} (miss_queue_full={})",
            fetch,
            self.inner.miss_queue_full()
        );

        if self.inner.miss_queue_full() {
            // let mut stats = self.inner.stats.lock();
            // if let Some(kernel_launch_id) = fetch.kernel_launch_id() {

            let num_accesses = self.inner.num_accesses_stat(&fetch);
            let kernel_stats = self.inner.stats.get_mut(fetch.kernel_launch_id());
            kernel_stats.inc(
                fetch.allocation_id(),
                fetch.access_kind(),
                cache::AccessStat::ReservationFailure(cache::ReservationFailure::MISS_QUEUE_FULL),
                num_accesses,
                // if self.inner.cache_config.accelsim_compat {
                //     1
                // } else {
                //     fetch.access.num_transactions()
                // },
            );
            // }
            // cannot handle request this cycle
            return cache::RequestStatus::RESERVATION_FAIL;
        }

        // on miss, generate write through
        let event = cache::Event::WriteRequestSent;
        self.send_write_request(fetch, event, time, events);
        cache::RequestStatus::MISS
    }

    fn write_hit(
        &mut self,
        addr: address,
        cache_index: usize,
        fetch: &mem_fetch::MemFetch,
        time: u64,
        events: &mut Vec<cache::Event>,
        probe_status: cache::RequestStatus,
    ) -> cache::RequestStatus {
        use cache::config::WritePolicy;
        let func = match self.inner.cache_config.write_policy {
            // TODO: make read only policy deprecated
            // READ_ONLY is now a separate cache class, config is deprecated
            WritePolicy::READ_ONLY => unimplemented!("todo: remove the read only cache write policy / writable data cache set as READ_ONLY"),
            WritePolicy::WRITE_BACK => Self::write_hit_write_back,
            WritePolicy::WRITE_THROUGH => unimplemented!("WritePolicy::WRITE_THROUGH"),
            WritePolicy::WRITE_EVICT => unimplemented!("WritePolicy::WRITE_EVICT"),
            WritePolicy::LOCAL_WRITE_BACK_GLOBAL_WRITE_THROUGH => unimplemented!("WritePolicy::LOCAL_WB_GLOBAL_WT"),
            // WritePolicy::LOCAL_WB_GLOBAL_WT => Self::write_hit_global_write_evict_local_write_back,
        };
        (func)(self, addr, cache_index, fetch, time, events, probe_status)
    }
}

impl<MC, CC, S> crate::engine::cycle::Component for Data<MC, CC, S> {
    // impl<CC, S> crate::engine::cycle::Component for Data<CC, S> {
    fn cycle(&mut self, cycle: u64) {
        self.inner.cycle(cycle);
    }
}

impl<MC, CC> cache::Cache<stats::cache::PerKernel> for Data<MC, CC, stats::cache::PerKernel>
// impl<CC> cache::Cache<stats::cache::PerKernel> for Data<CC, stats::cache::PerKernel>
where
    MC: crate::mcu::MemoryController,
    CC: cache::CacheController,
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    // fn per_kernel_stats(&self) -> &Arc<Mutex<stats::cache::PerKernel>> {
    fn per_kernel_stats(&self) -> &stats::cache::PerKernel {
        &self.inner.stats
    }

    fn per_kernel_stats_mut(&mut self) -> &mut stats::cache::PerKernel {
        &mut self.inner.stats
    }

    fn controller(&self) -> &dyn cache::CacheController {
        &self.inner.cache_controller
    }

    fn write_state(
        &self,
        csv_writer: &mut csv::Writer<std::io::BufWriter<std::fs::File>>,
    ) -> eyre::Result<()> {
        self.inner.tag_array.write_state(csv_writer)
    }

    fn access(
        &mut self,
        addr: address,
        fetch: mem_fetch::MemFetch,
        events: &mut Vec<cache::Event>,
        time: u64,
    ) -> cache::RequestStatus {
        let super::base::Base {
            ref cache_controller,
            ref cache_config,
            kind,
            ..
        } = self.inner;

        debug_assert_eq!(&fetch.access.addr, &addr);
        if !fetch.access_kind().is_inst() {
            debug_assert!(fetch.data_size() <= cache_config.atom_size);
        }

        if kind == cache::base::Kind::OffChip {
            debug_assert!(
                fetch.sub_partition_id() >= self.inner.id,
                "L2 #{} received {} [addr={}] with destination {} (addr gives {})",
                self.inner.id,
                fetch,
                fetch.addr(),
                fetch.sub_partition_id(),
                self.mem_controller
                    .to_physical_address(fetch.addr())
                    .sub_partition,
            );
        }

        let is_write = fetch.is_write();
        let access_kind = fetch.access_kind();
        let allocation_id = fetch.allocation_id();
        let kernel_launch_id = fetch.kernel_launch_id();
        let block_addr = cache_controller.block_addr(addr);

        log::debug!(
            "{}::data_cache::access({fetch}, write = {is_write}, size = {}, block = {block_addr}, time = {})",
            self.inner.name,
            fetch.data_size(), time,
        );

        let dbg_fetch = if DEBUG_PRINT {
            Some(fetch.clone())
        } else {
            None
        };

        let probe = self
            .inner
            .tag_array
            .probe(block_addr, &fetch, is_write, true);
        let probe_status = probe.map_or(cache::RequestStatus::RESERVATION_FAIL, |(_, s)| s);

        if time == 0 {
            debug_assert!(matches!(
                probe_status,
                cache::RequestStatus::HIT | cache::RequestStatus::HIT_RESERVED
            ));
        }

        let access_status =
            self.process_tag_probe(is_write, probe, addr, fetch.clone(), events, time);

        let access_stat = if self.inner.cache_config.accelsim_compat {
            cache::select_status_accelsim_compat(probe_status, access_status)
        } else {
            cache::select_status(probe_status, access_status)
        };

        if DEBUG_PRINT {
            let dbg_fetch = dbg_fetch.unwrap();
            log::debug!(
                "{}::access({}) => status={:?} (probe={:?} access={:?}) accesses={} size={} bytes={}",
                self.inner.name,
                &dbg_fetch,
                access_stat,
                probe_status,
                access_status,
                self.inner.num_accesses_stat(&fetch),
                fetch.access.data_size(),
                fetch.access.byte_mask.count_ones(),
            );
            if self.inner.name.contains("L2") && dbg_fetch.core_id.is_some() {
                let is_write = dbg_fetch.is_write();
                let is_hit = matches!(
                    access_stat,
                    cache::RequestStatus::HIT | cache::RequestStatus::HIT_RESERVED
                );
                // if false || (!is_hit && is_write) {
                if !is_hit && !is_write {
                    // if !is_hit {
                    log::error!(
                        "{}::access({}, {}) => status={:?} (probe={:?} access={:?}) accesses={} size={} bytes={}",
                        self.inner.name,
                        &dbg_fetch,
                        dbg_fetch.addr(),
                        access_stat,
                        probe_status,
                        access_status,
                        self.inner.num_accesses_stat(&fetch),
                        fetch.access.data_size(),
                        fetch.access.byte_mask.count_ones(),
                    );
                }
            }
        }

        // let mut stats = self.inner.stats.lock();
        let num_accesses = self.inner.num_accesses_stat(&fetch);
        let kernel_stats = self.inner.stats.get_mut(kernel_launch_id);
        kernel_stats.inc(
            allocation_id,
            access_kind,
            cache::AccessStat::Status(access_stat),
            num_accesses,
            // if self.inner.cache_config.accelsim_compat {
            //     1
            // } else {
            //     fetch.access.num_transactions()
            // },
        );

        if crate::DEBUG_PRINT
            && (probe_status, access_status)
                != (
                    cache::RequestStatus::HIT_RESERVED,
                    cache::RequestStatus::RESERVATION_FAIL,
                )
        {
            use trace_model::ToBitString;
            let addr = fetch.relative_byte_addr();
            eprintln!(
                "{:>40}: cycle={:<5} fetch {:<40} inst={:<20} addr={:<5} ({:<4}) size={:<2} sector={} probe status={:<10?} access status={:<10?}",
                self.inner.name,
                time,
                fetch,
                fetch.instr.as_ref().map(ToString::to_string).as_deref().unwrap_or("?"),
                addr,
                fetch.relative_byte_addr() / 4,
                fetch.data_size(),
                trace_model::colorize_bits(fetch.access.sector_mask[..4].to_bit_string(), None),
                probe_status,
                access_status,
            );
        }
        // #[cfg(feature = "detailed-stats")]
        // kernel_stats.accesses.push((
        //     (&fetch).into(),
        //     allocation_id,
        //     stats::cache::AccessStatus((access_kind.into(), access_stat.into())),
        // ));
        access_status
    }

    fn write_allocate_policy(&self) -> cache::config::WriteAllocatePolicy {
        self.inner.cache_config.write_allocate_policy
    }

    fn next_access(&mut self) -> Option<mem_fetch::MemFetch> {
        self.inner.next_access()
    }

    fn ready_accesses(&self) -> Option<&VecDeque<mem_fetch::MemFetch>> {
        self.inner.ready_accesses()
    }

    fn has_ready_accesses(&self) -> bool {
        self.inner.has_ready_accesses()
    }

    fn fill(&mut self, fetch: mem_fetch::MemFetch, time: u64) {
        self.inner.fill(fetch, time);
    }

    fn waiting_for_fill(&self, fetch: &mem_fetch::MemFetch) -> bool {
        self.inner.waiting_for_fill(fetch)
    }

    fn flush(&mut self) -> usize {
        self.inner.flush()
    }

    fn invalidate(&mut self) {
        self.inner.invalidate();
    }

    fn invalidate_addr(&mut self, addr: address) {
        self.inner.invalidate_addr(addr);
    }

    fn num_used_lines(&self) -> usize {
        self.inner.tag_array.num_used_lines()
    }

    fn num_used_bytes(&self) -> u64 {
        self.inner.tag_array.num_used_lines() as u64 * self.inner.cache_config.line_size as u64
    }

    fn num_total_lines(&self) -> usize {
        self.inner.tag_array.num_total_lines()
    }
}

impl<MC, CC, S> cache::Bandwidth for Data<MC, CC, S> {
    // impl<CC, S> cache::Bandwidth for Data<CC, S> {
    fn has_free_data_port(&self) -> bool {
        self.inner.has_free_data_port()
    }

    fn has_free_fill_port(&self) -> bool {
        self.inner.has_free_fill_port()
    }
}
