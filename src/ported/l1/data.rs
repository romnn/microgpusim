use crate::config;
use crate::ported::{
    self, address, cache, cache_block, interconn as ic, mem_fetch, mshr, tag_array,
};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// First level data cache in Fermi.
///
/// The cache uses a write-evict (global) or write-back (local) policy
/// at the granularity of individual blocks.
/// (the policy used in fermi according to the CUDA manual)
///
/// TODO: use base cache here!
#[derive(Debug)]
pub struct Data<I> {
    pub inner: super::base::Base<I>,
    // core_id: usize,
    // cluster_id: usize,

    // pub stats: Arc<Mutex<stats::Stats>>,
    // config: Arc<config::GPUConfig>,
    // cache_config: Arc<config::CacheConfig>,

    // tag_array: tag_array::TagArray<usize>,
    // mshrs: mshr::MshrTable,
    // mem_port: Arc<I>,
    /// Specifies type of write allocate request (e.g., L1 or L2)
    write_alloc_type: mem_fetch::AccessKind,
    /// Specifies type of writeback request (e.g., L1 or L2)
    write_back_type: mem_fetch::AccessKind,
    // miss_queue: VecDeque<mem_fetch::MemFetch>,
    // miss_queue_status: mem_fetch::Status,
    // m_mshrs(config.m_mshr_entries, config.m_mshr_max_merge)
}

impl<I> Data<I>
where
    // I: ic::MemPort,
    I: ic::MemFetchInterface,
    // I: ic::Interconnect<crate::ported::core::Packet>,
{
    pub fn new(
        name: String,
        core_id: usize,
        cluster_id: usize,
        cycle: ported::Cycle,
        mem_port: Arc<I>,
        stats: Arc<Mutex<stats::Cache>>,
        config: Arc<config::GPUConfig>,
        cache_config: Arc<config::CacheConfig>,
        write_alloc_type: mem_fetch::AccessKind,
        write_back_type: mem_fetch::AccessKind,
    ) -> Self {
        let inner = super::base::Base::new(
            name,
            core_id,
            cluster_id,
            cycle,
            mem_port,
            stats,
            config,
            cache_config,
        );
        // let tag_array = tag_array::TagArray::new(core_id, 0, cache_config.clone());
        // let mshrs = mshr::MshrTable::new(cache_config.mshr_entries, cache_config.mshr_max_merge);
        Self {
            inner,
            // core_id,
            // cluster_id,
            // mem_port,
            // config,
            // stats,
            // cache_config,
            // tag_array,
            // mshrs,
            // miss_queue: VecDeque::new(),
            // miss_queue_status: mem_fetch::Status::INITIALIZED,
            write_alloc_type,
            write_back_type,
        }
    }

    pub fn cache_config(&self) -> &Arc<config::CacheConfig> {
        &self.inner.cache_config
    }

    /// Write-back hit: Mark block as modified
    fn write_hit_write_back(
        &mut self,
        addr: address,
        cache_index: Option<usize>,
        fetch: mem_fetch::MemFetch,
        time: u64,
        events: &mut Vec<cache::Event>,
        probe_status: cache::RequestStatus,
    ) -> cache::RequestStatus {
        debug_assert_eq!(addr, fetch.addr());

        let block_addr = self.inner.cache_config.block_addr(addr);
        log::debug!(
            "handling WRITE HIT WRITE BACK for {} (block_addr={}, cache_idx={:?})",
            fetch,
            block_addr,
            cache_index,
        );

        // update LRU state
        let tag_array::AccessStatus {
            status,
            index,
            writeback,
            evicted,
            ..
        } = self.inner.tag_array.access(block_addr, &fetch, time);
        let cache_index = index.unwrap();
        let mut block = self.inner.tag_array.get_block_mut(cache_index);
        let was_modified_before = block.is_modified();
        block.set_status(cache_block::Status::MODIFIED, fetch.access_sector_mask());
        block.set_byte_mask(fetch.access_byte_mask());
        if !was_modified_before {
            self.inner.tag_array.num_dirty += 1;
        }
        self.update_readable(&fetch, cache_index);

        cache::RequestStatus::HIT
    }

    fn update_readable(&mut self, fetch: &mem_fetch::MemFetch, cache_index: usize) {
        use crate::ported::mem_sub_partition::{SECTOR_CHUNCK_SIZE, SECTOR_SIZE};
        let mut block = self.inner.tag_array.get_block_mut(cache_index);
        for i in 0..SECTOR_CHUNCK_SIZE as usize {
            let sector_mask = fetch.access_sector_mask();
            if sector_mask[i] {
                let mut all_set = true;
                for k in (i * SECTOR_SIZE as usize)..((i + 1) * SECTOR_SIZE as usize) {
                    // If any bit in the byte mask (within the sector) is not set,
                    // the sector is unreadble
                    if !block.dirty_byte_mask()[k] {
                        all_set = false;
                        break;
                    }
                }
                if all_set {
                    block.set_readable(true, fetch.access_sector_mask());
                }
            }
        }
    }

    fn read_hit(
        &mut self,
        addr: address,
        cache_index: Option<usize>,
        // cache_index: usize,
        fetch: mem_fetch::MemFetch,
        time: u64,
        events: &mut Vec<cache::Event>,
        probe_status: cache::RequestStatus,
    ) -> cache::RequestStatus {
        let super::base::Base {
            ref mut tag_array,
            ref cache_config,
            ..
        } = self.inner;
        let block_addr = cache_config.block_addr(addr);
        let access_status = tag_array.access(block_addr, &fetch, time);
        let block_index = access_status.index.expect("read hit has index");

        // Atomics treated as global read/write requests:
        // Perform read, mark line as MODIFIED
        if fetch.is_atomic() {
            debug_assert_eq!(*fetch.access_kind(), mem_fetch::AccessKind::GLOBAL_ACC_R);
            let block = tag_array.get_block_mut(block_index);
            let was_modified_before = block.is_modified();
            block.set_status(cache_block::Status::MODIFIED, fetch.access_sector_mask());
            block.set_byte_mask(fetch.access_byte_mask());
            if !was_modified_before {
                tag_array.num_dirty += 1;
            }
        }
        return cache::RequestStatus::HIT;
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

    /// Baseline read miss
    ///
    /// Send read request to lower level memory and perform
    /// write-back as necessary.
    fn read_miss(
        &mut self,
        addr: address,
        cache_index: Option<usize>,
        // cache_index: usize,
        fetch: mem_fetch::MemFetch,
        time: u64,
        events: &mut Vec<cache::Event>,
        // events: &[cache::Event],
        probe_status: cache::RequestStatus,
    ) -> cache::RequestStatus {
        if !self.inner.miss_queue_can_fit(1) {
            // cannot handle request this cycle
            // (might need to generate two requests)
            let mut stats = self.inner.stats.lock().unwrap();
            stats.inc(
                *fetch.access_kind(),
                cache::AccessStat::ReservationFailure(cache::ReservationFailure::MISS_QUEUE_FULL),
                1,
            );
            return cache::RequestStatus::RESERVATION_FAIL;
        }

        let block_addr = self.inner.cache_config.block_addr(addr);
        let (should_miss, writeback, evicted) = self.inner.send_read_request(
            addr,
            block_addr,
            cache_index.unwrap(),
            fetch.clone(),
            time,
            events,
            false,
            false,
        );

        let writeback_policy = self.inner.cache_config.write_policy;
        log::debug!(
            "handling READ MISS for {} (should miss={}, writeback={}, writeback policy={:?})",
            fetch,
            should_miss,
            writeback,
            writeback_policy,
        );

        if should_miss {
            // If evicted block is modified and not a write-through
            // (already modified lower level)
            if writeback && writeback_policy != config::CacheWritePolicy::WRITE_THROUGH {
                if let Some(evicted) = evicted {
                    let debug_fetch = fetch.to_string();

                    let is_write = true;
                    let writeback_access = mem_fetch::MemAccess::new(
                        self.write_back_type,
                        evicted.block_addr,
                        evicted.allocation.clone(),
                        evicted.modified_size as u32,
                        is_write,
                        *fetch.access_warp_mask(),
                        evicted.byte_mask,
                        evicted.sector_mask,
                    );
                    // dbg!(&writeback_access);

                    let mut writeback_fetch = mem_fetch::MemFetch::new(
                        fetch.instr,
                        writeback_access,
                        &*self.inner.config,
                        if is_write {
                            ported::WRITE_PACKET_SIZE
                        } else {
                            ported::READ_PACKET_SIZE
                        }
                        .into(),
                        0,
                        0,
                        0,
                    );
                    // dbg!(&writeback_fetch);

                    // the evicted block may have wrong chip id when
                    // advanced L2 hashing is used, so set the right chip
                    // address from the original mf
                    writeback_fetch.tlx_addr.chip = fetch.tlx_addr.chip;
                    writeback_fetch.tlx_addr.sub_partition = fetch.tlx_addr.sub_partition;
                    let event = cache::Event {
                        kind: cache::EventKind::WRITE_BACK_REQUEST_SENT,
                        evicted_block: None, // drop evicted?
                    };

                    log::trace!(
                        "handling READ MISS for {}: => sending writeback {}",
                        debug_fetch,
                        writeback_fetch
                    );

                    self.send_write_request(writeback_fetch, event, time, events);
                }
            }
            return cache::RequestStatus::MISS;
        }

        return cache::RequestStatus::RESERVATION_FAIL;
    }

    fn write_miss_no_write_allocate(
        &mut self,
        addr: address,
        cache_index: Option<usize>,
        fetch: mem_fetch::MemFetch,
        time: u64,
        events: &mut Vec<cache::Event>,
        probe_status: cache::RequestStatus,
    ) -> cache::RequestStatus {
        debug_assert_eq!(addr, fetch.addr());
        log::debug!(
            "handling WRITE MISS NO WRITE ALLOCATE for {} (miss_queue_full={})",
            fetch,
            self.inner.miss_queue_full()
        );

        if self.inner.miss_queue_full() {
            let mut stats = self.inner.stats.lock().unwrap();
            stats.inc(
                *fetch.access_kind(),
                cache::AccessStat::ReservationFailure(cache::ReservationFailure::MISS_QUEUE_FULL),
                1,
            );
            // cannot handle request this cycle
            return cache::RequestStatus::RESERVATION_FAIL;
        }

        // on miss, generate write through
        // (no write buffering -- too many threads for that)
        let event = cache::Event {
            kind: cache::EventKind::WRITE_REQUEST_SENT,
            evicted_block: None,
        };
        self.send_write_request(fetch, event, time, events);
        cache::RequestStatus::MISS
    }

    fn write_miss_write_allocate_naive(
        &mut self,
        addr: address,
        cache_index: Option<usize>,
        fetch: mem_fetch::MemFetch,
        time: u64,
        events: &mut Vec<cache::Event>,
        probe_status: cache::RequestStatus,
    ) -> cache::RequestStatus {
        // todo!("write_miss_write_allocate_naive");

        // what exactly is the difference between the addr and the fetch addr?
        debug_assert_eq!(addr, fetch.addr());

        let block_addr = self.inner.cache_config.block_addr(addr);
        let mshr_addr = self.inner.cache_config.mshr_addr(fetch.addr());

        // Write allocate, maximum 3 requests:
        //  (write miss, read request, write back request)
        //
        //  Conservatively ensure the worst-case request can be handled this cycle
        let mshr_hit = self.inner.mshrs.probe(mshr_addr);
        let mshr_free = !self.inner.mshrs.full(mshr_addr);
        let mshr_miss_but_free = !mshr_hit && mshr_free && !self.inner.miss_queue_full();

        log::debug!("handling write miss for {} (block addr={}, mshr addr={}, mshr hit={} mshr avail={}, miss queue full={})", &fetch, block_addr, mshr_addr, mshr_hit, mshr_free, self.inner.miss_queue_can_fit(2));

        if !self.inner.miss_queue_can_fit(2) || (!(mshr_hit && mshr_free) && !mshr_miss_but_free) {
            // check what is the exact failure reason
            let failure = if !self.inner.miss_queue_can_fit(2) {
                cache::ReservationFailure::MISS_QUEUE_FULL
            } else if mshr_hit && !mshr_free {
                cache::ReservationFailure::MSHR_MERGE_ENTRY_FAIL
            } else if !mshr_hit && !mshr_free {
                cache::ReservationFailure::MSHR_ENTRY_FAIL
            } else {
                panic!("write_miss_write_allocate_naive bad reason");
            };

            let mut stats = self.inner.stats.lock().unwrap();
            stats.inc(
                *fetch.access_kind(),
                cache::AccessStat::ReservationFailure(failure),
                1,
            );
            log::debug!("handling write miss for {}: RESERVATION FAIL", &fetch);
            return cache::RequestStatus::RESERVATION_FAIL;
        }

        let event = cache::Event {
            kind: cache::EventKind::WRITE_REQUEST_SENT,
            evicted_block: None,
        };

        self.send_write_request(fetch.clone(), event, time, events);

        let is_write = false;
        let new_access = mem_fetch::MemAccess::new(
            self.write_alloc_type,
            fetch.addr(),
            fetch.access.allocation.clone(),
            self.cache_config().atom_size(),
            is_write, // Now performing a read
            *fetch.access_warp_mask(),
            *fetch.access_byte_mask(),
            *fetch.access_sector_mask(),
        );

        let new_fetch = mem_fetch::MemFetch::new(
            None,
            new_access,
            &self.inner.config,
            fetch.control_size,
            fetch.warp_id,
            fetch.core_id,
            fetch.cluster_id,
        );

        // Send read request resulting from write miss
        let is_read_only = false;
        let is_write_allocate = true;
        let (should_miss, writeback, evicted) = self.inner.send_read_request(
            addr,
            block_addr,
            cache_index.unwrap(),
            new_fetch,
            time,
            events,
            is_read_only,
            is_write_allocate,
        );

        events.push(cache::Event {
            kind: cache::EventKind::WRITE_ALLOCATE_SENT,
            evicted_block: None,
        });

        if should_miss {
            // If evicted block is modified and not a write-through
            // (already modified lower level)
            // log::debug!(
            //     "evicted block: {:?}",
            //     evicted.as_ref().map(|e| e.block_addr)
            // );
            let not_write_through =
                self.cache_config().write_policy != config::CacheWritePolicy::WRITE_THROUGH;

            if writeback && not_write_through {
                if let Some(evicted) = evicted {
                    log::debug!("evicted block: {:?}", evicted.block_addr);

                    // SECTOR_MISS and HIT_RESERVED should not send write back
                    debug_assert_eq!(probe_status, cache::RequestStatus::MISS);

                    let is_write = true;
                    let writeback_access = mem_fetch::MemAccess::new(
                        self.write_back_type,
                        evicted.block_addr,
                        evicted.allocation.clone(),
                        evicted.modified_size,
                        is_write,
                        *fetch.access_warp_mask(),
                        evicted.byte_mask,
                        evicted.sector_mask,
                    );
                    let control_size = writeback_access.control_size();
                    let mut writeback_fetch = mem_fetch::MemFetch::new(
                        None,
                        writeback_access,
                        &self.inner.config,
                        control_size,
                        0, // warp id
                        0, // self.core_id,
                        0, // self.cluster_id,
                    );

                    // the evicted block may have wrong chip id when advanced L2 hashing
                    // is used, so set the right chip address from the original mf
                    writeback_fetch.tlx_addr.chip = fetch.tlx_addr.chip;
                    writeback_fetch.tlx_addr.sub_partition = fetch.tlx_addr.sub_partition;
                    let event = cache::Event {
                        kind: cache::EventKind::WRITE_BACK_REQUEST_SENT,
                        evicted_block: Some(evicted),
                    };

                    self.send_write_request(writeback_fetch, event, time, events);
                }
            }
            return cache::RequestStatus::MISS;
        }

        cache::RequestStatus::RESERVATION_FAIL
    }

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
                    cache::AccessStat::ReservationFailure(
                        cache::ReservationFailure::MISS_QUEUE_FULL,
                    ),
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

                if writeback && cache_config.write_policy != config::CacheWritePolicy::WRITE_THROUGH
                {
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

    fn write_miss(
        &mut self,
        addr: address,
        cache_index: Option<usize>,
        fetch: mem_fetch::MemFetch,
        time: u64,
        events: &mut Vec<cache::Event>,
        probe_status: cache::RequestStatus,
    ) -> cache::RequestStatus {
        let func = match self.inner.cache_config.write_allocate_policy {
            config::CacheWriteAllocatePolicy::NO_WRITE_ALLOCATE => {
                Self::write_miss_no_write_allocate
            }
            config::CacheWriteAllocatePolicy::WRITE_ALLOCATE => {
                Self::write_miss_write_allocate_naive
            }
            config::CacheWriteAllocatePolicy::FETCH_ON_WRITE => {
                Self::write_miss_write_allocate_fetch_on_write
            }
            config::CacheWriteAllocatePolicy::LAZY_FETCH_ON_READ => {
                Self::write_miss_write_allocate_lazy_fetch_on_read
            }
        };
        (func)(self, addr, cache_index, fetch, time, events, probe_status)
    }

    fn write_hit(
        &mut self,
        addr: address,
        cache_index: Option<usize>,
        // cache_index: usize,
        fetch: mem_fetch::MemFetch,
        time: u64,
        events: &mut Vec<cache::Event>,
        probe_status: cache::RequestStatus,
    ) -> cache::RequestStatus {
        let func = match self.inner.cache_config.write_policy {
            // TODO: make read only policy deprecated
            // READ_ONLY is now a separate cache class, config is deprecated
            config::CacheWritePolicy::READ_ONLY => unimplemented!("todo: remove the read only cache write policy / writable data cache set as READ_ONLY"),
            config::CacheWritePolicy::WRITE_BACK => Self::write_hit_write_back,
            config::CacheWritePolicy::WRITE_THROUGH => unimplemented!("Self::wr_hit_wt"),
            config::CacheWritePolicy::WRITE_EVICT => unimplemented!("Self::wr_hit_we"),
            config::CacheWritePolicy::LOCAL_WB_GLOBAL_WT => unimplemented!("Self::wr_hit_global_we_local_wb"),
        };
        (func)(self, addr, cache_index, fetch, time, events, probe_status)
    }

    // A general function that takes the result of a tag_array probe.
    //
    // It performs the correspding functions based on the
    // cache configuration.
    fn process_tag_probe(
        &mut self,
        is_write: bool,
        probe_status: cache::RequestStatus,
        addr: address,
        cache_index: Option<usize>,
        fetch: mem_fetch::MemFetch,
        events: &mut Vec<cache::Event>,
        time: u64,
    ) -> cache::RequestStatus {
        // dbg!(cache_index, probe_status);
        // Each function pointer ( m_[rd/wr]_[hit/miss] ) is set in the
        // data_cache constructor to reflect the corresponding cache
        // configuration options.
        //
        // Function pointers were used to avoid many long conditional
        // branches resulting from many cache configuration options.
        // let time = self.inner.cycle.get();
        let mut access_status = probe_status;
        let data_size = fetch.data_size;

        if is_write {
            if probe_status == cache::RequestStatus::HIT {
                // let cache_index = cache_index.expect("hit has cache idx");
                access_status =
                    self.write_hit(addr, cache_index, fetch, time, events, probe_status);
            } else if probe_status != cache::RequestStatus::RESERVATION_FAIL
                || (probe_status == cache::RequestStatus::RESERVATION_FAIL
                    && self.inner.cache_config.write_allocate_policy
                        == config::CacheWriteAllocatePolicy::NO_WRITE_ALLOCATE)
            {
                access_status =
                    self.write_miss(addr, cache_index, fetch, time, events, probe_status);
            } else {
                // the only reason for reservation fail here is LINE_ALLOC_FAIL
                // (i.e all lines are reserved)
                let mut stats = self.inner.stats.lock().unwrap();
                stats.inc(
                    *fetch.access_kind(),
                    cache::AccessStat::ReservationFailure(
                        cache::ReservationFailure::LINE_ALLOC_FAIL,
                    ),
                    1,
                );
            }
        } else {
            if probe_status == cache::RequestStatus::HIT {
                access_status = self.read_hit(addr, cache_index, fetch, time, events, probe_status);
            } else if probe_status != cache::RequestStatus::RESERVATION_FAIL {
                access_status =
                    self.read_miss(addr, cache_index, fetch, time, events, probe_status);
            } else {
                // the only reason for reservation fail here is LINE_ALLOC_FAIL
                // (i.e all lines are reserved)
                let mut stats = self.inner.stats.lock().unwrap();
                stats.inc(
                    *fetch.access_kind(),
                    cache::AccessStat::ReservationFailure(
                        cache::ReservationFailure::LINE_ALLOC_FAIL,
                    ),
                    1,
                );
            }
        }

        self.inner
            .bandwidth
            .use_data_port(data_size, access_status, events);

        access_status
    }
}

impl<I> cache::Component for Data<I>
where
    I: ic::MemFetchInterface,
{
    fn cycle(&mut self) {
        // panic!("l2 data cache cycle");
        self.inner.cycle()
    }
}

impl<I> cache::Cache for Data<I>
where
    // I: ic::MemPort,
    I: ic::MemFetchInterface + 'static,
    // I: ic::Interconnect<crate::ported::core::Packet>,
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn stats(&self) -> &Arc<Mutex<stats::Cache>> {
        &self.inner.stats
    }

    fn access(
        &mut self,
        addr: address,
        fetch: mem_fetch::MemFetch,
        events: &mut Vec<cache::Event>,
        time: u64,
    ) -> cache::RequestStatus {
        let super::base::Base {
            ref cache_config, ..
        } = self.inner;

        debug_assert_eq!(&fetch.access.addr, &addr);
        debug_assert!(fetch.data_size <= cache_config.atom_size());

        let is_write = fetch.is_write();
        let access_kind = *fetch.access_kind();
        let block_addr = cache_config.block_addr(addr);

        log::debug!(
            "{}::data_cache::access({fetch}, write = {is_write}, size = {}, block = {block_addr}, time = {})",
            self.inner.name,
            fetch.data_size, time,
        );

        let dbg_fetch = fetch.clone();

        let (cache_index, probe_status) = self
            .inner
            .tag_array
            .probe(block_addr, &fetch, is_write, true);
        // dbg!((cache_index, probe_status));

        let access_status = self.process_tag_probe(
            is_write,
            probe_status,
            addr,
            cache_index,
            fetch,
            events,
            time,
        );
        // dbg!(&access_status);

        log::debug!(
            "{}::access({}) => probe status={:?} access status={:?}",
            self.inner.name,
            &dbg_fetch,
            probe_status,
            access_status
        );

        {
            let mut stats = self.inner.stats.lock().unwrap();
            let stat_cache_request_status = match probe_status {
                cache::RequestStatus::HIT_RESERVED
                    if access_status != cache::RequestStatus::RESERVATION_FAIL =>
                {
                    probe_status
                }
                cache::RequestStatus::SECTOR_MISS
                    if access_status != cache::RequestStatus::MISS =>
                {
                    probe_status
                }
                status => access_status,
            };
            stats.inc(
                access_kind,
                cache::AccessStat::Status(stat_cache_request_status),
                1,
            );
        }
        // m_stats.inc_stats_pw(
        // mf->get_access_type(),
        // m_stats.select_stats_status(probe_status, access_status));
        access_status
    }

    fn write_allocate_policy(&self) -> config::CacheWriteAllocatePolicy {
        self.cache_config().write_allocate_policy
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

    // fn fill(&mut self, fetch: &mut mem_fetch::MemFetch) {
    fn fill(&mut self, fetch: mem_fetch::MemFetch, time: u64) {
        self.inner.fill(fetch, time)
    }

    fn waiting_for_fill(&self, fetch: &mem_fetch::MemFetch) -> bool {
        self.inner.waiting_for_fill(fetch)
    }

    // fn data_port_free(&self) -> bool {
    //
    //     self.inner.data_port_free()
    // }

    // fn flush(&mut self) {
    //     self.inner.flush()
    //     // self.tag_array.flush();
    // }

    // /// Invalidate all entries in cache
    // fn invalidate(&mut self) {
    //     self.inner.invalidate();
    // }
}

impl<I> cache::CacheBandwidth for Data<I> {
    fn has_free_data_port(&self) -> bool {
        self.inner.has_free_data_port()
        // todo!("l1 data: has_free_data_port");
        // false
    }

    fn has_free_fill_port(&self) -> bool {
        self.inner.has_free_fill_port()
        // todo!("l1 data: has_free_fill_port");
        // false
    }
}

#[cfg(test)]
mod tests {
    use super::Data;
    use crate::config;
    use crate::ported::{
        self, cache::Cache, instruction, interconn as ic, mem_fetch, parse_commands,
        scheduler as sched,
    };
    use itertools::Itertools;
    use std::collections::VecDeque;
    use std::path::PathBuf;
    use std::rc::Rc;
    use std::sync::{Arc, Mutex};
    use trace_model::{Command, KernelLaunch, MemAccessTraceEntry};

    #[derive(Debug)]
    struct MockFetchInterconn {}

    impl ic::MemFetchInterface for MockFetchInterconn {
        fn full(&self, size: u32, write: bool) -> bool {
            false
        }
        fn push(&self, fetch: mem_fetch::MemFetch) {}
    }

    fn concat<T>(
        mut a: impl IntoIterator<Item = T>,
        b: impl IntoIterator<Item = T>,
    ) -> impl Iterator<Item = T> {
        a.into_iter().chain(b.into_iter())
    }

    #[ignore = "todo"]
    #[test]
    fn test_ref_data_l1() {
        let control_size = 0;
        let warp_id = 0;
        let core_id = 0;
        let cluster_id = 0;
        // let type_id = bindings::cache_access_logger_types::NORMALS as i32;

        // let l1 = bindings::l1_cache::new(0, interconn, cache_config);
        // let cache_config = bindings::cache_config::new();

        // let mut cache_config = bridge::cache_config::new_cache_config();
        // dbg!(&cache_config.pin_mut().is_streaming());

        // let params = bindings::cache_config_params { disabled: false };
        // let mut cache_config = bridge::cache_config::new_cache_config(params);
        // dbg!(&cache_config.pin_mut().is_streaming());

        // let tag_array = bindings::tag_array::new(cache_config, core_id, type_id);
        // let fetch = bindings::mem_fetch_t::new(
        //     instr,
        //     access,
        //     &config,
        //     control_size,
        //     warp_id,
        //     core_id,
        //     cluster_id,
        // );
        // let status = l1.access(0x00000000, &fetch, vec![]);
        // dbg!(&status);
        // let status = l1.access(0x00000000, &fetch, vec![]);
        // dbg!(&status);

        assert!(false);
    }

    #[ignore = "todo"]
    #[test]
    fn test_data_l1_full_trace() {
        let control_size = 0;
        // let warp_id = 0;
        let core_id = 0;
        let cluster_id = 0;

        let stats = Arc::new(Mutex::new(stats::Cache::default()));
        let config = Arc::new(config::GPUConfig::default());
        let cache_config = config.data_cache_l1.clone().unwrap();
        let interconn = Arc::new(MockFetchInterconn {});
        let cycle: ported::Cycle = Rc::new(ported::AtomicCycle::new(0));
        let mut l1 = Data::new(
            "l1-data".to_string(),
            core_id,
            cluster_id,
            cycle,
            interconn,
            stats.clone(),
            config.clone(),
            Arc::clone(&cache_config.inner),
            mem_fetch::AccessKind::L1_WR_ALLOC_R,
            mem_fetch::AccessKind::L1_WRBK_ACC,
        );

        let trace_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test-apps/vectoradd/traces/vectoradd-100-32-trace/");
        // let command_traces_path =
        //     traces_dircommands.json");
        dbg!(&trace_dir);
        let mut commands: Vec<Command> =
            parse_commands(&trace_dir.join("commands.json")).expect("parse trace commands");

        dbg!(&commands);
        // let mut kernels: VecDeque<Arc<KernelInfo>> = VecDeque::new();
        let mut kernels: VecDeque<_> = VecDeque::new();
        for cmd in commands {
            match cmd {
                Command::MemcpyHtoD { .. } => {}
                Command::MemAlloc { .. } => {}
                // Command::MemcpyHtoD {
                //     allocation_name,
                //     dest_device_addr,
                //     num_bytes,
                // } => {
                //     // sim.memcopy_to_gpu(*dest_device_addr, *num_bytes, allocation_name);
                // }
                Command::KernelLaunch(launch) => {
                    let kernel = ported::KernelInfo::from_trace(&trace_dir, launch.clone());
                    // kernels.push_back(Arc::new(kernel));
                    kernels.push_back(kernel);
                }
            }
        }
        // dbg!(&kernels);

        // for kernel in &mut kernels {
        //     let mut block_iter = kernel.next_block_iter.lock().unwrap();
        //     while let Some(block) = block_iter.next() {
        //         dbg!(&block);
        //         let mut trace_pos = kernel.trace_pos.write().unwrap();
        //         // let mut lock = kernel.trace_iter.write().unwrap();
        //         // let trace_iter = lock.take_while_ref(|entry| entry.block_id == block);
        //         while *trace_pos < kernel.trace.len() {
        //             // for trace in trace_iter {
        //             let trace = &kernel.trace[*trace_pos];
        //             if trace.block_id > block.into() {
        //                 break;
        //             }
        //
        //             *trace_pos += 1;
        //
        //             // dbg!(&trace);
        //             let warp_id = trace.warp_id_in_block as usize;
        //             if warp_id != 0 {
        //                 continue;
        //             }
        //             let mut instr =
        //                 instruction::WarpInstruction::from_trace(&kernel, trace.clone());
        //
        //             let mut accesses = instr
        //                 .generate_mem_accesses(&*config)
        //                 .expect("generated acceseses");
        //             // dbg!(&accesses);
        //             for access in &accesses {
        //                 log::debug!(
        //                     "block {} warp {}: {} access {}",
        //                     &block,
        //                     &warp_id,
        //                     if access.is_write { "store" } else { "load" },
        //                     &access.addr
        //                 );
        //             }
        //             assert_eq!(accesses.len(), 1);
        //
        //             let access = accesses.remove(0);
        //             // let access = mem_fetch::MemAccess::from_instr(&instr).unwrap();
        //             let fetch = mem_fetch::MemFetch::new(
        //                 Some(instr),
        //                 access,
        //                 &config,
        //                 control_size,
        //                 warp_id,
        //                 core_id,
        //                 cluster_id,
        //             );
        //             let mut events = Vec::new();
        //             let status = l1.access(fetch.access.addr, fetch, &mut events);
        //             // let status = l1.access(0x00000000, fetch.clone(), None);
        //             dbg!(&status);
        //         }
        //     }
        //     // while let Some(trace_instr) = kernel.trace_iter.write().unwrap().next() {
        //     //     // dbg!(&instr);
        //     //     let mut instr = instruction::WarpInstruction::from_trace(&kernel, trace_instr);
        //     //     let mut accesses = instr
        //     //         .generate_mem_accesses(&*config)
        //     //         .expect("generated acceseses");
        //     //     // dbg!(&accesses);
        //     //     assert_eq!(accesses.len(), 1);
        //     //     for access in &accesses {
        //     //         // log::debug!(
        //     //         //     "block {} warp {}: access {}",
        //     //         //     &access.block, &access.warp_id, &access.addr
        //     //         // );
        //     //         // log::debug!("{}", &access);
        //     //     }
        //     //     // continue;
        //     //
        //     //     let access = accesses.remove(0);
        //     //     // let access = mem_fetch::MemAccess::from_instr(&instr).unwrap();
        //     //     let fetch = mem_fetch::MemFetch::new(
        //     //         instr,
        //     //         access,
        //     //         &config,
        //     //         control_size,
        //     //         warp_id,
        //     //         core_id,
        //     //         cluster_id,
        //     //     );
        //     //     let status = l1.access(fetch.access.addr, fetch, None);
        //     //     // let status = l1.access(0x00000000, fetch.clone(), None);
        //     //     dbg!(&status);
        //     // }
        // }

        // let mut stats = STATS.lock().unwrap();
        dbg!(&stats.lock().unwrap());

        // let mut warps: Vec<sched::SchedulerWarp> = Vec::new();
        // for kernel in kernels {
        //     loop {
        //         assert!(!warps.is_empty());
        //         kernel.next_threadblock_traces(&mut warps);
        //         dbg!(&warps);
        //         break;
        //     }
        // }

        assert!(false);
    }

    #[ignore = "todo"]
    #[test]
    fn test_data_l1_single_access() {
        let control_size = 0;
        let warp_id = 0;
        let core_id = 0;
        let cluster_id = 0;

        let stats = Arc::new(Mutex::new(stats::Cache::default()));
        let config = Arc::new(config::GPUConfig::default());
        let cache_config = config.data_cache_l1.clone().unwrap();
        let interconn = Arc::new(MockFetchInterconn {});
        let cycle: ported::Cycle = Rc::new(ported::AtomicCycle::new(0));

        let mut l1 = Data::new(
            "l1-data".to_string(),
            core_id,
            cluster_id,
            cycle,
            interconn,
            stats.clone(),
            config.clone(),
            Arc::clone(&cache_config.inner),
            mem_fetch::AccessKind::L1_WR_ALLOC_R,
            mem_fetch::AccessKind::L1_WRBK_ACC,
        );

        let trace_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test-apps/vectoradd/traces/vectoradd-100-32-trace/");

        let launch = trace_model::KernelLaunch {
            name: "void vecAdd<float>(float*, float*, float*, int)".into(),
            trace_file: "kernel-0-trace".into(),
            id: 0,
            grid: trace_model::Dim { x: 1, y: 1, z: 1 },
            block: trace_model::Dim {
                x: 1024,
                y: 1,
                z: 1,
            },
            shared_mem_bytes: 0,
            num_registers: 8,
            binary_version: 61,
            stream_id: 0,
            shared_mem_base_addr: 140663786045440,
            local_mem_base_addr: 140663752491008,
            nvbit_version: "1.5.5".to_string(),
        };
        let kernel = crate::ported::KernelInfo::from_trace(trace_dir, launch);

        let trace_instr = trace_model::MemAccessTraceEntry {
            cuda_ctx: 0,
            sm_id: 0,
            kernel_id: 0,
            block_id: trace_model::Dim::ZERO,
            warp_size: 32,
            warp_id_in_sm: 3,
            warp_id_in_block: 3,
            line_num: 0,
            instr_data_width: 4,
            instr_opcode: "LDG.E.CG".to_string(),
            instr_offset: 176,
            instr_idx: 16,
            instr_predicate: nvbit_model::Predicate {
                num: 0,
                is_neg: false,
                is_uniform: false,
            },
            instr_mem_space: nvbit_model::MemorySpace::Global,
            instr_is_mem: true,
            instr_is_load: true,
            instr_is_store: false,
            instr_is_extended: true,
            active_mask: 15,
            // todo: use real values here
            dest_regs: [0; 1],
            num_dest_regs: 0,
            src_regs: [0; 5],
            num_src_regs: 0,
            addrs: concat(
                [
                    140663086646144,
                    140663086646148,
                    140663086646152,
                    140663086646156,
                ],
                [0; 32 - 4],
            )
            .collect::<Vec<_>>()
            .try_into()
            .unwrap(),
        };
        let mut instr = instruction::WarpInstruction::from_trace(&kernel, trace_instr);
        dbg!(&instr);
        let mut accesses = instr
            .generate_mem_accesses(&*config)
            .expect("generated acceseses");
        assert_eq!(accesses.len(), 1);

        let access = accesses.remove(0);
        // let access = mem_fetch::MemAccess::from_instr(&instr).unwrap();
        let fetch = mem_fetch::MemFetch::new(
            Some(instr),
            access,
            &config,
            control_size,
            warp_id,
            core_id,
            cluster_id,
        );
        // let status = l1.access(0x00000000, fetch.clone(), None);
        let time = 0;
        let mut events = Vec::new();
        let status = l1.access(fetch.addr(), fetch.clone(), &mut events, time);
        dbg!(&status);
        let mut events = Vec::new();
        let status = l1.access(fetch.addr(), fetch, &mut events, time);
        dbg!(&status);

        // let mut stats = STATS.lock().unwrap();
        dbg!(&stats.lock().unwrap());
        assert!(false);
    }
}
