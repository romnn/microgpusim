use crate::config;
use crate::ported::{
    self, address, cache, cache_block, interconn as ic, mem_fetch, mshr, stats::Stats, tag_array,
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
    inner: super::base::Base<I>,
    // core_id: usize,
    // cluster_id: usize,

    // pub stats: Arc<Mutex<Stats>>,
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
        mem_port: Arc<I>,
        stats: Arc<Mutex<Stats>>,
        config: Arc<config::GPUConfig>,
        cache_config: Arc<config::CacheConfig>,
    ) -> Self {
        let inner = super::base::Base::new(
            name,
            core_id,
            cluster_id,
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
            write_alloc_type: mem_fetch::AccessKind::L1_WR_ALLOC_R,
            write_back_type: mem_fetch::AccessKind::L1_WRBK_ACC,
        }
    }

    pub fn cache_config(&self) -> &Arc<config::CacheConfig> {
        &self.inner.cache_config
    }

    fn wr_hit_wb(&self) -> usize {
        0
    }

    fn read_hit(
        &mut self,
        addr: address,
        cache_index: Option<usize>,
        // cache_index: usize,
        fetch: mem_fetch::MemFetch,
        time: usize,
        events: Option<&mut Vec<cache::Event>>,
        // events: &[cache::Event],
        probe_status: cache::RequestStatus,
    ) -> cache::RequestStatus {
        let super::base::Base {
            ref mut tag_array,
            ref cache_config,
            ..
        } = self.inner;
        let block_addr = cache_config.block_addr(addr);
        let access_status = tag_array.access(block_addr, time, &fetch);
        let block_index = access_status.index.expect("read hit has index");

        // Atomics treated as global read/write requests:
        // Perform read, mark line as MODIFIED
        if fetch.is_atomic() {
            debug_assert_eq!(*fetch.access_kind(), mem_fetch::AccessKind::GLOBAL_ACC_R);
            let block = tag_array.get_block_mut(block_index);
            block.set_status(cache_block::State::MODIFIED, fetch.access_sector_mask());
            block.set_byte_mask(fetch.access_byte_mask());
            if !block.is_modified() {
                tag_array.num_dirty += 1;
            }
        }
        return cache::RequestStatus::HIT;
    }

    /// Checks whether this request can be handled in this cycle.
    ///
    /// num_miss equals max # of misses to be handled on this cycle.
    // pub fn miss_queue_can_fit(&self, n: usize) -> bool {
    //     self.miss_queue.len() + n < self.cache_config.miss_queue_size
    // }
    //
    // pub fn miss_queue_full(&self) -> bool {
    //     self.miss_queue.len() >= self.cache_config.miss_queue_size
    // }

    /// Read miss handler.
    ///
    /// Check MSHR hit or MSHR available
    // pub fn send_read_request(
    //     &mut self,
    //     addr: address,
    //     block_addr: u64,
    //     cache_index: Option<usize>,
    //     mut fetch: mem_fetch::MemFetch,
    //     time: usize,
    //     // events: &mut Option<Vec<cache::Event>>,
    //     // events: &mut Option<&mut Vec<cache::Event>>,
    //     read_only: bool,
    //     write_allocate: bool,
    // ) -> (bool, bool, Option<tag_array::EvictedBlockInfo>) {
    //     let mut should_miss = false;
    //     let mut writeback = false;
    //     let mut evicted = None;
    //
    //     let mshr_addr = self.cache_config.mshr_addr(fetch.addr());
    //     let mshr_hit = self.mshrs.probe(mshr_addr);
    //     let mshr_full = self.mshrs.full(mshr_addr);
    //     let mut cache_index = cache_index.expect("cache index");
    //
    //     if mshr_hit && !mshr_full {
    //         if read_only {
    //             self.tag_array.access(block_addr, time, &fetch);
    //         } else {
    //             tag_array::AccessStatus {
    //                 writeback,
    //                 evicted,
    //                 ..
    //             } = self.tag_array.access(block_addr, time, &fetch);
    //         }
    //
    //         self.mshrs.add(mshr_addr, fetch.clone());
    //         // m_stats.inc_stats(mf->get_access_type(), MSHR_HIT);
    //         let mut stats = self.stats.lock().unwrap();
    //         stats.inc_access(
    //             *fetch.access_kind(),
    //             cache::AccessStat::Status(cache::RequestStatus::MSHR_HIT),
    //         );
    //
    //         should_miss = true;
    //     } else if !mshr_hit && !mshr_full && !self.miss_queue_full() {
    //         if read_only {
    //             self.tag_array.access(block_addr, time, &fetch);
    //         } else {
    //             tag_array::AccessStatus {
    //                 writeback,
    //                 evicted,
    //                 ..
    //             } = self.tag_array.access(block_addr, time, &fetch);
    //         }
    //
    //         // m_extra_mf_fields[mf] = extra_mf_fields(
    //         //     mshr_addr, mf->get_addr(), cache_index, mf->get_data_size(), m_config);
    //         fetch.data_size = self.cache_config.atom_size() as u32;
    //         fetch.access.addr = mshr_addr;
    //
    //         self.mshrs.add(mshr_addr, fetch.clone());
    //         self.miss_queue.push_back(fetch.clone());
    //         fetch.set_status(self.miss_queue_status, time);
    //         if !write_allocate {
    //             // if let Some(events) = events {
    //             //     let event = cache::Event::new(cache::EventKind::READ_REQUEST_SENT);
    //             //     events.push(event);
    //             // }
    //         }
    //
    //         should_miss = true;
    //     } else if mshr_hit && mshr_full {
    //         // m_stats.inc_fail_stats(fetch.access_kind(), MSHR_MERGE_ENRTY_FAIL);
    //     } else if !mshr_hit && mshr_full {
    //         // m_stats.inc_fail_stats(fetch.access_kind(), MSHR_ENRTY_FAIL);
    //     } else {
    //         panic!("mshr full?");
    //     }
    //     (should_miss, write_allocate, evicted)
    // }

    /// Sends write request to lower level memory (write or writeback)
    pub fn send_write_request(
        &mut self,
        mut fetch: mem_fetch::MemFetch,
        request: cache::Event,
        time: usize,
        // events: &Option<&mut Vec<cache::Event>>,
    ) {
        println!("data_cache::send_write_request(...)");
        // if let Some(events) = events {
        //     events.push(request);
        // }
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
        time: usize,
        // events: Option<&mut Vec<cache::Event>>,
        // events: &[cache::Event],
        probe_status: cache::RequestStatus,
    ) -> cache::RequestStatus {
        // dbg!((
        //     &self.inner.miss_queue.len(),
        //     &self.inner.cache_config.miss_queue_size
        // ));
        // dbg!(&self.inner.miss_queue_can_fit(1));
        if !self.inner.miss_queue_can_fit(1) {
            // cannot handle request this cycle
            // (might need to generate two requests)
            // m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
            return cache::RequestStatus::RESERVATION_FAIL;
        }

        let block_addr = self.inner.cache_config.block_addr(addr);
        let (should_miss, writeback, evicted) = self.inner.send_read_request(
            addr,
            block_addr,
            cache_index.unwrap(),
            fetch.clone(),
            time,
            // events.as_mut().cloned(),
            false,
            false,
        );
        // dbg!((&should_miss, &writeback, &evicted));

        if should_miss {
            // If evicted block is modified and not a write-through
            // (already modified lower level)
            if writeback
                && self.inner.cache_config.write_policy != config::CacheWritePolicy::WRITE_THROUGH
            {
                if let Some(evicted) = evicted {
                    let wr = true;
                    let access = mem_fetch::MemAccess::new(
                        self.write_back_type,
                        evicted.block_addr,
                        evicted.modified_size as u32,
                        wr,
                        *fetch.access_warp_mask(),
                        evicted.byte_mask,
                        evicted.sector_mask,
                    );

                    // (access, NULL, wr ? WRITE_PACKET_SIZE : READ_PACKET_SIZE, -1,
                    //   m_core_id, m_cluster_id, m_memory_config, cycle);
                    let mut writeback_fetch = mem_fetch::MemFetch::new(
                        fetch.instr,
                        access,
                        &*self.inner.config,
                        if wr {
                            ported::WRITE_PACKET_SIZE
                        } else {
                            ported::READ_PACKET_SIZE
                        }
                        .into(),
                        0,
                        0,
                        0,
                    );

                    //     None,
                    //     access,
                    //     // self.write_back_type,
                    //     &*self.config.l1_cache.unwrap(),
                    //     // evicted.block_addr,
                    //     // evicted.modified_size,
                    //     // true,
                    //     // fetch.access_warp_mask(),
                    //     // evicted.byte_mask,
                    //     // evicted.sector_mask,
                    //     // m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle,
                    //     // -1, -1, -1, NULL,
                    // );
                    // the evicted block may have wrong chip id when
                    // advanced L2 hashing is used, so set the right chip
                    // address from the original mf
                    writeback_fetch.tlx_addr.chip = fetch.tlx_addr.chip;
                    writeback_fetch.tlx_addr.sub_partition = fetch.tlx_addr.sub_partition;
                    let event = cache::Event {
                        kind: cache::EventKind::WRITE_BACK_REQUEST_SENT,
                        evicted_block: None,
                    };

                    self.send_write_request(
                        writeback_fetch,
                        event,
                        time,
                        // &events,
                    );
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
        time: usize,
        events: Option<&mut Vec<cache::Event>>,
        probe_status: cache::RequestStatus,
    ) -> cache::RequestStatus {
        if self.inner.miss_queue_full() {
            // m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
            // cannot handle request this cycle
            return cache::RequestStatus::RESERVATION_FAIL;
        }

        // on miss, generate write through
        // (no write buffering -- too many threads for that)
        let event = cache::Event {
            kind: cache::EventKind::WRITE_REQUEST_SENT,
            evicted_block: None,
        };
        self.send_write_request(fetch, event, time); // , events);
        cache::RequestStatus::MISS
    }

    fn write_miss_write_allocate_naive(
        &mut self,
        addr: address,
        cache_index: Option<usize>,
        fetch: mem_fetch::MemFetch,
        time: usize,
        events: Option<&mut Vec<cache::Event>>,
        probe_status: cache::RequestStatus,
    ) -> cache::RequestStatus {
        // TODO: what exactly is the difference between the
        // addr and the fetch addr?
        let block_addr = self.inner.cache_config.block_addr(addr);
        let mshr_addr = self.inner.cache_config.mshr_addr(fetch.addr());

        // Write allocate, maximum 3 requests:
        //  (write miss, read request, write back request)
        //
        //  Conservatively ensure the worst-case request can be handled this cycle
        let mshr_hit = self.inner.mshrs.probe(mshr_addr);
        let mshr_free = !self.inner.mshrs.full(mshr_addr);
        let mshr_miss_but_free = !mshr_hit && mshr_free && !self.inner.miss_queue_full();
        if !self.inner.miss_queue_can_fit(2) || (!(mshr_hit && mshr_free) && !mshr_miss_but_free) {
            // check what is the exactly the failure reason
            if !self.inner.miss_queue_can_fit(2) {
                // m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
            } else if mshr_hit && !mshr_free {
                // m_stats.inc_fail_stats(mf->get_access_type(), MSHR_MERGE_ENRTY_FAIL);
            } else if !mshr_hit && !mshr_free {
                // m_stats.inc_fail_stats(mf->get_access_type(), MSHR_ENRTY_FAIL);
            } else {
                panic!("write_miss_write_allocate_naive bad reason");
            }
        }
        // todo!("write_miss_write_allocate_naive");
        cache::RequestStatus::RESERVATION_FAIL
    }

    fn write_miss_write_allocate_fetch_on_write(
        &mut self,
        addr: address,
        cache_index: Option<usize>,
        fetch: mem_fetch::MemFetch,
        time: usize,
        events: Option<&mut Vec<cache::Event>>,
        probe_status: cache::RequestStatus,
    ) -> cache::RequestStatus {
        // let super::base::Base { ref cache_config, ref mut tag_array, .. } = self.inner;
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
                // m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
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
            } = self.inner.tag_array.access(block_addr, time, &fetch);
            // , cache_index);
            // , wb, evicted, mf);
            debug_assert_ne!(status, cache::RequestStatus::HIT);
            let block = self.inner.tag_array.get_block_mut(index.unwrap());
            block.set_status(cache_block::State::MODIFIED, fetch.access_sector_mask());
            block.set_byte_mask(fetch.access_byte_mask());
            if status == cache::RequestStatus::HIT_RESERVED {
                block.set_ignore_on_fill(true, fetch.access_sector_mask());
            }
            if !block.is_modified() {
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
        time: usize,
        events: Option<&mut Vec<cache::Event>>,
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
        time: usize,
        events: Option<&mut Vec<cache::Event>>,
        // events: &[cache::Event],
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
        // todo!("handle write miss");
    }

    fn write_hit(
        &self,
        addr: address,
        cache_index: Option<usize>,
        // cache_index: usize,
        fetch: mem_fetch::MemFetch,
        time: usize,
        // events: &[cache::Event],
        events: Option<&mut Vec<cache::Event>>,
        probe_status: cache::RequestStatus,
    ) -> cache::RequestStatus {
        let func = match self.inner.cache_config.write_policy {
            // TODO: make read only policy deprecated
            // READ_ONLY is now a separate cache class, config is deprecated
            config::CacheWritePolicy::READ_ONLY => unimplemented!("todo: remove the read only cache write policy / writable data cache set as READ_ONLY"),
            config::CacheWritePolicy::WRITE_BACK => Self::wr_hit_wb,
            config::CacheWritePolicy::WRITE_THROUGH => Self::wr_hit_wb,
            // m_wr_hit = &data_cache::wr_hit_wt;
            config::CacheWritePolicy::WRITE_EVICT => Self::wr_hit_wb,
            // m_wr_hit = &data_cache::wr_hit_we;
            config::CacheWritePolicy::LOCAL_WB_GLOBAL_WT => Self::wr_hit_wb,
            // m_wr_hit = &data_cache::wr_hit_global_we_local_wb;
        };
        todo!("handle write hit");
        cache::RequestStatus::MISS
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
        events: Option<&mut Vec<cache::Event>>,
        // events: &[cache::Event],
    ) -> cache::RequestStatus {
        // dbg!(cache_index, probe_status);
        // Each function pointer ( m_[rd/wr]_[hit/miss] ) is set in the
        // data_cache constructor to reflect the corresponding cache
        // configuration options.
        //
        // Function pointers were used to avoid many long conditional
        // branches resulting from many cache configuration options.
        let time = 0;
        let mut access_status = probe_status;
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
                // the only reason for reservation fail here is
                // LINE_ALLOC_FAIL (i.e all lines are reserved)
                // m_stats.inc_fail_stats(mf->get_access_type(), LINE_ALLOC_FAIL);
            }
        } else {
            if probe_status == cache::RequestStatus::HIT {
                access_status = self.read_hit(addr, cache_index, fetch, time, events, probe_status);
            } else if probe_status != cache::RequestStatus::RESERVATION_FAIL {
                access_status = self.read_miss(
                    addr,
                    cache_index,
                    fetch,
                    time,
                    // events,
                    probe_status,
                );
            } else {
                // the only reason for reservation fail here is
                // LINE_ALLOC_FAIL (i.e all lines are reserved)
                // m_stats.inc_fail_stats(mf->get_access_type(), LINE_ALLOC_FAIL);
            }
        }

        // m_bandwidth_management.use_data_port(mf, access_status, events);
        access_status
    }

    // Sends next request to lower level of memory
    // pub fn cycle(&mut self) {
    //     println!(
    //         "baseline cache cycle: miss queue size {}",
    //         self.miss_queue.len()
    //     );
    //     if let Some(fetch) = self.miss_queue.front() {
    //         dbg!(&fetch);
    //         if !self.mem_port.full(fetch.data_size, fetch.is_write()) {
    //             if let Some(fetch) = self.miss_queue.pop_front() {
    //                 self.mem_port.push(fetch);
    //             }
    //         }
    //     }
    //     // bool data_port_busy = !m_bandwidth_management.data_port_free();
    //     // bool fill_port_busy = !m_bandwidth_management.fill_port_free();
    //     // m_stats.sample_cache_port_utility(data_port_busy, fill_port_busy);
    //     // m_bandwidth_management.replenish_port_bandwidth();
    // }
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
    I: ic::MemFetchInterface,
    // I: ic::Interconnect<crate::ported::core::Packet>,
{
    fn access(
        &mut self,
        addr: address,
        fetch: mem_fetch::MemFetch,
        events: Option<&mut Vec<cache::Event>>,
    ) -> cache::RequestStatus {
        // let super::base::Base { ref cache_config, ref mut tag_array, ref mut stats, .. } = self.inner;
        let super::base::Base {
            ref cache_config, ..
        } = self.inner;
        // is this always true?
        debug_assert_eq!(&fetch.access.addr, &addr);
        debug_assert!(fetch.data_size <= cache_config.atom_size());

        let is_write = fetch.is_write();
        let access_kind = *fetch.access_kind();
        let block_addr = cache_config.block_addr(addr);

        println!(
            "data_cache::access({addr}, write = {is_write}, size = {}, block = {block_addr})",
            fetch.data_size,
        );

        let (cache_index, probe_status) = self
            .inner
            .tag_array
            .probe(block_addr, &fetch, is_write, true);
        // dbg!((cache_index, probe_status));

        let access_status =
            self.process_tag_probe(is_write, probe_status, addr, cache_index, fetch, events);
        // dbg!(&access_status);

        {
            // let mut stats = STATS.lock().unwrap();
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
            stats.inc_access(
                access_kind,
                cache::AccessStat::Status(stat_cache_request_status),
            );
            // stats
            //     .accesses
            //     .entry((access_kind, stat_cache_request_status))
            //     .and_modify(|s| *s += 1)
            //     .or_insert(1);
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

    fn has_ready_accesses(&self) -> bool {
        self.inner.has_ready_accesses()
    }

    fn fill(&self, fetch: &mem_fetch::MemFetch) {
        self.inner.fill(fetch)
    }

    // fn data_port_free(&self) -> bool {
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
        cache::Cache, instruction, interconn as ic, mem_fetch, parse_commands, scheduler as sched,
        stats::Stats, KernelInfo,
    };
    use itertools::Itertools;
    use playground::{bindings, bridge};
    use std::collections::VecDeque;
    use std::path::PathBuf;
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

    #[test]
    fn test_ref_data_l1() {
        let control_size = 0;
        let warp_id = 0;
        let core_id = 0;
        let cluster_id = 0;
        let type_id = bindings::cache_access_logger_types::NORMALS as i32;

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

    #[test]
    fn test_data_l1_full_trace() {
        let control_size = 0;
        // let warp_id = 0;
        let core_id = 0;
        let cluster_id = 0;

        let stats = Arc::new(Mutex::new(Stats::default()));
        let config = Arc::new(config::GPUConfig::default());
        let cache_config = config.data_cache_l1.clone().unwrap();
        let interconn = Arc::new(MockFetchInterconn {});
        let mut l1 = Data::new(
            "l1-data".to_string(),
            core_id,
            cluster_id,
            interconn,
            stats.clone(),
            config.clone(),
            cache_config,
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
                Command::MemcpyHtoD {
                    dest_device_addr,
                    num_bytes,
                } => {
                    // sim.memcopy_to_gpu(*dest_device_addr, *num_bytes);
                }
                Command::KernelLaunch(launch) => {
                    let kernel = KernelInfo::from_trace(&trace_dir, launch.clone());
                    // kernels.push_back(Arc::new(kernel));
                    kernels.push_back(kernel);
                }
            }
        }
        // dbg!(&kernels);

        for kernel in &mut kernels {
            let mut block_iter = kernel.next_block_iter.lock().unwrap();
            while let Some(block) = block_iter.next() {
                dbg!(&block);
                let mut lock = kernel.trace_iter.write().unwrap();
                let trace_iter = lock.take_while_ref(|entry| entry.block_id == block);
                for trace in trace_iter {
                    // dbg!(&trace);
                    let warp_id = trace.warp_id as usize;
                    if warp_id != 0 {
                        continue;
                    }
                    let instr = instruction::WarpInstruction::from_trace(&kernel, trace);

                    let mut accesses = instr
                        .generate_mem_accesses(&*config)
                        .expect("generated acceseses");
                    // dbg!(&accesses);
                    for access in &accesses {
                        println!(
                            "block {} warp {}: {} access {}",
                            &block,
                            &warp_id,
                            if access.is_write { "store" } else { "load" },
                            &access.addr
                        );
                        // println!("{}", &access);
                    }
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
                    let status = l1.access(fetch.access.addr, fetch, None);
                    // let status = l1.access(0x00000000, fetch.clone(), None);
                    dbg!(&status);
                }
            }
            // while let Some(trace_instr) = kernel.trace_iter.write().unwrap().next() {
            //     // dbg!(&instr);
            //     let mut instr = instruction::WarpInstruction::from_trace(&kernel, trace_instr);
            //     let mut accesses = instr
            //         .generate_mem_accesses(&*config)
            //         .expect("generated acceseses");
            //     // dbg!(&accesses);
            //     assert_eq!(accesses.len(), 1);
            //     for access in &accesses {
            //         // println!(
            //         //     "block {} warp {}: access {}",
            //         //     &access.block, &access.warp_id, &access.addr
            //         // );
            //         // println!("{}", &access);
            //     }
            //     // continue;
            //
            //     let access = accesses.remove(0);
            //     // let access = mem_fetch::MemAccess::from_instr(&instr).unwrap();
            //     let fetch = mem_fetch::MemFetch::new(
            //         instr,
            //         access,
            //         &config,
            //         control_size,
            //         warp_id,
            //         core_id,
            //         cluster_id,
            //     );
            //     let status = l1.access(fetch.access.addr, fetch, None);
            //     // let status = l1.access(0x00000000, fetch.clone(), None);
            //     dbg!(&status);
            // }
        }

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

    #[test]
    fn test_data_l1_single_access() {
        let control_size = 0;
        let warp_id = 0;
        let core_id = 0;
        let cluster_id = 0;

        let stats = Arc::new(Mutex::new(Stats::default()));
        let config = Arc::new(config::GPUConfig::default());
        let cache_config = config.data_cache_l1.clone().unwrap();
        let interconn = Arc::new(MockFetchInterconn {});
        let mut l1 = Data::new(
            "l1-data".to_string(),
            core_id,
            cluster_id,
            interconn,
            stats.clone(),
            config.clone(),
            cache_config,
        );

        let trace_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test-apps/vectoradd/traces/vectoradd-100-32-trace/");

        let launch = trace_model::KernelLaunch {
            name: "void vecAdd<float>(float*, float*, float*, int)".into(),
            trace_file: "kernel-0-trace".into(),
            id: 0,
            grid: nvbit_model::Dim { x: 1, y: 1, z: 1 },
            block: nvbit_model::Dim {
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
            kernel_id: 0,
            block_id: nvbit_model::Dim { x: 0, y: 0, z: 0 },
            warp_id: 3,
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
            instr_is_load: true,
            instr_is_store: false,
            instr_is_extended: true,
            active_mask: 15,
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
        let status = l1.access(fetch.addr(), fetch.clone(), None);
        dbg!(&status);
        let status = l1.access(fetch.addr(), fetch, None);
        dbg!(&status);

        // let mut stats = STATS.lock().unwrap();
        dbg!(&stats.lock().unwrap());
        assert!(false);
    }
}
