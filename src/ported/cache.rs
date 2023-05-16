use super::{address, interconn as ic, mem_fetch, mshr, stats::STATS, tag_array};
use crate::config;
use std::collections::VecDeque;
use std::sync::Arc;

#[derive(Debug)]
pub struct TextureL1 {
    id: usize,
    interconn: ic::Interconnect,
}

impl TextureL1 {
    pub fn new(id: usize, interconn: ic::Interconnect) -> Self {
        Self { id, interconn }
    }

    // pub fn new(name: String) -> Self {
    //     Self { name }
    // }
    pub fn cycle(&mut self) {}

    pub fn fill(&self, fetch: &mem_fetch::MemFetch) {}

    pub fn has_free_fill_port(&self) -> bool {
        false
    }
}

#[derive(Debug, Default)]
pub struct ConstL1 {}

impl ConstL1 {
    pub fn cycle(&mut self) {}

    pub fn fill(&self, fetch: &mem_fetch::MemFetch) {}

    pub fn has_free_fill_port(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum CacheBlockState {
    INVALID = 0,
    RESERVED,
    VALID,
    MODIFIED,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum CacheRequestStatus {
    HIT = 0,
    HIT_RESERVED,
    MISS,
    RESERVATION_FAIL,
    SECTOR_MISS,
    MSHR_HIT,
    NUM_CACHE_REQUEST_STATUS,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum CacheReservationFailure {
    /// all line are reserved
    LINE_ALLOC_FAIL = 0,
    /// MISS queue (i.e. interconnect or DRAM) is full
    MISS_QUEUE_FULL,
    MSHR_ENRTY_FAIL,
    MSHR_MERGE_ENRTY_FAIL,
    MSHR_RW_PENDING,
    NUM_CACHE_RESERVATION_FAIL_STATUS,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum CacheEventKind {
    WRITE_BACK_REQUEST_SENT,
    READ_REQUEST_SENT,
    WRITE_REQUEST_SENT,
    WRITE_ALLOCATE_SENT,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct CacheEvent {
    kind: CacheEventKind,

    // if it was write_back event, fill the the evicted block info
    evicted_block: Option<tag_array::EvictedBlockInfo>,
}

impl CacheEvent {
    pub fn new(kind: CacheEventKind) -> Self {
        Self {
            kind,
            evicted_block: None,
        }
    }
}

// #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
// pub enum WriteAllocatePolicy {
//     L1_WR_ALLOC_R,
// }
//
// #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
// pub enum WriteBackPolicy{
// }

/// First level data cache in Fermi.
///
/// The cache uses a write-evict (global) or write-back (local) policy
/// at the granularity of individual blocks.
/// (the policy used in fermi according to the CUDA manual)
#[derive(Debug)]
pub struct DataL1<I> {
    core_id: usize,

    config: Arc<config::GPUConfig>,
    cache_config: Arc<config::CacheConfig>,

    tag_array: tag_array::TagArray<usize>,
    mshrs: mshr::MshrTable,
    fetch_interconn: I,

    /// Specifies type of write allocate request (e.g., L1 or L2)
    write_alloc_type: mem_fetch::AccessKind,
    /// Specifies type of writeback request (e.g., L1 or L2)
    write_back_type: mem_fetch::AccessKind,

    miss_queue: VecDeque<mem_fetch::MemFetch>,
    miss_queue_status: mem_fetch::Status,
    // m_mshrs(config.m_mshr_entries, config.m_mshr_max_merge)
}

impl<I> DataL1<I> {
    pub fn new(
        core_id: usize,
        fetch_interconn: I,
        config: Arc<config::GPUConfig>,
        cache_config: Arc<config::CacheConfig>,
    ) -> Self {
        let tag_array = tag_array::TagArray::new(core_id, 0, cache_config.clone());
        let mshrs = mshr::MshrTable::new(cache_config.mshr_entries, cache_config.mshr_max_merge);
        Self {
            core_id,
            fetch_interconn,
            config,
            cache_config,
            tag_array,
            mshrs,
            miss_queue: VecDeque::new(),
            miss_queue_status: mem_fetch::Status::INITIALIZED,
            write_alloc_type: mem_fetch::AccessKind::L1_WR_ALLOC_R,
            write_back_type: mem_fetch::AccessKind::L1_WRBK_ACC,
        }
    }

    pub fn access(
        &mut self,
        addr: address,
        fetch: mem_fetch::MemFetch,
        events: Option<&mut Vec<CacheEvent>>,
    ) -> CacheRequestStatus {
        debug_assert!(fetch.data_size as usize <= self.cache_config.atom_size());

        let is_write = fetch.is_write();
        let access_kind = *fetch.access_kind();
        let block_addr = self.cache_config.block_addr(addr);

        println!(
            "data_cache::access({addr}, write = {is_write}, size = {}, block = {block_addr})",
            fetch.data_size,
        );

        let (cache_index, probe_status) = self.tag_array.probe(block_addr, &fetch, is_write, true);
        dbg!((cache_index, probe_status));

        let access_status =
            self.process_tag_probe(is_write, probe_status, addr, cache_index, fetch, events);

        let mut stats = STATS.lock().unwrap();
        let stat_cache_request_status = match probe_status {
            CacheRequestStatus::HIT_RESERVED
                if access_status != CacheRequestStatus::RESERVATION_FAIL =>
            {
                probe_status
            }
            CacheRequestStatus::SECTOR_MISS if access_status != CacheRequestStatus::MISS => {
                probe_status
            }
            status => access_status,
        };
        stats
            .accesses
            .entry((access_kind, stat_cache_request_status))
            .and_modify(|s| *s += 1)
            .or_insert(1);
        drop(stats);
        // m_stats.inc_stats_pw(
        // mf->get_access_type(),
        // m_stats.select_stats_status(probe_status, access_status));
        access_status
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
        events: Option<&mut Vec<CacheEvent>>,
        // events: &[CacheEvent],
        probe_status: CacheRequestStatus,
    ) -> CacheRequestStatus {
        let block_addr = self.cache_config.block_addr(addr);
        let access_status = self.tag_array.access(block_addr, time, &fetch);
        let block_index = access_status.index.expect("read hit has index");

        // Atomics treated as global read/write requests:
        // Perform read, mark line as MODIFIED
        if fetch.is_atomic() {
            debug_assert_eq!(*fetch.access_kind(), mem_fetch::AccessKind::GLOBAL_ACC_R);
            let block = self.tag_array.get_block_mut(block_index);
            block.set_status(CacheBlockState::MODIFIED, fetch.access_sector_mask());
            block.set_byte_mask(fetch.access_byte_mask());
            if !block.is_modified() {
                self.tag_array.num_dirty += 1;
            }
        }
        return CacheRequestStatus::HIT;
    }

    /// Checks whether this request can be handled in this cycle.
    ///
    /// num_miss equals max # of misses to be handled on this cycle.
    pub fn miss_queue_can_fit(&self, n: usize) -> bool {
        self.miss_queue.len() + n < self.cache_config.miss_queue_size
    }

    pub fn miss_queue_full(&self) -> bool {
        self.miss_queue.len() >= self.cache_config.miss_queue_size
    }

    /// Are any (accepted) accesses that had to wait for memory now ready?
    ///
    /// (does not include accesses that "HIT")
    pub fn ready_for_access(&self) -> bool {
        self.mshrs.ready_for_access()
    }

    /// Pop next ready access (does not include accesses that "HIT")
    pub fn next_access(&mut self) -> Option<mem_fetch::MemFetch> {
        self.mshrs.next_access()
    }

    // flush all entries in cache
    pub fn flush(&mut self) {
        self.tag_array.flush();
    }

    // invalidate all entries in cache
    pub fn invalidate(&mut self) {
        self.tag_array.invalidate();
    }

    /// Read miss handler.
    ///
    /// Check MSHR hit or MSHR available
    pub fn send_read_request(
        &mut self,
        addr: address,
        block_addr: u64,
        cache_index: Option<usize>,
        mut fetch: mem_fetch::MemFetch,
        time: usize,
        // events: &mut Option<Vec<CacheEvent>>,
        // events: &mut Option<&mut Vec<CacheEvent>>,
        read_only: bool,
        write_allocate: bool,
    ) -> (bool, bool, Option<tag_array::EvictedBlockInfo>) {
        let mut should_miss = false;
        let mut writeback = false;
        let mut evicted = None;

        let mshr_addr = self.cache_config.mshr_addr(fetch.addr());
        let mshr_hit = self.mshrs.probe(mshr_addr);
        let mshr_full = self.mshrs.full(mshr_addr);
        let mut cache_index = cache_index.expect("cache index");

        if mshr_hit && !mshr_full {
            if read_only {
                self.tag_array.access(block_addr, time, &fetch);
            } else {
                tag_array::TagArrayAccessStatus {
                    writeback,
                    evicted,
                    ..
                } = self.tag_array.access(block_addr, time, &fetch);
            }

            self.mshrs.add(mshr_addr, fetch.clone());
            // m_stats.inc_stats(mf->get_access_type(), MSHR_HIT);
            should_miss = true;
        } else if !mshr_hit && !mshr_full && !self.miss_queue_full() {
            if read_only {
                self.tag_array.access(block_addr, time, &fetch);
            } else {
                tag_array::TagArrayAccessStatus {
                    writeback,
                    evicted,
                    ..
                } = self.tag_array.access(block_addr, time, &fetch);
            }

            // m_extra_mf_fields[mf] = extra_mf_fields(
            //     mshr_addr, mf->get_addr(), cache_index, mf->get_data_size(), m_config);
            fetch.data_size = self.cache_config.atom_size() as u32;
            fetch.set_addr(mshr_addr);

            self.mshrs.add(mshr_addr, fetch.clone());
            self.miss_queue.push_back(fetch.clone());
            fetch.set_status(self.miss_queue_status, time);
            if !write_allocate {
                // if let Some(events) = events {
                //     let event = CacheEvent::new(CacheEventKind::READ_REQUEST_SENT);
                //     events.push(event);
                // }
            }

            should_miss = true;
        } else if mshr_hit && mshr_full {
            // m_stats.inc_fail_stats(fetch.access_kind(), MSHR_MERGE_ENRTY_FAIL);
        } else if !mshr_hit && mshr_full {
            // m_stats.inc_fail_stats(fetch.access_kind(), MSHR_ENRTY_FAIL);
        } else {
            panic!("mshr full?");
        }
        (should_miss, write_allocate, evicted)
    }

    /// Sends write request to lower level memory (write or writeback)
    pub fn send_write_request(
        &mut self,
        mut fetch: mem_fetch::MemFetch,
        request: CacheEvent,
        time: usize,
        // events: &Option<&mut Vec<CacheEvent>>,
    ) {
        println!("data_cache::send_write_request(...)");
        // if let Some(events) = events {
        //     events.push(request);
        // }
        fetch.set_status(self.miss_queue_status, time);
        self.miss_queue.push_back(fetch);
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
        // events: Option<&mut Vec<CacheEvent>>,
        // events: &[CacheEvent],
        probe_status: CacheRequestStatus,
    ) -> CacheRequestStatus {
        dbg!(&self.miss_queue);
        dbg!(&self.miss_queue_can_fit(1));
        if !self.miss_queue_can_fit(1) {
            // cannot handle request this cycle
            // (might need to generate two requests)
            // m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
            return CacheRequestStatus::RESERVATION_FAIL;
        }

        let block_addr = self.cache_config.block_addr(addr);
        let (should_miss, writeback, evicted) = self.send_read_request(
            addr,
            block_addr,
            cache_index,
            fetch.clone(),
            time,
            // events.as_mut().cloned(),
            false,
            false,
        );
        dbg!((&should_miss, &writeback, &evicted));

        if should_miss {
            // If evicted block is modified and not a write-through
            // (already modified lower level)
            if writeback
                && self.cache_config.write_policy != config::CacheWritePolicy::WRITE_THROUGH
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
                        &*self.config,
                        if wr {
                            super::WRITE_PACKET_SIZE
                        } else {
                            super::READ_PACKET_SIZE
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
                    self.send_write_request(
                        writeback_fetch,
                        CacheEvent {
                            kind: CacheEventKind::WRITE_BACK_REQUEST_SENT,
                            evicted_block: None,
                        },
                        time,
                        // &events,
                    );
                }
            }
            return CacheRequestStatus::MISS;
        }

        return CacheRequestStatus::RESERVATION_FAIL;
    }

    fn write_miss_no_write_allocate(&mut self) {}
    fn write_miss_write_allocate_naive(&mut self) {}
    fn write_miss_write_allocate_fetch_on_write(&mut self) {}
    fn write_miss_write_allocate_lazy_fetch_on_read(&mut self) {}

    fn write_miss(
        &mut self,
        addr: address,
        cache_index: Option<usize>,
        fetch: mem_fetch::MemFetch,
        time: usize,
        events: Option<&mut Vec<CacheEvent>>,
        // events: &[CacheEvent],
        probe_status: CacheRequestStatus,
    ) -> CacheRequestStatus {
        let func = match self.cache_config.write_allocate_policy {
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
            } // default:
              //   assert(0 && "Error: Must set valid cache write miss policy\n");
              //   break;  // Need to set a write miss function
        };
        todo!("handle write miss");
    }

    fn write_hit(
        &self,
        addr: address,
        cache_index: Option<usize>,
        // cache_index: usize,
        fetch: mem_fetch::MemFetch,
        time: usize,
        // events: &[CacheEvent],
        events: Option<&mut Vec<CacheEvent>>,
        probe_status: CacheRequestStatus,
    ) -> CacheRequestStatus {
        let func = match self.cache_config.write_policy {
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
        CacheRequestStatus::MISS
    }

    // A general function that takes the result of a tag_array probe.
    //
    // It performs the correspding functions based on the
    // cache configuration.
    fn process_tag_probe(
        &mut self,
        is_write: bool,
        probe_status: CacheRequestStatus,
        addr: address,
        cache_index: Option<usize>,
        fetch: mem_fetch::MemFetch,
        events: Option<&mut Vec<CacheEvent>>,
        // events: &[CacheEvent],
    ) -> CacheRequestStatus {
        dbg!(cache_index, probe_status);
        // Each function pointer ( m_[rd/wr]_[hit/miss] ) is set in the
        // data_cache constructor to reflect the corresponding cache
        // configuration options.
        //
        // Function pointers were used to avoid many long conditional
        // branches resulting from many cache configuration options.
        let time = 0;
        let mut access_status = probe_status;
        if is_write {
            if probe_status == CacheRequestStatus::HIT {
                // let cache_index = cache_index.expect("hit has cache idx");
                access_status =
                    self.write_hit(addr, cache_index, fetch, time, events, probe_status);
            } else if probe_status != CacheRequestStatus::RESERVATION_FAIL
                || (probe_status == CacheRequestStatus::RESERVATION_FAIL
                    && self.cache_config.write_allocate_policy
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
            if probe_status == CacheRequestStatus::HIT {
                access_status = self.read_hit(addr, cache_index, fetch, time, events, probe_status);
            } else if probe_status != CacheRequestStatus::RESERVATION_FAIL {
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

    // data_cache(name, config, core_id, type_id, memport, mfcreator, status,
    // , L1_WRBK_ACC, gpu) {}

    /// A general function that takes the result of a tag_array probe
    ///  and performs the correspding functions based on the cache configuration
    ///  The access fucntion calls this function
    // enum cache_request_status process_tag_probe(bool wr,
    //                                             enum cache_request_status status,
    //                                             new_addr_type addr,
    //                                             unsigned cache_index,
    //                                             mem_fetch *mf, unsigned time,
    //                                             std::list<cache_event> &events);

    // /// Sends write request to lower level memory (write or writeback)
    // void send_write_request(mem_fetch *mf, cache_event request, unsigned time,
    //                         std::list<cache_event> &events);
    // void update_m_readable(mem_fetch *mf, unsigned cache_index);
    //
    // // Member Function pointers - Set by configuration options
    // // to the functions below each grouping
    // /******* Write-hit configs *******/
    // enum cache_request_status (data_cache::*m_wr_hit)(
    //     new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    //     std::list<cache_event> &events, enum cache_request_status status);
    // /// Marks block as MODIFIED and updates block LRU
    // enum cache_request_status wr_hit_wb(
    //     new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    //     std::list<cache_event> &events,
    //     enum cache_request_status status);  // write-back
    // enum cache_request_status wr_hit_wt(
    //     new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    //     std::list<cache_event> &events,
    //     enum cache_request_status status);  // write-through
    //
    // /// Marks block as INVALID and sends write request to lower level memory
    // enum cache_request_status wr_hit_we(
    //     new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    //     std::list<cache_event> &events,
    //     enum cache_request_status status);  // write-evict
    // enum cache_request_status wr_hit_global_we_local_wb(
    //     new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    //     std::list<cache_event> &events, enum cache_request_status status);
    // // global write-evict, local write-back
    //
    // /******* Write-miss configs *******/
    // enum cache_request_status (data_cache::*m_wr_miss)(
    //     new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    //     std::list<cache_event> &events, enum cache_request_status status);
    // /// Sends read request, and possible write-back request,
    // //  to lower level memory for a write miss with write-allocate
    // enum cache_request_status wr_miss_wa_naive(
    //     new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    //     std::list<cache_event> &events,
    //     enum cache_request_status
    //         status);  // write-allocate-send-write-and-read-request
    // enum cache_request_status wr_miss_wa_fetch_on_write(
    //     new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    //     std::list<cache_event> &events,
    //     enum cache_request_status
    //         status);  // write-allocate with fetch-on-every-write
    // enum cache_request_status wr_miss_wa_lazy_fetch_on_read(
    //     new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    //     std::list<cache_event> &events,
    //     enum cache_request_status status);  // write-allocate with read-fetch-only
    // enum cache_request_status wr_miss_wa_write_validate(
    //     new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    //     std::list<cache_event> &events,
    //     enum cache_request_status
    //         status);  // write-allocate that writes with no read fetch
    // enum cache_request_status wr_miss_no_wa(
    //     new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    //     std::list<cache_event> &events,
    //     enum cache_request_status status);  // no write-allocate
    //
    // // Currently no separate functions for reads
    // /******* Read-hit configs *******/
    // enum cache_request_status (data_cache::*m_rd_hit)(
    //     new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    //     std::list<cache_event> &events, enum cache_request_status status);
    // enum cache_request_status rd_hit_base(new_addr_type addr,
    //                                       unsigned cache_index, mem_fetch *mf,
    //                                       unsigned time,
    //                                       std::list<cache_event> &events,
    //                                       enum cache_request_status status);
    //
    // /******* Read-miss configs *******/
    // enum cache_request_status (data_cache::*m_rd_miss)(
    //     new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    //     std::list<cache_event> &events, enum cache_request_status status);
    // enum cache_request_status rd_miss_base(new_addr_type addr,
    //                                        unsigned cache_index, mem_fetch *mf,
    //                                        unsigned time,
    //                                        std::list<cache_event> &events,
    //                                        enum cache_request_status status);

    // virtual void init(mem_fetch_allocator *mfcreator) {
    //     m_memfetch_creator = mfcreator;
    //
    //     // Set read hit function
    //     m_rd_hit = &data_cache::rd_hit_base;
    //
    //     // Set read miss function
    //     m_rd_miss = &data_cache::rd_miss_base;
    //
    //     // Set write hit function
    //     switch (m_config.m_write_policy) {
    //       // READ_ONLY is now a separate cache class, config is deprecated
    //       case READ_ONLY:
    //         assert(0 && "Error: Writable Data_cache set as READ_ONLY\n");
    //         break;
    //       case WRITE_BACK:
    //         m_wr_hit = &data_cache::wr_hit_wb;
    //         break;
    //       case WRITE_THROUGH:
    //         m_wr_hit = &data_cache::wr_hit_wt;
    //         break;
    //       case WRITE_EVICT:
    //         m_wr_hit = &data_cache::wr_hit_we;
    //         break;
    //       case LOCAL_WB_GLOBAL_WT:
    //         m_wr_hit = &data_cache::wr_hit_global_we_local_wb;
    //         break;
    //       default:
    //         assert(0 && "Error: Must set valid cache write policy\n");
    //         break;  // Need to set a write hit function
    //     }
    //
    //     // Set write miss function
    //     switch (m_config.m_write_alloc_policy) {
    //       case NO_WRITE_ALLOCATE:
    //         m_wr_miss = &data_cache::wr_miss_no_wa;
    //         break;
    //       case WRITE_ALLOCATE:
    //         m_wr_miss = &data_cache::wr_miss_wa_naive;
    //         break;
    //       case FETCH_ON_WRITE:
    //         m_wr_miss = &data_cache::wr_miss_wa_fetch_on_write;
    //         break;
    //       case LAZY_FETCH_ON_READ:
    //         m_wr_miss = &data_cache::wr_miss_wa_lazy_fetch_on_read;
    //         break;
    //       default:
    //         assert(0 && "Error: Must set valid cache write miss policy\n");
    //         break;  // Need to set a write miss function
    //     }
    //   }

    pub fn cycle(&mut self) {}

    pub fn fill(&self, fetch: &mem_fetch::MemFetch) {}

    pub fn has_free_fill_port(&self) -> bool {
        false
    }

    // virtual enum cache_request_status access(
    //     new_addr_type addr, mem_fetch *mf,
    //    unsigned time,
    //    std::list<cache_event> &events);
}

#[cfg(test)]
mod tests {
    use super::DataL1;
    use crate::config;
    use crate::ported::{instruction, mem_fetch, parse_commands, stats::STATS, KernelInfo};
    use playground::{bindings, bridge};
    use std::collections::VecDeque;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};
    use trace_model::{Command, KernelLaunch, MemAccessTraceEntry};

    struct Interconnect {}

    impl Interconnect {}

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
        let config = Arc::new(config::GPUConfig::default());
        let cache_config = config.data_cache_l1.clone().unwrap();
        let interconn = Interconnect {};
        let mut l1 = DataL1::new(0, interconn, config.clone(), cache_config);

        let traces_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let command_traces_path =
            traces_dir.join("test-apps/vectoradd/traces/vectoradd-100-32-trace/commands.json");
        dbg!(&command_traces_path);
        let mut commands: Vec<Command> =
            parse_commands(&command_traces_path).expect("parse trace commands");

        dbg!(&commands);
        let mut kernels: VecDeque<Arc<KernelInfo>> = VecDeque::new();
        for cmd in commands {
            match cmd {
                Command::MemcpyHtoD {
                    dest_device_addr,
                    num_bytes,
                } => {
                    // sim.memcopy_to_gpu(*dest_device_addr, *num_bytes);
                }
                Command::KernelLaunch(launch) => {
                    let kernel = KernelInfo::new(launch.clone());
                    kernels.push_back(Arc::new(kernel));
                }
            }
        }
        dbg!(&kernels);

        let control_size = 0;
        let warp_id = 0;
        let core_id = 0;
        let cluster_id = 0;

        use crate::ported::scheduler as sched;
        for kernel in kernels {
            while let Some(trace_instr) = kernel.trace_iter.write().unwrap().next() {
                // dbg!(&instr);
                let mut instr = instruction::WarpInstruction::from_trace(&kernel, trace_instr);
                let mut accesses = instr
                    .generate_mem_accesses(&*config)
                    .expect("generated acceseses");
                // dbg!(&accesses);
                assert_eq!(accesses.len(), 1);
                for access in &accesses {
                    println!("{}", &access);
                }

                let access = accesses.remove(0);
                // let access = mem_fetch::MemAccess::from_instr(&instr).unwrap();
                let fetch = mem_fetch::MemFetch::new(
                    instr,
                    access,
                    &config,
                    control_size,
                    warp_id,
                    core_id,
                    cluster_id,
                );
                let status = l1.access(0x00000000, fetch.clone(), None);
                dbg!(&status);
            }
        }

        let mut stats = STATS.lock().unwrap();
        dbg!(&stats);

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
        let config = Arc::new(config::GPUConfig::default());
        let cache_config = config.data_cache_l1.clone().unwrap();
        let interconn = Interconnect {};
        let mut l1 = DataL1::new(0, interconn, config.clone(), cache_config);

        let control_size = 0;
        let warp_id = 0;
        let core_id = 0;
        let cluster_id = 0;

        let kernel = crate::ported::KernelInfo::new(trace_model::KernelLaunch {
            name: "void vecAdd<float>(float*, float*, float*, int)".to_string(),
            trace_file: "./test-apps/vectoradd/traces/vectoradd-100-32-trace/kernel-0-trace".into(),
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
        });

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
            instr,
            access,
            &config,
            control_size,
            warp_id,
            core_id,
            cluster_id,
        );
        let status = l1.access(0x00000000, fetch.clone(), None);
        dbg!(&status);
        let status = l1.access(0x00000000, fetch, None);
        dbg!(&status);

        let mut stats = STATS.lock().unwrap();
        dbg!(&stats);
        assert!(false);
    }
}
