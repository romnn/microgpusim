use super::mem_fetch::BitString;
use super::{
    address, cache,
    cache::{Cache, CacheBandwidth},
    dram,
    fifo::{FifoQueue, Queue},
    interconn as ic, l2, mem_fetch, Packet,
};
use crate::config::{self, CacheConfig, GPUConfig};
use console::style;
use std::cell::RefCell;
use std::collections::{HashSet, VecDeque};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

pub const MAX_MEMORY_ACCESS_SIZE: u32 = 128;

/// four sectors
pub const SECTOR_CHUNCK_SIZE: u32 = 4;

/// Sector size is 32 bytes width
pub const SECTOR_SIZE: u32 = 32;

pub fn was_write_sent(events: &[cache::Event]) -> bool {
    events
        .iter()
        .any(|event| event.kind == cache::EventKind::WRITE_REQUEST_SENT)
}

pub fn was_writeback_sent(events: &[cache::Event]) -> Option<&cache::Event> {
    events
        .iter()
        .find(|event| event.kind == cache::EventKind::WRITE_BACK_REQUEST_SENT)
}

pub fn was_read_sent(events: &[cache::Event]) -> bool {
    events
        .iter()
        .any(|event| event.kind == cache::EventKind::READ_REQUEST_SENT)
}

pub fn was_writeallocate_sent(events: &[cache::Event]) -> bool {
    events
        .iter()
        .any(|event| event.kind == cache::EventKind::WRITE_ALLOCATE_SENT)
}

#[derive()]
// pub struct MemorySubPartition<I, Q = FifoQueue<mem_fetch::MemFetch>> {
pub struct MemorySubPartition<Q = FifoQueue<mem_fetch::MemFetch>> {
    pub id: usize,
    pub partition_id: usize,
    // pub cluster_id: usize,
    // pub core_id: usize,
    /// memory configuration
    pub config: Arc<GPUConfig>,
    pub stats: Arc<Mutex<stats::Stats>>,

    /// queues
    pub interconn_to_l2_queue: Q,
    pub l2_to_dram_queue: Arc<Mutex<Q>>,
    pub dram_to_l2_queue: Q,
    /// L2 cache hit response queue
    pub l2_to_interconn_queue: Q,
    rop_queue: VecDeque<mem_fetch::MemFetch>,

    // fetch_interconn: Arc<I>,
    // l2_cache: Option<l2::DataL2<I>>,
    pub l2_cache: Option<Box<dyn cache::Cache>>,
    // l2_cache: Option<l2::DataL2<ic::ToyInterconnect<Packet>>>,

    // class mem_fetch *L2dramout;
    wb_addr: Option<u64>,

    // class memory_stats_t *m_stats;
    request_tracker: HashSet<mem_fetch::MemFetch>,

    // This is a cycle offset that has to be applied to the l2 accesses to account
    // for the cudamemcpy read/writes. We want GPGPU-Sim to only count cycles for
    // kernel execution but we want cudamemcpy to go through the L2. Everytime an
    // access is made from cudamemcpy this counter is incremented, and when the l2
    // is accessed (in both cudamemcpyies and otherwise) this value is added to
    // the gpgpu-sim cycle counters.
    memcpy_cycle_offset: usize,
}

// impl<I, Q> MemorySubPartition<I, Q>
impl<Q> MemorySubPartition<Q>
where
    Q: Queue<mem_fetch::MemFetch> + 'static,
    // I: ic::MemFetchInterface + 'static,
{
    pub fn new(
        id: usize,
        partition_id: usize,
        // core_id: usize,
        // fetch_interconn: Arc<I>,
        config: Arc<GPUConfig>,
        stats: Arc<Mutex<stats::Stats>>,
    ) -> Self {
        // need to migrate memory config for this
        // assert!(id < config.num_mem_sub_partition);
        // assert!(id < config.numjjjkk);

        let interconn_to_l2_queue = Q::new(
            "icnt-to-L2",
            Some(0),
            Some(config.dram_partition_queue_interconn_to_l2),
        );
        let l2_to_dram_queue = Arc::new(Mutex::new(Q::new(
            "L2-to-dram",
            Some(0),
            Some(config.dram_partition_queue_l2_to_dram),
        )));
        let dram_to_l2_queue = Q::new(
            "dram-to-L2",
            Some(0),
            Some(config.dram_partition_queue_dram_to_l2),
        );
        let l2_to_interconn_queue = Q::new(
            "L2-to-icnt",
            Some(0),
            Some(config.dram_partition_queue_l2_to_interconn),
        );

        let l2_cache: Option<Box<dyn cache::Cache>> = match &config.data_cache_l2 {
            Some(l2_config) => {
                let l2_mem_port = Arc::new(ic::L2Interface {
                    l2_to_dram_queue: Arc::clone(&l2_to_dram_queue),
                });

                let cache_stats = Arc::new(Mutex::new(stats::Cache::default()));
                Some(Box::new(l2::DataL2::new(
                    format!("mem-sub-{}-{}", id, style("L2-CACHE").green()),
                    0, // core_id,
                    0, // cluster_id,
                    l2_mem_port,
                    cache_stats,
                    config.clone(),
                    l2_config.clone(),
                )))
            }
            None => None,
        };

        Self {
            id,
            partition_id,
            // cluster_id,
            // core_id,
            config,
            stats,
            // fetch_interconn,
            l2_cache,
            wb_addr: None,
            memcpy_cycle_offset: 0,
            interconn_to_l2_queue,
            l2_to_dram_queue,
            dram_to_l2_queue,
            l2_to_interconn_queue,
            rop_queue: VecDeque::new(),
            request_tracker: HashSet::new(),
        }
    }

    fn breakdown_request_to_sector_requests(
        &self,
        fetch: mem_fetch::MemFetch,
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

    pub fn push(&mut self, fetch: mem_fetch::MemFetch) {
        // todo!("mem sub partition: push");
        // m_stats->memlatstat_icnt2mem_pop(m_req);
        let mut requests = Vec::new();
        let l2_config = self.config.data_cache_l2.as_ref().unwrap();
        if l2_config.kind == config::CacheKind::Sector {
            requests.extend(self.breakdown_request_to_sector_requests(fetch));
        } else {
            requests.push(fetch);
        }

        for mut fetch in requests.drain(..) {
            // self.request_tracker.insert(fetch);
            assert!(!self.interconn_to_l2_queue.full());
            fetch.set_status(mem_fetch::Status::IN_PARTITION_ICNT_TO_L2_QUEUE, 0);

            // EDIT: we could skip the rop queue here, but then its harder to debug
            // self.interconn_to_l2_queue.enqueue(fetch);

            if fetch.is_texture() {
                self.interconn_to_l2_queue.enqueue(fetch);
                // fetch.status = mem_fetch::Status::IN_PARTITION_ICNT_TO_L2_QUEUE;
                // m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
            } else {
                // rop_delay_t r;
                // r.req = req;
                // r.ready_cycle = cycle + m_config->rop_latency;
                // m_rop.push(r);
                // req->set_status(IN_PARTITION_ROP_DELAY,
                // m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
                log::trace!("{}: {fetch}", style("PUSH TO ROP").red());
                self.rop_queue.push_back(fetch);
                // assert!(!self.interconn_to_l2_queue.full());
                // fetch.set_status(mem_fetch::Status::IN_PARTITION_ICNT_TO_L2_QUEUE, 0);
                // self.interconn_to_l2_queue.enqueue(fetch);
            }
        }
    }

    pub fn full(&self, size: u32) -> bool {
        self.interconn_to_l2_queue.full()
    }

    pub fn interconn_to_l2_can_fit(&self, size: usize) -> bool {
        self.interconn_to_l2_queue.can_fit(size)
    }

    pub fn busy(&self) -> bool {
        !self.request_tracker.is_empty()
    }

    pub fn flush_l2(&mut self) {
        if let Some(l2) = &mut self.l2_cache {
            l2.flush();
        }
    }

    pub fn invalidate_l2(&mut self) {
        if let Some(l2) = &mut self.l2_cache {
            l2.invalidate();
        }
    }

    pub fn pop(&mut self) -> Option<mem_fetch::MemFetch> {
        use mem_fetch::AccessKind;

        let fetch = self.l2_to_interconn_queue.dequeue()?;
        // self.request_tracker.remove(fetch);
        if fetch.is_atomic() {
            // fetch.do_atomic();
            unimplemented!("atomic memory operation");
        }
        // panic!(
        //     "l2 to dram queue fetch: access kind = {:?}",
        //     fetch.access_kind(),
        // );
        match fetch.access_kind() {
            // writeback accesses not counted
            AccessKind::L2_WRBK_ACC | AccessKind::L1_WRBK_ACC => None,
            _ => Some(fetch),
        }
    }

    pub fn top(&mut self) -> Option<&mem_fetch::MemFetch> {
        use super::AccessKind;
        match self
            .l2_to_interconn_queue
            .first()
            .map(|fetch| fetch.access_kind())
        {
            Some(AccessKind::L2_WRBK_ACC | AccessKind::L1_WRBK_ACC) => {
                self.l2_to_interconn_queue.dequeue();
                // self.request_tracker.remove(fetch);
                return None;
            }
            _ => {}
        }

        self.l2_to_interconn_queue.first()
    }

    // pub fn full(&self) -> bool {
    //     self.interconn_to_l2_queue.full()
    // }
    //
    // pub fn has_available_size(&self, size: usize) -> bool {
    //     self.interconn_to_l2_queue.has_available_size(size)
    // }

    pub fn set_done(&self, fetch: &mem_fetch::MemFetch) {
        todo!("mem sub partition: set done");
    }

    pub fn dram_l2_queue_push(&mut self, fetch: &mem_fetch::MemFetch) {
        todo!("mem sub partition: dram l2 queue push");
    }

    pub fn dram_l2_queue_full(&self) -> bool {
        todo!("mem sub partition: dram l2 queue full");
    }

    pub fn cache_cycle(&mut self, cycle: usize) {
        use config::CacheWriteAllocatePolicy;
        use mem_fetch::{AccessKind, Status};

        let log_line = style(format!(
            " => memory sub partition[{}] cache cycle {}",
            self.id, cycle
        ))
        .blue();

        log::trace!(
            "{}: rop queue={:?}, icnt to l2 queue={}, l2 to icnt queue={}, l2 to dram queue={}",
            log_line,
            self.rop_queue
                .iter()
                .map(|f| f.to_string())
                .collect::<Vec<_>>(),
            self.interconn_to_l2_queue,
            self.l2_to_interconn_queue,
            self.l2_to_dram_queue.lock().unwrap(),
        );

        // L2 fill responses
        if let Some(ref mut l2_cache) = self.l2_cache {
            let queue_full = self.l2_to_interconn_queue.full();

            log::trace!(
                "{}: l2 cache ready accesses={:?} l2 to icnt queue full={}",
                log_line,
                l2_cache
                    .ready_accesses()
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .map(|fetch| fetch.to_string())
                    .collect::<Vec<_>>(),
                queue_full,
            );

            // todo: move config into l2
            let l2_config = self.config.data_cache_l2.as_ref().unwrap();
            // if !l2_config.disabled {}
            if l2_cache.has_ready_accesses() && !queue_full {
                let mut fetch = l2_cache.next_access().unwrap();
                // panic!("fetch from l2 cache ready");

                // Don't pass write allocate read request back to upper level cache
                if fetch.access_kind() != &AccessKind::L2_WR_ALLOC_R {
                    fetch.set_reply();
                    fetch.set_status(Status::IN_PARTITION_L2_TO_ICNT_QUEUE, 0);
                    // m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
                    self.l2_to_interconn_queue.enqueue(fetch);
                } else {
                    if l2_config.write_allocate_policy == CacheWriteAllocatePolicy::FETCH_ON_WRITE {
                        todo!("fetch on write: l2 to icnt queue");
                        let mut original_write_fetch = *fetch.original_fetch.unwrap();
                        original_write_fetch.set_reply();
                        original_write_fetch
                            .set_status(mem_fetch::Status::IN_PARTITION_L2_TO_ICNT_QUEUE, 0);
                        // m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
                        self.l2_to_interconn_queue.enqueue(original_write_fetch);
                    }
                    // self.request_tracker.remove(fetch);
                    // delete mf;
                }
            }
        }

        // DRAM to L2 (texture) and icnt (not texture)
        if let Some(reply) = self.dram_to_l2_queue.first() {
            match self.l2_cache {
                Some(ref mut l2_cache) if l2_cache.waiting_for_fill(&reply) => {
                    if l2_cache.has_free_fill_port() {
                        let mut reply = self.dram_to_l2_queue.dequeue().unwrap();
                        log::trace!("filling L2 with {}", &reply);
                        reply.set_status(mem_fetch::Status::IN_PARTITION_L2_FILL_QUEUE, 0);
                        l2_cache.fill(reply)
                        // l2_cache.fill(&mut reply)
                        // reply will be gone forever at this point
                        // m_dram_L2_queue->pop();
                    } else {
                        log::trace!("skip filling L2 with {}: no free fill port", &reply);
                    }
                }
                _ if !self.l2_to_interconn_queue.full() => {
                    let mut reply = self.dram_to_l2_queue.dequeue().unwrap();
                    if reply.is_write() && reply.kind == mem_fetch::Kind::WRITE_ACK {
                        reply.set_status(mem_fetch::Status::IN_PARTITION_L2_TO_ICNT_QUEUE, 0);
                    }
                    // m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
                    log::trace!("pushing {} to interconn queue", &reply);
                    self.l2_to_interconn_queue.enqueue(reply);
                }
                _ => {
                    log::trace!(
                        "skip pushing {} to interconn queue: l2 to interconn queue full",
                        &reply
                    );
                }
            }
        }

        // prior L2 misses inserted into m_L2_dram_queue here
        if let Some(ref mut l2_cache) = self.l2_cache {
            l2_cache.cycle();
        }

        // new L2 texture accesses and/or non-texture accesses
        if !self.l2_to_dram_queue.lock().unwrap().full() {
            // && !self.interconn_to_l2_queue.empty() {
            if let Some(fetch) = self.interconn_to_l2_queue.first() {
                // let l2_cache_config = self.config.data_cache_l2.as_ref();
                if let Some(ref mut l2_cache) = self.l2_cache {
                    if (self.config.data_cache_l2_texture_only && fetch.is_texture())
                        || !self.config.data_cache_l2_texture_only
                    {
                        // L2 is enabled and access is for L2
                        // todo!("l2 is enabled and have access for L2");
                        let output_full = self.l2_to_interconn_queue.full();
                        let port_free = l2_cache.has_free_data_port();
                        if !output_full && port_free {
                            // std::list<cache_event> events;
                            let mut events = Vec::new();
                            // let events = None;
                            let status = l2_cache.access(
                                fetch.addr(),
                                fetch.clone(),
                                // m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle + m_memcpy_cycle_offset,
                                &mut events,
                            );
                            let write_sent = was_write_sent(&events);
                            let read_sent = was_read_sent(&events);
                            log::trace!(
                                "probing L2 cache address={}, status={:?}",
                                fetch.addr(),
                                status
                            );

                            if status == cache::RequestStatus::HIT {
                                let mut fetch = self.interconn_to_l2_queue.dequeue().unwrap();
                                if !write_sent {
                                    // L2 cache replies
                                    assert!(!read_sent);
                                    if fetch.access_kind() == &mem_fetch::AccessKind::L1_WRBK_ACC {
                                        // m_request_tracker.erase(mf);
                                        // delete mf;
                                    } else {
                                        fetch.set_reply();
                                        fetch.set_status(
                                            mem_fetch::Status::IN_PARTITION_L2_TO_ICNT_QUEUE,
                                            0,
                                        );
                                        self.l2_to_interconn_queue.enqueue(fetch);
                                    }
                                    // m_icnt_L2_queue->pop();
                                } else {
                                    assert!(write_sent);
                                    // m_icnt_L2_queue->pop();
                                }
                            } else if status != cache::RequestStatus::RESERVATION_FAIL {
                                // L2 cache accepted request
                                let mut fetch = self.interconn_to_l2_queue.dequeue().unwrap();
                                let wa_policy = l2_cache.write_allocate_policy();
                                // let is_fetch_on_write = l2_cache.write_allocate_policy()
                                //     == config::CacheWriteAllocatePolicy::FETCH_ON_WRITE;
                                let should_fetch = matches!(
                                    wa_policy,
                                    config::CacheWriteAllocatePolicy::FETCH_ON_WRITE
                                        | config::CacheWriteAllocatePolicy::LAZY_FETCH_ON_READ
                                );
                                if fetch.is_write()
                                    && should_fetch
                                    && !was_writeallocate_sent(&events)
                                {
                                    if fetch.access_kind() == &mem_fetch::AccessKind::L1_WRBK_ACC {
                                        //     m_request_tracker.erase(mf);
                                        //     delete mf;
                                    } else {
                                        fetch.set_reply();
                                        fetch.set_status(
                                            mem_fetch::Status::IN_PARTITION_L2_TO_ICNT_QUEUE,
                                            0,
                                        );
                                        // m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
                                        panic!("l2 to interconn queue push");
                                        self.l2_to_interconn_queue.enqueue(fetch);
                                    }
                                }
                            } else {
                                // reservation fail
                                assert!(!write_sent);
                                assert!(!read_sent);
                                // L2 cache lock-up: will try again next cycle
                            }
                        }
                    }
                } else {
                    // L2 is disabled or non-texture access to texture-only L2
                    let mut fetch = self.interconn_to_l2_queue.dequeue().unwrap();
                    fetch.set_status(mem_fetch::Status::IN_PARTITION_L2_TO_DRAM_QUEUE, 0);

                    self.l2_to_dram_queue.lock().unwrap().enqueue(fetch);
                }
            }
        }

        // rop delay queue
        // if (!m_rop.empty() && (cycle >= m_rop.front().ready_cycle) &&
        //     !m_icnt_L2_queue->full()) {
        if !self.interconn_to_l2_queue.full() {
            if let Some(mut fetch) = self.rop_queue.pop_front() {
                log::trace!("{}: {fetch}", style("POP FROM ROP").red());
                fetch.set_status(mem_fetch::Status::IN_PARTITION_ICNT_TO_L2_QUEUE, 0);
                // m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
                self.interconn_to_l2_queue.enqueue(fetch);
            }
        }
    }
}

#[derive()]
// pub struct MemoryPartitionUnit<I> {
pub struct MemoryPartitionUnit {
    id: usize,
    // cluster_id: usize,
    // core_id: usize,
    dram: dram::DRAM,
    pub dram_latency_queue: VecDeque<mem_fetch::MemFetch>,
    // fetch_interconn: Arc<I>,
    // pub sub_partitions: Vec<Rc<RefCell<MemorySubPartition<I>>>>,
    pub sub_partitions: Vec<Rc<RefCell<MemorySubPartition<FifoQueue<mem_fetch::MemFetch>>>>>,
    arbitration_metadata: dram::ArbitrationMetadata,

    config: Arc<GPUConfig>,
    stats: Arc<Mutex<stats::Stats>>,
}

impl MemoryPartitionUnit
// impl<I> MemoryPartitionUnit<I>
// where
//     I: ic::MemFetchInterface + 'static,
{
    pub fn new(
        id: usize,
        // cluster_id: usize,
        // core_id: usize,
        // fetch_interconn: Arc<I>,
        config: Arc<GPUConfig>,
        stats: Arc<Mutex<stats::Stats>>,
    ) -> Self {
        let num_sub_partitions = config.num_sub_partition_per_memory_channel;
        let sub_partitions: Vec<_> = (0..num_sub_partitions)
            .map(|i| {
                let sub_id = id * num_sub_partitions + i;

                let sub = Rc::new(RefCell::new(MemorySubPartition::new(
                    sub_id,
                    id,
                    // core_id,
                    // fetch_interconn.clone(),
                    config.clone(),
                    stats.clone(),
                )));
                // let l2_port = Arc::new(ic::L2Interface {
                //     sub_partition_unit: sub.clone(),
                // });
                // let l2_cache: Option<Box<dyn cache::Cache>> = match &config.data_cache_l2 {

                // if let Some(l2_config) = &config.data_cache_l2 {
                //     sub.borrow_mut().l2_cache = Some(Box::new(l2::DataL2::new(
                //         0, // core_id,
                //         0, // cluster_id,
                //         l2_port,
                //         stats.clone(),
                //         config.clone(),
                //         l2_config.clone(),
                //     )));
                // }

                sub
            })
            .collect();

        let dram = dram::DRAM::new(config.clone(), stats.clone());
        let arbitration_metadata = dram::ArbitrationMetadata::new(&*config);
        Self {
            id,
            // cluster_id,
            // core_id,
            // fetch_interconn,
            config,
            stats,
            dram,
            dram_latency_queue: VecDeque::new(),
            arbitration_metadata,
            sub_partitions,
        }
    }

    pub fn busy(&self) -> bool {
        self.sub_partitions
            .iter()
            .any(|sub| sub.try_borrow().unwrap().busy())
    }

    // pub fn sub_partition(&self, p: usize) -> {
    //     self.sub_partitions[
    // }

    fn global_sub_partition_id_to_local_id(&self, global_sub_partition_id: usize) -> usize {
        let mut local_id = global_sub_partition_id;
        local_id -= self.id * self.config.num_sub_partition_per_memory_channel;
        local_id
    }

    pub fn handle_memcpy_to_gpu(
        &self,
        addr: address,
        global_subpart_id: usize,
        mask: mem_fetch::MemAccessSectorMask,
    ) {
        let p = self.global_sub_partition_id_to_local_id(global_subpart_id);
        // log::trace!(
        //       "copy engine request received for address={}, local_subpart={}, global_subpart={}, sector_mask={}",
        //       addr, p, global_subpart_id, mask.to_bit_string());

        // self.mem_sub_partititon[p].force_l2_tag_update(addr, mask);
        // m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle, mask);
    }

    pub fn cache_cycle(&mut self, cycle: usize) {
        // todo!("mem partition unit: cache_cycle");
        // for p < m_config->m_n_sub_partition_per_memory_channel
        for mem_sub in self.sub_partitions.iter_mut() {
            mem_sub.borrow_mut().cache_cycle(cycle);
        }
    }

    pub fn simple_dram_cycle(&mut self) {
        log::trace!("{} ...", style("simple dram cycle").red());
        // pop completed memory request from dram and push it to dram-to-L2 queue
        // of the original sub partition
        // if !self.dram_latency_queue.is_empty() &&
        //     ((m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) >=
        //      m_dram_latency_queue.front().ready_cycle)) {
        if let Some(mut returned_fetch) = self.dram_latency_queue.front_mut() {
            if !matches!(
                returned_fetch.access_kind(),
                mem_fetch::AccessKind::L1_WRBK_ACC | mem_fetch::AccessKind::L2_WRBK_ACC
            ) {
                self.dram.access(returned_fetch);

                returned_fetch.set_reply(); // todo: is it okay to do that here?
                log::trace!(
                    "got {} fetch return from dram latency queue (write={})",
                    returned_fetch,
                    returned_fetch.is_write()
                );

                let dest_global_spid = returned_fetch.sub_partition_id();
                let dest_spid = self.global_sub_partition_id_to_local_id(dest_global_spid);
                let mut sub = self.sub_partitions[dest_spid].borrow_mut();
                debug_assert_eq!(sub.id, dest_global_spid);

                if !sub.dram_to_l2_queue.full() {
                    // here we could set reply
                    let mut returned_fetch = self.dram_latency_queue.pop_front().unwrap();
                    // dbg!(&returned_fetch);
                    // returned_fetch.set_reply();

                    if returned_fetch.access_kind() == &mem_fetch::AccessKind::L1_WRBK_ACC {
                        // sub.set_done(returned_fetch);
                        // delete mf_return;
                    } else {
                        returned_fetch
                            .set_status(mem_fetch::Status::IN_PARTITION_DRAM_TO_L2_QUEUE, 0);
                        // m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
                        self.arbitration_metadata.return_credit(dest_spid);
                        // log::trace!(
                        //     "mem_fetch request {:?} return from dram to sub partition {}",
                        //     returned_fetch, dest_spid
                        // );

                        debug_assert!(returned_fetch.is_reply());
                        sub.dram_to_l2_queue.enqueue(returned_fetch);
                    }
                } else {
                    // panic!("fyi: simple dram model stall");
                }
            } else {
                log::trace!(
                    "DROPPING {} fetch return from dram latency queue (write={})",
                    returned_fetch,
                    returned_fetch.is_write()
                );

                // this->set_done(mf_return);
                // delete mf_return;
                self.dram_latency_queue.pop_front();
            }
        }

        // L2->DRAM queue to DRAM latency queue
        // Arbitrate among multiple L2 subpartitions
        let last_issued_partition = self.arbitration_metadata.last_borrower();
        for sub_id in 0..self.sub_partitions.len() {
            // for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
            //      p++) {
            let spid = (sub_id + last_issued_partition + 1) % self.sub_partitions.len();
            // self.config->m_n_sub_partition_per_memory_channel;
            let mut sub = self.sub_partitions[spid].borrow_mut();
            debug_assert_eq!(sub.id, spid);
            // if !sub.l2_to_dram_queue.is_empty() && self.can_issue_to_dram(spid) {
            // let sub = self.sub_partitions[inner_sub_partition_id].borrow();

            // let sub_partition_contention = sub.l2_to_dram_queue.lock().unwrap().full();
            // let sub_partition_contention = sub.l2_to_dram_queue.lock().unwrap().full();
            let sub_partition_contention = sub.dram_to_l2_queue.full();
            let has_dram_resource = self.arbitration_metadata.has_credits(spid);
            let can_issue_to_dram = has_dram_resource && !sub_partition_contention;

            {
                log::debug!("checking sub partition[{spid}]:");
                log::debug!(
                    "\t icnt to l2 queue ({:3}) = {}",
                    sub.interconn_to_l2_queue.len(),
                    style(&sub.interconn_to_l2_queue).red()
                );
                log::debug!(
                    "\t l2 to icnt queue ({:3}) = {}",
                    sub.l2_to_interconn_queue.len(),
                    style(&sub.l2_to_interconn_queue).red()
                );
                let l2_to_dram_queue = sub.l2_to_dram_queue.lock().unwrap();
                log::debug!(
                    "\t l2 to dram queue ({:3}) = {}",
                    l2_to_dram_queue.len(),
                    style(&l2_to_dram_queue).red()
                );
                log::debug!(
                    "\t dram to l2 queue ({:3}) = {}",
                    sub.dram_to_l2_queue.len(),
                    style(&sub.dram_to_l2_queue).red()
                );
                let dram_latency_queue: Vec<_> = self
                    .dram_latency_queue
                    .iter()
                    .map(|f| f.to_string())
                    .collect();
                log::debug!(
                    "\t dram latency queue ({:3}) = {:?}",
                    dram_latency_queue.len(),
                    style(&dram_latency_queue).red()
                );
                log::debug!(
                    "\t can issue to dram={} dram to l2 queue full={}",
                    can_issue_to_dram,
                    sub.dram_to_l2_queue.full()
                );
                // log::debug!("");
            }

            if can_issue_to_dram {
                let mut l2_to_dram_queue = sub.l2_to_dram_queue.lock().unwrap();
                if let Some(fetch) = l2_to_dram_queue.first() {
                    if self.dram.full(fetch.is_write()) {
                        break;
                    }

                    let mut fetch = l2_to_dram_queue.dequeue().unwrap();
                    log::trace!(
                        "simple dram: issue {} from sub partition {} to DRAM",
                        &fetch,
                        sub.id
                    );
                    // log::trace!(
                    //     "issue mem_fetch request {:?} from sub partition {} to dram",
                    //     fetch, spid
                    // );
                    // dram_delay_t d;
                    // d.req = mf;
                    // d.ready_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                    //                 m_config->dram_latency;
                    fetch.set_status(mem_fetch::Status::IN_PARTITION_DRAM_LATENCY_QUEUE, 0);
                    // m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
                    self.dram_latency_queue.push_back(fetch);
                    self.arbitration_metadata.borrow_credit(spid);
                    break; // the DRAM should only accept one request per cycle
                }
            }
        }
    }

    // pub fn dram_cycle(&mut self) {
    //     todo!("deprecated: dram cycle");
    //     use mem_fetch::{AccessKind, Status};
    //     // todo!("mem partition unit: dram_cycle");
    //     // TODO
    //     return;
    //
    //     // pop completed memory request from dram and push it to
    //     // dram-to-L2 queue of the original sub partition
    //     if let Some(return_fetch) = self.dram.return_queue_top() {
    //         panic!("have completed memory request from DRAM");
    //         let dest_global_spid = return_fetch.sub_partition_id() as usize;
    //         let dest_spid = self.global_sub_partition_id_to_local_id(dest_global_spid);
    //         let mem_sub = self.sub_partitions[dest_spid].borrow();
    //         debug_assert_eq!(mem_sub.id, dest_global_spid);
    //         if !mem_sub.dram_l2_queue_full() {
    //             if return_fetch.access_kind() == &AccessKind::L1_WRBK_ACC {
    //                 mem_sub.set_done(return_fetch);
    //                 // delete mf_return;
    //             } else {
    //                 mem_sub.dram_l2_queue_push(return_fetch);
    //                 return_fetch.set_status(Status::IN_PARTITION_DRAM_TO_L2_QUEUE, 0);
    //                 // m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
    //                 // m_arbitration_metadata.return_credit(dest_spid);
    //                 log::trace!(
    //                     "mem_fetch request {:?} return from dram to sub partition {}",
    //                     return_fetch, dest_spid
    //                 );
    //             }
    //             self.dram.return_queue_pop();
    //         }
    //     } else {
    //         self.dram.return_queue_pop();
    //     }
    //
    //     self.dram.cycle();
    //
    //     // L2->DRAM queue to DRAM latency queue
    //     // Arbitrate among multiple L2 subpartitions
    //     let num_sub_partitions = self.config.num_sub_partition_per_memory_channel;
    //     let last_issued_partition = self.arbitration_metadata.last_borrower;
    //
    //     for p in 0..num_sub_partitions {
    //         let spid = (p + last_issued_partition + 1) % num_sub_partitions;
    //         let sub = self.sub_partitions[spid].borrow_mut();
    //         if !sub.l2_to_dram_queue.is_empty() && self.can_issue_to_dram(spid) {
    //             let Some(fetch) = sub.l2_to_dram_queue.first() else { break; };
    //             if self.dram.full(fetch.is_write()) {
    //                 break;
    //             }
    //
    //             let fetch = sub.l2_to_dram_queue.dequeue().unwrap();
    //             panic!("issue mem_fetch from sub partition to dram");
    //             log::trace!(
    //                 "Issue mem_fetch request {} from sub partition {} to dram",
    //                 fetch.addr(),
    //                 spid,
    //             );
    //             // dram_delay_t d;
    //             // d.req = mf;
    //             // d.ready_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
    //             //                 m_config->dram_latency;
    //             // m_dram_latency_queue.push_back(d);
    //             // mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE,
    //             //                m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
    //             // m_arbitration_metadata.borrow_credit(spid);
    //             // break; // the DRAM should only accept one request per cycle
    //         }
    //     }
    // }

    // determine whether a given subpartition can issue to DRAM
    // fn can_issue_to_dram(&self, inner_sub_partition_id: usize) -> bool {
    //     let sub = self.sub_partitions[inner_sub_partition_id].borrow();
    //     let sub_partition_contention = sub.dram_to_l2_queue.full();
    //     let has_dram_resource = self
    //         .arbitration_metadata
    //         .has_credits(inner_sub_partition_id);
    //
    //     log::trace!(
    //         "sub partition {} sub_partition_contention={} has_dram_resource={}",
    //         inner_sub_partition_id, sub_partition_contention, has_dram_resource
    //     );
    //
    //     has_dram_resource && !sub_partition_contention
    // }
}
