use crate::config::{self, GPUConfig};
use crate::ported::{
    self, address, cache,
    fifo::{FifoQueue, Queue},
    interconn as ic, l2, mem_fetch,
};
use console::style;

use std::collections::{HashSet, VecDeque};

use std::sync::{Arc, Mutex};

pub const MAX_MEMORY_ACCESS_SIZE: u32 = 128;

/// four sectors
pub const SECTOR_CHUNCK_SIZE: u32 = 4;

/// Sector size is 32 bytes width
pub const SECTOR_SIZE: u32 = 32;

#[must_use] pub fn was_write_sent(events: &[cache::Event]) -> bool {
    events
        .iter()
        .any(|event| event.kind == cache::EventKind::WRITE_REQUEST_SENT)
}

#[must_use] pub fn was_writeback_sent(events: &[cache::Event]) -> Option<&cache::Event> {
    events
        .iter()
        .find(|event| event.kind == cache::EventKind::WRITE_BACK_REQUEST_SENT)
}

#[must_use] pub fn was_read_sent(events: &[cache::Event]) -> bool {
    events
        .iter()
        .any(|event| event.kind == cache::EventKind::READ_REQUEST_SENT)
}

#[must_use] pub fn was_writeallocate_sent(events: &[cache::Event]) -> bool {
    events
        .iter()
        .any(|event| event.kind == cache::EventKind::WRITE_ALLOCATE_SENT)
}

#[derive()]
pub struct MemorySubPartition<Q = FifoQueue<mem_fetch::MemFetch>> {
    pub id: usize,
    pub partition_id: usize,
    pub config: Arc<GPUConfig>,
    pub stats: Arc<Mutex<stats::Stats>>,

    /// queues
    pub interconn_to_l2_queue: Q,
    pub l2_to_dram_queue: Arc<Mutex<Q>>,
    pub dram_to_l2_queue: Q,
    /// L2 cache hit response queue
    pub l2_to_interconn_queue: Q,
    rop_queue: VecDeque<mem_fetch::MemFetch>,

    pub l2_cache: Option<Box<dyn cache::Cache>>,

    request_tracker: HashSet<mem_fetch::MemFetch>,

    // This is a cycle offset that has to be applied to the l2 accesses to account
    // for the cudamemcpy read/writes. We want GPGPU-Sim to only count cycles for
    // kernel execution but we want cudamemcpy to go through the L2. Everytime an
    // access is made from cudamemcpy this counter is incremented, and when the l2
    // is accessed (in both cudamemcpyies and otherwise) this value is added to
    // the gpgpu-sim cycle counters.
    memcpy_cycle_offset: u64,
}

impl<Q> MemorySubPartition<Q>
where
    Q: Queue<mem_fetch::MemFetch> + 'static,
{
    pub fn new(
        id: usize,
        partition_id: usize,
        cycle: ported::Cycle,
        config: Arc<GPUConfig>,
        stats: Arc<Mutex<stats::Stats>>,
    ) -> Self {
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
                    cycle,
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
            l2_cache,
            memcpy_cycle_offset: 0,
            interconn_to_l2_queue,
            l2_to_dram_queue,
            dram_to_l2_queue,
            l2_to_interconn_queue,
            rop_queue: VecDeque::new(),
            request_tracker: HashSet::new(),
        }
    }

    pub fn force_l2_tag_update(
        &mut self,
        addr: address,
        mask: mem_fetch::MemAccessSectorMask,
        time: u64,
    ) {
        if let Some(ref mut l2_cache) = self.l2_cache {
            l2_cache.force_tag_access(addr, time + self.memcpy_cycle_offset, mask);
            self.memcpy_cycle_offset += 1;
        }
    }

    fn breakdown_request_to_sector_requests(
        &self,
        _fetch: mem_fetch::MemFetch,
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
        // m_stats->memlatstat_icnt2mem_pop(m_req);
        let mut requests = Vec::new();
        let l2_config = self.config.data_cache_l2.as_ref().unwrap();
        if l2_config.inner.kind == config::CacheKind::Sector {
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
                log::debug!("{}: {fetch}", style("PUSH TO ROP").red());
                self.rop_queue.push_back(fetch);
                // assert!(!self.interconn_to_l2_queue.full());
                // fetch.set_status(mem_fetch::Status::IN_PARTITION_ICNT_TO_L2_QUEUE, 0);
                // self.interconn_to_l2_queue.enqueue(fetch);
            }
        }
    }

    pub fn full(&self, _size: u32) -> bool {
        self.interconn_to_l2_queue.full()
    }

    pub fn interconn_to_l2_can_fit(&self, size: usize) -> bool {
        self.interconn_to_l2_queue.can_fit(size)
    }

    pub fn busy(&self) -> bool {
        !self.request_tracker.is_empty()
    }

    pub fn flush_l2(&mut self) -> Option<usize> {
        self.l2_cache.as_mut().map(|l2| l2.flush())
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
            unimplemented!("atomic memory operation");
        }
        match fetch.access_kind() {
            // writeback accesses not counted
            AccessKind::L2_WRBK_ACC | AccessKind::L1_WRBK_ACC => None,
            _ => Some(fetch),
        }
    }

    pub fn top(&mut self) -> Option<&mem_fetch::MemFetch> {
        use super::AccessKind;
        if let Some(AccessKind::L2_WRBK_ACC | AccessKind::L1_WRBK_ACC) = self
            .l2_to_interconn_queue
            .first()
            .map(ported::mem_fetch::MemFetch::access_kind)
        {
            self.l2_to_interconn_queue.dequeue();
            // self.request_tracker.remove(fetch);
            return None;
        }

        self.l2_to_interconn_queue.first()
    }

    pub fn set_done(&mut self, fetch: &mem_fetch::MemFetch) {
        self.request_tracker.remove(fetch);
    }

    pub fn dram_l2_queue_push(&mut self, _fetch: &mem_fetch::MemFetch) {
        todo!("mem sub partition: dram l2 queue push");
    }

    pub fn dram_l2_queue_full(&self) -> bool {
        todo!("mem sub partition: dram l2 queue full");
    }

    pub fn cache_cycle(&mut self, cycle: u64) {
        use config::CacheWriteAllocatePolicy;
        use mem_fetch::{AccessKind, Status};

        let log_line = style(format!(
            " => memory sub partition[{}] cache cycle {}",
            self.id, cycle
        ))
        .blue();

        log::debug!(
            "{}: rop queue={:?}, icnt to l2 queue={}, l2 to icnt queue={}, l2 to dram queue={}",
            log_line,
            self.rop_queue
                .iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>(),
            self.interconn_to_l2_queue,
            self.l2_to_interconn_queue,
            self.l2_to_dram_queue.lock().unwrap(),
        );

        // L2 fill responses
        if let Some(ref mut l2_cache) = self.l2_cache {
            let queue_full = self.l2_to_interconn_queue.full();

            log::debug!(
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
            if l2_cache.has_ready_accesses() && !queue_full {
                let mut fetch = l2_cache.next_access().unwrap();

                // Don't pass write allocate read request back to upper level cache
                if fetch.access_kind() != &AccessKind::L2_WR_ALLOC_R {
                    fetch.set_reply();
                    fetch.set_status(Status::IN_PARTITION_L2_TO_ICNT_QUEUE, 0);
                    // m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
                    self.l2_to_interconn_queue.enqueue(fetch);
                } else if l2_config.inner.write_allocate_policy
                    == CacheWriteAllocatePolicy::FETCH_ON_WRITE
                {
                    let mut original_write_fetch = *fetch.original_fetch.unwrap();
                    original_write_fetch.set_reply();
                    original_write_fetch
                        .set_status(mem_fetch::Status::IN_PARTITION_L2_TO_ICNT_QUEUE, 0);
                    // m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
                    self.l2_to_interconn_queue.enqueue(original_write_fetch);
                    todo!("fetch on write: l2 to icnt queue");
                }
            }
        }

        let time = cycle + self.memcpy_cycle_offset;

        // DRAM to L2 (texture) and icnt (not texture)
        if let Some(reply) = self.dram_to_l2_queue.first() {
            match self.l2_cache {
                Some(ref mut l2_cache) if l2_cache.waiting_for_fill(reply) => {
                    if l2_cache.has_free_fill_port() {
                        let mut reply = self.dram_to_l2_queue.dequeue().unwrap();
                        log::debug!("filling L2 with {}", &reply);
                        reply.set_status(mem_fetch::Status::IN_PARTITION_L2_FILL_QUEUE, 0);
                        l2_cache.fill(reply, time);
                        // reply will be gone forever at this point
                        // m_dram_L2_queue->pop();
                    } else {
                        log::debug!("skip filling L2 with {}: no free fill port", &reply);
                    }
                }
                _ if !self.l2_to_interconn_queue.full() => {
                    let mut reply = self.dram_to_l2_queue.dequeue().unwrap();
                    if reply.is_write() && reply.kind == mem_fetch::Kind::WRITE_ACK {
                        reply.set_status(mem_fetch::Status::IN_PARTITION_L2_TO_ICNT_QUEUE, 0);
                    }
                    // m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
                    log::debug!("pushing {} to interconn queue", &reply);
                    self.l2_to_interconn_queue.enqueue(reply);
                }
                _ => {
                    log::debug!(
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
            if let Some(fetch) = self.interconn_to_l2_queue.first() {
                if let Some(ref mut l2_cache) = self.l2_cache {
                    if !self.config.data_cache_l2_texture_only || fetch.is_texture() {
                        // L2 is enabled and access is for L2
                        let output_full = self.l2_to_interconn_queue.full();
                        let port_free = l2_cache.has_free_data_port();

                        if !output_full && port_free {
                            let mut events = Vec::new();
                            let status =
                                l2_cache.access(fetch.addr(), fetch.clone(), &mut events, time);
                            let write_sent = was_write_sent(&events);
                            let read_sent = was_read_sent(&events);
                            log::debug!(
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
                                } else {
                                    assert!(write_sent);
                                }
                            } else if status != cache::RequestStatus::RESERVATION_FAIL {
                                // L2 cache accepted request
                                let mut fetch = self.interconn_to_l2_queue.dequeue().unwrap();
                                let wa_policy = l2_cache.write_allocate_policy();
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
                                        self.l2_to_interconn_queue.enqueue(fetch);
                                        panic!("l2 to interconn queue push");
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
                log::debug!("{}: {fetch}", style("POP FROM ROP").red());
                fetch.set_status(mem_fetch::Status::IN_PARTITION_ICNT_TO_L2_QUEUE, 0);
                // m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
                self.interconn_to_l2_queue.enqueue(fetch);
            }
        }
    }
}
