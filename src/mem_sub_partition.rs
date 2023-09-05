use crate::{address, cache, config, fifo::Fifo, interconn::Packet, mem_fetch};
use console::style;

use std::collections::{HashSet, VecDeque};

use crate::sync::{Arc, Mutex};

pub const MAX_MEMORY_ACCESS_SIZE: u32 = 128;

/// four sectors
pub const SECTOR_CHUNCK_SIZE: u32 = 4;

/// Sector size is 32 bytes width
pub const SECTOR_SIZE: u32 = 32;

// pub struct MemorySubPartition<Q = Fifo<mem_fetch::MemFetch>> {
pub struct MemorySubPartition {
    pub id: usize,
    pub partition_id: usize,
    pub config: Arc<config::GPU>,
    pub stats: Arc<Mutex<stats::Stats>>,

    /// queues
    pub interconn_to_l2_queue: Fifo<Packet<mem_fetch::MemFetch>>,
    // pub interconn_to_l2_queue: Box<dyn ic::Connection<ic::Packet<mem_fetch::MemFetch>>>,
    pub l2_to_dram_queue: Arc<Mutex<Fifo<Packet<mem_fetch::MemFetch>>>>,
    pub dram_to_l2_queue: Fifo<Packet<mem_fetch::MemFetch>>,
    /// L2 cache hit response queue
    pub l2_to_interconn_queue: Fifo<Packet<mem_fetch::MemFetch>>,
    rop_queue: VecDeque<(u64, mem_fetch::MemFetch)>,

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

// impl<Q> std::fmt::Debug for MemorySubPartition<Q> {
impl std::fmt::Debug for MemorySubPartition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemorySubPartition").finish()
    }
}

const NO_FETCHES: VecDeque<mem_fetch::MemFetch> = VecDeque::new();

impl MemorySubPartition
// impl<Q> MemorySubPartition<Q>
// where
//     Q: Queue<mem_fetch::MemFetch> + 'static,
{
    pub fn new(
        id: usize,
        partition_id: usize,
        config: Arc<config::GPU>,
        stats: Arc<Mutex<stats::Stats>>,
    ) -> Self {
        let interconn_to_l2_queue = Fifo::new(
            // "icnt-to-L2",
            Some(0),
            Some(config.dram_partition_queue_interconn_to_l2),
        );
        let l2_to_dram_queue = Arc::new(Mutex::new(Fifo::new(
            // "L2-to-dram",
            Some(0),
            Some(config.dram_partition_queue_l2_to_dram),
        )));
        let dram_to_l2_queue = Fifo::new(
            // "dram-to-L2",
            Some(0),
            Some(config.dram_partition_queue_dram_to_l2),
        );
        let l2_to_interconn_queue = Fifo::new(
            // "L2-to-icnt",
            Some(0),
            Some(config.dram_partition_queue_l2_to_interconn),
        );

        let l2_cache: Option<Box<dyn cache::Cache>> = match &config.data_cache_l2 {
            Some(l2_config) => {
                // let l2_mem_port = Arc::new(ic::L2Interface {
                //     l2_to_dram_queue: Arc::clone(&l2_to_dram_queue),
                // });

                let cache_stats = Arc::new(Mutex::new(stats::Cache::default()));
                let mut data_l2 = cache::DataL2::new(
                    format!("mem-sub-{}-{}", id, style("L2-CACHE").green()),
                    0, // core_id,
                    0, // cluster_id,
                    // Arc::clone(&l2_to_dram_queue),
                    cache_stats,
                    config.clone(),
                    l2_config.clone(),
                );
                data_l2.set_top_port(l2_to_dram_queue.clone());
                Some(Box::new(data_l2))
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
        sector_mask: &mem_fetch::SectorMask,
        time: u64,
    ) {
        if let Some(ref mut l2_cache) = self.l2_cache {
            l2_cache.force_tag_access(addr, time + self.memcpy_cycle_offset, sector_mask);
            self.memcpy_cycle_offset += 1;
        }
    }

    pub fn push(&mut self, fetch: mem_fetch::MemFetch, time: u64) {
        // m_stats->memlatstat_icnt2mem_pop(m_req);
        let mut requests = Vec::new();
        let l2_config = self.config.data_cache_l2.as_ref().unwrap();
        if l2_config.inner.kind == config::CacheKind::Sector {
            // requests.extend(self.breakdown_request_to_sector_requests(&fetch));
            unimplemented!("sector cache");
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
                fetch.status = mem_fetch::Status::IN_PARTITION_ICNT_TO_L2_QUEUE;
                self.interconn_to_l2_queue
                    .enqueue(Packet { data: fetch, time });
            } else {
                let ready_cycle = time + self.config.l2_rop_latency;
                fetch.status = mem_fetch::Status::IN_PARTITION_ROP_DELAY;
                log::debug!("{}: {fetch}", style("PUSH TO ROP").red());
                self.rop_queue.push_back((ready_cycle, fetch));
            }
        }
    }

    #[must_use]
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
        use mem_fetch::access::Kind as AccessKind;

        let fetch = self.l2_to_interconn_queue.dequeue()?.into_inner();
        self.request_tracker.remove(&fetch);
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
        use mem_fetch::access::Kind as AccessKind;
        if let Some(AccessKind::L2_WRBK_ACC | AccessKind::L1_WRBK_ACC) = self
            .l2_to_interconn_queue
            .first()
            .map(|packet| packet.data.access_kind())
        {
            let fetch = self.l2_to_interconn_queue.dequeue().unwrap();
            self.request_tracker.remove(&fetch);
            return None;
        }

        self.l2_to_interconn_queue.first().map(AsRef::as_ref)
    }

    pub fn set_done(&mut self, fetch: &mem_fetch::MemFetch) {
        self.request_tracker.remove(fetch);
    }

    #[tracing::instrument]
    pub fn cache_cycle(&mut self, cycle: u64) {
        use mem_fetch::{access::Kind as AccessKind, Status};

        let log_line = style(format!(
            " => memory sub partition[{}] cache cycle {}",
            self.id, cycle
        ))
        .blue();

        // log::debug!(
        //     "{}: rop queue={:?}, icnt to l2 queue={}, l2 to icnt queue={}, l2 to dram queue={}",
        //     log_line,
        //     self.rop_queue
        //         .iter()
        //         .map(|(ready_cycle, fetch)| (ready_cycle, fetch.to_string()))
        //         .collect::<Vec<_>>(),
        //     self.interconn_to_l2_queue,
        //     self.l2_to_interconn_queue,
        //     self.l2_to_dram_queue.try_lock(),
        // );

        // L2 fill responses
        if let Some(ref mut l2_cache) = self.l2_cache {
            let queue_full = self.l2_to_interconn_queue.full();

            log::debug!(
                "{}: l2 cache ready accesses={:?} l2 to icnt queue full={}",
                log_line,
                l2_cache
                    .ready_accesses()
                    .unwrap_or(&NO_FETCHES)
                    .iter()
                    .map(std::string::ToString::to_string)
                    .collect::<Vec<_>>(),
                queue_full,
            );

            // todo: move config into l2
            let l2_config = self.config.data_cache_l2.as_ref().unwrap();
            if l2_cache.has_ready_accesses() && !queue_full {
                let mut fetch = l2_cache.next_access().unwrap();

                // Don't pass write allocate read request back to top level cache
                if fetch.access_kind() != &AccessKind::L2_WR_ALLOC_R {
                    fetch.set_reply();
                    fetch.set_status(Status::IN_PARTITION_L2_TO_ICNT_QUEUE, 0);
                    // m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
                    self.l2_to_interconn_queue.enqueue(Packet {
                        data: fetch,
                        time: cycle,
                    });
                } else if l2_config.inner.write_allocate_policy
                    == cache::config::WriteAllocatePolicy::FETCH_ON_WRITE
                {
                    let mut original_write_fetch = *fetch.original_fetch.unwrap();
                    original_write_fetch.set_reply();
                    original_write_fetch
                        .set_status(mem_fetch::Status::IN_PARTITION_L2_TO_ICNT_QUEUE, 0);
                    // m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
                    self.l2_to_interconn_queue.enqueue(Packet {
                        data: original_write_fetch,
                        time: cycle,
                    });
                    todo!("fetch on write: l2 to icnt queue");
                }
            }
        }

        let mem_copy_time = cycle + self.memcpy_cycle_offset;

        // DRAM to L2 (texture) and icnt (not texture)
        if let Some(reply) = self.dram_to_l2_queue.first() {
            match self.l2_cache {
                Some(ref mut l2_cache) if l2_cache.waiting_for_fill(reply) => {
                    if l2_cache.has_free_fill_port() {
                        let mut reply = self.dram_to_l2_queue.dequeue().unwrap().into_inner();
                        log::debug!("filling L2 with {}", &reply);
                        reply.set_status(mem_fetch::Status::IN_PARTITION_L2_FILL_QUEUE, 0);
                        l2_cache.fill(reply, mem_copy_time);
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
            l2_cache.cycle(cycle);
        }

        // new L2 texture accesses and/or non-texture accesses
        let mut l2_to_dram_queue = self.l2_to_dram_queue.try_lock();
        if !l2_to_dram_queue.full() {
            if let Some(fetch) = self.interconn_to_l2_queue.first().map(Packet::as_ref) {
                if let Some(ref mut l2_cache) = self.l2_cache {
                    if !self.config.data_cache_l2_texture_only || fetch.is_texture() {
                        // L2 is enabled and access is for L2
                        let output_full = self.l2_to_interconn_queue.full();
                        let port_free = l2_cache.has_free_data_port();

                        if !output_full && port_free {
                            let mut events = Vec::new();
                            let status = l2_cache.access(
                                fetch.addr(),
                                fetch.clone(),
                                &mut events,
                                mem_copy_time,
                            );
                            let write_sent = cache::event::was_write_sent(&events);
                            let read_sent = cache::event::was_read_sent(&events);
                            log::debug!(
                                "probing L2 cache address={}, status={:?}",
                                fetch.addr(),
                                status
                            );

                            if status == cache::RequestStatus::HIT {
                                let mut fetch = self.interconn_to_l2_queue.dequeue().unwrap();
                                if write_sent {
                                    assert!(write_sent);
                                } else {
                                    // L2 cache replies
                                    assert!(!read_sent);
                                    if fetch.access_kind() == &mem_fetch::access::Kind::L1_WRBK_ACC
                                    {
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
                                }
                            } else if status != cache::RequestStatus::RESERVATION_FAIL {
                                // L2 cache accepted request
                                let mut fetch = self.interconn_to_l2_queue.dequeue().unwrap();
                                let wa_policy = l2_cache.write_allocate_policy();
                                let should_fetch = matches!(
                                    wa_policy,
                                    cache::config::WriteAllocatePolicy::FETCH_ON_WRITE
                                        | cache::config::WriteAllocatePolicy::LAZY_FETCH_ON_READ
                                );
                                if fetch.is_write()
                                    && should_fetch
                                    && !cache::event::was_writeallocate_sent(&events)
                                {
                                    if fetch.access_kind() == &mem_fetch::access::Kind::L1_WRBK_ACC
                                    {
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

                    l2_to_dram_queue.enqueue(fetch);
                }
            }
        }

        // rop delay queue
        // if (!m_rop.empty() && (cycle >= m_rop.front().ready_cycle) &&
        //     !m_icnt_L2_queue->full()) {
        if !self.interconn_to_l2_queue.full() {
            match self.rop_queue.front() {
                Some((ready_cycle, _)) if cycle >= *ready_cycle => {
                    let (_, mut fetch) = self.rop_queue.pop_front().unwrap();
                    log::debug!("{}: {fetch}", style("POP FROM ROP").red());
                    fetch.set_status(mem_fetch::Status::IN_PARTITION_ICNT_TO_L2_QUEUE, 0);
                    // m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
                    self.interconn_to_l2_queue.enqueue(Packet {
                        data: fetch,
                        time: cycle,
                    });
                }
                _ => {}
            }
        }
    }
}
