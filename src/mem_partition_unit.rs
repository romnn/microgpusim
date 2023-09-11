use crate::sync::{Arc, Mutex};
use crate::{
    address, arbitration, config, dram, ic::Packet, mem_fetch,
    mem_sub_partition::MemorySubPartition,
};
use console::style;
use mem_fetch::ToBitString;
use std::collections::VecDeque;

pub struct MemoryPartitionUnit {
    id: usize,
    dram: dram::DRAM,
    pub dram_latency_queue: VecDeque<(u64, mem_fetch::MemFetch)>,
    pub sub_partitions: Vec<Arc<Mutex<MemorySubPartition>>>,
    pub arbiter: Box<dyn arbitration::Arbiter>,

    config: Arc<config::GPU>,
    #[allow(dead_code)]
    stats: Arc<Mutex<stats::PerKernel>>,
}

impl std::fmt::Debug for MemoryPartitionUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryPartitionUnit").finish()
    }
}

impl MemoryPartitionUnit {
    pub fn new(id: usize, config: Arc<config::GPU>, stats: Arc<Mutex<stats::PerKernel>>) -> Self {
        let num_sub_partitions = config.num_sub_partitions_per_memory_controller;
        let sub_partitions: Vec<_> = (0..num_sub_partitions)
            .map(|i| {
                let sub_id = id * num_sub_partitions + i;

                Arc::new(Mutex::new(MemorySubPartition::new(
                    sub_id,
                    id,
                    Arc::clone(&config),
                    Arc::clone(&stats),
                )))
            })
            .collect();

        let dram = dram::DRAM::new(config.clone(), stats.clone());
        let arb_config: arbitration::Config = (&(*config)).into();
        let arbiter = Box::new(arbitration::ArbitrationUnit::new(&arb_config));
        Self {
            id,
            config,
            stats,
            dram,
            dram_latency_queue: VecDeque::new(),
            arbiter,
            sub_partitions,
        }
    }

    #[must_use]
    #[inline]
    pub fn busy(&self) -> bool {
        self.sub_partitions.iter().any(|sub| sub.try_lock().busy())
    }

    #[must_use]
    #[inline]
    fn global_sub_partition_id_to_local_id(&self, global_sub_partition_id: usize) -> usize {
        let mut local_id = global_sub_partition_id;
        local_id -= self.id * self.config.num_sub_partitions_per_memory_controller;
        local_id
    }

    #[inline]
    pub fn handle_memcpy_to_gpu(
        &self,
        addr: address,
        global_subpart_id: usize,
        sector_mask: &mem_fetch::SectorMask,
        time: u64,
    ) {
        let local_subpart_id = self.global_sub_partition_id_to_local_id(global_subpart_id);

        log::trace!(
            "copy engine request received for address={}, local_subpart={}, global_subpart={}, sector_mask={}",
            addr, local_subpart_id, global_subpart_id, sector_mask.to_bit_string());

        self.sub_partitions[local_subpart_id]
            .lock()
            .force_l2_tag_update(addr, sector_mask, time);
    }

    #[inline]
    pub fn cache_cycle(&mut self, cycle: u64) {
        for mem_sub in &mut self.sub_partitions {
            mem_sub.try_lock().cache_cycle(cycle);
        }
    }

    #[inline]
    pub fn set_done(&mut self, fetch: &mem_fetch::MemFetch) {
        use mem_fetch::access::Kind as AccessKind;
        let global_spid = fetch.sub_partition_id();
        let sub_partition_id = self.global_sub_partition_id_to_local_id(global_spid);
        let mut sub = self.sub_partitions[sub_partition_id].try_lock();
        debug_assert_eq!(sub.id, global_spid);
        if matches!(
            fetch.access_kind(),
            AccessKind::L1_WRBK_ACC | AccessKind::L2_WRBK_ACC
        ) {
            self.arbiter.return_credit(sub_partition_id);
            log::trace!(
                "mem_fetch request {} return from dram to sub partition {}",
                fetch,
                sub_partition_id
            );
        }
        sub.set_done(fetch);
    }

    #[tracing::instrument]
    pub fn simple_dram_cycle(&mut self, cycle: u64) {
        use mem_fetch::access::Kind as AccessKind;
        log::debug!("{} ...", style("simple dram cycle").red());
        // pop completed memory request from dram and push it to dram-to-L2 queue
        // of the original sub partition
        // if !self.dram_latency_queue.is_empty() &&
        //     ((m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) >=
        //      m_dram_latency_queue.front().ready_cycle)) {
        match self.dram_latency_queue.front_mut() {
            Some((ready_cycle, returned_fetch))
                if cycle >= *ready_cycle
                    && matches!(
                        returned_fetch.access_kind(),
                        AccessKind::L1_WRBK_ACC | AccessKind::L2_WRBK_ACC
                    ) =>
            {
                log::debug!(
                    "DROPPING {} fetch return from dram latency queue (write={})",
                    returned_fetch,
                    returned_fetch.is_write()
                );

                let (_, returned_fetch) = self.dram_latency_queue.pop_front().unwrap();
                self.set_done(&returned_fetch);
            }
            Some((ready_cycle, returned_fetch)) if cycle >= *ready_cycle => {
                self.dram.access(returned_fetch);

                returned_fetch.set_reply(); // todo: is it okay to do that here?
                log::debug!(
                    "got {} fetch return from dram latency queue (write={})",
                    returned_fetch,
                    returned_fetch.is_write()
                );

                let dest_global_spid = returned_fetch.sub_partition_id();
                let dest_spid = self.global_sub_partition_id_to_local_id(dest_global_spid);
                let mut sub = self.sub_partitions[dest_spid].try_lock();
                debug_assert_eq!(sub.id, dest_global_spid);

                // depending on which sub the fetch is for, we race for the sub

                // this is fine
                if sub.dram_to_l2_queue.full() {
                    // panic!("fyi: simple dram model stall");
                } else {
                    let (_, mut returned_fetch) = self.dram_latency_queue.pop_front().unwrap();
                    // dbg!(&returned_fetch);
                    // returned_fetch.set_reply();

                    if returned_fetch.access_kind() == AccessKind::L1_WRBK_ACC {
                        sub.set_done(&returned_fetch);
                    } else {
                        returned_fetch
                            .set_status(mem_fetch::Status::IN_PARTITION_DRAM_TO_L2_QUEUE, 0);
                        self.arbiter.return_credit(dest_spid);
                        // log::debug!(
                        //     "mem_fetch request {:?} return from dram to sub partition {}",
                        //     returned_fetch, dest_spid
                        // );

                        debug_assert!(returned_fetch.is_reply());
                        sub.dram_to_l2_queue.enqueue(Packet {
                            data: returned_fetch,
                            time: cycle,
                        });
                    }
                }
            }
            // not ready
            None | Some(_) => {}
        }

        // L2->DRAM queue to DRAM latency queue
        // Arbitrate among multiple L2 subpartitions
        let last_issued_partition = self.arbiter.last_borrower();
        for sub_id in 0..self.sub_partitions.len() {
            let spid = (sub_id + last_issued_partition + 1) % self.sub_partitions.len();
            let sub = self.sub_partitions[spid].try_lock();

            let sub_partition_contention = sub.dram_to_l2_queue.full();
            let has_dram_resource = self.arbiter.has_credits(spid);
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
                let l2_to_dram_queue = sub.l2_to_dram_queue.try_lock();
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
                    .map(|(_, fetch)| fetch.to_string())
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
            }

            if can_issue_to_dram {
                let mut l2_to_dram_queue = sub.l2_to_dram_queue.lock();
                if let Some(fetch) = l2_to_dram_queue.first() {
                    if self.dram.full(fetch.is_write()) {
                        break;
                    }

                    let mut fetch = l2_to_dram_queue.dequeue().unwrap().into_inner();
                    log::debug!(
                        "simple dram: issue {} from sub partition {} to DRAM",
                        &fetch,
                        sub.id
                    );
                    // log::debug!(
                    //     "issue mem_fetch request {:?} from sub partition {} to dram",
                    //     fetch, spid
                    // );
                    let ready_cycle = cycle + self.config.dram_latency as u64;
                    fetch.set_status(mem_fetch::Status::IN_PARTITION_DRAM_LATENCY_QUEUE, 0);
                    self.dram_latency_queue.push_back((ready_cycle, fetch));
                    self.arbiter.borrow_credit(spid);

                    // DRAM should only accept one request per cycle
                    break;
                }
            }
        }
    }
}
