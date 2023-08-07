use super::{
    address, config, dram,
    fifo::{FifoQueue, Queue},
    mem_fetch,
    mem_fetch::BitString,
    mem_sub_partition::MemorySubPartition,
    Cycle,
};
use console::style;
// use std::cell::RefCell;
use std::collections::VecDeque;
// use std::rc::Rc;
use std::sync::{Arc, Mutex};

#[derive()]
pub struct MemoryPartitionUnit {
    id: usize,
    dram: dram::DRAM,
    pub dram_latency_queue: VecDeque<mem_fetch::MemFetch>,
    // pub sub_partitions: Vec<Rc<RefCell<MemorySubPartition<FifoQueue<mem_fetch::MemFetch>>>>>,
    pub sub_partitions: Vec<Arc<Mutex<MemorySubPartition<FifoQueue<mem_fetch::MemFetch>>>>>,
    pub arbitration_metadata: super::arbitration::ArbitrationMetadata,

    config: Arc<config::GPUConfig>,
    #[allow(dead_code)]
    stats: Arc<Mutex<stats::Stats>>,
}

impl MemoryPartitionUnit {
    pub fn new(
        id: usize,
        cycle: Cycle,
        config: Arc<config::GPUConfig>,
        stats: Arc<Mutex<stats::Stats>>,
    ) -> Self {
        let num_sub_partitions = config.num_sub_partition_per_memory_channel;
        let sub_partitions: Vec<_> = (0..num_sub_partitions)
            .map(|i| {
                let sub_id = id * num_sub_partitions + i;

                // Rc::new(RefCell::new(MemorySubPartition::new(
                Arc::new(Mutex::new(MemorySubPartition::new(
                    sub_id,
                    id,
                    cycle.clone(),
                    Arc::clone(&config),
                    Arc::clone(&stats),
                )))
            })
            .collect();

        let dram = dram::DRAM::new(config.clone(), stats.clone());
        let arbitration_metadata = super::arbitration::ArbitrationMetadata::new(&config);
        Self {
            id,
            config,
            stats,
            dram,
            dram_latency_queue: VecDeque::new(),
            arbitration_metadata,
            sub_partitions,
        }
    }

    #[must_use]
    pub fn busy(&self) -> bool {
        self.sub_partitions
            .iter()
            .any(|sub| sub.try_lock().unwrap().busy())
        // .any(|sub| sub.try_borrow().unwrap().busy())
    }

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
        time: u64,
    ) {
        let local_subpart_id = self.global_sub_partition_id_to_local_id(global_subpart_id);

        log::trace!(
            "copy engine request received for address={}, local_subpart={}, global_subpart={}, sector_mask={}",
            addr, local_subpart_id, global_subpart_id, mask.to_bit_string());

        self.sub_partitions[local_subpart_id]
            // .borrow_mut()
            .lock()
            .unwrap()
            .force_l2_tag_update(addr, mask, time);
    }

    pub fn cache_cycle(&mut self, cycle: u64) {
        for mem_sub in &mut self.sub_partitions {
            // mem_sub.borrow_mut().cache_cycle(cycle);
            mem_sub.try_lock().unwrap().cache_cycle(cycle);
        }
    }

    pub fn set_done(&mut self, fetch: mem_fetch::MemFetch) {
        let global_spid = fetch.sub_partition_id();
        let spid = self.global_sub_partition_id_to_local_id(global_spid);
        // let mut sub = self.sub_partitions[spid].try_borrow_mut().unwrap();
        let mut sub = self.sub_partitions[spid].try_lock().unwrap();
        debug_assert_eq!(sub.id, global_spid);
        if matches!(
            fetch.access_kind(),
            mem_fetch::AccessKind::L1_WRBK_ACC | mem_fetch::AccessKind::L2_WRBK_ACC
        ) {
            self.arbitration_metadata.return_credit(spid);
            log::trace!(
                "mem_fetch request {} return from dram to sub partition {}",
                fetch,
                spid
            );
        }
        sub.set_done(&fetch);
    }

    pub fn simple_dram_cycle(&mut self) {
        log::debug!("{} ...", style("simple dram cycle").red());
        // pop completed memory request from dram and push it to dram-to-L2 queue
        // of the original sub partition
        // if !self.dram_latency_queue.is_empty() &&
        //     ((m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) >=
        //      m_dram_latency_queue.front().ready_cycle)) {
        if let Some(returned_fetch) = self.dram_latency_queue.front_mut() {
            if !matches!(
                returned_fetch.access_kind(),
                mem_fetch::AccessKind::L1_WRBK_ACC | mem_fetch::AccessKind::L2_WRBK_ACC
            ) {
                self.dram.access(returned_fetch);

                returned_fetch.set_reply(); // todo: is it okay to do that here?
                log::debug!(
                    "got {} fetch return from dram latency queue (write={})",
                    returned_fetch,
                    returned_fetch.is_write()
                );

                let dest_global_spid = returned_fetch.sub_partition_id();
                let dest_spid = self.global_sub_partition_id_to_local_id(dest_global_spid);
                // let mut sub = self.sub_partitions[dest_spid].borrow_mut();
                let mut sub = self.sub_partitions[dest_spid].try_lock().unwrap();
                debug_assert_eq!(sub.id, dest_global_spid);

                if !sub.dram_to_l2_queue.full() {
                    // here we could set reply
                    let mut returned_fetch = self.dram_latency_queue.pop_front().unwrap();
                    // dbg!(&returned_fetch);
                    // returned_fetch.set_reply();

                    if returned_fetch.access_kind() == &mem_fetch::AccessKind::L1_WRBK_ACC {
                        sub.set_done(&returned_fetch);
                    } else {
                        returned_fetch
                            .set_status(mem_fetch::Status::IN_PARTITION_DRAM_TO_L2_QUEUE, 0);
                        // m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
                        self.arbitration_metadata.return_credit(dest_spid);
                        // log::debug!(
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
                log::debug!(
                    "DROPPING {} fetch return from dram latency queue (write={})",
                    returned_fetch,
                    returned_fetch.is_write()
                );

                let returned_fetch = self.dram_latency_queue.pop_front().unwrap();
                self.set_done(returned_fetch);
            }
        }

        // L2->DRAM queue to DRAM latency queue
        // Arbitrate among multiple L2 subpartitions
        let last_issued_partition = self.arbitration_metadata.last_borrower();
        for sub_id in 0..self.sub_partitions.len() {
            let spid = (sub_id + last_issued_partition + 1) % self.sub_partitions.len();
            // let sub = self.sub_partitions[spid].borrow_mut();
            let sub = self.sub_partitions[spid].try_lock().unwrap();

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
                    .map(std::string::ToString::to_string)
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
                let mut l2_to_dram_queue = sub.l2_to_dram_queue.lock().unwrap();
                if let Some(fetch) = l2_to_dram_queue.first() {
                    if self.dram.full(fetch.is_write()) {
                        break;
                    }

                    let mut fetch = l2_to_dram_queue.dequeue().unwrap();
                    log::debug!(
                        "simple dram: issue {} from sub partition {} to DRAM",
                        &fetch,
                        sub.id
                    );
                    // log::debug!(
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
}
