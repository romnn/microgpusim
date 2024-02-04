use super::{config, fifo::Fifo, interconn as ic, mem_fetch, Core};
use crate::sync::{atomic, Arc, Mutex, RwLock};
use console::style;
use crossbeam::utils::CachePadded;
use std::collections::VecDeque;

// pub trait IssueBlock {}

// pub struct BlockIssuer {
//     pub block_issue_next_core: usize,
//     // pub block_issue_next_core: Mutex<usize>,
// }
//
// impl BlockIssuer {
//     pub fn new(num_cores: usize) -> Self {
//         Self {
//             block_issue_next_core: num_cores - 1,
//         }
//     }
// }

pub type ResponseQueue = Arc<Mutex<Fifo<ic::Packet<mem_fetch::MemFetch>>>>;

#[derive()]
pub struct Cluster<I, MC> {
    pub cluster_id: usize,

    // pub warp_instruction_unique_uid: Arc<CachePadded<atomic::AtomicU64>>,
    // pub cores: Vec<Arc<RwLock<Core<I, MC>>>>,
    // pub cores: Vec<Core<I, MC>>>,
    // pub cores: Box<[Core<I, MC>]>,
    pub cores: Box<[Arc<RwLock<Core<I, MC>>>]>,

    // queues going to the cores
    pub core_instr_fetch_response_queue: Box<[ResponseQueue]>,
    pub core_load_store_response_queue: Box<[ResponseQueue]>,

    pub config: Arc<config::GPU>,
    pub interconn: Arc<I>,

    pub core_sim_order: Arc<Mutex<VecDeque<usize>>>,
    // pub issuer: BlockIssuer,
    // pub block_issue_next_core: Mutex<usize>,
    pub block_issue_next_core: usize,

    // pub response_fifo: RwLock<VecDeque<mem_fetch::MemFetch>>,
    pub response_fifo: VecDeque<mem_fetch::MemFetch>,
}

impl<I, MC> std::fmt::Debug for Cluster<I, MC> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Cluster@{}", self.cluster_id)
    }
}

impl<I, MC> Cluster<I, MC>
where
    I: ic::Interconnect<ic::Packet<mem_fetch::MemFetch>>,
    MC: crate::mcu::MemoryController,
{
    pub fn new(
        cluster_id: usize,
        warp_instruction_unique_uid: &Arc<CachePadded<atomic::AtomicU64>>,
        allocations: &Arc<super::allocation::Allocations>,
        interconn: &Arc<I>,
        config: &Arc<config::GPU>,
        mem_controller: &Arc<MC>,
    ) -> Self {
        let num_cores_per_cluster = config.num_cores_per_simt_cluster;
        let core_sim_order = (0..num_cores_per_cluster).collect();

        let core_instr_fetch_response_queue: Box<[ResponseQueue]> = (0..num_cores_per_cluster)
            .map(|_| Arc::new(Mutex::new(Fifo::new(None))))
            .collect();

        let core_load_store_response_queue: Box<[ResponseQueue]> = (0..num_cores_per_cluster)
            .map(|_| Arc::new(Mutex::new(Fifo::new(None))))
            .collect();

        let cores = (0..num_cores_per_cluster)
            .map(|local_core_id| {
                let global_core_id = config.global_core_id(cluster_id, local_core_id);
                let core = Core::new(
                    global_core_id,
                    local_core_id,
                    cluster_id,
                    Arc::clone(&core_instr_fetch_response_queue[local_core_id]),
                    Arc::clone(&core_load_store_response_queue[local_core_id]),
                    Arc::clone(allocations),
                    Arc::clone(warp_instruction_unique_uid),
                    Arc::clone(interconn),
                    Arc::clone(config),
                    Arc::clone(mem_controller),
                );
                // core
                Arc::new(RwLock::new(core))
            })
            .collect();

        let block_issue_next_core = num_cores_per_cluster - 1;
        // let issuer = BlockIssuer::new(num_cores);

        let response_fifo = VecDeque::new();
        // let response_fifo = RwLock::new(response_fifo);
        let mut cluster = Self {
            cluster_id,
            // warp_instruction_unique_uid: Arc::clone(warp_instruction_unique_uid),
            config: config.clone(),
            interconn: interconn.clone(),
            cores,
            core_instr_fetch_response_queue,
            core_load_store_response_queue,
            core_sim_order: Arc::new(Mutex::new(core_sim_order)),
            // issuer,
            block_issue_next_core,
            // block_issue_next_core: Mutex::new(block_issue_next_core),
            response_fifo,
        };
        cluster.reinit();
        cluster
    }

    fn reinit(&self) {
        for core in self.cores.iter() {
            // core.write()
            core.try_write()
                .reinit(0, self.config.max_threads_per_core, true);
        }
    }

    pub fn num_active_sms(&self) -> usize {
        self.cores
            .iter()
            .filter(|core| core.try_read().is_active())
            // .filter(|core| core.try_read().is_active())
            .count()
    }

    pub fn num_active_threads(&self) -> usize {
        self.cores
            .iter()
            .map(|core| core.try_read().num_active_threads())
            // .map(|core| core.try_read().num_active_threads())
            .sum()
    }

    #[tracing::instrument]
    pub fn interconn_cycle(&mut self, cycle: u64) {
        use mem_fetch::access::Kind as AccessKind;

        let response_fifo = &mut self.response_fifo;
        // let mut response_fifo = self.response_fifo.write();

        log::debug!(
            "{}",
            style(format!(
                "cycle {:02} cluster {}: interconn cycle (response fifo={:?})",
                cycle,
                self.cluster_id,
                response_fifo
                    .iter()
                    .map(std::string::ToString::to_string)
                    .collect::<Vec<_>>(),
            ))
            .cyan()
        );

        // Handle received package
        if let Some(fetch) = response_fifo.front() {
            // let core_id = self
            //     .config
            //     .global_core_id_to_core_id(fetch.core_id.unwrap());
            let global_core_id = fetch.global_core_id.unwrap();
            // let local_core_id = fetch.global_core_id.unwrap();
            let (_, local_core_id) = self.config.cluster_and_local_core_id(global_core_id);

            // we should not fully lock a core as we completely block a full core cycle
            // let core = self.cores[core_id].read();

            // let core = &self.cores[core_id];
            // assert_eq!(core.cluster_id, self.cluster_id);
            // assert_eq!((self.cluster_id, core_id), core.id());

            log::debug!(
                "have fetch {} for core {:?} ({}): ldst unit response buffer full={}",
                fetch,
                (self.cluster_id, local_core_id),
                // core.id(),
                global_core_id,
                // core.ldst_unit_response_buffer_full(),
                // core.load_store_unit.response_queue.lock().full(),
                self.core_load_store_response_queue[local_core_id]
                    .lock()
                    .full(),
            );

            match fetch.access_kind() {
                AccessKind::INST_ACC_R => {
                    // forward instruction fetch response to core
                    // if core.fetch_unit_response_buffer_full() {
                    //     log::debug!("instr access fetch {} NOT YET ACCEPTED", fetch);
                    // } else {
                    //     let fetch = response_fifo.pop_front().unwrap();
                    //     log::debug!("accepted instr access fetch {}", fetch);
                    //     core.accept_fetch_response(fetch, cycle);
                    // }
                    // let mut instr_fetch_response_queue = core.instr_fetch_response_queue.lock();
                    let mut instr_fetch_response_queue =
                        self.core_instr_fetch_response_queue[local_core_id].lock();
                    if instr_fetch_response_queue.full() {
                        log::debug!("instr access fetch {} NOT YET ACCEPTED", fetch);
                    } else {
                        let fetch = response_fifo.pop_front().unwrap();
                        log::debug!("accepted instr access fetch {}", fetch);
                        instr_fetch_response_queue.enqueue(ic::Packet { fetch, time: cycle });
                    }
                }
                _ => {
                    // if !core.ldst_unit_response_buffer_full() {
                    //     // Forward load store unit response to core
                    //     let fetch = response_fifo.pop_front().unwrap();
                    //     log::debug!("accepted ldst unit fetch {}", fetch);
                    //     // m_memory_stats->memlatstat_read_done(mf);
                    //     core.accept_ldst_unit_response(fetch, cycle);
                    // } else {
                    //     log::debug!("ldst unit fetch {} NOT YET ACCEPTED", fetch);
                    // }
                    // let mut load_store_response_queue = core.load_store_unit.response_queue.lock();
                    let mut load_store_response_queue =
                        self.core_load_store_response_queue[local_core_id].lock();
                    if !load_store_response_queue.full() {
                        // Forward load store unit response to core
                        let fetch = response_fifo.pop_front().unwrap();
                        log::debug!("accepted ldst unit fetch {}", fetch);
                        // m_memory_stats->memlatstat_read_done(mf);
                        // core.accept_ldst_unit_response(fetch, cycle);
                        load_store_response_queue.enqueue(ic::Packet { fetch, time: cycle });
                    } else {
                        log::debug!("ldst unit fetch {} NOT YET ACCEPTED", fetch);
                    }
                }
            }
        }

        let eject_buffer_size = self.config.num_cluster_ejection_buffer_size;
        if response_fifo.len() >= eject_buffer_size {
            log::debug!(
                "skip: ejection buffer full ({}/{})",
                response_fifo.len(),
                eject_buffer_size
            );
            return;
        }

        // Receive a packet from interconnect
        let Some(ic::Packet { mut fetch, .. }) = self.interconn.pop(self.cluster_id) else {
            return;
        };
        log::debug!(
            "{}",
            style(format!(
                "cycle {:02} cluster {}: got fetch from interconn: {}",
                cycle, self.cluster_id, fetch,
            ))
            .cyan()
        );

        debug_assert_eq!(fetch.cluster_id, Some(self.cluster_id));

        fetch.status = mem_fetch::Status::IN_CLUSTER_TO_SHADER_QUEUE;
        response_fifo.push_back(fetch);
    }

    pub fn cache_flush(&mut self) {
        for core in self.cores.iter() {
            core.try_write().cache_flush();
            // core.write().cache_flush();
        }
    }

    // pub fn is_cache_flushed(&self) {
    //     for core in &self.cores {
    //         core.read().flushed();
    //     }
    // }

    pub fn cache_invalidate(&mut self) {
        for core in self.cores.iter() {
            core.try_write().cache_invalidate();
            // core.write().cache_invalidate();
        }
    }

    #[tracing::instrument(name = "cluster_issue_block_to_core")]
    // pub fn issue_block_to_core(&mut self, sim: &mut Simulator<I, MC>, cycle: u64) -> usize
    pub fn issue_block_to_core_deterministic(
        &mut self,
        kernel_manager: &mut dyn crate::kernel_manager::SelectKernel,
        cycle: u64,
    ) -> usize
    where
        MC: std::fmt::Debug + crate::mcu::MemoryController,
    {
        let num_cores = self.cores.len();

        log::debug!(
            "cluster {}: issue block to core for {} cores",
            self.cluster_id,
            num_cores
        );
        let mut num_blocks_issued = 0;

        // let mut block_issue_next_core = self.block_issue_next_core.try_lock();
        // let block_issue_next_core = self.block_issue_next_core;

        for core_id in 0..num_cores {
            let core_id = (core_id + self.block_issue_next_core + 1) % num_cores;
            let core = &self.cores[core_id];
            let issued = core.try_write().issue_block(kernel_manager, cycle);
            if issued {
                num_blocks_issued += 1;
                self.block_issue_next_core = core_id;
                break;
            }
        }

        //     // let core = self.cores[core_id].read();
        //
        //     // let kernel: Option<Arc<Kernel>> = if self.config.concurrent_kernel_sm {
        //     //     // always select latest issued kernel
        //     //     // kernel = sim.select_kernel()
        //     //     // sim.select_kernel().map(Arc::clone);
        //     //     unimplemented!("concurrent kernel sm");
        //     // } else {
        //     // let mut current_kernel: Option<Arc<_>> = core.current_kernel.try_lock().as_ref().map(Arc::clone);
        //
        //     // let mut current_kernel = core.current_kernel.as_ref(); // .map(Arc::clone);
        //     // let current_kernel = &mut core.current_kernel; // .map(Arc::clone);
        //
        //     let should_select_new_kernel = if let Some(ref current) = core.current_kernel {
        //         // if no more blocks left, get new kernel once current block completes
        //         current.no_more_blocks_to_run() && core.num_active_threads() == 0
        //     } else {
        //         // core was not assigned a kernel yet
        //         true
        //     };
        //
        //     if should_select_new_kernel {
        //         core.current_kernel = kernel_manager.select_kernel();
        //     } else {
        //     }
        //
        //     // let mut new_kernel = None;
        //     // if should_select_new_kernel {
        //     //     new_kernel = kernel_manager.select_kernel();
        //     // }
        //     // if should_select_new_kernel {
        //     //     current_kernel = new_kernel.as_ref();
        //     // }
        //
        //     if let Some(kernel) = core.current_kernel.as_deref() {
        //         log::debug!(
        //             "core {}-{}: selected kernel {} more blocks={} can issue={}",
        //             self.cluster_id,
        //             core_id,
        //             kernel,
        //             !kernel.no_more_blocks_to_run(),
        //             core.can_issue_block(&*kernel),
        //         );
        //
        //         let can_issue = !kernel.no_more_blocks_to_run() && core.can_issue_block(&*kernel);
        //         // drop(core);
        //         if can_issue {
        //             // let mut core = self.cores[core_id].write();
        //             // let core = &mut self.cores[core_id];
        //             core.issue_block(&*kernel, cycle);
        //             num_blocks_issued += 1;
        //             self.block_issue_next_core = core_id;
        //             break;
        //         }
        //     } else {
        //         log::debug!(
        //             "core {}-{}: selected kernel NULL",
        //             self.cluster_id,
        //             core.core_id,
        //         );
        //     }
        // }
        num_blocks_issued
    }
}

#[cfg(test)]
pub mod tests {
    #[test]
    fn test_global_to_local_core_id() {
        assert_eq!(4 % 4, 0);
        assert_eq!(5 % 4, 1);
        assert_eq!(1 % 4, 1);
        assert_eq!(0 % 4, 0);
        assert_eq!(8 % 4, 0);
    }
}
