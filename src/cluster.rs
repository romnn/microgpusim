use super::{
    config, core, interconn as ic, kernel::Kernel, mem_fetch, MockSimulator, Packet, SIMTCore,
};
use console::style;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::{atomic, Arc, Mutex};

#[derive(Debug)]
pub struct SIMTCoreCluster<I> {
    pub cluster_id: usize,
    pub cycle: super::Cycle,
    pub warp_instruction_unique_uid: Arc<atomic::AtomicU64>,
    pub cores: Mutex<Vec<SIMTCore<I>>>,
    pub config: Arc<config::GPUConfig>,
    pub stats: Arc<Mutex<stats::Stats>>,

    pub interconn: Arc<I>,

    pub core_sim_order: VecDeque<usize>,
    pub block_issue_next_core: Mutex<usize>,
    pub response_fifo: VecDeque<mem_fetch::MemFetch>,
}

impl<I> SIMTCoreCluster<I>
where
    I: ic::Interconnect<Packet> + 'static,
{
    pub fn new(
        cluster_id: usize,
        cycle: super::Cycle,
        warp_instruction_unique_uid: Arc<atomic::AtomicU64>,
        allocations: Rc<RefCell<super::Allocations>>,
        interconn: Arc<I>,
        stats: Arc<Mutex<stats::Stats>>,
        config: Arc<config::GPUConfig>,
    ) -> Self {
        let num_cores = config.num_cores_per_simt_cluster;
        let block_issue_next_core = Mutex::new(num_cores - 1);
        let mut cluster = Self {
            cluster_id,
            cycle: Rc::clone(&cycle),
            warp_instruction_unique_uid: Arc::clone(&warp_instruction_unique_uid),
            config: config.clone(),
            stats: stats.clone(),
            interconn: interconn.clone(),
            cores: Mutex::new(Vec::new()),
            core_sim_order: VecDeque::new(),
            block_issue_next_core,
            response_fifo: VecDeque::new(),
        };
        let cores = (0..num_cores)
            .map(|core_id| {
                cluster.core_sim_order.push_back(core_id);
                let id = config.global_core_id(cluster_id, core_id);
                SIMTCore::new(
                    id,
                    cluster_id,
                    Rc::clone(&allocations),
                    Rc::clone(&cycle),
                    Arc::clone(&warp_instruction_unique_uid),
                    Arc::clone(&interconn),
                    Arc::clone(&stats),
                    Arc::clone(&config),
                )
            })
            .collect();
        cluster.cores = Mutex::new(cores);
        cluster.reinit();
        cluster
    }

    fn reinit(&mut self) {
        for core in self.cores.lock().unwrap().iter_mut() {
            core.reinit(0, self.config.max_threads_per_core, true);
        }
    }

    pub fn num_active_sms(&self) -> usize {
        self.cores
            .lock()
            .unwrap()
            .iter()
            .filter(|c| c.active())
            .count()
    }

    pub fn not_completed(&self) -> usize {
        self.cores
            .lock()
            .unwrap()
            .iter()
            .map(core::SIMTCore::not_completed)
            .sum()
    }

    pub fn warp_waiting_at_barrier(&self, _warp_id: usize) -> bool {
        todo!("cluster: warp_waiting_at_barrier");
        // self.barriers.warp_waiting_at_barrier(warp_id)
    }

    pub fn warp_waiting_at_mem_barrier(&self, _warp_id: usize) -> bool {
        todo!("cluster: warp_waiting_at_mem_barrier");
        // if (!m_warp[warp_id]->get_membar()) return false;
        // if (!m_scoreboard->pendingWrites(warp_id)) {
        //   m_warp[warp_id]->clear_membar();
        //   if (m_gpu->get_config().flush_l1()) {
        //     // Mahmoud fixed this on Nov 2019
        //     // Invalidate L1 cache
        //     // Based on Nvidia Doc, at MEM barrier, we have to
        //     //(1) wait for all pending writes till they are acked
        //     //(2) invalidate L1 cache to ensure coherence and avoid reading stall data
        //     cache_invalidate();
        //     // TO DO: you need to stall the SM for 5k cycles.
        //   }
        //   return false;
        // }
        // return true;
    }

    pub fn interconn_cycle(&mut self) {
        use mem_fetch::AccessKind;

        log::debug!(
            "{}",
            style(format!(
                "cycle {:02} cluster {}: interconn cycle (response fifo={:?})",
                self.cycle.get(),
                self.cluster_id,
                self.response_fifo
                    .iter()
                    .map(std::string::ToString::to_string)
                    .collect::<Vec<_>>(),
            ))
            .cyan()
        );

        if let Some(fetch) = self.response_fifo.front() {
            let core_id = self.config.global_core_id_to_core_id(fetch.core_id);

            let mut cores = self.cores.lock().unwrap();
            let core = &mut cores[core_id];

            match *fetch.access_kind() {
                AccessKind::INST_ACC_R => {
                    // this could be the reason
                    if !core.fetch_unit_response_buffer_full() {
                        let fetch = self.response_fifo.pop_front().unwrap();
                        log::debug!("accepted instr access fetch {}", fetch);
                        core.accept_fetch_response(fetch);
                    } else {
                        log::debug!("instr access fetch {} NOT YET ACCEPTED", fetch);
                    }
                }
                _ => {
                    // this could be the reason
                    if !core.ldst_unit_response_buffer_full() {
                        let fetch = self.response_fifo.pop_front().unwrap();
                        log::debug!("accepted ldst unit fetch {}", fetch);
                        // m_memory_stats->memlatstat_read_done(mf);
                        core.accept_ldst_unit_response(fetch);
                    } else {
                        log::debug!("ldst unit fetch {} NOT YET ACCEPTED", fetch);
                    }
                }
            }
        }

        // this could be the reason?
        let eject_buffer_size = self.config.num_cluster_ejection_buffer_size;
        if self.response_fifo.len() >= eject_buffer_size {
            log::debug!(
                "skip: ejection buffer full ({}/{})",
                self.response_fifo.len(),
                eject_buffer_size
            );
            return;
        }

        let Some(Packet::Fetch(mut fetch)) = self.interconn.pop(self.cluster_id) else {
            return;
        };
        log::debug!(
            "{}",
            style(format!(
                "cycle {:02} cluster {}: got fetch from interconn: {}",
                self.cycle.get(),
                self.cluster_id,
                fetch,
            ))
            .cyan()
        );

        debug_assert_eq!(fetch.cluster_id, self.cluster_id);
        // debug_assert!(matches!(
        //     fetch.kind,
        //     mem_fetch::Kind::READ_REPLY | mem_fetch::Kind::WRITE_ACK
        // ));

        // The packet size varies depending on the type of request:
        // - For read request and atomic request, the packet contains the data
        // - For write-ack, the packet only has control metadata
        let _packet_size = if fetch.is_write() {
            fetch.control_size
        } else {
            fetch.data_size
        };
        // m_stats->m_incoming_traffic_stats->record_traffic(mf, packet_size);
        fetch.status = mem_fetch::Status::IN_CLUSTER_TO_SHADER_QUEUE;
        self.response_fifo.push_back(fetch);

        // m_stats->n_mem_to_simt[m_cluster_id] += mf->get_num_flits(false);
    }

    pub fn cache_flush(&mut self) {
        let mut cores = self.cores.lock().unwrap();
        for core in cores.iter_mut() {
            core.cache_flush();
        }
    }

    pub fn cache_invalidate(&mut self) {
        let mut cores = self.cores.lock().unwrap();
        for core in cores.iter_mut() {
            core.cache_invalidate();
        }
    }

    pub fn cycle(&mut self) {
        log::debug!("cluster {} cycle {}", self.cluster_id, self.cycle.get());
        let mut cores = self.cores.lock().unwrap();

        for core_id in &self.core_sim_order {
            cores[*core_id].cycle()
        }

        if let config::SchedulingOrder::RoundRobin = self.config.simt_core_sim_order {
            self.core_sim_order.rotate_left(1);
            // let first = self.core_sim_order.pop_front().unwrap();
            // self.core_sim_order.push_back(first);
        }
    }

    pub fn issue_block_to_core(&self, sim: &MockSimulator<I>) -> usize {
        let mut cores = self.cores.lock().unwrap();
        let num_cores = cores.len();

        log::debug!(
            "cluster {}: issue block to core for {} cores",
            self.cluster_id,
            num_cores
        );
        let mut num_blocks_issued = 0;

        let mut block_issue_next_core = self.block_issue_next_core.lock().unwrap();

        for core_id in 0..num_cores {
            let core_id = (core_id + *block_issue_next_core + 1) % num_cores;
            let core = &mut cores[core_id];
            let kernel: Option<Arc<Kernel>> = if self.config.concurrent_kernel_sm {
                // always select latest issued kernel
                // kernel = sim.select_kernel()
                // sim.select_kernel().map(Arc::clone);
                unimplemented!("concurrent kernel sm");
            } else {
                let mut current_kernel = core.inner.current_kernel.as_ref();
                let should_select_new_kernel = if let Some(current) = current_kernel {
                    // if no more blocks left, get new kernel once current block completes
                    current.no_more_blocks_to_run() && core.not_completed() == 0
                } else {
                    // core was not assigned a kernel yet
                    true
                };

                if let Some(current) = current_kernel {
                    log::debug!(
                        "core {}-{}: current kernel {}, more blocks={}, completed={}",
                        self.cluster_id,
                        core_id,
                        current,
                        !current.no_more_blocks_to_run(),
                        core.not_completed() == 0,
                    );
                }

                if should_select_new_kernel {
                    current_kernel = sim.select_kernel();
                    if let Some(k) = current_kernel {
                        core.set_kernel(Arc::clone(k));
                    }
                }

                current_kernel.map(Arc::clone)
            };
            if let Some(kernel) = kernel {
                log::debug!(
                    "core {}-{}: selected kernel {} more blocks={} can issue={}",
                    self.cluster_id,
                    core_id,
                    kernel,
                    !kernel.no_more_blocks_to_run(),
                    core.can_issue_block(&kernel),
                );

                if !kernel.no_more_blocks_to_run() && core.can_issue_block(&kernel) {
                    core.issue_block(kernel);
                    num_blocks_issued += 1;
                    *block_issue_next_core = core_id;
                    break;
                }
            } else {
                log::debug!(
                    "core {}-{}: selected kernel NULL",
                    self.cluster_id,
                    core.inner.core_id,
                );
            }
        }
        num_blocks_issued
    }
}
