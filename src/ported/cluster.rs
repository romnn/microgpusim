use super::{interconn as ic, mem_fetch, stats::Stats, MockSimulator, Packet, SIMTCore};
use crate::config::GPUConfig;
use console::style;
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

#[derive(Debug)]
pub struct SIMTCoreCluster<I> {
    pub cluster_id: usize,
    pub cycle: super::Cycle,
    pub cores: Mutex<Vec<SIMTCore<I>>>,
    pub config: Arc<GPUConfig>,
    pub stats: Arc<Mutex<Stats>>,

    pub interconn: Arc<I>,

    pub core_sim_order: Vec<usize>,
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
        interconn: Arc<I>,
        stats: Arc<Mutex<Stats>>,
        config: Arc<GPUConfig>,
    ) -> Self {
        let num_cores = config.num_cores_per_simt_cluster;
        let block_issue_next_core = Mutex::new(num_cores - 1);
        let mut cluster = Self {
            cluster_id,
            cycle: Rc::clone(&cycle),
            config: config.clone(),
            stats: stats.clone(),
            interconn: interconn.clone(),
            cores: Mutex::new(Vec::new()),
            core_sim_order: Vec::new(),
            block_issue_next_core,
            response_fifo: VecDeque::new(),
        };
        let cores = (0..num_cores)
            .map(|core_id| {
                cluster.core_sim_order.push(core_id);
                let id = config.global_core_id(cluster_id, core_id);
                SIMTCore::new(
                    id,
                    cluster_id,
                    Rc::clone(&cycle),
                    interconn.clone(),
                    stats.clone(),
                    config.clone(),
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
            .map(|c| c.not_completed())
            .sum()
    }

    pub fn warp_waiting_at_barrier(&self, warp_id: usize) -> bool {
        todo!("cluster: warp_waiting_at_barrier");
        // self.barriers.warp_waiting_at_barrier(warp_id)
    }

    pub fn warp_waiting_at_mem_barrier(&self, warp_id: usize) -> bool {
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

        // println!(
        //     "cluster {}: {} (response fifo={})",
        //     self.cluster_id,
        //     style("interconn cycle").cyan(),
        //     self.response_fifo.len(),
        // );
        println!(
            "{}",
            style(format!(
                "cycle {:02} cluster {}: interconn cycle (response fifo={:?})",
                self.cycle.get(),
                self.cluster_id,
                self.response_fifo
                    .iter()
                    .map(|fetch| fetch.to_string())
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
                        println!("accepted instr access fetch {}", fetch);
                        core.accept_fetch_response(fetch);
                    } else {
                        println!("instr access fetch {} NOT YET ACCEPTED", fetch);
                    }
                }
                _ => {
                    // this could be the reason
                    if !core.ldst_unit_response_buffer_full() {
                        let fetch = self.response_fifo.pop_front().unwrap();
                        println!("accepted ldst unit fetch {}", fetch);
                        // m_memory_stats->memlatstat_read_done(mf);
                        core.accept_ldst_unit_response(fetch);
                    } else {
                        println!("ldst unit fetch {} NOT YET ACCEPTED", fetch);
                    }
                }
            }
        }

        // this could be the reason?
        let eject_buffer_size = self.config.num_cluster_ejection_buffer_size;
        if self.response_fifo.len() >= eject_buffer_size {
            println!(
                "skip: ejection buffer full ({}/{})",
                self.response_fifo.len(),
                eject_buffer_size
            );
            return;
        }

        let Some(Packet::Fetch(mut fetch)) = self.interconn.pop(self.cluster_id) else {
            return;
        };
        println!(
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
        let packet_size = if fetch.is_write() {
            fetch.control_size
        } else {
            fetch.data_size
        };
        // m_stats->m_incoming_traffic_stats->record_traffic(mf, packet_size);
        fetch.status = mem_fetch::Status::IN_CLUSTER_TO_SHADER_QUEUE;
        self.response_fifo.push_back(fetch.clone());

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
        let mut cores = self.cores.lock().unwrap();
        for core in cores.iter_mut() {
            core.cycle()
        }
    }

    pub fn issue_block_to_core(&self, sim: &MockSimulator<I>) -> usize {
        println!("cluster {}: issue block 2 core", self.cluster_id);
        let mut num_blocks_issued = 0;

        let mut block_issue_next_core = self.block_issue_next_core.lock().unwrap();
        let mut cores = self.cores.lock().unwrap();
        let num_cores = cores.len();
        // dbg!(&sim.select_kernel());

        for (i, core) in cores.iter_mut().enumerate() {
            // debug_assert_eq!(i, core.id);
            let core_id = (i + *block_issue_next_core + 1) % num_cores;
            let mut kernel = None;
            if self.config.concurrent_kernel_sm {
                // always select latest issued kernel
                kernel = sim.select_kernel()
            } else {
                if core
                    .inner
                    .current_kernel
                    .as_ref()
                    .map(|current| !current.no_more_blocks_to_run())
                    .unwrap_or(true)
                {
                    // wait until current kernel finishes
                    if core.inner.num_active_warps == 0 {
                        kernel = sim.select_kernel();
                        if let Some(k) = kernel {
                            core.set_kernel(k.clone());
                        }
                    }
                }
            }
            println!(
                "core {}-{}: current kernel {}",
                self.cluster_id,
                core.inner.core_id,
                &core.inner.current_kernel.is_some()
            );
            println!(
                "core {}-{}: selected kernel {:?}",
                self.cluster_id,
                core.inner.core_id,
                kernel.as_ref().map(|k| k.name())
            );
            if let Some(kernel) = kernel {
                // dbg!(&kernel.no_more_blocks_to_run());
                // dbg!(&core.can_issue_block(&*kernel));
                if !kernel.no_more_blocks_to_run() && core.can_issue_block(&*kernel) {
                    core.issue_block(kernel.clone());
                    num_blocks_issued += 1;
                    *block_issue_next_core = core_id;
                    break;
                }
            }
        }
        num_blocks_issued

        // pub fn id(&self) -> (usize, usize) {
        //         self.id,
        //         core.id,
        //
        // }
        //       unsigned num_blocks_issued = 0;
        // for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++) {
        //   unsigned core =
        //       (i + m_cta_issue_next_core + 1) % m_config->n_simt_cores_per_cluster;
        //
        //   kernel_info_t *kernel;
        //   // Jin: fetch kernel according to concurrent kernel setting
        //   if (m_config->gpgpu_concurrent_kernel_sm) {  // concurrent kernel on sm
        //     // always select latest issued kernel
        //     kernel_info_t *k = m_gpu->select_kernel();
        //     kernel = k;
        //   } else {
        //     // first select core kernel, if no more cta, get a new kernel
        //     // only when core completes
        //     kernel = m_core[core]->get_kernel();
        //     if (!m_gpu->kernel_more_cta_left(kernel)) {
        //       // wait till current kernel finishes
        //       if (m_core[core]->get_not_completed() == 0) {
        //         kernel_info_t *k = m_gpu->select_kernel();
        //         if (k) m_core[core]->set_kernel(k);
        //         kernel = k;
        //       }
        //     }
        //   }
        //
        //   if (m_gpu->kernel_more_cta_left(kernel) &&
        //       //            (m_core[core]->get_n_active_cta() <
        //       //            m_config->max_cta(*kernel)) ) {
        //       m_core[core]->can_issue_1block(*kernel)) {
        //     m_core[core]->issue_block2core(*kernel);
        //     num_blocks_issued++;
        //     m_cta_issue_next_core = core;
        //     break;
        //   }
        // }
        // return num_blocks_issued;
    }
}
