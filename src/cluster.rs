use super::{config, interconn as ic, mem_fetch, Core, MockSimulator};
use console::style;
use crossbeam::utils::CachePadded;

use std::collections::VecDeque;

use crate::sync::{atomic, Arc, Mutex, RwLock};

#[derive(Debug)]
pub struct Cluster<I> {
    pub cluster_id: usize,
    pub warp_instruction_unique_uid: Arc<CachePadded<atomic::AtomicU64>>,
    pub cores: Vec<Arc<RwLock<Core<I>>>>,
    pub config: Arc<config::GPU>,
    pub stats: Arc<Mutex<stats::Stats>>,

    pub interconn: Arc<I>,

    pub core_sim_order: Arc<Mutex<VecDeque<usize>>>,
    pub block_issue_next_core: Mutex<usize>,
    pub response_fifo: VecDeque<mem_fetch::MemFetch>,
}

impl<I> Cluster<I>
where
    I: ic::Interconnect<ic::Packet<mem_fetch::MemFetch>>,
{
    pub fn new(
        cluster_id: usize,
        warp_instruction_unique_uid: &Arc<CachePadded<atomic::AtomicU64>>,
        allocations: &super::allocation::Ref,
        interconn: &Arc<I>,
        stats: &Arc<Mutex<stats::Stats>>,
        config: &Arc<config::GPU>,
    ) -> Self {
        let num_cores = config.num_cores_per_simt_cluster;
        let block_issue_next_core = num_cores - 1;
        let core_sim_order = (0..num_cores).collect();
        let cores = (0..num_cores)
            .map(|core_id| {
                let id = config.global_core_id(cluster_id, core_id);
                let core = Core::new(
                    id,
                    cluster_id,
                    Arc::clone(allocations),
                    Arc::clone(warp_instruction_unique_uid),
                    Arc::clone(interconn),
                    Arc::clone(stats),
                    Arc::clone(config),
                );
                Arc::new(RwLock::new(core))
            })
            .collect();
        let mut cluster = Self {
            cluster_id,
            warp_instruction_unique_uid: Arc::clone(warp_instruction_unique_uid),
            config: config.clone(),
            stats: stats.clone(),
            interconn: interconn.clone(),
            cores,
            core_sim_order: Arc::new(Mutex::new(core_sim_order)),
            block_issue_next_core: Mutex::new(block_issue_next_core),
            response_fifo: VecDeque::new(),
        };
        cluster.reinit();
        cluster
    }

    fn reinit(&mut self) {
        for core in &self.cores {
            core.write()
                .reinit(0, self.config.max_threads_per_core, true);
        }
    }

    pub fn num_active_sms(&self) -> usize {
        self.cores
            .iter()
            .filter(|core| core.try_read().is_active())
            .count()
    }

    pub fn not_completed(&self) -> usize {
        self.cores
            .iter()
            .map(|core| core.try_read().not_completed())
            .sum()
    }

    #[tracing::instrument]
    pub fn interconn_cycle(&mut self, cycle: u64) {
        use mem_fetch::AccessKind;

        log::debug!(
            "{}",
            style(format!(
                "cycle {:02} cluster {}: interconn cycle (response fifo={:?})",
                cycle,
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

            // we should not fully lock a core as we completely block a full core cycle
            let core = self.cores[core_id].read();

            match *fetch.access_kind() {
                AccessKind::INST_ACC_R => {
                    // this could be the reason
                    if core.fetch_unit_response_buffer_full() {
                        log::debug!("instr access fetch {} NOT YET ACCEPTED", fetch);
                    } else {
                        let fetch = self.response_fifo.pop_front().unwrap();
                        log::debug!("accepted instr access fetch {}", fetch);
                        core.accept_fetch_response(fetch, cycle);
                    }
                }
                _ if !core.ldst_unit_response_buffer_full() => {
                    let fetch = self.response_fifo.pop_front().unwrap();
                    log::debug!("accepted ldst unit fetch {}", fetch);
                    // m_memory_stats->memlatstat_read_done(mf);
                    core.accept_ldst_unit_response(fetch, cycle);
                }
                _ => {
                    log::debug!("ldst unit fetch {} NOT YET ACCEPTED", fetch);
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

        let Some(packet) = self.interconn.pop(self.cluster_id) else {
            return;
        };
        let mut fetch = packet.data;
        log::debug!(
            "{}",
            style(format!(
                "cycle {:02} cluster {}: got fetch from interconn: {}",
                cycle, self.cluster_id, fetch,
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
        // let _packet_size = if fetch.is_write() {
        //     fetch.control_size()
        // } else {
        //     fetch.data_size()
        // };
        // m_stats->m_incoming_traffic_stats->record_traffic(mf, packet_size);
        fetch.status = mem_fetch::Status::IN_CLUSTER_TO_SHADER_QUEUE;
        self.response_fifo.push_back(fetch);

        // m_stats->n_mem_to_simt[m_cluster_id] += mf->get_num_flits(false);
    }

    pub fn cache_flush(&self) {
        for core in &self.cores {
            core.write().cache_flush();
        }
    }

    pub fn cache_invalidate(&self) {
        for core in &self.cores {
            core.write().cache_invalidate();
        }
    }

    // pub fn cycle(&mut self) {
    //     log::debug!("cluster {} cycle {}", self.cluster_id, self.cycle.get());
    //     for core_id in &self.core_sim_order {
    //         self.cores[*core_id]lock().cycle();
    //     }
    //
    //     if let config::SchedulingOrder::RoundRobin = self.config.simt_core_sim_order {
    //         self.core_sim_order.rotate_left(1);
    //     }
    // }

    #[tracing::instrument(name = "cluster_issue_block_to_core")]
    pub fn issue_block_to_core(&self, sim: &MockSimulator<I>, cycle: u64) -> usize {
        let num_cores = self.cores.len();

        log::debug!(
            "cluster {}: issue block to core for {} cores",
            self.cluster_id,
            num_cores
        );
        let mut num_blocks_issued = 0;

        let mut block_issue_next_core = self.block_issue_next_core.try_lock();

        for core_id in 0..num_cores {
            let core_id = (core_id + *block_issue_next_core + 1) % num_cores;
            let core = self.cores[core_id].read();

            // let kernel: Option<Arc<Kernel>> = if self.config.concurrent_kernel_sm {
            //     // always select latest issued kernel
            //     // kernel = sim.select_kernel()
            //     // sim.select_kernel().map(Arc::clone);
            //     unimplemented!("concurrent kernel sm");
            // } else {
            let mut current_kernel: Option<Arc<_>> =
                core.current_kernel.try_lock().as_ref().map(Arc::clone);
            let should_select_new_kernel = if let Some(ref current) = current_kernel {
                // if no more blocks left, get new kernel once current block completes
                current.no_more_blocks_to_run() && core.not_completed() == 0
            } else {
                // core was not assigned a kernel yet
                true
            };

            if should_select_new_kernel {
                current_kernel = sim.select_kernel();
            }

            if let Some(kernel) = current_kernel {
                log::debug!(
                    "core {}-{}: selected kernel {} more blocks={} can issue={}",
                    self.cluster_id,
                    core_id,
                    kernel,
                    !kernel.no_more_blocks_to_run(),
                    core.can_issue_block(&kernel),
                );

                let can_issue = !kernel.no_more_blocks_to_run() && core.can_issue_block(&kernel);
                drop(core);
                if can_issue {
                    let mut core = self.cores[core_id].write();
                    core.issue_block(&kernel, cycle);
                    num_blocks_issued += 1;
                    *block_issue_next_core = core_id;
                    break;
                }
            } else {
                log::debug!(
                    "core {}-{}: selected kernel NULL",
                    self.cluster_id,
                    core.core_id,
                );
            }
        }
        num_blocks_issued
    }
}
