use super::{config, interconn as ic, mem_fetch, Core, MockSimulator};
use crate::sync::{atomic, Arc, Mutex, RwLock};
use console::style;
use crossbeam::utils::CachePadded;
use std::collections::VecDeque;

#[derive()]
pub struct Cluster<I, MC> {
    pub cluster_id: usize,
    pub warp_instruction_unique_uid: Arc<CachePadded<atomic::AtomicU64>>,
    pub cores: Vec<Arc<RwLock<Core<I, MC>>>>,
    pub config: Arc<config::GPU>,
    pub interconn: Arc<I>,

    pub core_sim_order: Arc<Mutex<VecDeque<usize>>>,
    pub block_issue_next_core: Mutex<usize>,
    pub response_fifo: RwLock<VecDeque<mem_fetch::MemFetch>>,
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
        allocations: &super::allocation::Ref,
        interconn: &Arc<I>,
        config: &Arc<config::GPU>,
        mem_controller: &Arc<MC>,
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
                    Arc::clone(config),
                    Arc::clone(mem_controller),
                );
                Arc::new(RwLock::new(core))
            })
            .collect();
        let cluster = Self {
            cluster_id,
            warp_instruction_unique_uid: Arc::clone(warp_instruction_unique_uid),
            config: config.clone(),
            interconn: interconn.clone(),
            cores,
            core_sim_order: Arc::new(Mutex::new(core_sim_order)),
            block_issue_next_core: Mutex::new(block_issue_next_core),
            response_fifo: RwLock::new(VecDeque::new()),
        };
        cluster.reinit();
        cluster
    }

    fn reinit(&self) {
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
    pub fn interconn_cycle(&self, cycle: u64) {
        let mut response_fifo = self.response_fifo.write();
        use mem_fetch::access::Kind as AccessKind;

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
            let core_id = self
                .config
                .global_core_id_to_core_id(fetch.core_id.unwrap());

            // we should not fully lock a core as we completely block a full core cycle
            let core = self.cores[core_id].read();
            assert_eq!(core.cluster_id, self.cluster_id);

            log::debug!(
                "have fetch {} for core {:?} ({}): ldst unit response buffer full={}",
                fetch,
                core.id(),
                fetch.core_id.unwrap(),
                core.ldst_unit_response_buffer_full(),
            );

            match fetch.access_kind() {
                AccessKind::INST_ACC_R => {
                    // forward instruction fetch response to core
                    if core.fetch_unit_response_buffer_full() {
                        log::debug!("instr access fetch {} NOT YET ACCEPTED", fetch);
                    } else {
                        let fetch = response_fifo.pop_front().unwrap();
                        log::debug!("accepted instr access fetch {}", fetch);
                        core.accept_fetch_response(fetch, cycle);
                    }
                }
                _ if !core.ldst_unit_response_buffer_full() => {
                    // Forward load store unit response to core
                    let fetch = response_fifo.pop_front().unwrap();
                    log::debug!("accepted ldst unit fetch {}", fetch);
                    // m_memory_stats->memlatstat_read_done(mf);
                    core.accept_ldst_unit_response(fetch, cycle);
                }
                _ => {
                    log::debug!("ldst unit fetch {} NOT YET ACCEPTED", fetch);
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

        debug_assert_eq!(fetch.cluster_id, Some(self.cluster_id));

        fetch.status = mem_fetch::Status::IN_CLUSTER_TO_SHADER_QUEUE;
        response_fifo.push_back(fetch);
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

    #[tracing::instrument(name = "cluster_issue_block_to_core")]
    pub fn issue_block_to_core(&self, sim: &MockSimulator<I, MC>, cycle: u64) -> usize
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
                    core.can_issue_block(&*kernel),
                );

                let can_issue = !kernel.no_more_blocks_to_run() && core.can_issue_block(&*kernel);
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
