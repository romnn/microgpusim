use super::{config, interconn as ic, mem_fetch, Core};
use crate::sync::{atomic, Arc, Mutex};
use console::style;
use ic::SharedConnection;
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

pub type ResponseQueue = ic::shared::UnboundedChannel<ic::Packet<mem_fetch::MemFetch>>;

#[derive()]
pub struct Cluster<I, MC> {
    pub cluster_id: usize,

    pub cores: Box<[Arc<Mutex<Core<I, MC>>>]>,

    // queues going to the cores
    pub core_instr_fetch_response_queue: Box<[ResponseQueue]>,
    pub core_load_store_response_queue: Box<[ResponseQueue]>,

    pub config: Arc<config::GPU>,
    pub interconn: Arc<I>,

    pub core_sim_order: Arc<Mutex<VecDeque<usize>>>,
    // pub issuer: BlockIssuer,
    // pub block_issue_next_core: Mutex<usize>,
    pub block_issue_next_core: usize,

    pub response_fifo: VecDeque<mem_fetch::MemFetch>,

    /// Custom callback handler that is called when a fetch is
    /// returned to its issuer.
    pub fetch_return_callback: Option<Box<dyn Fn(u64, &mem_fetch::MemFetch) + Send + Sync>>,
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
        warp_instruction_unique_uid: &Arc<atomic::AtomicU64>,
        allocations: &Arc<super::allocation::Allocations>,
        interconn: &Arc<I>,
        config: &Arc<config::GPU>,
        mem_controller: &Arc<MC>,
    ) -> Self {
        let num_cores_per_cluster = config.num_cores_per_simt_cluster;
        let core_sim_order = (0..num_cores_per_cluster).collect();

        let core_instr_fetch_response_queue: Box<[ResponseQueue]> = (0..num_cores_per_cluster)
            .map(|_| ResponseQueue::default())
            .collect();

        let core_load_store_response_queue: Box<[ResponseQueue]> = (0..num_cores_per_cluster)
            .map(|_| ResponseQueue::default())
            .collect();

        let cores = (0..num_cores_per_cluster)
            .map(|local_core_id| {
                let global_core_id = config.global_core_id(cluster_id, local_core_id);
                let core = Core::new(
                    global_core_id,
                    local_core_id,
                    cluster_id,
                    core_instr_fetch_response_queue[local_core_id].clone(),
                    core_load_store_response_queue[local_core_id].clone(),
                    Arc::clone(allocations),
                    Arc::clone(warp_instruction_unique_uid),
                    Arc::clone(interconn),
                    Arc::clone(config),
                    Arc::clone(mem_controller),
                );
                Arc::new(Mutex::new(core))
            })
            .collect();

        let block_issue_next_core = num_cores_per_cluster - 1;
        // let issuer = BlockIssuer::new(num_cores);

        let response_fifo = VecDeque::new();
        let cluster = Self {
            cluster_id,
            config: config.clone(),
            interconn: interconn.clone(),
            cores,
            core_instr_fetch_response_queue,
            core_load_store_response_queue,
            core_sim_order: Arc::new(Mutex::new(core_sim_order)),
            block_issue_next_core,
            response_fifo,
            fetch_return_callback: None,
        };
        cluster.reinit();
        cluster
    }

    fn reinit(&self) {
        for core in self.cores.iter() {
            core.try_lock()
                .reinit(0, self.config.max_threads_per_core, true);
        }
    }

    pub fn num_active_sms(&self) -> usize {
        self.cores
            .iter()
            .filter(|core| core.try_lock().is_active())
            .count()
    }

    pub fn num_active_threads(&self) -> usize {
        self.cores
            .iter()
            .map(|core| core.try_lock().num_active_threads())
            .sum()
    }

    #[tracing::instrument]
    pub fn interconn_cycle(&mut self, cycle: u64) {
        use mem_fetch::access::Kind as AccessKind;

        let response_fifo = &mut self.response_fifo;

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
        if let Some(fetch) = response_fifo.pop_front() {
            let global_core_id = fetch.global_core_id.unwrap();
            let (_, local_core_id) = self.config.cluster_and_local_core_id(global_core_id);

            log::debug!(
                "have fetch {} for core {:?} ({}): ldst unit response buffer full={}",
                fetch,
                (self.cluster_id, local_core_id),
                global_core_id,
                false,
            );

            if let Some(fetch_return_cb) = &self.fetch_return_callback {
                fetch_return_cb(cycle, &fetch);
            }

            match fetch.access_kind() {
                AccessKind::INST_ACC_R => {
                    let instr_fetch_response_queue =
                        &self.core_instr_fetch_response_queue[local_core_id];

                    match instr_fetch_response_queue.try_send(ic::Packet { fetch, time: cycle }) {
                        Ok(_) => {
                            log::debug!("core accepted instr fetch");
                        }
                        Err(rejected) => {
                            log::debug!("instr access fetch {} NOT YET ACCEPTED", rejected.fetch);
                            response_fifo.push_front(rejected.fetch)
                        }
                    }
                }
                _ => {
                    let load_store_response_queue =
                        &self.core_load_store_response_queue[local_core_id];

                    match load_store_response_queue.try_send(ic::Packet { fetch, time: cycle }) {
                        Ok(_) => {
                            log::debug!("core accepted load store unit fetch");
                        }
                        Err(rejected) => {
                            log::debug!("ldst unit fetch {} NOT YET ACCEPTED", rejected.fetch);
                            response_fifo.push_front(rejected.fetch)
                        }
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
            core.try_lock().cache_flush();
        }
    }

    pub fn cache_invalidate(&mut self) {
        for core in self.cores.iter() {
            core.try_lock().cache_invalidate();
        }
    }

    #[tracing::instrument(name = "cluster_issue_block_to_core")]
    pub fn issue_block_to_core_deterministic<K>(
        &mut self,
        kernel_manager: &mut K,
        cycle: u64,
    ) -> usize
    where
        MC: std::fmt::Debug + crate::mcu::MemoryController,
        K: crate::kernel_manager::SelectKernel,
    {
        let num_cores = self.cores.len();

        log::debug!(
            "cluster {}: issue block to core for {} cores",
            self.cluster_id,
            num_cores
        );
        let mut num_blocks_issued = 0;

        for core_id in 0..num_cores {
            let core_id = (core_id + self.block_issue_next_core + 1) % num_cores;
            let core = &self.cores[core_id];
            let mut core = core.try_lock();
            let issued = core.maybe_issue_block(kernel_manager, cycle);
            if issued {
                num_blocks_issued += 1;
                self.block_issue_next_core = core_id;
                break;
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
