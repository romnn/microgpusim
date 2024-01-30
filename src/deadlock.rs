use crate::{interconn as ic, mem_fetch, register_set};

#[derive(Debug, PartialEq, Eq)]
pub struct State {
    pub interconn_to_l2_queue: Vec<Vec<mem_fetch::MemFetch>>,
    pub l2_to_interconn_queue: Vec<Vec<mem_fetch::MemFetch>>,
    pub l2_to_dram_queue: Vec<Vec<mem_fetch::MemFetch>>,
    pub dram_to_l2_queue: Vec<Vec<mem_fetch::MemFetch>>,
    pub dram_latency_queue: Vec<Vec<(u64, mem_fetch::MemFetch)>>,
    pub functional_unit_pipelines: Vec<Vec<register_set::RegisterSet>>,
    // pub operand_collectors: Vec<Option<operand_collector::OperandCollectorRegisterFileUnit>>,
    // pub schedulers: Vec<Vec<sched::Scheduler>>,
    // functional_unit_pipelines
    // schedulers
    // operand_collectors
}

impl State {
    #[must_use]
    pub fn new(total_cores: usize, num_mem_partitions: usize, num_sub_partitions: usize) -> Self {
        Self {
            // per sub partition
            interconn_to_l2_queue: vec![vec![]; num_sub_partitions],
            l2_to_interconn_queue: vec![vec![]; num_sub_partitions],
            l2_to_dram_queue: vec![vec![]; num_sub_partitions],
            dram_to_l2_queue: vec![vec![]; num_sub_partitions],
            // per partition
            dram_latency_queue: vec![vec![]; num_mem_partitions],
            // per core
            functional_unit_pipelines: vec![vec![]; total_cores],
            // operand_collectors: vec![None; total_cores],
            // schedulers: vec![vec![]; total_cores],
        }
    }
}

impl<I, MC> super::MockSimulator<I, MC>
where
    I: ic::Interconnect<ic::Packet<mem_fetch::MemFetch>> + 'static,
{
    pub fn gather_state(&self) -> State {
        let total_cores = self.config.total_cores();
        let num_partitions = self.mem_partition_units.len();
        let num_sub_partitions = self.mem_sub_partitions.len();

        let mut state = State::new(total_cores, num_partitions, num_sub_partitions);

        for (cluster_id, cluster) in self.clusters.iter().enumerate() {
            // let cluster = cluster.try_read();
            for (core_id, core) in cluster.cores.iter().enumerate() {
                let core = core.try_read();
                let global_core_id = cluster_id * self.config.num_cores_per_simt_cluster + core_id;
                assert_eq!(core.core_id, global_core_id);

                // this is the one we will use (unless the assertion is ever false)
                let core_id = core.core_id;

                // core: functional units
                for (fu_id, fu) in core.functional_units.iter().enumerate() {
                    // let issue_port = core.issue_ports[fu_id];
                    let issue_port = fu.issue_port();
                    let issue_reg: register_set::RegisterSet =
                        core.pipeline_reg[issue_port as usize].try_lock().clone();
                    assert_eq!(issue_port, issue_reg.stage);

                    state.functional_unit_pipelines[core_id].push(issue_reg);
                }
                // core: operand collector
                // state.operand_collectors[core_id] =
                //     Some(core.inner.operand_collector.borrow().clone());
                // core: schedulers
                // state.schedulers[core_id].extend(core.schedulers.iter().map(Into::into));
            }
        }
        for (partition_id, partition) in self.mem_partition_units.iter().enumerate() {
            state.dram_latency_queue[partition_id]
                .extend(partition.try_read().dram_latency_queue.clone().into_iter());
        }
        for (sub_id, sub) in self.mem_sub_partitions.iter().enumerate() {
            let sub = sub.try_lock();
            for (dest_queue, src_queue) in [
                (
                    &mut state.interconn_to_l2_queue[sub_id],
                    &sub.interconn_to_l2_queue,
                ),
                (
                    &mut state.l2_to_interconn_queue[sub_id],
                    &sub.l2_to_interconn_queue,
                ),
                (
                    &mut state.l2_to_dram_queue[sub_id],
                    &sub.l2_to_dram_queue.try_lock(),
                ),
                (&mut state.dram_to_l2_queue[sub_id], &sub.dram_to_l2_queue),
            ] {
                dest_queue.extend(src_queue.clone().into_iter().map(ic::Packet::into_inner));
            }
        }
        state
    }
}
