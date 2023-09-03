pub mod gto;
pub mod ordering;

use crate::sync::{Arc, Mutex, RwLock};
use crate::{
    config,
    core::{PipelineStage, WarpIssuer},
    opcodes::ArchOp,
    scoreboard::{self, Access},
    warp,
};
use console::style;
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ExecUnitKind {
    NONE = 0,
    SP = 1,
    #[allow(dead_code)]
    SFU = 2,
    MEM = 3,
    #[allow(dead_code)]
    DP = 4,
    INT = 5,
    #[allow(dead_code)]
    TENSOR = 6,
    #[allow(dead_code)]
    SPECIALIZED = 7,
}

pub trait Scheduler: Send + Sync + std::fmt::Debug + 'static {
    fn issue_to(&mut self, core: &dyn WarpIssuer, cycle: u64);

    fn add_supervised_warp(&mut self, warp: warp::Ref);

    fn prioritized_warps(&self) -> &VecDeque<(usize, warp::Ref)>;

    /// Order warps based on scheduling policy.
    fn order_warps(&mut self, core: &dyn WarpIssuer);
}

impl std::fmt::Debug for &dyn WarpIssuer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WarpIssuer").finish()
    }
}

#[derive(Debug)]
pub struct Base {
    id: usize,
    cluster_id: usize,
    core_id: usize,

    /// This is the prioritized warp list that is looped over each cycle to
    /// determine which warp gets to issue.
    next_cycle_prioritized_warps: VecDeque<(usize, warp::Ref)>,

    // Supervised warps keeps all warps this scheduler can arbitrate between.
    //
    // This is useful in systems where there is more than one warp scheduler.
    // In a single scheduler system, this is simply all the warps
    // assigned to this core.
    supervised_warps: VecDeque<warp::Ref>,
    warps: Vec<warp::Ref>,

    /// This is the iterator pointer to the last supervised warp issued
    last_supervised_issued_idx: usize,
    num_issued_last_cycle: usize,

    scoreboard: Arc<RwLock<scoreboard::Scoreboard>>,

    config: Arc<config::GPU>,
    stats: Arc<Mutex<stats::scheduler::Scheduler>>,
}

impl Base {
    pub fn new(
        id: usize,
        cluster_id: usize,
        core_id: usize,
        warps: Vec<warp::Ref>,
        scoreboard: Arc<RwLock<scoreboard::Scoreboard>>,
        stats: Arc<Mutex<stats::scheduler::Scheduler>>,
        config: Arc<config::GPU>,
    ) -> Self {
        Self {
            id,
            cluster_id,
            core_id,
            next_cycle_prioritized_warps: VecDeque::new(),
            supervised_warps: VecDeque::new(),
            last_supervised_issued_idx: 0,
            warps,
            num_issued_last_cycle: 0,
            stats,
            scoreboard,
            config,
        }
    }

    fn prioritized_warps(&self) -> &VecDeque<(usize, warp::Ref)> {
        &self.next_cycle_prioritized_warps
    }

    #[tracing::instrument(name = "scheduler_issue")]
    #[must_use]
    #[inline]
    fn issue(
        &self,
        warp: &mut warp::Warp,
        stage: PipelineStage,
        unit: ExecUnitKind,
        prev_issued_exec_unit: ExecUnitKind,
        core: &dyn WarpIssuer,
        cycle: u64,
    ) -> bool {
        let free_register = core.has_free_register(stage, self.id);
        let can_dual_issue =
            !self.config.dual_issue_only_to_different_exec_units || prev_issued_exec_unit != unit;

        if free_register && can_dual_issue {
            core.issue_warp(stage, warp, self.id, cycle).is_ok()
        } else {
            log::debug!("issue failed: no free mem port register");
            false
        }
    }

    fn issue_to(&mut self, core: &dyn WarpIssuer, cycle: u64) {
        log::debug!("{}: cycle", style("base scheduler").yellow());

        let mut valid_inst = false;
        let mut ready_inst = false;
        let mut issued_inst = false;

        for (next_warp_supervised_idx, next_warp_rc) in &self.next_cycle_prioritized_warps {
            // don't consider warps that are not yet valid
            let mut next_warp = next_warp_rc.try_lock();
            let (warp_id, dyn_warp_id) = (next_warp.warp_id, next_warp.dynamic_warp_id);

            if next_warp.done_exit() {
                continue;
            }
            let inst_count = next_warp.instruction_count();
            if inst_count == 0 {
                log::debug!("next warp: {:#?}", &next_warp);
            }
            // assert!(inst_count > 0);
            if inst_count > 1 {
                log::debug!(
                    "core[{}][{}] scheduler[{}]: \n\t => testing (warp_id={}, dynamic_warp_id={}, trace_pc={}, pc={:?}, ibuffer={:?}, {} instructions)",
                    self.cluster_id,
                    self.core_id,
                    self.id,
                    warp_id, dyn_warp_id,
                    next_warp.trace_pc,
                    next_warp.pc(),
                    next_warp.instr_buffer.iter().filter_map(Option::as_ref).map(|i| i.pc).collect::<Vec<_>>(), inst_count,
                );
            }
            let mut checked = 0;
            let mut num_issued = 0;

            let mut prev_issued_exec_unit = ExecUnitKind::NONE;
            let max_issue = self.config.max_instruction_issue_per_warp;
            // In this mode, we only allow dual issue to diff execution
            // units (as in Maxwell and Pascal)
            let dual_issue_only_to_different_exec_units =
                self.config.dual_issue_only_to_different_exec_units;

            if inst_count > 1 {
                if next_warp.ibuffer_empty() {
                    log::debug!(
                        "warp (warp_id={}, dynamic_warp_id={}) fails as ibuffer_empty",
                        warp_id,
                        dyn_warp_id
                    );
                }

                if next_warp.waiting()
                    || core.warp_waiting_at_barrier(warp_id)
                    || core.warp_waiting_at_mem_barrier(&mut next_warp)
                {
                    log::debug!(
                        "warp (warp_id={}, dynamic_warp_id={}) is waiting",
                        warp_id,
                        dyn_warp_id
                    );
                }
            }

            let warp = self.warps.get(warp_id).unwrap();

            // todo: what is the difference? why dont we just use next_warp?
            debug_assert!(Arc::ptr_eq(warp, next_warp_rc));
            drop(next_warp);

            let mut warp = warp.try_lock();

            while !(warp.waiting()
                || core.warp_waiting_at_barrier(warp_id)
                || core.warp_waiting_at_mem_barrier(&mut warp)
                || warp.ibuffer_empty())
                && checked < max_issue
                && checked <= num_issued
                && num_issued < max_issue
            {
                let mut warp_inst_issued = false;
                checked += 1;

                let Some(instr) = warp.ibuffer_peek() else {
                    continue;
                };
                log::debug!(
                    "Warp (warp_id={}, dynamic_warp_id={}) instruction buffer[{}] has valid instruction ({}, op={:?})",
                    warp_id, dyn_warp_id, warp.next, instr, instr.opcode.category
                );

                valid_inst = true;
                if self.scoreboard.try_read().has_collision(warp_id, instr) {
                    log::debug!(
                        "Warp (warp_id={}, dynamic_warp_id={}) {}",
                        warp_id,
                        dyn_warp_id,
                        style("fails scoreboard").yellow(),
                    );
                    continue;
                }

                log::debug!(
                    "Warp (warp_id={}, dynamic_warp_id={}) {}",
                    warp_id,
                    dyn_warp_id,
                    style("passes scoreboard").yellow(),
                );
                ready_inst = true;

                debug_assert!(warp.has_instr_in_pipeline());

                match instr.opcode.category {
                    ArchOp::LOAD_OP
                    | ArchOp::STORE_OP
                    | ArchOp::MEMORY_BARRIER_OP
                    | ArchOp::TENSOR_CORE_LOAD_OP
                    | ArchOp::TENSOR_CORE_STORE_OP => {
                        if self.issue(
                            &mut warp,
                            PipelineStage::ID_OC_MEM,
                            ExecUnitKind::MEM,
                            prev_issued_exec_unit,
                            core,
                            cycle,
                        ) {
                            num_issued += 1;
                            issued_inst = true;
                            warp_inst_issued = true;
                            prev_issued_exec_unit = ExecUnitKind::MEM;
                        }
                        // let mem_stage = PipelineStage::ID_OC_MEM;
                        //
                        // let free_register = core.has_free_register(mem_stage, self.id);
                        // let can_dual_issue = !dual_issue_only_to_different_exec_units
                        //     || prev_issued_exec_unit != ExecUnitKind::MEM;
                        //
                        // if free_register && can_dual_issue {
                        //     // warp.ibuffer_step();
                        //     // let instr = warp.ibuffer_take().unwrap();
                        //     // debug_assert_eq!(warp_id, warp.warp_id);
                        //     if core
                        //         .issue_warp(mem_stage, &mut warp, self.id)
                        //         // .issue_warp(mem_stage, warp_id, self.id)
                        //         // .issue_warp(mem_stage, &mut warp, instr, self.id)
                        //         .is_ok()
                        //     {
                        //         num_issued += 1;
                        //         issued_inst = true;
                        //         warp_inst_issued = true;
                        //         prev_issued_exec_unit = ExecUnitKind::MEM;
                        //     }
                        // } else {
                        //     log::debug!("issue failed: no free mem port register");
                        // }
                    }
                    op @ (ArchOp::NO_OP
                    | ArchOp::ALU_OP
                    | ArchOp::SP_OP
                    | ArchOp::INT_OP
                    | ArchOp::BARRIER_OP
                    | ArchOp::ALU_SFU_OP
                    | ArchOp::BRANCH_OP
                    | ArchOp::CALL_OPS
                    | ArchOp::RET_OPS
                    | ArchOp::EXIT_OPS) => {
                        // op => {
                        //     if op != ArchOp::TENSOR_CORE_OP
                        //         && op != ArchOp::SFU_OP
                        //         && op != ArchOp::DP_OP
                        //         && (op as usize) < opcodes::SPEC_UNIT_START_ID
                        //     {
                        let mut execute_on_sp = false;
                        let mut execute_on_int = false;

                        let sp_pipe_avail = self.config.num_sp_units > 0
                            && core.has_free_register(PipelineStage::ID_OC_SP, self.id);
                        let int_pipe_avail = self.config.num_int_units > 0
                            && core.has_free_register(PipelineStage::ID_OC_INT, self.id);

                        // if INT unit pipline exist, then execute ALU and INT
                        // operations on INT unit and SP-FPU on SP unit (like in Volta)
                        // if INT unit pipline does not exist, then execute all ALU, INT
                        // and SP operations on SP unit (as in Fermi, Pascal GPUs)
                        let int_can_dual_issue = !dual_issue_only_to_different_exec_units
                            || prev_issued_exec_unit != ExecUnitKind::INT;

                        let sp_can_dual_issue = !dual_issue_only_to_different_exec_units
                            || prev_issued_exec_unit != ExecUnitKind::SP;

                        if int_pipe_avail && op != ArchOp::SP_OP && int_can_dual_issue {
                            execute_on_int = true;
                        } else if sp_pipe_avail
                            && (self.config.num_int_units == 0
                                || (self.config.num_int_units > 0 && op == ArchOp::SP_OP))
                            && sp_can_dual_issue
                        {
                            execute_on_sp = true;
                        }

                        log::debug!(
                            "execute on INT={} execute on SP={}",
                            execute_on_int,
                            execute_on_sp
                        );

                        let issue_target = if execute_on_sp {
                            Some((PipelineStage::ID_OC_SP, ExecUnitKind::SP))
                        } else if execute_on_int {
                            Some((PipelineStage::ID_OC_INT, ExecUnitKind::INT))
                        } else {
                            None
                        };

                        if let Some((stage, unit)) = issue_target {
                            if self.issue(
                                &mut warp,
                                stage,
                                unit,
                                prev_issued_exec_unit,
                                core,
                                cycle,
                            ) {
                                num_issued += 1;
                                issued_inst = true;
                                warp_inst_issued = true;
                                prev_issued_exec_unit = unit;
                            }
                        }

                        // if let Some((stage, unit)) = issue_target {
                        //     // let instr = warp.ibuffer_take().unwrap();
                        //     // warp.ibuffer_step();
                        //     debug_assert_eq!(warp.warp_id, warp_id);
                        //     if core.issue_warp(stage, &mut warp, self.id).is_ok() {
                        //         // .issue_warp(stage, &mut warp, instr, self.id).is_ok() {
                        //         num_issued += 1;
                        //         issued_inst = true;
                        //         warp_inst_issued = true;
                        //         prev_issued_exec_unit = unit;
                        //     }
                        // }
                    }
                    op @ (ArchOp::DP_OP | ArchOp::SFU_OP) => {
                        let dp_can_dual_issue = !dual_issue_only_to_different_exec_units
                            || prev_issued_exec_unit != ExecUnitKind::DP;
                        let sfu_can_dual_issue = !dual_issue_only_to_different_exec_units
                            || prev_issued_exec_unit != ExecUnitKind::SFU;

                        let dp_pipe_avail = self.config.num_dp_units > 0
                            && core.has_free_register(PipelineStage::ID_OC_DP, self.id);

                        let sfu_pipe_avail = self.config.num_sfu_units > 0
                            && core.has_free_register(PipelineStage::ID_OC_SFU, self.id);

                        let issue_target = match op {
                            // case 2
                            ArchOp::DP_OP
                                if self.config.num_dp_units > 0
                                    && dp_can_dual_issue
                                    && dp_pipe_avail =>
                            {
                                Some((PipelineStage::ID_OC_DP, ExecUnitKind::DP))
                            }
                            // case 3
                            ArchOp::DP_OP
                                if self.config.num_dp_units == 0
                                    && sfu_can_dual_issue
                                    && sfu_pipe_avail =>
                            {
                                Some((PipelineStage::ID_OC_SFU, ExecUnitKind::SFU))
                            }
                            ArchOp::SFU_OP if sfu_can_dual_issue && sfu_pipe_avail => {
                                Some((PipelineStage::ID_OC_SFU, ExecUnitKind::SFU))
                            }
                            _ => None,
                        };
                        log::trace!(
                            "dp/sfu issue for {}: {:?} [DP: units={}, dual_issue={}, avail={}] [SFU: units={}, dual_issue={}, avail={}]",
                            instr,
                            issue_target,
                            self.config.num_dp_units,
                            dp_can_dual_issue,
                            dp_pipe_avail,
                            self.config.num_sfu_units,
                            sfu_can_dual_issue,
                            sfu_pipe_avail,
                        );

                        if let Some((stage, unit)) = issue_target {
                            if self.issue(
                                &mut warp,
                                stage,
                                unit,
                                prev_issued_exec_unit,
                                core,
                                cycle,
                            ) {
                                num_issued += 1;
                                issued_inst = true;
                                warp_inst_issued = true;
                                prev_issued_exec_unit = unit;
                            }
                        }
                    }
                    op @ ArchOp::TENSOR_CORE_OP => {
                        unimplemented!("op {:?} not implemented", op);
                    }
                    op @ (ArchOp::SPECIALIZED_UNIT_1_OP
                    | ArchOp::SPECIALIZED_UNIT_2_OP
                    | ArchOp::SPECIALIZED_UNIT_3_OP
                    | ArchOp::SPECIALIZED_UNIT_4_OP
                    | ArchOp::SPECIALIZED_UNIT_5_OP
                    | ArchOp::SPECIALIZED_UNIT_6_OP
                    | ArchOp::SPECIALIZED_UNIT_7_OP
                    | ArchOp::SPECIALIZED_UNIT_8_OP) => {
                        unimplemented!("op {:?} not implemented", op);
                    }
                }
                if warp_inst_issued {
                    log::debug!(
                        "Warp (warp_id={}, dynamic_warp_id={}) issued {} instructions",
                        warp_id,
                        dyn_warp_id,
                        num_issued
                    );
                    // ibuffer step is always called after ibuffer take
                    // warp.ibuffer_step();
                }
            }
            if num_issued > 0 {
                self.last_supervised_issued_idx = *next_warp_supervised_idx;
                self.num_issued_last_cycle = num_issued;
                #[cfg(feature = "stats")]
                {
                    let mut stats = self.stats.lock();
                    if num_issued == 1 {
                        stats.num_single_issue += 1;
                    } else {
                        stats.num_dual_issue += 1;
                    }
                }
                break;
            }
        }

        // issue stall statistics
        #[cfg(feature = "stats")]
        {
            let mut stats = self.stats.lock();
            if !valid_inst {
                // idle or control hazard
                stats.issue_raw_hazard_stall += 1;
            } else if !ready_inst {
                // waiting for RAW hazards (possibly due to memory)
                stats.issue_control_hazard_stall += 1;
            } else if !issued_inst {
                // pipeline stalled
                stats.issue_pipeline_stall += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::testing;
    use std::ptr;

    #[ignore = "todo"]
    #[test]
    fn test_shd_warp() {
        use playground::types::trace_shd_warp::new_trace_shd_warp;
        let core = ptr::null_mut();
        let warp_size = 32;
        let mut warp = unsafe { new_trace_shd_warp(core, warp_size) };
        warp.pin_mut().reset();
        dbg!(&warp.get_n_completed());
        dbg!(&warp.hardware_done());
        dbg!(&warp.functional_done());
    }

    #[test]
    fn test_skip_iterator_indexing() {
        let issued_warp_id = 3;
        let supervised_warp_ids = [1, 2, 3, 4, 5];
        let mut last_supervised_idx = 0;

        for (idx, id) in supervised_warp_ids.iter().enumerate() {
            if *id == issued_warp_id {
                last_supervised_idx = idx;
            }
        }
        assert_eq!(
            supervised_warp_ids.get(last_supervised_idx),
            Some(&issued_warp_id)
        );
    }

    impl From<&dyn super::Scheduler> for testing::state::Scheduler {
        fn from(scheduler: &dyn super::Scheduler) -> Self {
            let prioritized_warp_ids: Vec<_> = scheduler
                .prioritized_warps()
                .iter()
                .map(|(_idx, warp)| {
                    let warp = warp.try_lock();
                    (warp.warp_id, warp.dynamic_warp_id)
                })
                .collect();
            Self {
                prioritized_warp_ids,
            }
        }
    }
}
