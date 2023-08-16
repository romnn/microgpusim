pub mod gto;
pub mod ordering;

use crate::{config::GPUConfig, core::PipelineStage, opcodes, scoreboard, warp};
use console::style;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, RwLock};

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

pub trait SchedulerUnit: Send + Sync + std::fmt::Debug + 'static {
    // fn cycle(&mut self, _core: &mut dyn super::core::WarpIssuer);
    // fn cycle(&mut self, _core: &mut dyn super::core::WarpIssuer);
    fn cycle(&mut self, _core: &dyn super::core::WarpIssuer);

    fn add_supervised_warp(&mut self, warp: warp::Ref);

    fn prioritized_warps(&self) -> &VecDeque<(usize, warp::Ref)>;

    /// Order warps based on scheduling policy.
    fn order_warps(&mut self);
}

#[derive(Debug)]
pub struct BaseSchedulerUnit {
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

    config: Arc<GPUConfig>,
    stats: Arc<Mutex<stats::scheduler::Scheduler>>,
}

impl BaseSchedulerUnit {
    pub fn new(
        id: usize,
        cluster_id: usize,
        core_id: usize,
        warps: Vec<warp::Ref>,
        scoreboard: Arc<RwLock<scoreboard::Scoreboard>>,
        stats: Arc<Mutex<stats::scheduler::Scheduler>>,
        config: Arc<GPUConfig>,
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

    // fn cycle(&mut self, issuer: &mut dyn super::core::WarpIssuer) {
    fn cycle(&mut self, issuer: &dyn super::core::WarpIssuer) {
        log::debug!("{}: cycle", style("base scheduler").yellow());

        let mut valid_inst = false;
        let mut ready_inst = false;
        let mut issued_inst = false;

        for (next_warp_supervised_idx, next_warp_rc) in &self.next_cycle_prioritized_warps {
            // don't consider warps that are not yet valid
            // let next_warp = next_warp_rc.try_borrow().unwrap();
            let next_warp = next_warp_rc.lock().unwrap();
            let (warp_id, dyn_warp_id) = (next_warp.warp_id, next_warp.dynamic_warp_id);

            if next_warp.done_exit() {
                continue;
            }
            let inst_count = next_warp.instruction_count();
            if inst_count == 0 {
                log::debug!("next warp: {:#?}", &next_warp);
            }
            assert!(inst_count > 0);
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
            let mut issued = 0;

            let mut prev_issued_exec_unit = ExecUnitKind::NONE;
            let max_issue = self.config.max_instruction_issue_per_warp;
            // In tis mode, we only allow dual issue to diff execution
            // units (as in Maxwell and Pascal)
            let diff_exec_units = self.config.dual_issue_diff_exec_units;

            if inst_count > 1 {
                if next_warp.ibuffer_empty() {
                    log::debug!(
                        "warp (warp_id={}, dynamic_warp_id={}) fails as ibuffer_empty",
                        warp_id,
                        dyn_warp_id
                    );
                }

                if next_warp.waiting() {
                    log::debug!(
                        "warp (warp_id={}, dynamic_warp_id={}) is waiting for completion",
                        warp_id,
                        dyn_warp_id
                    );
                }
            }

            let warp = self.warps.get(warp_id).unwrap();

            // todo: what is the difference? why dont we just use next_warp?
            debug_assert!(Arc::ptr_eq(warp, next_warp_rc));
            // debug_assert!(Rc::ptr_eq(warp, next_warp_rc));
            drop(next_warp);

            let mut warp = warp.lock().unwrap();
            // let mut warp = warp.try_borrow_mut().unwrap();
            // let warp = warp.try_borrow_mut().unwrap();
            while !warp.waiting()
                && !warp.ibuffer_empty()
                && checked < max_issue
                && checked <= issued
                && issued < max_issue
            {
                let mut warp_inst_issued = false;

                if let Some(instr) = warp.ibuffer_peek() {
                    log::debug!(
                        "Warp (warp_id={}, dynamic_warp_id={}) instruction buffer[{}] has valid instruction {}",
                        warp_id, dyn_warp_id, warp.next, instr,
                    );

                    valid_inst = true;
                    if !self
                        .scoreboard
                        .read()
                        .unwrap()
                        .has_collision(warp_id, instr)
                    {
                        log::debug!(
                            "Warp (warp_id={}, dynamic_warp_id={}) {}",
                            warp_id,
                            dyn_warp_id,
                            style("passes scoreboard").yellow(),
                        );
                        ready_inst = true;

                        debug_assert!(warp.has_instr_in_pipeline());

                        use opcodes::ArchOp;
                        match instr.opcode.category {
                            ArchOp::LOAD_OP
                            | ArchOp::STORE_OP
                            | ArchOp::MEMORY_BARRIER_OP
                            | ArchOp::TENSOR_CORE_LOAD_OP
                            | ArchOp::TENSOR_CORE_STORE_OP => {
                                let mem_stage = PipelineStage::ID_OC_MEM;

                                let free_register = issuer.has_free_register(mem_stage, self.id);

                                if free_register
                                    && (!diff_exec_units
                                        || prev_issued_exec_unit != ExecUnitKind::MEM)
                                {
                                    // if !diff_exec_units || prev_issued_exec_unit != ExecUnitKind::MEM {
                                    let instr = warp.ibuffer_take().unwrap();
                                    debug_assert_eq!(warp_id, warp.warp_id);
                                    if issuer
                                        .issue_warp(mem_stage, &mut warp, instr, self.id)
                                        .is_ok()
                                    {
                                        issued += 1;
                                        issued_inst = true;
                                        warp_inst_issued = true;
                                        prev_issued_exec_unit = ExecUnitKind::MEM;
                                    }
                                } else {
                                    log::debug!("issue failed: no free mem port register");
                                }
                            }
                            op => {
                                if op != ArchOp::TENSOR_CORE_OP
                                    && op != ArchOp::SFU_OP
                                    && op != ArchOp::DP_OP
                                    && (op as usize) < opcodes::SPEC_UNIT_START_ID
                                {
                                    let mut execute_on_sp = false;
                                    let mut execute_on_int = false;

                                    let sp_pipe_avail = self.config.num_sp_units > 0
                                        && issuer
                                            .has_free_register(PipelineStage::ID_OC_SP, self.id);
                                    let int_pipe_avail = self.config.num_int_units > 0
                                        && issuer
                                            .has_free_register(PipelineStage::ID_OC_INT, self.id);

                                    // if INT unit pipline exist, then execute ALU and INT
                                    // operations on INT unit and SP-FPU on SP unit (like in Volta)
                                    // if INT unit pipline does not exist, then execute all ALU, INT
                                    // and SP operations on SP unit (as in Fermi, Pascal GPUs)
                                    if int_pipe_avail
                                        && op != ArchOp::SP_OP
                                        && !(diff_exec_units
                                            && prev_issued_exec_unit == ExecUnitKind::INT)
                                    {
                                        execute_on_int = true;
                                    } else if sp_pipe_avail
                                        && (self.config.num_int_units == 0
                                            || (self.config.num_int_units > 0
                                                && op == ArchOp::SP_OP))
                                        && !(diff_exec_units
                                            && prev_issued_exec_unit == ExecUnitKind::SP)
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
                                        let instr = warp.ibuffer_take().unwrap();
                                        debug_assert_eq!(warp.warp_id, warp_id);
                                        if issuer
                                            .issue_warp(stage, &mut warp, instr, self.id)
                                            .is_ok()
                                        {
                                            issued += 1;
                                            issued_inst = true;
                                            warp_inst_issued = true;
                                            prev_issued_exec_unit = unit;
                                        }
                                    }
                                }
                            } // op => unimplemented!("op {:?} not implemented", op),
                        }
                    } else {
                        log::debug!(
                            "Warp (warp_id={}, dynamic_warp_id={}) {}",
                            warp_id,
                            dyn_warp_id,
                            style("fails scoreboard").yellow(),
                        );
                    }
                }
                if warp_inst_issued {
                    log::debug!(
                        "Warp (warp_id={}, dynamic_warp_id={}) issued {} instructions",
                        warp_id,
                        dyn_warp_id,
                        issued
                    );
                    warp.ibuffer_step();
                }
                checked += 1;
            }
            drop(warp);
            if issued > 0 {
                // This might be a bit inefficient, but we need to maintain
                // two ordered list for proper scheduler execution.
                // We could remove the need for this loop by associating a
                // supervised_is index with each entry in the
                // m_next_cycle_prioritized_warps vector.
                // For now, just run through until you find the right warp_id
                // for (sup_idx, supervised) in self.supervised_warps.iter().enumerate() {
                //     // if *next_warp_rc.try_borrow().unwrap() == *supervised.try_borrow().unwrap() {
                //     // todo!("check if this deadlocks.. (it will). if so just order next cycle prio using sort_by_key and using enumerate to maintain the indexes");
                //     let next_warp_rc = next_warp_rc.try_lock().unwrap();
                //     let nw = (
                //         next_warp_rc.block_id,
                //         next_warp_rc.warp_id,
                //         next_warp_rc.dynamic_warp_id,
                //     );
                //     drop(next_warp_rc);
                //
                //     let supervised = supervised.try_lock().unwrap();
                //     let sup = (
                //         supervised.block_id,
                //         supervised.warp_id,
                //         supervised.dynamic_warp_id,
                //     );
                //
                //     drop(supervised);
                //
                //     // if *next_warp_rc.try_lock().unwrap() == *supervised.try_lock().unwrap() {
                //     if nw == sup {
                //         self.last_supervised_issued_idx = sup_idx;
                //     }
                // }

                self.last_supervised_issued_idx = *next_warp_supervised_idx;

                self.num_issued_last_cycle = issued;
                let mut stats = self.stats.lock().unwrap();
                if issued == 1 {
                    stats.num_single_issue += 1;
                } else {
                    stats.num_dual_issue += 1;
                }
                break;
            }
        }

        // issue stall statistics
        let mut stats = self.stats.lock().unwrap();
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
        assert!(false);
    }

    #[test]
    fn test_skip_iterator_indexing() {
        let issued_warp_id = 3;
        let supervised_warp_ids = vec![1, 2, 3, 4, 5];
        let mut last_supervised_idx = 0;

        for (idx, id) in supervised_warp_ids.iter().enumerate() {
            if *id == issued_warp_id {
                last_supervised_idx = idx;
            }
        }
        assert_eq!(
            supervised_warp_ids.iter().nth(last_supervised_idx),
            Some(&issued_warp_id)
        );
    }

    impl From<&dyn super::SchedulerUnit> for testing::state::Scheduler {
        fn from(scheduler: &dyn super::SchedulerUnit) -> Self {
            let prioritized_warp_ids: Vec<_> = scheduler
                .prioritized_warps()
                .iter()
                .map(|(_idx, warp)| {
                    let warp = warp.try_lock().unwrap();
                    (warp.warp_id, warp.dynamic_warp_id)
                })
                .collect();
            Self {
                prioritized_warp_ids,
            }
        }
    }
}
