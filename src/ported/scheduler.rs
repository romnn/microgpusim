use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::{Arc, Mutex, RwLock};

use trace_model::MemAccessTraceEntry;

use super::core::PipelineStage;
use super::{instruction::WarpInstruction, opcodes, register_set, scoreboard};
use crate::config::GPUConfig;
use bitvec::{array::BitArray, BitArr};
use console::style;

pub type ThreadActiveMask = BitArr!(for 32, in u32);

// type CoreWarp = Arc<Mutex<SchedulerWarp>>;
pub type CoreWarp = Rc<RefCell<SchedulerWarp>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ExecUnitKind {
    NONE = 0,
    SP = 1,
    SFU = 2,
    MEM = 3,
    DP = 4,
    INT = 5,
    TENSOR = 6,
    SPECIALIZED = 7,
}

// // todo: how to do that when not in exec driven?
// pub fn get_pdom_stack_top_info(warp_id: usize, instr: &WarpInstruction) -> (usize, usize) {
//     todo!("get_pdom_stack_top_info");
//     (0, 0)
// }

#[derive(Debug)]
pub struct SchedulerWarp {
    pub block_id: u64,
    pub dynamic_warp_id: usize,
    pub warp_id: usize,
    pub kernel: Option<Arc<super::KernelInfo>>,

    pub trace_pc: usize,
    pub active_mask: ThreadActiveMask,
    pub trace_instructions: VecDeque<WarpInstruction>,

    // state
    pub done_exit: bool,
    pub num_instr_in_pipeline: usize,
    pub num_outstanding_stores: usize,
    pub num_outstanding_atomics: usize,
    pub has_imiss_pending: bool,
    pub instr_buffer: Vec<Option<WarpInstruction>>,
    pub next: usize,
}

impl PartialEq for SchedulerWarp {
    fn eq(&self, other: &Self) -> bool {
        self.kernel == other.kernel
            && self.block_id == other.block_id
            && self.warp_id == other.warp_id
            && self.dynamic_warp_id == other.dynamic_warp_id
    }
}

const IBUFFER_SIZE: usize = 2;

impl Default for SchedulerWarp {
    fn default() -> Self {
        let instr_buffer = vec![None; IBUFFER_SIZE];
        Self {
            block_id: 0,
            dynamic_warp_id: u32::MAX as usize,
            warp_id: u32::MAX as usize,
            kernel: None,
            trace_pc: 0,
            trace_instructions: VecDeque::new(),
            active_mask: BitArray::ZERO,
            done_exit: false,
            num_instr_in_pipeline: 0,
            num_outstanding_stores: 0,
            num_outstanding_atomics: 0,
            has_imiss_pending: false,
            instr_buffer,
            next: 0,
        }
    }
}

impl SchedulerWarp {
    // #[deprecated]
    // pub fn new(
    //     &mut self,
    //     start_pc: Option<usize>,
    //     block_id: u64,
    //     warp_id: usize,
    //     // warp_size: usize,
    //     dynamic_warp_id: usize,
    //     active_mask: ThreadActiveMask,
    // ) {
    //     self.block_id = block_id;
    //     self.warp_id = warp_id;
    //     self.dynamic_warp_id = dynamic_warp_id;
    //     // self.next_pc = start_pc;
    //     // assert(self.num_completed >= active.count());
    //     // assert(n_completed <= m_warp_size);
    //     self.active_mask = active_mask;
    // }

    pub fn init(
        &mut self,
        start_pc: Option<usize>,
        block_id: u64,
        warp_id: usize,
        dynamic_warp_id: usize,
        active_mask: ThreadActiveMask,
        kernel: Arc<super::KernelInfo>,
    ) {
        self.block_id = block_id;
        self.warp_id = warp_id;
        self.dynamic_warp_id = dynamic_warp_id;
        self.done_exit = false;
        self.kernel = Some(kernel);
        // self.next_pc = start_pc;
        // assert(self.num_completed >= active.count());
        // assert(n_completed <= m_warp_size);
        self.active_mask = active_mask;
    }

    pub fn reset(&mut self) {
        debug_assert_eq!(self.num_outstanding_stores, 0);
        debug_assert_eq!(self.num_instr_in_pipeline, 0);
        self.has_imiss_pending = false;
        // self.warp_id = 0; // should be none
        // self.dynamic_warp_id = 0; // should be none
        self.warp_id = u32::MAX as usize;
        self.dynamic_warp_id = u32::MAX as usize;

        self.active_mask.fill(false);
        // m_n_atomic = 0;
        // m_membar = false;
        self.done_exit = true;
        // self.last_fetch = 0;
        self.next = 0;
        //
        // // Jin: cdp support
        // m_cdp_latency = 0;
        // m_cdp_dummy = false;
        // todo!("reset shd warp");
    }

    // pub fn set_has_imiss_pending(&self, value: bool) {
    //     self.imiss_pending = true;
    //     todo!("scheduler warp: set imiss pending");
    // }

    // pub fn inc_instr_in_pipeline(&self) {
    //     todo!("scheduler warp: inc_instr_in_pipeline");
    // }

    pub fn current_instr(&self) -> Option<&WarpInstruction> {
        // let trace_pc = *self.trace_pc.lock().unwrap();
        // let trace_instructions = self.trace_instructions.lock().unwrap();
        // trace_instructions.get(trace_pc)
        self.trace_instructions.get(self.trace_pc)
    }

    pub fn push_trace_instruction(&mut self, instr: WarpInstruction) {
        // todo!("scheduler warp: push trace instr");
        // self.trace_instructions.lock().unwrap().push_back(instr);
        self.trace_instructions.push_back(instr);
    }

    // todo: might do the conversion using `from_trace` during initialization
    // so this is not a special case and we support execution driven later on?
    // pub fn next_trace_inst(&mut self) -> Option<WarpInstruction> {
    pub fn next_trace_inst(&mut self) -> Option<WarpInstruction> {
        // todo!("scheduler warp: next trace instr");
        // let mut trace_pc = self.trace_pc.lock().unwrap();
        // let trace_instructions = self.trace_instructions.lock().unwrap();
        let trace_instr = self.trace_instructions.get(self.trace_pc)?;
        // let Some(trace_instr) = self.trace_instructions.get(self.trace_pc) else {
        //     return None;
        // };
        // let warp_instr = WarpInstruction::from_trace(&*self.kernel, trace_instr.clone());
        // new_inst->parse_from_trace_struct(
        //     warp_traces[trace_pc], m_kernel_info->OpcodeMap,
        //     m_kernel_info->m_tconfig, m_kernel_info->m_kernel_trace_info);
        self.trace_pc += 1;
        Some(trace_instr.clone())
    }

    pub fn instruction_count(&self) -> usize {
        self.trace_instructions.len()
    }

    // pub fn trace_start_pc(&self) -> Option<usize> {
    //     // debug_assert!(!self.trace_instructions.is_empty());
    //     // let trace_instructions = self.trace_instructions.lock().unwrap();
    //     self.trace_instructions.front().map(|instr| instr.pc)
    // }

    pub fn pc(&self) -> Option<usize> {
        // debug_assert!(!self.trace_instructions.is_empty());
        // let trace_pc = *self.trace_pc.lock().unwrap();
        // let trace_instructions = self.trace_instructions.lock().unwrap();
        debug_assert!(self.trace_pc <= self.instruction_count());
        self.trace_instructions
            .get(self.trace_pc)
            .map(|instr| instr.pc)
    }

    pub fn done(&self) -> bool {
        // let trace_instructions = self.trace_instructions.lock().unwrap();
        self.trace_pc == self.instruction_count()
    }

    pub fn clear(&mut self) {
        // todo: should we actually clear schedule warps or just swap
        // *self.trace_pc.lock().unwrap() = 0;
        self.trace_pc = 0;
        // let mut trace_instructions = self.trace_instructions.lock().unwrap();
        self.trace_instructions.clear();
    }

    // pub fn dec_instr_in_pipeline(&mut self) {
    //     todo!("sched warp: dec instr in pipeline");
    // }
    //
    // pub fn inc_instr_in_pipeline(&mut self) {
    //     todo!("sched warp: inc instr in pipeline");
    // }

    // pub fn ibuffer_fill(&mut self, slot: usize, instr: WarpInstruction) {
    pub fn ibuffer_fill(&mut self, slot: usize, instr: WarpInstruction) {
        // todo!("sched warp: ibuffer fill");
        debug_assert!(slot < self.instr_buffer.len());
        self.instr_buffer[slot] = Some(instr);
        self.next = 0;
    }

    pub fn ibuffer_size(&self) -> usize {
        self.instr_buffer.iter().filter(|x| x.is_some()).count()
    }

    pub fn ibuffer_empty(&self) -> bool {
        self.instr_buffer.iter().all(Option::is_none)
    }

    pub fn ibuffer_flush(&mut self) {
        // todo!("sched warp: ibuffer_flush");
        for i in self.instr_buffer.iter_mut() {
            if i.is_some() {
                self.num_instr_in_pipeline -= 1;
            }
            *i = None;
        }
    }

    pub fn ibuffer_peek(&self) -> Option<&WarpInstruction> {
        self.instr_buffer[self.next].as_ref()
    }

    // #[deprecated(note = "should check ibuffer next instr for none")]
    // pub fn ibuffer_next_valid(&self) -> bool {
    //     self.instr_buffer[self.next].is_some()
    // }

    pub fn ibuffer_take(&mut self) -> Option<WarpInstruction> {
        // if self.instr_buffer[self.next].is_some() {
        //     // usually in flush, however when flushed we already took out all instructions
        //     self.num_instr_in_pipeline -= 1;
        // }
        self.instr_buffer[self.next].take()
    }

    pub fn ibuffer_step(&mut self) {
        // todo!("sched warp: ibuffer_step");
        self.next = (self.next + 1) % IBUFFER_SIZE;
    }

    pub fn done_exit(&self) -> bool {
        // todo!("sched warp: done exit");
        self.done_exit
        // false
    }

    pub fn hardware_done(&self) -> bool {
        self.functional_done() && self.stores_done() && self.num_instr_in_pipeline == 0
        // todo!("sched warp: hardware done");
    }

    pub fn has_instr_in_pipeline(&self) -> bool {
        self.num_instr_in_pipeline > 0
        // todo!("sched warp: instructions in pipeline");
    }

    pub fn stores_done(&self) -> bool {
        self.num_outstanding_stores == 0
        // todo!("sched warp: stores done");
    }

    pub fn num_completed(&self) -> usize {
        self.active_mask.count_zeros()
    }

    pub fn set_thread_completed(&mut self, thread_id: usize) {
        self.active_mask.set(thread_id, false);
    }

    pub fn functional_done(&self) -> bool {
        self.active_mask.not_any()
        // self.num_completed() == self.warp_size
    }

    // pub fn set_next_pc(&mut self, pc: usize) {
    //     self.next_pc = Some(pc);
    //     // todo!("sched warp: set_next_pc");
    //     // *self.next_pc.lock().unwrap() = Some(pc);
    // }

    // pub fn imiss_pending(&self) -> bool {
    //     self.has_imiss_pending
    //     // todo!("sched warp: imiss pending");
    // }

    pub fn waiting(&self) -> bool {
        // todo!("sched warp: waiting");
        if self.functional_done() {
            // waiting to be initialized with a kernel
            true
        // } else if core.warp_waiting_at_barrier(self.warp_id) {
        //     // waiting for other warps in block to reach barrier
        //     true
        // } else if core.warp_waiting_at_mem_barrier(self.warp_id) {
        //     // waiting for memory barrier
        //     true
        } else if self.num_outstanding_atomics > 0 {
            // waiting for atomic operation to complete at memory:
            // this stall is not required for accurate timing model,
            // but rather we stall here since if a call/return
            // instruction occurs in the meantime the functional
            // execution of the atomic when it hits DRAM can cause
            // the wrong register to be read.
            true
        } else {
            false
        }
    }

    pub fn dynamic_warp_id(&self) -> usize {
        self.dynamic_warp_id
    }
}

fn sort_warps_by_oldest_dynamic_id(lhs: &CoreWarp, rhs: &CoreWarp) -> std::cmp::Ordering {
    let lhs = lhs.try_borrow().unwrap();
    let rhs = rhs.try_borrow().unwrap();
    if lhs.done_exit() || lhs.waiting() {
        std::cmp::Ordering::Greater
    } else if rhs.done_exit() || rhs.waiting() {
        std::cmp::Ordering::Less
    } else {
        lhs.dynamic_warp_id().cmp(&rhs.dynamic_warp_id())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ordering {
    // The item that issued last is prioritized first then the
    // sorted result of the priority_function
    GREEDY_THEN_PRIORITY_FUNC = 0,
    // No greedy scheduling based on last to issue.
    //
    // Only the priority function determines priority
    PRIORITY_FUNC_ONLY,
    // NUM_ORDERING,
}

#[derive(Debug)]
pub struct BaseSchedulerUnit {
    // pub struct BaseSchedulerUnit<'a> {
    // <'a> {
    id: usize,
    /// This is the prioritized warp list that is looped over each cycle to
    /// determine which warp gets to issue.
    // next_cycle_prioritized_warps: VecDeque<&'a CoreWarp>,
    next_cycle_prioritized_warps: VecDeque<CoreWarp>,
    // The m_supervised_warps list is all the warps this scheduler is
    // supposed to arbitrate between.
    // This is useful in systems where there is more than one warp scheduler.
    // In a single scheduler system, this is simply all the warps
    // assigned to this core.
    // supervised_warps: VecDeque<&'a CoreWarp>,
    supervised_warps: VecDeque<CoreWarp>,
    /// This is the iterator pointer to the last supervised warp you issued
    // last_supervised_issued: Vec<SchedulerWarp>,
    // last_supervised_issued: std::slice::Iter<'a, SchedulerWarp>,
    last_supervised_issued_idx: usize,
    // scheduler: GTOScheduler,

    // warps: &'a Vec<SchedulerWarp>,
    warps: Vec<CoreWarp>,
    // warps: &'a Vec<Option<SchedulerWarp>>,
    // register_set *m_mem_out;
    // std::vector<register_set *> &m_spec_cores_out;
    num_issued_last_cycle: usize,
    current_turn_warp: usize,

    // mem_out: &'a register_set::RegisterSet,
    // mem_out: Arc<register_set::RegisterSet>,

    // core: &'a super::core::InnerSIMTCore,
    scoreboard: Arc<RwLock<scoreboard::Scoreboard>>,
    config: Arc<GPUConfig>,
    stats: Arc<Mutex<stats::Stats>>,
}

// impl<'a> BaseSchedulerUnit<'a> {
impl BaseSchedulerUnit {
    pub fn new(
        id: usize,
        // warps: &'a Vec<SchedulerWarp>,
        warps: Vec<CoreWarp>,
        // warps: &'a Vec<Option<SchedulerWarp>>,
        // mem_out: &'a register_set::RegisterSet,
        // core: &'a super::core::InnerSIMTCore,
        scoreboard: Arc<RwLock<scoreboard::Scoreboard>>,
        stats: Arc<Mutex<stats::Stats>>,
        config: Arc<GPUConfig>,
    ) -> Self {
        let supervised_warps = VecDeque::new();
        Self {
            id,
            next_cycle_prioritized_warps: VecDeque::new(),
            supervised_warps,
            last_supervised_issued_idx: 0,
            // warps: Vec::new(),
            warps,
            num_issued_last_cycle: 0,
            current_turn_warp: 0,
            // mem_out,
            // core,
            scoreboard,
            config,
            stats,
        }
    }

    // pub fn add_supervised_warp_id(&mut self, warp_id: usize) {
    //     // let warp = self.warps[warp_id].as_ref().unwrap();
    //     let warp = &self.warps[warp_id];
    //     self.supervised_warps.push_back(warp);
    // }

    fn prioritized_warps(&self) -> &VecDeque<CoreWarp> {
        &self.next_cycle_prioritized_warps
    }

    // fn cycle(&mut self, core: ()) {
    // fn cycle<I>(&mut self, core: &mut super::core::InnerSIMTCore<I>) {
    fn cycle(&mut self, issuer: &mut dyn super::core::WarpIssuer) {
        println!("{}: cycle", style("base scheduler").yellow());

        // there was one warp with a valid instruction to issue (didn't require flush due to control hazard)
        let mut valid_inst = false;
        // of the valid instructions, there was one not waiting for pending register writes
        let mut ready_inst = false;
        // of these we issued one
        let mut issued_inst = false;

        // dbg!(&self.next_cycle_prioritized_warps.len());
        // dbg!(&self.supervised_warps.len());
        // dbg!(&self.last_supervised_issued_idx);
        //
        // dbg!(&self
        //     .warps
        //     .iter()
        //     .map(|w| w.lock().unwrap().instruction_count())
        //     .sum::<usize>());
        // dbg!(&self
        //     .supervised_warps
        //     .iter()
        //     .map(|w| w.lock().unwrap().instruction_count())
        //     .sum::<usize>());
        //
        // dbg!(&self
        //     .next_cycle_prioritized_warps
        //     .iter()
        //     .map(|w| w.lock().unwrap().instruction_count())
        //     .sum::<usize>());

        // println!(
        //     "supervised warps: {:#?}",
        //     self.supervised_warps
        //         .iter()
        //         .map(|w| w.lock().unwrap().instruction_count())
        //         .filter(|&c| c > 0)
        //         .collect::<Vec<_>>()
        // );
        // println!(
        //     "next_cycle_prioritized_warps: {:#?}",
        //     self.next_cycle_prioritized_warps
        //         .iter()
        //         .map(|w| w.lock().unwrap().instruction_count())
        //         .filter(|&c| c > 0)
        //         .collect::<Vec<_>>()
        // );

        // println!("next cycle prio warp");
        for next_warp_rc in &self.next_cycle_prioritized_warps {
            // don't consider warps that are not yet valid
            let next_warp = next_warp_rc.try_borrow().unwrap();
            let (warp_id, dyn_warp_id) = (next_warp.warp_id, next_warp.dynamic_warp_id);
            // println!("locked next warp = {}", warp_id);

            if next_warp.done_exit() {
                continue;
            }
            let inst_count = next_warp.instruction_count();
            if inst_count == 0 {
                println!("next warp: {:#?}", &next_warp);
            }
            debug_assert!(inst_count > 0);
            if inst_count > 1 {
                println!(
                    "scheduler: \n\t => testing (warp_id={}, dynamic_warp_id={}, trace_pc={}, pc={:?}, ibuffer={:?}, {} instructions)",
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
                    println!(
                        "warp (warp_id={}, dynamic_warp_id={}) fails as ibuffer_empty",
                        warp_id, dyn_warp_id
                    );
                }

                if next_warp.waiting() {
                    println!(
                        "warp (warp_id={}, dynamic_warp_id={}) is waiting for completion",
                        warp_id, dyn_warp_id
                    );
                }
            }

            let warp = self.warps.get(warp_id).unwrap();

            // todo: what is the difference? why dont we just use next_warp?
            debug_assert!(Rc::ptr_eq(warp, next_warp_rc));
            drop(next_warp);

            // println!("locking warp = {}", warp_id);
            let mut warp = warp.try_borrow_mut().unwrap();
            // println!("locked warp {}", warp_id);
            // .as_mut()
            // .as_ref()
            // .unwrap();
            while !warp.waiting()
                && !warp.ibuffer_empty()
                && checked < max_issue
                && checked <= issued
                && issued < max_issue
            {
                // let valid = warp.ibuffer_next_valid();
                let mut warp_inst_issued = false;

                if let Some(instr) = warp.ibuffer_peek() {
                    // let (pc, rpc) = get_pdom_stack_top_info(warp_id, instr);
                    println!(
                        "Warp (warp_id={}, dynamic_warp_id={}) instruction buffer[{}] has valid instruction {}",
                        warp_id, dyn_warp_id, warp.next, instr,
                    );

                    // In trace-driven mode, we assume no control hazard, meaning
                    // that `pc == rpc == instr.pc`
                    // if pc != instr.pc {
                    //     println!(
                    //         "Warp (warp_id {}, dynamic_warp_id {}) control hazard instruction flush",
                    //         warp_id, dyn_warp_id);
                    //     // control hazard
                    //     warp.set_next_pc(pc);
                    //     warp.ibuffer_flush();
                    // } else {
                    valid_inst = true;
                    if !self
                        .scoreboard
                        .read()
                        .unwrap()
                        .has_collision(warp_id, instr)
                    {
                        println!(
                            "Warp (warp_id={}, dynamic_warp_id={}) {}",
                            warp_id,
                            dyn_warp_id,
                            style("passes scoreboard").yellow(),
                        );
                        ready_inst = true;

                        // let active_mask = core.active_mask(warp_id, instr);

                        debug_assert!(warp.has_instr_in_pipeline());

                        use opcodes::ArchOp;
                        match instr.opcode.category {
                            ArchOp::LOAD_OP
                            | ArchOp::STORE_OP
                            | ArchOp::MEMORY_BARRIER_OP
                            | ArchOp::TENSOR_CORE_LOAD_OP
                            | ArchOp::TENSOR_CORE_STORE_OP => {
                                // if warp.warp_id == 3 {
                                //     super::debug_break(format!(
                                //         "scheduled mem instr for warp id 3: {}",
                                //         instr
                                //     ));
                                // }
                                let mem_stage = PipelineStage::ID_OC_MEM;

                                let free_register = issuer.has_free_register(mem_stage, self.id);

                                if free_register
                                    && (!diff_exec_units
                                        || prev_issued_exec_unit != ExecUnitKind::MEM)
                                {
                                    let instr = warp.ibuffer_take().unwrap();
                                    debug_assert_eq!(warp_id, warp.warp_id);
                                    issuer.issue_warp(mem_stage, &mut warp, instr, self.id);
                                    // .issue_warp(mem_stage, &mut warp, instr, warp_id, self.id);
                                    issued += 1;
                                    issued_inst = true;
                                    warp_inst_issued = true;
                                    prev_issued_exec_unit = ExecUnitKind::MEM;
                                } else {
                                    // panic!("issue failed: free register={}", free_register);
                                }
                            }
                            // ArchOp::EXIT_OPS => {}
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

                                    println!(
                                        "execute on INT={} execute on SP={}",
                                        execute_on_int, execute_on_sp
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
                                        issuer.issue_warp(stage, &mut warp, instr, self.id);
                                        // .issue_warp(stage, &mut warp, instr, warp_id, self.id);
                                        issued += 1;
                                        issued_inst = true;
                                        warp_inst_issued = true;
                                        prev_issued_exec_unit = unit;
                                    }
                                }
                                // else if ((m_shader->m_config->gpgpu_num_dp_units > 0) &&
                                //                          (pI->op == DP_OP) &&
                                //                          !(diff_exec_units && previous_issued_inst_exec_type ==
                                //                                                   exec_unit_type_t::DP)) {
                                // } else if (((m_shader->m_config->gpgpu_num_dp_units == 0 &&
                                //                          pI->op == DP_OP) ||
                                //                         (pI->op == SFU_OP) || (pI->op == ALU_SFU_OP)) &&
                                //                        !(diff_exec_units && previous_issued_inst_exec_type ==
                                //                                                 exec_unit_type_t::SFU)) {
                                // } else if ((pI->op == TENSOR_CORE_OP) &&
                                //                          !(diff_exec_units && previous_issued_inst_exec_type ==
                                //                                                   exec_unit_type_t::TENSOR)) {
                                // } else if ((pI->op >= SPEC_UNIT_START_ID) &&
                                //                          !(diff_exec_units &&
                                //                            previous_issued_inst_exec_type ==
                                //                                exec_unit_type_t::SPECIALIZED)) {
                                // }
                            } // op => unimplemented!("op {:?} not implemented", op),
                        }
                    } else {
                        println!(
                            "Warp (warp_id={}, dynamic_warp_id={}) {}",
                            warp_id,
                            dyn_warp_id,
                            style("fails scoreboard").yellow(),
                        );
                    }
                    // }
                }
                // else if (valid) {
                //   // this case can happen after a return instruction in diverged warp
                //   SCHED_DPRINTF(
                //       "Warp (warp_id %u, dynamic_warp_id %u) return from diverged warp "
                //       "flush\n",
                //       (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());
                //   warp(warp_id).set_next_pc(pc);
                //   warp(warp_id).ibuffer_flush();
                // }
                if warp_inst_issued {
                    println!(
                        "Warp (warp_id={}, dynamic_warp_id={}) issued {} instructions",
                        warp_id, dyn_warp_id, issued
                    );
                    // m_stats->event_warp_issued(m_shader->get_sid(), warp_id, num_issued, warp(warp_id).get_dynamic_warp_id());
                    warp.ibuffer_step();
                }
                checked += 1;
            }
            // drop(next_warp);
            drop(warp);
            if issued > 0 {
                // This might be a bit inefficient, but we need to maintain
                // two ordered list for proper scheduler execution.
                // We could remove the need for this loop by associating a
                // supervised_is index with each entry in the
                // m_next_cycle_prioritized_warps vector.
                // For now, just run through until you find the right warp_id
                for (sup_idx, supervised) in self.supervised_warps.iter().enumerate() {
                    // if *next_warp == *supervised.lock().unwrap().warp_id {
                    // println!("locking supervised[{}]", sup_idx);
                    // if dynamicwarp_id == supervised.try_borrow().unwrap().warp_id {
                    // if warp.borrow() == supervised.borrow() {
                    if *next_warp_rc.try_borrow().unwrap() == *supervised.try_borrow().unwrap() {
                        // test
                        self.last_supervised_issued_idx = sup_idx;
                    }
                }
                self.num_issued_last_cycle = issued;
                if issued == 1 {
                    // m_stats->single_issue_nums[m_id]++;
                } else if issued > 1 {
                    // m_stats->dual_issue_nums[m_id]++;
                }
                break;
            } else {
                // println!("WARN: issued should be > 0");
            }
        }

        // issue stall statistics:
        if !valid_inst {
            // idle or control hazard
            // m_stats.shader_cycle_distro[0]++;
        } else if !ready_inst {
            // waiting for RAW hazards (possibly due to memory)
            // m_stats.shader_cycle_distro[1]++;
        } else if !issued_inst {
            // pipeline stalled
            // m_stats.shader_cycle_distro[2]++;
        }

        // todo!("base scheduler unit: cycle");
    }
}

pub trait SchedulerUnit {
    fn cycle(&mut self, core: &mut dyn super::core::WarpIssuer) {
        // fn cycle(&mut self, core: ()) {
        // fn cycle(&mut self) {
        todo!("scheduler unit: cycle");
    }

    // fn done_adding_supervised_warps(&mut self) {
    //     todo!("scheduler unit: done_adding_supervised_warps");
    // }

    fn add_supervised_warp(&mut self, warp: CoreWarp) {
        todo!("scheduler unit: add supervised warp id");
    }

    fn prioritized_warps(&self) -> &VecDeque<CoreWarp>;

    // self.scheduler
    // self.inner.supervised_warps

    // fn add_supervised_warp_id(&mut self, warp_id: usize) {
    //     todo!("scheduler unit: add supervised warp id");
    // }

    /// Order warps based on scheduling policy.
    ///
    /// Derived classes can override this function to populate
    /// m_supervised_warps with their scheduling policies
    fn order_warps(
        &mut self,
        // out: &mut VecDeque<SchedulerWarp>,
        // warps: &mut Vec<SchedulerWarp>,
        // last_issued_warps: &Vec<SchedulerWarp>,
        // num_warps_to_add: usize,
    ) {
        todo!("scheduler unit: order warps")
    }
}

#[derive(Debug)]
pub struct LrrScheduler {
    inner: BaseSchedulerUnit,
}

pub fn all_different<T>(values: &[Rc<RefCell<T>>]) -> bool {
    for (vi, v) in values.iter().enumerate() {
        for (vii, vv) in values.iter().enumerate() {
            let should_be_equal = vi == vii;
            let are_equal = Rc::ptr_eq(v, vv);
            if should_be_equal && !are_equal {
                return false;
            }
            if !should_be_equal && are_equal {
                return false;
            }
        }
    }
    true
}

// pub struct LrrScheduler<'a> {
//     inner: BaseSchedulerUnit<'a>,
// }

// impl<'a> BaseSchedulerUnit<'a> {
impl BaseSchedulerUnit {
    fn order_by_priority<F>(&mut self, ordering: Ordering, priority_func: F)
    where
        F: FnMut(&CoreWarp, &CoreWarp) -> std::cmp::Ordering,
    {
        // gto_scheduler::scheduler_unit BEFORE: m_next_cycle_prioritized_warps: [31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,]
        // gto_scheduler::scheduler_unit AFTER: m_next_cycle_prioritized_warps: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,]

        // todo!("base scheduler unit: order by priority");
        let num_warps_to_add = self.supervised_warps.len();
        let out = &mut self.next_cycle_prioritized_warps;

        debug_assert!(num_warps_to_add <= self.warps.len());
        out.clear();

        let mut last_issued_iter = self.warps.iter().skip(self.last_supervised_issued_idx);
        debug_assert!(all_different(&self.warps));

        // TODO: maybe we actually should make a copy of the supervised warps to not actually
        // reorder those for stability

        let mut supervised_warps: Vec<_> = self.supervised_warps.clone().into_iter().collect();
        supervised_warps.sort_by(priority_func);

        debug_assert!(all_different(&self.supervised_warps.make_contiguous()));
        debug_assert!(all_different(&supervised_warps));

        // self.supervised_warps
        //     .make_contiguous()
        //     .sort_by(priority_func);

        match ordering {
            Ordering::GREEDY_THEN_PRIORITY_FUNC => {
                let greedy_value = last_issued_iter.next();
                if let Some(greedy) = greedy_value {
                    out.push_back(Rc::clone(greedy));
                }

                println!(
                    "added greedy warp (last supervised issued idx={}): {:?}",
                    self.last_supervised_issued_idx,
                    &greedy_value.map(|w| w.borrow().dynamic_warp_id)
                );

                // self.supervised_warps
                //     .make_contiguous()
                //     .sort_by(priority_func);

                // self.supervised_warpsself.supervised_warps.any( .iter()

                out.extend(
                    // self.supervised_warps
                    supervised_warps
                        .into_iter()
                        .take(num_warps_to_add)
                        // .filter(|&warp| {
                        .filter(|warp| {
                            if let Some(greedy) = greedy_value {
                                // note: this could defo deadlock because mutex locks both for
                                // write, we should use the id or whatever warp uses
                                // return *w.borrow() != *greedy.borrow();
                                // return w != greedy;
                                // return w != greedy;
                                // Rc::ptr_eq(w, greedy)
                                // let greedy_cell: &RefCell<_> = greedy.as_ref();
                                // let warp_cell: &RefCell<_> = warp.as_ref();
                                // dbg!(greedy_cell.borrow().dynamic_warp_id);
                                // dbg!(warp_cell.borrow().dynamic_warp_id);
                                // std::ptr::eq(greedy_cell.as_ptr(), warp_cell.as_ptr())
                                // false
                                // std::ptr::eq(Rc::as_ref(w),
                                // println!(
                                //     "greedy@{:?} warp@{:?}",
                                //     Rc::as_ptr(greedy),
                                //     Rc::as_ptr(warp)
                                // );
                                let already_added = Rc::ptr_eq(greedy, warp);
                                !already_added
                            } else {
                                true
                            }
                        }),
                    // .map(Rc::clone),
                );
            }
            Ordering::PRIORITY_FUNC_ONLY => {
                // self.supervised_warps
                //     .make_contiguous()
                //     .sort_by(priority_func);
                out.extend(
                    // self.supervised_warps
                    supervised_warps.into_iter().take(num_warps_to_add), // .map(Rc::clone),
                );
            }
        }
        assert_eq!(num_warps_to_add, out.len());
    }

    fn order_rrr(
        &mut self,
        // out: &mut VecDeque<SchedulerWarp>,
        // warps: &mut Vec<SchedulerWarp>,
        // std::vector<T> &result_list, const typename std::vector<T> &input_list,
        // const typename std::vector<T>::const_iterator &last_issued_from_input,
        // unsigned num_warps_to_add)
    ) {
        unimplemented!("order rrr is untested");
        let num_warps_to_add = self.supervised_warps.len();
        let out = &mut self.next_cycle_prioritized_warps;
        // order_lrr(
        //     &mut self.inner.next_cycle_prioritized_warps,
        //     &mut self.inner.supervised_warps,
        //     &mut self.inner.last_supervised_issued_idx,
        //     // &mut self.inner.last_supervised_issued(),
        //     num_warps_to_add,
        // );

        out.clear();

        let current_turn_warp_ref = self.warps.get(self.current_turn_warp).unwrap();
        let current_turn_warp = current_turn_warp_ref.try_borrow().unwrap();
        // .as_ref()
        // .unwrap();

        if self.num_issued_last_cycle > 0
            || current_turn_warp.done_exit()
            || current_turn_warp.waiting()
        {
            // std::vector<shd_warp_t *>::const_iterator iter =
            //   (last_issued_from_input == input_list.end()) ?
            //     input_list.begin() : last_issued_from_input + 1;

            let mut iter = self
                .supervised_warps
                .iter()
                .skip(self.last_supervised_issued_idx + 1)
                .chain(self.supervised_warps.iter());

            for w in iter.take(num_warps_to_add) {
                let warp = w.try_borrow().unwrap();
                let warp_id = warp.warp_id;
                if !warp.done_exit() && !warp.waiting() {
                    out.push_back(w.clone());
                    self.current_turn_warp = warp_id;
                    break;
                }
            }
            // for (unsigned count = 0; count < num_warps_to_add; ++iter, ++count) {
            //   if (iter == input_list.end()) {
            //   iter = input_list.begin();
            //   }
            //   unsigned warp_id = (*iter)->get_warp_id();
            //   if (!(*iter)->done_exit() && !(*iter)->waiting()) {
            //     result_list.push_back(*iter);
            //     m_current_turn_warp = warp_id;
            //     break;
            //   }
            // }
        } else {
            out.push_back(current_turn_warp_ref.clone());
        }
    }

    fn order_lrr(
        &mut self,
        // out: &mut VecDeque<SchedulerWarp>,
        // warps: &mut Vec<SchedulerWarp>,
        // // last_issued_warps: &Vec<SchedulerWarp>,
        // // last_issued_warps: impl Iterator<Item=SchedulerWarp>,
        // // last_issued_warps: &mut std::slice::Iter<'_, SchedulerWarp>,
        // // last_issued_warps: impl Iterator<Item = &'a SchedulerWarp>,
        // last_issued_warp_idx: &mut usize,
        // num_warps_to_add: usize,
    ) {
        unimplemented!("order lrr is not tested");
        let num_warps_to_add = self.supervised_warps.len();
        let out = &mut self.next_cycle_prioritized_warps;

        debug_assert!(num_warps_to_add <= self.warps.len());
        out.clear();
        // if last_issued_warps
        //   typename std::vector<T>::const_iterator iter = (last_issued_from_input == input_list.end()) ? input_list.begin()
        //                                                    : last_issued_from_input + 1;
        //
        let mut last_issued_iter = self.warps.iter().skip(self.last_supervised_issued_idx);

        let mut iter = last_issued_iter.chain(self.warps.iter());
        // .filter_map(|x| x.as_ref());
        // .filter_map(|x| x.as_ref());

        out.extend(iter.take(num_warps_to_add).cloned());
        // for count in 0..num_warps_to_add {
        //     let Some(warp) = iter.next() else {
        //         return;
        //     };
        //     // if (iter == input_list.end()) {
        //     //   iter = input_list.begin();
        //     // }
        //     out.push_back(warp.clone());
        // }
        // todo!("order lrr: order warps")
    }
}

impl SchedulerUnit for LrrScheduler {
    // impl<'a> SchedulerUnit for LrrScheduler<'a> {
    fn order_warps(
        &mut self,
        // out: &mut VecDeque<SchedulerWarp>,
        // warps: &mut Vec<SchedulerWarp>,
        // last_issued_warps: &Vec<SchedulerWarp>,
        // num_warps_to_add: usize,
    ) {
        self.inner.order_lrr();
        // let num_warps_to_add = self.inner.supervised_warps.len();
        // order_lrr(
        //     &mut self.inner.next_cycle_prioritized_warps,
        //     &mut self.inner.supervised_warps,
        //     &mut self.inner.last_supervised_issued_idx,
        //     // &mut self.inner.last_supervised_issued(),
        //     num_warps_to_add,
        // );
    }

    fn add_supervised_warp(&mut self, warp: CoreWarp) {
        self.inner.supervised_warps.push_back(warp);
        // self.inner.add_supervised_warp_id(warp_id);
    }

    fn prioritized_warps(&self) -> &VecDeque<CoreWarp> {
        self.inner.prioritized_warps()
    }

    // fn add_supervised_warp_id(&mut self, warp_id: usize) {
    //     self.inner.add_supervised_warp_id(warp_id);
    // }

    // fn done_adding_supervised_warps(&mut self) {
    //     self.inner.last_supervised_issued_idx = self.inner.supervised_warps.len();
    // }

    // fn cycle<I>(&mut self, core: &mut super::core::InnerSIMTCore<I>) {
    // fn cycle(&mut self, core: ()) {
    fn cycle(&mut self, issuer: &mut dyn super::core::WarpIssuer) {
        println!("lrr scheduler: cycle enter");
        self.order_warps();
        self.inner.cycle(issuer);
        println!("lrr scheduler: cycle exit");
    }
}

// impl<'a> LrrScheduler<'a> {
impl LrrScheduler {
    // fn order_warps(
    //     &self,
    //     out: &mut VecDeque<SchedulerWarp>,
    //     warps: &mut Vec<SchedulerWarp>,
    //     last_issued_warps: &Vec<SchedulerWarp>,
    //     num_warps_to_add: usize,
    // ) {
    //     todo!("scheduler unit: order warps")
    // }

    pub fn new(
        id: usize,
        // warps: &'a Vec<SchedulerWarp>,
        warps: Vec<CoreWarp>,
        // warps: &'a Vec<Option<SchedulerWarp>>,
        // mem_out: &'a register_set::RegisterSet,
        // core: &'a super::core::InnerSIMTCore,
        scoreboard: Arc<RwLock<scoreboard::Scoreboard>>,
        stats: Arc<Mutex<stats::Stats>>,
        config: Arc<GPUConfig>,
    ) -> Self {
        // todo!("lrr scheduler: new");
        let inner = BaseSchedulerUnit::new(
            id, // mem_out, core,
            warps, scoreboard, stats, config,
        );
        Self { inner }
    }
    // lrr_scheduler(shader_core_stats *stats, shader_core_ctx *shader,
    //               Scoreboard *scoreboard, simt_stack **simt,
    //               std::vector<shd_warp_t *> *warp, register_set *sp_out,
    //               register_set *dp_out, register_set *sfu_out,
    //               register_set *int_out, register_set *tensor_core_out,
    //               std::vector<register_set *> &spec_cores_out,
    //               register_set *mem_out, int id)
    //     : scheduler_unit(stats, shader, scoreboard, simt, warp, sp_out, dp_out,
    //                      sfu_out, int_out, tensor_core_out, spec_cores_out,
    //                      mem_out, id) {}

    // virtual void order_warps();
}

#[derive(Debug)]
pub struct GTOScheduler {
    inner: BaseSchedulerUnit,
}

impl GTOScheduler {
    pub fn new(
        id: usize,
        warps: Vec<CoreWarp>,
        scoreboard: Arc<RwLock<scoreboard::Scoreboard>>,
        stats: Arc<Mutex<stats::Stats>>,
        config: Arc<GPUConfig>,
    ) -> Self {
        let inner = BaseSchedulerUnit::new(
            id, // mem_out, core,
            warps, scoreboard, stats, config,
        );
        Self { inner }
    }
}

impl GTOScheduler {
    fn debug_warp_ids(&self) -> Vec<usize> {
        self.inner
            .next_cycle_prioritized_warps
            .iter()
            .map(|w| w.borrow().warp_id)
            .collect()
    }

    fn debug_dynamic_warp_ids(&self) -> Vec<usize> {
        self.inner
            .next_cycle_prioritized_warps
            .iter()
            .map(|w| w.borrow().dynamic_warp_id())
            .collect()
    }
}

impl SchedulerUnit for GTOScheduler {
    fn order_warps(&mut self) {
        // order_by_priority(
        //     m_next_cycle_prioritized_warps,
        //     m_supervised_warps,
        //     m_last_supervised_issued,
        //     m_supervised_warps.size(),
        //     ORDERING_GREEDY_THEN_PRIORITY_FUNC,
        //     scheduler_unit::sort_warps_by_oldest_dynamic_id,
        // );
        //x

        // let before = self.inner.next_cycle_prioritized_warps.len();
        self.inner.order_by_priority(
            Ordering::GREEDY_THEN_PRIORITY_FUNC,
            sort_warps_by_oldest_dynamic_id,
        );
        // let after = self.inner.next_cycle_prioritized_warps.len();
        // assert_eq!(before, after);
    }

    fn add_supervised_warp(&mut self, warp: CoreWarp) {
        self.inner.supervised_warps.push_back(warp);
    }

    fn prioritized_warps(&self) -> &VecDeque<CoreWarp> {
        self.inner.prioritized_warps()
    }

    // fn done_adding_supervised_warps(&mut self) {
    //     // self.inner.last_supervised_issued_idx = self.inner.supervised_warps.len();
    //     self.inner.last_supervised_issued_idx = 0;
    // }

    // fn cycle(&mut self, core: ()) {
    fn cycle(&mut self, issuer: &mut dyn super::core::WarpIssuer) {
        println!("gto scheduler: cycle enter");
        println!(
            "gto scheduler: BEFORE: prioritized warp ids: {:?}",
            self.debug_warp_ids()
        );
        println!(
            "gto scheduler: BEFORE: prioritized dynamic warp ids: {:?}",
            self.debug_dynamic_warp_ids()
        );

        self.order_warps();

        println!(
            "gto scheduler: AFTER: prioritized warp ids: {:?}",
            self.debug_warp_ids()
        );
        println!(
            "gto scheduler: AFTER: prioritized dynamic warp ids: {:?}",
            self.debug_dynamic_warp_ids()
        );

        self.inner.cycle(issuer);
        println!("gto scheduler: cycle exit");
    }
}

impl GTOScheduler {
    pub fn order_warps(
        &self,
        out: &mut VecDeque<SchedulerWarp>,
        warps: &mut Vec<SchedulerWarp>,
        last_issued_warps: &Vec<SchedulerWarp>,
        num_warps_to_add: usize,
    ) {
        // let mut next_cycle_prioritized_warps = Vec::new();
        //
        // let mut supervised_warps = Vec::new(); // input
        // let mut last_issued_from_input = Vec::new(); // last issued
        // let num_warps_to_add = supervised_warps.len();
        debug_assert!(num_warps_to_add <= warps.len());

        // scheduler_unit::sort_warps_by_oldest_dynamic_id

        // ORDERING_GREEDY_THEN_PRIORITY_FUNC
        out.clear();
        // let greedy_value = last_issued_warps.first();
        // if let Some(greedy_value) = greedy_value {
        //     out.push_back(greedy_value.clone());
        // }
        //
        // warps.sort_by(sort_warps_by_oldest_dynamic_id);
        // out.extend(
        //     warps
        //         .iter()
        //         .take_while(|w| match greedy_value {
        //             None => true,
        //             Some(val) => *w != val,
        //         })
        //         .take(num_warps_to_add)
        //         .cloned(),
        // );

        //     typename std::vector<T>::iterator iter = temp.begin();
        //     for (unsigned count = 0; count < num_warps_to_add; ++count, ++iter) {
        //       if (*iter != greedy_value) {
        //         result_list.push_back(*iter);
        //       }
        //     }

        //   result_list.clear();
        //   typename std::vector<T> temp = input_list;
        //
        //   if (ORDERING_GREEDY_THEN_PRIORITY_FUNC == ordering) {
        //     T greedy_value = *last_issued_from_input;
        //     result_list.push_back(greedy_value);
        //
        //     std::sort(temp.begin(), temp.end(), priority_func);
        //     typename std::vector<T>::iterator iter = temp.begin();
        //     for (unsigned count = 0; count < num_warps_to_add; ++count, ++iter) {
        //       if (*iter != greedy_value) {
        //         result_list.push_back(*iter);
        //       }
        //     }
        //   } else if (ORDERED_PRIORITY_FUNC_ONLY == ordering) {
        //     std::sort(temp.begin(), temp.end(), priority_func);
        //     typename std::vector<T>::iterator iter = temp.begin();
        //     for (unsigned count = 0; count < num_warps_to_add; ++count, ++iter) {
        //       result_list.push_back(*iter);
        //     }
        //   } else {
        //     fprintf(stderr, "Unknown ordering - %d\n", ordering);
        //     abort();
        //   }

        // order by priority
        // (m_next_cycle_prioritized_warps, m_supervised_warps,
        //                 m_last_supervised_issued, m_supervised_warps.size(),
        //                 ORDERING_GREEDY_THEN_PRIORITY_FUNC,
        //                 scheduler_unit::sort_warps_by_oldest_dynamic_id);
    }
}

#[cfg(test)]
mod tests {
    use crate::ported::testing;
    use std::ops::Deref;
    use std::ptr;

    #[ignore = "todo"]
    #[test]
    fn test_shd_warp() {
        use playground::trace_shd_warp::new_trace_shd_warp;
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
            supervised_warp_ids.iter().skip(last_supervised_idx).next(),
            Some(&issued_warp_id)
        );
    }

    impl From<&Box<dyn super::SchedulerUnit>> for testing::state::Scheduler {
        fn from(scheduler: &Box<dyn super::SchedulerUnit>) -> Self {
            let prioritized_warps = scheduler
                .prioritized_warps()
                .iter()
                .map(|warp| warp.borrow().warp_id)
                .collect();
            Self { prioritized_warps }
        }
    }
}
