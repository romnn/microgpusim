use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use trace_model::MemAccessTraceEntry;

use super::{instruction::WarpInstruction, opcodes, register_set, scoreboard, stats::Stats};
use crate::config::GPUConfig;
use bitvec::{array::BitArray, BitArr};

pub type ThreadActiveMask = BitArr!(for 32, in u32);

type CoreWarp = Arc<Mutex<SchedulerWarp>>;

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

#[derive(Debug)]
pub struct SchedulerWarp {
    pub block_id: u64,
    pub dynamic_warp_id: usize,
    pub warp_id: usize,
    pub kernel_id: usize,
    // todo: what is next and trace pc??
    // pub trace_pc: Mutex<usize>,
    pub trace_pc: usize,
    // pub next_pc: Mutex<Option<usize>>,
    pub next_pc: Option<usize>,
    pub active_mask: ThreadActiveMask,
    // pub trace_instructions: Mutex<VecDeque<WarpInstruction>>,
    pub trace_instructions: VecDeque<WarpInstruction>,

    // state
    pub done_exit: bool,
    pub num_instr_in_pipeline: usize,
    pub num_outstanding_stores: usize,
    pub num_outstanding_atomics: usize,
    pub has_imiss_pending: bool,
    pub instr_buffer: Vec<Option<WarpInstruction>>,
    pub next: usize,
    // pub trace_instructions: Vec<MemAccessTraceEntry>,
    // pub warp_traces: Vec<Mem>,
    // pub instructions: Vec<>,
    // pub kernel: Arc<super::KernelInfo>,
    // pub config: Arc<GPUConfig>,
}

impl PartialEq for SchedulerWarp {
    fn eq(&self, other: &Self) -> bool {
        self.kernel_id == other.kernel_id
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
            dynamic_warp_id: 0,
            warp_id: 0,
            kernel_id: 0,
            // todo: what is next and trace pc??
            trace_pc: 0,
            next_pc: None,
            trace_instructions: VecDeque::new(),
            active_mask: BitArray::ZERO,
            // pub trace_instructions: Vec<MemAccessTraceEntry>,
            // pub warp_traces: Vec<Mem>,
            // pub instructions: Vec<>,
            // kernel: Arc<super::KernelInfo>,
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

// todo: how to do that when not in exec driven?
pub fn get_pdom_stack_top_info(warp_id: usize, instr: &WarpInstruction) -> (usize, usize) {
    todo!("get_pdom_stack_top_info");
    (0, 0)
}

impl SchedulerWarp {
    #[deprecated]
    pub fn new(
        &mut self,
        start_pc: Option<usize>,
        block_id: u64,
        warp_id: usize,
        // warp_size: usize,
        dynamic_warp_id: usize,
        active_mask: ThreadActiveMask,
    ) {
        self.block_id = block_id;
        self.warp_id = warp_id;
        self.dynamic_warp_id = dynamic_warp_id;
        self.next_pc = start_pc;
        // assert(self.num_completed >= active.count());
        // assert(n_completed <= m_warp_size);
        self.active_mask = active_mask;
    }

    // todo: just use fields direclty now?
    // #[deprecated]
    pub fn init(
        &mut self,
        start_pc: Option<usize>,
        block_id: u64,
        warp_id: usize,
        dynamic_warp_id: usize,
        active_mask: ThreadActiveMask,
    ) {
        self.block_id = block_id;
        self.warp_id = warp_id;
        self.dynamic_warp_id = dynamic_warp_id;
        self.next_pc = start_pc;
        // assert(self.num_completed >= active.count());
        // assert(n_completed <= m_warp_size);
        self.active_mask = active_mask;
    }

    pub fn set_has_imiss_pending(&self, value: bool) {
        todo!("scheduler warp: set imiss pending");
    }

    pub fn inc_instr_in_pipeline(&self) {
        todo!("scheduler warp: inc_instr_in_pipeline");
    }

    pub fn num_completed(&self) -> usize {
        self.active_mask.count_zeros()
    }

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
        dbg!(&self.trace_pc);
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

    pub fn trace_start_pc(&self) -> Option<usize> {
        // debug_assert!(!self.trace_instructions.is_empty());
        // let trace_instructions = self.trace_instructions.lock().unwrap();
        self.trace_instructions.front().map(|instr| instr.pc)
    }

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

    pub fn ibuffer_next_inst(&self) -> Option<&WarpInstruction> {
        self.instr_buffer[self.next].as_ref()
    }

    #[deprecated(note = "should check ibuffer next instr for none")]
    pub fn ibuffer_next_valid(&self) -> bool {
        self.instr_buffer[self.next].is_some()
    }

    pub fn ibuffer_free(&mut self) {
        self.instr_buffer[self.next] = None
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
        self.functional_done() && self.stores_done() && self.num_instr_in_pipeline < 1
        // todo!("sched warp: hardware done");
    }

    pub fn has_instr_in_pipeline(&self) -> bool {
        self.num_instr_in_pipeline > 0
        // todo!("sched warp: instructions in pipeline");
    }

    pub fn stores_done(&self) -> bool {
        self.num_outstanding_stores > 0
        // todo!("sched warp: stores done");
    }

    pub fn functional_done(&self) -> bool {
        // todo: is that correct?
        self.active_mask.is_empty()
        // self.num_completed() == self.warp_size
        // todo!("sched warp: functional done");
    }

    pub fn set_next_pc(&mut self, pc: usize) {
        self.next_pc = Some(pc);
        // todo!("sched warp: set_next_pc");
        // *self.next_pc.lock().unwrap() = Some(pc);
    }

    pub fn imiss_pending(&self) -> bool {
        self.has_imiss_pending
        // todo!("sched warp: imiss pending");
    }

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

fn sort_warps_by_oldest_dynamic_id(lhs: &SchedulerWarp, rhs: &SchedulerWarp) -> std::cmp::Ordering {
    if lhs.done_exit() || lhs.waiting() {
        std::cmp::Ordering::Greater
    } else if rhs.done_exit() || rhs.waiting() {
        std::cmp::Ordering::Less
    } else {
        lhs.dynamic_warp_id().cmp(&rhs.dynamic_warp_id())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OrderingType {
    // The item that issued last is prioritized first then the
    // sorted result of the priority_function
    ORDERING_GREEDY_THEN_PRIORITY_FUNC = 0,
    // No greedy scheduling based on last to issue.
    //
    // Only the priority function determines priority
    ORDERED_PRIORITY_FUNC_ONLY,
    NUM_ORDERING,
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

    // core: &'a super::core::InnerSIMTCore,
    scoreboard: Arc<scoreboard::Scoreboard>,
    config: Arc<GPUConfig>,
    stats: Arc<Mutex<Stats>>,
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
        scoreboard: Arc<scoreboard::Scoreboard>,
        stats: Arc<Mutex<Stats>>,
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

    // fn cycle<I>(&mut self, core: &mut super::core::InnerSIMTCore<I>) {
    fn cycle(&mut self, core: ()) {
        println!("base scheduler: cycle");

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

        println!(
            "supervised warps: {:#?}",
            self.supervised_warps
                .iter()
                .map(|w| w.lock().unwrap().instruction_count())
                .filter(|&c| c > 0)
                .collect::<Vec<_>>()
        );
        println!(
            "next_cycle_prioritized_warps: {:#?}",
            self.next_cycle_prioritized_warps
                .iter()
                .map(|w| w.lock().unwrap().instruction_count())
                .filter(|&c| c > 0)
                .collect::<Vec<_>>()
        );

        for next_warp in &self.next_cycle_prioritized_warps {
            // don't consider warps that are not yet valid
            let next_warp = next_warp.lock().unwrap();
            let (warp_id, dyn_warp_id) = (next_warp.warp_id, next_warp.dynamic_warp_id);
            if next_warp.done_exit() {
                continue;
            }
            let inst_count = next_warp.instruction_count();
            // dbg!(inst_count);
            if inst_count > 0 {
                println!(
                "scheduler: \n\t => testing (warp_id {}, dynamic_warp_id {}, pc {:?}, {} instructions)",
                warp_id, dyn_warp_id,
                next_warp.trace_start_pc(), inst_count,
            );
            }
            let mut checked = 0;
            let mut issued = 0;

            let mut prev_issued_exec_unit = ExecUnitKind::NONE;
            let max_issue = self.config.max_instruction_issue_per_warp;
            // In tis mode, we only allow dual issue to diff execution
            // units (as in Maxwell and Pascal)
            let diff_exec_units = self.config.dual_issue_diff_exec_units;

            if inst_count > 0 {
                if next_warp.ibuffer_empty() {
                    println!(
                        "warp (warp_id {}, dynamic_warp_id {}) fails as ibuffer_empty",
                        warp_id, dyn_warp_id
                    );
                }

                if next_warp.waiting() {
                    println!(
                        "warp (warp_id {}, dynamic_warp_id {}) fails as waiting for barrier",
                        warp_id, dyn_warp_id
                    );
                }
            }

            // println!("locking warp {}", warp_id);
            let warp = self
                .warps
                // .get_mut(warp_id)
                .get(warp_id)
                .unwrap();

            // todo: what is the difference? why dont we just use next_warp?
            drop(next_warp);
            let mut warp = warp.lock().unwrap();
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
                // dbg!(&checked);
                // dbg!(&issued);
                // dbg!(&max_issue);
                // let valid = warp.ibuffer_next_valid();
                let mut warp_inst_issued = false;

                if let Some(instr) = warp.ibuffer_next_inst() {
                    let (pc, rpc) = get_pdom_stack_top_info(warp_id, instr);
                    println!(
                        "Warp (warp_id {}, dynamic_warp_id {}) has valid instruction ({})",
                        warp_id,
                        dyn_warp_id,
                        // instr
                        // m_shader->m_config->gpgpu_ctx->func_sim->ptx_get_insn_str(pc).c_str()
                        "todo",
                    );

                    if pc != instr.pc {
                        println!(
                            "Warp (warp_id {}, dynamic_warp_id {}) control hazard instruction flush",
                            warp_id, dyn_warp_id);
                        // control hazard
                        warp.set_next_pc(pc);
                        warp.ibuffer_flush();
                    } else {
                        valid_inst = true;
                        if !self.scoreboard.check_collision(warp_id, instr) {
                            println!(
                                "Warp (warp_id {}, dynamic_warp_id {}) passes scoreboard",
                                warp_id, dyn_warp_id
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
                                    // if self
                                    //     .mem_out
                                    //     .has_free_sub_core(self.config.sub_core_model, self.id)
                                    //     && (!diff_exec_units
                                    //         || prev_issued_exec_unit != ExecUnitKind::MEM)
                                    // {
                                    //     core.issue_warp(
                                    //         mem_out,
                                    //         instr,
                                    //         active_mask,
                                    //         next_warp.warp_id,
                                    //         self.id,
                                    //     );
                                    //     issued += 1;
                                    //     issued_inst = true;
                                    //     warp_inst_issued = true;
                                    //     prev_issued_exec_unit = ExecUnitKind::MEM;
                                    // }
                                }
                                op => unimplemented!("op {:?} no implemented", op),
                            }
                        } else {
                            println!(
                                "Warp (warp_id {}, dynamic_warp_id {}) fails scoreboard",
                                warp_id, dyn_warp_id
                            );
                        }
                    }
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
                        "Warp (warp_id {}, dynamic_warp_id {}) issued {} instructions",
                        warp_id, dyn_warp_id, issued
                    );
                    // m_stats->event_warp_issued(m_shader->get_sid(), warp_id, num_issued, warp(warp_id).get_dynamic_warp_id());
                    warp.ibuffer_step();
                    // self.do_on_warp_issued(next_warp.warp_id, issued, next_warp);
                }
                checked += 1;
            }
            if issued > 0 {
                // This might be a bit inefficient, but we need to maintain
                // two ordered list for proper scheduler execution.
                // We could remove the need for this loop by associating a
                // supervised_is index with each entry in the
                // m_next_cycle_prioritized_warps vector.
                // For now, just run through until you find the right warp_id
                for (sup_idx, supervised) in self.supervised_warps.iter().enumerate() {
                    // if *next_warp == *supervised.lock().unwrap().warp_id {
                    if warp_id == supervised.lock().unwrap().warp_id {
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
                println!("WARN: issued should be > 0");
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
    // fn cycle<I>(&mut self, core: &mut super::core::InnerSIMTCore<I>) {
    fn cycle(&mut self, core: ()) {
        // fn cycle(&mut self) {
        todo!("scheduler unit: cycle");
    }

    fn done_adding_supervised_warps(&mut self) {
        todo!("scheduler unit: done_adding_supervised_warps");
    }

    fn add_supervised_warp(&mut self, warp: CoreWarp) {
        todo!("scheduler unit: add supervised warp id");
    }

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
// pub struct LrrScheduler<'a> {
//     inner: BaseSchedulerUnit<'a>,
// }

impl BaseSchedulerUnit {
    // impl<'a> BaseSchedulerUnit<'a> {
    fn order_by_priority(&mut self) {
        todo!("base scheduler unit: order by priority");
    }

    fn order_rrr(
        &mut self,
        // out: &mut VecDeque<SchedulerWarp>,
        // warps: &mut Vec<SchedulerWarp>,
        // std::vector<T> &result_list, const typename std::vector<T> &input_list,
        // const typename std::vector<T>::const_iterator &last_issued_from_input,
        // unsigned num_warps_to_add)
    ) {
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
        let current_turn_warp = current_turn_warp_ref.lock().unwrap();
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
                let warp = w.lock().unwrap();
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
        let num_warps_to_add = self.supervised_warps.len();
        let out = &mut self.next_cycle_prioritized_warps;

        debug_assert!(num_warps_to_add <= self.warps.len());
        out.clear();
        // if last_issued_warps
        //   typename std::vector<T>::const_iterator iter = (last_issued_from_input == input_list.end()) ? input_list.begin()
        //                                                    : last_issued_from_input + 1;
        //
        let mut last_iter = self.warps.iter().skip(self.last_supervised_issued_idx);

        let mut iter = last_iter.chain(self.warps.iter());
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

    // fn add_supervised_warp_id(&mut self, warp_id: usize) {
    //     self.inner.add_supervised_warp_id(warp_id);
    // }

    fn done_adding_supervised_warps(&mut self) {
        self.inner.last_supervised_issued_idx = self.inner.supervised_warps.len();
    }

    // fn cycle<I>(&mut self, core: &mut super::core::InnerSIMTCore<I>) {
    fn cycle(&mut self, core: ()) {
        println!("lrr scheduler: cycle enter");
        self.order_warps();
        self.inner.cycle(core);
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
        scoreboard: Arc<scoreboard::Scoreboard>,
        stats: Arc<Mutex<Stats>>,
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
pub struct GTOScheduler {}

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
    use playground::{bindings, bridge};
    use std::ptr;

    #[test]
    fn test_shd_warp() {
        use bridge::trace_shd_warp::new_trace_shd_warp;
        let core = ptr::null_mut();
        let warp_size = 32;
        let mut warp = unsafe { new_trace_shd_warp(core, warp_size) };
        warp.pin_mut().reset();
        dbg!(&warp.get_n_completed());
        dbg!(&warp.hardware_done());
        dbg!(&warp.functional_done());
        assert!(false);
    }
}
