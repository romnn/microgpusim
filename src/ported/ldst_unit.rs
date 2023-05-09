use crate::config::GPUConfig;
use crate::ported::mem_fetch::AccessKind;

use super::{
    address,
    instruction::{CacheOperator, WarpInstruction},
    mem_fetch::MemFetch,
};
use nvbit_model::MemorySpace;
use std::collections::{HashMap, VecDeque};
use trace_model::MemAccessTraceEntry;

/// A general function to order things in an priority-based way.
/// The core usage of the function is similar to order_lrr.
/// The explanation of the additional parameters (beyond order_lrr)
/// explains the further extensions.
///
/// **Ordering**:
/// An enum that determines how the age function will be treated
/// in prioritization see the definition of OrderingType.
///
/// **priority_function**:
/// This function is used to sort the input_list.
/// It is passed to stl::sort as the sorting fucntion.
/// So, if you wanted to sort a list of integer warp_ids with the
/// oldest warps having the most priority, then the priority_function
/// would compare the age of the two warps.
///
// fn order_by_priority<T>(
//     warps: Vec<T>,
//     last_issued_from_input: Vec<T>,
//     count: usize,
//     // ordering: OrderingKind,
//     priority_func: String,
// ) {
// }
//
// template <class T>
// void scheduler_unit::order_by_priority(
//     std::vector<T> &result_list, const typename std::vector<T> &input_list,
//     const typename std::vector<T>::const_iterator &last_issued_from_input,
//     unsigned num_warps_to_add, OrderingType ordering,
//     bool (*priority_func)(T lhs, T rhs)) {
//   assert(num_warps_to_add <= input_list.size());
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
// }

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SchedulerWarp {
    block_id: usize,
    dynamic_warp_id: usize,
    warp_id: usize,
    done: bool,
}

impl SchedulerWarp {
    pub fn ibuffer_empty(&self) -> bool {
        self.done
    }

    pub fn done_exit(&self) -> bool {
        self.done
    }

    pub fn waiting(&self) -> bool {
        false
        //       if (functional_done()) {
        //   // waiting to be initialized with a kernel
        //   return true;
        // } else if (m_shader->warp_waiting_at_barrier(m_warp_id)) {
        //   // waiting for other warps in CTA to reach barrier
        //   return true;
        // } else if (m_shader->warp_waiting_at_mem_barrier(m_warp_id)) {
        //   // waiting for memory barrier
        //   return true;
        // } else if (m_n_atomic > 0) {
        //   // waiting for atomic operation to complete at memory:
        //   // this stall is not required for accurate timing model, but rather we
        //   // stall here since if a call/return instruction occurs in the meantime
        //   // the functional execution of the atomic when it hits DRAM can cause
        //   // the wrong register to be read.
        //   return true;
        // }
        // return false;
    }

    pub fn dynamic_warp_id(&self) -> usize {
        self.dynamic_warp_id
    }
}

pub trait SchedulerPolicy {
    fn order_warps(&self);
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

#[derive(Debug)]
pub struct GTOScheduler {}

impl GTOScheduler {
    fn order_warps(
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
        let greedy_value = last_issued_warps.first();
        if let Some(greedy_value) = greedy_value {
            out.push_back(greedy_value.clone());
        }

        warps.sort_by(sort_warps_by_oldest_dynamic_id);
        out.extend(
            warps
                .iter()
                .take_while(|w| match greedy_value {
                    None => true,
                    Some(val) => *w != val,
                })
                .take(num_warps_to_add)
                .cloned(),
        );

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

#[derive(Debug)]
pub struct SchedulerUnit {
    id: usize,
    /// This is the prioritized warp list that is looped over each cycle to
    /// determine which warp gets to issue.
    next_cycle_prioritized_warps: VecDeque<SchedulerWarp>,
    // The m_supervised_warps list is all the warps this scheduler is
    // supposed to arbitrate between.
    // This is useful in systems where there is more than one warp scheduler.
    // In a single scheduler system, this is simply all the warps
    // assigned to this core.
    supervised_warps: Vec<SchedulerWarp>,
    /// This is the iterator pointer to the last supervised warp you issued
    last_supervised_issued: Vec<SchedulerWarp>,
    scheduler: GTOScheduler,
    warps: Vec<SchedulerWarp>,
    // register_set *m_mem_out;
    // std::vector<register_set *> &m_spec_cores_out;
    num_issued_last_cycle: usize,
}

impl SchedulerUnit {
    pub fn cycle(&mut self) {
        println!("scheduler unit cycle");
        // there was one warp with a valid instruction to issue (didn't require flush due to control hazard)
        let valid_inst = false;
        // of the valid instructions, there was one not waiting for pending register writes
        let ready_inst = false;
        // of these we issued one
        let issued_inst = false;

        let num_warps_to_add = self.supervised_warps.len();
        self.scheduler.order_warps(
            &mut self.next_cycle_prioritized_warps,
            &mut self.supervised_warps,
            &self.last_supervised_issued,
            num_warps_to_add,
        );
        for warp in &self.next_cycle_prioritized_warps {
            println!(
                "testing (warp_id {}, dynamic_warp_id {})",
                warp.warp_id, warp.dynamic_warp_id,
            );
            // let warp_id = warp.warp_id;
            let checked = 0;
            let issued = 0;

            if warp.ibuffer_empty() {
                println!(
                    "warp (warp_id {}, dynamic_warp_id {}) fails as ibuffer_empty",
                    warp.warp_id, warp.dynamic_warp_id
                );
            }

            if warp.waiting() {
                println!(
                    "warp (warp_id {}, dynamic_warp_id {}) fails as waiting for barrier",
                    warp.warp_id, warp.dynamic_warp_id
                );
            }
        }
    }
}

/// Register that can hold multiple instructions.
#[derive(Debug)]
pub struct RegisterSet {
    name: String,
    regs: Vec<Option<WarpInstruction>>,
}

impl RegisterSet {
    pub fn new(size: usize, name: String) -> Self {
        let regs = (0..size).map(|_| None).collect();
        Self { regs, name }
    }

    pub fn has_free(&self) -> bool {
        self.regs.iter().any(|r| match r {
            Some(r) => r.empty,
            None => true,
        })
    }

    pub fn has_free_sub_core(&self, sub_core_model: bool, reg_id: usize) -> bool {
        // in subcore model, each sched has a one specific
        // reg to use (based on sched id)
        if !sub_core_model {
            return self.has_free();
        }

        debug_assert!(reg_id < self.regs.len());
        self.regs
            .get(reg_id)
            .and_then(Option::as_ref)
            .map(|r| r.empty)
            .unwrap_or(false)
    }

    pub fn has_ready(&self) -> bool {
        self.regs.iter().any(|r| match r {
            Some(r) => !r.empty,
            None => false,
        })
    }

    // pub fn has_ready_sub_core(&self, sub_core_model: bool, reg_id: usize) -> bool {
    pub fn has_ready_sub_core(&self, reg_id: usize) -> bool {
        // if !sub_core_model {
        //     return self.has_ready();
        // }

        debug_assert!(reg_id < self.regs.len());
        match self.get_ready_sub_core(reg_id) {
            Some(ready) => !ready.empty,
            None => true,
        }
    }

    pub fn ready_reg_id(&self) -> Option<usize> {
        // for sub core model we need to figure which reg_id has
        // the ready warp this function should only be called
        // if has_ready() was true
        debug_assert!(self.has_ready());
        let mut non_empty = self
            .regs
            .iter()
            .map(Option::as_ref)
            .filter_map(|r| r)
            .filter(|r| !r.empty);

        let mut ready: Option<&WarpInstruction> = None;
        let mut reg_id = None;
        for (i, reg) in non_empty.enumerate() {
            match ready {
                Some(ready) if ready.warp_id < reg.warp_id => {
                    // ready is oldest
                }
                _ => {
                    ready.insert(reg);
                    reg_id = Some(i);
                }
            }
        }
        reg_id
    }

    // pub fn schd_id(&self, reg_id: usize) -> usize {
    //     debug_assert!(!self.regs[reg_id].empty);
    //     self.regs[reg_id].schd_id()
    // }

    pub fn get_ready_sub_core(&self, reg_id: usize) -> Option<&WarpInstruction> {
        debug_assert!(reg_id < self.regs.len());
        self.regs
            .get(reg_id)
            .and_then(Option::as_ref)
            .filter(|r| r.empty)
    }

    pub fn get_free(&self) -> Option<&WarpInstruction> {
        let mut free = self
            .regs
            .iter()
            .filter_map(|r| r.as_ref())
            .filter(|r| r.empty);
        free.next()
    }

    pub fn get_free_sub_core(&self, reg_id: usize) -> Option<&WarpInstruction> {
        // in subcore model, each sched has a one specific reg
        // to use (based on sched id)
        debug_assert!(reg_id < self.regs.len());
        self.regs
            .get(reg_id)
            .and_then(Option::as_ref)
            .filter(|r| r.empty)
    }

    pub fn size(&self) -> usize {
        self.regs.len()
    }

    pub fn empty(&self) -> bool {
        false
    }
}

impl std::fmt::Display for RegisterSet {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_list().entries(self.regs.iter()).finish()
    }
}

//   void move_in(warp_inst_t *&src) {
//     warp_inst_t **free = get_free();
//     move_warp(*free, src);
//   }
//
//   void move_in(bool sub_core_model, unsigned reg_id, warp_inst_t *&src) {
//     warp_inst_t **free;
//     if (!sub_core_model) {
//       free = get_free();
//     } else {
//       assert(reg_id < regs.size());
//       free = get_free(sub_core_model, reg_id);
//     }
//     move_warp(*free, src);
//   }
//
//   void move_out_to(warp_inst_t *&dest) {
//     warp_inst_t **ready = get_ready();
//     move_warp(dest, *ready);
//   }
//
//   void move_out_to(bool sub_core_model, unsigned reg_id, warp_inst_t *&dest) {
//     if (!sub_core_model) {
//       return move_out_to(dest);
//     }
//     warp_inst_t **ready = get_ready(sub_core_model, reg_id);
//     assert(ready != NULL);
//     move_warp(dest, *ready);
//   }
//
//   warp_inst_t **get_ready() {
//     warp_inst_t **ready;
//     ready = NULL;
//     for (unsigned i = 0; i < regs.size(); i++) {
//       if (not regs[i]->empty()) {
//         if (ready and (*ready)->get_uid() < regs[i]->get_uid()) {
//           // ready is oldest
//         } else {
//           ready = &regs[i];
//         }
//       }
//     }
//     return ready;
//   }

// fn move_warp(lhs: &PipelineStage, rhs: &PipelineStage) {}

#[derive(Debug)]
pub struct L1Cache {}

impl L1Cache {
    pub fn cycle(&mut self) {}

    pub fn fill(&self, fetch: &MemFetch) {}

    pub fn has_free_fill_port(&self) -> bool {
        false
    }
}

pub trait MemFetchInterconnect {
    fn full(&self, size: usize, write: bool) -> bool;
    fn push(&mut self, fetch: MemFetch);
}

#[derive(Debug)]
pub struct Interconnect {}

impl Interconnect {
    pub fn push(&mut self, fetch: MemFetch) {}

    pub fn full(&self, size: usize, write: bool) -> bool {
        false
    }
}

pub static READ_PACKET_SIZE: usize = 8;

// bytes: 6 address, 2 miscelaneous.
pub static WRITE_PACKET_SIZE: usize = 8;

pub static WRITE_MASK_SIZE: usize = 8;

#[derive(Debug)]
pub struct LoadStoreUnit {
    pipeline_depth: usize,
    pipeline_reg: Vec<RegisterSet>,
    response_fifo: VecDeque<MemFetch>,
    texture_l1: L1Cache,
    const_l1: L1Cache,
    data_l1: Option<L1Cache>,
    config: GPUConfig,
    next_global: Option<MemFetch>,
    dispatch_reg: WarpInstruction,
    /// Pending writes warp -> register -> count
    pending_writes: HashMap<usize, HashMap<u32, usize>>,
    interconn: Interconnect,
}

impl LoadStoreUnit {
    pub fn writeback(&self) {}

    pub fn cycle(&mut self) {
        use super::instruction::CacheOperator;

        self.writeback();
        for stage in 0..(self.pipeline_depth - 1) {
            let current = &self.pipeline_reg[stage];
            let next = &self.pipeline_reg[stage + 1];
            if current.empty() && !next.empty() {
                // move_warp(current, next);
            }
        }

        if let Some(fetch) = self.response_fifo.front().cloned() {
            match fetch.access_kind() {
                AccessKind::TEXTURE_ACC_R => {
                    if self.texture_l1.has_free_fill_port() {
                        self.texture_l1.fill(&fetch);
                        // self.response_fifo.fill(mem_fetch);
                        self.response_fifo.pop_front();
                    }
                }
                AccessKind::CONST_ACC_R => {
                    if self.const_l1.has_free_fill_port() {
                        // fetch.set_status(IN_SHADER_FETCHED)
                        self.const_l1.fill(&fetch);
                        // self.response_fifo.fill(mem_fetch);
                        self.response_fifo.pop_front();
                    }
                }
                _ => {
                    if fetch.kind == super::MemFetchKind::WRITE_ACK
                        || (self.config.perfect_mem && fetch.is_write())
                    {
                        // m_core->store_ack(mf);
                        self.response_fifo.pop_front();
                    } else {
                        // L1 cache is write evict:
                        // allocate line on load miss only
                        debug_assert!(fetch.is_write());
                        let mut bypass_l1 = false;

                        if self.data_l1.is_none()
                            || fetch.instr.cache_operator == CacheOperator::GLOBAL
                        {
                            bypass_l1 = true;
                        } else if fetch.access_kind() == AccessKind::GLOBAL_ACC_R
                            || fetch.access_kind() == AccessKind::GLOBAL_ACC_W
                        {
                            // global memory access
                            if self.config.global_mem_skip_l1_data_cache {
                                bypass_l1 = true;
                            }
                        }

                        match &self.data_l1 {
                            Some(l1d) if !bypass_l1 => {
                                if l1d.has_free_fill_port() {
                                    l1d.fill(&fetch);
                                    self.response_fifo.pop_front();
                                }
                            }
                            _ => {
                                if self.next_global.is_none() {
                                    // fetch.set_status(IN_SHADER_FETCHED);
                                    self.response_fifo.pop_front();
                                    self.next_global.insert(fetch.clone());
                                }
                            }
                        }
                    }
                }
            }
        }

        self.texture_l1.cycle();
        self.const_l1.cycle();
        if let Some(data_l1) = &mut self.data_l1 {
            data_l1.cycle();
            // if (m_config->m_L1D_config.l1_latency > 0) L1_latency_queue_cycle();
        }

        // let pipe_reg = &self.dispatch_reg;
        // enum mem_stage_stall_type rc_fail = NO_RC_FAIL;
        // mem_stage_access_type type;
        // bool done = true;
        // done &= shared_cycle(pipe_reg, rc_fail, type);
        // done &= constant_cycle(pipe_reg, rc_fail, type);
        // done &= texture_cycle(pipe_reg, rc_fail, type);
        // done &= memory_cycle(pipe_reg, rc_fail, type);
        self.memory_cycle(); // &self.dispatch_reg);
                             // m_mem_rc = rc_fail;
        let done = true;
        let mut num_stall_scheduler_mem = 0;
        if !done {
            // log stall types and return
            // debug_assert!(rc_fail != NO_RC_FAIL);
            num_stall_scheduler_mem += 1;
            // m_stats->gpu_stall_shd_mem_breakdown[type][rc_fail]++;
            return;
        }

        let pipe_reg = &self.dispatch_reg;
        if !pipe_reg.empty {
            let warp_id = pipe_reg.warp_id;
            if pipe_reg.is_load() {
                if pipe_reg.memory_space == MemorySpace::Shared {
                    let slot = &self.pipeline_reg[self.config.shared_memory_latency - 1];
                    if slot.empty() {
                        // new shared memory request
                        // move_warp(&slot, self.dispatch_reg);
                        self.dispatch_reg.clear();
                    }
                } else {
                    let mut pending_requests = false;
                    for reg_id in pipe_reg.outputs {
                        let mut pending = self.pending_writes.get_mut(&warp_id).unwrap();
                        if reg_id > 0 {
                            match pending.get(&reg_id) {
                                Some(&p) if p > 0 => {
                                    pending_requests = true;
                                    break;
                                }
                                _ => {
                                    // this instruction is done already
                                    pending.remove(&reg_id);
                                }
                            }
                        }
                    }
                    if !pending_requests {
                        // core.warp_inst_complete(self.dispatch_reg);
                        // self.scoreboard.release_registers(self.dispatch_reg);
                    }
                    // core.dec_inst_in_pipeline(warp_id);
                    self.dispatch_reg.clear();
                }
            } else {
                // stores exit pipeline here
                // core.dec_inst_in_pipeline(warp_id);
                // core.warp_inst_complete(self.dispatch_reg);
                self.dispatch_reg.clear();
            }
        }
    }

    fn shared_cycle(&mut self) {}

    fn constant_cycle(&mut self) {}

    fn texture_cycle(&mut self) {}

    // fn memory_cycle(&mut self, instr: &WarpInstruction) -> bool {
    fn memory_cycle(&mut self) -> bool {
        let instr = &mut self.dispatch_reg;
        if instr.empty
            || (instr.memory_space != MemorySpace::Global
                && instr.memory_space != MemorySpace::Local)
        {
            return true;
        }
        if instr.active_thread_count() == 0 {
            return true;
        }
        if instr.mem_access_queue.is_empty() {
            return true;
        }

        // mem_stage_stall_type stall_cond = NO_RC_FAIL;
        let Some(access) = instr.mem_access_queue.back() else {
            return true;
        };
        let mut bypass_l1 = false;

        if self.data_l1.is_none() || instr.cache_operator == CacheOperator::GLOBAL {
            bypass_l1 = true;
        } else if instr.memory_space == MemorySpace::Global {
            // global memory access
            // skip L1 cache if the option is enabled
            if self.config.global_mem_skip_l1_data_cache
                && instr.cache_operator != CacheOperator::L1
            {
                bypass_l1 = true;
            }
        }

        if bypass_l1 {
            // bypass L1 cache
            let control_size = if instr.is_store() {
                WRITE_PACKET_SIZE
            } else {
                READ_PACKET_SIZE
            };
            let size = access.req_size + control_size;

            println!("Interconnect addr: {}, size={}", access.addr, size);
            if self
                .interconn
                .full(size, instr.is_store() || instr.is_atomic())
            {
                // stall_cond = ICNT_RC_FAIL;
            } else {
                let fetch = MemFetch::alloc(instr.clone(), access.clone());
                self.interconn.push(fetch);
                instr.mem_access_queue.pop_back();
                // // inst.clear_active( access.get_warp_mask() );
                // if (inst.is_load()) {
                //   for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++)
                //     if (inst.out[r] > 0)
                //       assert(m_pending_writes[inst.warp_id()][inst.out[r]] > 0);
                // } else if (inst.is_store())
                //   m_core->inc_store_req(inst.warp_id());
            }
        } else {
            debug_assert!(instr.cache_operator != CacheOperator::UNDEFINED);
            // stall_cond = process_memory_access_queue_l1cache(m_L1D, inst);
        }

        false
    }
}

//   if (!inst.accessq_empty() && stall_cond == NO_RC_FAIL)
//     stall_cond = COAL_STALL;
//   if (stall_cond != NO_RC_FAIL) {
//     stall_reason = stall_cond;
//     bool iswrite = inst.is_store();
//     if (inst.space.is_local())
//       access_type = (iswrite) ? L_MEM_ST : L_MEM_LD;
//     else
//       access_type = (iswrite) ? G_MEM_ST : G_MEM_LD;
//   }
//   return inst.accessq_empty();

// void ldst_unit::cycle() {
//   writeback();
//
//   for (unsigned stage = 0; (stage + 1) < m_pipeline_depth; stage++)
//     if (m_pipeline_reg[stage]->empty() && !m_pipeline_reg[stage + 1]->empty())
//       move_warp(m_pipeline_reg[stage], m_pipeline_reg[stage + 1]);
//
//   if (!m_response_fifo.empty()) {
//     mem_fetch *mf = m_response_fifo.front();
//     if (mf->get_access_type() == TEXTURE_ACC_R) {
//       if (m_L1T->fill_port_free()) {
//         m_L1T->fill(mf, m_core->get_gpu()->gpu_sim_cycle +
//                             m_core->get_gpu()->gpu_tot_sim_cycle);
//         m_response_fifo.pop_front();
//       }
//     } else if (mf->get_access_type() == CONST_ACC_R) {
//       if (m_L1C->fill_port_free()) {
//         mf->set_status(IN_SHADER_FETCHED,
//                        m_core->get_gpu()->gpu_sim_cycle +
//                            m_core->get_gpu()->gpu_tot_sim_cycle);
//         m_L1C->fill(mf, m_core->get_gpu()->gpu_sim_cycle +
//                             m_core->get_gpu()->gpu_tot_sim_cycle);
//         m_response_fifo.pop_front();
//       }
//     } else {
//       if (mf->get_type() == WRITE_ACK ||
//           (m_config->gpgpu_perfect_mem && mf->get_is_write())) {
//         m_core->store_ack(mf);
//         m_response_fifo.pop_front();
//         delete mf;
//       } else {
//         assert(!mf->get_is_write());  // L1 cache is write evict, allocate line
//                                       // on load miss only
//
//         bool bypassL1D = false;
//         if (CACHE_GLOBAL == mf->get_inst().cache_op || (m_L1D == NULL)) {
//           bypassL1D = true;
//         } else if (mf->get_access_type() == GLOBAL_ACC_R ||
//                    mf->get_access_type() ==
//                        GLOBAL_ACC_W) {  // global memory access
//           if (m_core->get_config()->gmem_skip_L1D) bypassL1D = true;
//         }
//         if (bypassL1D) {
//           if (m_next_global == NULL) {
//             mf->set_status(IN_SHADER_FETCHED,
//                            m_core->get_gpu()->gpu_sim_cycle +
//                                m_core->get_gpu()->gpu_tot_sim_cycle);
//             m_response_fifo.pop_front();
//             m_next_global = mf;
//           }
//         } else {
//           if (m_L1D->fill_port_free()) {
//             m_L1D->fill(mf, m_core->get_gpu()->gpu_sim_cycle +
//                                 m_core->get_gpu()->gpu_tot_sim_cycle);
//             m_response_fifo.pop_front();
//           }
//         }
//       }
//     }
//   }
//
//   m_L1T->cycle();
//   m_L1C->cycle();
//   if (m_L1D) {
//     m_L1D->cycle();
//     if (m_config->m_L1D_config.l1_latency > 0) L1_latency_queue_cycle();
//   }
//
//   warp_inst_t &pipe_reg = *m_dispatch_reg;
//   enum mem_stage_stall_type rc_fail = NO_RC_FAIL;
//   mem_stage_access_type type;
//   bool done = true;
//   done &= shared_cycle(pipe_reg, rc_fail, type);
//   done &= constant_cycle(pipe_reg, rc_fail, type);
//   done &= texture_cycle(pipe_reg, rc_fail, type);
//   done &= memory_cycle(pipe_reg, rc_fail, type);
//   m_mem_rc = rc_fail;
//
//   if (!done) {  // log stall types and return
//     assert(rc_fail != NO_RC_FAIL);
//     m_stats->gpgpu_n_stall_shd_mem++;
//     m_stats->gpu_stall_shd_mem_breakdown[type][rc_fail]++;
//     return;
//   }
//
//   if (!pipe_reg.empty()) {
//     unsigned warp_id = pipe_reg.warp_id();
//     if (pipe_reg.is_load()) {
//       if (pipe_reg.space.get_type() == shared_space) {
//         if (m_pipeline_reg[m_config->smem_latency - 1]->empty()) {
//           // new shared memory request
//           move_warp(m_pipeline_reg[m_config->smem_latency - 1], m_dispatch_reg);
//           m_dispatch_reg->clear();
//         }
//       } else {
//         // if( pipe_reg.active_count() > 0 ) {
//         //    if( !m_operand_collector->writeback(pipe_reg) )
//         //        return;
//         //}
//
//         bool pending_requests = false;
//         for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
//           unsigned reg_id = pipe_reg.out[r];
//           if (reg_id > 0) {
//             if (m_pending_writes[warp_id].find(reg_id) !=
//                 m_pending_writes[warp_id].end()) {
//               if (m_pending_writes[warp_id][reg_id] > 0) {
//                 pending_requests = true;
//                 break;
//               } else {
//                 // this instruction is done already
//                 m_pending_writes[warp_id].erase(reg_id);
//               }
//             }
//           }
//         }
//         if (!pending_requests) {
//           m_core->warp_inst_complete(*m_dispatch_reg);
//           m_scoreboard->releaseRegisters(m_dispatch_reg);
//         }
//         m_core->dec_inst_in_pipeline(warp_id);
//         m_dispatch_reg->clear();
//       }
//     } else {
//       // stores exit pipeline here
//       m_core->dec_inst_in_pipeline(warp_id);
//       m_core->warp_inst_complete(*m_dispatch_reg);
//       m_dispatch_reg->clear();
//     }
//   }
// }

// bool ldst_unit::shared_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
//                              mem_stage_access_type &fail_type) {
//   if (inst.space.get_type() != shared_space) return true;
//
//   if (inst.active_count() == 0) return true;
//
//   if (inst.has_dispatch_delay()) {
//     m_stats->gpgpu_n_shmem_bank_access[m_sid]++;
//   }
//
//   bool stall = inst.dispatch_delay();
//   if (stall) {
//     fail_type = S_MEM;
//     rc_fail = BK_CONF;
//   } else
//     rc_fail = NO_RC_FAIL;
//   return !stall;
// }
//
// mem_stage_stall_type ldst_unit::process_cache_access(
//     cache_t *cache, new_addr_type address, warp_inst_t &inst,
//     std::list<cache_event> &events, mem_fetch *mf,
//     enum cache_request_status status) {
//   mem_stage_stall_type result = NO_RC_FAIL;
//   bool write_sent = was_write_sent(events);
//   bool read_sent = was_read_sent(events);
//   if (write_sent) {
//     unsigned inc_ack = (m_config->m_L1D_config.get_mshr_type() == SECTOR_ASSOC)
//                            ? (mf->get_data_size() / SECTOR_SIZE)
//                            : 1;
//
//     for (unsigned i = 0; i < inc_ack; ++i)
//       m_core->inc_store_req(inst.warp_id());
//   }
//   if (status == HIT) {
//     assert(!read_sent);
//     inst.accessq_pop_back();
//     if (inst.is_load()) {
//       for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++)
//         if (inst.out[r] > 0) m_pending_writes[inst.warp_id()][inst.out[r]]--;
//     }
//     if (!write_sent) delete mf;
//   } else if (status == RESERVATION_FAIL) {
//     result = BK_CONF;
//     assert(!read_sent);
//     assert(!write_sent);
//     delete mf;
//   } else {
//     assert(status == MISS || status == HIT_RESERVED);
//     // inst.clear_active( access.get_warp_mask() ); // threads in mf writeback
//     // when mf returns
//     inst.accessq_pop_back();
//   }
//   if (!inst.accessq_empty() && result == NO_RC_FAIL) result = COAL_STALL;
//   return result;
// }
//
// mem_stage_stall_type ldst_unit::process_memory_access_queue(cache_t *cache,
//                                                             warp_inst_t &inst) {
//   mem_stage_stall_type result = NO_RC_FAIL;
//   if (inst.accessq_empty()) return result;
//
//   if (!cache->data_port_free()) return DATA_PORT_STALL;
//
//   // const mem_access_t &access = inst.accessq_back();
//   mem_fetch *mf = m_mf_allocator->alloc(
//       inst, inst.accessq_back(),
//       m_core->get_gpu()->gpu_sim_cycle + m_core->get_gpu()->gpu_tot_sim_cycle);
//   std::list<cache_event> events;
//   enum cache_request_status status = cache->access(
//       mf->get_addr(), mf,
//       m_core->get_gpu()->gpu_sim_cycle + m_core->get_gpu()->gpu_tot_sim_cycle,
//       events);
//   return process_cache_access(cache, mf->get_addr(), inst, events, mf, status);
// }
//
// mem_stage_stall_type ldst_unit::process_memory_access_queue_l1cache(
//     l1_cache *cache, warp_inst_t &inst) {
//   mem_stage_stall_type result = NO_RC_FAIL;
//   if (inst.accessq_empty()) return result;
//
//   if (m_config->m_L1D_config.l1_latency > 0) {
//     for (int j = 0; j < m_config->m_L1D_config.l1_banks;
//          j++) {  // We can handle at max l1_banks reqs per cycle
//
//       if (inst.accessq_empty()) return result;
//
//       mem_fetch *mf =
//           m_mf_allocator->alloc(inst, inst.accessq_back(),
//                                 m_core->get_gpu()->gpu_sim_cycle +
//                                     m_core->get_gpu()->gpu_tot_sim_cycle);
//       unsigned bank_id = m_config->m_L1D_config.set_bank(mf->get_addr());
//       assert(bank_id < m_config->m_L1D_config.l1_banks);
//
//       if ((l1_latency_queue[bank_id][m_config->m_L1D_config.l1_latency - 1]) ==
//           NULL) {
//         l1_latency_queue[bank_id][m_config->m_L1D_config.l1_latency - 1] = mf;
//
//         if (mf->get_inst().is_store()) {
//           unsigned inc_ack =
//               (m_config->m_L1D_config.get_mshr_type() == SECTOR_ASSOC)
//                   ? (mf->get_data_size() / SECTOR_SIZE)
//                   : 1;
//
//           for (unsigned i = 0; i < inc_ack; ++i)
//             m_core->inc_store_req(inst.warp_id());
//         }
//
//         inst.accessq_pop_back();
//       } else {
//         result = BK_CONF;
//         delete mf;
//         break;  // do not try again, just break from the loop and try the next
//                 // cycle
//       }
//     }
//     if (!inst.accessq_empty() && result != BK_CONF) result = COAL_STALL;
//
//     return result;
//   } else {
//     mem_fetch *mf =
//         m_mf_allocator->alloc(inst, inst.accessq_back(),
//                               m_core->get_gpu()->gpu_sim_cycle +
//                                   m_core->get_gpu()->gpu_tot_sim_cycle);
//     std::list<cache_event> events;
//     enum cache_request_status status = cache->access(
//         mf->get_addr(), mf,
//         m_core->get_gpu()->gpu_sim_cycle + m_core->get_gpu()->gpu_tot_sim_cycle,
//         events);
//     return process_cache_access(cache, mf->get_addr(), inst, events, mf,
//                                 status);
//   }
// }
//
// void ldst_unit::L1_latency_queue_cycle() {
//   for (int j = 0; j < m_config->m_L1D_config.l1_banks; j++) {
//     if ((l1_latency_queue[j][0]) != NULL) {
//       mem_fetch *mf_next = l1_latency_queue[j][0];
//       std::list<cache_event> events;
//       enum cache_request_status status =
//           m_L1D->access(mf_next->get_addr(), mf_next,
//                         m_core->get_gpu()->gpu_sim_cycle +
//                             m_core->get_gpu()->gpu_tot_sim_cycle,
//                         events);
//
//       bool write_sent = was_write_sent(events);
//       bool read_sent = was_read_sent(events);
//
//       if (status == HIT) {
//         assert(!read_sent);
//         l1_latency_queue[j][0] = NULL;
//         if (mf_next->get_inst().is_load()) {
//           for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++)
//             if (mf_next->get_inst().out[r] > 0) {
//               assert(m_pending_writes[mf_next->get_inst().warp_id()]
//                                      [mf_next->get_inst().out[r]] > 0);
//               unsigned still_pending =
//                   --m_pending_writes[mf_next->get_inst().warp_id()]
//                                     [mf_next->get_inst().out[r]];
//               if (!still_pending) {
//                 m_pending_writes[mf_next->get_inst().warp_id()].erase(
//                     mf_next->get_inst().out[r]);
//                 m_scoreboard->releaseRegister(mf_next->get_inst().warp_id(),
//                                               mf_next->get_inst().out[r]);
//                 m_core->warp_inst_complete(mf_next->get_inst());
//               }
//             }
//         }
//
//         // For write hit in WB policy
//         if (mf_next->get_inst().is_store() && !write_sent) {
//           unsigned dec_ack =
//               (m_config->m_L1D_config.get_mshr_type() == SECTOR_ASSOC)
//                   ? (mf_next->get_data_size() / SECTOR_SIZE)
//                   : 1;
//
//           mf_next->set_reply();
//
//           for (unsigned i = 0; i < dec_ack; ++i) m_core->store_ack(mf_next);
//         }
//
//         if (!write_sent) delete mf_next;
//
//       } else if (status == RESERVATION_FAIL) {
//         assert(!read_sent);
//         assert(!write_sent);
//       } else {
//         assert(status == MISS || status == HIT_RESERVED);
//         l1_latency_queue[j][0] = NULL;
//         if (m_config->m_L1D_config.get_write_policy() != WRITE_THROUGH &&
//             mf_next->get_inst().is_store() &&
//             (m_config->m_L1D_config.get_write_allocate_policy() ==
//                  FETCH_ON_WRITE ||
//              m_config->m_L1D_config.get_write_allocate_policy() ==
//                  LAZY_FETCH_ON_READ) &&
//             !was_writeallocate_sent(events)) {
//           unsigned dec_ack =
//               (m_config->m_L1D_config.get_mshr_type() == SECTOR_ASSOC)
//                   ? (mf_next->get_data_size() / SECTOR_SIZE)
//                   : 1;
//           mf_next->set_reply();
//           for (unsigned i = 0; i < dec_ack; ++i) m_core->store_ack(mf_next);
//           if (!write_sent && !read_sent) delete mf_next;
//         }
//       }
//     }
//
//     for (unsigned stage = 0; stage < m_config->m_L1D_config.l1_latency - 1;
//          ++stage)
//       if (l1_latency_queue[j][stage] == NULL) {
//         l1_latency_queue[j][stage] = l1_latency_queue[j][stage + 1];
//         l1_latency_queue[j][stage + 1] = NULL;
//       }
//   }
// }
//
// bool ldst_unit::constant_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
//                                mem_stage_access_type &fail_type) {
//   if (inst.empty() || ((inst.space.get_type() != const_space) &&
//                        (inst.space.get_type() != param_space_kernel)))
//     return true;
//   if (inst.active_count() == 0) return true;
//
//   mem_stage_stall_type fail;
//   if (m_config->perfect_inst_const_cache) {
//     fail = NO_RC_FAIL;
//     unsigned access_count = inst.accessq_count();
//     while (inst.accessq_count() > 0) inst.accessq_pop_back();
//     if (inst.is_load()) {
//       for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++)
//         if (inst.out[r] > 0) m_pending_writes[inst.warp_id()][inst.out[r]] -= access_count;
//     }
//   } else {
//     fail = process_memory_access_queue(m_L1C, inst);
//   }
//
//   if (fail != NO_RC_FAIL) {
//     rc_fail = fail;  // keep other fails if this didn't fail.
//     fail_type = C_MEM;
//     if (rc_fail == BK_CONF or rc_fail == COAL_STALL) {
//       m_stats->gpgpu_n_cmem_portconflict++;  // coal stalls aren't really a bank
//                                              // conflict, but this maintains
//                                              // previous behavior.
//     }
//   }
//   return inst.accessq_empty();  // done if empty.
// }
//
// bool ldst_unit::texture_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
//                               mem_stage_access_type &fail_type) {
//   if (inst.empty() || inst.space.get_type() != tex_space) return true;
//   if (inst.active_count() == 0) return true;
//   mem_stage_stall_type fail = process_memory_access_queue(m_L1T, inst);
//   if (fail != NO_RC_FAIL) {
//     rc_fail = fail;  // keep other fails if this didn't fail.
//     fail_type = T_MEM;
//   }
//   return inst.accessq_empty();  // done if empty.
// }

//
// bool ldst_unit::response_buffer_full() const {
//   return m_response_fifo.size() >= m_config->ldst_unit_response_queue_size;
// }
//
// void ldst_unit::fill(mem_fetch *mf) {
//   mf->set_status(
//       IN_SHADER_LDST_RESPONSE_FIFO,
//       m_core->get_gpu()->gpu_sim_cycle + m_core->get_gpu()->gpu_tot_sim_cycle);
//   m_response_fifo.push_back(mf);
// }
//
// void ldst_unit::flush() {
//   // Flush L1D cache
//   m_L1D->flush();
// }
//
// void ldst_unit::invalidate() {
//   // Flush L1D cache
//   m_L1D->invalidate();
// }
