use super::{
    cache, interconn as ic, l1, mem_fetch, operand_collector as opcoll, register_set::RegisterSet,
    scheduler as sched, stats::Stats,
};
use crate::{config::GPUConfig, ported::operand_collector::OperandCollectorRegisterFileUnit};
use bitvec::{array::BitArray, BitArr};
use console::style;
use ic::MemPort;
use std::sync::{Arc, Mutex};

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

pub trait SimdFunctionUnit {
    // modifiers
    fn cycle(&mut self);
    fn issue(&mut self, source_reg: RegisterSet);
    // fn compute_active_lanes_in_pipeline(&mut self);

    // accessors
    fn clock_multiplier(&self) -> usize {
        1
    }
    fn can_issue(&self, instr: &WarpInstruction) -> bool;
    fn active_lanes_in_pipeline(&self) -> usize;
    // {
    //     return m_dispatch_reg->empty() && !occupied.test(inst.latency);
    //   }
    fn is_issue_partitioned(&self) -> bool;
    fn issue_reg_id(&self) -> usize;
    fn stallable(&self) -> bool;

    // void simd_function_unit::issue(register_set &source_reg) {
    //   bool partition_issue =
    //       m_config->sub_core_model && this->is_issue_partitioned();
    //   source_reg.move_out_to(partition_issue, this->get_issue_reg_id(),
    //                          m_dispatch_reg);
    //   occupied.set(m_dispatch_reg->latency);
    // }
}

pub static READ_PACKET_SIZE: u8 = 8;

// bytes: 6 address, 2 miscelaneous.
pub static WRITE_PACKET_SIZE: u8 = 8;

pub static WRITE_MASK_SIZE: u8 = 8;

pub const MAX_ALU_LATENCY: usize = 512;

#[derive()]
pub struct PipelinedSimdUnitImpl {
    result_port: Option<RegisterSet>,
    pipeline_depth: usize,
    pipeline_reg: Vec<WarpInstruction>,
    issue_reg_id: usize,
    active_insts_in_pipeline: usize,
    dispatch_reg: Option<WarpInstruction>,
    occupied: BitArr!(for MAX_ALU_LATENCY),
    config: Arc<GPUConfig>,
}

impl PipelinedSimdUnitImpl {
    pub fn new(
        result_port: Option<RegisterSet>,
        depth: usize,
        config: Arc<GPUConfig>,
        issue_reg_id: usize,
    ) -> Self {
        let pipeline_reg = (0..depth)
            // .map(|_| RegisterSet::new(5, "regiserset".into()))
            .map(|_| WarpInstruction::default())
            .collect();
        Self {
            result_port,
            pipeline_depth: depth,
            pipeline_reg,
            issue_reg_id,
            active_insts_in_pipeline: 0,
            dispatch_reg: None,
            occupied: BitArray::ZERO,
            config,
        }
    }
}

// pub trait PipelinedSimdUnit {
//     // same as simdfunctionunit
// }

// #[derive()]
// pub struct SimdUnitImpl {
//     // pipeline_depth: usize,
//     // pipeline_reg: Vec<WarpInstruction>,
//     // issue_reg_id: usize,
//     // active_insts_in_pipeline: usize,
//     dispatch_reg: Option<WarpInstruction>,
//     config: Arc<GPUConfig>,
//     occupied: BitArr!(for MAX_ALU_LATENCY),
// }
//
// impl SimdFunctionUnit for SimdUnitImpl {
//     fn issue(&mut self, src_reg: RegisterSet) {
//         let partition_issue = self.config.sub_core_model && self.is_issue_partitioned();
//         // src_reg.move_out_to(partition_issue, self.issue_reg_id(), self.dispatch_reg);
//         // self.occupied.set(self.dispatch_reg.latency, true);
//         todo!("pipelined simd unit: issue");
//     }
// }

impl SimdFunctionUnit for PipelinedSimdUnitImpl {
    fn active_lanes_in_pipeline(&self) -> usize {
        let mut active_lanes: sched::ThreadActiveMask = BitArray::ZERO;
        // if self.config.
        for stage in &self.pipeline_reg {
            active_lanes |= stage.active_mask;
        }
        // for (unsigned stage = 0; (stage + 1) < m_pipeline_depth; stage++) {
        //   if (!m_pipeline_reg[stage]->empty())
        //     active_lanes |= m_pipeline_reg[stage]->get_active_mask();
        // }
        active_lanes.count_ones()
    }

    fn cycle(&mut self) {
        if !self.pipeline_reg[0].empty() {
            if let Some(port) = &mut self.result_port {
                port.move_in(self.pipeline_reg[0].clone());
            }
            debug_assert!(self.active_insts_in_pipeline > 0);
            self.active_insts_in_pipeline -= 1;
        }
        if self.active_insts_in_pipeline > 0 {
            for stage in 0..self.pipeline_reg.len() - 1 {
                let current = &self.pipeline_reg[stage];
                let next = &self.pipeline_reg[stage + 1];
                // move_warp(next, current);
            }
        }
        if let Some(dispatch) = &self.dispatch_reg {
            // if !dispatch.empty() && !dispatch.dispatch_delay() {
            //     // let start_stage = dispatch.latency - dispatch.initiation_interval;
            //     // move_warp(m_pipeline_reg[start_stage], m_dispatch_reg);
            //     self.active_insts_in_pipeline += 1;
            // }
        }
        self.occupied.shift_right(1);

        todo!("pipelined simd unit: cycle");
    }

    fn issue(&mut self, src_reg: RegisterSet) {
        let partition_issue = self.config.sub_core_model && self.is_issue_partitioned();
        // let ready_reg = src_reg.get_ready(partition_issue, self.issue_reg_id());
        // // self.core.incexecstat((*ready_reg));
        //
        // // from simd function unit
        // src_reg.move_out_to(partition_issue, self.issue_reg_id(), self.dispatch_reg);
        if let Some(dispatch) = &self.dispatch_reg {
            self.occupied.set(dispatch.latency, true);
        }

        todo!("pipelined simd unit: issue");
    }

    // accessors
    fn clock_multiplier(&self) -> usize {
        1
    }
    fn can_issue(&self, instr: &WarpInstruction) -> bool {
        todo!("pipelined simd unit: can issue");
    }
    // fn active_lanes_in_pipeline(&self) -> usize;
    // {
    //     return m_dispatch_reg->empty() && !occupied.test(inst.latency);
    //   }
    fn is_issue_partitioned(&self) -> bool {
        todo!("pipelined simd unit: is issue partitioned");
    }
    fn issue_reg_id(&self) -> usize {
        todo!("pipelined simd unit: issue reg id");
    }
    fn stallable(&self) -> bool {
        todo!("pipelined simd unit: stallable");
    }
}

#[derive()]
pub struct LoadStoreUnit<I> {
    core_id: usize,
    cluster_id: usize,
    // pipeline_depth: usize,
    // pipeline_reg: Vec<RegisterSet>,
    // pipeline_reg: Vec<WarpInstruction>,
    next_writeback: Option<WarpInstruction>,
    response_fifo: VecDeque<MemFetch>,
    // texture_l1: l1::TextureL1,
    // const_l1: l1::ConstL1,
    // todo: how to use generic interface here
    // data_l1: Option<l1::Data<I>>,
    data_l1: Option<Box<dyn cache::Cache>>,
    config: Arc<GPUConfig>,
    next_global: Option<MemFetch>,
    // dispatch_reg: Option<WarpInstruction>,
    /// Pending writes warp -> register -> count
    pending_writes: HashMap<usize, HashMap<u32, usize>>,
    // interconn: ic::Interconnect,
    // interconn: Arc<dyn ic::MemFetchInterface>,
    interconn: Arc<I>,
    pipelined_simd_unit: PipelinedSimdUnitImpl,
    operand_collector: opcoll::OperandCollectorRegisterFileUnit,
}

impl<I> LoadStoreUnit<I>
where
    I: ic::MemFetchInterface,
{
    pub fn new(
        core_id: usize,
        cluster_id: usize,
        interconn: Arc<I>,
        // interconn: Arc<dyn ic::MemFetchInterface>,
        config: Arc<GPUConfig>,
        stats: Arc<Mutex<Stats>>,
    ) -> Self {
        // pipelined_simd_unit(NULL, config, 3, core, 0),
        // pipelined_simd_unit(NULL, config, config->smem_latency, core, 0),
        let pipeline_depth = config.shared_memory_latency;
        let pipelined_simd_unit =
            PipelinedSimdUnitImpl::new(None, pipeline_depth, config.clone(), 0);
        //
        // see pipelined_simd_unit::pipelined_simd_unit
        debug_assert!(config.shared_memory_latency > 1);
        //
        // let texture_l1 = l1::TextureL1::new(core_id, interconn.clone());
        // let const_l1 = l1::ConstL1::default();

        // m_L1T = new tex_cache(L1T_name, m_config->m_L1T_config, m_sid,
        //                 get_shader_texture_cache_id(), icnt, IN_L1T_MISS_QUEUE,
        //                 IN_SHADER_L1T_ROB);
        // m_L1C = new read_only_cache(L1C_name, m_config->m_L1C_config, m_sid,
        //                       get_shader_constant_cache_id(), icnt,
        //                       IN_L1C_MISS_QUEUE);
        //m_L1D = new l1_cache(L1D_name, m_config->m_L1D_config, m_sid,
        // get_shader_normal_cache_id(), m_icnt, m_mf_allocator,
        // IN_L1D_MISS_QUEUE, core->get_gpu());

        let mut l1_latency_queue: Vec<Vec<Option<mem_fetch::MemFetch>>> = Vec::new();
        let data_l1 = if let Some(l1_config) = &config.data_cache_l1 {
            // initialize latency queue
            debug_assert!(config.l1_latency > 0);
            l1_latency_queue = (0..config.l1_banks)
                .map(|bank| vec![None; config.l1_latency])
                .collect();

            // initialize l1 data cache
            Some(l1::Data::new(
                core_id,
                cluster_id,
                interconn.clone(),
                stats.clone(),
                config.clone(),
                l1_config.clone(),
            ))
        } else {
            None
        };

        let num_banks = 0;
        let operand_collector = OperandCollectorRegisterFileUnit::new(num_banks);
        Self {
            core_id,
            cluster_id,
            // const_l1,
            // texture_l1,
            data_l1: None,
            // dispatch_reg: None,
            // pipeline_depth,
            // pipeline_reg,
            next_writeback: None,
            next_global: None,
            pending_writes: HashMap::new(),
            response_fifo: VecDeque::new(),
            interconn,
            pipelined_simd_unit,
            config,
            operand_collector,
        }
    }

    pub fn response_buffer_full(&self) -> bool {
        self.response_fifo.len() >= self.config.num_ldst_response_buffer_size
    }

    pub fn flush(&mut self) {
        if let Some(l1) = &mut self.data_l1 {
            todo!("flush data l1");
            // l1.flush();
        }
    }

    pub fn invalidate(&mut self) {
        if let Some(l1) = &mut self.data_l1 {
            todo!("invalidate data l1");
            // l1.invalidate();
        }
    }

    pub fn fill(&mut self, mut fetch: MemFetch) {
        fetch.status = mem_fetch::Status::IN_SHADER_LDST_RESPONSE_FIFO;
        self.response_fifo.push_back(fetch);
    }

    pub fn writeback(&mut self) {
        // process next instruction that is going to writeback
        // if !self.next_writeback.empty()) {
        if let Some(next_writeback) = self.next_writeback.take() {
            if self.operand_collector.writeback(&next_writeback) {
                let mut instr_completed = false;
                // for r in 0..MAX_OUTPUT_VALUES {
                for out in next_writeback.outputs {
                    // 0..MAX_OUTPUT_VALUES {
                    // for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
                    // if next_writeback.out[r] > 0 {
                    if out > 0 {
                        if next_writeback.memory_space != MemorySpace::Shared {
                            let pending_write = self
                                .pending_writes
                                .get_mut(&next_writeback.warp_id)
                                .unwrap();
                            debug_assert!(pending_write[&out] > 0);
                            let still_pending = pending_write[&out] - 1;
                            if still_pending == 0 {
                                pending_write.remove(&out);
                                // m_scoreboard->releaseRegister(m_next_wb.warp_id(),
                                //                                 m_next_wb.out[r]);
                                instr_completed = true;
                            }
                        } else {
                            // shared
                            // m_scoreboard->releaseRegister(m_next_wb.warp_id(),
                            //                               m_next_wb.out[r]);
                            instr_completed = true;
                        }
                    }
                }
                if instr_completed {
                    // self.core.warp_inst_complete(next_writeback);
                }
                // m_next_wb.clear();
                // m_last_inst_gpu_sim_cycle = m_core->get_gpu()->gpu_sim_cycle;
                // m_last_inst_gpu_tot_sim_cycle = m_core->get_gpu()->gpu_tot_sim_cycle;
            }
        }

        //       unsigned serviced_client = -1;
        // for (unsigned c = 0; m_next_wb.empty() && (c < m_num_writeback_clients);
        //      c++) {
        //   unsigned next_client = (c + m_writeback_arb) % m_num_writeback_clients;
        //   switch (next_client) {
        //     case 0:  // shared memory
        //       if (!m_pipeline_reg[0]->empty()) {
        //         m_next_wb = *m_pipeline_reg[0];
        //         if (m_next_wb.isatomic()) {
        //           m_next_wb.do_atomic();
        //           m_core->decrement_atomic_count(m_next_wb.warp_id(),
        //                                          m_next_wb.active_count());
        //         }
        //         m_core->dec_inst_in_pipeline(m_pipeline_reg[0]->warp_id());
        //         m_pipeline_reg[0]->clear();
        //         serviced_client = next_client;
        //       }
        //       break;
        //     case 1:  // texture response
        //       if (m_L1T->access_ready()) {
        //         mem_fetch *mf = m_L1T->next_access();
        //         m_next_wb = mf->get_inst();
        //         delete mf;
        //         serviced_client = next_client;
        //       }
        //       break;
        //     case 2:  // const cache response
        //       if (m_L1C->access_ready()) {
        //         mem_fetch *mf = m_L1C->next_access();
        //         m_next_wb = mf->get_inst();
        //         delete mf;
        //         serviced_client = next_client;
        //       }
        //       break;
        //     case 3:  // global/local
        //       if (m_next_global) {
        //         m_next_wb = m_next_global->get_inst();
        //         if (m_next_global->isatomic()) {
        //           m_core->decrement_atomic_count(
        //               m_next_global->get_wid(),
        //               m_next_global->get_access_warp_mask().count());
        //         }
        //         delete m_next_global;
        //         m_next_global = NULL;
        //         serviced_client = next_client;
        //       }
        //       break;
        //     case 4:
        //       if (m_L1D && m_L1D->access_ready()) {
        //         mem_fetch *mf = m_L1D->next_access();
        //         m_next_wb = mf->get_inst();
        //         delete mf;
        //         serviced_client = next_client;
        //       }
        //       break;
        //     default:
        //       abort();
        //   }
        // }
        // // update arbitration priority only if:
        // // 1. the writeback buffer was available
        // // 2. a client was serviced
        // if (serviced_client != (unsigned)-1) {
        //   m_writeback_arb = (serviced_client + 1) % m_num_writeback_clients;
        // }

        todo!("ldst unit writeback");
    }

    fn shared_cycle(&mut self) {}

    fn constant_cycle(&mut self) {}

    fn texture_cycle(&mut self) {}

    fn memory_cycle(&mut self) -> bool {
        let simd_unit = &mut self.pipelined_simd_unit;

        println!(
            "core {}-{}: {}",
            self.core_id,
            self.cluster_id,
            style("load store unit: memory cycle").magenta()
        );

        let Some(instr) = &mut simd_unit.dispatch_reg else {
            return true;
        };
        println!("memory cycle for instruction: {}", &instr);

        if instr.memory_space != MemorySpace::Global && instr.memory_space != MemorySpace::Local {
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
            let size = access.req_size_bytes + control_size as u32;

            println!("Interconnect addr: {}, size={}", access.addr, size);
            if self
                .interconn
                .full(size, instr.is_store() || instr.is_atomic())
            {
                // stall_cond = ICNT_RC_FAIL;
            } else {
                // let fetch = self.new_mem_fetch(instr.clone(), access.clone());
                let fetch = {
                    let size = if access.is_write {
                        WRITE_PACKET_SIZE
                    } else {
                        READ_PACKET_SIZE
                    } as u32;

                    let warp_id = instr.warp_id;
                    mem_fetch::MemFetch::new(
                        Some(instr.clone()),
                        access.clone(),
                        &self.config,
                        size,
                        warp_id,
                        self.core_id,
                        self.cluster_id,
                    )
                };

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
}

impl<I> SimdFunctionUnit for LoadStoreUnit<I>
where
    I: ic::MemFetchInterface,
{
    fn active_lanes_in_pipeline(&self) -> usize {
        let active = self.pipelined_simd_unit.active_lanes_in_pipeline();
        debug_assert!(active <= self.config.warp_size);
        active
        // m_core->incfumemactivelanes_stat(active_count);
        // todo!("load store unit: active lanes in pipeline");
    }

    fn issue(&mut self, source_reg: RegisterSet) {
        todo!("load store unit: issue");
    }

    fn clock_multiplier(&self) -> usize {
        1
    }

    fn can_issue(&self, instr: &WarpInstruction) -> bool {
        todo!("load store unit: can issue");
    }

    fn is_issue_partitioned(&self) -> bool {
        // load store unit issue is not partitioned
        false
    }

    fn issue_reg_id(&self) -> usize {
        todo!("load store unit: issue reg id");
    }

    fn stallable(&self) -> bool {
        todo!("load store unit: stallable");
    }

    fn cycle(&mut self) {
        use super::instruction::CacheOperator;

        println!(
            "core {}-{}: {}",
            self.core_id,
            self.cluster_id,
            style("load store unit: cycle").magenta()
        );

        // self.writeback();

        let simd_unit = &mut self.pipelined_simd_unit;
        debug_assert!(simd_unit.pipeline_depth > 0);
        for stage in 0..(simd_unit.pipeline_depth - 1) {
            let current = &simd_unit.pipeline_reg[stage];
            let next = &simd_unit.pipeline_reg[stage + 1];
            if current.empty() && !next.empty() {
                todo!("move warp");
                // move_warp(&mut simd_unit.pipeline_reg, stage + 1, stage);
            }
        }

        dbg!(&simd_unit.pipeline_reg);
        dbg!(&self.response_fifo);
        if let Some(mut fetch) = self.response_fifo.front().cloned() {
            match fetch.access_kind() {
                mem_fetch::AccessKind::TEXTURE_ACC_R => {
                    // if self.texture_l1.has_free_fill_port() {
                    //     self.texture_l1.fill(&fetch);
                    //     // self.response_fifo.fill(mem_fetch);
                    //     self.response_fifo.pop_front();
                    // }
                }
                mem_fetch::AccessKind::CONST_ACC_R => {
                    // if self.const_l1.has_free_fill_port() {
                    //     // fetch.set_status(IN_SHADER_FETCHED)
                    //     self.const_l1.fill(&fetch);
                    //     // self.response_fifo.fill(mem_fetch);
                    //     self.response_fifo.pop_front();
                    // }
                }
                _ => {
                    if fetch.kind == mem_fetch::Kind::WRITE_ACK
                        || (self.config.perfect_mem && fetch.is_write())
                    {
                        // m_core->store_ack(mf);
                        self.response_fifo.pop_front();
                    } else {
                        // L1 cache is write evict:
                        // allocate line on load miss only
                        debug_assert!(fetch.is_write());
                        let mut bypass_l1 = false;

                        // let cache_op = fetch.instr.map(|i| i.cache_operator);
                        if self.data_l1.is_none() {
                            // matches!(cache_op, Some(CacheOperator::GLOBAL)) {
                            // {
                            bypass_l1 = true;
                        } else if fetch.access_kind() == &mem_fetch::AccessKind::GLOBAL_ACC_R
                            || fetch.access_kind() == &mem_fetch::AccessKind::GLOBAL_ACC_W
                        {
                            // global memory access
                            if self.config.global_mem_skip_l1_data_cache {
                                bypass_l1 = true;
                            }
                        }

                        match &self.data_l1 {
                            Some(l1d) if !bypass_l1 => {
                                // if l1d.has_free_fill_port() {
                                //     l1d.fill(&fetch);
                                //     self.response_fifo.pop_front();
                                // }
                            }
                            _ => {
                                if self.next_global.is_none() {
                                    fetch.set_status(mem_fetch::Status::IN_SHADER_FETCHED, 0);
                                    self.response_fifo.pop_front();
                                    self.next_global.insert(fetch.clone());
                                }
                            }
                        }
                    }
                }
            }
        }

        // self.texture_l1.cycle();
        // self.const_l1.cycle();
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
        drop(simd_unit);
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

        // let pipe_reg = &self.dispatch_reg;
        // if !pipe_reg.empty {
        let simd_unit = &mut self.pipelined_simd_unit;
        if let Some(pipe_reg) = simd_unit.dispatch_reg.take() {
            let warp_id = pipe_reg.warp_id;
            if pipe_reg.is_load() {
                if pipe_reg.memory_space == MemorySpace::Shared {
                    let slot = &mut simd_unit.pipeline_reg[self.config.shared_memory_latency - 1];
                    if slot.empty() {
                        // new shared memory request
                        //
                        // move_warp(&mut self.pipeline_reg, stage + 1, stage);
                        // move_warp(&slot, self.dispatch_reg);
                        *slot = pipe_reg;
                        // self.dispatch_reg.clear();

                        // pipe_reg = None;
                        // self.dispatch_reg.take();
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
                    // self.dispatch_reg.clear();
                    // self.dispatch_reg = None;
                }
            } else {
                // stores exit pipeline here
                // core.dec_inst_in_pipeline(warp_id);
                // core.warp_inst_complete(self.dispatch_reg);
                // self.dispatch_reg.clear();
                // self.dispatch_reg = None;
            }
        }
    }

    // fn new_mem_fetch(
    //     &self,
    //     instr: WarpInstruction,
    //     access: mem_fetch::MemAccess,
    // ) -> mem_fetch::MemFetch {
    //     let size = if access.is_write {
    //         WRITE_PACKET_SIZE
    //     } else {
    //         READ_PACKET_SIZE
    //     };
    //
    //     let warp_id = instr.warp_id;
    //     mem_fetch::MemFetch::new(
    //         instr,
    //         access,
    //         GPUConfig::default(),
    //         size,
    //         warp_id,
    //         self.core_id,
    //         self.cluster_id,
    //     )
    //     // Self {
    //     //     instr,
    //     //     access,
    //     // }
    //     // access, &inst_copy,
    //     // inst.warp_id(), m_core_id, m_cluster_id, m_memory_config, cycle);
    //     // return mf;
    // }
}

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
