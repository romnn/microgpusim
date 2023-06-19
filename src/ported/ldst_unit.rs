use super::{
    cache, interconn as ic, l1, mem_fetch, operand_collector as opcoll,
    register_set::{self, RegisterSet},
    scheduler as sched,
    scoreboard::Scoreboard,
    simd_function_unit as fu,
    stats::Stats,
};
use crate::{config::GPUConfig, ported::operand_collector::OperandCollectorRegisterFileUnit};
use bitvec::{array::BitArray, BitArr};
use console::style;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex, RwLock};

use super::{
    address,
    instruction::{CacheOperator, WarpInstruction},
    mem_fetch::MemFetch,
};
use nvbit_model::MemorySpace;
use std::collections::{HashMap, VecDeque};
use trace_model::MemAccessTraceEntry;

pub static READ_PACKET_SIZE: u8 = 8;

// bytes: 6 address, 2 miscelaneous.
pub static WRITE_PACKET_SIZE: u8 = 8;

pub static WRITE_MASK_SIZE: u8 = 8;

#[derive()]
pub struct LoadStoreUnit<I> {
    // pub struct LoadStoreUnit {
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
    stats: Arc<Mutex<Stats>>,
    scoreboard: Arc<RwLock<Scoreboard>>,
    next_global: Option<MemFetch>,
    // dispatch_reg: Option<WarpInstruction>,
    /// Pending writes warp -> register -> count
    pending_writes: HashMap<usize, HashMap<u32, usize>>,
    // interconn: ic::Interconnect,
    // fetch_interconn: Arc<dyn ic::MemFetchInterface>,
    fetch_interconn: Arc<I>,
    pipelined_simd_unit: fu::PipelinedSimdUnitImpl,
    operand_collector: Rc<RefCell<opcoll::OperandCollectorRegisterFileUnit>>,
    // phantom: std::marker::PhantomData<I>,
}

impl<I> std::fmt::Debug for LoadStoreUnit<I> {
    // impl std::fmt::Debug for LoadStoreUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadStoreUnit")
            .field("core_id", &self.core_id)
            .field("cluster_id", &self.cluster_id)
            .field("response_fifo_size", &self.response_fifo.len())
            .finish()
    }
}

impl<I> LoadStoreUnit<I>
// impl LoadStoreUnit
where
    I: ic::MemFetchInterface,
    // I: ic::Interconnect<super::core::Packet>,
{
    pub fn new(
        core_id: usize,
        cluster_id: usize,
        fetch_interconn: Arc<I>,
        // interconn: Arc<I>,
        // fetch_interconn: Arc<dyn ic::MemFetchInterface>,
        operand_collector: Rc<RefCell<OperandCollectorRegisterFileUnit>>,
        scoreboard: Arc<RwLock<Scoreboard>>,
        config: Arc<GPUConfig>,
        stats: Arc<Mutex<Stats>>,
    ) -> Self {
        // pipelined_simd_unit(NULL, config, 3, core, 0),
        // pipelined_simd_unit(NULL, config, config->smem_latency, core, 0),
        let pipeline_depth = config.shared_memory_latency;
        let pipelined_simd_unit =
            fu::PipelinedSimdUnitImpl::new(None, pipeline_depth, config.clone(), 0);
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

        // let fetch_interconn = Arc::new(ic::MockCoreMemoryInterface {});

        let mut l1_latency_queue: Vec<Vec<Option<mem_fetch::MemFetch>>> = Vec::new();
        let data_l1 = if let Some(l1_config) = &config.data_cache_l1 {
            // initialize latency queue
            debug_assert!(config.l1_latency > 0);
            l1_latency_queue = (0..config.l1_banks)
                .map(|bank| vec![None; config.l1_latency])
                .collect();

            // initialize l1 data cache
            Some(l1::Data::new(
                format!("ldst-unit-{}-{}-L1-DATA-CACHE", cluster_id, core_id),
                core_id,
                cluster_id,
                fetch_interconn.clone(),
                // interconn.clone(),
                stats.clone(),
                config.clone(),
                l1_config.clone(),
            ))
        } else {
            None
        };

        let num_banks = 0;
        // let operand_collector = OperandCollectorRegisterFileUnit::new(num_banks);
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
            // interconn,
            fetch_interconn,
            pipelined_simd_unit,
            config,
            stats,
            scoreboard,
            operand_collector,
            // phantom: std::marker::PhantomData,
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
            l1.invalidate();
        }
    }

    pub fn fill(&mut self, mut fetch: MemFetch) {
        fetch.status = mem_fetch::Status::IN_SHADER_LDST_RESPONSE_FIFO;
        self.response_fifo.push_back(fetch);
    }

    pub fn writeback(&mut self) {
        // process next instruction that is going to writeback
        // if !self.next_writeback.empty()) {
        if let Some(ref next_writeback) = self.next_writeback {
            if self
                .operand_collector
                .try_borrow_mut()
                .unwrap()
                .writeback(&next_writeback)
            {
                let mut next_writeback = self.next_writeback.take().unwrap();
                let mut instr_completed = false;
                // for r in 0..MAX_OUTPUT_VALUES {
                for out in next_writeback.outputs() {
                    // 0..MAX_OUTPUT_VALUES {
                    // for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
                    // if next_writeback.out[r] > 0 {
                    // if *out > 0 {
                    debug_assert!(*out > 0);
                    if next_writeback.memory_space != MemorySpace::Shared {
                        let pending_write = self
                            .pending_writes
                            .get_mut(&next_writeback.warp_id)
                            .unwrap();
                        debug_assert!(pending_write[out] > 0);
                        let still_pending = pending_write[out] - 1;
                        if still_pending == 0 {
                            pending_write.remove(out);
                            self.scoreboard
                                .write()
                                .unwrap()
                                .release_register(next_writeback.warp_id, *out);
                            instr_completed = true;
                        }
                    } else {
                        // shared
                        self.scoreboard
                            .write()
                            .unwrap()
                            .release_register(next_writeback.warp_id, *out);
                        instr_completed = true;
                    }
                    // }
                }
                if instr_completed {
                    warp_inst_complete(&mut next_writeback, &mut self.stats.lock().unwrap());
                }
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

        // dbg!(&bypass_l1);
        if bypass_l1 {
            // bypass L1 cache
            let control_size = if instr.is_store() {
                WRITE_PACKET_SIZE
            } else {
                READ_PACKET_SIZE
            };
            let size = access.req_size_bytes + control_size as u32;

            // println!("Interconnect addr: {}, size={}", access.addr, size);
            // todo!("load store unit interconn full");
            if self
                .fetch_interconn
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

                // self.interconn.push(fetch);
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

impl<I> fu::SimdFunctionUnit for LoadStoreUnit<I>
// impl fu::SimdFunctionUnit for LoadStoreUnit
where
    I: ic::MemFetchInterface,
    // I: ic::Interconnect<super::core::Packet>,
{
    fn active_lanes_in_pipeline(&self) -> usize {
        let active = self.pipelined_simd_unit.active_lanes_in_pipeline();
        debug_assert!(active <= self.config.warp_size);
        active
        // m_core->incfumemactivelanes_stat(active_count);
        // todo!("load store unit: active lanes in pipeline");
    }

    fn issue(&mut self, source_reg: &mut RegisterSet) {
        // warp_inst_t *inst = *(reg_set.get_ready());
        //
        // // record how many pending register writes/memory accesses there are for this
        // // instruction
        // assert(inst->empty() == false);
        // if (inst->is_load() and inst->space.get_type() != shared_space) {
        //   unsigned warp_id = inst->warp_id();
        //   unsigned n_accesses = inst->accessq_count();
        //   for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
        //     unsigned reg_id = inst->out[r];
        //     if (reg_id > 0) {
        //       m_pending_writes[warp_id][reg_id] += n_accesses;
        //     }
        //   }
        // }
        //
        // inst->op_pipe = MEM__OP;
        // // stat collection
        // m_core->mem_instruction_stats(*inst);
        // m_core->incmem_stat(m_core->get_config()->warp_size, 1);

        self.pipelined_simd_unit.issue(source_reg);
        // todo!("load store unit: issue");
    }

    fn clock_multiplier(&self) -> usize {
        1
    }

    fn can_issue(&self, instr: &WarpInstruction) -> bool {
        use super::opcodes::ArchOp;
        match instr.opcode.category {
            ArchOp::LOAD_OP
            | ArchOp::TENSOR_CORE_LOAD_OP
            | ArchOp::STORE_OP
            | ArchOp::TENSOR_CORE_STORE_OP
            | ArchOp::MEMORY_BARRIER_OP => self.pipelined_simd_unit.dispatch_reg.is_none(),
            _ => false,
        }
        // todo!("load store unit: can issue");
    }

    fn is_issue_partitioned(&self) -> bool {
        // load store unit issue is not partitioned
        false
    }

    fn issue_reg_id(&self) -> usize {
        todo!("load store unit: issue reg id");
    }

    fn stallable(&self) -> bool {
        true
        // todo!("load store unit: stallable");
    }

    fn cycle(&mut self) {
        use super::instruction::CacheOperator;

        println!(
            "core {}-{}: {} (response fifo size={})",
            self.core_id,
            self.cluster_id,
            style("load store unit::cycle()").magenta(),
            self.response_fifo.len(),
        );

        // self.writeback();

        let simd_unit = &mut self.pipelined_simd_unit;
        debug_assert!(simd_unit.pipeline_depth > 0);
        for stage in 0..(simd_unit.pipeline_depth - 1) {
            // let mut current = simd_unit.pipeline_reg[stage];
            let next = &simd_unit.pipeline_reg[stage + 1];
            // dbg!((&current, &next));

            // if current.empty() && !next.empty() {
            if let Some(next) = next {
                if let Some(current) = simd_unit.pipeline_reg[stage].take() {
                    todo!("move warp");
                    // register_set::move_warp(&mut simd_unit.pipeline_reg, stage + 1, stage);
                    // register_set::move_warp(&mut simd_unit.pipeline_reg, stage + 1, stage);
                }
            }
        }

        if let Some(mut fetch) = self.response_fifo.front().cloned() {
            match fetch.access_kind() {
                mem_fetch::AccessKind::TEXTURE_ACC_R => {
                    todo!("ldst unit: tex access");
                    // if self.texture_l1.has_free_fill_port() {
                    //     self.texture_l1.fill(&fetch);
                    //     // self.response_fifo.fill(mem_fetch);
                    //     self.response_fifo.pop_front();
                    // }
                }
                mem_fetch::AccessKind::CONST_ACC_R => {
                    todo!("ldst unit: const access");
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
                                todo!("ldst unit: data l1 fill");
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
        if let Some(ref pipe_reg) = simd_unit.dispatch_reg {
            // panic!("ldst unit got instr from dispatch reg");
            let warp_id = pipe_reg.warp_id;
            if pipe_reg.is_load() {
                if pipe_reg.memory_space == MemorySpace::Shared {
                    let slot = &mut simd_unit.pipeline_reg[self.config.shared_memory_latency - 1];
                    if slot.is_none() {
                        // new shared memory request
                        let pipe_reg = simd_unit.dispatch_reg.take();
                        register_set::move_warp(pipe_reg, slot);
                    }
                } else {
                    let mut pending_requests = false;
                    for reg_id in pipe_reg.outputs() {
                        let pending = self.pending_writes.entry(warp_id).or_default();
                        if *reg_id > 0 {
                            match pending.get(reg_id) {
                                Some(&p) if p > 0 => {
                                    pending_requests = true;
                                    break;
                                }
                                _ => {
                                    // this instruction is done already
                                    pending.remove(reg_id);
                                }
                            }
                        }
                    }

                    let mut dispatch_reg = simd_unit.dispatch_reg.take().unwrap();

                    if !pending_requests {
                        warp_inst_complete(&mut dispatch_reg, &mut self.stats.lock().unwrap());
                        self.scoreboard
                            .write()
                            .unwrap()
                            .release_registers(&dispatch_reg);
                    }
                    // core.dec_inst_in_pipeline(warp_id);
                    simd_unit.dispatch_reg = None;
                }
            } else {
                // stores exit pipeline here
                // core.dec_inst_in_pipeline(warp_id);
                // todo!("warp instruction complete");
                let mut dispatch_reg = simd_unit.dispatch_reg.take().unwrap();
                warp_inst_complete(&mut dispatch_reg, &mut self.stats.lock().unwrap());
            }
        }
    }
}

pub fn warp_inst_complete(instr: &mut WarpInstruction, stats: &mut Stats) {
    // if (inst.op_pipe == SP__OP)
    //   m_stats->m_num_sp_committed[m_sid]++;
    // else if (inst.op_pipe == SFU__OP)
    //   m_stats->m_num_sfu_committed[m_sid]++;
    // else if (inst.op_pipe == MEM__OP)
    //   m_stats->m_num_mem_committed[m_sid]++;
    //
    // if (m_config->gpgpu_clock_gated_lanes == false)
    //   m_stats->m_num_sim_insn[m_sid] += m_config->warp_size;
    // else
    //   m_stats->m_num_sim_insn[m_sid] += inst.active_count();

    // m_stats->m_num_sim_winsn[m_sid]++;
    // m_gpu->gpu_sim_insn += inst.active_count();
    // instr.completed(m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle);
}
