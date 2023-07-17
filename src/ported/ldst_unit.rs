use super::{
    cache, interconn as ic, l1, mem_fetch, mshr, operand_collector as opcoll,
    register_set::{self, RegisterSet},
    scheduler as sched,
    scoreboard::Scoreboard,
    simd_function_unit as fu,
};
use crate::{config, ported::operand_collector::OperandCollectorRegisterFileUnit};
use bitvec::{array::BitArray, BitArr};
use console::style;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex, RwLock};
use strum::EnumCount;

use super::{
    address,
    instruction::{CacheOperator, MemorySpace, WarpInstruction},
    mem_fetch::MemFetch,
};
use std::collections::{HashMap, VecDeque};
use trace_model::MemAccessTraceEntry;

fn new_mem_fetch(
    access: mem_fetch::MemAccess,
    instr: WarpInstruction,
    config: &config::GPUConfig,
    core_id: usize,
    cluster_id: usize,
) -> mem_fetch::MemFetch {
    let warp_id = instr.warp_id;
    let control_size = access.control_size();
    mem_fetch::MemFetch::new(
        Some(instr),
        access,
        &config,
        control_size,
        warp_id,
        core_id,
        cluster_id,
    )
}

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
    warps: Vec<sched::CoreWarp>,
    // texture_l1: l1::TextureL1,
    // const_l1: l1::ConstL1,
    // todo: how to use generic interface here
    // data_l1: Option<l1::Data<I>>,
    pub data_l1: Option<Box<dyn cache::Cache>>,
    config: Arc<config::GPUConfig>,
    pub stats: Arc<Mutex<stats::Stats>>,
    scoreboard: Arc<RwLock<Scoreboard>>,
    next_global: Option<MemFetch>,
    // dispatch_reg: Option<WarpInstruction>,
    /// Pending writes warp -> register -> count
    pending_writes: HashMap<usize, HashMap<u32, usize>>,
    l1_latency_queue: Vec<Vec<Option<mem_fetch::MemFetch>>>,
    // interconn: ic::Interconnect,
    // fetch_interconn: Arc<dyn ic::MemFetchInterface>,
    fetch_interconn: Arc<I>,
    pipelined_simd_unit: fu::PipelinedSimdUnitImpl,
    operand_collector: Rc<RefCell<opcoll::OperandCollectorRegisterFileUnit>>,

    /// round-robin arbiter for writeback contention between L1T, L1C, shared
    writeback_arb: usize,
    num_writeback_clients: usize,
    // phantom: std::marker::PhantomData<I>,
}

impl<I> std::fmt::Display for LoadStoreUnit<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.pipelined_simd_unit.name)
    }
}

impl<I> std::fmt::Debug for LoadStoreUnit<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(&self.pipelined_simd_unit.name)
            .field("core_id", &self.core_id)
            .field("cluster_id", &self.cluster_id)
            .field("response_fifo_size", &self.response_fifo.len())
            .finish()
    }
}

#[derive(strum::EnumCount, strum::FromRepr, Hash, PartialEq, Eq, Clone, Copy, Debug)]
#[repr(usize)]
enum WritebackClient {
    SharedMemory = 0,
    L1T = 1,
    L1C = 2,
    // (uncached)
    GlobalLocal = 3,
    L1D = 4,
}

#[derive(strum::EnumCount, strum::FromRepr, Hash, PartialEq, Eq, Clone, Copy, Debug)]
#[repr(usize)]
enum MemStageAccessKind {
    C_MEM,
    T_MEM,
    S_MEM,
    G_MEM_LD,
    L_MEM_LD,
    G_MEM_ST,
    L_MEM_ST,
    // N_MEM_STAGE_ACCESS_TYPE
}

#[derive(strum::EnumCount, strum::FromRepr, Hash, PartialEq, Eq, Clone, Copy, Debug)]
#[repr(usize)]
enum MemStageStallKind {
    NO_RC_FAIL = 0,
    BK_CONF,
    MSHR_RC_FAIL,
    ICNT_RC_FAIL,
    COAL_STALL,
    TLB_STALL,
    DATA_PORT_STALL,
    WB_ICNT_RC_FAIL,
    WB_CACHE_RSRV_FAIL,
}

impl<I> LoadStoreUnit<I>
// impl LoadStoreUnit
where
    I: ic::MemFetchInterface + 'static,
    // I: ic::Interconnect<super::core::Packet>,
{
    pub fn new(
        id: usize,
        core_id: usize,
        cluster_id: usize,
        warps: Vec<sched::CoreWarp>,
        fetch_interconn: Arc<I>,
        // interconn: Arc<I>,
        // fetch_interconn: Arc<dyn ic::MemFetchInterface>,
        operand_collector: Rc<RefCell<OperandCollectorRegisterFileUnit>>,
        scoreboard: Arc<RwLock<Scoreboard>>,
        config: Arc<config::GPUConfig>,
        stats: Arc<Mutex<stats::Stats>>,
        cycle: super::Cycle,
        // issue_reg_id: usize,
    ) -> Self {
        // pipelined_simd_unit(NULL, config, 3, core, 0),
        // pipelined_simd_unit(NULL, config, config->smem_latency, core, 0),
        let pipeline_depth = config.shared_memory_latency;
        let pipelined_simd_unit = fu::PipelinedSimdUnitImpl::new(
            id,
            "LdstUnit".to_string(),
            None,
            pipeline_depth,
            config.clone(),
            cycle,
            0,
        );
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
        let data_l1: Option<Box<dyn cache::Cache + 'static>> =
            if let Some(l1_config) = &config.data_cache_l1 {
                // initialize latency queue
                debug_assert!(l1_config.l1_latency > 0);
                l1_latency_queue = (0..l1_config.l1_banks)
                    .map(|bank| vec![None; l1_config.l1_latency])
                    .collect();

                // initialize l1 data cache
                let cache_stats = Arc::new(Mutex::new(stats::Cache::default()));
                Some(Box::new(l1::Data::new(
                    format!("ldst-unit-{}-{}-L1-DATA-CACHE", cluster_id, core_id),
                    core_id,
                    cluster_id,
                    fetch_interconn.clone(),
                    cache_stats,
                    config.clone(),
                    l1_config.inner.clone(),
                    mem_fetch::AccessKind::L1_WR_ALLOC_R,
                    mem_fetch::AccessKind::L1_WRBK_ACC,
                ))) //  as Option<Box<dyn cache::Cache + 'static>>
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
            // data_l1: None,
            data_l1,
            // dispatch_reg: None,
            // pipeline_depth,
            // pipeline_reg,
            warps,
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
            num_writeback_clients: WritebackClient::COUNT,
            writeback_arb: 0,
            l1_latency_queue,
            // phantom: std::marker::PhantomData,
        }
    }

    fn get_pending_writes(&mut self, warp_id: usize, reg_id: u32) -> &mut usize {
        self.pending_writes
            .entry(warp_id)
            .or_default()
            .entry(reg_id)
            .or_default()
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
        println!(
            "{} (next_writeback={:?} arb={})",
            style("ldst unit writeback").magenta(),
            self.next_writeback.as_ref().map(|wb| wb.to_string()),
            self.writeback_arb
        );

        // this processes the next writeback
        if let Some(ref next_writeback) = self.next_writeback {
            // println!(
            //     "{}",
            //     style(format!("ldst unit writeback {}", next_writeback)).magenta()
            // );

            if self
                .operand_collector
                .try_borrow_mut()
                .unwrap()
                .writeback(&next_writeback)
            {
                let mut next_writeback = self.next_writeback.take().unwrap();

                let mut instr_completed = false;
                for out_reg in next_writeback.outputs() {
                    debug_assert!(*out_reg > 0);
                    // if next_writeback.warp_id == 3 {
                    //     super::debug_break("process writeback for warp 3");
                    // }
                    if next_writeback.memory_space != Some(MemorySpace::Shared) {
                        let pending = self
                            .pending_writes
                            .entry(next_writeback.warp_id)
                            .or_default();
                        debug_assert!(pending[out_reg] > 0);
                        let still_pending = pending[out_reg] - 1;
                        if still_pending == 0 {
                            pending.remove(out_reg);
                            self.scoreboard
                                .write()
                                .unwrap()
                                .release_register(next_writeback.warp_id, *out_reg);
                            instr_completed = true;
                        }
                    } else {
                        // shared
                        self.scoreboard
                            .write()
                            .unwrap()
                            .release_register(next_writeback.warp_id, *out_reg);
                        instr_completed = true;
                    }
                }
                if instr_completed {
                    super::warp_inst_complete(&mut next_writeback, &self.stats);
                }
                // m_last_inst_gpu_sim_cycle = m_core->get_gpu()->gpu_sim_cycle;
                // m_last_inst_gpu_tot_sim_cycle = m_core->get_gpu()->gpu_tot_sim_cycle;
            }
        }

        // this arbitrates between the writeback clients
        // sets next writeback for writeback in the next cycle
        let mut serviced_client = None;
        for client in 0..self.num_writeback_clients {
            if self.next_writeback.is_some() {
                break;
            }
            let next_client = (client + self.writeback_arb) % self.num_writeback_clients;
            match WritebackClient::from_repr(next_client).unwrap() {
                WritebackClient::SharedMemory => {
                    if let Some(pipe_reg) = self.pipelined_simd_unit.pipeline_reg[0].take() {
                        if pipe_reg.is_atomic() {
                            // pipe_reg.do_atomic();
                            // m_core->decrement_atomic_count(m_next_wb.warp_id(),
                            //                                m_next_wb.active_count());
                        }

                        self.warps[pipe_reg.warp_id]
                            .try_borrow_mut()
                            .unwrap()
                            .num_instr_in_pipeline -= 1;
                        self.next_writeback = Some(pipe_reg);
                        serviced_client = Some(next_client);
                    }
                }
                WritebackClient::L1T => {
                    // texture response
                    // todo!("texture l1 writeback service");
                    // if self.texture_l1.access_ready() {
                    //     //   mem_fetch *mf = m_L1T->next_access();
                    //     //   m_next_wb = mf->get_inst();
                    //     //   delete mf;
                    //     serviced_client = Some(next_client);
                    // }
                }
                WritebackClient::L1C => {
                    // const cache response
                    // todo!("constant l1 writeback service");
                    // if (m_L1C->access_ready()) {
                    //   mem_fetch *mf = m_L1C->next_access();
                    //   m_next_wb = mf->get_inst();
                    //   delete mf;
                    // serviced_client = Some(next_client);
                    // },
                }
                WritebackClient::GlobalLocal => {
                    // global/local
                    if let Some(next_global) = self.next_global.take() {
                        if next_global.warp_id == 3 {
                            super::debug_break("global writeback for warp 3");
                        }
                        println!(
                            "{}",
                            style(format!(
                                "ldst unit writeback: has global {:?} ({})",
                                &next_global.instr.as_ref().map(|i| i.to_string()),
                                &next_global.addr()
                            ))
                            .magenta(),
                        );
                        if next_global.is_atomic() {
                            // m_core->decrement_atomic_count(
                            //     m_next_global->get_wid(),
                            //     m_next_global->get_access_warp_mask().count());
                        }

                        self.next_writeback = next_global.instr;
                        serviced_client = Some(next_client);
                    }
                }
                WritebackClient::L1D => {
                    if let Some(ref mut data_l1) = self.data_l1 {
                        if let Some(fetch) = data_l1.next_access() {
                            self.next_writeback = fetch.instr;
                            serviced_client = Some(next_client);
                        }
                    }
                }
            }
        }

        // update arbitration priority only if:
        // 1. the writeback buffer was available
        // 2. a client was serviced
        if let Some(serviced) = serviced_client {
            self.writeback_arb = (serviced + 1) % self.num_writeback_clients;
            println!(
                "{} {:?} ({})",
                style("ldst unit writeback: serviced client").magenta(),
                WritebackClient::from_repr(serviced),
                serviced,
            );
        }

        // todo!("ldst unit writeback");
    }

    fn shared_cycle(
        &mut self,
        stall_kind: &mut MemStageStallKind,
        kind: &mut MemStageAccessKind,
    ) -> bool {
        let Some(dispatch_instr) = &mut self.pipelined_simd_unit.dispatch_reg else {
            return true;
        };
        println!("shared cycle for instruction: {}", &dispatch_instr);

        if dispatch_instr.memory_space != Some(MemorySpace::Shared) {
            // shared cycle is done
            return true;
        }

        if dispatch_instr.active_thread_count() == 0 {
            // shared cycle is done
            return true;
        }

        if dispatch_instr.has_dispatch_delay() {
            // self.inner.stats.num_shared_mem_bank_access[self.core_id] += 1;
        }

        dispatch_instr.dec_dispatch_delay();
        let has_stall = dispatch_instr.has_dispatch_delay();
        if has_stall {
            *kind = MemStageAccessKind::S_MEM;
            *stall_kind = MemStageStallKind::BK_CONF;
        } else {
            *stall_kind = MemStageStallKind::NO_RC_FAIL;
        }
        !has_stall
    }

    fn constant_cycle(
        &mut self,
        rc_fail: &mut MemStageStallKind,
        kind: &mut MemStageAccessKind,
    ) -> bool {
        false
    }

    fn texture_cycle(
        &mut self,
        rc_fail: &mut MemStageStallKind,
        kind: &mut MemStageAccessKind,
    ) -> bool {
        false
    }

    fn memory_cycle(
        &mut self,
        rc_fail: &mut MemStageStallKind,
        kind: &mut MemStageAccessKind,
    ) -> bool {
        // println!(
        //     "core {}-{}: {}",
        //     self.core_id,
        //     self.cluster_id,
        //     style("load store unit: memory cycle").magenta()
        // );

        // let simd_unit = &mut self.pipelined_simd_unit;
        let Some(dispatch_instr) = &self.pipelined_simd_unit.dispatch_reg else {
            return true;
        };
        println!("memory cycle for instruction: {}", &dispatch_instr);

        if !matches!(
            dispatch_instr.memory_space,
            Some(MemorySpace::Global | MemorySpace::Local)
        ) {
            return true;
        }
        if dispatch_instr.active_thread_count() == 0 {
            return true;
        }
        if dispatch_instr.mem_access_queue.is_empty() {
            return true;
        }

        let mut bypass_l1 = false;

        if self.data_l1.is_none() || dispatch_instr.cache_operator == CacheOperator::GLOBAL {
            bypass_l1 = true;
        } else if dispatch_instr.memory_space == Some(MemorySpace::Global) {
            // global memory access
            // skip L1 cache if the option is enabled
            if self.config.global_mem_skip_l1_data_cache
                && dispatch_instr.cache_operator != CacheOperator::L1
            {
                bypass_l1 = true;
            }
        }

        let Some(access) = dispatch_instr.mem_access_queue.back() else {
            return true;
        };

        println!(
            "memory cycle for instruction {} => access: {}",
            &dispatch_instr, access
        );

        let mut stall_cond = MemStageStallKind::NO_RC_FAIL;
        if bypass_l1 {
            // bypass L1 cache
            debug_assert_eq!(dispatch_instr.is_store(), access.is_write); // "this must not hold?");
            let control_size = if dispatch_instr.is_store() {
                mem_fetch::WRITE_PACKET_SIZE
            } else {
                mem_fetch::READ_PACKET_SIZE
            };
            let size = access.req_size_bytes + control_size as u32;

            // println!("Interconnect addr: {}, size={}", access.addr, size);
            if self.fetch_interconn.full(
                size,
                dispatch_instr.is_store() || dispatch_instr.is_atomic(),
            ) {
                stall_cond = MemStageStallKind::ICNT_RC_FAIL;
            } else {
                let pending = &self.pending_writes[&dispatch_instr.warp_id];
                if dispatch_instr.is_load() {
                    for out_reg in dispatch_instr.outputs() {
                        debug_assert!(pending[out_reg] > 0);
                    }
                } else if dispatch_instr.is_store() {
                    // m_core->inc_store_req(inst.warp_id());
                }

                // shouldnt we remove the dispatch reg here?
                let instr = &mut self.pipelined_simd_unit.dispatch_reg.as_mut().unwrap();
                let access = instr.mem_access_queue.pop_back().unwrap();

                let mut fetch = new_mem_fetch(
                    access,
                    instr.clone(),
                    &self.config,
                    self.core_id,
                    self.cluster_id,
                );

                self.fetch_interconn.push(fetch);
            }
        } else {
            debug_assert_ne!(dispatch_instr.cache_operator, CacheOperator::UNDEFINED);
            stall_cond = self.process_memory_access_queue_l1cache();
        }

        let dispatch_instr = self.pipelined_simd_unit.dispatch_reg.as_ref().unwrap();

        if !dispatch_instr.mem_access_queue.is_empty()
            && stall_cond == MemStageStallKind::NO_RC_FAIL
        {
            stall_cond = MemStageStallKind::COAL_STALL;
        }
        println!("memory instruction stall cond: {:?}", &stall_cond);
        if stall_cond != MemStageStallKind::NO_RC_FAIL {
            *rc_fail = stall_cond;
            if dispatch_instr.memory_space == Some(MemorySpace::Local) {
                *kind = if dispatch_instr.is_store() {
                    MemStageAccessKind::L_MEM_ST
                } else {
                    MemStageAccessKind::L_MEM_LD
                };
            } else {
                *kind = if dispatch_instr.is_store() {
                    MemStageAccessKind::G_MEM_ST
                } else {
                    MemStageAccessKind::G_MEM_LD
                };
            }
        }
        dispatch_instr.mem_access_queue.is_empty()
    }

    fn process_memory_access_queue_l1cache(
        &mut self,
        // instr: &mut WarpInstruction,
    ) -> MemStageStallKind {
        let mut stall_cond = MemStageStallKind::NO_RC_FAIL;
        let Some(instr) = &mut self.pipelined_simd_unit.dispatch_reg else {
            return stall_cond;
        };

        let Some(access) = instr.mem_access_queue.back() else {
            return stall_cond;
        };

        let l1d_config = self.config.data_cache_l1.as_ref().unwrap();

        if l1d_config.l1_latency > 0 {
            // We can handle at max l1_banks reqs per cycle
            for bank in 0..l1d_config.l1_banks {
                let Some(access) = instr.mem_access_queue.back() else {
                    break;
                };

                // let is_store = instr.is_store();
                // let new_instr = instr.clone();
                // drop(instr);
                // mem_fetch *mf =
                //     m_mf_allocator->alloc(inst, inst.accessq_back(),
                //                           m_core->get_gpu()->gpu_sim_cycle +
                //                               m_core->get_gpu()->gpu_tot_sim_cycle);
                let bank_id = l1d_config.set_bank(access.addr) as usize;
                debug_assert!(bank_id < l1d_config.l1_banks);

                let slot = self.l1_latency_queue[bank_id]
                    .get_mut(l1d_config.l1_latency - 1)
                    .unwrap();
                if slot.is_none() {
                    let is_store = instr.is_store();
                    let access = instr.mem_access_queue.pop_back().unwrap();
                    let new_instr = instr.clone();
                    let fetch = new_mem_fetch(
                        access,
                        new_instr,
                        &self.config,
                        self.core_id,
                        self.cluster_id,
                    );
                    *slot = Some(fetch);

                    if is_store {
                        //       unsigned inc_ack =
                        //           (m_config->m_L1D_config.get_mshr_type() == SECTOR_ASSOC)
                        //               ? (mf->get_data_size() / SECTOR_SIZE)
                        //               : 1;
                        //
                        //       for (unsigned i = 0; i < inc_ack; ++i)
                        //         m_core->inc_store_req(inst.warp_id());
                    }
                    //
                } else {
                    stall_cond = MemStageStallKind::BK_CONF;
                    // delete mf;
                    // do not try again, just break from the loop and try the next cycle
                    break;
                }
            }
            if !instr.mem_access_queue.is_empty() && stall_cond != MemStageStallKind::BK_CONF {
                stall_cond = MemStageStallKind::COAL_STALL;
            }

            stall_cond
        } else {
            let new_instr = instr.clone();
            let fetch = new_mem_fetch(
                access.clone(),
                new_instr,
                &self.config,
                self.core_id,
                self.cluster_id,
            );
            let mut events = Vec::new();
            let status = self
                .data_l1
                .as_mut()
                .unwrap()
                .access(fetch.addr(), fetch, &mut events);
            todo!("process cache access");
            // self.process_cache_access(cache, fetch.addr(), instr, fetch, status)
            stall_cond
        }
    }

    fn process_cache_access(
        &mut self,
        cache: (),
        addr: address,
        instr: &mut WarpInstruction,
        events: &mut Vec<cache::Event>,
        fetch: mem_fetch::MemFetch,
        status: cache::RequestStatus,
    ) -> MemStageStallKind {
        let mut stall_cond = MemStageStallKind::NO_RC_FAIL;
        let write_sent = super::was_write_sent(events);
        let read_sent = super::was_read_sent(events);
        // if write_sent {
        // unsigned inc_ack = (m_config->m_L1D_config.get_mshr_type() == SECTOR_ASSOC)
        //                        ? (mf->get_data_size() / SECTOR_SIZE)
        //                        : 1;

        // for (unsigned i = 0; i < inc_ack; ++i)
        // self.warps[instr.warp_id].try_borrow_mut().unwrap().num_outstanding_stores += 1;
        // m_core->inc_store_req(inst.warp_id());
        // }
        if status == cache::RequestStatus::HIT {
            // assert(!read_sent);
            instr.mem_access_queue.pop_back();
            if instr.is_load() {
                for out_reg in instr.outputs() {
                    let mut pending = self
                        .pending_writes
                        .get_mut(&instr.warp_id)
                        .and_then(|p| p.get_mut(out_reg))
                        .unwrap();
                    *pending -= 1;
                }
            }
            // if (!write_sent)
            //   delete mf;
        } else if status == cache::RequestStatus::RESERVATION_FAIL {
            stall_cond = MemStageStallKind::BK_CONF;
            // assert(!read_sent);
            // assert(!write_sent);
            // delete mf;
        } else {
            debug_assert!(matches!(
                status,
                cache::RequestStatus::MISS | cache::RequestStatus::HIT_RESERVED
            ));
            instr.mem_access_queue.pop_back();
        }
        if !instr.mem_access_queue.is_empty() && stall_cond == MemStageStallKind::NO_RC_FAIL {
            stall_cond = MemStageStallKind::COAL_STALL;
        }
        stall_cond
    }

    fn l1_latency_queue_cycle(&mut self) {
        let l1_cache = self.data_l1.as_mut().unwrap();
        let l1_config = self.config.data_cache_l1.as_ref().unwrap();
        for bank in 0..l1_config.l1_banks {
            if let Some(next_fetch) = &self.l1_latency_queue[bank][0] {
                let mut events = Vec::new();
                let access_status =
                    l1_cache.access(next_fetch.addr(), next_fetch.clone(), &mut events);

                let write_sent = super::was_write_sent(&events);
                let read_sent = super::was_read_sent(&events);
                let write_allocate_sent = super::was_writeallocate_sent(&events);

                let dec_ack = if l1_config.inner.mshr_kind == mshr::Kind::SECTOR_ASSOC {
                    next_fetch.data_size / super::mem_sub_partition::SECTOR_SIZE
                } else {
                    1
                };

                if access_status == cache::RequestStatus::HIT {
                    debug_assert!(!read_sent);
                    let mut next_fetch = self.l1_latency_queue[bank][0].take().unwrap();
                    let instr = next_fetch.instr.as_mut().unwrap();

                    if instr.is_load() {
                        let mut completed = false;
                        for out_reg in instr.outputs() {
                            let pending = self.pending_writes.get_mut(&instr.warp_id).unwrap();
                            debug_assert!(pending[out_reg] > 0);
                            let still_pending = pending[out_reg] - 1;
                            if still_pending > 0 {
                                pending.remove(&out_reg);
                                self.scoreboard
                                    .write()
                                    .unwrap()
                                    .release_register(instr.warp_id, *out_reg);
                                completed = true;
                            }
                        }
                        if completed {
                            super::warp_inst_complete(instr, &self.stats);
                        }
                    }

                    // For write hit in WB policy
                    if instr.is_store() && !write_sent {
                        next_fetch.set_reply();

                        for ack in 0..dec_ack {
                            // core.store_ack(next_fetch);
                        }
                    }

                    if !write_sent {
                        // delete mf_next;
                    }
                } else if access_status == cache::RequestStatus::RESERVATION_FAIL {
                    debug_assert!(!read_sent);
                    debug_assert!(!write_sent);
                } else {
                    debug_assert!(matches!(
                        access_status,
                        cache::RequestStatus::MISS | cache::RequestStatus::HIT_RESERVED
                    ));
                    let mut next_fetch = self.l1_latency_queue[bank][0].take().unwrap();
                    let instr = next_fetch.instr.as_ref().unwrap();

                    let should_fetch = matches!(
                        l1_config.inner.write_allocate_policy,
                        config::CacheWriteAllocatePolicy::FETCH_ON_WRITE
                            | config::CacheWriteAllocatePolicy::LAZY_FETCH_ON_READ
                    );
                    if l1_config.inner.write_policy != config::CacheWritePolicy::WRITE_THROUGH
                        && instr.is_store()
                        && should_fetch
                        && !write_allocate_sent
                    {
                        next_fetch.set_reply();
                        for ack in 0..dec_ack {
                            // core.store_ack(next_fetch);
                        }

                        if !write_sent && !read_sent {
                            // delete mf_next
                        };
                    }
                }
            }

            for stage in 0..l1_config.l1_latency - 1 {
                if self.l1_latency_queue[bank][stage].is_none() {
                    self.l1_latency_queue[bank][stage] =
                        self.l1_latency_queue[bank][stage + 1].take();
                }
            }
        }
    }
}

impl<I> fu::SimdFunctionUnit for LoadStoreUnit<I>
// impl fu::SimdFunctionUnit for LoadStoreUnit
where
    I: ic::MemFetchInterface + 'static,
    // I: ic::Interconnect<super::core::Packet>,
{
    fn active_lanes_in_pipeline(&self) -> usize {
        let active = self.pipelined_simd_unit.active_lanes_in_pipeline();
        debug_assert!(active <= self.config.warp_size);
        active
        // m_core->incfumemactivelanes_stat(active_count);
        // todo!("load store unit: active lanes in pipeline");
    }

    fn pipeline(&self) -> &Vec<Option<WarpInstruction>> {
        &self.pipelined_simd_unit.pipeline_reg
    }

    fn issue(&mut self, instr: WarpInstruction) {
        // fn issue(&mut self, source_reg: &mut RegisterSet) {
        // let Some((_, Some(instr))) = source_reg.get_ready() else {
        //     panic!("issue to non ready");
        // };

        // record how many pending register writes/memory accesses there are for this
        // instruction
        if instr.is_load() && instr.memory_space != Some(MemorySpace::Shared) {
            let warp_id = instr.warp_id;
            let num_accessess = instr.mem_access_queue.len();
            for out_reg in instr.outputs() {
                let pending = self.pending_writes.entry(warp_id).or_default();
                *pending.entry(*out_reg).or_default() += num_accessess;
            }
        }

        // inst->op_pipe = MEM__OP;

        // m_core->mem_instruction_stats(*inst);
        if let Some(mem_space) = instr.memory_space {
            let mut stats = self.stats.lock().unwrap();
            let active_count = instr.active_thread_count() as u64;
            stats
                .instructions
                .inc(mem_space, instr.is_store(), active_count);
            // match instr.memory_space {
            //     Some(MemorySpace::Local | MemorySpace::Global) => {
            //         if instr.is_store() {
            //             stats.counters.num_mem_store_instructions += active_count;
            //         } else {
            //             stats.counters.num_mem_load_instructions += active_count;
            //         }
            //     }
            //     Some(MemorySpace::Shared) => {
            //         stats.counters.num_shared_mem_instructions += active_count;
            //         if instr.is_store() {
            //             stats.counters.num_shared_mem_store_instructions += active_count;
            //         } else {
            //             stats.counters.num_shared_mem_load_instructions += active_count;
            //         }
            //     }
            //     Some(MemorySpace::Texture) => {}
            //         stats.counters.num_tex_mem_instructions += active_count;
            //     Some(MemorySpace::Constant) => {
            //         stats.counters.num_const_mem_instructions += active_count;
            //     }
            //     None => {}
            // }
        }

        // switch (inst.space.get_type()) {
        //   case undefined_space:
        //   case reg_space:
        //     break;
        //   case shared_space:
        //     m_stats->gpgpu_n_shmem_insn += active_count;
        //     break;
        //   case sstarr_space:
        //     m_stats->gpgpu_n_sstarr_insn += active_count;
        //     break;
        //   case const_space:
        //     m_stats->gpgpu_n_const_insn += active_count;
        //     break;
        //   case param_space_kernel:
        //   case param_space_local:
        //     m_stats->gpgpu_n_param_insn += active_count;
        //     break;
        //   case tex_space:
        //     m_stats->gpgpu_n_tex_insn += active_count;
        //     break;
        //   case global_space:
        //   case local_space:
        //     if (inst.is_store())
        //       m_stats->gpgpu_n_store_insn += active_count;
        //     else
        //       m_stats->gpgpu_n_load_insn += active_count;
        //     break;
        //   default:
        //     abort();
        // }

        // m_core->incmem_stat(m_core->get_config()->warp_size, 1);

        self.pipelined_simd_unit.issue(instr);
        // self.pipelined_simd_unit.issue(source_reg);
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
    }

    fn cycle(&mut self) {
        use super::instruction::CacheOperator;

        println!(
            "fu[{:03}] {:<10} cycle={:03}: \tpipeline={:?} ({} active) \tresponse fifo={:?}",
            self.pipelined_simd_unit.id,
            self.pipelined_simd_unit.name,
            self.pipelined_simd_unit.cycle.get(),
            self.pipelined_simd_unit
                .pipeline_reg
                .iter()
                .map(|reg| reg.as_ref().map(|r| r.to_string()))
                .collect::<Vec<_>>(),
            self.pipelined_simd_unit.num_active_instr_in_pipeline(),
            self.response_fifo
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>(),
        );

        self.writeback();

        let simd_unit = &mut self.pipelined_simd_unit;
        debug_assert!(simd_unit.pipeline_depth > 0);
        for stage in 0..(simd_unit.pipeline_depth - 1) {
            let current = simd_unit.pipeline_reg[stage].take();
            let next = &mut simd_unit.pipeline_reg[stage + 1];
            register_set::move_warp(
                current,
                next,
                format!(
                    "load store unit: move warp from stage {} to {}",
                    stage,
                    stage + 1
                ),
            );
        }

        if let Some(ref fetch) = self.response_fifo.front() {
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
                        debug_assert!(!fetch.is_write());
                        let mut bypass_l1 = false;

                        // let cache_op = fetch.instr.map(|i| i.cache_operator);
                        if self.data_l1.is_none() {
                            // matches!(cache_op, Some(CacheOperator::GLOBAL)) {
                            // {
                            bypass_l1 = true;
                        } else if matches!(
                            fetch.access_kind(),
                            mem_fetch::AccessKind::GLOBAL_ACC_R
                                | mem_fetch::AccessKind::GLOBAL_ACC_W
                        ) {
                            // global memory access
                            if self.config.global_mem_skip_l1_data_cache {
                                bypass_l1 = true;
                            }
                        }

                        if bypass_l1 {
                            if self.next_global.is_none() {
                                let mut fetch = self.response_fifo.pop_front().unwrap();
                                fetch.set_status(mem_fetch::Status::IN_SHADER_FETCHED, 0);
                                // m_core->get_gpu()->gpu_sim_cycle +
                                //     m_core->get_gpu()->gpu_tot_sim_cycle);
                                self.next_global.insert(fetch);
                            }
                        } else {
                            let l1d = self.data_l1.as_mut().unwrap();
                            if l1d.has_free_fill_port() {
                                let mut fetch = self.response_fifo.pop_front().unwrap();
                                l1d.fill(fetch);
                            }
                        }
                    }
                }
            }
        }

        drop(simd_unit);

        // self.texture_l1.cycle();
        // self.const_l1.cycle();
        if let Some(data_l1) = &mut self.data_l1 {
            data_l1.cycle();
            let cache_config = self.config.data_cache_l1.as_ref().unwrap();
            debug_assert_eq!(cache_config.l1_latency, 1);
            if cache_config.l1_latency > 0 {
                self.l1_latency_queue_cycle();
            }
        }

        // let pipe_reg = &self.dispatch_reg;
        let mut stall_kind = MemStageStallKind::NO_RC_FAIL;
        let mut access_kind = MemStageAccessKind::C_MEM;
        let mut done = true;
        done &= self.shared_cycle(&mut stall_kind, &mut access_kind);
        // done &= self.constant_cycle(&mut stall_kind, &mut access_kind);
        // done &= self.texture_cycle(&mut stall_kind, &mut access_kind);
        done &= self.memory_cycle(&mut stall_kind, &mut access_kind);

        // let mut num_stall_scheduler_mem = 0;
        if !done {
            // log stall types and return
            debug_assert_ne!(stall_kind, MemStageStallKind::NO_RC_FAIL);
            // num_stall_scheduler_mem += 1;
            // m_stats->gpu_stall_shd_mem_breakdown[type][rc_fail]++;
            return;
        }

        // let pipe_reg = &self.dispatch_reg;
        // if !pipe_reg.empty {
        let simd_unit = &mut self.pipelined_simd_unit;
        if let Some(ref pipe_reg) = simd_unit.dispatch_reg {
            // ldst unit got instr from dispatch reg
            let warp_id = pipe_reg.warp_id;
            if pipe_reg.is_load() {
                if pipe_reg.memory_space == Some(MemorySpace::Shared) {
                    let pipe_slot_idx = self.config.shared_memory_latency - 1;
                    let pipe_slot = &mut simd_unit.pipeline_reg[pipe_slot_idx];
                    if pipe_slot.is_none() {
                        // new shared memory request
                        let dispatch_reg = simd_unit.dispatch_reg.take();
                        register_set::move_warp(
                            dispatch_reg,
                            pipe_slot,
                            format!(
                                "load store unit: move warp from dispatch register to pipeline[{}]",
                                pipe_slot_idx,
                            ),
                        );
                    }
                } else {
                    let pending = self.pending_writes.entry(warp_id).or_default();

                    let mut has_pending_requests = false;

                    for reg_id in pipe_reg.outputs() {
                        match pending.get(reg_id) {
                            Some(&p) if p > 0 => {
                                has_pending_requests = true;
                                break;
                            }
                            _ => {
                                // this instruction is done already
                                pending.remove(reg_id);
                            }
                        }
                        // }
                    }

                    let mut dispatch_reg = simd_unit.dispatch_reg.take().unwrap();

                    if !has_pending_requests && warp_id == 3 {
                        panic!("rrr");
                    }
                    if !has_pending_requests {
                        // warp_inst_complete(&mut dispatch_reg, &mut self.stats.lock().unwrap());
                        super::warp_inst_complete(&mut dispatch_reg, &self.stats);

                        self.scoreboard
                            .write()
                            .unwrap()
                            .release_registers(&dispatch_reg);
                    }
                    self.warps[warp_id]
                        .try_borrow_mut()
                        .unwrap()
                        .num_instr_in_pipeline -= 1;
                    simd_unit.dispatch_reg = None;
                }
            } else {
                // stores exit pipeline here
                self.warps[warp_id]
                    .try_borrow_mut()
                    .unwrap()
                    .num_instr_in_pipeline -= 1;
                // todo!("warp instruction complete");
                let mut dispatch_reg = simd_unit.dispatch_reg.take().unwrap();
                // warp_inst_complete(&mut dispatch_reg, &mut self.stats.lock().unwrap());
                super::warp_inst_complete(&mut dispatch_reg, &self.stats);
            }
        }
    }
}
