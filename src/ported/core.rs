use super::instruction::WarpInstruction;
use super::scheduler::SchedulerWarp;
use super::{
    address, barrier, cache, ldst_unit, opcodes, operand_collector as opcoll, register_set,
    scoreboard, simd_function_unit as fu,
    stats::{CacheStats, Stats},
    KernelInfo, LoadStoreUnit, MockSimulator,
};
use super::{interconn as ic, l1, mem_fetch, scheduler as sched};
use crate::config::{self, GPUConfig};
use crate::ported::mem_fetch::BitString;
use bitvec::{array::BitArray, BitArr};
use color_eyre::eyre;
use console::style;
use fu::SimdFunctionUnit;
use itertools::Itertools;
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use std::sync::{Arc, Mutex, RwLock, Weak};
use strum::{EnumCount, IntoEnumIterator};

// Volta max shmem size is 96kB
pub const SHARED_MEM_SIZE_MAX: usize = 96 * (1 << 10);
// Volta max local mem is 16kB
pub const LOCAL_MEM_SIZE_MAX: usize = 1 << 14;
// Volta Titan V has 80 SMs
pub const MAX_STREAMING_MULTIPROCESSORS: usize = 80;
// Max 2048 threads / SM
pub const MAX_THREAD_PER_SM: usize = 1 << 11;
// MAX 64 warps / SM
pub const MAX_WARP_PER_SM: usize = 1 << 6;

// todo: is this generic enough?
// Set a hard limit of 32 CTAs per shader (cuda only has 8)
pub const MAX_CTA_PER_SHADER: usize = 32;
pub const MAX_BARRIERS_PER_CTA: usize = 16;

pub const WARP_PER_CTA_MAX: usize = 64;
pub type WarpMask = BitArr!(for WARP_PER_CTA_MAX);

/// Start of the program memory space
///
/// Note: should be distinct from other memory spaces.
pub const PROGRAM_MEM_START: usize = 0xF0000000;

#[derive(Debug)]
pub struct ThreadState {
    pub block_id: usize,
    pub active: bool,
    pub pc: usize,
}

impl ThreadState {}

#[derive(Debug)]
pub struct ThreadInfo {}

#[derive(Debug, Default)]
pub struct InstrFetchBuffer {
    valid: bool,
    pc: address,
    num_bytes: usize,
    warp_id: usize,
}

impl InstrFetchBuffer {
    pub fn new() -> Self {
        Self {
            valid: false,
            pc: 0,
            num_bytes: 0,
            warp_id: 0,
        }
    }
}

type ResultBus = BitArr!(for fu::MAX_ALU_LATENCY);

#[derive()]
pub struct InnerSIMTCore<I> {
    pub core_id: usize,
    pub cluster_id: usize,
    pub cycle: super::Cycle,
    pub stats: Arc<Mutex<Stats>>,
    pub config: Arc<GPUConfig>,
    pub current_kernel: Option<Arc<KernelInfo>>,
    pub last_warp_fetched: Option<usize>,
    pub interconn: Arc<I>,
    pub load_store_unit: Arc<Mutex<LoadStoreUnit<ic::CoreMemoryInterface<Packet>>>>,
    pub active_thread_mask: BitArr!(for MAX_THREAD_PER_SM),
    pub occupied_hw_thread_ids: BitArr!(for MAX_THREAD_PER_SM),
    pub dynamic_warp_id: usize,
    pub num_active_blocks: usize,
    pub num_active_warps: usize,
    pub num_active_threads: usize,
    pub num_occupied_threads: usize,

    pub max_blocks_per_shader: usize,
    pub thread_block_size: usize,
    pub occupied_block_to_hw_thread_id: HashMap<usize, usize>,
    pub block_status: [usize; MAX_CTA_PER_SHADER],

    pub instr_l1_cache: Box<dyn cache::Cache>,
    pub instr_fetch_buffer: InstrFetchBuffer,
    pub warps: Vec<sched::CoreWarp>,
    pub thread_state: Vec<Option<ThreadState>>,
    pub thread_info: Vec<Option<ThreadInfo>>,
    pub scoreboard: Arc<RwLock<scoreboard::Scoreboard>>,
    pub operand_collector: Rc<RefCell<opcoll::OperandCollectorRegisterFileUnit>>,
    pub pipeline_reg: Vec<Rc<RefCell<register_set::RegisterSet>>>,
    pub result_busses: Vec<ResultBus>,
    pub barriers: barrier::BarrierSet,
}

impl<I> std::fmt::Debug for InnerSIMTCore<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InnerSIMTCore")
            .field("core_id", &self.core_id)
            .field("cluster_id", &self.cluster_id)
            .finish()
    }
}

#[derive(Debug)]
pub enum Packet {
    Fetch(mem_fetch::MemFetch),
}

impl std::fmt::Display for Packet {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Packet::Fetch(fetch) => write!(f, "{}", fetch),
        }
    }
}

pub trait WarpIssuer {
    fn issue_warp(
        &mut self,
        stage: PipelineStage,
        warp: &mut SchedulerWarp,
        next_inst: WarpInstruction,
        warp_id: usize,
        sch_id: usize,
    );

    fn has_free_register(&self, stage: PipelineStage, register_id: usize) -> bool;
}

impl<I> WarpIssuer for InnerSIMTCore<I>
where
    I: ic::Interconnect<Packet> + 'static,
{
    fn has_free_register(&self, stage: PipelineStage, register_id: usize) -> bool {
        let pipeline_stage = self.pipeline_reg[stage as usize].borrow();

        if self.config.sub_core_model {
            pipeline_stage.has_free_sub_core(register_id)
        } else {
            pipeline_stage.has_free()
        }
    }

    fn issue_warp(
        &mut self,
        stage: PipelineStage,
        warp: &mut SchedulerWarp,
        next_instr: WarpInstruction,
        warp_id: usize,
        sch_id: usize,
    ) {
        println!(
            "{}",
            style(format!(
                "cycle {:02} issue {} for warp {}",
                self.cycle.get(),
                next_instr,
                warp_id
            ))
            .yellow()
        );

        debug_assert_eq!(warp.warp_id, next_instr.warp_id);

        let mut pipeline_stage = self.pipeline_reg[stage as usize].borrow_mut();
        let pipe_reg = if self.config.sub_core_model {
            pipeline_stage.get_free_sub_core_mut(sch_id).unwrap()
        } else {
            pipeline_stage.get_free_mut().unwrap()
        };

        *pipe_reg = Some(next_instr);

        let pipe_reg_mut = pipe_reg.as_mut().unwrap();
        pipe_reg_mut.issue(
            pipe_reg_mut.active_mask,
            warp_id,
            0,
            warp.dynamic_warp_id,
            sch_id,
        );

        for t in 0..self.config.warp_size {
            if pipe_reg_mut.active_mask[t] {
                let warp_id = pipe_reg_mut.warp_id;
                let thread_id = self.config.warp_size * warp_id + t;

                if pipe_reg_mut.is_atomic() {
                    // warp.inc_n_atomic();
                }

                // if instr.memory_space.is_local() && (inst.is_load() || inst.is_store())) {
                //    new_addr_type localaddrs[MAX_ACCESSES_PER_INSN_PER_THREAD];
                //    unsigned num_addrs;
                //    num_addrs = translate_local_memaddr(
                //        inst.get_addr(t), tid,
                //        m_config->n_simt_clusters * m_config->n_simt_cores_per_cluster,
                //        inst.data_size, (new_addr_type *)localaddrs);
                //    inst.set_addr(t, (new_addr_type *)localaddrs, num_addrs);
                //  }
                if pipe_reg_mut.opcode.category == opcodes::ArchOp::EXIT_OPS {
                    warp.set_thread_completed(t);
                }
            }
        }

        // here, we generate memory acessess
        if pipe_reg_mut.is_load() || pipe_reg_mut.is_store() {
            if let Some(accesses) = pipe_reg_mut.generate_mem_accesses(&self.config) {
                pipe_reg_mut.mem_access_queue.extend(accesses);
            }
        }

        let pipe_reg_ref = pipe_reg.as_ref().unwrap();

        println!(
            "{} (done={} ({}/{}), functional done={}, hardware done={}, stores done={} ({} stores), instr in pipeline = {}, active_threads={})",
            style(format!("checking if warp {} did exit", warp.warp_id)).yellow(),
            warp.done(),
            warp.trace_pc,
            warp.instruction_count(),
            warp.functional_done(),
            warp.hardware_done(),
            warp.stores_done(),
            warp.num_outstanding_stores,
            warp.num_instr_in_pipeline,
            warp.active_mask.count_ones(),
        );

        if warp.done() && warp.functional_done() {
            warp.ibuffer_flush();
            // note: not modeling barriers for now
            // self.barriers.warp_exit(pipe_reg_ref.warp_id);
        }

        // let mut warp = self.warps.get_mut(warp_id).unwrap().lock().unwrap();
        if pipe_reg_ref.opcode.category == opcodes::ArchOp::BARRIER_OP {
            // m_warp[warp_id]->store_info_of_last_inst_at_barrier(*pipe_reg);
            // self.barriers.warp_reaches_barrier(warp.block_id, warp_id, next_inst);
        } else if pipe_reg_ref.opcode.category == opcodes::ArchOp::MEMORY_BARRIER_OP {
            // m_warp[warp_id]->set_membar();
            // warp.set_membar();
        }

        println!(
            "{} ({:?}) for instr {}",
            style(format!(
                "reserving {} registers",
                pipe_reg_ref.outputs().count()
            ))
            .yellow(),
            pipe_reg_ref.outputs().collect::<Vec<_>>(),
            pipe_reg_ref
        );

        self.scoreboard
            .write()
            .unwrap()
            .reserve_registers(pipe_reg_ref);

        println!(
            "post issue register set of {:?} pipeline: {}",
            stage, pipeline_stage
        );
    }
}

#[derive()]
pub struct SIMTCore<I> {
    pub issue_ports: Vec<PipelineStage>,
    pub dispatch_ports: Vec<PipelineStage>,
    pub functional_units: Vec<Arc<Mutex<dyn SimdFunctionUnit>>>,
    pub schedulers: Vec<Box<dyn sched::SchedulerUnit>>,
    pub inner: InnerSIMTCore<I>,
}

impl<I> std::fmt::Debug for SIMTCore<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.inner, f)
    }
}

#[derive(strum::EnumIter, strum::EnumCount, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum PipelineStage {
    ID_OC_SP = 0,
    ID_OC_DP = 1,
    ID_OC_INT = 2,
    ID_OC_SFU = 3,
    ID_OC_MEM = 4,
    OC_EX_SP = 5,
    OC_EX_DP = 6,
    OC_EX_INT = 7,
    OC_EX_SFU = 8,
    OC_EX_MEM = 9,
    EX_WB = 10,
    ID_OC_TENSOR_CORE = 11,
    OC_EX_TENSOR_CORE = 12,
}

impl<I> SIMTCore<I>
where
    I: ic::Interconnect<Packet> + 'static,
{
    pub fn new(
        core_id: usize,
        cluster_id: usize,
        cycle: super::Cycle,
        interconn: Arc<I>,
        stats: Arc<Mutex<Stats>>,
        config: Arc<GPUConfig>,
    ) -> Self {
        let thread_info: Vec<_> = (0..config.max_threads_per_core).map(|_| None).collect();
        let thread_state: Vec<_> = (0..config.max_threads_per_core).map(|_| None).collect();

        let warps: Vec<_> = (0..config.max_warps_per_core())
            .map(|_| Rc::new(RefCell::new(SchedulerWarp::default())))
            .collect();

        let port = Arc::new(ic::CoreMemoryInterface {
            cluster_id,
            stats: stats.clone(),
            config: config.clone(),
            interconn: interconn.clone(),
        });
        let cache_stats = Arc::new(Mutex::new(CacheStats::default()));
        let instr_l1_cache = l1::ReadOnly::new(
            format!(
                "core-{}-{}-{}",
                cluster_id,
                core_id,
                style("READONLY-INSTR-CACHE").green()
            ),
            core_id,
            cluster_id,
            port.clone(),
            cache_stats,
            config.clone(),
            config.inst_cache_l1.as_ref().unwrap().clone(),
        );

        // todo: are those parameters correct?
        // m_barriers(this, config->max_warps_per_shader, config->max_cta_per_core, config->max_barriers_per_cta, config->warp_size);
        let barriers = barrier::BarrierSet::new(
            config.max_warps_per_core(),
            config.max_concurrent_blocks_per_core,
            config.num_cta_barriers,
            config.warp_size,
        );

        let scoreboard = Arc::new(RwLock::new(scoreboard::Scoreboard::new(
            core_id,
            cluster_id,
            config.max_warps_per_core(),
        )));

        let mut operand_collector = opcoll::OperandCollectorRegisterFileUnit::new(config.clone());

        let operand_collector = Rc::new(RefCell::new(operand_collector));

        // pipeline_stages is the sum of normal pipeline stages
        // and specialized_unit stages * 2 (for ID and EX)
        // let total_pipeline_stages = PipelineStage::COUNT + config.num_specialized_unit.len() * 2;;
        // let pipeline_reg = (0..total_pipeline_stages)

        let pipeline_reg: Vec<_> = PipelineStage::iter()
            .map(|stage| {
                let pipeline_width = config.pipeline_widths.get(&stage).copied().unwrap_or(0);
                register_set::RegisterSet::new(stage, pipeline_width)
            })
            .collect();

        // SKIPPING SPECIALIZED UNITS
        // for (int j = 0; j < m_config->m_specialized_unit.size(); j++) {
        //   m_pipeline_reg.push_back(
        //       register_set(m_config->m_specialized_unit[j].id_oc_spec_reg_width,
        //                    m_config->m_specialized_unit[j].name));
        //   m_config->m_specialized_unit[j].ID_OC_SPEC_ID = m_pipeline_reg.size() - 1;
        //   m_specilized_dispatch_reg.push_back(
        //       &m_pipeline_reg[m_pipeline_reg.size() - 1]);
        // }
        // for (int j = 0; j < m_config->m_specialized_unit.size(); j++) {
        //   m_pipeline_reg.push_back(
        //       register_set(m_config->m_specialized_unit[j].oc_ex_spec_reg_width,
        //                    m_config->m_specialized_unit[j].name));
        //   m_config->m_specialized_unit[j].OC_EX_SPEC_ID = m_pipeline_reg.size() - 1;
        // }

        if config.sub_core_model {
            // in subcore model, each scheduler should has its own
            // issue register, so ensure num scheduler = reg width
            debug_assert_eq!(
                config.num_schedulers_per_core,
                pipeline_reg[PipelineStage::ID_OC_SP as usize].size()
            );
            debug_assert_eq!(
                config.num_schedulers_per_core,
                pipeline_reg[PipelineStage::ID_OC_SFU as usize].size()
            );
            debug_assert_eq!(
                config.num_schedulers_per_core,
                pipeline_reg[PipelineStage::ID_OC_MEM as usize].size()
            );
            // if (m_config->gpgpu_tensor_core_avail)
            //   assert(m_config->gpgpu_num_sched_per_core ==
            //          m_pipeline_reg[ID_OC_TENSOR_CORE].get_size());
            // if (m_config->gpgpu_num_dp_units > 0)
            //   assert(m_config->gpgpu_num_sched_per_core ==
            //          m_pipeline_reg[ID_OC_DP].get_size());
            // if (m_config->gpgpu_num_int_units > 0)
            //   assert(m_config->gpgpu_num_sched_per_core ==
            //          m_pipeline_reg[ID_OC_INT].get_size());
            // for (int j = 0; j < m_config->m_specialized_unit.size(); j++) {
            //   if (m_config->m_specialized_unit[j].num_units > 0)
            //     assert(m_config->gpgpu_num_sched_per_core ==
            //            m_config->m_specialized_unit[j].id_oc_spec_reg_width);
            // }
        }

        let fetch_interconn = Arc::new(ic::CoreMemoryInterface {
            cluster_id,
            interconn: interconn.clone(),
            stats: stats.clone(),
            config: config.clone(),
        });

        let load_store_unit = Arc::new(Mutex::new(LoadStoreUnit::new(
            core_id,
            cluster_id,
            warps.clone(),
            fetch_interconn.clone(),
            operand_collector.clone(),
            scoreboard.clone(),
            config.clone(),
            stats.clone(),
            Rc::clone(&cycle),
        )));

        // there are as many result buses as the width of the EX_WB stage
        let result_busses: Vec<_> = (0..pipeline_reg[PipelineStage::EX_WB as usize].size())
            .map(|_| BitArray::ZERO)
            .collect();

        let pipeline_reg: Vec<_> = pipeline_reg
            .into_iter()
            .map(|reg| Rc::new(RefCell::new(reg)))
            .collect();

        let mut inner = InnerSIMTCore {
            core_id,
            cluster_id,
            cycle,
            stats,
            config: config.clone(),
            current_kernel: None,
            last_warp_fetched: None,
            active_thread_mask: BitArray::ZERO,
            occupied_hw_thread_ids: BitArray::ZERO,
            dynamic_warp_id: 0,
            num_active_blocks: 0,
            num_active_warps: 0,
            num_active_threads: 0,
            num_occupied_threads: 0,
            max_blocks_per_shader: 0,
            thread_block_size: 0,
            occupied_block_to_hw_thread_id: HashMap::new(),
            block_status: [0; MAX_CTA_PER_SHADER],
            instr_l1_cache: Box::new(instr_l1_cache),
            instr_fetch_buffer: InstrFetchBuffer::default(),
            interconn,
            load_store_unit,
            warps: warps.clone(),
            pipeline_reg,
            result_busses,
            scoreboard: scoreboard.clone(),
            operand_collector,
            barriers,
            thread_state,
            thread_info,
        };
        let mut core = Self {
            inner,
            schedulers: Vec::new(),
            issue_ports: Vec::new(),
            dispatch_ports: Vec::new(),
            functional_units: Vec::new(),
        };

        core.init_schedulers();
        core.init_functional_units();
        core.init_operand_collectors();
        core
    }

    fn init_operand_collectors(&mut self) {
        let mut operand_collector = self.inner.operand_collector.try_borrow_mut().unwrap();

        // configure generic collectors
        operand_collector.add_cu_set(
            opcoll::OperandCollectorUnitKind::GEN_CUS,
            self.inner.config.operand_collector_num_units_gen,
            self.inner.config.operand_collector_num_out_ports_gen,
        );

        for i in 0..self.inner.config.operand_collector_num_in_ports_gen {
            let mut in_ports = opcoll::PortVec::new();
            let mut out_ports = opcoll::PortVec::new();
            let mut cu_sets: Vec<opcoll::OperandCollectorUnitKind> = Vec::new();

            in_ports.push(self.inner.pipeline_reg[PipelineStage::ID_OC_SP as usize].clone());
            // in_ports.push_back(&m_pipeline_reg[ID_OC_SFU]);
            // in_ports.push(&self.pipeline_reg[ID_OC_MEM]);
            out_ports.push(self.inner.pipeline_reg[PipelineStage::OC_EX_SP as usize].clone());
            // out_ports.push_back(&m_pipeline_reg[OC_EX_SFU]);
            // out_ports.push(&self.pipeline_reg[OC_EX_MEM]);
            // if (m_config->gpgpu_tensor_core_avail) {
            //   in_ports.push_back(&m_pipeline_reg[ID_OC_TENSOR_CORE]);
            //   out_ports.push_back(&m_pipeline_reg[OC_EX_TENSOR_CORE]);
            // }
            // if (m_config->gpgpu_num_dp_units > 0) {
            //   in_ports.push_back(&m_pipeline_reg[ID_OC_DP]);
            //   out_ports.push_back(&m_pipeline_reg[OC_EX_DP]);
            // }
            // if (m_config->gpgpu_num_int_units > 0) {
            //   in_ports.push_back(&m_pipeline_reg[ID_OC_INT]);
            //   out_ports.push_back(&m_pipeline_reg[OC_EX_INT]);
            // }
            // if (m_config->m_specialized_unit.size() > 0) {
            //   for (unsigned j = 0; j < m_config->m_specialized_unit.size(); ++j) {
            //     in_ports.push_back(
            //         &m_pipeline_reg[m_config->m_specialized_unit[j].ID_OC_SPEC_ID]);
            //     out_ports.push_back(
            //         &m_pipeline_reg[m_config->m_specialized_unit[j].OC_EX_SPEC_ID]);
            //   }
            // }
            // cu_sets.push_back((unsigned)GEN_CUS);
            // m_operand_collector.add_port(in_ports, out_ports, cu_sets);
            // in_ports.clear(), out_ports.clear(), cu_sets.clear();
            cu_sets.push(opcoll::OperandCollectorUnitKind::GEN_CUS);
            operand_collector.add_port(in_ports, out_ports, cu_sets);
            // in_ports.clear();
            // out_ports.clear();
            // cu_sets.clear();
        }

        // let enable_specialized_operand_collector = true;
        if self.inner.config.enable_specialized_operand_collector {
            // only added two
            operand_collector.add_cu_set(
                opcoll::OperandCollectorUnitKind::SP_CUS,
                self.inner.config.operand_collector_num_units_sp,
                self.inner.config.operand_collector_num_out_ports_sp,
            );
            operand_collector.add_cu_set(
                opcoll::OperandCollectorUnitKind::MEM_CUS,
                self.inner.config.operand_collector_num_units_mem,
                self.inner.config.operand_collector_num_out_ports_mem,
            );

            for i in 0..self.inner.config.operand_collector_num_in_ports_sp {
                let mut in_ports = opcoll::PortVec::new();
                let mut out_ports = opcoll::PortVec::new();
                let mut cu_sets: Vec<opcoll::OperandCollectorUnitKind> = Vec::new();

                in_ports.push(self.inner.pipeline_reg[PipelineStage::ID_OC_SP as usize].clone());
                out_ports.push(self.inner.pipeline_reg[PipelineStage::OC_EX_SP as usize].clone());
                cu_sets.push(opcoll::OperandCollectorUnitKind::SP_CUS);
                cu_sets.push(opcoll::OperandCollectorUnitKind::GEN_CUS);
                operand_collector.add_port(in_ports, out_ports, cu_sets);
            }

            for i in 0..self.inner.config.operand_collector_num_in_ports_mem {
                let mut in_ports = opcoll::PortVec::new();
                let mut out_ports = opcoll::PortVec::new();
                let mut cu_sets: Vec<opcoll::OperandCollectorUnitKind> = Vec::new();

                in_ports.push(self.inner.pipeline_reg[PipelineStage::ID_OC_MEM as usize].clone());
                out_ports.push(self.inner.pipeline_reg[PipelineStage::OC_EX_MEM as usize].clone());
                cu_sets.push(opcoll::OperandCollectorUnitKind::MEM_CUS);
                cu_sets.push(opcoll::OperandCollectorUnitKind::GEN_CUS);
                operand_collector.add_port(in_ports, out_ports, cu_sets);
            }
        }

        // this must be called after we add the collector unit sets!
        operand_collector.init(self.inner.config.num_reg_banks);
    }

    fn init_functional_units(&mut self) {
        // single precision units
        for u in 0..self.inner.config.num_sp_units {
            self.functional_units
                .push(Arc::new(Mutex::new(super::SPUnit::new(
                    u, // id
                    Rc::clone(&self.inner.pipeline_reg[PipelineStage::EX_WB as usize]),
                    Arc::clone(&self.inner.config),
                    Arc::clone(&self.inner.stats),
                    Rc::clone(&self.inner.cycle),
                    u, // issue reg id
                ))));
            self.dispatch_ports.push(PipelineStage::ID_OC_SP);
            self.issue_ports.push(PipelineStage::OC_EX_SP);
        }

        // load store unit
        self.functional_units
            .push(self.inner.load_store_unit.clone());
        self.dispatch_ports.push(PipelineStage::OC_EX_MEM);
        self.issue_ports.push(PipelineStage::OC_EX_MEM);

        debug_assert_eq!(self.functional_units.len(), self.issue_ports.len());
        debug_assert_eq!(self.functional_units.len(), self.dispatch_ports.len());
    }

    fn init_schedulers(&mut self) {
        // let scheduler_kind = config::SchedulerKind::LRR;
        let scheduler_kind = config::SchedulerKind::GTO;

        self.schedulers = (0..self.inner.config.num_schedulers_per_core)
            .map(|sched_id| match scheduler_kind {
                config::SchedulerKind::LRR => {
                    let mem_out = &self.inner.pipeline_reg[PipelineStage::ID_OC_MEM as usize];
                    Box::new(sched::LrrScheduler::new(
                        // &self.inner.warps,
                        sched_id,
                        self.inner.warps.clone(),
                        // mem_out,
                        // &self.inner,
                        self.inner.scoreboard.clone(),
                        self.inner.stats.clone(),
                        self.inner.config.clone(),
                        // m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
                        // &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
                        // &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
                        // &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
                        // &m_pipeline_reg[ID_OC_MEM], i
                    )) as Box<dyn sched::SchedulerUnit>
                    // self.schedulers.push_back(Box::new(lrr));
                }
                config::SchedulerKind::GTO => {
                    Box::new(sched::GTOScheduler::new(
                        // &self.inner.warps,
                        sched_id,
                        self.inner.warps.clone(),
                        // mem_out,
                        // &self.inner,
                        self.inner.scoreboard.clone(),
                        self.inner.stats.clone(),
                        self.inner.config.clone(),
                        // &self.inner.pipeline_reg[PipelineStage::ID_OC_MEM as usize],
                        // m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
                        // &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
                        // &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
                        // &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
                        // &m_pipeline_reg[ID_OC_MEM], i

                        // ORIGINAL PARAMS
                        // m_stats,
                        // this,
                        // m_scoreboard,
                        // m_simt_stack,
                        // &m_warp,
                        // &m_pipeline_reg[ID_OC_SP],
                        // &m_pipeline_reg[ID_OC_DP],
                        // &m_pipeline_reg[ID_OC_SFU],
                        // &m_pipeline_reg[ID_OC_INT],
                        // &m_pipeline_reg[ID_OC_TENSOR_CORE],
                        // m_specilized_dispatch_reg,
                        // &m_pipeline_reg[ID_OC_MEM],
                        // i,
                    )) as Box<dyn sched::SchedulerUnit>
                    // schedulers.push_back(gto);
                }
                //     SchedulerKind::TwoLevelActive => {
                // Box::new(sched::TwoLevelActiveScheduler::new(
                //                   m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
                //                   &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
                //                   &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
                //                   &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
                //                   &m_pipeline_reg[ID_OC_MEM], i, m_config->gpgpu_scheduler_string);
                //               schedulers.push_back(tla);
                //         },
                other => todo!("scheduler: {:?}", &other),
                //         SchedulerKind::RRR => {
                //                     let rrr = RrrScheduler::new(
                //                   m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
                //                   &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
                //                   &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
                //                   &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
                //                   &m_pipeline_reg[ID_OC_MEM], i);
                //               schedulers.push_back(rrr);
                //         },
                //             SchedulerKind::OldestFirst => {
                //                     let oldest = OldestScheduler::new(
                //                   m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
                //                   &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
                //                   &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
                //                   &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
                //                   &m_pipeline_reg[ID_OC_MEM], i);
                //               schedulers.push_back(oldest);
                //         },
                //             SchedulerKind::WarpLimiting => {
                //                     let swl = SwlScheduler::new(
                //                   m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
                //                   &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
                //                   &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
                //                   &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
                //                   &m_pipeline_reg[ID_OC_MEM], i, m_config->gpgpu_scheduler_string);
                //               schedulers.push_back(swl);
                //         },
            })
            .collect();
        // }

        for (i, warp) in self.inner.warps.iter().enumerate() {
            // distribute i's evenly though schedulers;
            let sched_idx = i % self.inner.config.num_schedulers_per_core;
            let scheduler = &mut self.schedulers[sched_idx];
            scheduler.add_supervised_warp(warp.clone());
        }
        for scheduler in self.schedulers.iter_mut() {
            // todo!("call done_adding_supervised_warps");
            scheduler.done_adding_supervised_warps();
        }
        // for (unsigned i = 0; i < m_config->gpgpu_num_sched_per_core; ++i) {
        //   schedulers[i]->done_adding_supervised_warps();
        // }
    }

    pub fn active(&self) -> bool {
        self.inner.num_active_blocks > 0
    }

    /// return the next pc of a thread
    pub fn next_pc(&mut self, thread_id: usize) -> Option<usize> {
        // if (tid == -1) return -1;
        // PC should already be updatd to next PC at this point (was
        // set in shader_decode() last time thread ran)
        self.inner
            .thread_state
            .get(thread_id)
            .map(Option::as_ref)
            .flatten()
            .map(|t| t.pc)
    }

    fn register_thread_in_block_exited(&mut self, block_id: usize, kernel: Option<&KernelInfo>) {
        let current_kernel: &mut Option<_> =
            &mut self.inner.current_kernel.as_ref().map(|k| k.as_ref());

        debug_assert!(self.inner.block_status[block_id] > 0);
        self.inner.block_status[block_id] -= 1;
        if self.inner.block_status[block_id] == 0 {
            // Increment the completed CTAs
            //   m_stats->ctas_completed++;
            //   m_gpu->inc_completed_cta();
            self.inner.num_active_blocks -= 1;
            //   m_barriers.deallocate_barrier(cta_num);
            //   shader_CTA_count_unlog(m_sid, 1);
            //
            //   SHADER_DPRINTF(
            //       LIVENESS,
            //       "GPGPU-Sim uArch: Finished CTA #%u (%lld,%lld), %u CTAs running\n",
            //       cta_num, m_gpu->gpu_sim_cycle, m_gpu->gpu_tot_sim_cycle,
            //       m_n_active_cta);
            //
            if self.inner.num_active_blocks == 0 {
                //     SHADER_DPRINTF(
                //         LIVENESS,
                //         "GPGPU-Sim uArch: Empty (last released kernel %u \'%s\').\n",
                //         kernel->get_uid(), kernel->name().c_str());
                //     fflush(stdout);
                //
                // Shader can only be empty when no more cta are dispatched
                if kernel != *current_kernel {
                    // debug_assert!(current_kernel.is_none() || kernel.no_more_blocks_to_run());
                }
                *current_kernel = None;
            }
            //
            //   // Jin: for concurrent kernels on sm
            // self.release_shader_resource_1block(cta_num, kernel);
            //   kernel->dec_running();
            if let Some(kernel) = kernel {
                if kernel.no_more_blocks_to_run() {
                    if !kernel.running() {
                        //       SHADER_DPRINTF(LIVENESS,
                        //                      "GPGPU-Sim uArch: GPU detected kernel %u \'%s\' "
                        //                      "finished on shader %u.\n",
                        //                      kernel->get_uid(), kernel->name().c_str(), m_sid);
                        //
                        if *current_kernel == Some(&kernel) {
                            *current_kernel = None;
                        }
                        // m_gpu->set_kernel_done(kernel);
                    }
                }
            }
        }
    }

    fn fetch(&mut self) {
        println!(
            "{}",
            style(format!(
                "cycle {:03} core {:?}: fetch (fetch buffer valid={}, l1i ready={:?})",
                self.inner.cycle.get(),
                self.id(),
                self.inner.instr_fetch_buffer.valid,
                self.inner
                    .instr_l1_cache
                    .ready_accesses()
                    .cloned()
                    .unwrap_or_default()
                    .iter()
                    .map(|access| access.to_string())
                    .collect::<Vec<_>>(),
            ))
            .green()
        );

        if !self.inner.instr_fetch_buffer.valid {
            if self.inner.instr_l1_cache.has_ready_accesses() {
                let fetch = self.inner.instr_l1_cache.next_access().unwrap();
                let warp = self.inner.warps.get_mut(fetch.warp_id).unwrap();
                let mut warp = warp.try_borrow_mut().unwrap();
                warp.has_imiss_pending = false;

                let pc = warp.pc().unwrap() as u64;
                self.inner.instr_fetch_buffer = InstrFetchBuffer {
                    valid: true,
                    pc,
                    num_bytes: fetch.data_size as usize,
                    warp_id: fetch.warp_id,
                };

                // verify that we got the instruction we were expecting.
                // TODO: this does not work because the fetch.addr() is not the same anymore?
                // it gets changed to the block addr on the way and not ever changed back..
                // debug_assert_eq!(
                //     warp.pc(),
                //     Some(fetch.addr() as usize - super::PROGRAM_MEM_START)
                // );

                self.inner.instr_fetch_buffer.valid = true;
                // warp.set_last_fetch(m_gpu->gpu_sim_cycle);
                // drop(fetch);
            } else {
                // find an active warp with space in
                // instruction buffer that is not
                // already waiting on a cache miss and get
                // next 1-2 instructions from instruction cache
                let max_warps = self.inner.config.max_warps_per_core();

                // println!(
                //     "{}: instr fetch buffer not valid (checking {max_warps} warps now)",
                //     style("empty instruction cache").red(),
                // );

                for warp_id in 0..max_warps {
                    let warp = self.inner.warps[warp_id].try_borrow().unwrap();

                    let pending_writes = self
                        .inner
                        .scoreboard
                        .read()
                        .unwrap()
                        .pending_writes(warp_id)
                        .clone();

                    if warp.functional_done() && warp.hardware_done() && warp.done_exit() {
                        continue;
                    }
                    println!(
                        "checking warp_id = {} (instruction count={}, hardware_done={}, functional_done={}, instr in pipe={}, stores={}, done_exit={}, pending writes={:?})",
                        &warp_id,
                        warp.instruction_count(),
                        warp.hardware_done(),
                        warp.functional_done(),
                        warp.num_instr_in_pipeline,
                        warp.num_outstanding_stores,
                        warp.done_exit(),
                        pending_writes.iter().sorted().collect::<Vec<_>>()
                    );
                }

                println!("\n\n");
                for i in 0..max_warps {
                    let last = self.inner.last_warp_fetched.unwrap_or(0);
                    let warp_id = (last + 1 + i) % max_warps;

                    let warp = self.inner.warps[warp_id].try_borrow().unwrap();

                    let block_id = warp.block_id as usize;
                    let kernel = warp.kernel.as_ref().map(Arc::clone);

                    let pending_writes = self
                        .inner
                        .scoreboard
                        .read()
                        .unwrap()
                        .pending_writes(warp_id)
                        .clone();

                    if !(warp.hardware_done() && warp.functional_done() && warp.done_exit()) {
                        println!(
                            "\n checking warp_id = {} (instruction count={}, hardware_done={}, functional_done={}, instr in pipe={}, stores={}, done_exit={}, pending writes={:?})",
                            &warp_id,
                            warp.instruction_count(),
                            warp.hardware_done(),
                            warp.functional_done(),
                            warp.num_instr_in_pipeline,
                            warp.num_outstanding_stores,
                            warp.done_exit(),
                        pending_writes.iter().sorted().collect::<Vec<_>>()
                        );
                    }

                    let did_maybe_exit =
                        warp.hardware_done() && pending_writes.is_empty() && !warp.done_exit();

                    drop(warp);

                    // check if this warp has finished executing and can be reclaimed.
                    let mut did_exit = false;
                    if did_maybe_exit {
                        for t in 0..self.inner.config.warp_size {
                            let tid = warp_id * self.inner.config.warp_size + t;
                            if let Some(Some(state)) = self.inner.thread_state.get_mut(tid) {
                                if state.active {
                                    state.active = false;
                                    self.register_thread_in_block_exited(
                                        block_id,
                                        kernel.as_ref().map(Arc::as_ref),
                                    );

                                    // if let Some(Some(thread_info)) =
                                    //     self.inner.thread_info.get(tid).map(Option::as_ref)
                                    // {
                                    //     // self.register_thread_in_block_exited(block_id, &(m_thread[tid]->get_kernel()));
                                    //     self.register_thread_in_block_exited(
                                    //         block_id,
                                    //         thread_info.kernel,
                                    //         // kernel.as_ref().map(Arc::as_ref),
                                    //     );
                                    // } else {
                                    //     self.register_thread_in_block_exited(
                                    //         block_id,
                                    //         kernel.as_ref().map(Arc::as_ref),
                                    //     );
                                    // }
                                    self.inner.num_active_threads -= 1;
                                    self.inner.active_thread_mask.set(tid, false);
                                    did_exit = true;
                                }
                            }
                        }
                        self.inner.num_active_warps -= 1;
                        debug_assert!(self.inner.num_active_warps >= 0);
                    }

                    let mut warp = self.inner.warps[warp_id].try_borrow_mut().unwrap();
                    if did_exit {
                        // todo!("first warp did exit");
                        println!("warp_id = {} exited", &warp_id);
                        // if warp_id == 3 {
                        //     panic!("warp 3 exited");
                        // }

                        warp.done_exit = true;
                    }

                    let icache_config = self.inner.config.inst_cache_l1.as_ref().unwrap();
                    let should_fetch_instruction = !warp.trace_instructions.is_empty()
                        && !warp.functional_done()
                        && !warp.has_imiss_pending
                        && warp.ibuffer_empty();

                    // this code fetches instructions
                    // from the i-cache or generates memory
                    if should_fetch_instruction {
                        let instr = warp.current_instr().unwrap();
                        let pc = warp.pc().unwrap();
                        let ppc = pc + PROGRAM_MEM_START;

                        println!(
                            "\t fetching instr {} for warp_id = {} (pc={}, ppc={})",
                            &instr, warp.warp_id, pc, ppc,
                        );

                        let mut num_bytes = 16;
                        let line_size = icache_config.line_size as usize;
                        let offset_in_block = pc & (line_size - 1);
                        if offset_in_block + num_bytes > line_size as usize {
                            num_bytes = line_size as usize - offset_in_block;
                        }
                        let access = mem_fetch::MemAccess::new(
                            mem_fetch::AccessKind::INST_ACC_R,
                            ppc as u64,
                            num_bytes as u32,
                            false,
                            // todo: is this correct?
                            BitArray::ZERO,
                            BitArray::ZERO,
                            BitArray::ZERO,
                        );
                        let fetch = mem_fetch::MemFetch::new(
                            None,
                            access,
                            &*self.inner.config,
                            mem_fetch::READ_PACKET_SIZE.into(),
                            warp_id,
                            self.inner.core_id,
                            self.inner.cluster_id,
                        );

                        let status = if self.inner.config.perfect_inst_const_cache {
                            // shader_cache_access_log(m_sid, INSTRUCTION, 0);
                            cache::RequestStatus::HIT
                        } else {
                            let mut events = Vec::new();
                            self.inner
                                .instr_l1_cache
                                .access(ppc as address, fetch, &mut events)
                        };

                        println!("L1I->access(addr={}) -> status = {:?}", ppc, status);

                        self.inner.last_warp_fetched = Some(warp_id);

                        if status == cache::RequestStatus::MISS {
                            // let warp = self.inner.warps.get_mut(warp_id).unwrap();
                            // let warp = warp.lock().unwrap();
                            // .as_mut()
                            // .unwrap();
                            warp.has_imiss_pending = true;
                            // warp.set_last_fetch(m_gpu->gpu_sim_cycle);
                        } else if status == cache::RequestStatus::HIT {
                            self.inner.instr_fetch_buffer = InstrFetchBuffer {
                                valid: true,
                                pc: pc as u64,
                                num_bytes,
                                warp_id,
                            };
                            // m_warp[warp_id]->set_last_fetch(m_gpu->gpu_sim_cycle);
                            // delete mf;
                        } else {
                            debug_assert_eq!(status, cache::RequestStatus::RESERVATION_FAIL);
                            // delete mf;
                        }
                        break;
                    }
                    // }
                }
                // println!("\n\n");
            }
        }
        self.inner.instr_l1_cache.cycle();
    }

    /// shader core decode pipeline stage
    ///
    /// NOTE: inst fetch buffer valid after 279 cycles
    ///
    /// investigate:
    /// - fetch buffer becomes valid when icache has access ready
    /// - icache has access ready whenm mshrs has next access
    /// - mshrs has next access when mshrs::current_response queue is not empty
    /// - mshrs::current_response is pushed into by mshr_table::mark_ready
    /// - mshr_table::mark_ready is called by baseline_cache::fill
    /// - only trace_shader_core_ctx::accept_fetch_response calls baseline_cache::fill
    /// - only void simt_core_cluster::icnt_cycle() calls accept_fetch_response when there is a
    /// response
    fn decode(&mut self) {
        let InstrFetchBuffer {
            valid, pc, warp_id, ..
        } = self.inner.instr_fetch_buffer;

        let core_id = self.id();
        // println!("core {:?}: {}", core_id, style("decode").red());
        println!(
            "{}",
            style(format!(
                "cycle {:03} core {:?}: decode (fetch buffer valid={})",
                self.inner.cycle.get(),
                self.id(),
                valid
            ))
            .blue()
        );

        if !valid {
            return;
        }

        // decode 1 or 2 instructions and buffer them
        let mut warp = self
            .inner
            .warps
            .get_mut(warp_id)
            .unwrap()
            .try_borrow_mut()
            .unwrap();
        debug_assert_eq!(warp.warp_id, warp_id);
        let instr1 = warp.next_trace_inst();
        let instr2 = if instr1.is_some() {
            warp.next_trace_inst()
        } else {
            None
        };

        // debug: print all instructions in this warp
        for (trace_pc, trace_instr) in warp.trace_instructions.iter().enumerate() {
            println!(
                "====> warp[warp_id={:03}][trace_pc={:03}]:\t {}\t\t active={} \tpc = {}",
                warp_id,
                trace_pc,
                trace_instr,
                trace_instr.active_mask.to_bit_string(),
                trace_instr.pc
            );
        }
        drop(warp);

        if let Some(instr1) = instr1 {
            self.decode_instruction(warp_id, instr1, 0);
        }

        if let Some(instr2) = instr2 {
            self.decode_instruction(warp_id, instr2, 1);
        }

        self.inner.instr_fetch_buffer.valid = false;
    }

    fn decode_instruction(&mut self, warp_id: usize, instr: WarpInstruction, slot: usize) {
        let core_id = self.id();
        let warp = self.inner.warps.get_mut(warp_id).unwrap();
        let mut warp = warp.try_borrow_mut().unwrap();

        println!(
            "====> warp[warp_id={:03}] ibuffer fill at slot {:01} with instruction {}",
            warp.warp_id, slot, instr,
        );

        warp.ibuffer_fill(slot, instr);
        warp.num_instr_in_pipeline += 1;

        // self.stats->m_num_decoded_insn[m_sid]++;
        // use super::instruction::ArchOp;
        // match instr1.opcode.category {
        //     ArchOp::INT_OP | ArchOp::UN_OP => {
        //         //these counters get added up in mcPat to compute scheduler power
        //         // m_stats->m_num_INTdecoded_insn[m_sid]++;
        //     }
        //     ArchOp::FP_OP => {
        //         // m_stats->m_num_FPdecoded_insn[m_sid]++;
        //     }
        //     _ => {}
        // }

        // drop(instr1);
        // drop(warp);
    }

    fn issue(&mut self) {
        for scheduler in &mut self.schedulers {
            scheduler.cycle(&mut self.inner);
            // scheduler.cycle(());
        }
    }

    fn writeback(&mut self) {
        // from the functional units
        let mut exec_writeback_pipeline =
            self.inner.pipeline_reg[PipelineStage::EX_WB as usize].borrow_mut();
        println!(
            "{}",
            style(format!(
                "cycle {:03} core {:?}: writeback: ex wb pipeline={}",
                self.inner.cycle.get(),
                self.id(),
                exec_writeback_pipeline
            ))
            .cyan()
        );
        let max_committed_thread_instructions =
            self.inner.config.warp_size * exec_writeback_pipeline.size();

        // m_stats->m_pipeline_duty_cycle[m_sid] =
        //     ((float)(m_stats->m_num_sim_insn[m_sid] -
        //              m_stats->m_last_num_sim_insn[m_sid])) /
        //     max_committed_thread_instructions;
        //
        // m_stats->m_last_num_sim_insn[m_sid] = m_stats->m_num_sim_insn[m_sid];
        // m_stats->m_last_num_sim_winsn[m_sid] = m_stats->m_num_sim_winsn[m_sid];
        //
        // let preg = ex_wb_stage.get_ready_mut();
        // let pipe_reg = (preg == NULL) ? NULL : *preg;
        // while preg.is_some() { // && !pipe_reg.empty() {
        while let Some(mut ready) = exec_writeback_pipeline
            .get_ready_mut()
            .and_then(|(_, r)| r.take())
        {
            println!("ready for writeback: {}", ready);

            // Right now, the writeback stage drains all waiting instructions
            // assuming there are enough ports in the register file or the
            // conflicts are resolved at issue.
            //
            // The operand collector writeback can generally generate a stall
            // However, here, the pipelines should be un-stallable. This is
            // guaranteed because this is the first time the writeback function
            // is called after the operand collector's step function, which
            // resets the allocations. There is one case which could result in
            // the writeback function returning false (stall), which is when
            // an instruction tries to modify two registers (GPR and predicate)
            // To handle this case, we ignore the return value (thus allowing
            // no stalling).
            //
            self.inner.operand_collector.borrow_mut().writeback(&ready);
            self.inner
                .scoreboard
                .write()
                .unwrap()
                .release_registers(&ready);
            self.inner.warps[ready.warp_id]
                .try_borrow_mut()
                .unwrap()
                .num_instr_in_pipeline -= 1;
            super::warp_inst_complete(&mut ready, &mut self.inner.stats.lock().unwrap());

            //   m_gpu->gpu_sim_insn_last_update_sid = m_sid;
            //   m_gpu->gpu_sim_insn_last_update = m_gpu->gpu_sim_cycle;
            //   m_last_inst_gpu_sim_cycle = m_gpu->gpu_sim_cycle;
            //   m_last_inst_gpu_tot_sim_cycle = m_gpu->gpu_tot_sim_cycle;
            // preg = m_pipeline_reg[EX_WB].get_ready();
            //   pipe_reg = (preg == NULL) ? NULL : *preg;
        }
    }

    fn execute(&mut self) {
        use mem_fetch::BitString;

        let core_id = self.id();
        println!(
            "{}",
            style(format!(
                "cycle {:03} core {:?} execute: ",
                self.inner.cycle.get(),
                core_id
            ))
            .red()
        );

        for (i, res_bus) in self.inner.result_busses.iter_mut().enumerate() {
            res_bus.shift_right(1);
            println!(
                "res bus {:03}[:128]: {}",
                i,
                &res_bus.to_bit_string()[0..128]
            );
        }

        for (fu_id, fu) in self.functional_units.iter_mut().enumerate() {
            let mut fu = fu.try_lock().unwrap();

            let issue_port = self.issue_ports[fu_id];
            {
                let issue_inst = self.inner.pipeline_reg[issue_port as usize].borrow();
                println!(
                    "fu[{:03}] {:<10} before \t{:?}={}",
                    &fu_id,
                    fu.to_string(),
                    issue_port,
                    issue_inst
                );
            }

            fu.cycle();
            fu.active_lanes_in_pipeline();

            let mut issue_inst = self.inner.pipeline_reg[issue_port as usize].borrow_mut();
            println!(
                "fu[{:03}] {:<10} after \t{:?}={}",
                &fu_id,
                fu.to_string(),
                issue_port,
                issue_inst
            );

            let partition_issue = self.inner.config.sub_core_model && fu.is_issue_partitioned();
            let ready_reg: Option<&mut Option<WarpInstruction>> = if partition_issue {
                let reg_id = fu.issue_reg_id();
                issue_inst.get_ready_sub_core_mut(reg_id)
            } else {
                issue_inst.get_ready_mut().map(|(_, r)| r)
            };

            let Some(ready_reg) = ready_reg else {
                // continue
                continue;
            };

            // if let Some(ref mut ready_reg @ Some(instr)) = ready_reg {
            if let Some(ref instr) = ready_reg {
                // todo!("ready for issue to functional unit");
                if fu.can_issue(instr) {
                    let schedule_wb_now = !fu.stallable();
                    let result_bus = self
                        .inner
                        .result_busses
                        .iter_mut()
                        .filter(|bus| !bus[instr.latency])
                        .next();

                    println!(
                        "{}",
                        style(format!(
                            "cycle {:03} core={:?} execute: {} ready for issue to fu[{:03}]={}",
                            self.inner.cycle.get(),
                            core_id,
                            instr,
                            fu_id,
                            fu,
                        ))
                        .red()
                    );

                    let mut issued = true;
                    match result_bus {
                        Some(result_bus) if schedule_wb_now => {
                            debug_assert!(instr.latency < fu::MAX_ALU_LATENCY);
                            result_bus.set(instr.latency, true);
                            // fu.issue(&mut issue_inst);
                            // let ready_reg = ready_reg.take();
                            fu.issue(ready_reg.take().unwrap());
                        }
                        _ if !schedule_wb_now => {
                            // fu.issue(&mut issue_inst);
                            fu.issue(ready_reg.take().unwrap());
                        }
                        _ => {
                            // stall issue (cannot reserve result bus)
                            issued = false;
                        }
                    }
                }
            }
        }
    }

    pub fn cycle(&mut self) {
        println!(
            "{}",
            style(format!(
                "cycle {:03} core {:?}: core cycle",
                self.inner.cycle.get(),
                self.id()
            ))
            .blue()
        );

        if !self.is_active() && self.not_completed() == 0 {
            panic!("core done");
            return;
        }
        // m_stats->shader_cycles[m_sid]++;
        self.writeback();
        self.execute();
        // self.read_operands();
        for _ in 0..self.inner.config.reg_file_port_throughput {
            self.inner
                .operand_collector
                .try_borrow_mut()
                .unwrap()
                .step();
        }

        self.issue();
        for i in 0..self.inner.config.inst_fetch_throughput {
            self.decode();
            self.fetch();
        }
    }

    pub fn cache_flush(&mut self) {
        let mut unit = self.inner.load_store_unit.try_lock().unwrap();
        unit.flush();
    }

    pub fn cache_invalidate(&mut self) {
        let mut unit = self.inner.load_store_unit.try_lock().unwrap();
        unit.invalidate();
    }

    pub fn ldst_unit_response_buffer_full(&self) -> bool {
        self.inner
            .load_store_unit
            .lock()
            .unwrap()
            .response_buffer_full()
    }

    pub fn fetch_unit_response_buffer_full(&self) -> bool {
        false
    }

    pub fn accept_fetch_response(&mut self, mut fetch: mem_fetch::MemFetch) {
        fetch.status = mem_fetch::Status::IN_SHADER_FETCHED;
        self.inner.instr_l1_cache.fill(&mut fetch);
    }

    pub fn accept_ldst_unit_response(&self, fetch: mem_fetch::MemFetch) {
        // todo!("core: accept_ldst_unit_response");
        self.inner.load_store_unit.lock().unwrap().fill(fetch);
    }

    pub fn not_completed(&self) -> usize {
        self.inner.num_active_threads
        // todo!("core: not completed");
    }

    pub fn is_active(&self) -> bool {
        self.inner.num_active_blocks > 0
    }

    pub fn set_kernel(&mut self, kernel: Arc<KernelInfo>) {
        println!(
            "kernel {} ({}) bind to core {:?}",
            kernel.uid,
            kernel.name(),
            self.id()
        );
        self.inner.current_kernel = Some(kernel);
    }

    pub fn find_available_hw_thread_id(
        &mut self,
        thread_block_size: usize,
        occupy: bool,
    ) -> Option<usize> {
        let mut step = 0;
        while step < self.inner.config.max_threads_per_core {
            let mut hw_thread_id = step;
            while hw_thread_id < step + thread_block_size {
                if self.inner.occupied_hw_thread_ids[hw_thread_id] {
                    break;
                }
            }
            // consecutive non-active
            if hw_thread_id == step + thread_block_size {
                break;
            }
            step += thread_block_size;
        }
        if step >= self.inner.config.max_threads_per_core {
            // didn't find
            None
        } else {
            if occupy {
                for hw_thread_id in step..step + thread_block_size {
                    self.inner.occupied_hw_thread_ids.set(hw_thread_id, true);
                }
            }
            Some(step)
        }
    }
    //     int shader_core_ctx::find_available_hwtid(unsigned int cta_size, bool occupy) {
    //   unsigned int step;
    //   for (step = 0; step < m_config->n_thread_per_shader; step += cta_size) {
    //     unsigned int hw_tid;
    //     for (hw_tid = step; hw_tid < step + cta_size; hw_tid++) {
    //       if (m_occupied_hwtid.test(hw_tid)) break;
    //     }
    //     if (hw_tid == step + cta_size)  // consecutive non-active
    //       break;
    //   }
    //   if (step >= m_config->n_thread_per_shader)  // didn't find
    //     return -1;
    //   else {
    //     if (occupy) {
    //       for (unsigned hw_tid = step; hw_tid < step + cta_size; hw_tid++)
    //         m_occupied_hwtid.set(hw_tid);
    //     }
    //     return step;
    //   }
    // }

    pub fn occupy_resource_for_block(&mut self, kernel: &KernelInfo, occupy: bool) -> bool {
        let thread_block_size = self.inner.config.threads_per_block_padded(kernel);
        if self.inner.num_occupied_threads + thread_block_size
            > self.inner.config.max_threads_per_core
        {
            return false;
        }
        if self
            .find_available_hw_thread_id(thread_block_size, false)
            .is_none()
        {
            return false;
        }
        todo!();
        return true;
    }
    //     bool shader_core_ctx::occupy_shader_resource_1block(kernel_info_t &k,
    //                                                     bool occupy) {
    //   unsigned threads_per_cta = k.threads_per_cta();
    //   const class function_info *kernel = k.entry();
    //   unsigned int padded_cta_size = threads_per_cta;
    //   unsigned int warp_size = m_config->warp_size;
    //   if (padded_cta_size % warp_size)
    //     padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);
    //
    //   if (m_occupied_n_threads + padded_cta_size > m_config->n_thread_per_shader)
    //     return false;
    //
    //   if (find_available_hwtid(padded_cta_size, false) == -1) return false;
    //
    //   const struct gpgpu_ptx_sim_info *kernel_info = ptx_sim_kernel_info(kernel);
    //
    //   if (m_occupied_shmem + kernel_info->smem > m_config->gpgpu_shmem_size)
    //     return false;
    //
    //   unsigned int used_regs = padded_cta_size * ((kernel_info->regs + 3) & ~3);
    //   if (m_occupied_regs + used_regs > m_config->gpgpu_shader_registers)
    //     return false;
    //
    //   if (m_occupied_ctas + 1 > m_config->max_cta_per_core) return false;
    //
    //   if (occupy) {
    //     m_occupied_n_threads += padded_cta_size;
    //     m_occupied_shmem += kernel_info->smem;
    //     m_occupied_regs += (padded_cta_size * ((kernel_info->regs + 3) & ~3));
    //     m_occupied_ctas++;
    //
    //     SHADER_DPRINTF(LIVENESS,
    //                    "GPGPU-Sim uArch: Occupied %u threads, %u shared mem, %u "
    //                    "registers, %u ctas, on shader %d\n",
    //                    m_occupied_n_threads, m_occupied_shmem, m_occupied_regs,
    //                    m_occupied_ctas, m_sid);
    //   }
    //
    //   return true;
    // }

    pub fn can_issue_block(&mut self, kernel: &KernelInfo) -> bool {
        let max_blocks = self.inner.config.max_blocks(kernel).unwrap();
        if self.inner.config.concurrent_kernel_sm {
            if max_blocks < 1 {
                return false;
            }
            self.occupy_resource_for_block(kernel, false)
        } else {
            self.inner.num_active_blocks < max_blocks
        }
    }

    /// m_not_completed
    // pub fn active_warps(&self) -> usize {
    //     0
    // }

    fn set_max_blocks(&mut self, kernel: &KernelInfo) -> eyre::Result<()> {
        // calculate the max cta count and cta size for local memory address mapping
        self.inner.max_blocks_per_shader = self.inner.config.max_blocks(kernel)?;
        self.inner.thread_block_size = self.inner.config.threads_per_block_padded(kernel);
        Ok(())
    }

    pub fn id(&self) -> (usize, usize) {
        (self.inner.cluster_id, self.inner.core_id)
    }

    pub fn init_warps_from_traces(
        &mut self,
        kernel: &KernelInfo,
        start_thread: usize,
        end_thread: usize,
    ) {
        let start_warp = start_thread / self.inner.config.warp_size;
        let end_warp = (end_thread / self.inner.config.warp_size)
            + if end_thread % self.inner.config.warp_size != 0 {
                1
            } else {
                0
            };

        debug_assert!(!self.inner.warps.is_empty());
        kernel.next_threadblock_traces(&mut self.inner.warps);
    }

    pub fn init_warps(
        &mut self,
        block_hw_id: usize,
        start_thread: usize,
        end_thread: usize,
        block_id: u64,
        thread_block_size: usize,
        kernel: Arc<KernelInfo>,
    ) {
        println!("core {:?}: init warps for block {}", self.id(), &block_id);
        println!("kernel: {}", &kernel);

        // shader_core_ctx::init_warps
        let start_pc = self.next_pc(start_thread);
        let kernel_id = kernel.uid;
        // if self.config.model == POST_DOMINATOR {
        let start_warp = start_thread / self.inner.config.warp_size;
        let warp_per_cta = thread_block_size / self.inner.config.warp_size;
        let end_warp = end_thread / self.inner.config.warp_size
            + if end_thread % self.inner.config.warp_size == 0 {
                0
            } else {
                1
            };
        for warp_id in start_warp..end_warp {
            let mut num_active = 0;

            let mut local_active_thread_mask: sched::ThreadActiveMask = BitArray::ZERO;
            for warp_thread_id in 0..self.inner.config.warp_size {
                let hwtid = warp_id * self.inner.config.warp_size + warp_thread_id;
                if hwtid < end_thread {
                    num_active += 1;
                    debug_assert!(!self.inner.active_thread_mask[hwtid]);
                    self.inner.active_thread_mask.set(hwtid, true);
                    local_active_thread_mask.set(warp_thread_id, true);
                }
            }
            // self.simt_stack[i].launch(start_pc, self.active_threads);
            // self.inner.warps.insert(
            //     warp_id,
            //     // Some(SchedulerWarp {
            //     // SchedulerWarp {
            //     Arc::new(Mutex::new(SchedulerWarp {
            //         next_pc: start_pc,
            //         warp_id,
            //         block_id,
            //         dynamic_warp_id: self.inner.dynamic_warp_id,
            //         active_mask: local_active_thread_mask,
            //         ..SchedulerWarp::default()
            //     })),
            // );
            self.inner.warps[warp_id].try_borrow_mut().unwrap().init(
                start_pc,
                block_id,
                warp_id,
                self.inner.dynamic_warp_id,
                local_active_thread_mask,
                kernel.clone(),
            );

            self.inner.dynamic_warp_id += 1;
            self.inner.num_active_warps += 1;
            self.inner.num_active_threads += num_active;
        }

        println!("initialized {} warps", &self.inner.warps.len());
        self.init_warps_from_traces(&kernel, start_thread, end_thread);
    }

    pub fn reinit(&mut self, start_thread: usize, end_thread: usize, reset_not_completed: bool) {
        if reset_not_completed {
            self.inner.num_active_threads = 0;
            self.inner.active_thread_mask.fill(false);

            // Jin: for concurrent kernels on a SM
            // m_occupied_n_threads = 0;
            // m_occupied_shmem = 0;
            // m_occupied_regs = 0;
            // m_occupied_ctas = 0;
            // m_occupied_hwtid.reset();
            // m_occupied_cta_to_hwtid.clear();
            self.inner.num_active_warps = 0;
        }
        for t in start_thread..end_thread {
            self.inner.thread_state[t] = None;
        }
        let warp_size = self.inner.config.warp_size;

        for w in (start_thread / warp_size)..(end_thread / warp_size) {
            // println!("warp = {}/{}", w, self.inner.warps.len());
            self.inner.warps[w].try_borrow_mut().unwrap().reset();
            // simt_stack[i]->reset();
        }
    }

    pub fn issue_block(&mut self, kernel: Arc<KernelInfo>) -> () {
        println!(
            "core {:?}: issue one block from kernel {} ({})",
            self.id(),
            kernel.uid,
            kernel.name()
        );
        if !self.inner.config.concurrent_kernel_sm {
            self.set_max_blocks(&*kernel);
        } else {
            let num = self.occupy_resource_for_block(&*kernel, true);
            assert!(num);
        }

        // kernel.inc_running();

        // find a free CTA context
        let max_blocks_per_core = if self.inner.config.concurrent_kernel_sm {
            self.inner.max_blocks_per_shader
        } else {
            self.inner.config.max_concurrent_blocks_per_core
        };
        let free_block_hw_id = (0..max_blocks_per_core)
            .filter(|i| self.inner.block_status[*i] == 0)
            .next()
            .unwrap();

        // determine hardware threads and warps that will be used for this block
        let thread_block_size = kernel.threads_per_block();
        let padded_thread_block_size = self.inner.config.threads_per_block_padded(&*kernel);

        // hw warp id = hw thread id mod warp size, so we need to find a range
        // of hardware thread ids corresponding to an integral number of hardware
        // thread ids
        let (start_thread, end_thread) = if !self.inner.config.concurrent_kernel_sm {
            let start_thread = free_block_hw_id * padded_thread_block_size;
            let end_thread = start_thread + thread_block_size;
            (start_thread, end_thread)
        } else {
            let start_thread = self
                .find_available_hw_thread_id(padded_thread_block_size, true)
                .unwrap();
            let end_thread = start_thread + thread_block_size;

            assert!(!self
                .inner
                .occupied_block_to_hw_thread_id
                .contains_key(&free_block_hw_id));
            self.inner
                .occupied_block_to_hw_thread_id
                .insert(free_block_hw_id, start_thread);
            (start_thread, end_thread)
        };

        // reset state of the selected hardware thread and warp contexts
        // panic!("reinit {:?}", (start_thread, end_thread));
        self.reinit(start_thread, end_thread, false);
        debug_assert!(self
            .inner
            .warps
            .iter()
            .all(|w| w.try_borrow().unwrap().done_exit()));

        // initalize scalar threads and determine which hardware warps they are
        // allocated to bind functional simulation state of threads to hardware
        // resources (simulation)
        let mut warps: WarpMask = BitArray::ZERO;
        // let block_id = kernel.next_block_id();
        let Some(block) = kernel.current_block() else {
            panic!("kernel has no block");

        };
        let mut num_threads_in_block = 0;
        for i in start_thread..end_thread {
            self.inner.thread_state[i] = Some(ThreadState {
                block_id: free_block_hw_id,
                active: true,
                pc: 0, // todo
            });
            let warp_id = i / self.inner.config.warp_size;
            if !kernel.no_more_blocks_to_run() {
                if !kernel.more_threads_in_block() {
                    kernel.next_thread_iter.lock().unwrap().next();
                }

                // we just incremented the thread id so this is not the same
                if !kernel.more_threads_in_block() {
                    kernel.next_block_iter.lock().unwrap().next();
                    *kernel.next_thread_iter.lock().unwrap() =
                        kernel.config.block.into_iter().peekable();
                }
                num_threads_in_block += 1;
            }

            warps.set(warp_id, true);
        }

        self.inner.block_status[free_block_hw_id] = num_threads_in_block;
        self.init_warps(
            free_block_hw_id,
            start_thread,
            end_thread,
            block.id(),
            kernel.threads_per_block(),
            kernel,
        );
        self.inner.num_active_blocks += 1;

        //
        //   warp_set_t warps;
        //   unsigned nthreads_in_block = 0;
        //   function_info *kernel_func_info = kernel.entry();
        //   symbol_table *symtab = kernel_func_info->get_symtab();
        //   unsigned ctaid = kernel.get_next_cta_id_single();
        //   checkpoint *g_checkpoint = new checkpoint();
        //   for (unsigned i = start_thread; i < end_thread; i++) {
        //     m_threadState[i].m_cta_id = free_cta_hw_id;
        //     unsigned warp_id = i / m_config->warp_size;
        //     nthreads_in_block += sim_init_thread(
        //         kernel, &m_thread[i], m_sid, i, cta_size - (i - start_thread),
        //         m_config->n_thread_per_shader, this, free_cta_hw_id, warp_id,
        //         m_cluster->get_gpu());
        //     m_threadState[i].m_active = true;
        //     // load thread local memory and register file
        //     if (m_gpu->resume_option == 1 && kernel.get_uid() == m_gpu->resume_kernel &&
        //         ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
        //       char fname[2048];
        //       snprintf(fname, 2048, "checkpoint_files/thread_%d_%d_reg.txt",
        //                i % cta_size, ctaid);
        //       m_thread[i]->resume_reg_thread(fname, symtab);
        //       char f1name[2048];
        //       snprintf(f1name, 2048, "checkpoint_files/local_mem_thread_%d_%d_reg.txt",
        //                i % cta_size, ctaid);
        //       g_checkpoint->load_global_mem(m_thread[i]->m_local_mem, f1name);
        //     }
        //     //
        //     warps.set(warp_id);
        //   }
        //   assert(nthreads_in_block > 0 &&
        //          nthreads_in_block <=
        //              m_config->n_thread_per_shader);  // should be at least one, but
        //                                               // less than max
        //   m_cta_status[free_cta_hw_id] = nthreads_in_block;
        //
        //   if (m_gpu->resume_option == 1 && kernel.get_uid() == m_gpu->resume_kernel &&
        //       ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
        //     char f1name[2048];
        //     snprintf(f1name, 2048, "checkpoint_files/shared_mem_%d.txt", ctaid);
        //
        //     g_checkpoint->load_global_mem(m_thread[start_thread]->m_shared_mem, f1name);
        //   }
        //   // now that we know which warps are used in this CTA, we can allocate
        //   // resources for use in CTA-wide barrier operations
        //   m_barriers.allocate_barrier(free_cta_hw_id, warps);
        //
        //   // initialize the SIMT stacks and fetch hardware
        //   init_warps(free_cta_hw_id, start_thread, end_thread, ctaid, cta_size, kernel);
        //   m_n_active_cta++;
        //
        //   shader_CTA_count_log(m_sid, 1);
        //   SHADER_DPRINTF(LIVENESS,
        //                  "GPGPU-Sim uArch: cta:%2u, start_tid:%4u, end_tid:%4u, "
        //                  "initialized @(%lld,%lld), kernel_uid:%u, kernel_name:%s\n",
        //                  free_cta_hw_id, start_thread, end_thread, m_gpu->gpu_sim_cycle,
        //                  m_gpu->gpu_tot_sim_cycle, kernel.get_uid(), kernel.get_name().c_str());
        // }
    }
    //
    //   warp_set_t warps;
    //   unsigned nthreads_in_block = 0;
    //   function_info *kernel_func_info = kernel.entry();
    //   symbol_table *symtab = kernel_func_info->get_symtab();
    //   unsigned ctaid = kernel.get_next_cta_id_single();
    //   checkpoint *g_checkpoint = new checkpoint();
    //   for (unsigned i = start_thread; i < end_thread; i++) {
    //     m_threadState[i].m_cta_id = free_cta_hw_id;
    //     unsigned warp_id = i / m_config->warp_size;
    //     nthreads_in_block += sim_init_thread(
    //         kernel, &m_thread[i], m_sid, i, cta_size - (i - start_thread),
    //         m_config->n_thread_per_shader, this, free_cta_hw_id, warp_id,
    //         m_cluster->get_gpu());
    //     m_threadState[i].m_active = true;
    //     // load thread local memory and register file
    //     if (m_gpu->resume_option == 1 && kernel.get_uid() == m_gpu->resume_kernel &&
    //         ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
    //       char fname[2048];
    //       snprintf(fname, 2048, "checkpoint_files/thread_%d_%d_reg.txt",
    //                i % cta_size, ctaid);
    //       m_thread[i]->resume_reg_thread(fname, symtab);
    //       char f1name[2048];
    //       snprintf(f1name, 2048, "checkpoint_files/local_mem_thread_%d_%d_reg.txt",
    //                i % cta_size, ctaid);
    //       g_checkpoint->load_global_mem(m_thread[i]->m_local_mem, f1name);
    //     }
    //     //
    //     warps.set(warp_id);
    //   }
    //   assert(nthreads_in_block > 0 &&
    //          nthreads_in_block <=
    //              m_config->n_thread_per_shader);  // should be at least one, but
    //                                               // less than max
    //   m_cta_status[free_cta_hw_id] = nthreads_in_block;
    //
    //   if (m_gpu->resume_option == 1 && kernel.get_uid() == m_gpu->resume_kernel &&
    //       ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
    //     char f1name[2048];
    //     snprintf(f1name, 2048, "checkpoint_files/shared_mem_%d.txt", ctaid);
    //
    //     g_checkpoint->load_global_mem(m_thread[start_thread]->m_shared_mem, f1name);
    //   }
    //   // now that we know which warps are used in this CTA, we can allocate
    //   // resources for use in CTA-wide barrier operations
    //   m_barriers.allocate_barrier(free_cta_hw_id, warps);
    //
    //   // initialize the SIMT stacks and fetch hardware
    //   init_warps(free_cta_hw_id, start_thread, end_thread, ctaid, cta_size, kernel);
    //   m_n_active_cta++;
    //
    //   shader_CTA_count_log(m_sid, 1);
    //   SHADER_DPRINTF(LIVENESS,
    //                  "GPGPU-Sim uArch: cta:%2u, start_tid:%4u, end_tid:%4u, "
    //                  "initialized @(%lld,%lld), kernel_uid:%u, kernel_name:%s\n",
    //                  free_cta_hw_id, start_thread, end_thread, m_gpu->gpu_sim_cycle,
    //                  m_gpu->gpu_tot_sim_cycle, kernel.get_uid(), kernel.get_name().c_str());
    // }
}

// pub trait CoreMemoryInterface {
//   fn full(&self, size: usize, write: bool) -> bool;
//   fn push(&mut self, size: usize, write: bool) -> bool;
// }
