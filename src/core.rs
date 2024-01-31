use super::{
    address, barrier, cache, config, func_unit as fu,
    instruction::WarpInstruction,
    interconn as ic,
    kernel::Kernel,
    mem_fetch, opcodes,
    operand_collector::{self, OperandCollector, RegisterFileUnit},
    register_set, scheduler, scoreboard, warp,
};
use crate::func_unit::SimdFunctionUnit;
use crate::sync::Mutex;

use barrier::Barrier;
use bitvec::{array::BitArray, BitArr};
use color_eyre::eyre;
use console::style;
use crossbeam::utils::CachePadded;
use mem_fetch::access::Kind as AccessKind;
use register_set::Access as RegisterSetAccess;
use std::collections::{HashMap, VecDeque};
use std::sync::{atomic, Arc};
use strum::IntoEnumIterator;
use trace_model::ToBitString;

pub type WarpMask = BitArr!(for crate::MAX_WARPS_PER_CTA);

#[derive(Debug)]
pub struct ThreadState {
    pub active: bool,
    pub pc: usize,
}

#[derive(Debug, Default)]
pub struct InstrFetchBuffer {
    valid: bool,
    warp_id: usize,
}

impl InstrFetchBuffer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            valid: false,
            warp_id: 0,
        }
    }
}

type ResultBus = BitArr!(for fu::MAX_ALU_LATENCY);

pub trait WarpIssuer {
    fn issue_warp(
        &mut self,
        stage: PipelineStage,
        warp: &mut warp::Warp,
        scheduler_id: usize,
        cycle: u64,
    ) -> eyre::Result<()>;

    fn has_free_register(&self, stage: PipelineStage, scheduler_id: usize) -> bool;

    fn has_collision(&self, warp_id: usize, instr: &WarpInstruction) -> bool;

    #[must_use]
    fn warp_waiting_at_barrier(&self, warp_id: usize) -> bool;

    #[must_use]
    fn warp_waiting_at_mem_barrier(&self, warp_id: &warp::Warp) -> bool;
}

#[derive()]
pub struct CoreIssuer<'a> {
    pub config: &'a config::GPU,
    pub pipeline_reg: &'a mut [register_set::RegisterSet],
    pub warp_instruction_unique_uid: &'a CachePadded<atomic::AtomicU64>,
    pub allocations: &'a super::allocation::Ref,
    pub core_id: usize,
    pub thread_block_size: usize,
    pub current_kernel_max_blocks: usize,
    pub scoreboard: &'a mut dyn scoreboard::Access<WarpInstruction>,
    pub barriers: &'a mut barrier::BarrierSet,
}

impl<'a> std::fmt::Debug for CoreIssuer<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoreIssuer").finish()
    }
}

impl<'a> WarpIssuer for CoreIssuer<'a> {
    fn has_free_register(&self, stage: PipelineStage, scheduler_id: usize) -> bool {
        // locking here blocks when we run schedulers in parallel
        let pipeline_stage = &self.pipeline_reg[stage as usize];

        if self.config.sub_core_model {
            pipeline_stage
                .get(scheduler_id)
                .map(Option::as_ref)
                .flatten()
                .is_none()
        } else {
            pipeline_stage.has_free()
        }
    }

    fn has_collision(&self, warp_id: usize, instr: &WarpInstruction) -> bool {
        self.scoreboard.has_collision(warp_id, instr)
    }

    #[tracing::instrument(name = "core_issue_warp")]
    fn issue_warp(
        &mut self,
        stage: PipelineStage,
        warp: &mut warp::Warp,
        scheduler_id: usize,
        cycle: u64,
    ) -> eyre::Result<()> {
        let pipeline_stage = &mut self.pipeline_reg[stage as usize];
        let pipeline_stage_copy = pipeline_stage.clone();
        let free = if self.config.sub_core_model {
            pipeline_stage.get_free_sub_core_mut(scheduler_id)
        } else {
            pipeline_stage.get_free_mut()
        };
        let (reg_idx, pipe_reg) = free.ok_or(eyre::eyre!("no free register"))?;

        let mut next_instr = warp.ibuffer_take().unwrap();
        warp.ibuffer_step();

        log::debug!(
            "{} by scheduler {} to pipeline[{:?}][{}] {}",
            style(format!(
                "cycle {:02} issue {} for warp {}",
                cycle, next_instr, warp.warp_id
            ))
            .yellow(),
            scheduler_id,
            stage,
            reg_idx,
            pipeline_stage_copy,
        );

        // this sets all the info for the warp instruction in pipe reg
        next_instr.uid = self
            .warp_instruction_unique_uid
            .fetch_add(1, atomic::Ordering::SeqCst);

        next_instr.warp_id = warp.warp_id;
        next_instr.issue_cycle = Some(cycle);
        next_instr.dispatch_delay_cycles = next_instr.initiation_interval;
        next_instr.scheduler_id = Some(scheduler_id);

        let mut pipe_reg_mut = next_instr;

        debug_assert_eq!(warp.warp_id, pipe_reg_mut.warp_id);

        for t in 0..self.config.warp_size {
            if pipe_reg_mut.active_mask[t] {
                let warp_id = pipe_reg_mut.warp_id;
                let thread_id = self.config.warp_size * warp_id + t;

                if pipe_reg_mut.is_atomic() {
                    todo!("atomics");
                    // warp.inc_n_atomic();
                }

                if pipe_reg_mut.memory_space == Some(super::instruction::MemorySpace::Local)
                    && (pipe_reg_mut.is_load() || pipe_reg_mut.is_store())
                {
                    let total_cores =
                        self.config.num_simt_clusters * self.config.num_cores_per_simt_cluster;
                    let translated_local_addresses = translate_local_memaddr(
                        pipe_reg_mut.threads[t].mem_req_addr[0],
                        self.core_id,
                        thread_id,
                        total_cores,
                        pipe_reg_mut.data_size,
                        self.thread_block_size,
                        self.current_kernel_max_blocks,
                        &self.config,
                    );

                    debug_assert!(
                        translated_local_addresses.len()
                            < super::instruction::MAX_ACCESSES_PER_THREAD_INSTRUCTION
                    );
                    pipe_reg_mut.set_addresses(t, translated_local_addresses);
                }

                if pipe_reg_mut.opcode.category == opcodes::ArchOp::EXIT_OPS {
                    warp.set_thread_completed(t);
                }
            }
        }

        // here, we generate memory acessess
        if pipe_reg_mut.is_load() || pipe_reg_mut.is_store() {
            if let Some(accesses) =
                pipe_reg_mut.generate_mem_accesses(&self.config, &self.allocations)
            {
                for access in accesses {
                    if let AccessKind::LOCAL_ACC_W | AccessKind::LOCAL_ACC_R = access.kind {
                        panic!("have local access!");
                    }
                    if matches!(
                        access.kind,
                        AccessKind::L1_WRBK_ACC
                            | AccessKind::L2_WRBK_ACC
                            | AccessKind::INST_ACC_R
                            | AccessKind::L1_WR_ALLOC_R
                            | AccessKind::L2_WR_ALLOC_R
                    ) {
                        panic!(
                            "generated {:?} access from instruction {}",
                            &access.kind, &pipe_reg_mut
                        );
                    }

                    log::trace!(
                        "generate_mem_accesses: adding access {} to instruction {}",
                        &access,
                        &pipe_reg_mut
                    );
                    pipe_reg_mut.mem_access_queue.push_back(access);
                }
            }

            log::trace!(
                "generated mem accesses: {:?}",
                pipe_reg_mut
                    .mem_access_queue
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
            );
        }

        let pipe_reg_ref = pipe_reg_mut;

        log::debug!(
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
            self.barriers.warp_exited(pipe_reg_ref.warp_id);
        }

        if pipe_reg_ref.opcode.category == opcodes::ArchOp::BARRIER_OP {
            self.barriers
                .warp_reached_barrier(warp.block_id, &pipe_reg_ref);
        } else if pipe_reg_ref.opcode.category == opcodes::ArchOp::MEMORY_BARRIER_OP {
            warp.waiting_for_memory_barrier = true;
        }

        log::debug!(
            "{} ({:?}) for instr {}",
            style(format!(
                "reserving {} registers",
                pipe_reg_ref.outputs().count()
            ))
            .yellow(),
            pipe_reg_ref.outputs().collect::<Vec<_>>(),
            pipe_reg_ref
        );

        self.scoreboard.reserve_all(&pipe_reg_ref);

        *pipe_reg = Some(pipe_reg_ref);

        // log::debug!(
        //     "post issue register set of {:?} pipeline: {}",
        //     stage,
        //     pipeline_stage
        // );
        Ok(())
    }

    #[must_use]
    fn warp_waiting_at_barrier(&self, warp_id: usize) -> bool {
        self.barriers.is_waiting_at_barrier(warp_id)
    }

    #[must_use]
    fn warp_waiting_at_mem_barrier(&self, warp: &warp::Warp) -> bool {
        if !warp.waiting_for_memory_barrier {
            return false;
        }
        let has_pending_writes = !self.scoreboard.pending_writes(warp.warp_id).is_empty();

        // warp.waiting_for_memory_barrier = has_pending_writes;
        has_pending_writes

        // TODO: PERF roman this requires mutable access but should be fast
        // if has_pending_writes {
        //     true
        // } else {
        //     warp.waiting_for_memory_barrier = false;
        //     if self.config.flush_l1_cache {
        //         // Mahmoud fixed this on Nov 2019
        //         // Invalidate L1 cache
        //         // Based on Nvidia Doc, at MEM barrier, we have to
        //         //(1) wait for all pending writes till they are acked
        //         //(2) invalidate L1 cache to ensure coherence and avoid reading stall data
        //         *self.need_l1_flush.lock() = true;
        //         // todo!("cache invalidate");
        //         // self.cache_invalidate();
        //         // TO DO: you need to stall the SM for 5k cycles.
        //     }
        //     false
        // }
    }
}

#[derive(strum::EnumIter, strum::EnumCount, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum PipelineStage {
    /// Instruction Decode -> Operand Collector stage for single precision unit
    ID_OC_SP = 0,
    /// Instruction Decode -> Operand Collector stage for double precision unit
    ID_OC_DP = 1,
    /// Instruction Decode -> Operand Collector stage for integer unit
    ID_OC_INT = 2,
    /// Instruction Decode -> Operand Collector stage for special function unit
    ID_OC_SFU = 3,
    /// Instruction Decode -> Operand Collector stage for load store unit
    ID_OC_MEM = 4,
    /// Operand Collector -> Execution stage for single precision unit
    OC_EX_SP = 5,
    /// Operand Collector -> Execution stage for double precision unit
    OC_EX_DP = 6,
    /// Operand Collector -> Execution stage for integer precision unit
    OC_EX_INT = 7,
    /// Operand Collector -> Execution stage for special function unit
    OC_EX_SFU = 8,
    /// Operand Collector -> Execution stage for load store unit
    OC_EX_MEM = 9,
    /// Execution -> Writeback stage
    EX_WB = 10,
    /// Instruction Decode -> Operand Collector stage for tensor unit
    ID_OC_TENSOR_CORE = 11,
    /// Operand Collector -> Execution stage for tensor unit
    OC_EX_TENSOR_CORE = 12,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FetchResponseTarget {
    LoadStoreUnit,
    ICache,
}

type InterconnBuffer<T> = VecDeque<ic::Packet<(usize, T, u32)>>;

#[allow(clippy::module_name_repetitions)]
pub struct CoreMemoryConnection<C> {
    pub config: Arc<config::GPU>,
    pub stats: stats::PerKernel,
    pub cluster_id: usize,
    pub buffer: C,
}

#[allow(clippy::missing_fields_in_debug)]
impl<C> std::fmt::Debug for CoreMemoryConnection<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoreMemoryConnection").finish()
    }
}

impl<C> ic::Connection<ic::Packet<mem_fetch::MemFetch>> for CoreMemoryConnection<C>
where
    C: ic::BufferedConnection<ic::Packet<(usize, mem_fetch::MemFetch, u32)>>,
{
    // #[inline]
    fn can_send(&self, packets: &[u32]) -> bool {
        // let request_size: u32 = packets
        //     .iter()
        //     .map(|fetch| {
        //         if fetch.is_write() {
        //             fetch.size()
        //         } else {
        //             u32::from(mem_fetch::READ_PACKET_SIZE)
        //         }
        //     })
        //     .sum();
        // true
        self.buffer.can_send(packets)
        // self.interconn.has_buffer(self.cluster_id, request_size)
    }

    // #[inline]
    fn send(&mut self, packet: ic::Packet<mem_fetch::MemFetch>) {
        // self.core.interconn_simt_to_mem(fetch.get_num_flits(true));
        // self.cluster.interconn_inject_request_packet(fetch);

        let ic::Packet { data, time } = packet;
        let mut fetch = data;

        let access_kind = fetch.access_kind();
        debug_assert_eq!(fetch.is_write(), access_kind.is_write());
        let kernel_stats = self.stats.get_mut(fetch.kernel_launch_id());
        kernel_stats
            .accesses
            .inc(fetch.allocation_id(), access_kind, 1);

        let dest_sub_partition_id = fetch.sub_partition_id();
        let mem_dest = self.config.mem_id_to_device_id(dest_sub_partition_id);

        log::debug!(
            "cluster {} icnt_inject_request_packet({}) dest sub partition id={} dest mem node={}",
            self.cluster_id,
            fetch,
            dest_sub_partition_id,
            mem_dest
        );

        // The packet size varies depending on the type of request:
        // - For write request and atomic request, packet contains the data
        // - For read request (i.e. not write nor atomic), packet only has control metadata
        let packet_size = if fetch.is_write() || fetch.is_atomic() {
            fetch.size()
        } else {
            fetch.control_size()
        };
        // m_stats->m_outgoing_traffic_stats->record_traffic(mf, packet_size);
        fetch.status = mem_fetch::Status::IN_ICNT_TO_MEM;

        // if let Packet::Fetch(fetch) = packet {
        fetch.inject_cycle.get_or_insert(time);

        // self.interconn_queue
        //     .push_back((mem_dest, fetch, packet_size));
        // self.interconn
        //     .push(self.cluster_id, mem_dest, Packet::Fetch(fetch), packet_size);
        // self.interconn_port
        //     .lock()
        //     .push_back((mem_dest, fetch, packet_size));
        self.buffer.send(ic::Packet {
            data: (mem_dest, fetch, packet_size),
            time,
        });
    }
}

/// SIMT Core.
#[derive()]
pub struct Core<I, MC> {
    pub core_id: usize,
    pub cluster_id: usize,
    pub warp_instruction_unique_uid: Arc<CachePadded<atomic::AtomicU64>>,

    /// Core statistics per kernel.
    ///
    /// Stats are private so that consumers use the stats() method,
    /// which aggregates stats for the functional units and mem port.
    stats: stats::PerKernel,

    pub config: Arc<config::GPU>,
    pub mem_controller: Arc<MC>,
    pub current_kernel: Mutex<Option<Arc<dyn Kernel>>>,
    pub current_kernel_max_blocks: usize,
    pub last_warp_fetched: Option<usize>,
    pub interconn: Arc<I>,
    // check this again
    pub mem_port: Arc<Mutex<CoreMemoryConnection<InterconnBuffer<mem_fetch::MemFetch>>>>,
    pub load_store_unit: fu::LoadStoreUnit<MC>,
    pub active_thread_mask: BitArr!(for crate::MAX_THREADS_PER_SM),
    pub occupied_hw_thread_ids: BitArr!(for crate::MAX_THREADS_PER_SM),
    pub dynamic_warp_id: usize,
    pub num_active_blocks: usize,
    pub num_active_warps: usize,
    pub num_active_threads: usize,
    pub num_occupied_threads: usize,

    pub thread_block_size: usize,
    pub occupied_block_to_hw_thread_id: HashMap<usize, usize>,
    pub block_status: Box<[usize]>,

    // allocations are managed by the driver API, hence we need mutex.
    pub allocations: super::allocation::Ref,
    pub instr_l1_cache: Box<dyn cache::Cache<stats::cache::PerKernel>>,
    pub please_fill: Mutex<Vec<(FetchResponseTarget, mem_fetch::MemFetch, u64)>>,
    pub need_l1_flush: Mutex<bool>,
    pub instr_fetch_buffer: InstrFetchBuffer,
    pub warps: Vec<warp::Warp>,
    pub thread_state: Vec<Option<ThreadState>>,
    pub scoreboard: Box<dyn scoreboard::Access<WarpInstruction>>,
    pub barriers: barrier::BarrierSet,
    pub register_file: RegisterFileUnit,
    pub pipeline_reg: Vec<register_set::RegisterSet>,
    pub result_busses: Vec<ResultBus>,
    pub functional_units: Vec<Box<dyn fu::SimdFunctionUnit>>,
    pub schedulers: Vec<Box<dyn scheduler::Scheduler>>,
    pub scheduler_issue_priority: usize,

    /// Custom callback handler that is called when a fetch is returned to its issuer.
    pub fetch_return_callback: Option<Box<dyn Fn(u64, &mem_fetch::MemFetch) + Send + Sync>>,
}

#[allow(clippy::missing_fields_in_debug)]
impl<I, MC> std::fmt::Debug for Core<I, MC> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Core")
            .field("core_id", &self.core_id)
            .field("cluster_id", &self.cluster_id)
            .finish()
    }
}

impl<I, MC> Core<I, MC>
where
    I: ic::Interconnect<ic::Packet<mem_fetch::MemFetch>>,
    MC: crate::mcu::MemoryController,
{
    pub fn new(
        core_id: usize,
        cluster_id: usize,
        allocations: super::allocation::Ref,
        warp_instruction_unique_uid: Arc<CachePadded<atomic::AtomicU64>>,
        interconn: Arc<I>,
        config: Arc<config::GPU>,
        mem_controller: Arc<MC>,
    ) -> Self {
        let thread_state: Vec<_> = (0..config.max_threads_per_core).map(|_| None).collect();

        let stats = stats::PerKernel::new(config.as_ref().into());

        let warps: Vec<_> = (0..config.max_warps_per_core())
            .map(|_| warp::Warp::default())
            .collect();

        let mem_port = Arc::new(Mutex::new(CoreMemoryConnection {
            cluster_id,
            stats: stats.clone(),
            config: Arc::clone(&config),
            buffer: InterconnBuffer::<mem_fetch::MemFetch>::new(),
        }));

        let mut instr_l1_cache = cache::ReadOnly::new(
            core_id,
            format!(
                "core-{}-{}-{}",
                cluster_id,
                core_id,
                style("READONLY-INSTR-CACHE").green(),
            ),
            cache::base::Kind::OnChip,
            // cluster_id,
            // cache_stats,
            // config,
            config.inst_cache_l1.as_ref().unwrap().clone(),
            config.accelsim_compat,
        );
        instr_l1_cache.set_top_port(mem_port.clone());

        let scoreboard = Box::new(scoreboard::Scoreboard::new(&scoreboard::Config {
            core_id,
            cluster_id,
            max_warps: config.max_warps_per_core(),
        }));

        // pipeline_stages is the sum of normal pipeline stages
        // and specialized_unit stages * 2 (for ID and EX)
        // let total_pipeline_stages = PipelineStage::COUNT + config.num_specialized_unit.len() * 2;;
        // let pipeline_reg = (0..total_pipeline_stages)

        let pipeline_reg: Vec<_> = PipelineStage::iter()
            .map(|stage| {
                let pipeline_width = config.pipeline_widths.get(&stage).copied().unwrap_or(0);
                register_set::RegisterSet::new(stage, pipeline_width, stage as usize)
            })
            .collect();

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
        }

        // there are as many result buses as the width of the EX_WB stage
        let result_busses: Vec<_> = (0..pipeline_reg[PipelineStage::EX_WB as usize].size())
            .map(|_| BitArray::ZERO)
            .collect();

        let mut register_file = RegisterFileUnit::new(config.clone());
        // configure generic collectors
        Self::init_operand_collector(&mut register_file, &config);

        let load_store_unit = fu::LoadStoreUnit::new(
            core_id, // is the core id for now
            core_id,
            cluster_id,
            mem_port.clone(),
            config.clone(),
            mem_controller.clone(),
        );

        let scheduler_kind = config::SchedulerKind::GTO;

        let schedulers: Vec<Box<dyn scheduler::Scheduler>> = (0..config.num_schedulers_per_core)
            .map(|sched_id| {
                let scheduler: Box<dyn scheduler::Scheduler> = match scheduler_kind {
                    config::SchedulerKind::GTO => {
                        let gto = scheduler::gto::Scheduler::new(
                            sched_id,
                            cluster_id,
                            core_id,
                            config.clone(),
                        );
                        Box::new(gto)
                    }
                    scheduler_kind => unimplemented!("scheduler: {:?}", &scheduler_kind),
                };
                scheduler
            })
            .collect();

        let mut functional_units: Vec<Box<dyn fu::SimdFunctionUnit>> = Vec::new();

        // single precision units
        for issue_reg_id in 0..config.num_sp_units {
            functional_units.push(Box::new(fu::sp::SPUnit::new(
                issue_reg_id,
                Arc::clone(&config),
                issue_reg_id,
            )));
        }

        // double precision units
        for issue_reg_id in 0..config.num_dp_units {
            functional_units.push(Box::new(fu::DPUnit::new(
                issue_reg_id,
                Arc::clone(&config),
                issue_reg_id,
            )));
        }

        // integer units
        for issue_reg_id in 0..config.num_int_units {
            functional_units.push(Box::new(fu::IntUnit::new(
                issue_reg_id,
                Arc::clone(&config),
                issue_reg_id,
            )));
        }

        // special function units
        for issue_reg_id in 0..config.num_sfu_units {
            functional_units.push(Box::new(fu::SFU::new(
                issue_reg_id,
                Arc::clone(&config),
                issue_reg_id,
            )));
        }

        let barriers = barrier::Builder {
            // config.max_warps_per_core(),
            max_blocks_per_core: config.max_concurrent_blocks_per_core,
            max_barriers_per_block: config.max_barriers_per_block,
            warp_size: config.warp_size,
        }
        .build();

        Self {
            core_id,
            cluster_id,
            warp_instruction_unique_uid,
            stats,
            allocations,
            config,
            mem_controller,
            current_kernel: Mutex::new(None),
            current_kernel_max_blocks: 0,
            last_warp_fetched: None,
            active_thread_mask: BitArray::ZERO,
            occupied_hw_thread_ids: BitArray::ZERO,
            dynamic_warp_id: 0,
            num_active_blocks: 0,
            num_active_warps: 0,
            num_active_threads: 0,
            num_occupied_threads: 0,
            thread_block_size: 0,
            occupied_block_to_hw_thread_id: HashMap::new(),
            // todo: add configuration option for MAX_CTA_PER_SM
            block_status: utils::box_slice![0; 32],
            instr_l1_cache: Box::new(instr_l1_cache),
            please_fill: Mutex::new(Vec::new()),
            need_l1_flush: Mutex::new(false),
            instr_fetch_buffer: InstrFetchBuffer::default(),
            interconn,
            mem_port,
            load_store_unit,
            warps,
            pipeline_reg,
            result_busses,
            scoreboard,
            barriers,
            register_file,
            thread_state,
            schedulers,
            scheduler_issue_priority: 0,
            functional_units,
            fetch_return_callback: None,
        }
    }

    pub fn stats(&self) -> stats::PerKernel {
        let mut stats = self.stats.clone();

        // add l1i stats
        let l1i = self.instr_l1_cache.per_kernel_stats().clone();
        for (kernel_launch_id, cache_stats) in l1i.into_iter().enumerate() {
            let kernel_stats = stats.get_mut(Some(kernel_launch_id));
            kernel_stats.l1i_stats[self.core_id] += cache_stats.clone();
        }

        // add l1d stats
        if let Some(ref l1d) = self.load_store_unit.data_l1 {
            let l1d = l1d.per_kernel_stats().clone();
            for (kernel_launch_id, cache_stats) in l1d.into_iter().enumerate() {
                let kernel_stats = stats.get_mut(Some(kernel_launch_id));
                kernel_stats.l1d_stats[self.core_id] += cache_stats.clone();
            }
        }

        stats += self.mem_port.lock().stats.clone();
        stats
    }

    // #[inline]
    #[tracing::instrument]
    fn fetch(&mut self, cycle: u64) {
        // if log::log_enabled!(log::Level::Debug) {
        //     log::debug!(
        //         "{}",
        //         style(format!(
        //             "cycle {:03} core {:?}: fetch (fetch buffer valid={}, l1i ready={:?})",
        //             cycle,
        //             self.id(),
        //             self.instr_fetch_buffer.valid,
        //             self.instr_l1_cache
        //                 .ready_accesses()
        //                 .cloned()
        //                 .unwrap_or_default()
        //                 .iter()
        //                 .map(std::string::ToString::to_string)
        //                 .collect::<Vec<_>>(),
        //         ))
        //         .green()
        //     );
        // }

        if !self.instr_fetch_buffer.valid {
            if let Some(fetch) = self.instr_l1_cache.next_access() {
                let warp = self.warps.get_mut(fetch.warp_id).unwrap();
                warp.has_imiss_pending = false;

                self.instr_fetch_buffer = InstrFetchBuffer {
                    valid: true,
                    warp_id: fetch.warp_id,
                };

                // verify that we got the instruction we were expecting.
                debug_assert_eq!(
                    warp.pc(),
                    Some(fetch.addr() as usize - super::PROGRAM_MEM_START)
                );

                self.instr_fetch_buffer.valid = true;
                // warp.set_last_fetch(m_gpu->gpu_sim_cycle);
            } else {
                // find an active warp with space in
                // instruction buffer that is not
                // already waiting on a cache miss and get
                // next 1-2 instructions from instruction cache
                let max_warps = self.config.max_warps_per_core();

                // if false {
                //     for warp_id in 0..max_warps {
                //         // let warp = self.warps[warp_id].try_borrow().unwrap();
                //         let warp = self.warps[warp_id].try_lock();
                //         if warp.instruction_count() == 0 {
                //             // consider empty
                //             continue;
                //         }
                //         debug_assert_eq!(warp.warp_id, warp_id);
                //
                //         let sb = self.scoreboard.try_read();
                //         let pending_writes = sb.pending_writes(warp_id);
                //
                //         // if warp.functional_done() && warp.hardware_done() && warp.done_exit() {
                //         //     continue;
                //         // }
                //         log::debug!(
                //         "checking warp_id = {} dyn warp id = {} (instruction count={}, trace pc={} hardware_done={}, functional_done={}, instr in pipe={}, stores={}, done_exit={}, pending writes={:?})",
                //         &warp_id,
                //         warp.dynamic_warp_id(),
                //         warp.instruction_count(),
                //         warp.trace_pc,
                //         warp.hardware_done(),
                //         warp.functional_done(),
                //         warp.num_instr_in_pipeline,
                //         warp.num_outstanding_stores,
                //         warp.done_exit(),
                //         pending_writes.iter().sorted().collect::<Vec<_>>()
                //     );
                //     }
                // }

                for i in 0..max_warps {
                    let last = self.last_warp_fetched.unwrap_or(0);
                    let warp_id = (last + 1 + i) % max_warps;

                    let warp = self.warps.get(warp_id).unwrap();
                    debug_assert!(warp.warp_id == warp_id || warp.warp_id == u32::MAX as usize);

                    let block_hw_id = warp.block_id as usize;
                    debug_assert!(
                        block_hw_id < self.block_status.len(),
                        "block id is the hw block id for this core"
                    );

                    // todo: how expensive are arc clones?
                    let kernel = warp.kernel.as_ref().map(Arc::clone);

                    let has_pending_writes = !self.scoreboard.pending_writes(warp_id).is_empty();

                    let did_maybe_exit =
                        warp.hardware_done() && !has_pending_writes && !warp.done_exit();

                    // check if this warp has finished executing and can be reclaimed.
                    let mut did_exit = false;
                    if did_maybe_exit {
                        log::debug!("\tchecking if warp_id = {} did complete", warp_id);

                        for t in 0..self.config.warp_size {
                            let tid = warp_id * self.config.warp_size + t;
                            if let Some(Some(state)) = self.thread_state.get_mut(tid) {
                                if state.active {
                                    state.active = false;

                                    log::debug!(
                                        "thread {} of block {} completed ({} left)",
                                        tid,
                                        block_hw_id,
                                        self.block_status[block_hw_id]
                                    );
                                    self.register_thread_in_block_exited(block_hw_id, &kernel);

                                    // if let Some(Some(thread_info)) =
                                    //     self.thread_info.get(tid).map(Option::as_ref)
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
                                    self.num_active_threads -= 1;
                                    self.active_thread_mask.set(tid, false);
                                    did_exit = true;
                                }
                            }
                        }
                        self.num_active_warps -= 1;
                    }

                    let warp = self.warps.get_mut(warp_id).unwrap();
                    if did_exit {
                        warp.done_exit = true;
                    }

                    let icache_config = self.config.inst_cache_l1.as_ref().unwrap();
                    // !warp.trace_instructions.is_empty() &&
                    let should_fetch_instruction =
                        !warp.functional_done() && !warp.has_imiss_pending && warp.ibuffer_empty();

                    // this code fetches instructions
                    // from the i-cache or generates memory
                    if should_fetch_instruction {
                        if warp.current_instr().is_none() {
                            // warp.hardware_done() && pending_writes.is_empty() && !warp.done_exit()
                            // dbg!(&warp);
                            // dbg!(&warp.active_mask.to_bit_string());
                            // dbg!(&warp.num_completed());
                            // panic!("?");
                            // skip and do nothing (can happen during nondeterministic parallel)
                            continue;
                        }
                        let instr = warp.current_instr().unwrap();
                        let pc = warp.pc().unwrap();
                        let ppc = pc + crate::PROGRAM_MEM_START;

                        log::debug!(
                            "\t fetching instr {} for warp_id = {} (pc={}, ppc={})",
                            &instr,
                            warp.warp_id,
                            pc,
                            ppc,
                        );

                        let mut num_bytes = 16;
                        let line_size = icache_config.line_size as usize;
                        let offset_in_block = pc & (line_size - 1);
                        if offset_in_block + num_bytes > line_size {
                            num_bytes = line_size - offset_in_block;
                        }
                        let inst_alloc = &*crate::PROGRAM_MEM_ALLOC;
                        let access = mem_fetch::access::Builder {
                            kind: AccessKind::INST_ACC_R,
                            addr: ppc as u64,
                            kernel_launch_id: Some(instr.kernel_launch_id),
                            allocation: Some(inst_alloc.clone()),
                            req_size_bytes: num_bytes as u32,
                            is_write: false,
                            warp_active_mask: instr.active_mask,
                            // warp_active_mask: warp::ActiveMask::all_ones(),
                            // we dont need to set the sector and bit
                            // masks, because they will be computed
                            // when inserted into the memory sub partition
                            byte_mask: !mem_fetch::ByteMask::ZERO,
                            sector_mask: !mem_fetch::SectorMask::ZERO,
                        }
                        .build();

                        let physical_addr = self.mem_controller.to_physical_address(access.addr);

                        let fetch = mem_fetch::Builder {
                            instr: None,
                            access,
                            warp_id,
                            core_id: Some(self.core_id),
                            cluster_id: Some(self.cluster_id),
                            physical_addr,
                        }
                        .build();

                        let status = if self.config.perfect_inst_const_cache {
                            cache::RequestStatus::HIT
                        } else {
                            let mut events = Vec::new();
                            self.instr_l1_cache.access(
                                ppc as address,
                                fetch.clone(),
                                &mut events,
                                cycle,
                            )
                        };

                        // log::warn!("L1I access({}) -> {:?}", fetch, status);
                        self.last_warp_fetched = Some(warp_id);

                        if status == cache::RequestStatus::MISS {
                            warp.has_imiss_pending = true;
                            // warp.set_last_fetch(m_gpu->gpu_sim_cycle);
                        } else if status == cache::RequestStatus::HIT {
                            self.instr_fetch_buffer = InstrFetchBuffer {
                                valid: true,
                                warp_id,
                            };
                            // m_warp[warp_id]->set_last_fetch(m_gpu->gpu_sim_cycle);
                        } else {
                            debug_assert_eq!(status, cache::RequestStatus::RESERVATION_FAIL);
                        }
                        break;
                    }
                }
            }
        }
        if !self.config.perfect_inst_const_cache {
            self.instr_l1_cache.cycle(cycle);
        }
    }

    #[tracing::instrument]
    // #[inline]
    fn execute(&mut self, cycle: u64) {
        let core_id = self.id();
        log::debug!(
            "{}",
            style(format!("cycle {cycle:03} core {core_id:?} execute: ")).red()
        );

        for (_, res_bus) in self.result_busses.iter_mut().enumerate() {
            // note: in rust, shift left is semantically equal to "towards the zero index"
            res_bus.shift_left(1);
            // log::debug!(
            //     "res bus {:03}[:128]: {}",
            //     i,
            //     &res_bus.to_bit_string()[0..128]
            // );
        }

        let functional_units_iter = self
            .functional_units
            .iter_mut()
            .map(|fu| fu.as_mut() as &mut dyn SimdFunctionUnit)
            .chain(std::iter::once(
                &mut self.load_store_unit as &mut dyn SimdFunctionUnit,
            ));

        for fu in functional_units_iter {
            let fu_id = fu.id().to_string();
            let issue_port = fu.issue_port();
            let result_port = fu.result_port();

            log::debug!(
                "fu[{:03}] {:<10} before \t{:?}={}",
                &fu_id,
                fu.to_string(),
                issue_port,
                &self.pipeline_reg[issue_port as usize]
            );

            fu.cycle(
                &mut self.register_file,
                &mut *self.scoreboard,
                &mut self.warps,
                &mut self.stats,
                result_port.map(|port| &mut self.pipeline_reg[port as usize]),
                cycle,
            );
            fu.active_lanes_in_pipeline();

            log::debug!(
                "fu[{:03}] {:<10} after \t{:?}={}",
                &fu_id,
                fu.to_string(),
                issue_port,
                &self.pipeline_reg[issue_port as usize]
            );

            let partition_issue = self.config.sub_core_model && fu.is_issue_partitioned();

            let issue_inst = &mut self.pipeline_reg[issue_port as usize];
            let ready_reg: Option<&mut Option<WarpInstruction>> = if partition_issue {
                issue_inst
                    .get_ready_sub_core_mut(fu.issue_reg_id())
                    .map(|(_, r)| r)
            } else {
                issue_inst.get_ready_mut().map(|(_, r)| r)
            };

            let Some(ready_reg) = ready_reg else {
                continue;
            };

            log::trace!("occupied: {}", fu.occupied().to_bit_string());
            log::trace!(
                "{} checking {}: fu[{:03}] can issue={:?} latency={:?}",
                style(format!("cycle {cycle:03} core {core_id:?}: execute:",)).red(),
                crate::Optional(ready_reg.as_ref()),
                fu_id,
                ready_reg.as_ref().map(|instr| fu.can_issue(instr)),
                ready_reg.as_ref().map(|instr| instr.latency),
            );

            if let Some(ref instr) = ready_reg {
                if fu.can_issue(instr) {
                    let schedule_wb_now = !fu.stallable();
                    let result_bus = self
                        .result_busses
                        .iter_mut()
                        .find(|bus| !bus[instr.latency]);

                    log::debug!(
                        "{} {} (partition issue={}, schedule wb now={}, resbus={}, latency={:?}) ready for issue to fu[{:03}]={}",
                        style(format!(
                            "cycle {cycle:03} core {core_id:?}: execute:",
                        ))
                        .red(),
                        instr,
                        partition_issue,
                        schedule_wb_now,
                        result_bus.is_some(),
                        ready_reg.as_ref().map(|reg| reg.latency),
                        fu_id,
                        fu,
                    );

                    let mut issued = true;
                    match result_bus {
                        Some(result_bus) if schedule_wb_now => {
                            debug_assert!(instr.latency < fu::MAX_ALU_LATENCY);
                            result_bus.set(instr.latency, true);
                            // println!("execute {} [latency={}]", instr, instr.latency);
                            fu.issue(ready_reg.take().unwrap(), &mut self.stats);
                        }
                        _ if !schedule_wb_now => {
                            // println!("execute {} [latency={}]", instr, instr.latency);
                            fu.issue(ready_reg.take().unwrap(), &mut self.stats);
                        }
                        _ => {
                            // stall issue (cannot reserve result bus)
                            issued = false;
                        }
                    }
                    log::debug!("execute: issue={}", issued);
                }
            }
        }
    }
}

// Returns numbers of addresses in translated_addrs.
//
// Each addr points to a 4B (32-bit) word
#[must_use]
pub fn translate_local_memaddr(
    // &self,
    local_addr: address,
    core_id: usize,
    thread_id: usize,
    num_cores: usize,
    data_size: u32,
    thread_block_size: usize,
    current_kernel_max_blocks: usize,
    config: &config::GPU,
) -> Vec<address> {
    // During functional execution, each thread sees its own memory space for
    // local memory, but these need to be mapped to a shared address space for
    // timing simulation.  We do that mapping here.

    let (thread_base, max_concurrent_threads) = if config.local_mem_map {
        // Dnew = D*N + T%nTpC + nTpC*C
        // N = nTpC*nCpS*nS (max concurent threads)
        // C = nS*K + S (hw cta number per gpu)
        // K = T/nTpC   (hw cta number per core)
        // D = data index
        // T = thread
        // nTpC = number of threads per CTA
        // nCpS = number of CTA per shader
        //
        // for a given local memory address threads in a CTA map to
        // contiguous addresses, then distribute across memory space by CTAs
        // from successive shader cores first, then by successive CTA in same
        // shader core
        let kernel_padded_threads_per_cta = thread_block_size;
        // let kernel_max_cta_per_sm = self.block_status.len();

        let temp = core_id + num_cores * (thread_id / kernel_padded_threads_per_cta);
        let rest = thread_id % kernel_padded_threads_per_cta;
        let thread_base = 4 * (kernel_padded_threads_per_cta * temp + rest);
        let max_concurrent_threads =
            kernel_padded_threads_per_cta * current_kernel_max_blocks * num_cores;
        (thread_base, max_concurrent_threads)
    } else {
        // legacy mapping that maps the same address in the local memory
        // space of all threads to a single contiguous address region
        let thread_base = 4 * (config.max_threads_per_core * core_id + thread_id);
        let max_concurrent_threads = num_cores * config.max_threads_per_core;
        (thread_base, max_concurrent_threads)
    };
    debug_assert!(thread_base < 4 /*word size*/ * max_concurrent_threads);

    // If requested datasize > 4B, split into multiple 4B accesses
    // otherwise do one sub-4 byte memory access
    let mut translated_addresses = vec![];

    if data_size >= 4 {
        // >4B access, split into 4B chunks
        debug_assert_eq!(data_size % 4, 0); // Must be a multiple of 4B
        let num_accesses = data_size / 4;
        // max 32B
        debug_assert!(
            num_accesses <= super::instruction::MAX_ACCESSES_PER_THREAD_INSTRUCTION as u32
        );
        // Address must be 4B aligned - required if
        // accessing 4B per request, otherwise access
        // will overflow into next thread's space
        debug_assert_eq!(local_addr % 4, 0);
        for i in 0..num_accesses {
            let local_word = local_addr / 4 + u64::from(i);
            let linear_address: address = local_word * max_concurrent_threads as u64 * 4
                + thread_base as u64
                + crate::LOCAL_GENERIC_START;
            translated_addresses.push(linear_address);
        }
    } else {
        // Sub-4B access, do only one access
        debug_assert!(data_size > 0);
        let local_word = local_addr / 4;
        let local_word_offset = local_addr % 4;
        // Make sure access doesn't overflow into next 4B chunk
        debug_assert_eq!((local_addr + u64::from(data_size) - 1) / 4, local_word);
        let linear_address: address = local_word * max_concurrent_threads as u64 * 4
            + local_word_offset
            + thread_base as u64
            + crate::LOCAL_GENERIC_START;
        translated_addresses.push(linear_address);
    }
    translated_addresses
}

impl<I, MC> Core<I, MC>
where
    I: ic::Interconnect<ic::Packet<mem_fetch::MemFetch>>,
{
    // #[inline]
    pub fn cache_flush(&mut self) {
        self.load_store_unit.flush();
    }

    // pub fn is_cache_flushed(&mut self) {
    //     self.load_store_unit.is_flushed();
    // }

    // #[inline]
    pub fn cache_invalidate(&mut self) {
        self.load_store_unit.invalidate();
    }

    #[must_use]
    // #[inline]
    pub fn ldst_unit_response_buffer_full(&self) -> bool {
        self.load_store_unit.response_buffer_full()
    }

    #[must_use]
    // #[inline]
    pub fn fetch_unit_response_buffer_full(&self) -> bool {
        false
    }

    // #[inline]
    pub fn accept_fetch_response(&self, mut fetch: mem_fetch::MemFetch, time: u64) {
        fetch.status = mem_fetch::Status::IN_SHADER_FETCHED;
        self.please_fill
            .lock()
            .push((FetchResponseTarget::ICache, fetch, time));
    }

    // #[inline]
    pub fn accept_ldst_unit_response(&self, fetch: mem_fetch::MemFetch, time: u64) {
        self.please_fill
            .lock()
            .push((FetchResponseTarget::LoadStoreUnit, fetch, time));
    }

    #[must_use]
    // #[inline]
    pub fn num_active_threads(&self) -> usize {
        self.num_active_threads
    }

    #[must_use]
    // #[inline]
    pub fn is_active(&self) -> bool {
        self.num_active_blocks > 0
    }

    #[must_use]
    // #[inline]
    pub fn num_active_blocks(&self) -> usize {
        self.num_active_blocks
    }

    // #[inline]
    pub fn can_issue_block(&self, kernel: &dyn Kernel) -> bool {
        let max_blocks = self.config.max_blocks(kernel).unwrap();
        if self.config.concurrent_kernel_sm {
            if max_blocks < 1 {
                return false;
            }
            // self.occupy_resource_for_block(kernel, false);
            unimplemented!("concurrent kernel sm model");
        } else {
            self.num_active_blocks < max_blocks
        }
    }

    #[must_use]
    // #[inline]
    pub fn id(&self) -> (usize, usize) {
        (self.cluster_id, self.core_id)
    }

    #[tracing::instrument(name = "core_reinit")]
    // #[inline]
    pub fn reinit(&mut self, start_thread: usize, end_thread: usize, reset_not_completed: bool) {
        if reset_not_completed {
            self.num_active_warps = 0;
            self.num_active_threads = 0;
            self.active_thread_mask.fill(false);
            self.occupied_block_to_hw_thread_id.clear();
            self.occupied_hw_thread_ids.fill(false);
        }
        for t in start_thread..end_thread {
            self.thread_state[t] = None;
        }
        let warp_size = self.config.warp_size;

        let start_warp = start_thread / warp_size;
        let end_warp = end_thread / warp_size;
        log::debug!(
            "reset warps {}..{} (threads {}..{})",
            start_warp,
            end_warp,
            start_thread,
            end_thread
        );

        for w in start_warp..end_warp {
            self.warps[w].reset();
        }
    }

    #[tracing::instrument(name = "core_issue_block")]
    pub fn issue_block(&mut self, kernel: &Arc<dyn Kernel>, cycle: u64) {
        log::debug!("core {:?}: issue block", self.id());
        if self.config.concurrent_kernel_sm {
            // let occupied = self.occupy_resource_for_block(&*kernel, true);
            // assert!(occupied);
            unimplemented!("concurrent kernel sm");
        } else {
            // calculate the max cta count and cta size for local memory address mapping
            // self.max_blocks_per_sm = self.config.max_blocks(kernel).unwrap();
            self.current_kernel_max_blocks = self.config.max_blocks(&**kernel).unwrap();
            self.thread_block_size = self.config.threads_per_block_padded(&**kernel);
        }

        // find a free block context
        let max_blocks_per_core = if self.config.concurrent_kernel_sm {
            unimplemented!("concurrent kernel sm");
            // self.config.max_concurrent_blocks_per_core
        } else {
            self.block_status.len()
        };
        log::debug!(
            "core {:?}: free block status: {:?}",
            self.id(),
            self.block_status
        );

        debug_assert_eq!(
            self.num_active_blocks,
            self.block_status
                .iter()
                .filter(|&num_threads_in_block| *num_threads_in_block > 0)
                .count()
        );
        let Some(free_block_hw_id) = self.block_status[0..max_blocks_per_core]
            .iter()
            .position(|num_threads_in_block| *num_threads_in_block == 0)
        else {
            return;
        };

        // determine hardware threads and warps that will be used for this block
        let thread_block_size = kernel.config().threads_per_block();
        let padded_thread_block_size = self.config.threads_per_block_padded(&**kernel);

        // hw warp id = hw thread id mod warp size, so we need to find a range
        // of hardware thread ids corresponding to an integral number of hardware
        // thread ids
        let (start_thread, end_thread) = if self.config.concurrent_kernel_sm {
            let start_thread = self
                .find_available_hw_thread_id(padded_thread_block_size, true)
                .unwrap();
            let end_thread = start_thread + thread_block_size;

            assert!(!self
                .occupied_block_to_hw_thread_id
                .contains_key(&free_block_hw_id));
            self.occupied_block_to_hw_thread_id
                .insert(free_block_hw_id, start_thread);
            (start_thread, end_thread)
        } else {
            let start_thread = free_block_hw_id * padded_thread_block_size;
            let end_thread = start_thread + thread_block_size;
            (start_thread, end_thread)
        };

        // reset state of the selected hardware thread and warp contexts
        self.reinit(start_thread, end_thread, false);

        // initalize scalar threads and determine which hardware warps they are
        // allocated to bind functional simulation state of threads to hardware
        // resources (simulation)
        let mut warps = WarpMask::ZERO;
        let block = kernel.next_block().expect("kernel has current block");
        log::debug!(
            "core {:?}: issue block {} from kernel {}",
            self.id(),
            block,
            kernel,
        );
        let block_id = block.id();

        let mut num_threads_in_block = 0;
        for i in start_thread..end_thread {
            self.thread_state[i] = Some(ThreadState {
                // block_id: free_block_hw_id,
                active: true,
                pc: 0, // todo
            });
            let warp_id = i / self.config.warp_size;

            // TODO: removed this but is that fine?
            if !kernel.no_more_blocks_to_run() {
                //     if !kernel.more_threads_in_block() {
                //         kernel.next_thread_iterlock().next();
                //     }
                //
                //     // we just incremented the thread id so this is not the same
                //     if !kernel.more_threads_in_block() {
                //         kernel.next_block_iterlock().next();
                //         *kernel.next_thread_iterlock() =
                //             kernel.config.block.into_iter().peekable();
                //     }
                num_threads_in_block += 1;
            }

            warps.set(warp_id, true);
        }

        kernel.increment_running_blocks();

        self.block_status[free_block_hw_id] = num_threads_in_block;
        log::debug!(
            "num threads in block {}={} (hw {}) = {}",
            block,
            block_id,
            free_block_hw_id,
            num_threads_in_block
        );

        self.barriers.allocate(free_block_hw_id as u64, warps);

        self.init_warps(free_block_hw_id, start_thread, end_thread, block_id, kernel);
        self.num_active_blocks += 1;
    }
}

impl<I, MC> Core<I, MC>
where
    I: ic::Interconnect<ic::Packet<mem_fetch::MemFetch>>,
{
    // #[inline]
    fn init_operand_collector(register_file: &mut RegisterFileUnit, config: &config::GPU) {
        register_file.add_cu_set(
            operand_collector::Kind::GEN_CUS,
            config.operand_collector_num_units_gen,
            config.operand_collector_num_out_ports_gen,
        );

        for _i in 0..config.operand_collector_num_in_ports_gen {
            let mut in_ports = operand_collector::PortVec::new();
            let mut out_ports = operand_collector::PortVec::new();
            let mut cu_sets: Vec<operand_collector::Kind> = Vec::new();

            in_ports.push(PipelineStage::ID_OC_SP);
            in_ports.push(PipelineStage::ID_OC_SFU);
            in_ports.push(PipelineStage::ID_OC_MEM);
            out_ports.push(PipelineStage::OC_EX_SP);
            out_ports.push(PipelineStage::OC_EX_SFU);
            out_ports.push(PipelineStage::OC_EX_MEM);

            if config.num_dp_units > 0 {
                in_ports.push(PipelineStage::ID_OC_DP);
                out_ports.push(PipelineStage::OC_EX_DP);
            }
            if config.num_int_units > 0 {
                in_ports.push(PipelineStage::ID_OC_INT);
                out_ports.push(PipelineStage::OC_EX_INT);
            }

            cu_sets.push(operand_collector::Kind::GEN_CUS);
            register_file.add_port(in_ports, out_ports, cu_sets);
        }

        if config.enable_specialized_operand_collector {
            for (kind, num_collector_units, num_dispatch_units) in [
                (
                    operand_collector::Kind::SP_CUS,
                    config.operand_collector_num_units_sp,
                    config.operand_collector_num_out_ports_sp,
                ),
                (
                    operand_collector::Kind::DP_CUS,
                    config.operand_collector_num_units_dp,
                    config.operand_collector_num_out_ports_dp,
                ),
                (
                    operand_collector::Kind::SFU_CUS,
                    config.operand_collector_num_units_sfu,
                    config.operand_collector_num_out_ports_sfu,
                ),
                (
                    operand_collector::Kind::MEM_CUS,
                    config.operand_collector_num_units_mem,
                    config.operand_collector_num_out_ports_mem,
                ),
                (
                    operand_collector::Kind::INT_CUS,
                    config.operand_collector_num_units_int,
                    config.operand_collector_num_out_ports_int,
                ),
            ] {
                register_file.add_cu_set(kind, num_collector_units, num_dispatch_units);
            }

            for _ in 0..config.operand_collector_num_in_ports_sp {
                let in_ports = vec![PipelineStage::ID_OC_SP];
                let out_ports = vec![PipelineStage::OC_EX_SP];

                let cu_sets = vec![
                    operand_collector::Kind::SP_CUS,
                    operand_collector::Kind::GEN_CUS,
                ];

                register_file.add_port(in_ports, out_ports, cu_sets);
            }

            for _ in 0..config.operand_collector_num_in_ports_dp {
                let in_ports = vec![PipelineStage::ID_OC_DP];
                let out_ports = vec![PipelineStage::OC_EX_DP];

                let cu_sets = vec![
                    operand_collector::Kind::DP_CUS,
                    operand_collector::Kind::GEN_CUS,
                ];

                register_file.add_port(in_ports, out_ports, cu_sets);
            }

            for _ in 0..config.operand_collector_num_in_ports_sfu {
                let in_ports = vec![PipelineStage::ID_OC_SFU];
                let out_ports = vec![PipelineStage::OC_EX_SFU];

                let cu_sets = vec![
                    operand_collector::Kind::SFU_CUS,
                    operand_collector::Kind::GEN_CUS,
                ];

                register_file.add_port(in_ports, out_ports, cu_sets);
            }

            for _ in 0..config.operand_collector_num_in_ports_mem {
                let in_ports = vec![PipelineStage::ID_OC_MEM];
                let out_ports = vec![PipelineStage::OC_EX_MEM];

                let cu_sets = vec![
                    operand_collector::Kind::MEM_CUS,
                    operand_collector::Kind::GEN_CUS,
                ];

                register_file.add_port(in_ports, out_ports, cu_sets);
            }

            for _ in 0..config.operand_collector_num_in_ports_int {
                let in_ports = vec![PipelineStage::ID_OC_INT];
                let out_ports = vec![PipelineStage::OC_EX_INT];

                let cu_sets = vec![
                    operand_collector::Kind::INT_CUS,
                    operand_collector::Kind::GEN_CUS,
                ];

                register_file.add_port(in_ports, out_ports, cu_sets);
            }
        }

        // this must be called after we add the collector unit sets!
        register_file.init(config.num_reg_banks);
    }

    // #[inline]
    fn register_thread_in_block_exited(
        &mut self,
        block_hw_id: usize,
        kernel: &Option<Arc<dyn Kernel>>,
    ) {
        let current_kernel: &mut Option<_> = &mut *self.current_kernel.try_lock();
        debug_assert!(block_hw_id < self.block_status.len());
        debug_assert!(self.block_status[block_hw_id] > 0);
        self.block_status[block_hw_id] -= 1;

        // this is the last thread that exited
        if self.block_status[block_hw_id] == 0 {
            // deallocate barriers for this block
            self.barriers.deallocate(block_hw_id as u64);

            // increment the number of completed blocks
            self.num_active_blocks -= 1;
            if self.num_active_blocks == 0 {
                // Shader can only be empty when no more cta are dispatched
                if kernel.as_ref().map(|k| k.id()) != current_kernel.as_ref().map(|k| k.id()) {
                    // debug_assert!(current_kernel.is_none() || kernel.no_more_blocks_to_run());
                }
                *current_kernel = None;
            }
            //
            // self.release_shader_resource_1block(cta_num, kernel);
            if let Some(kernel) = kernel {
                kernel.decrement_running_blocks();
                if kernel.no_more_blocks_to_run()
                    && !kernel.running()
                    && current_kernel.as_ref().map(|k| k.id()) == Some(kernel.id())
                {
                    log::info!("kernel {} ({}) completed", kernel.name(), kernel.id());
                    *current_kernel = None;
                    // set done here
                }
            }
        }
    }

    /// Shader core decode
    #[tracing::instrument]
    // #[inline]
    fn decode(&mut self, cycle: u64) {
        let InstrFetchBuffer { valid, warp_id, .. } = self.instr_fetch_buffer;

        log::debug!(
            "{}",
            style(format!(
                "cycle {:03} core {:?}: decode (fetch buffer valid={})",
                cycle,
                self.id(),
                valid
            ))
            .blue()
        );

        if !valid {
            return;
        }

        // decode 1 or 2 instructions and buffer them
        let warp = self.warps.get_mut(warp_id).unwrap();
        debug_assert_eq!(warp.warp_id, warp_id);

        let instr1 = warp.next_trace_inst().cloned();
        let instr2 = if instr1.is_some() {
            warp.next_trace_inst().cloned()
        } else {
            None
        };

        // debug: print all instructions in this warp
        assert_eq!(warp.warp_id, warp_id);
        if false {
            let already_issued_trace_pc = warp.trace_pc;
            print_trace_instructions(&warp, already_issued_trace_pc);
        }

        if let Some(instr1) = instr1 {
            self.decode_instruction(warp_id, instr1, 0);
        }

        if let Some(instr2) = instr2 {
            self.decode_instruction(warp_id, instr2, 1);
        }

        self.instr_fetch_buffer.valid = false;
    }

    // #[inline]
    fn decode_instruction(&mut self, warp_id: usize, instr: WarpInstruction, slot: usize) {
        let warp = self.warps.get_mut(warp_id).unwrap();

        log::debug!(
            "====> warp[warp_id={:03}] ibuffer fill at slot {:01} with instruction {}",
            warp.warp_id,
            slot,
            instr,
        );

        warp.ibuffer_fill(slot, instr);
        warp.num_instr_in_pipeline += 1;
    }

    #[tracing::instrument]
    // #[inline]
    fn issue(&mut self, cycle: u64) {
        // fair round robin issue between schedulers
        let num_schedulers = self.schedulers.len();
        assert_eq!(num_schedulers, self.config.num_schedulers_per_core);

        for scheduler_idx in 0..num_schedulers {
            let scheduler_idx = (self.scheduler_issue_priority + scheduler_idx) % num_schedulers;

            // compute subset of warps supervised by this scheduler
            let scheduler_supervised_warps: Vec<_> = self
                .warps
                .iter_mut()
                .enumerate()
                .filter(|(i, _)| scheduler_idx == i % num_schedulers)
                .map(|(_, warp)| warp)
                .collect();

            let mut issuer = CoreIssuer {
                config: &self.config,
                pipeline_reg: &mut self.pipeline_reg,
                warp_instruction_unique_uid: &self.warp_instruction_unique_uid,
                allocations: &self.allocations,
                core_id: self.core_id,
                thread_block_size: self.thread_block_size,
                current_kernel_max_blocks: self.current_kernel_max_blocks,
                scoreboard: &mut *self.scoreboard,
                barriers: &mut self.barriers,
            };
            self.schedulers[scheduler_idx].issue_to(&mut issuer, scheduler_supervised_warps, cycle);
        }
        self.scheduler_issue_priority = (self.scheduler_issue_priority + 1) % num_schedulers;
    }

    #[tracing::instrument]
    // #[inline]
    fn writeback(&mut self, cycle: u64) {
        // from the functional units
        let id = self.id();
        let exec_writeback_pipeline = &mut self.pipeline_reg[PipelineStage::EX_WB as usize];

        log::debug!(
            "{}",
            style(format!(
                "cycle {:03} core {:?}: writeback: ex wb pipeline={}",
                cycle, id, exec_writeback_pipeline
            ))
            .cyan()
        );
        let _max_committed_thread_instructions =
            self.config.warp_size * exec_writeback_pipeline.size();

        // m_stats->m_pipeline_duty_cycle[m_sid] =
        //     ((float)(m_stats->m_num_sim_insn[m_sid] -
        //              m_stats->m_last_num_sim_insn[m_sid])) /
        //     max_committed_thread_instructions;
        //
        // m_stats->m_last_num_sim_insn[m_sid] = m_stats->m_num_sim_insn[m_sid];
        // m_stats->m_last_num_sim_winsn[m_sid] = m_stats->m_num_sim_winsn[m_sid];
        while let Some(mut ready) = exec_writeback_pipeline
            .get_ready_mut()
            .and_then(|(_, r)| r.take())
        {
            log::debug!("ready for writeback: {}", ready);

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
            self.register_file.writeback(&mut ready);
            self.scoreboard.release_all(&ready);
            let warp = self.warps.get_mut(ready.warp_id).unwrap();
            warp.num_instr_in_pipeline -= 1;
            warp_inst_complete(&mut ready, &mut self.stats);

            //   m_gpu->gpu_sim_insn_last_update_sid = m_sid;
            //   m_gpu->gpu_sim_insn_last_update = m_gpu->gpu_sim_cycle;
            //   m_last_inst_gpu_sim_cycle = m_gpu->gpu_sim_cycle;
            //   m_last_inst_gpu_tot_sim_cycle = m_gpu->gpu_tot_sim_cycle;
            // preg = m_pipeline_reg[EX_WB].get_ready();
            //   pipe_reg = (preg == NULL) ? NULL : *preg;
        }
    }

    #[must_use]
    // #[inline]
    fn find_available_hw_thread_id(
        &mut self,
        thread_block_size: usize,
        occupy: bool,
    ) -> Option<usize> {
        let mut step = 0;
        while step < self.config.max_threads_per_core {
            if self.occupied_hw_thread_ids[step..(step + thread_block_size)].not_any() {
                // found consecutive non-active
                break;
            }
            // for hw_thread_id in step..(step + thread_block_size) {
            //     if self.occupied_hw_thread_ids[hw_thread_id] {
            //         break;
            //     }
            // }
            // consecutive non-active
            // if hw_thread_id == step + thread_block_size {
            //     break;
            // }
            step += thread_block_size;
        }
        if step >= self.config.max_threads_per_core {
            // didn't find
            None
        } else {
            if occupy {
                self.occupied_hw_thread_ids[step..(step + thread_block_size)].fill(true);
                // for hw_thread_id in step..(step + thread_block_size) {
                //     self.occupied_hw_thread_ids.set(hw_thread_id, true);
                // }
            }
            Some(step)
        }
    }

    #[tracing::instrument(name = "core_init_warps_from_traces")]
    // #[inline]
    fn init_warps_from_traces(
        &mut self,
        kernel: &Arc<dyn Kernel>,
        start_warp: usize,
        end_warp: usize,
    ) {
        debug_assert!(!self.warps.is_empty());
        let selected_warps = &mut self.warps[start_warp..end_warp];
        for warp in &mut *selected_warps {
            warp.trace_instructions.clear();
            warp.kernel = Some(Arc::clone(kernel));
            warp.trace_pc = 0;
        }

        let have_block = crate::timeit!(
            "core::read_trace",
            kernel.next_threadblock_traces(selected_warps, &self.config)
        );
        if have_block {
            let kernel_stats = self.stats.get_mut(Some(kernel.id() as usize));
            kernel_stats.sim.num_blocks += 1;
        }
        log::debug!(
            "initialized traces {}..{} of {} warps",
            start_warp,
            end_warp,
            &self.warps.len()
        );
    }

    #[tracing::instrument(name = "core_init_warps")]
    // #[inline]
    fn init_warps(
        &mut self,
        block_hw_id: usize,
        start_thread: usize,
        end_thread: usize,
        block_id: u64,
        kernel: &Arc<dyn Kernel>,
    ) {
        // let threads_per_block = kernel.threads_per_block();
        let start_warp = start_thread / self.config.warp_size;
        let end_warp = end_thread / self.config.warp_size
            + usize::from(end_thread % self.config.warp_size != 0);
        for warp_id in start_warp..end_warp {
            let mut num_active = 0;

            let mut local_active_thread_mask = warp::ActiveMask::ZERO;
            for warp_thread_id in 0..self.config.warp_size {
                let hwtid = warp_id * self.config.warp_size + warp_thread_id;
                if hwtid < end_thread {
                    num_active += 1;
                    debug_assert!(!self.active_thread_mask[hwtid]);
                    self.active_thread_mask.set(hwtid, true);
                    local_active_thread_mask.set(warp_thread_id, true);
                }
            }

            // crate::WIP_STATS.lock().num_warps += 1;
            // crate::WIP_STATS.lock().warps_per_core[self.core_id] += 1;

            // self.warps[warp_id].try_lock().init(
            let warp = self.warps.get_mut(warp_id).unwrap();
            warp.init(
                block_hw_id as u64,
                warp_id,
                self.dynamic_warp_id,
                local_active_thread_mask,
                Arc::clone(kernel),
            );

            self.dynamic_warp_id += 1;
            self.num_active_warps += 1;
            self.num_active_threads += num_active;
        }

        log::debug!(
            "initialized warps {}..{} (threads {}..{}) for block {} (hw {})",
            start_warp,
            end_warp,
            start_thread,
            end_thread,
            block_id,
            block_hw_id,
        );
        self.init_warps_from_traces(kernel, start_warp, end_warp);
    }
}

impl<I, MC> crate::engine::cycle::Component for Core<I, MC>
where
    I: ic::Interconnect<ic::Packet<mem_fetch::MemFetch>>,
    MC: crate::mcu::MemoryController,
{
    #[tracing::instrument(name = "core_cycle")]
    fn cycle(&mut self, cycle: u64) {
        log::debug!(
            "{} \tactive={}, not completed={} ldst unit response buffer={}",
            style(format!(
                "cycle {:03} core {:?}: core cycle",
                cycle,
                self.id()
            ))
            .blue(),
            self.is_active(),
            self.num_active_threads(),
            self.load_store_unit.response_fifo.len()
        );

        // // workaround when l1 flush is enabled and we need to flush the L1 after a mem barrier
        // // FIXME: this is likely implemented wrong - causing a invalidations every cycle
        // let need_l1_flush = {
        //     let mut need_l1_flush_lock = self.need_l1_flush.lock();
        //     let need_l1_flush = *need_l1_flush_lock;
        //     *need_l1_flush_lock = false;
        //     need_l1_flush
        // };
        // if need_l1_flush {
        //     self.cache_invalidate();
        // }

        for (target, fetch, time) in self.please_fill.lock().drain(..) {
            if let Some(fetch_return_cb) = &self.fetch_return_callback {
                fetch_return_cb(cycle, &fetch);
            }
            match target {
                FetchResponseTarget::LoadStoreUnit => self.load_store_unit.fill(fetch),
                FetchResponseTarget::ICache => self.instr_l1_cache.fill(fetch, time),
            }
        }

        // if !self.is_active() && self.not_completed() == 0 {
        //     log::debug!(
        //         "{}",
        //         style(format!(
        //             "cycle {:03} core {:?}: core done",
        //             cycle,
        //             self.id()
        //         ))
        //         .blue(),
        //     );
        //     return;
        // }

        // m_stats->shader_cycles[m_sid]++;
        // "writeback"
        // self.writeback(cycle);
        crate::timeit!("core::writeback", self.writeback(cycle));
        // this made it already double the time per core cycle
        crate::timeit!("core::execute", self.execute(cycle));
        for _ in 0..self.config.reg_file_port_throughput {
            crate::timeit!(
                "core::operand collector",
                self.register_file.step(&mut self.pipeline_reg)
            );
        }

        crate::timeit!("core::issue", self.issue(cycle));
        for _i in 0..self.config.inst_fetch_throughput {
            crate::timeit!("core::decode", self.decode(cycle));
            crate::timeit!("core::fetch", self.fetch(cycle));
        }
    }
}

pub fn warp_inst_complete(instr: &mut WarpInstruction, stats: &mut stats::PerKernel) {
    let kernel_stats = stats.get_mut(Some(instr.kernel_launch_id as usize));
    kernel_stats.sim.instructions += instr.active_thread_count() as u64;
    // crate::WIP_STATS.lock().warp_instructions += 1;
}

#[allow(dead_code)]
fn print_trace_instructions(warp: &warp::Warp, already_issued_trace_pc: usize) {
    log::debug!(
        "====> instruction at trace pc < {:<4} already issued ...",
        already_issued_trace_pc
    );

    for (trace_pc, trace_instr) in warp
        .trace_instructions
        .iter()
        .enumerate()
        .skip(already_issued_trace_pc)
    {
        log::debug!(
            "====> warp[{:03}][trace_pc={:03}]:\t {}\t\t active={} \tpc={} idx={}",
            warp.warp_id,
            trace_pc,
            trace_instr,
            trace_instr.active_mask.to_bit_string(),
            trace_instr.pc,
            trace_instr.trace_idx
        );
    }
}
