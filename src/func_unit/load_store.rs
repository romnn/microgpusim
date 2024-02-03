use crate::sync::{Arc, Mutex};
use crate::{
    address, cache, config,
    core::PipelineStage,
    fifo::Fifo,
    func_unit as fu,
    instruction::{CacheOperator, MemorySpace, WarpInstruction},
    interconn as ic, mcu, mem_fetch, mem_sub_partition, mshr,
    operand_collector::OperandCollector,
    register_set::{self},
    scoreboard::{self, Access},
    warp,
};
use utils::box_slice;

use console::style;
use mem_fetch::{access::Kind as AccessKind, MemFetch};
use std::collections::{HashMap, VecDeque};
use strum::EnumCount;

#[allow(clippy::module_name_repetitions)]
pub struct LoadStoreUnit<MC> {
    /// Core ID
    core_id: usize,
    /// Cluster ID
    cluster_id: usize,
    /// Next writeback instruction
    next_writeback: Option<WarpInstruction>,
    /// Response fifo queue
    // pub response_queue: VecDeque<MemFetch>,
    pub response_queue: Arc<Mutex<Fifo<ic::Packet<MemFetch>>>>,
    pub data_l1: Option<Box<dyn cache::Cache<stats::cache::PerKernel>>>,
    /// Config
    config: Arc<config::GPU>,
    /// Memory controller
    mem_controller: Arc<MC>,
    /// Next global access
    next_global: Option<MemFetch>,
    /// Pending writes per register
    pub pending_writes: HashMap<usize, HashMap<u32, usize>>,

    /// L1 tag latency queue
    pub l1_latency_queue: Box<[Box<[Option<mem_fetch::MemFetch>]>]>,
    // pub l1_latency_queue: Box<[VecDeque<(u64, mem_fetch::MemFetch)>]>,
    /// L1 hit latency queue
    pub l1_hit_latency_queue: VecDeque<(u64, mem_fetch::MemFetch)>,

    /// Memory port
    // pub mem_port: ic::Port<mem_fetch::MemFetch>,
    inner: fu::PipelinedSimdUnit,

    /// Round-robin write-back arbiter.
    ///
    /// The arbiter handles contention between L1T, L1C, and shared memory
    writeback_arb: usize,
    /// Number of writeback clients (L1T, L1C, and shared memory)
    num_writeback_clients: usize,

    /// Callbacks
    pub l1_access_callback:
        Option<Box<dyn Fn(u64, &mem_fetch::MemFetch, cache::RequestStatus) + Send + Sync>>,
}

impl<MC> std::fmt::Display for LoadStoreUnit<MC> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner.name)
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
#[allow(dead_code)]
#[repr(usize)]
enum MemStageAccessKind {
    C_MEM,
    T_MEM,
    S_MEM,
    G_MEM_LD,
    L_MEM_LD,
    G_MEM_ST,
    L_MEM_ST,
}

#[derive(strum::EnumCount, strum::FromRepr, Hash, PartialEq, Eq, Clone, Copy, Debug)]
#[allow(dead_code)]
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

impl<MC> LoadStoreUnit<MC>
where
    MC: mcu::MemoryController,
{
    pub fn new(
        id: usize,
        core_id: usize,
        cluster_id: usize,
        // mem_port: ic::Port<mem_fetch::MemFetch>,
        config: Arc<config::GPU>,
        mem_controller: Arc<MC>,
    ) -> Self {
        let pipeline_depth = config.shared_memory_latency;
        let inner = fu::PipelinedSimdUnit::new(
            id,
            "LdstUnit".to_string(),
            pipeline_depth,
            config.clone(),
            0,
        );
        debug_assert!(config.shared_memory_latency > 1);

        let data_l1: Option<Box<dyn cache::Cache<_>>> =
            if let Some(l1_config) = &config.data_cache_l1 {
                // initialize l1 data cache
                let cache_stats = stats::cache::PerKernel::default();

                let cache_controller = cache::controller::pascal::L1DataCacheController::new(
                    cache::Config::new(l1_config.inner.as_ref(), config.accelsim_compat),
                    l1_config,
                    config.accelsim_compat,
                );

                let mut data_cache: cache::data::Data<
                    MC,
                    cache::controller::pascal::L1DataCacheController,
                    stats::cache::PerKernel,
                > = cache::data::Builder {
                    name: format!(
                        "ldst-unit-{cluster_id}-{core_id}-{}",
                        style("L1D-CACHE").green()
                    ),
                    id,
                    kind: cache::base::Kind::OnChip,
                    stats: cache_stats,
                    config: Arc::clone(&config),
                    mem_controller: mem_controller.clone(),
                    cache_controller,
                    cache_config: Arc::clone(&l1_config.inner),
                    write_alloc_type: AccessKind::L1_WR_ALLOC_R,
                    write_back_type: AccessKind::L1_WRBK_ACC,
                }
                .build();
                // data_cache.set_top_port(mem_port.clone());

                Some(Box::new(data_cache))
            } else {
                None
            };

        debug_assert!(data_l1.is_some());

        let l1_banks = config
            .data_cache_l1
            .as_ref()
            .map(|l1_config| l1_config.l1_banks)
            .unwrap_or(0);

        let l1_latency = config
            .data_cache_l1
            .as_ref()
            .map(|l1_config| l1_config.l1_latency)
            .unwrap_or(0);

        let l1_latency_queue = box_slice![box_slice![None; l1_latency]; l1_banks];
        // let l1_latency_queue = box_slice![VecDeque::new(); l1_banks];

        let l1_hit_latency_queue = VecDeque::new();

        // let response_queue = VecDeque::new();
        let response_queue = Fifo::new(Some(config.num_ldst_response_buffer_size));
        let response_queue = Arc::new(Mutex::new(response_queue));

        Self {
            core_id,
            cluster_id,
            data_l1,
            next_writeback: None,
            next_global: None,
            pending_writes: HashMap::new(),
            response_queue,
            // mem_port,
            inner,
            config,
            mem_controller: Arc::clone(&mem_controller),
            num_writeback_clients: WritebackClient::COUNT,
            writeback_arb: 0,
            l1_latency_queue,
            l1_hit_latency_queue,
            l1_access_callback: None,
        }
    }

    // #[inline]
    fn memory_cycle(
        &mut self,
        warps: &mut [warp::Warp],
        mem_port: &mut dyn ic::Connection<ic::Packet<mem_fetch::MemFetch>>,
        rc_fail: &mut MemStageStallKind,
        kind: &mut MemStageAccessKind,
        cycle: u64,
    ) -> bool {
        let Some(dispatch_instr) = &self.inner.dispatch_reg else {
            return true;
        };
        log::debug!("memory cycle for instruction: {}", &dispatch_instr);

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

        if self.data_l1.is_none() || dispatch_instr.cache_operator == Some(CacheOperator::Global) {
            bypass_l1 = true;
        } else if dispatch_instr.memory_space == Some(MemorySpace::Global) {
            // skip L1 if global memory access does not use L1 by default
            if self.config.global_mem_skip_l1_data_cache
                && dispatch_instr.cache_operator != Some(CacheOperator::L1)
            {
                bypass_l1 = true;
            }
        }

        // log::warn!("bypass l1={}", bypass_l1);
        let Some(access) = dispatch_instr.mem_access_queue.back() else {
            return true;
        };

        log::debug!(
            "memory cycle for instruction Some({}) => access: {} (bypass l1={}, queue={:?})",
            &dispatch_instr,
            access,
            bypass_l1,
            dispatch_instr
                .mem_access_queue
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
        );

        let mut stall_cond = MemStageStallKind::NO_RC_FAIL;
        if bypass_l1 {
            // bypass L1 cache
            debug_assert_eq!(dispatch_instr.is_store(), access.is_write);

            // let mut mem_port = self.mem_port.lock();

            let packet_size = if dispatch_instr.is_store() || dispatch_instr.is_atomic() {
                access.size()
            } else {
                access.control_size()
            };

            if mem_port.can_send(&[packet_size]) {
                if dispatch_instr.is_load() {
                    for out_reg in dispatch_instr.outputs() {
                        let pending = &self.pending_writes[&dispatch_instr.warp_id];
                        debug_assert!(pending[out_reg] > 0);
                    }
                } else if dispatch_instr.is_store() {
                    warps[dispatch_instr.warp_id].num_outstanding_stores += 1;
                }

                let instr = self.inner.dispatch_reg.as_mut().unwrap();
                let access = instr.mem_access_queue.pop_back().unwrap();

                let physical_addr = self.mem_controller.to_physical_address(access.addr);

                let fetch = mem_fetch::Builder {
                    instr: Some(instr.clone()),
                    access,
                    warp_id: instr.warp_id,
                    core_id: Some(self.core_id),
                    cluster_id: Some(self.cluster_id),
                    physical_addr,
                }
                .build();

                log::debug!("memory cycle for instruction {} => send {}", &instr, fetch);

                mem_port.send(ic::Packet { fetch, time: cycle });
            }
        } else {
            debug_assert_ne!(dispatch_instr.cache_operator, None);
            stall_cond = crate::timeit!(
                "core::execute::process_memory_access_queue_l1cache",
                self.process_memory_access_queue_l1cache(warps, cycle)
            );
        }

        let dispatch_instr = self.inner.dispatch_reg.as_ref().unwrap();

        if !dispatch_instr.mem_access_queue.is_empty()
            && stall_cond == MemStageStallKind::NO_RC_FAIL
        {
            stall_cond = MemStageStallKind::COAL_STALL;
        }

        log::debug!("memory instruction stall cond: {:?}", &stall_cond);
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
        warps: &mut [warp::Warp],
        cycle: u64,
    ) -> MemStageStallKind {
        let mut stall_cond = MemStageStallKind::NO_RC_FAIL;
        let Some(instr) = &mut self.inner.dispatch_reg else {
            return MemStageStallKind::NO_RC_FAIL;
        };

        let Some(access) = instr.mem_access_queue.back() else {
            return MemStageStallKind::NO_RC_FAIL;
        };
        let dbg_access = access.clone();

        let l1d_config = self.config.data_cache_l1.as_ref().unwrap();

        if l1d_config.l1_latency > 0 {
            // We can handle at most l1_banks reqs per cycle
            for _ in 0..l1d_config.l1_banks {
                let Some(access) = instr.mem_access_queue.back() else {
                    return MemStageStallKind::NO_RC_FAIL;
                };

                let bank_id = self
                    .data_l1
                    .as_ref()
                    .unwrap()
                    .controller()
                    .set_bank(access.addr) as usize;
                log::trace!("{}: {} -> bank {}", access, access.addr, bank_id);
                debug_assert!(bank_id < l1d_config.l1_banks);

                if log::log_enabled!(log::Level::Trace) {
                    log::trace!(
                        "computed bank id {} for access {} (access queue={:?} l1 latency queue={:?})",
                        bank_id,
                        access,
                        &instr
                            .mem_access_queue
                            .iter()
                            .map(ToString::to_string)
                            .collect::<Vec<_>>(),
                        self.l1_latency_queue[bank_id]
                            .iter()
                            // .map(|(cycle, fetch)| format!("{}<-{}", cycle, fetch))
                            .map(Option::as_ref)
                            .map(crate::Optional)
                            .collect::<Vec<_>>(),
                    );
                }

                let slot_idx = l1d_config.l1_latency - 1;
                let slot = &mut self.l1_latency_queue[bank_id][slot_idx];
                let has_slot = slot.is_none();

                // let ready_cycle = cycle + l1d_config.l1_latency as u64;
                // let has_slot = self.l1_latency_queue[bank_id]
                //     .back()
                //     .map(|(recent_ready_cycle, _)| *recent_ready_cycle < ready_cycle)
                //     .unwrap_or(true);

                if has_slot {
                    let is_store = instr.is_store();
                    let access = instr.mem_access_queue.pop_back().unwrap();

                    let physical_addr = self.mem_controller.to_physical_address(access.addr);

                    let mut fetch = mem_fetch::Builder {
                        instr: Some(instr.clone()),
                        access,
                        warp_id: instr.warp_id,
                        core_id: Some(self.core_id),
                        cluster_id: Some(self.cluster_id),
                        physical_addr,
                    }
                    .build();
                    // println!(
                    //     "add fetch {:<35} rel addr={:<4} bank={:<2} slot={:<4} at cycle={:<4}",
                    //     fetch.to_string(),
                    //     fetch.relative_byte_addr(),
                    //     bank_id,
                    //     slot_idx,
                    //     cycle
                    // );
                    fetch.inject_cycle = Some(cycle);

                    let data_size = fetch.data_size();
                    // self.l1_latency_queue[bank_id].push_back((ready_cycle, fetch));
                    *slot = Some(fetch);

                    if is_store {
                        let inc_ack = if l1d_config.inner.mshr_kind == mshr::Kind::SECTOR_ASSOC {
                            data_size / mem_sub_partition::SECTOR_SIZE
                        } else {
                            1
                        };

                        let warp = warps.get_mut(instr.warp_id).unwrap();
                        // let warp = warp.try_lock();
                        for _ in 0..inc_ack {
                            warp.num_outstanding_stores += 1;
                        }
                    }
                } else {
                    stall_cond = MemStageStallKind::BK_CONF;
                    if let Some(ref mut l1_cache) = self.data_l1 {
                        let stats = l1_cache.per_kernel_stats_mut();
                        let kernel_stats = stats.get_mut(access.kernel_launch_id());
                        kernel_stats.num_l1_cache_bank_conflicts += 1;
                    }

                    // do not try again, just break from the loop and try the next cycle
                    break;
                }
            }

            if !instr.mem_access_queue.is_empty() && stall_cond != MemStageStallKind::BK_CONF {
                stall_cond = MemStageStallKind::COAL_STALL;
            }

            log::trace!(
                "process_memory_access_queue_l1cache stall cond {:?} for access {} (access queue size={:?})",
                stall_cond,
                &dbg_access,
                &instr.mem_access_queue.iter().map(ToString::to_string).collect::<Vec<_>>(),
            );
            stall_cond
        } else {
            let physical_addr = self.mem_controller.to_physical_address(access.addr);

            let fetch = mem_fetch::Builder {
                instr: Some(instr.clone()),
                access: access.clone(),
                warp_id: instr.warp_id,
                core_id: Some(self.core_id),
                cluster_id: Some(self.cluster_id),
                physical_addr,
            }
            .build();
            let mut events = Vec::new();
            let _status =
                self.data_l1
                    .as_mut()
                    .unwrap()
                    .access(fetch.addr(), fetch, &mut events, cycle);
            todo!("process cache access");
            // self.process_cache_access(cache, fetch.addr(), instr, fetch, status)
            // stall_cond
        }
    }
}

impl<MC> LoadStoreUnit<MC> {
    // #[must_use]
    // pub fn response_buffer_full(&self) -> bool {
    //     self.response_queue.len() >= self.config.num_ldst_response_buffer_size
    // }

    pub fn flush(&mut self) {
        if let Some(ref mut l1) = self.data_l1 {
            l1.flush();
        }
    }

    pub fn invalidate(&mut self) {
        if let Some(ref mut l1) = self.data_l1 {
            l1.invalidate();
        }
    }

    pub fn fill(&self, mut fetch: MemFetch, time: u64) {
        fetch.status = mem_fetch::Status::IN_SHADER_LDST_RESPONSE_FIFO;
        // self.response_queue.push_back(fetch);
        self.response_queue
            .lock()
            .enqueue(ic::Packet { fetch, time });
    }

    pub fn writeback(
        &mut self,
        operand_collector: &mut dyn OperandCollector,
        scoreboard: &mut dyn scoreboard::Access<WarpInstruction>,
        warps: &mut [warp::Warp],
        stats: &mut stats::PerKernel,
        cycle: u64,
    ) {
        log::debug!(
            "{} (arb={}, writeback clients={})",
            style(format!("load store unit: cycle {cycle} writeback")).magenta(),
            self.writeback_arb,
            self.num_writeback_clients,
        );

        // this processes the next writeback
        if let Some(ref mut next_writeback) = self.next_writeback {
            log::trace!(
                "{} => next_writeback={} ({:?})",
                style("ldst unit writeback").magenta(),
                next_writeback,
                next_writeback.memory_space,
            );

            if operand_collector.writeback(next_writeback) {
                let mut next_writeback = self.next_writeback.take().unwrap();

                let mut instr_completed = false;
                for out_reg in next_writeback.outputs() {
                    debug_assert!(*out_reg > 0);

                    if next_writeback.memory_space == Some(MemorySpace::Shared) {
                        // shared
                        scoreboard.release(next_writeback.warp_id, *out_reg);
                        instr_completed = true;
                    } else {
                        let pending = self
                            .pending_writes
                            .entry(next_writeback.warp_id)
                            .or_default();
                        let still_pending = pending.get_mut(out_reg).unwrap();
                        debug_assert!(*still_pending > 0);
                        *still_pending -= 1;
                        log::trace!(
                            "{} => next_writeback={} dest register {}: pending writes={}",
                            style("ldst unit writeback").magenta(),
                            next_writeback,
                            out_reg,
                            still_pending,
                        );

                        if *still_pending == 0 {
                            pending.remove(out_reg);
                            scoreboard.release(next_writeback.warp_id, *out_reg);
                            instr_completed = true;
                        }
                    }
                }
                if instr_completed {
                    crate::warp_inst_complete(&mut next_writeback, &mut *stats);
                }
            }
        }

        // this arbitrates between the writeback clients
        // sets next writeback for writeback in the next cycle
        let mut serviced_client = None;
        for client in 0..self.num_writeback_clients {
            if self.next_writeback.is_some() {
                break;
            }
            let next_client_id = (client + self.writeback_arb) % self.num_writeback_clients;
            let next_client = WritebackClient::from_repr(next_client_id).unwrap();
            log::trace!("checking writeback client {:?}", next_client);

            #[allow(clippy::match_same_arms)]
            match next_client {
                WritebackClient::SharedMemory => {
                    if let Some(pipe_reg) = self.inner.pipeline_reg[0].take() {
                        if pipe_reg.is_atomic() {
                            todo!("atomics");
                        }

                        let warp = warps.get_mut(pipe_reg.warp_id).unwrap();
                        warp.num_instr_in_pipeline -= 1;
                        self.next_writeback = Some(pipe_reg);
                        serviced_client = Some(next_client_id);
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
                        log::debug!(
                            "{}",
                            style(format!(
                                "ldst unit writeback: has global {} ({})",
                                crate::Optional(next_global.instr.as_ref()),
                                &next_global.addr()
                            ))
                            .magenta(),
                        );
                        if next_global.is_atomic() {
                            todo!("atomics");
                        }

                        self.next_writeback = next_global.instr;
                        serviced_client = Some(next_client_id);
                    }
                }
                WritebackClient::L1D => {
                    debug_assert!(self.data_l1.is_some());
                    if let Some(ref mut data_l1) = self.data_l1 {
                        if let Some(fetch) = data_l1.next_access() {
                            log::trace!("l1 cache got ready access {} cycle={}", &fetch, cycle);
                            self.next_writeback = fetch.instr;
                            serviced_client = Some(next_client_id);
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
            log::debug!(
                "{} {:?} ({}) => next writeback={}",
                style("load store unit writeback serviced client").magenta(),
                WritebackClient::from_repr(serviced),
                serviced,
                crate::Optional(self.next_writeback.as_ref()),
            );
        }
    }

    #[must_use]
    // #[inline]
    fn shared_cycle(
        &mut self,
        _warps: &mut [warp::Warp],
        stall_kind: &mut MemStageStallKind,
        kind: &mut MemStageAccessKind,
        _cycle: u64,
    ) -> bool {
        let Some(dispatch_instr) = &mut self.inner.dispatch_reg else {
            return true;
        };
        log::debug!("shared cycle for instruction: {}", &dispatch_instr);

        if dispatch_instr.memory_space != Some(MemorySpace::Shared) {
            // shared cycle is done
            return true;
        }

        if dispatch_instr.active_thread_count() == 0 {
            // shared cycle is done
            return true;
        }

        if dispatch_instr.dispatch_delay_cycles > 0 {
            if let Some(ref mut l1_cache) = self.data_l1 {
                let stats = l1_cache.per_kernel_stats_mut();
                let kernel_stats = stats.get_mut(Some(dispatch_instr.kernel_launch_id));
                kernel_stats.num_shared_mem_bank_accesses += 1;
            }
        }

        dispatch_instr.dispatch_delay_cycles =
            dispatch_instr.dispatch_delay_cycles.saturating_sub(1);
        let has_stall = dispatch_instr.dispatch_delay_cycles > 0;
        if has_stall {
            *kind = MemStageAccessKind::S_MEM;
            *stall_kind = MemStageStallKind::BK_CONF;
            if let Some(ref mut l1_cache) = self.data_l1 {
                let stats = l1_cache.per_kernel_stats_mut();
                let kernel_stats = stats.get_mut(Some(dispatch_instr.kernel_launch_id));
                kernel_stats.num_shared_mem_bank_conflicts += 1;
            }
        } else {
            *stall_kind = MemStageStallKind::NO_RC_FAIL;
        }
        !has_stall
    }

    #[allow(clippy::unused_self)]
    #[must_use]
    // #[inline]
    fn constant_cycle(
        &mut self,
        _warps: &mut [warp::Warp],
        _rc_fail: &mut MemStageStallKind,
        _kind: &mut MemStageAccessKind,
        _cycle: u64,
    ) -> bool {
        true
    }

    #[allow(clippy::unused_self)]
    #[must_use]
    // #[inline]
    fn texture_cycle(
        &mut self,
        _warps: &mut [warp::Warp],
        _rc_fail: &mut MemStageStallKind,
        _kind: &mut MemStageAccessKind,
        _cycle: u64,
    ) -> bool {
        true
    }

    fn store_ack(&self, warps: &mut [warp::Warp], fetch: &mem_fetch::MemFetch) {
        debug_assert!(
            fetch.kind == mem_fetch::Kind::WRITE_ACK
                || (self.config.perfect_mem && fetch.is_write())
        );
        let warp = warps.get_mut(fetch.warp_id).unwrap();
        warp.num_outstanding_stores -= 1;
    }

    #[allow(dead_code)]
    fn process_cache_access(
        &mut self,
        warps: &mut [warp::Warp],
        _cache: (),
        _addr: address,
        instr: &mut WarpInstruction,
        events: &mut [cache::Event],
        fetch: &mem_fetch::MemFetch,
        status: cache::RequestStatus,
    ) -> MemStageStallKind {
        let mut stall_cond = MemStageStallKind::NO_RC_FAIL;
        let write_sent = cache::event::was_write_sent(events);
        let read_sent = cache::event::was_read_sent(events);
        if write_sent {
            let l1d_config = self.config.data_cache_l1.as_ref().unwrap();
            let inc_ack = if l1d_config.inner.mshr_kind == mshr::Kind::SECTOR_ASSOC {
                fetch.data_size() / mem_sub_partition::SECTOR_SIZE
            } else {
                1
            };

            let warp = warps.get_mut(instr.warp_id).unwrap();
            for _ in 0..inc_ack {
                warp.num_outstanding_stores += 1;
            }
        }
        if status == cache::RequestStatus::HIT {
            debug_assert!(!read_sent);
            instr.mem_access_queue.pop_back();
            if instr.is_load() {
                for out_reg in instr.outputs() {
                    let pending = self
                        .pending_writes
                        .get_mut(&instr.warp_id)
                        .and_then(|p| p.get_mut(out_reg))
                        .unwrap();
                    *pending -= 1;
                    log::trace!(
                        "warp {} register {}: decrement from {} to {}",
                        instr.warp_id,
                        out_reg,
                        *pending + 1,
                        *pending
                    );
                }
            }
        } else if status == cache::RequestStatus::RESERVATION_FAIL {
            stall_cond = MemStageStallKind::BK_CONF;
            debug_assert!(!read_sent);
            debug_assert!(!write_sent);
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

    fn l1_latency_queue_cycle(
        &mut self,
        scoreboard: &mut dyn Access<WarpInstruction>,
        warps: &mut [warp::Warp],
        stats: &mut stats::PerKernel,
        cycle: u64,
    ) {
        let l1_config = self.config.data_cache_l1.as_ref().unwrap();
        for bank_id in 0..l1_config.l1_banks {
            if let Some(fetch) = &self.l1_latency_queue[bank_id][0] {
                // let Some((ready_cycle, fetch)) = self.l1_latency_queue[bank_id].front() else {
                //     // no fetch available
                //     continue;
                // };
                //
                // if cycle <= *ready_cycle {
                //     // not ready yet
                //     continue;
                // }

                let mut events = Vec::new();

                let l1_cache = self.data_l1.as_mut().unwrap();
                let access_status =
                    l1_cache.access(fetch.addr(), fetch.clone(), &mut events, cycle);

                let write_sent = cache::event::was_write_sent(&events);
                let read_sent = cache::event::was_read_sent(&events);

                // [write sent={write_sent}, read sent={read_sent}, wr allocate sent={write_allocate_sent}]
                log::debug!(
                    "l1 cache access for warp={:<2} {} => {access_status:?} cycle={}",
                    fetch.warp_id,
                    &fetch,
                    cycle
                );

                let dec_ack = if l1_config.inner.mshr_kind.is_sector_assoc() {
                    fetch.data_size() / mem_sub_partition::SECTOR_SIZE
                } else {
                    1
                };

                if let Some(l1_access_callback) = &self.l1_access_callback {
                    l1_access_callback(cycle, &fetch, access_status);
                }

                if access_status == cache::RequestStatus::HIT {
                    debug_assert!(!read_sent);
                    // let (_, mut fetch) = self.l1_latency_queue[bank_id].pop_front().unwrap();
                    let mut fetch = self.l1_latency_queue[bank_id][0].take().unwrap();

                    // For write hit in WB policy
                    let instr = fetch.instr.as_mut().unwrap();
                    if instr.is_store() && !write_sent {
                        fetch.set_reply();

                        for _ in 0..dec_ack {
                            self.store_ack(warps, &fetch);
                        }
                    }

                    log::debug!("{}: {fetch}", style("PUSH TO L1 HIT LATENCY QUEUE").red());

                    if !self.inner.config.accelsim_compat {
                        let latency = if self.config.accelsim_compat {
                            0
                        } else {
                            l1_config.l1_hit_latency
                        };

                        self.l1_hit_latency_queue
                            .push_back((cycle + latency as u64, fetch));
                    } else {
                        let instr = fetch.instr.as_mut().unwrap();

                        if instr.is_load() {
                            let mut completed = false;
                            for out_reg in instr.outputs() {
                                let pending = self.pending_writes.get_mut(&instr.warp_id).unwrap();

                                let still_pending = pending.get_mut(out_reg).unwrap();
                                debug_assert!(*still_pending > 0);
                                *still_pending -= 1;
                                log::trace!(
                                    "warp {} register {}: decrement from {} to {}",
                                    instr.warp_id,
                                    out_reg,
                                    *still_pending + 1,
                                    *still_pending
                                );

                                if *still_pending == 0 {
                                    pending.remove(out_reg);
                                    log::trace!("l1 latency queue release registers");
                                    scoreboard.release(instr.warp_id, *out_reg);
                                    completed = true;
                                }
                            }
                            if completed {
                                crate::warp_inst_complete(instr, &mut *stats);
                            }
                        }
                    }
                } else if access_status.is_reservation_fail() {
                    debug_assert!(!read_sent);
                    debug_assert!(!write_sent);
                } else {
                    debug_assert!(matches!(
                        access_status,
                        cache::RequestStatus::MISS | cache::RequestStatus::HIT_RESERVED
                    ));
                    let mut fetch = self.l1_latency_queue[bank_id][0].take().unwrap();
                    let instr = fetch.instr.as_ref().unwrap();

                    let write_allocate_policy = l1_config.inner.write_allocate_policy;
                    let write_policy = l1_config.inner.write_policy;
                    let should_fetch = write_allocate_policy.is_fetch_on_write()
                        || write_allocate_policy.is_lazy_fetch_on_read();

                    let write_allocate_sent = cache::event::was_writeallocate_sent(&events);

                    if !write_policy.is_write_through()
                        && instr.is_store()
                        && should_fetch
                        && !write_allocate_sent
                    {
                        fetch.set_reply();
                        for _ in 0..dec_ack {
                            self.store_ack(warps, &fetch);
                        }
                    }
                }
            }

            for stage in 0..l1_config.l1_latency - 1 {
                if self.l1_latency_queue[bank_id][stage].is_none() {
                    self.l1_latency_queue[bank_id][stage] =
                        self.l1_latency_queue[bank_id][stage + 1].take();
                }
            }
        }

        if self.inner.config.accelsim_compat {
            return;
        }

        // check for ready L1 hits
        for _ in 0..l1_config.l1_banks {
            match self.l1_hit_latency_queue.front() {
                Some((ready_cycle, _)) if cycle >= *ready_cycle => {
                    let (_, mut fetch) = self.l1_hit_latency_queue.pop_front().unwrap();
                    log::debug!("{}: {fetch}", style("POP FROM L1 HIT LATENCY QUEUE").red());

                    if let Some(l1_access_callback) = &self.l1_access_callback {
                        l1_access_callback(cycle, &fetch, cache::RequestStatus::HIT);
                    }

                    let instr = fetch.instr.as_mut().unwrap();

                    if instr.is_load() {
                        let mut completed = false;
                        for out_reg in instr.outputs() {
                            let pending = self.pending_writes.get_mut(&instr.warp_id).unwrap();

                            let still_pending = pending.get_mut(out_reg).unwrap();
                            debug_assert!(*still_pending > 0);
                            *still_pending -= 1;
                            log::trace!(
                                "warp {} register {}: decrement from {} to {}",
                                instr.warp_id,
                                out_reg,
                                *still_pending + 1,
                                *still_pending
                            );

                            if *still_pending == 0 {
                                pending.remove(out_reg);
                                log::trace!("l1 latency queue release registers");
                                scoreboard.release(instr.warp_id, *out_reg);
                                completed = true;
                            }
                        }
                        if completed {
                            crate::warp_inst_complete(instr, &mut *stats);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    #[must_use]
    pub fn pending_writes(&self, warp_id: usize, reg_id: u32) -> Option<usize> {
        let pending = self.pending_writes.get(&warp_id)?;
        let pending = pending.get(&reg_id)?;
        Some(*pending)
    }

    pub fn pending_writes_mut(&mut self, warp_id: usize, reg_id: u32) -> &mut usize {
        let pending = self.pending_writes.entry(warp_id).or_default();
        pending.entry(reg_id).or_default()
    }
}

impl<MC> fu::SimdFunctionUnit for LoadStoreUnit<MC>
where
    MC: crate::mcu::MemoryController,
{
    fn active_lanes_in_pipeline(&self) -> usize {
        let active = self.inner.active_lanes_in_pipeline();
        debug_assert!(active <= self.config.warp_size);
        active
    }

    fn id(&self) -> &str {
        &self.inner.name
    }

    fn issue_port(&self) -> PipelineStage {
        PipelineStage::OC_EX_MEM
    }

    fn result_port(&self) -> Option<PipelineStage> {
        None
    }

    fn pipeline(&self) -> &Vec<Option<WarpInstruction>> {
        &self.inner.pipeline_reg
    }

    fn occupied(&self) -> &fu::OccupiedSlots {
        &self.inner.occupied
    }

    fn issue(&mut self, instr: WarpInstruction, stats: &mut stats::PerKernel) {
        // record how many pending register writes/memory accesses there are for this
        // instruction
        if instr.is_load() && instr.memory_space != Some(MemorySpace::Shared) {
            let warp_id = instr.warp_id;
            let num_accesses = instr.mem_access_queue.len();
            for out_reg in instr.outputs() {
                let still_pending = self.pending_writes_mut(warp_id, *out_reg);
                *still_pending += num_accesses;
                log::trace!(
                    "warp {} register {}: increment from {} to {}",
                    warp_id,
                    *out_reg,
                    *still_pending - num_accesses,
                    *still_pending
                );
            }
        }

        if let Some(mem_space) = instr.memory_space {
            let kernel_stats = stats.get_mut(Some(instr.kernel_launch_id));
            let active_count = instr.active_thread_count() as u64;
            kernel_stats
                .instructions
                .inc(None, mem_space, instr.is_store(), active_count);
        }

        self.inner.issue(instr);
    }

    fn clock_multiplier(&self) -> usize {
        1
    }

    fn can_issue(&self, instr: &WarpInstruction) -> bool {
        use crate::opcodes::ArchOp;
        match instr.opcode.category {
            ArchOp::LOAD_OP
            | ArchOp::TENSOR_CORE_LOAD_OP
            | ArchOp::STORE_OP
            | ArchOp::TENSOR_CORE_STORE_OP
            | ArchOp::MEMORY_BARRIER_OP => self.inner.dispatch_reg.is_none(),
            _ => false,
        }
    }

    fn is_issue_partitioned(&self) -> bool {
        // load store unit issue is not partitioned
        false
    }

    fn issue_reg_id(&self) -> usize {
        self.inner.issue_reg_id
    }

    fn stallable(&self) -> bool {
        // load store unit is stallable
        true
    }

    fn cycle(
        &mut self,
        operand_collector: &mut dyn OperandCollector,
        scoreboard: &mut dyn Access<WarpInstruction>,
        warps: &mut [warp::Warp],
        stats: &mut stats::PerKernel,
        mem_port: &mut dyn ic::Connection<ic::Packet<mem_fetch::MemFetch>>,
        _result_port: Option<&mut register_set::RegisterSet>,
        cycle: u64,
    ) {
        log::debug!(
            "fu[{:03}] {:<10} cycle={:03}: \tpipeline={:?} ({}/{} active) \tresponse fifo={:?}",
            self.inner.id,
            self.inner.name,
            cycle,
            self.inner
                .pipeline_reg
                .iter()
                .map(|reg| reg.as_ref().map(std::string::ToString::to_string))
                .collect::<Vec<_>>(),
            self.inner.num_active_instr_in_pipeline(),
            self.inner.pipeline_reg.len(),
            self.response_queue
                .lock()
                .iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>(),
        );

        self.writeback(operand_collector, scoreboard, warps, stats, cycle);

        let simd_unit = &mut self.inner;
        debug_assert!(simd_unit.pipeline_depth > 0);
        for stage in 0..(simd_unit.pipeline_depth - 1) {
            let current = stage + 1;
            let next = stage;

            if simd_unit.pipeline_reg[current].is_some() && simd_unit.pipeline_reg[next].is_none() {
                register_set::move_warp(
                    simd_unit.pipeline_reg[current].take(),
                    &mut simd_unit.pipeline_reg[next],
                    // format!(
                    //     "load store unit: move warp from stage {} to {}",
                    //     stage + 1,
                    //     stage,
                    // ),
                );
            } else {
                log::trace!("LdstUnit: skip moving {} to {}", stage + 1, stage);
            }
        }

        // if let Some(fetch) = self.response_queue.front() {
        {
            let mut response_queue_lock = self.response_queue.lock();

            if let Some(ic::Packet { fetch, .. }) = response_queue_lock.first() {
                match fetch.access_kind() {
                    AccessKind::TEXTURE_ACC_R => {
                        todo!("ldst unit: tex access");
                        // if self.texture_l1.has_free_fill_port() {
                        //     self.texture_l1.fill(&fetch);
                        //     // self.response_queue.fill(mem_fetch);
                        //     self.response_queue.pop_front();
                        // }
                    }
                    AccessKind::CONST_ACC_R => {
                        todo!("ldst unit: const access");
                        // if self.const_l1.has_free_fill_port() {
                        //     // fetch.set_status(IN_SHADER_FETCHED)
                        //     self.const_l1.fill(&fetch);
                        //     // self.response_queue.fill(mem_fetch);
                        //     self.response_queue.pop_front();
                        // }
                    }
                    _ => {
                        if fetch.kind == mem_fetch::Kind::WRITE_ACK
                            || (self.config.perfect_mem && fetch.is_write())
                        {
                            self.store_ack(warps, fetch);
                            // self.response_queue.pop_front();
                            response_queue_lock.dequeue();
                        } else {
                            // L1 cache is write evict:
                            // allocate line on load miss only
                            debug_assert!(!fetch.is_write());
                            let mut bypass_l1 = false;

                            let cache_op = fetch.instr.as_ref().and_then(|i| i.cache_operator);
                            if self.data_l1.is_none() || cache_op == Some(CacheOperator::Global) {
                                bypass_l1 = true;
                            } else if fetch.access_kind().is_global()
                                && self.config.global_mem_skip_l1_data_cache
                            {
                                bypass_l1 = true;
                            }

                            if bypass_l1 {
                                if self.next_global.is_none() {
                                    // let mut fetch = self.response_queue.pop_front().unwrap();
                                    let ic::Packet { mut fetch, .. } =
                                        response_queue_lock.dequeue().unwrap();
                                    fetch.set_status(mem_fetch::Status::IN_SHADER_FETCHED, 0);
                                    self.next_global = Some(fetch);
                                }
                            } else {
                                let l1d = self.data_l1.as_mut().unwrap();
                                if l1d.has_free_fill_port() {
                                    // let fetch = self.response_queue.pop_front().unwrap();
                                    let ic::Packet { fetch, time } =
                                        response_queue_lock.dequeue().unwrap();
                                    // eagerly release the lock
                                    drop(response_queue_lock);
                                    l1d.fill(fetch, time);
                                } else {
                                    log::trace!(
                                        "cannot fill L1 data cache with {}: no free fill port",
                                        fetch
                                    );
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
            data_l1.cycle(mem_port, cycle);

            let cache_config = self.config.data_cache_l1.as_ref().unwrap();
            debug_assert!(cache_config.l1_latency > 0);
            crate::timeit!(
                "core::execute::l1_latency_queue_cycle",
                self.l1_latency_queue_cycle(scoreboard, warps, stats, cycle)
            );
        }

        let mut stall_kind = MemStageStallKind::NO_RC_FAIL;
        let mut access_kind = MemStageAccessKind::C_MEM;
        let mut done = true;
        done &= self.shared_cycle(warps, &mut stall_kind, &mut access_kind, cycle);
        done &= self.constant_cycle(warps, &mut stall_kind, &mut access_kind, cycle);
        done &= self.texture_cycle(warps, &mut stall_kind, &mut access_kind, cycle);
        done &= self.memory_cycle(warps, mem_port, &mut stall_kind, &mut access_kind, cycle);

        if !done {
            // log stall types and return
            debug_assert_ne!(stall_kind, MemStageStallKind::NO_RC_FAIL);
            // num_stall_scheduler_mem += 1;
            // m_stats->gpu_stall_shd_mem_breakdown[type][rc_fail]++;
            return;
        }

        let simd_unit = &mut self.inner;
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
                        // let msg = format!(
                        //     "load store unit: move {:?} from dispatch register to pipeline[{}]",
                        //     dispatch_reg.as_ref().map(ToString::to_string),
                        //     pipe_slot_idx,
                        // );
                        register_set::move_warp(dispatch_reg, pipe_slot);
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
                    }

                    let mut dispatch_reg = simd_unit.dispatch_reg.take().unwrap();

                    if !has_pending_requests {
                        crate::warp_inst_complete(&mut dispatch_reg, &mut *stats);

                        scoreboard.release_all(&dispatch_reg);
                    }
                    let warp = warps.get_mut(warp_id).unwrap();
                    warp.num_instr_in_pipeline -= 1;
                    simd_unit.dispatch_reg = None;
                }
            } else {
                // stores exit pipeline here
                let warp = warps.get_mut(warp_id).unwrap();
                warp.num_instr_in_pipeline -= 1;
                let mut dispatch_reg = simd_unit.dispatch_reg.take().unwrap();

                // check for deadlocks due to scoreboard:
                //
                // make sure stores do not use destination registers
                assert_eq!(dispatch_reg.outputs().count(), 0);
                crate::warp_inst_complete(&mut dispatch_reg, &mut *stats);
            }
        }
    }
}
