use crate::sync::{Arc, Mutex, RwLock};
use crate::{
    address, cache, config, func_unit as fu,
    instruction::{CacheOperator, MemorySpace, WarpInstruction},
    interconn as ic, mcu, mem_fetch, mem_sub_partition, mshr, operand_collector as opcoll,
    register_set::{self},
    scoreboard::{Access, Scoreboard},
    warp,
};
use utils::box_slice;

use console::style;
use mem_fetch::{access::Kind as AccessKind, MemFetch};
use std::collections::{HashMap, VecDeque};
use strum::EnumCount;

#[allow(clippy::module_name_repetitions)]
pub struct LoadStoreUnit {
    core_id: usize,
    cluster_id: usize,
    next_writeback: Option<WarpInstruction>,
    pub response_fifo: VecDeque<MemFetch>,
    warps: Vec<warp::Ref>,
    pub data_l1: Option<Box<dyn cache::Cache<stats::cache::PerKernel>>>,
    config: Arc<config::GPU>,
    mem_controller: Arc<dyn mcu::MemoryController>,
    pub stats: Arc<Mutex<stats::PerKernel>>,
    scoreboard: Arc<RwLock<Scoreboard>>,
    next_global: Option<MemFetch>,
    pub pending_writes: HashMap<usize, HashMap<u32, usize>>,
    // pub l1_latency_queue: Box<[VecDeque<Option<mem_fetch::MemFetch>>]>,
    pub l1_latency_queue: Box<[Box<[Option<mem_fetch::MemFetch>]>]>,
    pub mem_port: ic::Port<mem_fetch::MemFetch>,
    inner: fu::PipelinedSimdUnit,

    operand_collector: Arc<Mutex<opcoll::RegisterFileUnit>>,

    /// round-robin arbiter for writeback contention between L1T, L1C, shared
    writeback_arb: usize,
    num_writeback_clients: usize,
}

impl std::fmt::Display for LoadStoreUnit {
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

impl LoadStoreUnit {
    pub fn new<MC>(
        id: usize,
        core_id: usize,
        cluster_id: usize,
        warps: Vec<warp::Ref>,
        mem_port: ic::Port<mem_fetch::MemFetch>,
        operand_collector: Arc<Mutex<opcoll::RegisterFileUnit>>,
        scoreboard: Arc<RwLock<Scoreboard>>,
        config: Arc<config::GPU>,
        // mem_controller: Arc<dyn mcu::MemoryController>,
        mem_controller: MC,
        stats: Arc<Mutex<stats::PerKernel>>,
    ) -> Self
    where
        MC: mcu::MemoryController + Clone,
    {
        let pipeline_depth = config.shared_memory_latency;
        let inner = fu::PipelinedSimdUnit::new(
            id,
            "LdstUnit".to_string(),
            None,
            pipeline_depth,
            config.clone(),
            // 0,
        );
        debug_assert!(config.shared_memory_latency > 1);

        let mut l1_latency_queue = box_slice![box_slice![None; 0]; 0];

        let data_l1: Option<Box<dyn cache::Cache<stats::cache::PerKernel>>> =
            if let Some(l1_config) = &config.data_cache_l1 {
                // initialize latency queue
                debug_assert!(l1_config.l1_latency > 0);
                l1_latency_queue =
                    box_slice![box_slice![None; l1_config.l1_latency]; l1_config.l1_banks];

                // initialize l1 data cache
                let cache_stats = Arc::new(Mutex::new(stats::cache::PerKernel::default()));
                // let mem_controller = crate::mcu::MemoryControllerUnit::new(&config).unwrap();

                let cache_controller = cache::controller::pascal::L1DataCacheController::new(
                    cache::Config::from(l1_config.inner.as_ref()),
                    l1_config,
                );

                let mut data_cache: cache::data::Data<
                    MC,
                    // Arc<dyn mcu::MemoryController>,
                    cache::controller::pascal::L1DataCacheController,
                    stats::cache::PerKernel,
                > = cache::data::Builder {
                    // name: format!("ldst-unit-{cluster_id}-{core_id}-L1-DATA-CACHE"),
                    name: format!(
                        "ldst-unit-{cluster_id}-{core_id}-{}",
                        style("L1D-CACHE").green()
                    ),
                    core_id,
                    cluster_id,
                    stats: cache_stats,
                    config: Arc::clone(&config),
                    mem_controller: mem_controller.clone(),
                    // &(mem_controller as Arc<dyn mcu::MemoryController>),
                    // mem_controller: Arc::clone(&mem_controller),
                    cache_controller,
                    cache_config: Arc::clone(&l1_config.inner),
                    write_alloc_type: AccessKind::L1_WR_ALLOC_R,
                    write_back_type: AccessKind::L1_WRBK_ACC,
                }
                .build();
                data_cache.set_top_port(mem_port.clone());
                // let _: &dyn cache::Cache<stats::cache::PerKernel> = &data_cache;

                Some(Box::new(data_cache))
            } else {
                None
            };

        assert!(data_l1.is_some());

        Self {
            core_id,
            cluster_id,
            data_l1,
            warps,
            next_writeback: None,
            next_global: None,
            pending_writes: HashMap::new(),
            response_fifo: VecDeque::new(),
            mem_port,
            inner,
            config,
            mem_controller: Arc::new(mem_controller),
            stats,
            scoreboard,
            operand_collector,
            num_writeback_clients: WritebackClient::COUNT,
            writeback_arb: 0,
            l1_latency_queue,
        }
    }

    #[must_use]
    pub fn response_buffer_full(&self) -> bool {
        self.response_fifo.len() >= self.config.num_ldst_response_buffer_size
    }

    pub fn flush(&mut self) {
        if let Some(l1) = &mut self.data_l1 {
            l1.flush();
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

    pub fn writeback(&mut self, cycle: u64) {
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

            if self.operand_collector.try_lock().writeback(next_writeback) {
                let mut next_writeback = self.next_writeback.take().unwrap();

                let mut instr_completed = false;
                for out_reg in next_writeback.outputs() {
                    debug_assert!(*out_reg > 0);

                    if next_writeback.memory_space == Some(MemorySpace::Shared) {
                        // shared
                        self.scoreboard
                            .try_write()
                            .release(next_writeback.warp_id, *out_reg);
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
                            self.scoreboard
                                .write()
                                .release(next_writeback.warp_id, *out_reg);
                            instr_completed = true;
                        }
                    }
                }
                if instr_completed {
                    crate::warp_inst_complete(&mut next_writeback, &self.stats);
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
                            // pipe_reg.do_atomic();
                            // m_core->decrement_atomic_count(m_next_wb.warp_id(),
                            //                                m_next_wb.active_count());
                        }

                        self.warps[pipe_reg.warp_id]
                            .try_lock()
                            .num_instr_in_pipeline -= 1;
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
                            // m_core->decrement_atomic_count(
                            //     m_next_global->get_wid(),
                            //     m_next_global->get_access_warp_mask().count());
                        }

                        self.next_writeback = next_global.instr;
                        serviced_client = Some(next_client_id);
                    }
                }
                WritebackClient::L1D => {
                    assert!(self.data_l1.is_some());
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
            if let Some(ref l1_cache) = self.data_l1 {
                let mut stats = l1_cache.per_kernel_stats().lock();
                let kernel_stats = stats.get_mut(dispatch_instr.kernel_launch_id);
                kernel_stats.num_shared_mem_bank_accesses += 1;
            }
        }

        dispatch_instr.dispatch_delay_cycles =
            dispatch_instr.dispatch_delay_cycles.saturating_sub(1);
        let has_stall = dispatch_instr.dispatch_delay_cycles > 0;
        if has_stall {
            *kind = MemStageAccessKind::S_MEM;
            *stall_kind = MemStageStallKind::BK_CONF;
            if let Some(ref l1_cache) = self.data_l1 {
                let mut stats = l1_cache.per_kernel_stats().lock();
                let kernel_stats = stats.get_mut(dispatch_instr.kernel_launch_id);
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
        _rc_fail: &mut MemStageStallKind,
        _kind: &mut MemStageAccessKind,
        _cycle: u64,
    ) -> bool {
        true
    }

    // #[inline]
    fn memory_cycle(
        &mut self,
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
            // debug_assert_eq!(access.req_size_bytes, dispatch_instr.data_size);
            // let control_size = if dispatch_instr.is_store() {
            //     mem_fetch::WRITE_PACKET_SIZE
            // } else {
            //     mem_fetch::READ_PACKET_SIZE
            // };
            // let size = access.req_size_bytes + u32::from(control_size);
            // debug_assert_eq!(access.size(), size);

            // // if self.fetch_interconn.full(
            // if false
            // // if self.interconn_full(
            // // size,
            // // dispatch_instr.is_store() || dispatch_instr.is_atomic(),
            // {
            //     // stall_cond = MemStageStallKind::ICNT_RC_FAIL;
            //     panic!("interconn full");
            // } else {
            let mut mem_port = self.mem_port.lock();

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
                    self.warps[dispatch_instr.warp_id]
                        .try_lock()
                        .num_outstanding_stores += 1;
                }

                let instr = self.inner.dispatch_reg.as_mut().unwrap();
                let access = instr.mem_access_queue.pop_back().unwrap();

                let physical_addr = self
                    // .config
                    // .address_mapping()
                    .mem_controller
                    .to_physical_address(access.addr);
                let partition_addr = self
                    // .config
                    // .address_mapping()
                    .mem_controller
                    .memory_partition_address(access.addr);

                let fetch = mem_fetch::Builder {
                    instr: Some(instr.clone()),
                    access,
                    warp_id: instr.warp_id,
                    core_id: Some(self.core_id),
                    cluster_id: Some(self.cluster_id),
                    physical_addr,
                    partition_addr,
                }
                .build();

                log::debug!("memory cycle for instruction {} => send {}", &instr, fetch);

                mem_port.send(ic::Packet {
                    data: fetch,
                    time: cycle,
                });
            }
        } else {
            debug_assert_ne!(dispatch_instr.cache_operator, CacheOperator::UNDEFINED);
            stall_cond = self.process_memory_access_queue_l1cache(cycle);
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

    fn store_ack(&self, fetch: &mem_fetch::MemFetch) {
        debug_assert!(
            fetch.kind == mem_fetch::Kind::WRITE_ACK
                || (self.config.perfect_mem && fetch.is_write())
        );
        // let mut warp = self.warps[fetch.warp_id].try_borrow_mut().unwrap();
        let mut warp = self.warps[fetch.warp_id].try_lock();
        warp.num_outstanding_stores -= 1;
    }

    fn process_memory_access_queue_l1cache(&mut self, cycle: u64) -> MemStageStallKind {
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
            // We can handle at max l1_banks reqs per cycle
            for _bank in 0..l1d_config.l1_banks {
                let Some(access) = instr.mem_access_queue.back() else {
                    return MemStageStallKind::NO_RC_FAIL;
                };

                let bank_id = l1d_config.compute_set_bank(access.addr) as usize;
                debug_assert!(bank_id < l1d_config.l1_banks);

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
                        .map(Option::as_ref)
                        .map(crate::Optional)
                        .collect::<Vec<_>>(),
                );

                let slot_idx = l1d_config.l1_latency - 1;
                let slot = &mut self.l1_latency_queue[bank_id][l1d_config.l1_latency - 1];
                if slot.is_none() {
                    let is_store = instr.is_store();
                    let access = instr.mem_access_queue.pop_back().unwrap();

                    let physical_addr = self
                        // .config
                        // .address_mapping()
                        .mem_controller
                        .to_physical_address(access.addr);
                    let partition_addr = self
                        // .config
                        // .address_mapping()
                        .mem_controller
                        .memory_partition_address(access.addr);

                    let mut fetch = mem_fetch::Builder {
                        instr: Some(instr.clone()),
                        access,
                        warp_id: instr.warp_id,
                        core_id: Some(self.core_id),
                        cluster_id: Some(self.cluster_id),
                        physical_addr,
                        partition_addr,
                    }
                    .build();
                    println!(
                        "add fetch {:<35} rel addr={:<4} bank={:<2} slot={:<4} at cycle={:<4}",
                        fetch.to_string(),
                        fetch.relative_byte_addr(),
                        bank_id,
                        slot_idx,
                        cycle
                    );
                    fetch.inject_cycle = Some(cycle);

                    let data_size = fetch.data_size();
                    *slot = Some(fetch);

                    if is_store {
                        let inc_ack = if l1d_config.inner.mshr_kind == mshr::Kind::SECTOR_ASSOC {
                            data_size / mem_sub_partition::SECTOR_SIZE
                        } else {
                            1
                        };

                        let mut warp = self.warps[instr.warp_id].try_lock();
                        for _ in 0..inc_ack {
                            warp.num_outstanding_stores += 1;
                        }
                    }
                } else {
                    stall_cond = MemStageStallKind::BK_CONF;
                    if let Some(ref l1_cache) = self.data_l1 {
                        let mut stats = l1_cache.per_kernel_stats().lock();
                        if let Some(kernel_launch_id) = access.kernel_launch_id() {
                            let kernel_stats = stats.get_mut(kernel_launch_id);
                            kernel_stats.num_l1_cache_bank_conflicts += 1;
                        }
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
            let physical_addr = self
                // .config
                // .address_mapping()
                .mem_controller
                .to_physical_address(access.addr);
            let partition_addr = self
                // .config
                // .address_mapping()
                .mem_controller
                .memory_partition_address(access.addr);

            let fetch = mem_fetch::Builder {
                instr: Some(instr.clone()),
                access: access.clone(),
                // &self.config,
                warp_id: instr.warp_id,
                core_id: Some(self.core_id),
                cluster_id: Some(self.cluster_id),
                physical_addr,
                partition_addr,
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

    #[allow(dead_code)]
    fn process_cache_access(
        &mut self,
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

            // let mut warp = self.warps[instr.warp_id].try_borrow_mut().unwrap();
            let mut warp = self.warps[instr.warp_id].try_lock();
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

    fn l1_latency_queue_cycle(&mut self, cycle: u64) {
        let l1_config = self.config.data_cache_l1.as_ref().unwrap();
        for bank in 0..l1_config.l1_banks {
            if let Some(next_fetch) = &self.l1_latency_queue[bank][0] {
                let mut events = Vec::new();

                let l1_cache = self.data_l1.as_mut().unwrap();
                let access_status =
                    l1_cache.access(next_fetch.addr(), next_fetch.clone(), &mut events, cycle);

                let write_sent = cache::event::was_write_sent(&events);
                let read_sent = cache::event::was_read_sent(&events);
                let write_allocate_sent = cache::event::was_writeallocate_sent(&events);

                log::debug!("l1 cache access for warp={:<2} {} => {access_status:?} cycle={} [write sent={write_sent}, read sent={read_sent}, wr allocate sent={write_allocate_sent}]", next_fetch.warp_id, &next_fetch, cycle);

                let dec_ack = if l1_config.inner.mshr_kind == mshr::Kind::SECTOR_ASSOC {
                    next_fetch.data_size() / mem_sub_partition::SECTOR_SIZE
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
                                self.scoreboard.try_write().release(instr.warp_id, *out_reg);
                                completed = true;
                            }
                        }
                        if completed {
                            crate::warp_inst_complete(instr, &self.stats);
                        }
                    }

                    // For write hit in WB policy
                    if instr.is_store() && !write_sent {
                        next_fetch.set_reply();

                        for _ in 0..dec_ack {
                            self.store_ack(&next_fetch);
                        }
                    }

                    dbg!(
                        &next_fetch.relative_byte_addr(),
                        cycle - next_fetch.inject_cycle.unwrap()
                    );
                    // self.l1_hit_callback()
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
                        cache::config::WriteAllocatePolicy::FETCH_ON_WRITE
                            | cache::config::WriteAllocatePolicy::LAZY_FETCH_ON_READ
                    );
                    if l1_config.inner.write_policy != cache::config::WritePolicy::WRITE_THROUGH
                        && instr.is_store()
                        && should_fetch
                        && !write_allocate_sent
                    {
                        next_fetch.set_reply();
                        for _ in 0..dec_ack {
                            self.store_ack(&next_fetch);
                        }
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

impl fu::SimdFunctionUnit for LoadStoreUnit
// impl<I> fu::SimdFunctionUnit for LoadStoreUnit<I>
// where
//     I: ic::MemFetchInterface,
{
    fn active_lanes_in_pipeline(&self) -> usize {
        let active = self.inner.active_lanes_in_pipeline();
        debug_assert!(active <= self.config.warp_size);
        active
    }

    fn id(&self) -> &str {
        &self.inner.name
    }

    fn pipeline(&self) -> &Vec<Option<WarpInstruction>> {
        &self.inner.pipeline_reg
    }

    fn occupied(&self) -> &fu::OccupiedSlots {
        &self.inner.occupied
    }

    fn issue(&mut self, instr: WarpInstruction) {
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

        // m_core->mem_instruction_stats(*inst);
        if let Some(mem_space) = instr.memory_space {
            let mut stats = self.stats.lock();
            let kernel_stats = stats.get_mut(instr.kernel_launch_id);
            let active_count = instr.active_thread_count() as u64;
            kernel_stats
                .instructions
                .inc(None, mem_space, instr.is_store(), active_count);
        }

        // m_core->incmem_stat(m_core->get_config()->warp_size, 1);

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

    // fn issue_reg_id(&self) -> usize {
    //     todo!("load store unit: issue reg id");
    // }

    fn stallable(&self) -> bool {
        // load store unit is stallable
        true
    }
}

impl crate::engine::cycle::Component for LoadStoreUnit {
    fn cycle(&mut self, cycle: u64) {
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
            self.response_fifo
                .iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>(),
        );

        self.writeback(cycle);

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

        if let Some(fetch) = self.response_fifo.front() {
            match fetch.access_kind() {
                AccessKind::TEXTURE_ACC_R => {
                    todo!("ldst unit: tex access");
                    // if self.texture_l1.has_free_fill_port() {
                    //     self.texture_l1.fill(&fetch);
                    //     // self.response_fifo.fill(mem_fetch);
                    //     self.response_fifo.pop_front();
                    // }
                }
                AccessKind::CONST_ACC_R => {
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
                        self.store_ack(fetch);
                        self.response_fifo.pop_front();
                    } else {
                        // L1 cache is write evict:
                        // allocate line on load miss only
                        debug_assert!(!fetch.is_write());
                        let mut bypass_l1 = false;

                        let cache_op = fetch.instr.as_ref().map(|i| i.cache_operator);
                        if self.data_l1.is_none() || cache_op == Some(CacheOperator::GLOBAL) {
                            bypass_l1 = true;
                        } else if fetch.access_kind().is_global()
                            && self.config.global_mem_skip_l1_data_cache
                        {
                            bypass_l1 = true;
                        }

                        if bypass_l1 {
                            if self.next_global.is_none() {
                                let mut fetch = self.response_fifo.pop_front().unwrap();
                                fetch.set_status(mem_fetch::Status::IN_SHADER_FETCHED, 0);
                                self.next_global = Some(fetch);
                            }
                        } else {
                            let l1d = self.data_l1.as_mut().unwrap();
                            if l1d.has_free_fill_port() {
                                let fetch = self.response_fifo.pop_front().unwrap();
                                l1d.fill(fetch, cycle);
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

        // self.texture_l1.cycle();
        // self.const_l1.cycle();
        if let Some(data_l1) = &mut self.data_l1 {
            data_l1.cycle(cycle);
            let cache_config = self.config.data_cache_l1.as_ref().unwrap();
            assert!(cache_config.l1_latency > 1);
            self.l1_latency_queue_cycle(cycle);
        }

        let mut stall_kind = MemStageStallKind::NO_RC_FAIL;
        let mut access_kind = MemStageAccessKind::C_MEM;
        let mut done = true;
        done &= self.shared_cycle(&mut stall_kind, &mut access_kind, cycle);
        done &= self.constant_cycle(&mut stall_kind, &mut access_kind, cycle);
        done &= self.texture_cycle(&mut stall_kind, &mut access_kind, cycle);
        done &= self.memory_cycle(&mut stall_kind, &mut access_kind, cycle);

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
                        crate::warp_inst_complete(&mut dispatch_reg, &self.stats);

                        self.scoreboard.try_write().release_all(&dispatch_reg);
                    }
                    self.warps[warp_id].try_lock().num_instr_in_pipeline -= 1;
                    simd_unit.dispatch_reg = None;
                }
            } else {
                // stores exit pipeline here
                self.warps[warp_id].try_lock().num_instr_in_pipeline -= 1;
                let mut dispatch_reg = simd_unit.dispatch_reg.take().unwrap();
                crate::warp_inst_complete(&mut dispatch_reg, &self.stats);
            }
        }
    }
}
