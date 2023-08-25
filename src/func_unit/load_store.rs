use crate::sync::{Arc, Mutex, RwLock};
use crate::{
    address, cache, config, func_unit as fu,
    instruction::{CacheOperator, MemorySpace, WarpInstruction},
    interconn as ic, mem_fetch,
    mem_fetch::MemFetch,
    mem_sub_partition, mshr, operand_collector as opcoll,
    register_set::{self},
    scoreboard::{Access, Scoreboard},
    warp,
};
use console::style;
use std::collections::{HashMap, VecDeque};
use strum::EnumCount;

fn new_mem_fetch(
    access: mem_fetch::MemAccess,
    instr: WarpInstruction,
    config: &config::GPU,
    core_id: usize,
    cluster_id: usize,
) -> mem_fetch::MemFetch {
    let warp_id = instr.warp_id;
    let control_size = access.control_size();
    mem_fetch::MemFetch::new(
        Some(instr),
        access,
        config,
        control_size,
        warp_id,
        core_id,
        cluster_id,
    )
}

#[allow(clippy::module_name_repetitions)]
// pub struct LoadStoreUnit<I> {
pub struct LoadStoreUnit {
    core_id: usize,
    cluster_id: usize,
    next_writeback: Option<WarpInstruction>,
    response_fifo: VecDeque<MemFetch>,
    warps: Vec<warp::Ref>,
    pub data_l1: Option<Box<dyn cache::Cache>>,
    config: Arc<config::GPU>,
    pub stats: Arc<Mutex<stats::Stats>>,
    scoreboard: Arc<RwLock<Scoreboard>>,
    next_global: Option<MemFetch>,
    pub pending_writes: HashMap<usize, HashMap<u32, usize>>,
    l1_latency_queue: Vec<Vec<Option<mem_fetch::MemFetch>>>,
    // #[allow(dead_code)]
    // interconn: Arc<dyn ic::Interconnect<ic::Packet<mem_fetch::MemFetch>>>,
    // fetch_interconn: Arc<I>,
    pub mem_port: ic::Port<mem_fetch::MemFetch>,
    inner: fu::PipelinedSimdUnit,

    operand_collector: Arc<Mutex<opcoll::RegisterFileUnit>>,

    /// round-robin arbiter for writeback contention between L1T, L1C, shared
    writeback_arb: usize,
    num_writeback_clients: usize,
}

// impl<I> std::fmt::Display for LoadStoreUnit<I> {
impl std::fmt::Display for LoadStoreUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner.name)
    }
}

// impl<I> std::fmt::Debug for LoadStoreUnit<I> {
impl std::fmt::Debug for LoadStoreUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(&self.inner.name)
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
    // N_MEM_STAGE_ACCESS_TYPE
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

impl LoadStoreUnit
// impl<I> LoadStoreUnit<I>
// where
//     I: ic::MemFetchInterface + 'static,
{
    pub fn new(
        id: usize,
        core_id: usize,
        cluster_id: usize,
        warps: Vec<warp::Ref>,
        // interconn: Arc<dyn ic::Interconnect<ic::Packet<mem_fetch::MemFetch>>>,
        // fetch_interconn: Arc<I>,
        mem_port: ic::Port<mem_fetch::MemFetch>,
        operand_collector: Arc<Mutex<opcoll::RegisterFileUnit>>,
        scoreboard: Arc<RwLock<Scoreboard>>,
        config: Arc<config::GPU>,
        stats: Arc<Mutex<stats::Stats>>,
    ) -> Self {
        let pipeline_depth = config.shared_memory_latency;
        let inner = fu::PipelinedSimdUnit::new(
            id,
            "LdstUnit".to_string(),
            None,
            pipeline_depth,
            config.clone(),
            0,
        );
        debug_assert!(config.shared_memory_latency > 1);

        let mut l1_latency_queue: Vec<Vec<Option<mem_fetch::MemFetch>>> = Vec::new();
        let data_l1: Option<Box<dyn cache::Cache + 'static>> =
            if let Some(l1_config) = &config.data_cache_l1 {
                // initialize latency queue
                debug_assert!(l1_config.l1_latency > 0);
                l1_latency_queue = (0..l1_config.l1_banks)
                    .map(|_bank| vec![None; l1_config.l1_latency])
                    .collect();

                // initialize l1 data cache
                let cache_stats = Arc::new(Mutex::new(stats::Cache::default()));
                let mut data_cache = cache::Data::new(
                    format!("ldst-unit-{cluster_id}-{core_id}-L1-DATA-CACHE"),
                    core_id,
                    cluster_id,
                    // Arc::clone(&fetch_interconn),
                    cache_stats,
                    Arc::clone(&config),
                    Arc::clone(&l1_config.inner),
                    mem_fetch::AccessKind::L1_WR_ALLOC_R,
                    mem_fetch::AccessKind::L1_WRBK_ACC,
                );
                data_cache.set_top_port(mem_port.clone());
                Some(Box::new(data_cache))
            } else {
                None
            };

        Self {
            core_id,
            cluster_id,
            data_l1,
            warps,
            next_writeback: None,
            next_global: None,
            pending_writes: HashMap::new(),
            response_fifo: VecDeque::new(),
            // interconn,
            // fetch_interconn,
            mem_port,
            inner,
            config,
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
        if let Some(_l1) = &mut self.data_l1 {
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
                            .write()
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
                                "ldst unit writeback: has global {:?} ({})",
                                &next_global
                                    .instr
                                    .as_ref()
                                    .map(std::string::ToString::to_string),
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
                    if let Some(ref mut data_l1) = self.data_l1 {
                        if let Some(fetch) = data_l1.next_access() {
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
                "{} {:?} ({}) => next writeback={:?}",
                style("load store unit writeback serviced client").magenta(),
                WritebackClient::from_repr(serviced),
                serviced,
                self.next_writeback.as_ref().map(ToString::to_string),
            );
        }
    }

    #[must_use]
    #[inline]
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
            #[cfg(feature = "stats")]
            {
                let _stats = self.stats.lock();
                // stats.num_shared_mem_bank_access[self.core_id] += 1;
            }
        }

        // dispatch_instr.dec_dispatch_delay();
        dispatch_instr.dispatch_delay_cycles =
            dispatch_instr.dispatch_delay_cycles.saturating_sub(1);
        let has_stall = dispatch_instr.dispatch_delay_cycles > 0;
        if has_stall {
            *kind = MemStageAccessKind::S_MEM;
            *stall_kind = MemStageStallKind::BK_CONF;
        } else {
            *stall_kind = MemStageStallKind::NO_RC_FAIL;
        }
        !has_stall
    }

    #[allow(clippy::unused_self)]
    #[must_use]
    #[inline]
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
    #[inline]
    fn texture_cycle(
        &mut self,
        _rc_fail: &mut MemStageStallKind,
        _kind: &mut MemStageAccessKind,
        _cycle: u64,
    ) -> bool {
        true
    }

    // fn interconn_full(&self) -> bool {
    //     let size = self
    //         .interconn_port
    //         .iter()
    //         // .map(|(_dest, _fetch, _packet_size, size)| {
    //         .map(|(_dest, _fetch, size)| {
    //             // dispatch_instr.is_store() || dispatch_instr.is_atomic(),
    //             *size
    //         })
    //         .sum();
    //     // let size: u32 = self
    //     //     .interconn_port
    //     //     .iter()
    //     //     .filter_map(|fetch| fetch.instr)
    //     //     .map(|instr| {
    //     //         let control_size = if instr.is_store() {
    //     //             mem_fetch::WRITE_PACKET_SIZE
    //     //         } else {
    //     //             mem_fetch::READ_PACKET_SIZE
    //     //         };
    //     //         let size = access.req_size_bytes + u32::from(control_size);
    //     //
    //     //         let is_write = instr.is_store() || instr.is_atomic();
    //     //         let request_size = if is_write {
    //     //             size
    //     //         } else {
    //     //             u32::from(mem_fetch::READ_PACKET_SIZE)
    //     //         };
    //     //         request_size
    //     //     })
    //     //     .sum();
    //     // false
    //     !self.interconn.has_buffer(self.cluster_id, size)
    // }
    //
    // fn interconn_push(&mut self, mut fetch: mem_fetch::MemFetch, time: u64) {
    //     {
    //         let mut stats = self.statslock();
    //         let access_kind = *fetch.access_kind();
    //         debug_assert_eq!(fetch.is_write(), access_kind.is_write());
    //         stats.accesses.inc(access_kind, 1);
    //     }
    //
    //     let dest_sub_partition_id = fetch.sub_partition_id();
    //     let mem_dest = self.config.mem_id_to_device_id(dest_sub_partition_id);
    //
    //     log::debug!(
    //         "cluster {} icnt_inject_request_packet({}) dest sub partition id={} dest mem node={}",
    //         self.cluster_id,
    //         fetch,
    //         dest_sub_partition_id,
    //         mem_dest
    //     );
    //
    //     // The packet size varies depending on the type of request:
    //     // - For write request and atomic request, packet contains the data
    //     // - For read request (i.e. not write nor atomic), packet only has control metadata
    //     let packet_size = if !fetch.is_write() && !fetch.is_atomic() {
    //         fetch.control_size()
    //     } else {
    //         // todo: is that correct now?
    //         fetch.size()
    //         // fetch.data_size
    //     };
    //
    //     // let instr = fetch.instr.as_ref().unwrap();
    //     // let request_size = if instr.is_store() || instr.is_atomic() {
    //     //     packet_size
    //     // } else {
    //     //     u32::from(mem_fetch::READ_PACKET_SIZE)
    //     // };
    //
    //     // {
    //     //     let control_size = if instr.is_store() {
    //     //         mem_fetch::WRITE_PACKET_SIZE
    //     //     } else {
    //     //         mem_fetch::READ_PACKET_SIZE
    //     //     };
    //     //     let size = fetch.access.req_size_bytes + u32::from(control_size);
    //     //     // debug_assert_eq!(fetch.access.size(), size);
    //     //     assert_eq!(packet_size, size);
    //     // }
    //
    //     // m_stats->m_outgoing_traffic_stats->record_traffic(mf, packet_size);
    //     fetch.status = mem_fetch::Status::IN_ICNT_TO_MEM;
    //
    //     // if let Packet::Fetch(fetch) = packet {
    //     fetch.pushed_cycle = Some(time);
    //
    //     // self.interconn_queue
    //     //     .push_back((mem_dest, fetch, packet_size));
    //     self.interconn.push(
    //         self.cluster_id,
    //         mem_dest,
    //         super::Packet::Fetch(fetch),
    //         packet_size,
    //     );
    //     // self.interconn_port
    //     //     .push_back((mem_dest, fetch, packet_size));
    // }

    #[inline]
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

        let Some(access) = dispatch_instr.mem_access_queue.back() else {
            return true;
        };

        log::debug!(
            "memory cycle for instruction {} => access: {} (bypass l1={})",
            &dispatch_instr,
            access,
            bypass_l1
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

                let fetch = new_mem_fetch(
                    access,
                    instr.clone(),
                    &self.config,
                    self.core_id,
                    self.cluster_id,
                );

                // self.interconn_push(fetch, time);
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
            return stall_cond;
        };

        let Some(access) = instr.mem_access_queue.back() else {
            return stall_cond;
        };
        let dbg_access = access.clone();

        let l1d_config = self.config.data_cache_l1.as_ref().unwrap();

        if l1d_config.l1_latency > 0 {
            // We can handle at max l1_banks reqs per cycle
            for _bank in 0..l1d_config.l1_banks {
                let Some(access) = instr.mem_access_queue.back() else {
                    break;
                };

                let bank_id = l1d_config.compute_set_bank(access.addr) as usize;
                debug_assert!(bank_id < l1d_config.l1_banks);

                log::trace!(
                    "computed bank id {} for access {} (access queue={:?})",
                    bank_id,
                    access,
                    &instr
                        .mem_access_queue
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                );

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
                    let data_size = fetch.data_size();
                    *slot = Some(fetch);

                    if is_store {
                        let inc_ack = if l1d_config.inner.mshr_kind == mshr::Kind::SECTOR_ASSOC {
                            data_size / mem_sub_partition::SECTOR_SIZE
                        } else {
                            1
                        };

                        // let mut warp = self.warps[instr.warp_id].try_borrow_mut().unwrap();
                        let mut warp = self.warps[instr.warp_id].try_lock();
                        for _ in 0..inc_ack {
                            warp.num_outstanding_stores += 1;
                        }
                    }
                } else {
                    stall_cond = MemStageStallKind::BK_CONF;
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
            let new_instr = instr.clone();
            let fetch = new_mem_fetch(
                access.clone(),
                new_instr,
                &self.config,
                self.core_id,
                self.cluster_id,
            );
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

                            if *still_pending > 0 {
                                pending.remove(out_reg);
                                log::trace!("l1 latency queue release registers");
                                self.scoreboard.write().release(instr.warp_id, *out_reg);
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
            #[cfg(feature = "stats")]
            {
                let mut stats = self.stats.lock();
                let active_count = instr.active_thread_count() as u64;
                stats
                    .instructions
                    .inc(mem_space, instr.is_store(), active_count);
            }
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

    fn issue_reg_id(&self) -> usize {
        todo!("load store unit: issue reg id");
    }

    fn stallable(&self) -> bool {
        // load store unit is stallable
        true
    }
}

impl crate::engine::cycle::Component for LoadStoreUnit
// impl<I> crate::engine::cycle::Component for LoadStoreUnit<I>
// where
//     I: ic::MemFetchInterface + 'static,
{
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
                                self.next_global = Some(fetch);
                            }
                        } else {
                            let l1d = self.data_l1.as_mut().unwrap();
                            if l1d.has_free_fill_port() {
                                let fetch = self.response_fifo.pop_front().unwrap();
                                l1d.fill(fetch, cycle);
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
            debug_assert_eq!(cache_config.l1_latency, 1);
            if cache_config.l1_latency > 0 {
                self.l1_latency_queue_cycle(cycle);
            }
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

                        self.scoreboard.write().release_all(&dispatch_reg);
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
