#![allow(
    clippy::upper_case_acronyms,
    non_camel_case_types,
    clippy::too_many_arguments,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]

pub mod allocation;
pub mod arbitration;
pub mod barrier;
pub mod cache;
pub mod cluster;
pub mod config;
pub mod core;
pub mod deadlock;
pub mod dram;
pub mod fifo;
pub mod func_unit;
pub mod instruction;
pub mod interconn;
pub mod kernel;
pub mod kernel_manager;
pub mod mcu;
pub mod mem_fetch;
pub mod mem_partition_unit;
pub mod mem_sub_partition;
pub mod mshr;
pub mod opcodes;
pub mod operand_collector;
#[cfg(feature = "parallel")]
pub mod parallel;
pub mod register_set;
pub mod scheduler;
pub mod scoreboard;
pub mod streams;
pub mod sync;
pub mod tag_array;
pub mod trace;
pub mod warp;

#[cfg(test)]
pub mod testing;

use self::core::{warp_inst_complete, Core, PipelineStage};
use allocation::{Allocation, Allocations};
use cache::Cache;
use cluster::Cluster;
pub use exec;
use interconn as ic;
use kernel::Kernel;
use mem_partition_unit::MemoryPartitionUnit;
use mem_sub_partition::{MemorySubPartition, SECTOR_SIZE};
use trace_model::{Command, ToBitString};

use crate::sync::{atomic, Arc, Mutex, RwLock};
use bitvec::array::BitArray;
use color_eyre::eyre::{self};
use console::style;
use itertools::Itertools;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};

pub type address = u64;

pub const DEBUG_PRINT: bool = false;

/// Clock domains
#[derive(Debug, Clone, Copy, Hash, strum::EnumIter, PartialEq, Eq, PartialOrd, Ord)]
#[repr(usize)]
pub enum ClockDomain {
    CORE,
    L2,
    DRAM,
    ICNT,
}

pub type ClockMask = bitvec::BitArr!(for 8, in u8);

impl ClockDomain {
    pub fn iter() -> <ClockDomain as strum::IntoEnumIterator>::Iterator {
        <Self as strum::IntoEnumIterator>::iter()
    }
}

impl std::ops::BitOrAssign<ClockDomain> for u64 {
    fn bitor_assign(&mut self, rhs: ClockDomain) {
        *self |= rhs as u64;
    }
}

impl std::ops::BitAnd<ClockDomain> for u64 {
    type Output = Self;
    fn bitand(self, rhs: ClockDomain) -> Self::Output {
        self & (rhs as u64)
    }
}

/// Start address of the global heap on device.
///
/// Note that the precise value is not actually very important.
/// However, the goal is to split addresses into different memory spaces to avoid clashes.
/// Moreover, requests to zero addresses are treated as no request.
pub const GLOBAL_HEAP_START: u64 = 0xC0000000;

// The maximum number of concurrent threads per SM.
pub const MAX_THREADS_PER_SM: usize = 2048;

/// The maximum number of warps assigned to one SM.
pub const MAX_WARPS_PER_SM: usize = 64;

/// The maximum number of barriers per block.
pub const MAX_BARRIERS_PER_CTA: usize = 16;

/// The maximum number of warps per block.
pub const MAX_WARPS_PER_CTA: usize = 64;

/// The warp size.
///
/// A warp is a group of `n` SIMD lanes (threads).
pub const WARP_SIZE: usize = 32;

// TODO: remove

// Volta max shmem size is 96kB
pub const SHARED_MEM_SIZE_MAX: u64 = 96 * (1 << 10);
// Volta max local mem is 16kB
pub const LOCAL_MEM_SIZE_MAX: u64 = 16 * (1 << 10);

pub const MAX_STREAMING_MULTIPROCESSORS: usize = 80;

pub const TOTAL_LOCAL_MEM_PER_SM: u64 = MAX_THREADS_PER_SM as u64 * LOCAL_MEM_SIZE_MAX;
pub const TOTAL_SHARED_MEM: u64 = MAX_STREAMING_MULTIPROCESSORS as u64 * SHARED_MEM_SIZE_MAX;
pub const TOTAL_LOCAL_MEM: u64 =
    MAX_STREAMING_MULTIPROCESSORS as u64 * MAX_THREADS_PER_SM as u64 * LOCAL_MEM_SIZE_MAX;
pub const SHARED_GENERIC_START: u64 = GLOBAL_HEAP_START - TOTAL_SHARED_MEM;
pub const LOCAL_GENERIC_START: u64 = SHARED_GENERIC_START - TOTAL_LOCAL_MEM;
pub const STATIC_ALLOC_LIMIT: u64 = GLOBAL_HEAP_START - (TOTAL_LOCAL_MEM + TOTAL_SHARED_MEM);

/// Start of the program memory space
///
/// Note: should be distinct from other memory spaces.
pub const PROGRAM_MEM_START: usize = 0xF000_0000;

pub static PROGRAM_MEM_ALLOC: once_cell::sync::Lazy<Allocation> =
    once_cell::sync::Lazy::new(|| Allocation {
        name: Some("PROGRAM_MEM".to_string()),
        id: 0,
        start_addr: PROGRAM_MEM_START as address,
        end_addr: None,
    });

#[must_use]
pub fn is_debug() -> bool {
    #[cfg(all(feature = "debug_build", feature = "release_build"))]
    compile_error!(r#"both feature "debug_build" or "release_build" are set."#);

    #[cfg(feature = "debug_build")]
    return true;
    #[cfg(feature = "release_build")]
    return false;
    #[cfg(not(any(feature = "debug_build", feature = "release_build")))]
    compile_error!(r#"neither feature "debug_build" or "release_build" is set."#);
}

pub struct Optional<T>(Option<T>);

impl<'a, T> std::fmt::Display for Optional<&'a T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Some(ref value) => write!(f, "Some({value})"),
            None => write!(f, "None"),
        }
    }
}

impl<'a, T> std::fmt::Debug for Optional<&'a T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct TotalDuration {
    count: u128,
    dur: std::time::Duration,
}

impl TotalDuration {
    pub fn add(&mut self, dur: std::time::Duration) {
        self.count += 1;
        self.dur += dur;
    }

    #[must_use]
    pub fn total(&self) -> &std::time::Duration {
        &self.dur
    }

    #[must_use]
    pub fn count(&self) -> u128 {
        self.count
    }

    #[must_use]
    pub fn mean(&self) -> std::time::Duration {
        let nanos = u64::try_from(self.dur.as_nanos() / self.count).unwrap();
        std::time::Duration::from_nanos(nanos)
    }
}

pub static TIMINGS: once_cell::sync::Lazy<Mutex<HashMap<&'static str, TotalDuration>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(HashMap::new()));

pub mod wip_stats {
    use crate::sync::Mutex;

    #[derive(Debug)]
    pub struct WIPStats {
        pub issued_instructions: u64,
        pub executed_instructions: u64,
        pub warp_instructions: u64,
        pub num_warps: u64,
        pub warps_per_core: Vec<u64>,
    }

    impl Default for WIPStats {
        fn default() -> Self {
            Self {
                issued_instructions: 0,
                executed_instructions: 0,
                warp_instructions: 0,
                num_warps: 0,
                warps_per_core: vec![0; 20 * 8],
            }
        }
    }

    pub static WIP_STATS: once_cell::sync::Lazy<Mutex<WIPStats>> =
        once_cell::sync::Lazy::new(|| Mutex::new(WIPStats::default()));
}

#[macro_export]
macro_rules! timeit {
    ($name:expr, $call:expr) => {{
        #[cfg(feature = "timings")]
        {
            let start = std::time::Instant::now();
            let res = $call;
            let dur = start.elapsed();
            let mut timings = $crate::TIMINGS.lock();
            timings.entry($name).or_default().add(dur);
            drop(timings);
            res
        }
        #[cfg(not(feature = "timings"))]
        $call
    }};
    ($call:expr) => {{
        $crate::timeit!(stringify!($call), $call)
    }};
}

#[derive(Clone, Debug, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub struct DebugState {
    pub core_orders_per_cluster: Vec<VecDeque<usize>>,
    pub last_cluster_issue: usize,
    pub last_issued_kernel: usize,
    pub block_issue_next_core_per_cluster: Vec<usize>,
}

#[derive()]
pub struct Simulator<I, MC> {
    stats: stats::PerKernel,
    config: Arc<config::GPU>,
    mem_controller: Arc<MC>,

    mem_partition_units: Box<[mem_partition_unit::MemoryPartitionUnit<MC>]>,

    pub kernel_manager: kernel_manager::KernelManager,
    pub stream_manager: streams::StreamManager,
    pub clusters: Box<[Cluster<I, MC>]>,

    /// Queue with capacity of the kernel window size
    ///
    /// One or multiple kernels in the window can be launched given
    /// sufficient resources and different streams.
    kernel_launch_window_queue: VecDeque<Arc<dyn Kernel>>,

    #[allow(dead_code)]
    warp_instruction_unique_uid: Arc<atomic::AtomicU64>,

    interconn: Arc<I>,

    block_issuer: BlockIssuer<I, MC>,
    // todo refactor into a block issuer trait
    // last_cluster_issue: Arc<Mutex<usize>>,
    // pub cluster_issuers: Box<[cluster::ClusterIssuer<cluster::CoreIssuer<Core<I, MC>>>]>,
    allocations: Arc<allocation::Allocations>,

    pub trace: trace::Trace,

    /// simulation options
    cycle_limit: Option<u64>,
    log_after_cycle: Option<u64>,

    /// clock domain times
    core_time: f64,
    dram_time: f64,
    icnt_time: f64,
    l2_time: f64,
}

impl<I, MC> std::fmt::Debug for Simulator<I, MC> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Simulator").finish()
    }
}

pub struct MemSubPartitionIter<'a, MC> {
    partition_units: &'a [MemoryPartitionUnit<MC>],
    sub_partitions_per_partition: usize,
    global_sub_id: usize,
}

impl<'a, MC> Iterator for MemSubPartitionIter<'a, MC> {
    type Item = &'a MemorySubPartition<MC>;

    fn next(&mut self) -> Option<Self::Item> {
        let partition_id = self.global_sub_id / self.sub_partitions_per_partition;
        let sub_id = self.global_sub_id % self.sub_partitions_per_partition;
        if partition_id >= self.partition_units.len() {
            return None;
        }
        let partition = &self.partition_units[partition_id];
        if sub_id >= partition.sub_partitions.len() {
            return None;
        }
        self.global_sub_id += 1;
        Some(&partition.sub_partitions[sub_id])
    }
}

impl<I, MC> Simulator<I, MC>
where
    I: ic::Interconnect<ic::Packet<mem_fetch::MemFetch>>,
    MC: mcu::MemoryController,
{
    pub fn new(interconn: Arc<I>, mem_controller: Arc<MC>, config: Arc<config::GPU>) -> Self
    where
        MC: mcu::MemoryController,
    {
        let stats = stats::PerKernel::new(config.as_ref().into());

        // let mem_controller: Arc<dyn mcu::MemoryController> = if config.accelsim_compat {
        //     Arc::new(mcu::MemoryControllerUnit::new(&config).unwrap())
        // } else {
        //     Arc::new(mcu::PascalMemoryControllerUnit::new(&config).unwrap())
        // };

        let num_mem_units = config.num_memory_controllers;

        let mem_partition_units: Box<[_]> = (0..num_mem_units)
            .map(|partition_id| {
                let unit = mem_partition_unit::MemoryPartitionUnit::new(
                    partition_id,
                    Arc::clone(&config),
                    mem_controller.clone(),
                );
                unit
            })
            .collect();

        let kernel_manager = kernel_manager::KernelManager::new(config.clone());
        let allocations = Arc::new(Allocations::default());

        let warp_instruction_unique_uid = Arc::new(atomic::AtomicU64::new(0));
        let clusters: Box<[_]> = (0..config.num_simt_clusters)
            .map(|i| {
                let cluster = Cluster::new(
                    i,
                    &warp_instruction_unique_uid,
                    &allocations,
                    &interconn,
                    &config,
                    &mem_controller,
                );
                cluster
            })
            .collect();

        // : Vec<cluster::ClusterIssuer<_>>
        let cluster_issuers: Box<[_]> = (0..config.num_simt_clusters)
            .map(|i| {
                let cores: Box<[_]> = clusters[i]
                    .cores
                    .iter()
                    .map(|core| cluster::CoreIssuer { core: core.clone() })
                    .collect();
                let cluster = cluster::ClusterIssuer::new(cores);
                cluster
            })
            .collect();

        // this causes first launch to use simt cluster
        // let last_cluster_issue = Arc::new(Mutex::new(config.num_simt_clusters - 1));
        assert_eq!(config.num_simt_clusters, cluster_issuers.len());
        let block_issuer = BlockIssuer {
            last_cluster_issue: cluster_issuers.len() - 1,
            num_simt_clusters: cluster_issuers.len(),
            cluster_issuers,
        };

        assert!(config.max_threads_per_core.rem_euclid(config.warp_size) == 0);
        let trace = trace::Trace::new(config.clone());

        let cycle_limit: Option<u64> = std::env::var("CYCLES")
            .ok()
            .as_deref()
            .map(str::parse)
            .and_then(Result::ok);

        let kernel_launch_window_queue: VecDeque<Arc<dyn Kernel>> =
            VecDeque::with_capacity(config.kernel_window_size());

        Self {
            config,
            mem_controller,
            stats,
            mem_partition_units,
            kernel_manager,
            stream_manager: streams::StreamManager::default(),
            interconn,
            kernel_launch_window_queue,
            clusters,
            warp_instruction_unique_uid,
            // cluster_issuers,
            // last_cluster_issue,
            block_issuer,
            allocations,
            trace,
            cycle_limit,
            log_after_cycle: None,
            core_time: 0.0,
            dram_time: 0.0,
            icnt_time: 0.0,
            l2_time: 0.0,
        }
    }

    pub fn active(&self) -> bool {
        for cluster in self.clusters.iter() {
            if cluster.num_active_threads() > 0 {
                return true;
            }
        }
        for unit in self.mem_partition_units.iter() {
            if unit.busy() {
                return true;
            }
        }
        if self.interconn.busy() {
            return true;
        }
        if self.kernel_manager.more_blocks_to_run() {
            return true;
        }
        false
    }

    pub fn launch(&mut self, kernel: Arc<dyn Kernel>, cycle: u64) -> eyre::Result<()> {
        let launch_latency = self.config.kernel_launch_latency
            + kernel.num_blocks() * self.config.block_launch_latency;
        self.kernel_manager
            .try_launch_kernel(kernel, launch_latency, cycle)?;
        Ok(())
    }

    fn next_clock_domain(&mut self) -> ClockMask {
        let mut smallest = [self.core_time, self.icnt_time, self.dram_time]
            .into_iter()
            .fold(f64::INFINITY, |a, b| a.min(b));
        let mut mask = bitvec::array::BitArray::ZERO;
        if self.l2_time <= smallest {
            smallest = self.l2_time;
            mask.set(ClockDomain::L2 as usize, true);
            self.l2_time += self.config.clock_frequencies.l2_period;
        }
        if self.icnt_time <= smallest {
            mask.set(ClockDomain::ICNT as usize, true);
            self.icnt_time += self.config.clock_frequencies.interconn_period;
        }
        if self.dram_time <= smallest {
            mask.set(ClockDomain::DRAM as usize, true);
            self.dram_time += self.config.clock_frequencies.dram_period;
        }
        if self.core_time <= smallest {
            mask.set(ClockDomain::CORE as usize, true);
            self.core_time += self.config.clock_frequencies.core_period;
        }
        mask
    }

    #[allow(clippy::overly_complex_bool_expr)]
    #[tracing::instrument(name = "cycle")]
    pub fn cycle(&mut self, mut cycle: u64) -> u64 {
        #[cfg(feature = "timings")]
        let start_total = std::time::Instant::now();
        let clock_mask = self.next_clock_domain();
        log::trace!("clock mask: {}", clock_mask.to_bit_string());

        // shader core loading (pop from ICNT into core)
        if !self.config.simulate_clock_domains || clock_mask[ClockDomain::CORE as usize] {
            #[cfg(feature = "timings")]
            let start = std::time::Instant::now();
            for cluster in self.clusters.iter_mut() {
                cluster.interconn_cycle(cycle);
            }
            #[cfg(feature = "timings")]
            TIMINGS
                .lock()
                .entry("cycle::interconn")
                .or_default()
                .add(start.elapsed());
        }

        // pop from memory controller to interconnect
        if !self.config.simulate_clock_domains || clock_mask[ClockDomain::ICNT as usize] {
            log::debug!(
                "POP from {} memory sub partitions",
                self.config.total_sub_partitions()
            );

            #[cfg(feature = "timings")]
            let start = std::time::Instant::now();
            for partition in self.mem_partition_units.iter_mut() {
                for mem_sub in partition.sub_partitions.iter_mut() {
                    if log::log_enabled!(log::Level::Debug) {
                        log::debug!("checking sub partition[{}]:", mem_sub.global_id);
                        log::debug!(
                            "\t icnt to l2 queue ({:<3}) = {}",
                            mem_sub.interconn_to_l2_queue.len(),
                            mem_sub.interconn_to_l2_queue
                        );
                        log::debug!(
                            "\t l2 to icnt queue ({:<3}) = {}",
                            mem_sub.l2_to_interconn_queue.len(),
                            mem_sub.l2_to_interconn_queue
                        );
                        let l2_to_dram_queue = &mem_sub.l2_to_dram_queue;
                        log::debug!(
                            "\t l2 to dram queue ({:<3}) = {}",
                            l2_to_dram_queue.len(),
                            l2_to_dram_queue
                        );
                        log::debug!(
                            "\t dram to l2 queue ({:<3}) = {}",
                            mem_sub.dram_to_l2_queue.len(),
                            mem_sub.dram_to_l2_queue
                        );
                        let dram_latency_queue: Vec<_> = partition
                            .dram_latency_queue
                            .iter()
                            .map(|(_, fetch)| fetch.to_string())
                            .collect();
                        log::debug!(
                            "\t dram latency queue ({:3}) = {:?}",
                            dram_latency_queue.len(),
                            style(&dram_latency_queue).red()
                        );
                    }

                    if let Some(fetch) = mem_sub.top() {
                        let response_packet_size = if fetch.is_write() {
                            fetch.control_size()
                        } else {
                            fetch.size()
                        };
                        let device = self.config.mem_id_to_device_id(mem_sub.global_id);
                        if self.interconn.has_buffer(device, response_packet_size) {
                            let mut fetch = mem_sub.pop().unwrap();
                            if let Some(cluster_id) = fetch.cluster_id {
                                fetch.set_status(mem_fetch::Status::IN_ICNT_TO_SHADER, 0);
                                self.interconn.push(
                                    device,
                                    cluster_id,
                                    ic::Packet { fetch, time: cycle },
                                    response_packet_size,
                                );
                            }
                        } else {
                            // for stats: self.gpu_stall_icnt2sh += 1;
                        }
                    }
                }
            }
            #[cfg(feature = "timings")]
            TIMINGS
                .lock()
                .entry("cycle::subpartitions")
                .or_default()
                .add(start.elapsed());
        }

        // DRAM
        if !self.config.simulate_clock_domains || clock_mask[ClockDomain::DRAM as usize] {
            #[cfg(feature = "timings")]
            let start = std::time::Instant::now();
            log::debug!("cycle for {} drams", self.mem_partition_units.len());
            for (_i, unit) in self.mem_partition_units.iter_mut().enumerate() {
                unit.simple_dram_cycle(cycle);
            }
            #[cfg(feature = "timings")]
            TIMINGS
                .lock()
                .entry("cycle::dram")
                .or_default()
                .add(start.elapsed());
        }

        // L2 operations
        if !self.config.simulate_clock_domains || clock_mask[ClockDomain::L2 as usize] {
            log::debug!(
                "moving mem requests from interconn to {} mem partitions",
                self.config.total_sub_partitions()
            );

            #[cfg(feature = "timings")]
            let start = std::time::Instant::now();
            for partition in self.mem_partition_units.iter_mut() {
                for mem_sub in partition.sub_partitions.iter_mut() {
                    // move memory request from interconnect into memory partition
                    // (if not backed up)
                    //
                    // Note:This needs to be called in DRAM clock domain if there
                    // is no L2 cache in the system In the worst case, we may need
                    // to push NUM_SECTORS requests, so ensure you have enough
                    // buffer for them
                    let device = self.config.mem_id_to_device_id(mem_sub.global_id);

                    if mem_sub
                        .interconn_to_l2_queue
                        .can_fit(mem_sub_partition::NUM_SECTORS as usize)
                    {
                        if let Some(ic::Packet { fetch, .. }) = self.interconn.pop(device) {
                            assert_eq!(fetch.sub_partition_id(), mem_sub.global_id);
                            log::debug!(
                                "got new fetch {} for mem sub partition {} ({})",
                                fetch,
                                mem_sub.global_id,
                                device
                            );

                            mem_sub.push(fetch, cycle);
                        }
                    } else {
                        log::debug!(
                            "SKIP sub partition {} ({}): DRAM full stall",
                            mem_sub.global_id,
                            device
                        );
                        let kernel_id = self
                            .kernel_manager
                            .current_kernel()
                            .as_ref()
                            .map(|kernel| kernel.id() as usize);
                        let kernel_stats = self.stats.get_mut(kernel_id);
                        kernel_stats.stall_dram_full += 1;
                    }
                    mem_sub.cycle(cycle);
                }
            }
            #[cfg(feature = "timings")]
            TIMINGS
                .lock()
                .entry("cycle::l2")
                .or_default()
                .add(start.elapsed());
        }

        // self.interconn_transfer();

        if !self.config.simulate_clock_domains || clock_mask[ClockDomain::CORE as usize] {
            #[cfg(feature = "timings")]
            let start = std::time::Instant::now();

            let mut active_clusters = utils::box_slice![false; self.clusters.len()];
            for (cluster_id, cluster) in self.clusters.iter_mut().enumerate() {
                assert_eq!(cluster_id, cluster.cluster_id);
                log::debug!("cluster {} cycle {}", cluster_id, cycle);
                let cores_completed = cluster.num_active_threads() == 0;
                let kernels_completed = self.kernel_manager.all_kernels_completed();

                let cluster_active = !(cores_completed && kernels_completed);
                active_clusters[cluster_id] = cluster_active;
                if !cluster_active {
                    continue;
                }

                let core_sim_order = cluster.core_sim_order.try_lock();
                for local_core_id in &*core_sim_order {
                    let core = &cluster.cores[*local_core_id];
                    let mut core = core.try_lock();
                    crate::timeit!("core::cycle", core.cycle(cycle));
                }
                // active_sms += cluster.num_active_sms();
            }

            #[cfg(debug_assertions)]
            {
                // sanity check that inactive clusters do no produce any messages
                for (cluster_id, cluster) in self.clusters.iter_mut().enumerate() {
                    if active_clusters[cluster_id] {
                        continue;
                    }
                    let core_sim_order = cluster.core_sim_order.try_lock();
                    for local_core_id in &*core_sim_order {
                        let core = &cluster.cores[*local_core_id];
                        let core = core.try_lock();
                        let mem_port = &core.mem_port;
                        assert_eq!(mem_port.buffer.len(), 0);
                    }
                }
            }

            for cluster in self.clusters.iter_mut() {
                let cluster_id = cluster.cluster_id;
                let mut core_sim_order = cluster.core_sim_order.try_lock();
                for local_core_id in &*core_sim_order {
                    let core = &cluster.cores[*local_core_id];
                    let mut core = core.try_lock();
                    let core_id = core.id();
                    let mem_port = &mut core.mem_port;
                    log::trace!(
                        "interconn buffer for core {:?}: {:?}",
                        core_id,
                        mem_port
                            .buffer
                            .iter()
                            .map(
                                |ic::Packet {
                                     fetch: (_dest, fetch, _size),
                                     ..
                                 }| fetch.to_string()
                            )
                            .collect::<Vec<_>>()
                    );

                    for ic::Packet {
                        fetch: (dest, fetch, size),
                        time,
                    } in mem_port.buffer.drain(..)
                    {
                        assert_eq!(
                            dest,
                            self.config.mem_id_to_device_id(fetch.sub_partition_id())
                        );
                        self.interconn
                            .push(cluster_id, dest, ic::Packet { fetch, time }, size);
                    }
                }
                if !active_clusters[cluster_id] {
                    // do not advance core sim order if cluster is inactive
                    continue;
                }
                if let config::SchedulingOrder::RoundRobin = self.config.simt_core_sim_order {
                    core_sim_order.rotate_left(1);
                }
            }

            #[cfg(feature = "timings")]
            TIMINGS
                .lock()
                .entry("cycle::core")
                .or_default()
                .add(start.elapsed());

            cycle += 1;

            crate::timeit!(
                "cycle::issue_block_to_core",
                self.block_issuer
                    .issue_blocks_to_core_deterministic(&mut self.kernel_manager, cycle) // self.issue_blocks_to_core_deterministic(&mut self.kernel_manager, cycle)
            );

            self.kernel_manager.decrement_launch_latency(1);

            // Depending on configuration, invalidate the caches
            // once all threads are completed.
            self.flush_caches(cycle);

            #[cfg(feature = "timings")]
            TIMINGS
                .lock()
                .entry("cycle::total")
                .or_default()
                .add(start_total.elapsed());
        }

        // self.debug_non_exit();
        cycle
    }

    pub fn flush_caches(&mut self, _cycle: u64) {
        let mut num_active_threads = 0;
        if self.config.flush_l1_cache {
            #[cfg(feature = "timings")]
            let start = std::time::Instant::now();

            let mut clusters_flushed = 0;
            // let mut lines_flushed = 0;
            for cluster in self.clusters.iter_mut() {
                let cluster_id = cluster.cluster_id;
                let num_cluster_active_threads = cluster.num_active_threads();
                log::trace!(
                    "cluster {}: {} threads not completed",
                    cluster_id,
                    num_cluster_active_threads
                );
                if num_cluster_active_threads == 0 {
                    cluster.cache_invalidate();
                    clusters_flushed += 1;
                } else {
                    num_active_threads += num_cluster_active_threads;
                }
            }
            #[cfg(feature = "timings")]
            TIMINGS
                .lock()
                .entry("cycle::flush_l1")
                .or_default()
                .add(start.elapsed());

            log::trace!(
                "l1 flush: {}/{} clusters flushed ({} active threads)",
                clusters_flushed,
                self.clusters.len(),
                num_active_threads
            );
        }

        match &self.config.data_cache_l2 {
            Some(l2_config) if self.config.flush_l2_cache => {
                let mut all_threads_complete = num_active_threads == 0;
                #[cfg(feature = "timings")]
                let start = std::time::Instant::now();
                if !self.config.flush_l1_cache {
                    for cluster in self.clusters.iter_mut() {
                        if cluster.num_active_threads() > 0 {
                            all_threads_complete = false;
                            break;
                        }
                    }
                }

                let should_flush = if self.config.accelsim_compat {
                    assert!(l2_config.inner.total_lines() > 0);
                    all_threads_complete && l2_config.inner.total_lines() > 0
                } else {
                    all_threads_complete
                };

                let mut num_flushed = 0;
                if should_flush {
                    for partition in self.mem_partition_units.iter_mut() {
                        for mem_sub in partition.sub_partitions.iter_mut() {
                            mem_sub.flush_l2();
                            num_flushed += 1;
                        }
                    }
                }

                #[cfg(feature = "timings")]
                TIMINGS
                    .lock()
                    .entry("cycle::flush_l2")
                    .or_default()
                    .add(start.elapsed());

                log::trace!(
                    "l2 flush: flushed {}/{} sub partitions ({} active threads)",
                    num_flushed,
                    self.config.total_sub_partitions(),
                    num_active_threads
                );
            }
            _ => {}
        }
    }

    #[allow(dead_code)]
    fn debug_non_exit(&self) {
        eprintln!("\n\n DEBUG NON EXIT");
        let all_clusters_completed = !self
            .clusters
            .iter()
            .any(|cluster| cluster.num_active_threads() > 0);
        eprintln!("kernels left: {}", self.kernels_left());
        eprintln!("commands left: {}", self.trace.commands_left());
        eprintln!("all clusters completed: {}", all_clusters_completed);
        for cluster in self.clusters.iter() {
            eprintln!(
                "cluster {} num active threads: {}",
                cluster.cluster_id,
                cluster.num_active_threads()
            );
        }

        for core in self
            .clusters
            .iter()
            .flat_map(|cluster| cluster.cores.iter())
        {
            let core = core.try_lock();
            let block_status: Vec<_> = core
                .active_threads_per_hardware_block
                .iter()
                .enumerate()
                .filter(|(_id, num_threads)| **num_threads > 0)
                .collect();
            eprintln!("core {:?}: blocks: {:?}", core.id(), block_status);
            eprintln!(
                "core {:?}: kernel: {:?}",
                core.id(),
                core.current_kernel
                    .as_ref()
                    .map(|kernel| kernel.to_string())
            );

            // let instr_fetch_response_queue =
            //     &self.clusters[core.cluster_id].core_instr_fetch_response_queue[core.local_core_id];
            // let load_store_response_queue =
            //     &self.clusters[core.cluster_id].core_load_store_response_queue[core.local_core_id];

            // eprintln!(
            //     "core {:?}: instr fetch response queue size: {}",
            //     core.id(),
            //     instr_fetch_response_queue.len(),
            // );
            // eprintln!(
            //     "core {:?}: load store response queue size: {}",
            //     core.id(),
            //     load_store_response_queue.len(),
            // );
        }

        if let Some(current_kernel) = self.kernel_manager.current_kernel() {
            eprintln!(
                "kernel manager: kernel {} done={} running={} more blocks to run={}",
                current_kernel,
                current_kernel.done(),
                current_kernel.running(),
                !current_kernel.no_more_blocks_to_run(),
            );
        } else {
            eprintln!("kernel manager: NO CURRENT KERNEL");
        }
        for (partition_id, partition) in self.mem_partition_units.iter().enumerate() {
            let busy = partition.sub_partitions.iter().any(|sub| sub.busy());

            eprintln!(
                "{:>20} {:>2} \tbusy={}",
                "partition unit", partition_id, busy,
            );

            for sub in partition.sub_partitions.iter() {
                let busy = sub.busy();
                let interconn_to_l2_queue = &sub.interconn_to_l2_queue;
                let l2_to_dram_queue = &sub.l2_to_dram_queue;
                let dram_to_l2_queue = &sub.dram_to_l2_queue;
                let l2_to_interconn_queue = &sub.l2_to_interconn_queue;
                let rop_queue = &sub.rop_queue;
                eprintln!(
                    "{:>20} {:>2} \tbusy={} \tpending={} \t{:>3} icnt->l2 \t{:>3} l2->dram \t{:>3} dram->l2 \t{:>3} l2->icnt \t{:>3} rop",
                    "sub partition",
                    sub.global_id, busy,
                    sub.request_tracker.len(),
                    interconn_to_l2_queue.len(),
                    l2_to_dram_queue.len(),
                    dram_to_l2_queue.len(),
                    l2_to_interconn_queue.len(),
                    rop_queue.len(),
                );
            }
        }

        eprintln!("interconn busy: {}", self.interconn.busy());
        // if self.interconn.busy() {
        //     for cluster_id in 0..self.config.num_simt_clusters {
        //         let queue = self
        //             .interconn
        //             .dest_queue(cluster_id)
        //             .try_lock()
        //             .iter()
        //             .sorted_by_key(|packet| packet.fetch.addr())
        //             .map(ToString::to_string)
        //             .collect::<Vec<_>>();
        //         if !queue.is_empty() {
        //             // log::trace!(
        //             eprintln!(
        //                 "cluster {cluster_id:<3} icnt: [{:<3}] {:?}...",
        //                 queue.len(),
        //                 queue.iter().next(),
        //             );
        //         }
        //     }
        //     for sub_id in 0..self.config.total_sub_partitions() {
        //         let mem_device = self.config.mem_id_to_device_id(sub_id);
        //         let queue = self
        //             .interconn
        //             .dest_queue(mem_device)
        //             .try_lock()
        //             .iter()
        //             .sorted_by_key(|packet| packet.fetch.addr())
        //             .map(ToString::to_string)
        //             .collect::<Vec<_>>();
        //         if !queue.is_empty() {
        //             // log::trace!(
        //             eprintln!(
        //                 "sub     {sub_id:<3} icnt: [{:<3}] {:?}...",
        //                 queue.len(),
        //                 queue.iter().next()
        //             );
        //         }
        //     }
        // }
        eprintln!("\n\n");
    }

    pub fn l2_used_bytes(&self) -> u64 {
        let sub_partitions = MemSubPartitionIter {
            partition_units: &self.mem_partition_units,
            sub_partitions_per_partition: self.config.num_sub_partitions_per_memory_controller,
            global_sub_id: 0,
        };

        sub_partitions
            .map(|sub| {
                if let Some(ref l2_cache) = sub.l2_cache {
                    l2_cache.num_used_bytes()
                } else {
                    0
                }
            })
            .sum()
    }

    pub fn l2_used_lines(&self) -> usize {
        let sub_partitions = MemSubPartitionIter {
            partition_units: &self.mem_partition_units,
            sub_partitions_per_partition: self.config.num_sub_partitions_per_memory_controller,
            global_sub_id: 0,
        };

        sub_partitions
            .map(|sub| {
                if let Some(ref l2_cache) = sub.l2_cache {
                    l2_cache.num_used_lines() as usize
                } else {
                    0
                }
            })
            .sum()
    }

    pub fn invalidate_l2(&mut self, start_addr: address, num_bytes: u64) {
        eprintln!("invalidate l2 cache [start={start_addr}, num bytes={num_bytes}]");
        for partition in self.mem_partition_units.iter_mut() {
            for mem_sub in partition.sub_partitions.iter_mut() {
                if let Some(ref mut l2_cache) = mem_sub.l2_cache {
                    let chunk_size = SECTOR_SIZE;
                    let num_chunks = (num_bytes as f64 / chunk_size as f64).ceil() as usize;

                    for chunk in 0..num_chunks {
                        let addr = start_addr + (chunk as u64 * chunk_size as u64);
                        l2_cache.invalidate_addr(addr);
                    }
                }
            }
        }
    }

    pub fn print_l1_cache(&self) {
        let mut num_total_lines = 0;
        let mut num_total_lines_used = 0;
        let total_cores = self.config.total_cores();
        for cluster in self.clusters.iter() {
            for core in cluster.cores.iter() {
                let core = core.try_lock();
                let ldst_unit = &core.load_store_unit;
                if let Some(ref l1_cache) = ldst_unit.data_l1 {
                    let num_lines_used = l1_cache.num_used_lines();
                    let num_lines = l1_cache.num_total_lines();
                    eprintln!(
                        "core {:>3}/{:<3}: L2D {:>5}/{:<5} lines used ({:2.2}%, {})",
                        core.global_core_id,
                        total_cores,
                        num_lines_used,
                        num_lines,
                        num_lines_used as f32 / num_lines as f32 * 100.0,
                        human_bytes::human_bytes(num_lines_used as f64 * 128.0),
                    );
                    num_total_lines += num_lines;
                    num_total_lines_used += num_lines_used;
                }
            }
        }
        eprintln!(
            "Total L1D {:>5}/{:<5} lines used ({:2.2}%, {})",
            num_total_lines_used,
            num_total_lines,
            num_total_lines_used as f32 / num_total_lines as f32 * 100.0,
            human_bytes::human_bytes(num_total_lines_used as f64 * 128.0),
        );
    }

    pub fn print_l2_cache(&self) {
        let mut num_total_lines = 0;
        let mut num_total_lines_used = 0;
        let num_sub_partitions = self.config.total_sub_partitions();

        let sub_partitions = MemSubPartitionIter {
            partition_units: &self.mem_partition_units,
            sub_partitions_per_partition: self.config.num_sub_partitions_per_memory_controller,
            global_sub_id: 0,
        };

        for sub in sub_partitions {
            if let Some(ref l2_cache) = sub.l2_cache {
                let num_lines_used = l2_cache.num_used_lines();
                let num_lines = l2_cache.num_total_lines();
                eprintln!(
                    "sub {:>3}/{:<3}: L2D {:>5}/{:<5} lines used ({:2.2}%, {})",
                    sub.global_id,
                    num_sub_partitions,
                    num_lines_used,
                    num_lines,
                    num_lines_used as f32 / num_lines as f32 * 100.0,
                    human_bytes::human_bytes(num_lines_used as f64 * 128.0),
                );
                num_total_lines += num_lines;
                num_total_lines_used += num_lines_used;
            }
        }
        eprintln!(
            "Total L2D {:>5}/{:<5} lines used ({:2.2}%, {})",
            num_total_lines_used,
            num_total_lines,
            num_total_lines_used as f32 / num_total_lines as f32 * 100.0,
            human_bytes::human_bytes(num_total_lines_used as f64 * 128.0),
        );
        // let stats = self.stats();
        // for (kernel_launch_id, kernel_stats) in stats.as_ref().iter().enumerate() {
        //     eprintln!(
        //         "L2D[kernel {}]: {:#?}",
        //         kernel_launch_id,
        //         &kernel_stats.l2d_stats.reduce()
        //     );
        // }
        // eprintln!("L2D[no kernel]: {:#?}", &stats.no_kernel.l2d_stats.reduce());
        // eprintln!("DRAM[no kernel]: {:#?}", &stats.no_kernel.dram.reduce());
    }

    // TODO: move to trait
    pub fn write_l1_cache_state(&self, path: &Path) -> eyre::Result<()> {
        // open csv writer
        let writer = utils::fs::open_writable(path)?;
        let mut csv_writer = csv::WriterBuilder::new()
            .flexible(false)
            .from_writer(writer);

        for cluster in self.clusters.iter() {
            for core in cluster.cores.iter() {
                let core = core.try_lock();
                let ldst_unit = &core.load_store_unit;
                if let Some(ref l1_cache) = ldst_unit.data_l1 {
                    l1_cache.write_state(&mut csv_writer)?;
                }
            }
        }
        Ok(())
    }

    // TODO: move to trait
    pub fn write_l2_cache_state(&self, path: &Path) -> eyre::Result<()> {
        // open csv writer
        let writer = utils::fs::open_writable(path)?;
        let mut csv_writer = csv::WriterBuilder::new()
            .flexible(false)
            .from_writer(writer);

        let sub_partitions = MemSubPartitionIter {
            partition_units: &self.mem_partition_units,
            sub_partitions_per_partition: self.config.num_sub_partitions_per_memory_controller,
            global_sub_id: 0,
        };

        for sub in sub_partitions {
            if let Some(ref l2_cache) = sub.l2_cache {
                l2_cache.write_state(&mut csv_writer)?;
            }
        }
        Ok(())
    }

    /// Allocate memory on simulated device.
    pub fn gpu_mem_alloc(
        &mut self,
        addr: address,
        num_bytes: u64,
        name: Option<String>,
        _cycle: u64,
    ) {
        log::info!(
            "CUDA mem alloc: {:<20} {:>15} ({:>5} f32) at address {addr:>20}",
            name.as_deref().unwrap_or("<unnamed>"),
            human_bytes::human_bytes(num_bytes as f64),
            num_bytes / 4,
        );

        // keep track of allocations
        assert!(
            addr % 128 == 0,
            "allocation start address ({}) is not 128B aligned)",
            addr
        );
        let alloc_range = addr..(addr + num_bytes);
        self.allocations.insert(alloc_range, name);
    }

    /// Collect simulation statistics.
    pub fn stats(&self) -> stats::PerKernel {
        let mut stats = self.stats.clone();

        let cores: Vec<_> = self
            .clusters
            .iter()
            .flat_map(|cluster| cluster.cores.iter())
            .collect();

        // collect statistics from cores
        for core in cores.iter() {
            let core = core.try_lock();
            stats += core.stats();
        }

        // collect statistics from mem sub partitions
        let sub_partitions = MemSubPartitionIter {
            partition_units: &self.mem_partition_units,
            sub_partitions_per_partition: self.config.num_sub_partitions_per_memory_controller,
            global_sub_id: 0,
        };

        for sub in sub_partitions {
            stats += sub.stats();
        }

        // collect statistics from mem partitions
        for partition in self.mem_partition_units.iter() {
            stats += partition.stats();
        }

        // Set the release mode
        let is_release_build = !is_debug();
        stats.no_kernel.sim.is_release_build = is_release_build;

        // Set the kernel info, which is only known at the top level.
        for (kernel_launch_id, kernel_stats) in stats.iter_mut().enumerate() {
            if let Some(kernel) = &self
                .kernel_manager
                .executed_kernels
                .try_read()
                .get(&(kernel_launch_id as u64))
            {
                let kernel_info = stats::KernelInfo {
                    name: kernel.config().unmangled_name.clone(),
                    mangled_name: kernel.config().mangled_name.clone(),
                    launch_id: kernel_launch_id,
                };
                kernel_stats.sim.kernel_name = kernel_info.name.clone();
                kernel_stats.sim.kernel_name_mangled = kernel_info.mangled_name.clone();
                kernel_stats.sim.kernel_launch_id = kernel_info.launch_id;
                kernel_stats.sim.is_release_build = is_release_build;

                kernel_stats.dram.kernel_info = kernel_info.clone();
                kernel_stats.accesses.kernel_info = kernel_info.clone();
                kernel_stats.instructions.kernel_info = kernel_info.clone();

                for cache_stats in [
                    &mut kernel_stats.l1i_stats,
                    &mut kernel_stats.l1c_stats,
                    &mut kernel_stats.l1t_stats,
                    &mut kernel_stats.l1d_stats,
                    &mut kernel_stats.l2d_stats,
                ] {
                    cache_stats.kernel_info = kernel_info.clone();
                }
            }
        }
        stats
    }

    /// Process commands greedily
    #[must_use]
    pub fn process_commands(&mut self, mut cycle: u64) -> u64 {
        let mut allocations_and_memcopies: Vec<Command> = Vec::new();

        let kernel_window_size = self.config.kernel_window_size();
        while self.kernel_launch_window_queue.len() < kernel_window_size {
            let Some(cmd) = self.trace.next_command().cloned() else {
                break;
            };
            match cmd {
                Command::KernelLaunch(launch) => {
                    self.handle_allocations_and_memcopies(&allocations_and_memcopies, cycle);

                    // TODO: clean up this mess which has nothing to do here
                    let output_cache_state = std::env::var("PCHASE_OUTPUT_CACHE_STATE")
                        .as_deref()
                        .unwrap_or("")
                        .to_lowercase()
                        == "yes";

                    if output_cache_state {
                        // we assume running pchase, so there is only a
                        // single allocation plus memcopy
                        // assert!(allocations_and_memcopies.len() == 2);
                        let alloc_sizes: Vec<_> = allocations_and_memcopies
                            .iter()
                            .map(|cmd| match cmd {
                                Command::MemAlloc(trace_model::command::MemAlloc {
                                    num_bytes,
                                    ..
                                }) => *num_bytes,
                                Command::MemcpyHtoD(trace_model::command::MemcpyHtoD {
                                    num_bytes,
                                    ..
                                }) => *num_bytes,
                                _ => unreachable!(),
                            })
                            .collect();
                        assert!(alloc_sizes.iter().all_equal());
                        let size_bytes = *alloc_sizes.first().unwrap();

                        let cache_state_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                            .join("benchmarks/debug/pchase_cache_state");
                        std::fs::create_dir_all(&cache_state_dir).ok();
                        let l2_cache_state_file = cache_state_dir.join(format!(
                            "pchase_{:0>12}_{}_l2_cache_state_before.csv",
                            size_bytes.to_string(),
                            utils::fs::Bytes(size_bytes as usize)
                                .to_string()
                                .replace(" ", "_")
                                .replace(".", "_")
                        ));
                        self.write_l2_cache_state(&l2_cache_state_file).unwrap();
                        eprintln!("wrote L2 cache state to {}", l2_cache_state_file.display());
                    }

                    allocations_and_memcopies.clear();

                    let kernel = kernel::trace::KernelTrace::new(
                        launch.clone(),
                        self.trace.traces_dir.as_ref().unwrap(),
                        &self.config,
                        self.config.memory_only,
                    );

                    eprintln!("kernel launch {}: {:#?}", launch.id, &launch);
                    let num_launched_kernels =
                        self.kernel_manager.executed_kernels.try_read().len();

                    match std::env::var("KERNEL_LIMIT")
                        .ok()
                        .map(|limit| limit.parse::<usize>())
                        .transpose()
                        .unwrap()
                    {
                        Some(kernel_limit) if num_launched_kernels < kernel_limit => {
                            log::info!(
                                "adding kernel {} ({}/{})",
                                kernel,
                                num_launched_kernels + 1,
                                kernel_limit
                            );
                            self.kernel_launch_window_queue.push_back(Arc::new(kernel));
                        }
                        Some(kernel_limit) => {
                            log::info!(
                                "skip kernel {} ({}/{})",
                                kernel,
                                num_launched_kernels + 1,
                                kernel_limit
                            );
                        }
                        None => {
                            log::info!("adding kernel {} (no limit)", kernel);
                            self.kernel_launch_window_queue.push_back(Arc::new(kernel));
                        }
                    }
                }
                cmd => allocations_and_memcopies.push(cmd),
            }
        }

        log::info!(
            "allocations: {:#?}",
            self.allocations
                .try_read()
                .iter()
                .map(|(_, alloc)| alloc.to_string())
                .collect::<Vec<_>>()
        );
        cycle
    }

    /// Lauch more kernels if possible.
    ///
    /// Launch all kernels within window that are on a stream that isn't already running
    pub fn launch_kernels(&mut self, cycle: u64) {
        log::debug!("launching kernels");
        let mut launch_queue: Vec<Arc<dyn Kernel>> = Vec::new();

        for kernel in &self.kernel_launch_window_queue {
            let stream_id = kernel.config().stream_id as usize;
            let stream_busy = self.stream_manager.is_busy(stream_id);
            let can_start_kernel = self.kernel_manager.can_start_kernel();
            if !stream_busy && can_start_kernel && !kernel.launched() {
                self.stream_manager.reserve_stream(stream_id);
                launch_queue.push(kernel.clone());
            }
        }

        for kernel in launch_queue {
            log::info!("launching kernel {}", kernel);
            let up_to_kernel: Option<u64> = std::env::var("UP_TO_KERNEL")
                .ok()
                .map(|s| s.parse())
                .transpose()
                .unwrap();
            if let Some(up_to_kernel) = up_to_kernel {
                assert!(kernel.id() <= up_to_kernel, "launching kernel {kernel}");
            }
            self.launch(kernel, cycle).unwrap();
        }
    }

    pub fn reached_limit(&self, cycle: u64) -> bool {
        matches!(self.cycle_limit, Some(limit) if cycle >= limit)
    }

    pub fn kernels_left(&self) -> bool {
        !self.kernel_launch_window_queue.is_empty()
    }

    pub fn run(&mut self) -> eyre::Result<std::time::Duration> {
        let start = std::time::Instant::now();
        dbg!(&self.config.parallelization);
        dbg!(&self.config.fill_l2_on_memcopy);
        TIMINGS.lock().clear();
        match self.config.parallelization {
            config::Parallelization::Serial => {
                self.run_to_completion()?;
            }
            #[cfg(feature = "parallel")]
            config::Parallelization::Deterministic => {
                self.run_to_completion_parallel_deterministic()?;
            }
            #[cfg(feature = "parallel")]
            config::Parallelization::Nondeterministic { run_ahead } => {
                self.run_to_completion_parallel_nondeterministic(run_ahead)?;
            }
        }

        let output_cache_state = std::env::var("OUTPUT_FINAL_CACHE_STATE")
            .as_deref()
            .unwrap_or("")
            .to_lowercase()
            == "yes";

        let debug_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("debug");
        if output_cache_state {
            // l2 cache
            self.print_l2_cache();
            self.write_l2_cache_state(&debug_dir.join("post_l2_cache_state.csv"))?;

            // l1 cache
            self.print_l1_cache();
            self.write_l1_cache_state(&debug_dir.join("post_l1_cache_state.csv"))?;
        }

        Ok(start.elapsed())
    }

    #[tracing::instrument]
    pub fn run_to_completion(&mut self) -> eyre::Result<()> {
        let mut cycle: u64 = 0;
        let mut last_state_change: Option<(deadlock::State, u64)> = None;
        TIMINGS.lock().clear();

        let log_every: u64 = std::env::var("LOG_EVERY")
            .ok()
            .as_deref()
            .map(str::parse)
            .transpose()?
            .unwrap_or(5_000);
        let mut last_time = std::time::Instant::now();

        log::info!("serial for {} cores", self.config.total_cores());

        while (self.trace.commands_left() || self.kernels_left()) && !self.reached_limit(cycle) {
            cycle = self.process_commands(cycle);
            self.launch_kernels(cycle);

            let start_cycle = cycle;

            let mut finished_kernel = None;
            loop {
                log::info!("======== cycle {cycle} ========");
                log::info!("");
                if (cycle - start_cycle) % log_every == 0 && (cycle - start_cycle) > 0 {
                    eprintln!(
                        "cycle {cycle:<10} ({:>8.4} cycle/sec)",
                        log_every as f64 / last_time.elapsed().as_secs_f64()
                    );
                    last_time = std::time::Instant::now()
                }

                log::info!("cycle {} active={}", cycle, &self.active());

                if self.reached_limit(cycle) {
                    break;
                }

                let old_cycle = cycle;
                cycle = self.cycle(cycle);
                assert!(cycle >= old_cycle);

                if !self.active() {
                    finished_kernel = self.kernel_manager.get_finished_kernel();
                    if finished_kernel.is_some() {
                        break;
                    }
                }

                match self.log_after_cycle {
                    Some(ref log_after_cycle) if cycle >= *log_after_cycle => {
                        eprintln!("initializing logging after cycle {cycle}");
                        init_logging();
                        self.log_after_cycle.take();
                        for (_, alloc) in self.allocations.try_read().iter() {
                            log::info!("allocation: {}", alloc);
                        }
                    }
                    _ => {}
                }

                // collect state
                if self.config.deadlock_check {
                    todo!("deadlock check");
                    let state = self.gather_state();
                    if let Some((_last_state, _update_cycle)) = &last_state_change {
                        // log::info!(
                        //     "current: {:?}",
                        //     &state
                        //         .interconn_to_l2_queue
                        //         .iter()
                        //         .map(|v| v.iter().map(ToString::to_string).collect())
                        //         .collect::<Vec<Vec<String>>>()
                        // );
                        // log::info!(
                        //     "last: {:?}",
                        //     &last_state
                        //         .interconn_to_l2_queue
                        //         .iter()
                        //         .map(|v| v.iter().map(ToString::to_string).collect())
                        //         .collect::<Vec<Vec<String>>>()
                        // );

                        // log::info!(
                        //     "interconn to l2 state updated? {}",
                        //     &state.interconn_to_l2_queue != &last_state.interconn_to_l2_queue
                        // );
                    }

                    const DEADLOCK_DETECTION_CYCLE: u64 = 300;
                    match &mut last_state_change {
                        Some((last_state, last_state_change_cycle))
                            if &state == last_state
                                && cycle - *last_state_change_cycle > DEADLOCK_DETECTION_CYCLE =>
                        {
                            panic!("deadlock after cycle {last_state_change_cycle} no progress until cycle {cycle}");
                        }
                        Some((ref mut last_state, ref mut last_state_change_cycle)) => {
                            // log::info!("deadlock check: updated state in cycle {}", cycle);
                            *last_state = state;
                            *last_state_change_cycle = cycle;
                        }
                        None => {
                            last_state_change = Some((state, cycle));
                        }
                    }
                }
            }

            log::debug!("checking for finished kernel in cycle {}", cycle);

            if let Some(kernel) = finished_kernel {
                self.cleanup_finished_kernel(&*kernel, cycle);
            }

            log::trace!(
                "commands left={} kernels left={}",
                self.trace.commands_left(),
                self.kernels_left()
            );
        }
        self.stats.no_kernel.sim.cycles = cycle;

        if let Some(log_after_cycle) = self.log_after_cycle {
            if log_after_cycle >= cycle {
                eprintln!("WARNING: log after {log_after_cycle} cycles but simulation ended after {cycle} cycles");
            }
        }
        log::info!("exit after {cycle} cycles");
        Ok(())
    }

    fn cleanup_finished_kernel(&mut self, kernel: &dyn Kernel, cycle: u64) {
        self.kernel_launch_window_queue
            .retain(|k| k.id() != kernel.id());
        self.stream_manager
            .release_stream(kernel.config().stream_id as usize);

        kernel.set_completed(cycle);
        let kernel_stats = self.stats.get_mut(Some(kernel.id() as usize));

        let elapsed_cycles = kernel.elapsed_cycles().unwrap_or(0);

        kernel_stats.sim.is_release_build = !is_debug();
        kernel_stats.sim.cycles = elapsed_cycles;

        let elapsed = kernel.elapsed_time();

        let elapsed_millis = elapsed
            .as_ref()
            .map(std::time::Duration::as_millis)
            .unwrap_or(0);

        kernel_stats.sim.elapsed_millis = elapsed_millis;
        log::info!(
            "finished kernel {}: {kernel} in {elapsed_cycles} cycles ({elapsed:?})",
            kernel.id(),
        );
    }

    // #[tracing::instrument]
    // fn issue_blocks_to_core_deterministic(&mut self, cycle: u64) {
    //     log::debug!("===> issue block to core");
    //     let mut last_cluster_issue = self.last_cluster_issue.try_lock();
    //     let last_issued = *last_cluster_issue;
    //     let num_clusters = self.config.num_simt_clusters;
    //     for cluster_id in 0..num_clusters {
    //         let cluster_id = (cluster_id + last_issued + 1) % num_clusters;
    //         let cluster = &mut self.clusters[cluster_id];
    //         let cluster_issuer = &mut self.cluster_issuers[cluster_id];
    //         debug_assert_eq!(cluster_id, cluster.cluster_id);
    //
    //         use cluster::ClusterIssue;
    //         let num_blocks_issued =
    //             cluster_issuer.issue_blocks_to_core_deterministic(&mut self.kernel_manager, cycle);
    //             // cluster.issue_block_to_core_deterministic(&mut self.kernel_manager, cycle);
    //         log::trace!(
    //             "cluster[{}] issued {} blocks",
    //             cluster_id,
    //             num_blocks_issued
    //         );
    //
    //         if num_blocks_issued > 0 {
    //             *last_cluster_issue = cluster_id;
    //         }
    //     }
    //
    //     // decrement kernel latency
    //     self.kernel_manager.decrement_launch_latency(1);
    // }
}

pub trait BlockIssue {
    fn issue_blocks_to_core_deterministic<K>(
        &mut self,
        kernel_manager: &mut K,
        cycle: u64,
    ) -> usize
    where
        K: crate::kernel_manager::SelectKernel;
}

#[derive(Debug)]
pub struct BlockIssuer<I, MC> {
    pub last_cluster_issue: usize,
    pub num_simt_clusters: usize,
    pub cluster_issuers: Box<[cluster::ClusterIssuer<cluster::CoreIssuer<Core<I, MC>>>]>,
}

impl<I, MC> BlockIssue for BlockIssuer<I, MC>
// impl BlockIssue for BlockIssuer
// impl<I, MC> BlockIssue for Simulator<I, MC>
// where
//     I: ic::Interconnect<ic::Packet<mem_fetch::MemFetch>>,
//     MC: mcu::MemoryController,
{
    // #[tracing::instrument]
    fn issue_blocks_to_core_deterministic<K>(&mut self, kernel_manager: &mut K, cycle: u64) -> usize
    where
        K: crate::kernel_manager::SelectKernel,
    {
        log::debug!("===> issue block to core");
        // let mut last_cluster_issue = self.last_cluster_issue.try_lock();
        let last_cluster_issue = &mut self.last_cluster_issue;
        let mut num_blocks_issued = 0;
        let last_issued = *last_cluster_issue;
        let num_clusters = self.num_simt_clusters;
        for cluster_id in 0..num_clusters {
            let cluster_id = (cluster_id + last_issued + 1) % num_clusters;
            //     let cluster = &mut self.clusters[cluster_id];
            // debug_assert_eq!(cluster_id, cluster.cluster_id);
            let cluster_issuer = &mut self.cluster_issuers[cluster_id];

            use cluster::ClusterIssue;
            let num_blocks_issued_to_cluster =
                cluster_issuer.issue_blocks_to_core_deterministic(kernel_manager, cycle);
            // cluster_issuer.issue_blocks_to_core_deterministic(&mut self.kernel_manager, cycle);
            // cluster.issue_block_to_core_deterministic(&mut self.kernel_manager, cycle);

            log::trace!(
                "cluster[{}] issued {} blocks",
                cluster_id,
                num_blocks_issued
            );

            if num_blocks_issued_to_cluster > 0 {
                *last_cluster_issue = cluster_id;
            }
            num_blocks_issued += num_blocks_issued_to_cluster;
        }

        // decrement kernel latency
        // kernel_manager.decrement_launch_latency(1);
        num_blocks_issued
    }
}

impl<I, MC> Simulator<I, MC>
where
    I: ic::Interconnect<ic::Packet<mem_fetch::MemFetch>>,
    MC: mcu::MemoryController,
{
    #[deprecated = "accelsim compat mode"]
    fn handle_memcpy_to_gpu_accelsim_compat(&mut self, write_addr: address, time: u64) {
        let num_sub_partitions = self.config.num_sub_partitions_per_memory_controller;
        let tlx_addr = self.mem_controller.to_physical_address(write_addr);
        let partition_id = tlx_addr.sub_partition / num_sub_partitions as u64;
        let sub_partition_id = tlx_addr.sub_partition % num_sub_partitions as u64;

        let partition = &mut self.mem_partition_units[partition_id as usize];

        let mut sector_mask: mem_fetch::SectorMask = BitArray::ZERO;
        // Sector chunk size is 4, so we get the highest 4 bits
        // of the address to set the sector mask
        sector_mask.set(((write_addr % 128) as u8 / 32) as usize, true);

        log::trace!(
            "memcopy to gpu: copy 32 byte chunk starting at {} to sub partition unit {} of partition unit {} ({}) (mask {})",
            write_addr,
            sub_partition_id,
            partition_id,
            tlx_addr.sub_partition,
            sector_mask.to_bit_string()
        );

        partition.memcpy_to_gpu_accelsim_compat(
            write_addr,
            tlx_addr.sub_partition as usize,
            &sector_mask,
            time,
        );
    }

    /// Simulate memory copy to simulated device.
    pub fn handle_allocations_and_memcopies(&mut self, commands: &Vec<Command>, mut cycle: u64) {
        // handle allocations (no copy)
        let (allocations, copies): (Vec<_>, Vec<_>) =
            commands
                .into_iter()
                .cloned()
                .partition_map(|cmd| match cmd {
                    Command::MemAlloc(alloc) => itertools::Either::Left(alloc),
                    Command::MemcpyHtoD(copy) => itertools::Either::Right(copy),
                    Command::KernelLaunch(_launch) => unreachable!(),
                });

        for trace_model::command::MemAlloc {
            allocation_name,
            device_ptr,
            num_bytes,
            ..
        } in &allocations
        {
            self.gpu_mem_alloc(*device_ptr, *num_bytes, allocation_name.clone(), cycle);
        }

        let l2_cache_size_bytes = self
            .config
            .data_cache_l2
            .as_ref()
            .map(|l2| self.config.total_sub_partitions() * l2.inner.total_bytes())
            .unwrap_or(0);

        let l2_prefetch_percent = self.config.l2_prefetch_percent.unwrap_or(100.0);
        let cache_bytes_used_before = self.l2_used_bytes();
        let mut cache_bytes_used = cache_bytes_used_before;

        let mut valid_prefill_memcopies = Vec::new();

        for trace_model::command::MemAlloc {
            allocation_name,
            device_ptr,
            num_bytes,
            fill_l2,
        } in allocations.clone().into_iter().rev()
        {
            let allocation_id = self
                .allocations
                .try_read()
                .iter()
                .find(|(_, alloc)| alloc.start_addr == device_ptr)
                .map(|(_, alloc)| alloc.id)
                .unwrap();

            if num_bytes > 64 {
                eprintln!(
                    "CUDA allocation={:<3} {}: {:>15} ({:>5} f32) to address {:>20}",
                    allocation_id,
                    allocation_name.as_deref().unwrap_or("<unnamed>"),
                    human_bytes::human_bytes(num_bytes as f64),
                    num_bytes / 4,
                    device_ptr,
                );
            }

            let alloc_range = device_ptr..device_ptr + num_bytes;

            // find a copy
            let copy = copies
                .iter()
                .find(|copy| alloc_range.contains(&copy.dest_device_addr));

            let percent =
                ((cache_bytes_used + num_bytes) as f32 / l2_cache_size_bytes as f32) * 100.0;
            let should_prefill = if allocations.len() == 1 {
                // for a single allocation, go all in
                // cache_bytes_used + num_bytes
                true
                // percent <= 100.0
            } else {
                // percent <= l2_prefetch_percent
                // should be less than 75 percent
                percent <= 60.0
            };

            let alignment = crate::exec::tracegen::ALIGNMENT_BYTES;

            let end_addr = device_ptr + num_bytes;

            // take the last 2.5 MB only
            // let valid_num_bytes = num_bytes.min((2.5 * crate::config::MB as f32) as u64);
            let valid_num_bytes = num_bytes;

            let start_addr = if valid_num_bytes != num_bytes {
                let start_addr = end_addr.saturating_sub(valid_num_bytes);
                let start_addr = utils::next_multiple(start_addr, alignment);
                assert!(
                    start_addr % alignment == 0,
                    "start address {} for allocation {} is not {} aligned",
                    allocation_id,
                    start_addr,
                    human_bytes::human_bytes(alignment as f32),
                );
                start_addr
            } else {
                device_ptr
            };

            // what does it take to make LRU miss all sets when
            // going over capacity
            // 12 partitions *128 sets * 128B line/1024 = 129KB

            // if *fill_l2 || (copy.is_some() && should_prefill) {
            let force_fill = fill_l2 == trace_model::command::L2Prefill::Force;
            let disabled = fill_l2 == trace_model::command::L2Prefill::NoPrefill;

            if force_fill || (!disabled && should_prefill) {
                valid_prefill_memcopies.push((start_addr, valid_num_bytes, allocation_id));
                cache_bytes_used += valid_num_bytes;
            }
        }

        eprintln!(
            "have {}/{} valid memcopies (requested={}, occupied={}, cache size={})",
            valid_prefill_memcopies
                .iter()
                .filter(|(_, num_bytes, _)| *num_bytes > 64)
                .count(),
            allocations.len() + copies.len(),
            human_bytes::human_bytes(copies.iter().map(|copy| copy.num_bytes).sum::<u64>() as f64),
            human_bytes::human_bytes(cache_bytes_used_before as f64),
            human_bytes::human_bytes(l2_cache_size_bytes as f64),
        );

        for (start_addr, num_bytes, allocation_id) in valid_prefill_memcopies
            .into_iter()
            .sorted_by_key(|(_, _, allocation_id)| *allocation_id)
        {
            cycle = crate::timeit!(
                "cycle::memcopy",
                self.memcopy_to_gpu(start_addr, num_bytes, cycle,)
            );
        }
    }

    /// Simulate memory copy to simulated device.
    #[allow(clippy::needless_pass_by_value)]
    #[must_use]
    pub fn memcopy_to_gpu(&mut self, addr: address, num_bytes: u64, mut cycle: u64) -> u64 {
        log::info!(
            "CUDA mem copy: {:>15} ({:>5} f32) to address {addr:>20}",
            human_bytes::human_bytes(num_bytes as f64),
            num_bytes / 4,
        );

        if self.config.fill_l2_on_memcopy {
            if self.config.accelsim_compat {
                // todo: remove this branch because accelsim is broken
                let chunk_size: u64 = 32;
                let chunks = (num_bytes as f64 / chunk_size as f64).ceil() as usize;
                for chunk in 0..chunks {
                    let write_addr = addr + (chunk as u64 * chunk_size);
                    self.handle_memcpy_to_gpu_accelsim_compat(write_addr, cycle);
                }
            } else {
                if let Some(ref l2_cache) = self.config.data_cache_l2 {
                    let l2_cache_size_bytes =
                        self.config.total_sub_partitions() * l2_cache.inner.total_bytes();

                    let bytes_used = self.l2_used_bytes();

                    // find allocation
                    let old_allocation_id = self
                        .allocations
                        .try_read()
                        .iter()
                        .find(|(_, alloc)| alloc.contains(addr))
                        .map(|(_, alloc)| alloc.id)
                        .unwrap();

                    let allocation_id = self
                        .allocations
                        .try_read()
                        .get(&addr)
                        .map(|alloc| alloc.id)
                        .unwrap();

                    assert_eq!(old_allocation_id, allocation_id);

                    let max_bytes = num_bytes;
                    let percent = (max_bytes as f32 / l2_cache_size_bytes as f32) * 100.0;

                    if num_bytes > 64 {
                        eprintln!(
                            "l2 cache prefill {}/{} ({}%) threshold={:?}% allocation={:?} used={}",
                            human_bytes::human_bytes(max_bytes as f64),
                            human_bytes::human_bytes(l2_cache_size_bytes as f64),
                            percent,
                            self.config.l2_prefetch_percent,
                            allocation_id,
                            human_bytes::human_bytes(bytes_used as f64),
                        );
                    }

                    let output_memcopy_l2_cache_state =
                        std::env::var("OUTPUT_L2_CACHE_STATE_MEMCPY")
                            .as_deref()
                            .unwrap_or("")
                            .to_lowercase()
                            == "yes";
                    let debug_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("debug");

                    cycle = self.fill_l2(addr, max_bytes, cycle);

                    if output_memcopy_l2_cache_state
                        && num_bytes > 64
                        && !self.config.accelsim_compat
                    {
                        self.write_l2_cache_state(
                            &debug_dir.join(format!("l2_cache_state_{}_after.csv", allocation_id)),
                        )
                        .unwrap();
                        self.print_l2_cache();
                    }
                }
            }
        }
        cycle
    }

    #[must_use]
    pub fn fill_l2(&mut self, addr: address, num_bytes: u64, mut cycle: u64) -> u64 {
        let alignment = crate::exec::tracegen::ALIGNMENT_BYTES;
        assert!(
            addr % alignment == 0,
            "start address {} is not {} aligned",
            addr,
            human_bytes::human_bytes(alignment as f32),
        );

        let chunk_size: u64 = 128;
        let num_chunks = (num_bytes as f64 / chunk_size as f64).ceil() as usize;

        let mut partition_hist = vec![0; self.mem_controller.num_memory_sub_partitions()];
        for chunk in 0..num_chunks {
            let chunk_addr = addr + (chunk as u64 * chunk_size);

            let kind = mem_fetch::access::Kind::GLOBAL_ACC_R;
            let allocation = self.allocations.try_read().get(&chunk_addr).cloned();
            let access = mem_fetch::access::Builder {
                kind,
                addr: chunk_addr,
                kernel_launch_id: None,
                allocation,
                req_size_bytes: chunk_size as u32,
                is_write: kind.is_write(),
                warp_active_mask: warp::ActiveMask::all_ones(),
                byte_mask: !mem_fetch::ByteMask::ZERO,
                sector_mask: !mem_fetch::SectorMask::ZERO,
            }
            .build();

            assert!(access.byte_mask.all());
            assert!(access.sector_mask.all());

            let physical_addr = self.mem_controller.to_physical_address(access.addr);

            let fetch = mem_fetch::Builder {
                instr: None,
                access,
                warp_id: 0,
                // the core id and cluster id are not set as no core/cluster explicitely requested the write.
                // the WRITE_ACK will not be forwarded.
                global_core_id: None,
                cluster_id: None,
                physical_addr,
            }
            .build();

            let dest_sub_partition_id = fetch.sub_partition_id();
            partition_hist[dest_sub_partition_id] += 1;
            let dest_mem_device = self.config.mem_id_to_device_id(dest_sub_partition_id);
            let packet_size = fetch.control_size();

            log::debug!("push transaction: {fetch} to device {dest_mem_device} (cluster_id={:?}, core_id={:?})", fetch.cluster_id, fetch.global_core_id);

            self.interconn.push(
                0,
                dest_mem_device,
                ic::Packet { fetch, time: 0 },
                packet_size,
            );
        }

        if num_bytes > 64 {
            eprintln!("memory partition histogram: {:?}", partition_hist);
        }

        let copy_start_cycle = cycle;
        while self.active() {
            for partition in self.mem_partition_units.iter_mut() {
                for mem_sub in partition.sub_partitions.iter_mut() {
                    if let Some(fetch) = mem_sub.top() {
                        let response_packet_size = if fetch.is_write() {
                            fetch.control_size()
                        } else {
                            fetch.size()
                        };
                        let device = self.config.mem_id_to_device_id(mem_sub.global_id);
                        if self.interconn.has_buffer(device, response_packet_size) {
                            mem_sub.pop().unwrap();
                        }
                    }
                }
            }

            for partition in self.mem_partition_units.iter_mut() {
                for mem_sub in partition.sub_partitions.iter_mut() {
                    let mem_sub_device = self.config.mem_id_to_device_id(mem_sub.global_id);

                    if mem_sub
                        .interconn_to_l2_queue
                        .can_fit(mem_sub_partition::NUM_SECTORS as usize)
                    {
                        if let Some(ic::Packet { fetch, .. }) = self.interconn.pop(mem_sub_device) {
                            assert_eq!(fetch.sub_partition_id(), mem_sub.global_id);
                            mem_sub.push(fetch, cycle);
                        }
                    } else {
                        log::trace!(
                            "SKIP sub partition {} ({mem_sub_device}): DRAM full stall",
                            mem_sub.global_id
                        );
                    }
                    mem_sub.cycle(cycle);
                }
            }

            for unit in self.mem_partition_units.iter_mut() {
                unit.simple_dram_cycle(cycle);
            }

            cycle += 1;
            let copy_duration = cycle - copy_start_cycle;
            if copy_duration > 0 && copy_duration % 10000 == 0 {
                log::warn!("fill l2 on memcopy: running for {copy_duration} cycles");
            }
        }

        // OPTIONAL: check prefilled chunks are HIT (debugging)
        if false {
            for chunk in 0..num_chunks {
                let chunk_addr = addr + (chunk as u64 * chunk_size);

                let allocation = self.allocations.try_read().get(&chunk_addr).cloned();
                let access = mem_fetch::access::Builder {
                    kind: mem_fetch::access::Kind::GLOBAL_ACC_R,
                    addr: chunk_addr,
                    kernel_launch_id: None,
                    allocation,
                    req_size_bytes: chunk_size as u32,
                    is_write: false,
                    warp_active_mask: warp::ActiveMask::all_ones(),
                    byte_mask: !mem_fetch::ByteMask::ZERO,
                    sector_mask: !mem_fetch::SectorMask::ZERO,
                }
                .build();

                let physical_addr = self.mem_controller.to_physical_address(access.addr);

                let fetch = mem_fetch::Builder {
                    instr: None,
                    access,
                    warp_id: 0,
                    // the core id and cluster id are not set as no core/cluster explicitely requested the write.
                    // the WRITE_ACK will not be forwarded.
                    global_core_id: None,
                    cluster_id: None,
                    physical_addr,
                }
                .build();

                let mut write_fetch = fetch.clone();
                write_fetch.access.kind = mem_fetch::access::Kind::GLOBAL_ACC_W;
                write_fetch.access.is_write = true;
                let read_fetch = fetch;

                let dest_global_sub_partition_id = read_fetch.sub_partition_id();
                let (partition_id, local_sub_id) = self
                    .config
                    .partition_and_local_sub_partition_id(dest_global_sub_partition_id);

                let partition = &mut self.mem_partition_units[partition_id];
                let sub = &mut partition.sub_partitions[local_sub_id];

                if let Some(ref _l2_cache) = sub.l2_cache {
                    use crate::mem_sub_partition::{
                        breakdown_request_to_sector_requests, NUM_SECTORS,
                    };
                    let mut read_sector_requests: [Option<mem_fetch::MemFetch>; NUM_SECTORS] =
                        [(); NUM_SECTORS].map(|_| None);
                    breakdown_request_to_sector_requests(
                        read_fetch,
                        &*self.mem_controller,
                        &mut read_sector_requests,
                    );

                    let mut write_sector_requests: [Option<mem_fetch::MemFetch>; NUM_SECTORS] =
                        [(); NUM_SECTORS].map(|_| None);
                    breakdown_request_to_sector_requests(
                        write_fetch,
                        &*self.mem_controller,
                        &mut write_sector_requests,
                    );

                    for read_fetch in read_sector_requests.into_iter().filter_map(|x| x) {
                        let addr = read_fetch.addr();
                        assert!(!read_fetch.is_write());
                        // let mut _events = Vec::new();
                        // let read_status = l2_cache.access(addr, read_fetch, &mut _events, 0);
                        // eprintln!("READ: checked {}: {:?}", addr, read_status);
                        // assert_eq!(read_status, cache::RequestStatus::HIT);
                    }

                    for write_fetch in write_sector_requests.into_iter().filter_map(|x| x) {
                        let addr = write_fetch.addr();
                        assert!(write_fetch.is_write());
                        // let mut _events = Vec::new();
                        // let write_status = l2_cache.access(addr, write_fetch, &mut _events, 0);
                        // eprintln!("WRITE: checked {}: {:?}", addr, write_status);
                        // assert_eq!(write_status, cache::RequestStatus::HIT);
                    }
                }
            }
        }

        let copy_duration = cycle - copy_start_cycle;
        log::debug!("l2 memcopy took {copy_duration} cycles");
        cycle
    }
}

pub fn save_stats_to_file(stats: &stats::PerKernel, path: &Path) -> eyre::Result<()> {
    use serde::Serialize;

    let path = path.with_extension("json");

    if let Some(parent) = &path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let output_file = utils::fs::open_writable(path)?;
    let mut json_serializer = serde_json::Serializer::with_formatter(
        output_file,
        serde_json::ser::PrettyFormatter::with_indent(b"    "),
    );
    stats.serialize(&mut json_serializer)?;
    Ok(())
}

#[cfg(feature = "deadlock_detection")]
const DEADLOCK_DETECTOR_THREAD: std::sync::OnceLock<std::thread::JoinHandle<()>> =
    std::sync::OnceLock::new();

pub fn init_deadlock_detector() {
    #[cfg(feature = "deadlock_detection")]
    DEADLOCK_DETECTOR_THREAD.get_or_init(|| {
        std::thread::spawn(move || loop {
            // Create a background thread which checks for deadlocks every 10s
            std::thread::sleep(std::time::Duration::from_secs(10));
            let deadlocks = parking_lot::deadlock::check_deadlock();
            if deadlocks.is_empty() {
                continue;
            }

            eprintln!("{} deadlocks detected", deadlocks.len());
            for (i, threads) in deadlocks.iter().enumerate() {
                eprintln!("Deadlock #{i}");
                for t in threads {
                    eprintln!("Thread Id {:#?}", t.thread_id());
                    eprintln!("{:#?}", t.backtrace());
                }
            }
        })
    });
}

pub fn accelmain(
    traces_dir: impl AsRef<Path>,
    config: impl Into<Arc<config::GPU>>,
) -> eyre::Result<config::GTX1080> {
    init_deadlock_detector();
    let config = config.into();
    let traces_dir = traces_dir.as_ref();
    let (traces_dir, commands_path) = if traces_dir.is_dir() {
        (traces_dir.to_path_buf(), traces_dir.join("commands.json"))
    } else {
        (
            traces_dir.parent().map(Path::to_path_buf).ok_or_else(|| {
                eyre::eyre!(
                    "could not determine trace dir from file {}",
                    traces_dir.display()
                )
            })?,
            traces_dir.to_path_buf(),
        )
    };

    let mut sim = config::GTX1080::new(config);
    sim.trace.add_commands(commands_path, traces_dir)?;
    sim.run()?;
    Ok(sim)
}

pub fn init_logging() {
    let mut log_builder = env_logger::Builder::new();
    log_builder.format(|buf, record| {
        use std::io::Write;
        let level_style = buf.default_level_style(record.level());
        writeln!(
            buf,
            "[ {}{}{} {} ] {}",
            level_style.render(),
            record.level(),
            level_style.render_reset(),
            record.module_path().unwrap_or(""),
            record.args()
        )
    });

    log_builder.filter_level(log::LevelFilter::Off);
    log_builder.parse_default_env();
    log_builder.init();
}

#[cfg(test)]
mod tests {}
