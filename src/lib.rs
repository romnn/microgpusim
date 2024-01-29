// #![feature(stmt_expr_attributes)]
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
pub mod engine;
pub mod fifo;
pub mod func_unit;
pub mod instruction;
pub mod interconn;
pub mod kernel;
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
pub mod sync;
pub mod tag_array;
pub mod warp;

#[cfg(test)]
pub mod testing;

use self::core::{warp_inst_complete, Core, PipelineStage};
use allocation::{Allocation, Allocations};
use cluster::Cluster;
use engine::cycle::Component;
pub use exec;
use interconn as ic;
use kernel::Kernel;
use mem_sub_partition::SECTOR_SIZE;
use trace_model::{Command, ToBitString};

use crate::sync::{atomic, Arc, Mutex, RwLock};
use bitvec::array::BitArray;
use color_eyre::eyre::{self};
use console::style;
use crossbeam::utils::CachePadded;
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

// /// Max number of SMs
// /// Volta Titan V has 80 SMs
// pub const MAX_STREAMING_MULTIPROCESSORS: usize = 80;

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

// pub const GLOBAL_HEAP_START: u64 = 0xC000_0000;
// // Volta max shmem size is 96kB
// pub const SHARED_MEM_SIZE_MAX: u64 = 96 * (1 << 10);

// The size of the local memory space.
//
// Volta max local mem is 16kB
// pub const LOCAL_MEM_SIZE_MAX: u64 = 1 << 14;
//
// // Volta Titan V has 80 SMs
// pub const MAX_STREAMING_MULTIPROCESSORS: u64 = 80;
//
// pub const TOTAL_SHARED_MEM: u64 = MAX_STREAMING_MULTIPROCESSORS * SHARED_MEM_SIZE_MAX;
//
// pub const TOTAL_LOCAL_MEM: u64 =
//     MAX_STREAMING_MULTIPROCESSORS * super::MAX_THREADS_PER_SM as u64 * LOCAL_MEM_SIZE_MAX;
//
// pub const SHARED_GENERIC_START: u64 = GLOBAL_HEAP_START - TOTAL_SHARED_MEM;
// pub const LOCAL_GENERIC_START: u64 = SHARED_GENERIC_START - TOTAL_LOCAL_MEM;

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

pub fn parse_commands(path: impl AsRef<Path>) -> eyre::Result<Vec<Command>> {
    let reader = utils::fs::open_readable(path.as_ref())?;
    let commands = serde_json::from_reader(reader)?;
    Ok(commands)
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
    once_cell::sync::Lazy::new(|| Mutex::new(HashMap::default()));

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

// trait MemoryPartitionUnit: std::fmt::Debug + Sync + Send + 'static {}
//
// impl MemoryPartitionUnit for mem_partition_unit::MemoryPartitionUnit {}

#[derive()]
pub struct MockSimulator<I, MC> {
    stats: Arc<Mutex<stats::PerKernel>>,
    config: Arc<config::GPU>,
    mem_controller: Arc<MC>,
    // mem_controller: Arc<dyn mcu::MemoryController>,
    // mem_partition_units: Vec<Arc<RwLock<dyn MemoryPartitionUnit>>>,
    mem_partition_units: Vec<Arc<RwLock<mem_partition_unit::MemoryPartitionUnit<MC>>>>,
    mem_sub_partitions: Vec<Arc<Mutex<mem_sub_partition::MemorySubPartition<MC>>>>,
    // we could remove the arcs on running and executed if we change to self: Arc<Self>
    pub running_kernels: Arc<RwLock<Vec<Option<(usize, Arc<dyn Kernel>)>>>>,
    // executed_kernels: Arc<Mutex<HashMap<u64, String>>>,
    executed_kernels: Arc<Mutex<HashMap<u64, Arc<dyn Kernel>>>>,
    pub current_kernel: Mutex<Option<Arc<dyn Kernel>>>,
    pub clusters: Vec<Arc<Cluster<I, MC>>>,
    #[allow(dead_code)]
    warp_instruction_unique_uid: Arc<CachePadded<atomic::AtomicU64>>,
    interconn: Arc<I>,

    // parallel_simulation: bool,
    last_cluster_issue: Arc<Mutex<usize>>,
    last_issued_kernel: Mutex<usize>,
    allocations: allocation::Ref,

    // for main run loop
    traces_dir: Option<PathBuf>,
    commands: Vec<Command>,
    command_idx: usize,
    kernels: VecDeque<Arc<dyn Kernel>>,
    kernel_window_size: usize,
    busy_streams: VecDeque<u64>,
    cycle_limit: Option<u64>,
    log_after_cycle: Option<u64>,
    // gpu_stall_icnt2sh: usize,
    partition_replies_in_parallel: usize,

    core_time: f64,
    dram_time: f64,
    icnt_time: f64,
    l2_time: f64,
}

impl<I, MC> std::fmt::Debug for MockSimulator<I, MC> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockSimulator").finish()
    }
}

pub trait FromConfig {
    fn from_config(config: &config::GPU) -> Self;
}

impl FromConfig for stats::Config {
    fn from_config(config: &config::GPU) -> Self {
        let num_total_cores = config.total_cores();
        let num_mem_units = config.num_memory_controllers;
        let num_sub_partitions = config.total_sub_partitions();
        let num_dram_banks = config.dram_timing_options.num_banks;

        Self {
            num_total_cores,
            num_mem_units,
            num_sub_partitions,
            num_dram_banks,
        }
    }
}

impl<I, MC> MockSimulator<I, MC>
where
    I: ic::Interconnect<ic::Packet<mem_fetch::MemFetch>>,
    MC: mcu::MemoryController,
{
    pub fn new(interconn: Arc<I>, mem_controller: Arc<MC>, config: Arc<config::GPU>) -> Self
    where
        MC: mcu::MemoryController,
    {
        let stats = Arc::new(Mutex::new(stats::PerKernel::new(
            stats::Config::from_config(&config),
        )));

        // let mem_controller: Arc<dyn mcu::MemoryController> = if config.accelsim_compat {
        //     Arc::new(mcu::MemoryControllerUnit::new(&config).unwrap())
        // } else {
        //     Arc::new(mcu::PascalMemoryControllerUnit::new(&config).unwrap())
        // };
        // TODO: REMOVE
        // let mem_controller: Arc<dyn mcu::MemoryController> =
        //     Arc::new(mcu::MemoryControllerUnit::new(&config).unwrap());

        let num_mem_units = config.num_memory_controllers;

        let mem_partition_units: Vec<_> = (0..num_mem_units)
            .map(|partition_id| {
                let unit = mem_partition_unit::MemoryPartitionUnit::new(
                    partition_id,
                    Arc::clone(&config),
                    mem_controller.clone(),
                    Arc::clone(&stats),
                );
                // Arc::new(RwLock::new(unit)) as Arc<RwLock<dyn MemoryPartitionUnit>>
                Arc::new(RwLock::new(unit))
            })
            .collect();

        let mut mem_sub_partitions = Vec::new();
        for partition in &mem_partition_units {
            mem_sub_partitions.extend(partition.try_read().sub_partitions.iter().cloned());
        }

        let max_concurrent_kernels = config.max_concurrent_kernels;
        let running_kernels = Arc::new(RwLock::new(vec![None; max_concurrent_kernels]));
        let allocations = Arc::new(RwLock::new(Allocations::default()));

        let warp_instruction_unique_uid = Arc::new(CachePadded::new(atomic::AtomicU64::new(0)));
        let clusters: Vec<_> = (0..config.num_simt_clusters)
            .map(|i| {
                let cluster = Cluster::new(
                    i,
                    &warp_instruction_unique_uid,
                    &allocations,
                    &interconn,
                    &stats,
                    &config,
                    &mem_controller,
                    // &(Arc::clone(&mem_controller) as Arc<dyn mcu::MemoryController>),
                );
                Arc::new(cluster)
            })
            .collect();

        let executed_kernels = Arc::new(Mutex::new(HashMap::new()));

        assert!(config.max_threads_per_core.rem_euclid(config.warp_size) == 0);
        // let _max_warps_per_shader = config.max_threads_per_core / config.warp_size;

        let window_size = if config.concurrent_kernel_sm {
            config.max_concurrent_kernels
        } else {
            1
        };
        assert!(window_size > 0);

        // todo: make this a hashset?
        let busy_streams: VecDeque<u64> = VecDeque::new();
        let mut kernels: VecDeque<Arc<dyn Kernel>> = VecDeque::new();
        kernels.reserve_exact(window_size);

        let cycle_limit: Option<u64> = std::env::var("CYCLES")
            .ok()
            .as_deref()
            .map(str::parse)
            .and_then(Result::ok);

        // this causes first launch to use simt cluster
        let last_cluster_issue = Arc::new(Mutex::new(config.num_simt_clusters - 1));

        Self {
            config,
            mem_controller,
            stats,
            mem_partition_units,
            mem_sub_partitions,
            interconn,
            // parallel_simulation: false,
            running_kernels,
            executed_kernels,
            current_kernel: Mutex::new(None),
            clusters,
            warp_instruction_unique_uid,
            last_cluster_issue,
            last_issued_kernel: Mutex::new(0),
            allocations,
            traces_dir: None,
            commands: Vec::new(),
            command_idx: 0,
            kernels,
            kernel_window_size: window_size,
            busy_streams,
            cycle_limit,
            log_after_cycle: None,
            partition_replies_in_parallel: 0,
            core_time: 0.0,
            dram_time: 0.0,
            icnt_time: 0.0,
            l2_time: 0.0,
        }
    }

    pub fn add_commands(
        &mut self,
        commands_path: impl AsRef<Path>,
        traces_dir: impl Into<PathBuf>,
    ) -> eyre::Result<()> {
        self.commands
            .extend(parse_commands(commands_path.as_ref())?);
        self.traces_dir = Some(traces_dir.into());
        Ok(())
    }

    /// Select the next kernel to run.
    ///
    /// Todo: used hack to allow selecting the kernel from the shader core,
    /// but we could maybe refactor
    pub fn select_kernel(&self) -> Option<Arc<dyn Kernel>> {
        let mut last_issued_kernel = self.last_issued_kernel.lock();
        let mut executed_kernels = self.executed_kernels.try_lock();
        let running_kernels = self.running_kernels.try_read();

        log::trace!(
            "select kernel: {} running kernels, last issued kernel={}",
            running_kernels.iter().filter_map(Option::as_ref).count(),
            last_issued_kernel
        );

        if let Some((launch_latency, ref last_kernel)) = running_kernels[*last_issued_kernel] {
            log::trace!(
            "select kernel: => running_kernels[{}] no more blocks to run={} {} kernel block latency={} launch uid={}",
            last_issued_kernel,
            last_kernel.no_more_blocks_to_run(),
            last_kernel.next_block().map(|block| format!("{}/{}", block, last_kernel.config().grid)).as_deref().unwrap_or(""),
            launch_latency, last_kernel.id());
        }

        // issue same kernel again
        match running_kernels[*last_issued_kernel] {
            Some((launch_latency, ref last_kernel))
                if !last_kernel.no_more_blocks_to_run() && launch_latency == 0 =>
            {
                let launch_id = last_kernel.id();
                executed_kernels
                    .entry(launch_id)
                    .or_insert(Arc::clone(last_kernel));
                return Some(last_kernel.clone());
            }
            _ => {}
        };

        // issue new kernel
        let num_kernels = running_kernels.len();
        let max_concurrent = self.config.max_concurrent_kernels;
        for n in 0..num_kernels {
            let idx = (n + *last_issued_kernel + 1) % max_concurrent;
            if let Some((launch_latency, ref kernel)) = running_kernels[idx] {
                log::trace!(
                  "select kernel: running_kernels[{}] more blocks left={}, kernel block latency={}",
                  idx,
                  !kernel.no_more_blocks_to_run(),
                  launch_latency,
                );
            }

            match running_kernels[idx] {
                Some((launch_latency, ref kernel))
                    if !kernel.no_more_blocks_to_run() && launch_latency == 0 =>
                {
                    *last_issued_kernel = idx;
                    let launch_id = kernel.id();
                    assert!(!executed_kernels.contains_key(&launch_id));
                    executed_kernels.insert(launch_id, Arc::clone(kernel));
                    return Some(Arc::clone(kernel));
                }
                _ => {}
            }
        }
        None
    }

    pub fn more_blocks_to_run(&self) -> bool {
        let running_kernels = self.running_kernels.try_read();
        running_kernels.iter().any(|kernel| match kernel {
            Some((_, kernel)) => !kernel.no_more_blocks_to_run(),
            None => false,
        })
    }

    pub fn active(&self) -> bool {
        // dbg!(self
        //     .clusters
        //     .iter()
        //     .any(|cluster| cluster.try_read().not_completed() > 0));
        // for partition in self.mem_partition_units.iter() {
        //     dbg!(partition
        //         .try_read()
        //         .sub_partitions
        //         .iter()
        //         .any(|unit| unit.try_lock().busy()));
        // }
        // dbg!(self.interconn.busy());
        // dbg!(self.more_blocks_to_run());

        for cluster in &self.clusters {
            if cluster.not_completed() > 0 {
                return true;
            }
        }
        for unit in &self.mem_partition_units {
            if unit.try_read().busy() {
                return true;
            }
        }
        if self.interconn.busy() {
            return true;
        }
        if self.more_blocks_to_run() {
            return true;
        }
        false
    }

    pub fn can_start_kernel(&self) -> bool {
        let running_kernels = self.running_kernels.try_read();
        running_kernels.iter().any(|kernel| match kernel {
            Some((_, kernel)) => kernel.done(),
            None => true,
        })
    }

    pub fn launch(&self, kernel: Arc<dyn Kernel>, cycle: u64) -> eyre::Result<()> {
        // kernel.set_launched();
        // eprintln!("launch kernel {} in cycle {}", kernel.id(), cycle);
        let threads_per_block = kernel.config().threads_per_block();
        let max_threads_per_block = self.config.max_threads_per_core;
        if threads_per_block > max_threads_per_block {
            log::error!("kernel block size is too large");
            log::error!(
                "CTA size (x*y*z) = {threads_per_block}, max supported = {max_threads_per_block}"
            );
            eyre::bail!("kernel block size is too large");
        }
        let mut running_kernels = self.running_kernels.try_write();
        let free_slot = running_kernels
            .iter_mut()
            .find(|slot| slot.is_none() || slot.as_ref().map_or(false, |(_, k)| k.done()))
            .ok_or(eyre::eyre!("no free slot for kernel"))?;

        kernel.set_started(cycle);
        // *kernel.start_time.lock() = Some(std::time::Instant::now());
        // *kernel.start_cycle.lock() = Some(cycle);

        *self.current_kernel.lock() = Some(Arc::clone(&kernel));
        let launch_latency = self.config.kernel_launch_latency
            + kernel.config().num_blocks() * self.config.block_launch_latency;
        *free_slot = Some((launch_latency, kernel));
        Ok(())
    }

    #[tracing::instrument]
    // #[inline]
    fn issue_block_to_core(&self, cycle: u64) {
        log::debug!("===> issue block to core");
        let mut last_cluster_issue = self.last_cluster_issue.try_lock();
        let last_issued = *last_cluster_issue;
        let num_clusters = self.config.num_simt_clusters;
        for cluster_id in 0..num_clusters {
            let cluster_id = (cluster_id + last_issued + 1) % num_clusters;
            let cluster = &self.clusters[cluster_id];
            debug_assert_eq!(cluster_id, cluster.cluster_id);
            let num_blocks_issued = cluster.issue_block_to_core(self, cycle);
            log::trace!(
                "cluster[{}] issued {} blocks",
                cluster_id,
                num_blocks_issued
            );

            if num_blocks_issued > 0 {
                *last_cluster_issue = cluster_id;
                // self.total_blocks_launched += num_blocks_issued;
            }
        }

        // decrement kernel latency
        for (launch_latency, _) in self
            .running_kernels
            .try_write()
            .iter_mut()
            .filter_map(Option::as_mut)
        {
            *launch_latency = launch_latency.saturating_sub(1);
        }
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
        // use bitvec::field::BitField;
        // let mut clock_mask_bits: bitvec::BitArr!(for 8, in u8) = bitvec::array::BitArray::ZERO;
        // clock_mask_bits.store(clock_mask);
        log::trace!("clock mask: {}", clock_mask.to_bit_string());

        // shader core loading (pop from ICNT into core)
        if !self.config.simulate_clock_domains || clock_mask[ClockDomain::CORE as usize] {
            #[cfg(feature = "timings")]
            let start = std::time::Instant::now();
            for cluster in &self.clusters {
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
                self.mem_sub_partitions.len()
            );

            #[cfg(feature = "timings")]
            let start = std::time::Instant::now();
            for (i, mem_sub) in self.mem_sub_partitions.iter().enumerate() {
                let mut mem_sub = mem_sub.try_lock();
                if log::log_enabled!(log::Level::Debug) {
                    log::debug!("checking sub partition[{i}]:");
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
                    let l2_to_dram_queue = mem_sub.l2_to_dram_queue.try_lock();
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
                    let partition = &self.mem_partition_units[mem_sub.partition_id];
                    let dram_latency_queue: Vec<_> = partition
                        .try_read()
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
                    let device = self.config.mem_id_to_device_id(i);
                    if self.interconn.has_buffer(device, response_packet_size) {
                        let mut fetch = mem_sub.pop().unwrap();
                        if let Some(cluster_id) = fetch.cluster_id {
                            fetch.set_status(mem_fetch::Status::IN_ICNT_TO_SHADER, 0);
                            // fetch.set_return_timestamp(gpu_sim_cycle + gpu_tot_sim_cycle);
                            // , gpu_sim_cycle + gpu_tot_sim_cycle);
                            // log::trace!("interconn push from memory sub partition {i}: {fetch} (cluster={:?}, core={:?})", fetch.cluster_id, fetch.core_id);
                            // eprintln!("interconn push from memory sub partition {i}: {fetch} (cluster={:?}, core={:?})", fetch.cluster_id, fetch.core_id);
                            self.interconn.push(
                                device,
                                cluster_id,
                                ic::Packet {
                                    data: fetch,
                                    time: cycle,
                                },
                                response_packet_size,
                            );
                            self.partition_replies_in_parallel += 1;
                        }
                    } else {
                        // self.gpu_stall_icnt2sh += 1;
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
                unit.try_write().simple_dram_cycle(cycle);
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
                self.mem_sub_partitions.len()
            );

            #[cfg(feature = "timings")]
            let start = std::time::Instant::now();
            // let mut parallel_mem_partition_reqs_per_cycle = 0;
            // let mut stall_dram_full = 0;
            for (sub_id, mem_sub) in self.mem_sub_partitions.iter_mut().enumerate() {
                let mut mem_sub = mem_sub.try_lock();
                // move memory request from interconnect into memory partition
                // (if not backed up)
                //
                // Note:This needs to be called in DRAM clock domain if there
                // is no L2 cache in the system In the worst case, we may need
                // to push NUM_SECTORS requests, so ensure you have enough
                // buffer for them
                let device = self.config.mem_id_to_device_id(sub_id);

                if mem_sub
                    .interconn_to_l2_queue
                    .can_fit(mem_sub_partition::NUM_SECTORS as usize)
                {
                    if let Some(packet) = self.interconn.pop(device) {
                        assert_eq!(packet.data.sub_partition_id(), sub_id);
                        log::debug!(
                            "got new fetch {} for mem sub partition {} ({})",
                            packet.data,
                            sub_id,
                            device
                        );

                        mem_sub.push(packet.data, cycle);
                        // self.parallel_mem_partition_reqs += 1;
                    }
                } else {
                    log::debug!(
                        "SKIP sub partition {} ({}): DRAM full stall",
                        sub_id,
                        device
                    );
                    // if let Some(kernel) = &*self.current_kernel.lock() {
                    let kernel_id = self
                        .current_kernel
                        .lock()
                        .as_ref()
                        .map(|kernel| kernel.id() as usize);
                    let mut stats = self.stats.lock();
                    let kernel_stats = stats.get_mut(kernel_id);
                    kernel_stats.stall_dram_full += 1;
                    // }
                }
                // we borrow all of sub here, which is a problem for the cyclic reference in l2
                // interface
                mem_sub.cycle(cycle);
            }
            #[cfg(feature = "timings")]
            TIMINGS
                .lock()
                .entry("cycle::l2")
                .or_default()
                .add(start.elapsed());
        }

        //   partiton_reqs_in_parallel += partiton_reqs_in_parallel_per_cycle;
        // if (partiton_reqs_in_parallel_per_cycle > 0) {
        //   partiton_reqs_in_parallel_util += partiton_reqs_in_parallel_per_cycle;
        //   gpu_sim_cycle_parition_util++;
        // }

        // self.interconn_transfer();

        if !self.config.simulate_clock_domains || clock_mask[ClockDomain::CORE as usize] {
            #[cfg(feature = "timings")]
            let start = std::time::Instant::now();
            // let kernels_completed = self
            //     .running_kernels
            //     .read()
            //     .iter()
            //     .filter_map(std::option::Option::as_ref)
            //     .all(|k| k.no_more_blocks_to_run());

            // let mut active_sms = 0;
            let mut active_clusters = utils::box_slice![false; self.clusters.len()];
            for (cluster_id, cluster) in self.clusters.iter().enumerate() {
                log::debug!("cluster {} cycle {}", cluster_id, cycle);
                let cores_completed = cluster.not_completed() == 0;
                let kernels_completed = self
                    .running_kernels
                    .read()
                    .iter()
                    .filter_map(Option::as_ref)
                    .all(|(_, k)| k.no_more_blocks_to_run());

                let cluster_active = !(cores_completed && kernels_completed);
                active_clusters[cluster_id] = cluster_active;
                if !cluster_active {
                    continue;
                }

                let core_sim_order = cluster.core_sim_order.try_lock();
                for core_id in &*core_sim_order {
                    let mut core = cluster.cores[*core_id].write();
                    crate::timeit!("core::cycle", core.cycle(cycle));
                }
                // active_sms += cluster.num_active_sms();
            }

            #[cfg(debug_assertions)]
            {
                // sanity check that inactive clusters do no produce any messages
                for (cluster_id, cluster) in self.clusters.iter().enumerate() {
                    if active_clusters[cluster_id] {
                        continue;
                    }
                    let core_sim_order = cluster.core_sim_order.try_lock();
                    for core_id in &*core_sim_order {
                        let core = cluster.cores[*core_id].try_read();
                        let mem_port = core.mem_port.lock();
                        assert_eq!(mem_port.buffer.len(), 0);
                    }
                }
            }

            for (cluster_id, cluster) in self.clusters.iter().enumerate() {
                let mut core_sim_order = cluster.core_sim_order.try_lock();
                for core_id in &*core_sim_order {
                    let core = cluster.cores[*core_id].try_read();
                    let mut mem_port = core.mem_port.lock();
                    log::trace!(
                        "interconn buffer for core {:?}: {:?}",
                        core.id(),
                        mem_port
                            .buffer
                            .iter()
                            .map(
                                |ic::Packet {
                                     data: (_dest, fetch, _size),
                                     ..
                                 }| fetch.to_string()
                            )
                            .collect::<Vec<_>>()
                    );

                    for ic::Packet {
                        data: (dest, fetch, size),
                        time,
                    } in mem_port.buffer.drain(..)
                    {
                        // log::trace!(
                        // eprintln!(
                        //     "interconn push from core {:?}: {fetch} (cluster={:?}, core={:?})",
                        //     core.id(),
                        //     fetch.cluster_id,
                        //     fetch.core_id,
                        // );
                        assert_eq!(
                            dest,
                            self.config.mem_id_to_device_id(fetch.sub_partition_id())
                        );
                        self.interconn.push(
                            core.cluster_id,
                            dest,
                            ic::Packet { data: fetch, time },
                            size,
                        );
                    }
                }
                if !active_clusters[cluster_id] {
                    // do not advance core sim order if cluster is inactive
                    continue;
                }
                if let config::SchedulingOrder::RoundRobin = self.config.simt_core_sim_order {
                    // eprintln!(
                    //     "SERIAL: cluster {} is active in cycle {}",
                    //     cluster.cluster_id, cycle
                    // );
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
                self.issue_block_to_core(cycle)
            );

            // if false {
            //     let state = crate::DebugState {
            //         core_orders_per_cluster: self
            //             .clusters
            //             .iter()
            //             .map(|cluster| cluster.core_sim_order.lock().clone())
            //             .collect(),
            //         last_cluster_issue: *self.last_cluster_issue.lock(),
            //         last_issued_kernel: *self.last_issued_kernel.lock(),
            //         block_issue_next_core_per_cluster: self
            //             .clusters
            //             .iter()
            //             .map(|cluster| *cluster.block_issue_next_core.lock())
            //             .collect(),
            //     };
            //     self.states.push((cycle, state));
            // }

            // self.decrement_kernel_latency();
            // }

            // Depending on configuration, invalidate the caches
            // once all of threads are completed.

            let mut not_completed = 0;
            let mut all_threads_complete = true;
            if self.config.flush_l1_cache {
                log::debug!("flushing l1 caches");
                for cluster in &mut self.clusters {
                    let cluster_id = cluster.cluster_id;
                    let cluster_not_completed = cluster.not_completed();
                    log::trace!(
                        "cluster {}: {} threads not completed",
                        cluster_id,
                        cluster_not_completed
                    );
                    if cluster_not_completed == 0 {
                        cluster.cache_invalidate();
                    } else {
                        not_completed += cluster_not_completed;
                        all_threads_complete = false;
                    }
                }
                log::trace!(
                    "all threads completed: {} ({} not completed)",
                    all_threads_complete,
                    not_completed
                );
            }

            if self.config.flush_l2_cache {
                if !self.config.flush_l1_cache {
                    for cluster in &mut self.clusters {
                        if cluster.not_completed() > 0 {
                            all_threads_complete = false;
                            break;
                        }
                    }
                }

                if let Some(l2_config) = &self.config.data_cache_l2 {
                    if all_threads_complete {
                        log::debug!("flushing l2 caches");
                        if l2_config.inner.total_lines() > 0 {
                            for (i, mem_sub) in self.mem_sub_partitions.iter_mut().enumerate() {
                                let mut mem_sub = mem_sub.try_lock();
                                let num_dirty_lines_flushed = mem_sub.flush_l2();
                                log::debug!(
                                    "dirty lines flushed from L2 {} is {:?}",
                                    i,
                                    num_dirty_lines_flushed
                                );
                            }
                        }
                    }
                }
            }

            #[cfg(feature = "timings")]
            TIMINGS
                .lock()
                .entry("cycle::total")
                .or_default()
                .add(start_total.elapsed());
        }

        // cycle += 1;

        // self.debug_non_exit();
        cycle
    }

    #[allow(dead_code)]
    fn debug_non_exit(&self) {
        log::trace!(
            "all clusters completed: {}",
            self.clusters
                .iter()
                .any(|cluster| cluster.not_completed() > 0)
        );
        for core in self
            .clusters
            .iter()
            .flat_map(|cluster| cluster.cores.clone())
        {
            let core = core.try_read();
            let block_status: Vec<_> = core.block_status.iter().enumerate().collect();
            let block_status: Vec<_> = block_status
                .into_iter()
                .filter(|(_id, num_threads)| **num_threads > 0)
                .collect();
            log::trace!("core {:?}: blocks: {:?}", core.id(), block_status);
        }
        // dbg!(self
        //     .clusters
        //     .iter()
        //     .flat_map(|cluster| cluster.cores.clone())
        //     .map(|core| core.try_read().block_status)
        //     // .sorted_by_key(|(block_hw_id, _)| block_hw_id)
        //     .collect::<Vec<_>>());
        for (partition_id, partition) in self.mem_partition_units.iter().enumerate() {
            log::trace!(
                "partition unit {}: busy={}",
                partition_id,
                partition
                    .try_read()
                    .sub_partitions
                    .iter()
                    .any(|unit| unit.try_lock().busy())
            );
        }
        log::trace!("interconn busy: {}", self.interconn.busy());
        log::trace!("more blocks to run: {}", self.more_blocks_to_run());

        if self.interconn.busy() {
            for cluster_id in 0..self.config.num_simt_clusters {
                let queue = self
                    .interconn
                    .dest_queue(cluster_id)
                    .try_lock()
                    .iter()
                    .sorted_by_key(|fetch| fetch.addr())
                    .map(ToString::to_string)
                    .collect::<Vec<_>>();
                if !queue.is_empty() {
                    log::trace!(
                        "cluster {cluster_id:<3} icnt: [{:<3}] {:?}...",
                        queue.len(),
                        queue.iter().next(),
                    );
                }
            }
            for sub_id in 0..self.config.total_sub_partitions() {
                let mem_device = self.config.mem_id_to_device_id(sub_id);
                let queue = self
                    .interconn
                    .dest_queue(mem_device)
                    .try_lock()
                    .iter()
                    .sorted_by_key(|fetch| fetch.addr())
                    .map(ToString::to_string)
                    .collect::<Vec<_>>();
                if !queue.is_empty() {
                    log::trace!(
                        "sub     {sub_id:<3} icnt: [{:<3}] {:?}...",
                        queue.len(),
                        queue.iter().next()
                    );
                }
            }
        }
    }

    pub fn l2_used_bytes(&self) -> u64 {
        self.mem_sub_partitions
            .iter()
            .map(|sub| {
                let sub = sub.lock();
                if let Some(ref l2_cache) = sub.l2_cache {
                    l2_cache.num_used_bytes()
                } else {
                    0
                }
            })
            .sum()
    }

    pub fn l2_used_lines(&self) -> usize {
        self.mem_sub_partitions
            .iter()
            .map(|sub| {
                let sub = sub.lock();
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
        for sub in &self.mem_sub_partitions {
            let mut sub = sub.lock();
            if let Some(ref mut l2_cache) = sub.l2_cache {
                let chunk_size = SECTOR_SIZE;
                let num_chunks = (num_bytes as f64 / chunk_size as f64).ceil() as usize;

                for chunk in 0..num_chunks {
                    let addr = start_addr + (chunk as u64 * chunk_size as u64);
                    l2_cache.invalidate_addr(addr);
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
                let core = core.read();
                let ldst_unit = core.load_store_unit.lock();
                if let Some(ref l1_cache) = ldst_unit.data_l1 {
                    let num_lines_used = l1_cache.num_used_lines();
                    let num_lines = l1_cache.num_total_lines();
                    eprintln!(
                        "core {:>3}/{:<3}: L2D {:>5}/{:<5} lines used ({:2.2}%, {})",
                        core.core_id,
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
        let num_sub_partitions = self.mem_sub_partitions.len();
        for sub in self.mem_sub_partitions.iter() {
            let sub = sub.lock();
            if let Some(ref l2_cache) = sub.l2_cache {
                let num_lines_used = l2_cache.num_used_lines();
                let num_lines = l2_cache.num_total_lines();
                eprintln!(
                    "sub {:>3}/{:<3}: L2D {:>5}/{:<5} lines used ({:2.2}%, {})",
                    sub.id,
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

    pub fn write_l1_cache_state(&self, path: &Path) -> eyre::Result<()> {
        // open csv writer
        let writer = utils::fs::open_writable(path)?;
        let mut csv_writer = csv::WriterBuilder::new()
            .flexible(false)
            .from_writer(writer);

        for cluster in self.clusters.iter() {
            for core in cluster.cores.iter() {
                let core = core.read();
                let ldst_unit = core.load_store_unit.lock();
                if let Some(ref l1_cache) = ldst_unit.data_l1 {
                    l1_cache.write_state(&mut csv_writer)?;
                }
            }
        }
        Ok(())
    }

    pub fn write_l2_cache_state(&self, path: &Path) -> eyre::Result<()> {
        // open csv writer
        let writer = utils::fs::open_writable(path)?;
        let mut csv_writer = csv::WriterBuilder::new()
            .flexible(false)
            .from_writer(writer);

        for sub in self.mem_sub_partitions.iter() {
            let sub = sub.lock();
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
        self.allocations.write().insert(alloc_range, name);
    }

    /// Collect simulation statistics.
    pub fn stats(&self) -> stats::PerKernel {
        let mut stats: stats::PerKernel = self.stats.lock().clone();

        let is_release_build = !is_debug();
        stats.no_kernel.sim.is_release_build = is_release_build;

        for (kernel_launch_id, kernel_stats) in stats.as_mut().iter_mut().enumerate() {
            if let Some(kernel) = &self.executed_kernels.lock().get(&(kernel_launch_id as u64)) {
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
        macro_rules! per_kernel_cache_stats {
            ($cache:expr) => {{
                $cache
                    .per_kernel_stats()
                    .try_lock()
                    .as_ref()
                    .iter()
                    .enumerate()
            }};
        }

        let cores = self
            .clusters
            .iter()
            .flat_map(|cluster| cluster.cores.clone());
        for core in cores {
            let core = core.try_read();
            for (kernel_launch_id, cache_stats) in per_kernel_cache_stats!(core.instr_l1_cache) {
                let kernel_stats = stats.get_mut(Some(kernel_launch_id));
                kernel_stats.l1i_stats[core.core_id] = cache_stats.clone();
            }

            let ldst_unit = &core.load_store_unit.try_lock();
            let data_l1 = ldst_unit.data_l1.as_ref().unwrap();
            for (kernel_launch_id, cache_stats) in per_kernel_cache_stats!(data_l1) {
                let kernel_stats = stats.get_mut(Some(kernel_launch_id));
                kernel_stats.l1d_stats[core.core_id] = cache_stats.clone();
            }
        }

        for sub in &self.mem_sub_partitions {
            let sub = sub.try_lock();
            let l2_cache = sub.l2_cache.as_ref().unwrap();
            for (kernel_launch_id, cache_stats) in per_kernel_cache_stats!(l2_cache) {
                let kernel_stats = stats.get_mut(Some(kernel_launch_id));
                kernel_stats.l2d_stats[sub.id] = cache_stats.clone();
            }
            stats.no_kernel.l2d_stats[sub.id] =
                l2_cache.per_kernel_stats().try_lock().no_kernel.clone();
        }
        stats
    }

    /// Process commands
    ///
    /// Take as many commands as possible until we have collected as many kernels to fill
    /// the `window_size` or processed every command.
    #[must_use]
    pub fn process_commands(&mut self, mut cycle: u64) -> u64 {
        let mut allocations_and_memcopies = Vec::new();
        while self.kernels.len() < self.kernel_window_size && self.command_idx < self.commands.len()
        {
            let cmd = self.commands[self.command_idx].clone();
            match cmd {
                // Command::MemcpyHtoD(trace_model::command::MemcpyHtoD {
                //     allocation_name,
                //     dest_device_addr,
                //     num_bytes,
                // }) => {
                //     cycle = crate::timeit!(
                //         "cycle::memcopy",
                //         self.memcopy_to_gpu(
                //             *dest_device_addr,
                //             *num_bytes,
                //             allocation_name.clone(),
                //             cycle,
                //             false,
                //         )
                //     );
                // }
                // Command::MemAlloc(trace_model::command::MemAlloc {
                //     allocation_name,
                //     device_ptr,
                //     fill_l2,
                //     num_bytes,
                // }) => {
                //     let fill_l2 = *fill_l2;
                //     let device_ptr = *device_ptr;
                //     let num_bytes = *num_bytes;
                //     let allocation_name = allocation_name.clone();
                //     self.gpu_mem_alloc(device_ptr, num_bytes, allocation_name.clone(), cycle);
                //     let has_memcopy = self.commands.iter().any(|cmd| match cmd {
                //         Command::MemcpyHtoD(trace_model::command::MemcpyHtoD {
                //             dest_device_addr,
                //             ..
                //         }) => {
                //             device_ptr <= *dest_device_addr
                //                 && *dest_device_addr < device_ptr + num_bytes
                //         }
                //         _ => false,
                //     });
                //
                //     if (true || fill_l2) && !has_memcopy {
                //         cycle = crate::timeit!(
                //             "cycle::memcopy",
                //             self.memcopy_to_gpu(
                //                 device_ptr,
                //                 num_bytes,
                //                 allocation_name.clone(),
                //                 cycle,
                //                 true,
                //             )
                //         );
                //     }
                // }
                Command::KernelLaunch(launch) => {
                    self.handle_allocations_and_memcopies(&allocations_and_memcopies, cycle);

                    // TODO: clean up this mess
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

                    let mut kernel = kernel::trace::KernelTrace::new(
                        launch.clone(),
                        self.traces_dir.as_ref().unwrap(),
                    );
                    kernel.memory_only = self.config.memory_only;
                    // let num_running_kernels = self
                    //     .running_kernels
                    //     .try_read()
                    //     .iter()
                    //     .map(Option::as_ref)
                    //     .filter(Option::is_some)
                    //     .count();
                    eprintln!("kernel launch {}: {:#?}", launch.id, &launch);
                    // todo!("temp");
                    let num_launched_kernels = self.executed_kernels.lock().len();

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
                            self.kernels.push_back(Arc::new(kernel));
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
                            self.kernels.push_back(Arc::new(kernel));
                        }
                    }
                }
                cmd => allocations_and_memcopies.push(cmd),
            }
            // cycle += 1;
            self.command_idx += 1;
        }
        let allocations = self.allocations.read();
        log::info!(
            "allocations: {:#?}",
            allocations
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
        log::trace!("launching kernels");
        let mut launch_queue: Vec<Arc<dyn Kernel>> = Vec::new();
        for kernel in &self.kernels {
            let stream_busy = self
                .busy_streams
                .iter()
                .any(|stream_id| *stream_id == kernel.config().stream_id);
            if !stream_busy && self.can_start_kernel() && !kernel.launched() {
                self.busy_streams.push_back(kernel.config().stream_id);
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

    pub fn commands_left(&self) -> bool {
        self.command_idx < self.commands.len()
    }

    pub fn kernels_left(&self) -> bool {
        !self.kernels.is_empty()
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

        while (self.commands_left() || self.kernels_left()) && !self.reached_limit(cycle) {
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

                // dbg!(&self.active());
                // dbg!(&self.reached_limit(cycle));
                // dbg!(&self.commands_left());
                // dbg!(&self.kernels_left());

                if self.reached_limit(cycle) {
                    // || !self.active() {
                    break;
                }

                let old_cycle = cycle;
                cycle = self.cycle(cycle);
                assert!(cycle >= old_cycle);

                if !self.active() {
                    finished_kernel = self.finished_kernel();
                    if finished_kernel.is_some() {
                        break;
                    }
                }

                match self.log_after_cycle {
                    Some(ref log_after_cycle) if cycle >= *log_after_cycle => {
                        eprintln!("initializing logging after cycle {cycle}");
                        init_logging();
                        self.log_after_cycle.take();
                        let allocations = self.allocations.read();
                        for (_, alloc) in allocations.iter() {
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
                self.commands_left(),
                self.kernels_left()
            );
        }
        self.stats.lock().no_kernel.sim.cycles = cycle;

        if let Some(log_after_cycle) = self.log_after_cycle {
            if log_after_cycle >= cycle {
                eprintln!("WARNING: log after {log_after_cycle} cycles but simulation ended after {cycle} cycles");
            }
        }
        log::info!("exit after {cycle} cycles");
        Ok(())
    }

    fn finished_kernel(&mut self) -> Option<Arc<dyn Kernel>> {
        // check running kernels
        let mut running_kernels = self.running_kernels.try_write().clone();
        let finished_kernel: Option<&mut Option<(_, Arc<dyn Kernel>)>> =
            running_kernels.iter_mut().find(|kernel| match kernel {
                // TODO: could also check here if !self.active()
                Some((_, k)) => {
                    // dbg!(
                    //     &self.active(),
                    //     &k.no_more_blocks_to_run(),
                    //     &k.running(),
                    //     &k.num_running_blocks(),
                    //     &k.launched()
                    // );
                    k.no_more_blocks_to_run() && !k.running() && k.launched()
                }
                _ => false,
            });
        // running_kernels.iter_mut().find_map(|kernel| match kernel {
        //     // TODO: could also check here if !self.active()
        //     Some((_, k)) if k.no_more_blocks_to_run() && !k.running() && k.launched() => {
        //         Some(kernel)
        //     }
        //     _ => None,
        // });
        finished_kernel.and_then(Option::take).map(|(_, k)| k)
        // if let Some(kernel) = finished_kernel {
        //     kernel.take().1
        // } else {
        //     None
        // }
    }

    fn cleanup_finished_kernel(&mut self, kernel: &dyn Kernel, cycle: u64) {
        // panic!("cleanup finished kernel {}", kernel.name());
        self.kernels.retain(|k| k.id() != kernel.id());
        self.busy_streams
            .retain(|stream| *stream != kernel.config().stream_id);

        kernel.set_completed(cycle);
        // let completion_time = std::time::Instant::now();
        // *kernel.completed_time.lock() = Some(completion_time);
        // *kernel.completed_cycle.lock() = Some(cycle);

        let mut stats = self.stats.lock();
        let kernel_stats = stats.get_mut(Some(kernel.id() as usize));

        kernel_stats.sim.is_release_build = !is_debug();
        // let elapsed_cycles = cycle - kernel.start_cycle.lock().unwrap_or(0);
        let elapsed_cycles = kernel.elapsed_cycles().unwrap_or(0);
        kernel_stats.sim.cycles = elapsed_cycles;
        let elapsed = kernel.elapsed_time();
        // let elapsed = kernel
        //     .start_time
        //     .lock()
        //     .map(|start_time| completion_time.duration_since(start_time));

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
}

impl<I, MC> MockSimulator<I, MC>
where
    I: ic::Interconnect<ic::Packet<mem_fetch::MemFetch>>,
    MC: mcu::MemoryController,
{
    fn copy_chunk_to_gpu(&self, write_addr: address, time: u64) {
        let num_sub_partitions = self.config.num_sub_partitions_per_memory_controller;
        let tlx_addr = self.mem_controller.to_physical_address(write_addr);
        let partition_id = tlx_addr.sub_partition / num_sub_partitions as u64;
        let sub_partition_id = tlx_addr.sub_partition % num_sub_partitions as u64;

        let partition = &self.mem_partition_units[partition_id as usize];

        let mut sector_mask: mem_fetch::SectorMask = BitArray::ZERO;
        // Sector chunk size is 4, so we get the highest 4 bits of the address
        // to set the sector mask
        sector_mask.set(((write_addr % 128) as u8 / 32) as usize, true);

        log::trace!(
            "memcopy to gpu: copy 32 byte chunk starting at {} to sub partition unit {} of partition unit {} ({}) (mask {})",
            write_addr,
            sub_partition_id,
            partition_id,
            tlx_addr.sub_partition,
            sector_mask.to_bit_string()
        );

        partition.try_read().handle_memcpy_to_gpu(
            write_addr,
            tlx_addr.sub_partition as usize,
            &sector_mask,
            time,
        );
    }

    /// Simulate memory copy to simulated device.
    // #[allow(clippy::needless_pass_by_value)]
    pub fn handle_allocations_and_memcopies(
        &mut self,
        commands: &Vec<Command>,
        // addr: address,
        // num_bytes: u64,
        // name: Option<String>,
        mut cycle: u64,
        // force: bool,
    ) {
        // handle allocations (no copy)
        let (mut allocations, copies): (Vec<_>, Vec<_>) = commands
            .into_iter()
            .cloned()
            .partition_map(|cmd| match cmd {
                Command::MemAlloc(alloc) => itertools::Either::Left(alloc),
                Command::MemcpyHtoD(copy) => itertools::Either::Right(copy),
                Command::KernelLaunch(_launch) => unreachable!(),
            });

        // allocations.sort_by_key(|alloc| let allocation_id = self
        //         .allocations
        //         .read()
        //         .iter()
        //         .find(|(_, alloc)| alloc.start_addr == *device_ptr)
        //         .map(|(_, alloc)| alloc.id)
        //         .unwrap();)

        for trace_model::command::MemAlloc {
            allocation_name,
            device_ptr,
            num_bytes,
            ..
        } in &allocations
        {
            self.gpu_mem_alloc(*device_ptr, *num_bytes, allocation_name.clone(), cycle);
        }
        // for cmd in allocations {
        //     if let Command::MemAlloc(trace_model::command::MemAlloc {
        //         allocation_name,
        //         device_ptr,
        //         num_bytes,
        //         ..
        //     }) = cmd
        //     {
        //         self.gpu_mem_alloc(*device_ptr, *num_bytes, allocation_name.clone(), cycle);
        //     }
        // }

        let l2_cache_size_bytes = self
            .config
            .data_cache_l2
            .as_ref()
            .map(|l2| self.mem_sub_partitions.len() * l2.inner.total_bytes())
            .unwrap_or(0);
        let l2_prefetch_percent = self.config.l2_prefetch_percent.unwrap_or(100.0);
        let mut cache_bytes_used = self.l2_used_bytes();

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
                .read()
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

            // dedup
            // if valid_prefill_memcopies
            //     .iter()
            //     .any(|(have_start, have_num_bytes, _)| {
            //         *have_start <= start_addr && start_addr < *have_start + *have_num_bytes
            //     })
            // {
            //     // skip
            //     continue;
            // }

            // if allocation_id == 3 {
            //     // invalidate everything before
            //     self.invalidate_l2()
            //     // continue;
            // }

            // check if there is enough space for this allocation
            // let should_prefill = true;
            // let should_prefill = if valid_prefill_allocations.len() > 0 {
            //     // already have an allocation
            //     let percent =
            //         ((num_bytes + cache_bytes_used) as f32 / l2_cache_size_bytes as f32) * 100.0;
            //     percent <= 75.0
            // } else {
            //     // first allocation
            //     let percent =
            //         ((cache_bytes_used + num_bytes) as f32 / l2_cache_size_bytes as f32) * 100.0;
            //     percent <= l2_prefetch_percent
            //     // if percent > l2_prefetch_percent {
            //     //     should_prefill = false
            //     // }
            // };
            //
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
            // let end_addr = utils::next_multiple(device_ptr + num_bytes, alignment);
            // assert!(
            //     end_addr % alignment == 0,
            //     "end address {} for allocation {} is not {} aligned",
            //     allocation_id,
            //     end_addr,
            //     human_bytes::human_bytes(alignment as f32),
            // );

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

            // what does it take to make LRU miss all sets when going over capacity
            // 12 partitions *128 sets * 128B line/1024 = 129KB

            // if *fill_l2 || (copy.is_some() && should_prefill) {
            let force_fill = fill_l2 == trace_model::command::L2Prefill::Force;
            let disabled = fill_l2 == trace_model::command::L2Prefill::NoPrefill;

            if force_fill || (!disabled && should_prefill) {
                valid_prefill_memcopies.push((start_addr, valid_num_bytes, allocation_id));
                cache_bytes_used += valid_num_bytes;
            }
        }

        // if false {
        //     for cmd in commands.iter().rev() {
        //         #[derive(Debug, PartialEq, Eq)]
        //         enum AllocationKind {
        //             MemCopy,
        //             Alloc,
        //         }
        //         let (force, start_addr, num_bytes, kind) = match cmd {
        //             Command::MemcpyHtoD(trace_model::command::MemcpyHtoD {
        //                 allocation_name,
        //                 dest_device_addr,
        //                 num_bytes,
        //             }) => {
        //                 // if *num_bytes > 64 {
        //                 //     eprintln!(
        //                 //         "CUDA mem copy: {:<20} {:>15} ({:>5} f32) to address {dest_device_addr:>20}",
        //                 //         allocation_name.as_deref().unwrap_or("<unnamed>"),
        //                 //         human_bytes::human_bytes(*num_bytes as f64),
        //                 //         *num_bytes / 4,
        //                 //     );
        //                 // }
        //                 (
        //                     trace_model::command::L2Prefill::Auto,
        //                     *dest_device_addr,
        //                     *num_bytes,
        //                     AllocationKind::MemCopy,
        //                 )
        //             }
        //             Command::MemAlloc(trace_model::command::MemAlloc {
        //                 allocation_name,
        //                 device_ptr,
        //                 fill_l2,
        //                 num_bytes,
        //                 ..
        //             }) => {
        //                 // if *num_bytes > 64 {
        //                 //     eprintln!(
        //                 //         "CUDA mem allocation: {:<20} {:>15} ({:>5} f32) to address {device_ptr:>20}",
        //                 //         allocation_name.as_deref().unwrap_or("<unnamed>"),
        //                 //         human_bytes::human_bytes(*num_bytes as f64),
        //                 //         *num_bytes / 4,
        //                 //     );
        //                 // }
        //                 // continue;
        //                 (*fill_l2, *device_ptr, *num_bytes, AllocationKind::Alloc)
        //             }
        //             _ => continue,
        //             // other => panic!("bad command {}", other),
        //         };
        //
        //         let allocation_id = self
        //             .allocations
        //             .read()
        //             .iter()
        //             .find(|(_, alloc)| alloc.start_addr == start_addr)
        //             .map(|(_, alloc)| alloc.id)
        //             .unwrap();
        //
        //         if num_bytes > 64 {
        //             eprintln!(
        //                 "CUDA mem {}: {:>15} ({:>5} f32) [allocation={}] to address {:>20}",
        //                 if kind == AllocationKind::MemCopy {
        //                     "copy"
        //                 } else {
        //                     "allocation"
        //                 },
        //                 // allocation_name.as_deref().unwrap_or("<unnamed>"),
        //                 human_bytes::human_bytes(num_bytes as f64),
        //                 num_bytes / 4,
        //                 allocation_id,
        //                 start_addr,
        //             );
        //         }
        //
        //         // dedup
        //         if valid_prefill_memcopies
        //             .iter()
        //             .any(|(have_start, have_num_bytes, _)| {
        //                 *have_start <= start_addr && start_addr < *have_start + *have_num_bytes
        //             })
        //         {
        //             // skip
        //             continue;
        //         }
        //
        //         // if allocation_id == 3 {
        //         //     // invalidate everything before
        //         //     self.invalidate_l2()
        //         //     // continue;
        //         // }
        //
        //         // check if there is enough space for this allocation
        //         // let should_prefill = true;
        //         // let should_prefill = if valid_prefill_allocations.len() > 0 {
        //         //     // already have an allocation
        //         //     let percent =
        //         //         ((num_bytes + cache_bytes_used) as f32 / l2_cache_size_bytes as f32) * 100.0;
        //         //     percent <= 75.0
        //         // } else {
        //         //     // first allocation
        //         //     let percent =
        //         //         ((cache_bytes_used + num_bytes) as f32 / l2_cache_size_bytes as f32) * 100.0;
        //         //     percent <= l2_prefetch_percent
        //         //     // if percent > l2_prefetch_percent {
        //         //     //     should_prefill = false
        //         //     // }
        //         // };
        //         //
        //         let should_prefill = {
        //             let percent = ((cache_bytes_used + num_bytes) as f32
        //                 / l2_cache_size_bytes as f32)
        //                 * 100.0;
        //             // percent <= l2_prefetch_percent
        //             // should be less than 75 percent
        //             percent <= 60.0
        //         };
        //
        //         if force || should_prefill {
        //             valid_prefill_memcopies.push((start_addr, num_bytes, allocation_id));
        //             cache_bytes_used += num_bytes;
        //         }
        //     }
        // }

        // let valid_prefill_allocations: Vec<_> = commands
        //     .iter()
        //     .rev()
        //     .take_while(|cmd| {
        //     let (start_address, num_bytes) = match cmd {
        //         Command::MemcpyHtoD(trace_model::command::MemcpyHtoD {
        //             dest_device_addr,
        //             num_bytes,
        //             ..
        //         }) => (dest_device_addr, num_bytes),
        //         Command::MemAlloc(trace_model::command::MemAlloc {
        //             device_ptr,
        //             fill_l2,
        //             num_bytes,
        //             ..
        //         }) => {
        //             if *fill_l2 {
        //                 return true;
        //             }
        //             (device_ptr, num_bytes)
        //         }
        //         other => panic!("bad command {}", other),
        //     };
        //     false
        // })
        // .collect();

        eprintln!(
            "have {} valid memcopies",
            valid_prefill_memcopies
                .iter()
                .filter(|(_, num_bytes, _)| *num_bytes > 64)
                .count(),
            // .filter_map(|(_, num_bytes)| if *num_bytes > 64 {
            //     Some(*num_bytes)
            // } else {
            //     None
            // })
            // .sum::<u64>(),
            // valid_prefill_allocations
        );

        // let mut completed_prefills: Vec<(u64, u64, usize)> = Vec::new();
        for (start_addr, num_bytes, allocation_id) in valid_prefill_memcopies
            .into_iter()
            .sorted_by_key(|(_, _, allocation_id)| *allocation_id)
        {
            // if allocation_id == 3 {
            //     for (prev_start, prev_num_bytes, _) in completed_prefills.drain(..) {
            //         self.invalidate_l2(prev_start, prev_num_bytes);
            //     }
            //     // let (prev_start, prev_num_bytes, _) = &valid_prefill_allocations[0];
            //     // self.invalidate_l2(*prev_start, *prev_num_bytes);
            //     // let (prev_start, prev_num_bytes, _) = &valid_prefill_allocations[1];
            //     // self.invalidate_l2(*prev_start, *prev_num_bytes);
            //     eprintln!(
            //         "post invalidation: bytes used={}",
            //         human_bytes::human_bytes(self.l2_used_bytes() as f64)
            //     );
            // }

            cycle = crate::timeit!(
                "cycle::memcopy",
                self.memcopy_to_gpu(
                    start_addr, num_bytes,
                    cycle,
                    // *dest_device_addr,
                    // *num_bytes,
                    // allocation_name.clone(),
                    // false,
                )
            );

            // completed_prefills.push((start_addr, num_bytes, allocation_id));
        }

        //     match cmd {
        //         // Command::MemcpyHtoD(trace_model::command::MemcpyHtoD {
        //         //     allocation_name,
        //         //     dest_device_addr,
        //         //     num_bytes,
        //         // }) => {}
        //         Command::MemAlloc(trace_model::command::MemAlloc {
        //             allocation_name,
        //             device_ptr,
        //             fill_l2,
        //             num_bytes,
        //         }) => {
        //             // let fill_l2 = *fill_l2;
        //             // let device_ptr = *device_ptr;
        //             // let num_bytes = *num_bytes;
        //             // let allocation_name = allocation_name.clone();
        //             self.gpu_mem_alloc(*device_ptr, *num_bytes, allocation_name.clone(), cycle);
        //             // let has_memcopy = self.commands.iter().any(|cmd| match cmd {
        //             //     Command::MemcpyHtoD(trace_model::command::MemcpyHtoD {
        //             //         dest_device_addr,
        //             //         ..
        //             //     }) => {
        //             //         device_ptr <= *dest_device_addr
        //             //             && *dest_device_addr < device_ptr + num_bytes
        //             //     }
        //             //     _ => false,
        //             // });
        //         }
        //         other => panic!("bad command {}", other),
        //     }
        // }
    }

    /// Simulate memory copy to simulated device.
    #[allow(clippy::needless_pass_by_value)]
    #[must_use]
    pub fn memcopy_to_gpu(
        &mut self,
        addr: address,
        num_bytes: u64,
        // name: Option<String>,
        mut cycle: u64,
        // force: bool,
    ) -> u64 {
        // {:<20}
        log::info!(
            "CUDA mem copy: {:>15} ({:>5} f32) to address {addr:>20}",
            // name.as_deref().unwrap_or("<unnamed>"),
            human_bytes::human_bytes(num_bytes as f64),
            num_bytes / 4,
        );
        // let alloc_range = addr..(addr + num_bytes);
        // self.allocations.write().insert(alloc_range, name);

        // let print_cache = |sim: &Self| {
        //     let mut num_total_lines = 0;
        //     let mut num_total_lines_used = 0;
        //     let num_sub_partitions = sim.mem_sub_partitions.len();
        //     for sub in sim.mem_sub_partitions.iter() {
        //         let sub = sub.lock();
        //         if let Some(ref l2_cache) = sub.l2_cache {
        //             let num_lines_used = l2_cache.num_used_lines();
        //             let num_lines = l2_cache.num_total_lines();
        //             eprintln!(
        //                 "sub {:>3}/{:<3}: L2D {:>5}/{:<5} lines used ({:2.2}%, {})",
        //                 sub.id,
        //                 num_sub_partitions,
        //                 num_lines_used,
        //                 num_lines,
        //                 num_lines_used as f32 / num_lines as f32 * 100.0,
        //                 human_bytes::human_bytes(num_lines_used as f64 * 128.0),
        //             );
        //             num_total_lines += num_lines;
        //             num_total_lines_used += num_lines_used;
        //         }
        //     }
        //     eprintln!(
        //         "Total L2D {:>5}/{:<5} lines used ({:2.2}%, {})",
        //         num_total_lines_used,
        //         num_total_lines,
        //         num_total_lines_used as f32 / num_total_lines as f32 * 100.0,
        //         human_bytes::human_bytes(num_total_lines_used as f64 * 128.0),
        //     );
        //     let stats = sim.stats();
        //     for (kernel_launch_id, kernel_stats) in stats.as_ref().iter().enumerate() {
        //         eprintln!(
        //             "L2D[kernel {}]: {:#?}",
        //             kernel_launch_id,
        //             &kernel_stats.l2d_stats.reduce()
        //         );
        //     }
        //     eprintln!("L2D[no kernel]: {:#?}", &stats.no_kernel.l2d_stats.reduce());
        //     eprintln!("DRAM[no kernel]: {:#?}", &stats.no_kernel.dram.reduce());
        // };

        // let write_l2_cache_state = |sim: &Self, path: &Path| -> eyre::Result<()> {
        //     // open csv writer
        //     let writer = utils::fs::open_writable(path)?;
        //     let mut csv_writer = csv::WriterBuilder::new()
        //         .flexible(false)
        //         .from_writer(writer);
        //
        //     for sub in sim.mem_sub_partitions.iter() {
        //         let sub = sub.lock();
        //         if let Some(ref l2_cache) = sub.l2_cache {
        //             l2_cache.write_state(&mut csv_writer)?;
        //         }
        //     }
        //     Ok(())
        // };

        if self.config.fill_l2_on_memcopy {
            if self.config.accelsim_compat {
                // todo: remove this branch because accelsim is broken
                let chunk_size: u64 = 32;
                let chunks = (num_bytes as f64 / chunk_size as f64).ceil() as usize;
                for chunk in 0..chunks {
                    let write_addr = addr + (chunk as u64 * chunk_size);
                    self.copy_chunk_to_gpu(write_addr, cycle);
                }
            } else {
                if let Some(ref l2_cache) = self.config.data_cache_l2 {
                    // let start = std::time::Instant::now();
                    let l2_cache_size_bytes =
                        self.mem_sub_partitions.len() * l2_cache.inner.total_bytes();

                    let bytes_used = self.l2_used_bytes();
                    // let bytes_used_percent =
                    //     (bytes_used as f32 / l2_cache_size_bytes as f32) * 100.0;

                    // find allocation
                    let allocation_id = self
                        .allocations
                        .read()
                        .iter()
                        .find(|(_, alloc)| {
                            // eprintln!(
                            //     "alloc {:>2}: {:>10} to {:>10}",
                            //     alloc.id,
                            //     alloc.start_addr,
                            //     alloc.end_addr.unwrap_or(alloc.start_addr)
                            // );
                            alloc.contains(addr)
                        })
                        .map(|(_, alloc)| alloc.id)
                        .unwrap();

                    // let size_below_threshold = self
                    //     .config
                    //     .l2_prefetch_percent
                    //     .map(|l2_prefetch_percent| percent <= l2_prefetch_percent)
                    //     .unwrap_or(true);

                    // let should_prefetch = force || (size_below_threshold && allocation_id == 1);
                    // let should_prefetch = force || (size_below_threshold && allocation_id == 3);
                    // * self.config.data_cache_l2.map(|l2| l2.line_size());
                    // let should_prefetch =
                    //     force || (bytes_used_percent <= 25.0 && size_below_threshold);
                    // let should_prefetch = force || allocation_id == 3;
                    // let should_prefetch = force || size_below_threshold;
                    // let should_prefetch = false;
                    // let should_prefetch = allocation_id != 3;

                    let max_bytes = num_bytes;
                    // let max_bytes = num_bytes.min(l2_cache_size_bytes as u64);
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

                    // if output_memcopy_l2_cache_state
                    //     && num_bytes > 64
                    //     && !self.config.accelsim_compat
                    // {
                    //     self.write_l2_cache_state(
                    //         &debug_dir.join(format!("l2_cache_state_{}_before.csv", allocation_id)),
                    //     )
                    //     .unwrap();
                    //     self.print_l2_cache();
                    // }

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

                    // if should_prefetch && num_bytes > 64 {
                    // eprintln!("memcopy completed in {:?}", start.elapsed());
                    // }
                }
            }
        }
        cycle
    }

    #[must_use]
    pub fn fill_l2(&mut self, addr: address, num_bytes: u64, mut cycle: u64) -> u64 {
        // assert!(
        //     addr % 256 == 0,
        //     "memcopy start address ({}) is not 256B aligned)",
        //     addr
        // );
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

            // let kind = mem_fetch::access::Kind::GLOBAL_ACC_W;
            let kind = mem_fetch::access::Kind::GLOBAL_ACC_R;
            let access = mem_fetch::access::Builder {
                kind,
                addr: chunk_addr,
                kernel_launch_id: None,
                allocation: self.allocations.try_read().get(&chunk_addr).cloned(),
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
                core_id: None,
                cluster_id: None,
                physical_addr,
            }
            .build();

            let dest_sub_partition_id = fetch.sub_partition_id();
            partition_hist[dest_sub_partition_id] += 1;
            let dest_mem_device = self.config.mem_id_to_device_id(dest_sub_partition_id);
            let packet_size = fetch.control_size();

            log::debug!("push transaction: {fetch} to device {dest_mem_device} (cluster_id={:?}, core_id={:?})", fetch.cluster_id, fetch.core_id);

            self.interconn.push(
                0,
                dest_mem_device,
                ic::Packet {
                    data: fetch,
                    time: 0,
                },
                packet_size,
            );
        }

        if num_bytes > 64 {
            eprintln!("memory partition histogram: {:?}", partition_hist);
        }

        let copy_start_cycle = cycle;
        while self.active() {
            for (sub_id, mem_sub) in self.mem_sub_partitions.iter().enumerate() {
                let mut mem_sub = mem_sub.try_lock();
                if let Some(fetch) = mem_sub.top() {
                    let response_packet_size = if fetch.is_write() {
                        fetch.control_size()
                    } else {
                        fetch.size()
                    };
                    let device = self.config.mem_id_to_device_id(sub_id);
                    if self.interconn.has_buffer(device, response_packet_size) {
                        mem_sub.pop().unwrap();
                    }
                }
            }

            for (sub_id, mem_sub) in self.mem_sub_partitions.iter_mut().enumerate() {
                let mut mem_sub = mem_sub.try_lock();
                let mem_sub_device = self.config.mem_id_to_device_id(sub_id);

                if mem_sub
                    .interconn_to_l2_queue
                    .can_fit(mem_sub_partition::NUM_SECTORS as usize)
                {
                    if let Some(packet) = self.interconn.pop(mem_sub_device) {
                        assert_eq!(packet.data.sub_partition_id(), sub_id);
                        mem_sub.push(packet.data, cycle);
                    }
                } else {
                    log::trace!("SKIP sub partition {sub_id} ({mem_sub_device}): DRAM full stall");
                }
                mem_sub.cycle(cycle);
            }

            for unit in self.mem_partition_units.iter_mut() {
                unit.try_write().simple_dram_cycle(cycle);
            }

            cycle += 1;
            let copy_duration = cycle - copy_start_cycle;
            if copy_duration > 0 && copy_duration % 10000 == 0 {
                log::warn!("fill l2 on memcopy: running for {copy_duration} cycles");
            }

            if log::log_enabled!(log::Level::Trace) {
                // dbg!(self
                //     .clusters
                //     .iter()
                //     .any(|cluster| cluster.try_read().not_completed() > 0));
                // for partition in self.mem_partition_units.iter() {
                //     dbg!(partition
                //         .try_read()
                //         .sub_partitions
                //         .iter()
                //         .any(|unit| unit.try_lock().busy()));
                // }
                // dbg!(self.interconn.busy());

                if self.interconn.busy() {
                    for cluster_id in 0..self.config.num_simt_clusters {
                        let queue = self
                            .interconn
                            .dest_queue(cluster_id)
                            .try_lock()
                            .iter()
                            .sorted_by_key(|fetch| fetch.addr())
                            .map(ToString::to_string)
                            .collect::<Vec<_>>();
                        if !queue.is_empty() {
                            log::trace!(
                                "cluster {cluster_id:<3} icnt: [{:<3}] {queue:?}",
                                queue.len()
                            );
                        }
                    }
                    for sub_id in 0..self.config.total_sub_partitions() {
                        let mem_device = self.config.mem_id_to_device_id(sub_id);
                        let queue = self
                            .interconn
                            .dest_queue(mem_device)
                            .try_lock()
                            .iter()
                            .sorted_by_key(|fetch| fetch.addr())
                            .map(ToString::to_string)
                            .collect::<Vec<_>>();
                        if !queue.is_empty() {
                            log::trace!("sub     {sub_id:<3} icnt: [{:<3}] {queue:?}", queue.len());
                        }
                    }
                }
                // dbg!(self.more_blocks_to_run());
                // dbg!(self.active());
            }
        }

        // OPTIONAL: check prefilled chunks are HIT (debugging)
        // if log::log_enabled!(log::Level::Trace) {
        if false {
            for chunk in 0..num_chunks {
                let chunk_addr = addr + (chunk as u64 * chunk_size);

                let access = mem_fetch::access::Builder {
                    kind: mem_fetch::access::Kind::GLOBAL_ACC_R,
                    addr: chunk_addr,
                    kernel_launch_id: None,
                    allocation: self.allocations.try_read().get(&chunk_addr).cloned(),
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
                    core_id: None,
                    cluster_id: None,
                    physical_addr,
                }
                .build();

                let mut write_fetch = fetch.clone();
                write_fetch.access.kind = mem_fetch::access::Kind::GLOBAL_ACC_W;
                write_fetch.access.is_write = true;
                let read_fetch = fetch;

                let dest_sub_partition_id = read_fetch.sub_partition_id();
                let sub = &self.mem_sub_partitions[dest_sub_partition_id];
                if let Some(ref mut l2_cache) = sub.lock().l2_cache {
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

    // debugging config
    // let config = Arc::new(config::GPUConfig {
    //     num_simt_clusters: 20,                   // 20
    //     num_cores_per_simt_cluster: 4,           // 1
    //     num_schedulers_per_core: 2,              // 1
    //     num_memory_controllers: 8,               // 8
    //     num_sub_partition_per_memory_channel: 2, // 2
    //     fill_l2_on_memcopy: true,                // true
    //     ..config::GPUConfig::default()
    // });

    // let interconn = Arc::new(ic::ToyInterconnect::new(
    //     config.num_simt_clusters,
    //     config.total_sub_partitions(),
    //     // config.num_memory_controllers * config.num_sub_partitions_per_memory_controller,
    // ));
    // let mut sim = MockSimulator::new(interconn, Arc::clone(&config));

    // match config.parallelization {
    //     config::Parallelization::Serial => {
    //         sim.run_to_completion()?;
    //     }
    //     #[cfg(feature = "parallel")]
    //     config::Parallelization::Deterministic => sim.run_to_completion_parallel_deterministic()?,
    //     #[cfg(feature = "parallel")]
    //     config::Parallelization::Nondeterministic(n) => {
    //         sim.run_to_completion_parallel_nondeterministic(n)?;
    //     }
    // }
    let mut sim = config::GTX1080::new(config);

    sim.add_commands(commands_path, traces_dir)?;
    sim.run()?;
    Ok(sim)
}

pub fn init_logging() {
    // let mut log_builder = env_logger::Builder::new();
    // log_builder.format(|buf, record| {
    //     writeln!(
    //         buf,
    //         // "{} [{}] - {}",
    //         "{}",
    //         // Local::now().format("%Y-%m-%dT%H:%M:%S"),
    //         // record.level(),
    //         record.args()
    //     )
    // });
    //
    // log_builder.parse_default_env();
    // log_builder.init();

    let mut log_builder = env_logger::Builder::new();
    log_builder.format(|buf, record| {
        use std::io::Write;
        let level_style = buf.default_level_style(record.level());
        writeln!(
            buf,
            "[ {}{}{} {} ] {}",
            // Local::now().format("%Y-%m-%dT%H:%M:%S"),
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
