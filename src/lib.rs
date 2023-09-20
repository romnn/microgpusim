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
// #![allow(warnings)]

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

pub use exec;

use self::core::{warp_inst_complete, Core, PipelineStage, MAX_THREAD_PER_SM, PROGRAM_MEM_START};
use allocation::Allocations;
use cluster::Cluster;
use engine::cycle::Component;
use interconn as ic;
use kernel::Kernel;
use trace_model::{Command, ToBitString};

use crate::sync::{atomic, Arc, Mutex, RwLock};
use bitvec::array::BitArray;
use color_eyre::eyre::{self};
use console::style;
use crossbeam::utils::CachePadded;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};

pub type address = u64;

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
// pub struct Optional<'a, T>(Option<&'a T>);
// pub struct Optional<'a, T>(Option<&'a T>);

// impl<T> std::fmt::Display for Optional<T>
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
        // match self.0 {
        //     Some(ref value) => write!(f, "Some({value})"),
        //     None => write!(f, "None"),
        // }
    }
}

#[derive(Default, Debug)]
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
    pub fn mean(&self) -> std::time::Duration {
        let nanos = u64::try_from(self.dur.as_nanos() / self.count).unwrap();
        std::time::Duration::from_nanos(nanos)
    }
}

pub static TIMINGS: once_cell::sync::Lazy<Mutex<HashMap<&'static str, TotalDuration>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(HashMap::default()));

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
pub struct MockSimulator<I> {
    /// Temporary for debugging
    // pub states: Vec<(u64, DebugState)>,
    stats: Arc<Mutex<stats::PerKernel>>,
    config: Arc<config::GPU>,
    // mem_partition_units: Vec<Arc<RwLock<dyn MemoryPartitionUnit>>>,
    mem_partition_units: Vec<Arc<RwLock<mem_partition_unit::MemoryPartitionUnit>>>,
    mem_sub_partitions: Vec<Arc<Mutex<mem_sub_partition::MemorySubPartition>>>,
    // we could remove the arcs on running and executed if we change to self: Arc<Self>
    pub running_kernels: Arc<RwLock<Vec<Option<Arc<kernel::Kernel>>>>>,
    // executed_kernels: Arc<Mutex<HashMap<u64, String>>>,
    executed_kernels: Arc<Mutex<HashMap<u64, Arc<kernel::Kernel>>>>,
    pub current_kernel: Mutex<Option<Arc<kernel::Kernel>>>,
    clusters: Vec<Arc<RwLock<Cluster<I>>>>,
    #[allow(dead_code)]
    warp_instruction_unique_uid: Arc<CachePadded<atomic::AtomicU64>>,
    interconn: Arc<I>,

    parallel_simulation: bool,
    last_cluster_issue: Arc<Mutex<usize>>,
    last_issued_kernel: Mutex<usize>,
    allocations: allocation::Ref,

    // for main run loop
    traces_dir: Option<PathBuf>,
    commands: Vec<Command>,
    command_idx: usize,
    kernels: VecDeque<Arc<kernel::Kernel>>,
    kernel_window_size: usize,
    busy_streams: VecDeque<u64>,
    cycle_limit: Option<u64>,
    log_after_cycle: Option<u64>,
    // gpu_stall_icnt2sh: usize,
    partition_replies_in_parallel: usize,
}

impl<I> std::fmt::Debug for MockSimulator<I> {
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

impl<I> MockSimulator<I>
where
    I: ic::Interconnect<ic::Packet<mem_fetch::MemFetch>>,
{
    pub fn new(interconn: Arc<I>, config: Arc<config::GPU>) -> Self {
        let stats = Arc::new(Mutex::new(stats::PerKernel::new(
            stats::Config::from_config(&config),
        )));

        let num_mem_units = config.num_memory_controllers;

        let mem_partition_units: Vec<_> = (0..num_mem_units)
            .map(|i| {
                let unit = mem_partition_unit::MemoryPartitionUnit::new(
                    i,
                    Arc::clone(&config),
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
                );
                Arc::new(RwLock::new(cluster))
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
        let mut kernels: VecDeque<Arc<kernel::Kernel>> = VecDeque::new();
        kernels.reserve_exact(window_size);

        let cycle_limit: Option<u64> = std::env::var("CYCLES")
            .ok()
            .as_deref()
            .map(str::parse)
            .and_then(Result::ok);

        // this causes first launch to use simt cluster
        let last_cluster_issue = Arc::new(Mutex::new(config.num_simt_clusters - 1));

        Self {
            // states: Vec::new(),
            config,
            stats,
            mem_partition_units,
            mem_sub_partitions,
            interconn,
            parallel_simulation: false,
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
    pub fn select_kernel(&self) -> Option<Arc<Kernel>> {
        let mut last_issued_kernel = self.last_issued_kernel.lock();
        let mut executed_kernels = self.executed_kernels.try_lock();
        let running_kernels = self.running_kernels.try_read();

        // issue same kernel again
        match running_kernels[*last_issued_kernel] {
            // && !kernel.kernel_TB_latency)
            Some(ref last_kernel) if !last_kernel.no_more_blocks_to_run() => {
                let launch_id = last_kernel.id();
                executed_kernels
                    .entry(launch_id)
                    // .or_insert(last_kernel.name().to_string());
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
            match running_kernels[idx] {
                // &&!kernel.kernel_TB_latency)
                Some(ref kernel) if !kernel.no_more_blocks_to_run() => {
                    *last_issued_kernel = idx;
                    let launch_id = kernel.id();
                    assert!(!executed_kernels.contains_key(&launch_id));
                    // executed_kernels.insert(launch_id, kernel.name().to_string());
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
            Some(kernel) => !kernel.no_more_blocks_to_run(),
            None => false,
        })
    }

    pub fn active(&self) -> bool {
        for cluster in &self.clusters {
            if cluster.try_read().not_completed() > 0 {
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
            Some(kernel) => kernel.done(),
            None => true,
        })
    }

    pub fn launch(&self, kernel: Arc<Kernel>, cycle: u64) -> eyre::Result<()> {
        // kernel.set_launched();
        // println!("launch kernel {} in cycle {}", kernel.id(), cycle);
        let threads_per_block = kernel.threads_per_block();
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
            .find(|slot| slot.is_none() || slot.as_ref().map_or(false, |k| k.done()))
            .ok_or(eyre::eyre!("no free slot for kernel"))?;

        *kernel.start_time.lock() = Some(std::time::Instant::now());
        *kernel.start_cycle.lock() = Some(cycle);

        *self.current_kernel.lock() = Some(Arc::clone(&kernel));
        *free_slot = Some(kernel);
        Ok(())
    }

    #[tracing::instrument]
    #[inline]
    fn issue_block_to_core(&self, cycle: u64) {
        log::debug!("===> issue block to core");
        let mut last_cluster_issue = self.last_cluster_issue.try_lock();
        let last_issued = *last_cluster_issue;
        let num_clusters = self.config.num_simt_clusters;
        for cluster_id in 0..num_clusters {
            let cluster_id = (cluster_id + last_issued + 1) % num_clusters;
            let cluster = self.clusters[cluster_id].read();
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
    }

    // pub fn set_cycle(&self, cycle: u64) {
    //     let mut stats = self.stats.lock();
    //     let kernel_stats = stats.get_mut(0);
    //     kernel_stats.sim.cycles = cycle;
    // }

    #[allow(clippy::overly_complex_bool_expr)]
    #[tracing::instrument(name = "cycle")]
    pub fn cycle(&mut self, cycle: u64) {
        #[cfg(feature = "timings")]
        let start_total = std::time::Instant::now();
        // int clock_mask = next_clock_domain();

        // shader core loading (pop from ICNT into core)
        #[cfg(feature = "timings")]
        let start = std::time::Instant::now();
        for cluster in &self.clusters {
            cluster.try_write().interconn_cycle(cycle);
        }
        #[cfg(feature = "timings")]
        TIMINGS
            .lock()
            .entry("cycle::interconn")
            .or_default()
            .add(start.elapsed());

        log::debug!(
            "POP from {} memory sub partitions",
            self.mem_sub_partitions.len()
        );

        // pop from memory controller to interconnect

        #[cfg(feature = "timings")]
        let start = std::time::Instant::now();
        for (i, mem_sub) in self.mem_sub_partitions.iter().enumerate() {
            let mut mem_sub = mem_sub.try_lock();
            {
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
                    let cluster_id = fetch.cluster_id;
                    fetch.set_status(mem_fetch::Status::IN_ICNT_TO_SHADER, 0);
                    // fetch.set_return_timestamp(gpu_sim_cycle + gpu_tot_sim_cycle);
                    // , gpu_sim_cycle + gpu_tot_sim_cycle);
                    // drop(fetch);
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
                } else {
                    // self.gpu_stall_icnt2sh += 1;
                }
            }
        }
        #[cfg(feature = "timings")]
        TIMINGS
            .lock()
            .entry("cycle::subs")
            .or_default()
            .add(start.elapsed());

        // DRAM
        #[cfg(feature = "timings")]
        let start = std::time::Instant::now();
        log::debug!("cycle for {} drams", self.mem_partition_units.len());
        for (_i, unit) in self.mem_partition_units.iter_mut().enumerate() {
            unit.try_write().simple_dram_cycle(cycle);

            // if self.config.simple_dram_model {
            //     unit.simple_dram_cycle();
            // } else {
            //     // Issue the dram command (scheduler + delay model)
            //     // unit.simple_dram_cycle();
            //     unimplemented!()
            // }
        }
        #[cfg(feature = "timings")]
        TIMINGS
            .lock()
            .entry("cycle::dram")
            .or_default()
            .add(start.elapsed());

        // L2 operations
        log::debug!(
            "moving mem requests from interconn to {} mem partitions",
            self.mem_sub_partitions.len()
        );

        #[cfg(feature = "timings")]
        let start = std::time::Instant::now();
        // let mut parallel_mem_partition_reqs_per_cycle = 0;
        // let mut stall_dram_full = 0;
        for (i, mem_sub) in self.mem_sub_partitions.iter_mut().enumerate() {
            let mut mem_sub = mem_sub.try_lock();
            // move memory request from interconnect into memory partition
            // (if not backed up)
            //
            // Note:This needs to be called in DRAM clock domain if there
            // is no L2 cache in the system In the worst case, we may need
            // to push SECTOR_CHUNCK_SIZE requests, so ensure you have enough
            // buffer for them
            let device = self.config.mem_id_to_device_id(i);

            // same as full with parameter overload
            if mem_sub
                .interconn_to_l2_queue
                .can_fit(mem_sub_partition::SECTOR_CHUNCK_SIZE as usize)
            {
                if let Some(packet) = self.interconn.pop(device) {
                    log::debug!(
                        "got new fetch {} for mem sub partition {} ({})",
                        packet.data,
                        i,
                        device
                    );

                    mem_sub.push(packet.data, cycle);
                    // self.parallel_mem_partition_reqs += 1;
                }
            } else {
                log::debug!("SKIP sub partition {} ({}): DRAM full stall", i, device);
                let mut stats = self.stats.lock();
                let kernel_stats = stats.get_mut(0);
                kernel_stats.stall_dram_full += 1;
            }
            // we borrow all of sub here, which is a problem for the cyclic reference in l2
            // interface
            mem_sub.cache_cycle(cycle);
        }
        #[cfg(feature = "timings")]
        TIMINGS
            .lock()
            .entry("cycle::l2")
            .or_default()
            .add(start.elapsed());

        //   partiton_reqs_in_parallel += partiton_reqs_in_parallel_per_cycle;
        // if (partiton_reqs_in_parallel_per_cycle > 0) {
        //   partiton_reqs_in_parallel_util += partiton_reqs_in_parallel_per_cycle;
        //   gpu_sim_cycle_parition_util++;
        // }

        // self.interconn_transfer();

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
            let cluster = cluster.try_read();
            let cores_completed = cluster.not_completed() == 0;
            let kernels_completed = self
                .running_kernels
                .read()
                .iter()
                .filter_map(std::option::Option::as_ref)
                .all(|k| k.no_more_blocks_to_run());

            let cluster_active = !(cores_completed && kernels_completed);
            active_clusters[cluster_id] = cluster_active;
            if !cluster_active {
                continue;
            }

            let core_sim_order = cluster.core_sim_order.try_lock();
            for core_id in &*core_sim_order {
                let mut core = cluster.cores[*core_id].write();
                crate::timeit!("serial core cycle", core.cycle(cycle));
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
                let cluster = cluster.try_read();
                let core_sim_order = cluster.core_sim_order.try_lock();
                for core_id in &*core_sim_order {
                    let core = cluster.cores[*core_id].try_read();
                    let port = core.mem_port.lock();
                    assert_eq!(port.buffer.len(), 0);
                }
            }
        }

        for (cluster_id, cluster) in self.clusters.iter().enumerate() {
            let cluster = cluster.try_read();
            let mut core_sim_order = cluster.core_sim_order.try_lock();
            for core_id in &*core_sim_order {
                let core = cluster.cores[*core_id].try_read();
                let mut port = core.mem_port.lock();
                for ic::Packet {
                    data: (dest, fetch, size),
                    time,
                } in port.buffer.drain(..)
                {
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
                // println!(
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

        crate::timeit!(self.issue_block_to_core(cycle));

        // if false {
        //     let state = crate::DebugState {
        //         core_orders_per_cluster: self
        //             .clusters
        //             .iter()
        //             .map(|cluster| cluster.read().core_sim_order.lock().clone())
        //             .collect(),
        //         last_cluster_issue: *self.last_cluster_issue.lock(),
        //         last_issued_kernel: *self.last_issued_kernel.lock(),
        //         block_issue_next_core_per_cluster: self
        //             .clusters
        //             .iter()
        //             .map(|cluster| *cluster.read().block_issue_next_core.lock())
        //             .collect(),
        //     };
        //     self.states.push((cycle, state));
        // }

        // self.decrement_kernel_latency();
        // }

        // Depending on configuration, invalidate the caches
        // once all of threads are completed.

        let mut all_threads_complete = true;
        if self.config.flush_l1_cache {
            for cluster in &mut self.clusters {
                if cluster.try_read().not_completed() == 0 {
                    cluster.try_write().cache_invalidate();
                } else {
                    all_threads_complete = false;
                }
            }
        }

        if self.config.flush_l2_cache {
            if !self.config.flush_l1_cache {
                for cluster in &mut self.clusters {
                    if cluster.try_read().not_completed() > 0 {
                        all_threads_complete = false;
                        break;
                    }
                }
            }

            if let Some(l2_config) = &self.config.data_cache_l2 {
                if all_threads_complete {
                    log::debug!("flushed L2 caches...");
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

    fn copy_chunk_to_gpu(&self, write_addr: address, time: u64) {
        let num_sub_partitions = self.config.num_sub_partitions_per_memory_controller;
        let tlx_addr = self
            .config
            .address_mapping()
            .to_physical_address(write_addr);
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
    #[allow(clippy::needless_pass_by_value)]
    pub fn memcopy_to_gpu(
        &mut self,
        addr: address,
        num_bytes: u64,
        name: Option<String>,
        cycle: u64,
    ) {
        log::info!(
            "CUDA mem copy: {:<20} {:>15} ({:>5} f32) to address {addr:>20}",
            name.as_deref().unwrap_or("<unnamed>"),
            human_bytes::human_bytes(num_bytes as f64),
            num_bytes / 4,
        );
        // let alloc_range = addr..(addr + num_bytes);
        // self.allocations.write().insert(alloc_range, name);
        //
        if self.config.fill_l2_on_memcopy {
            let chunk_size: u64 = 32;
            let chunks = (num_bytes as f64 / chunk_size as f64).ceil() as usize;
            for chunk in 0..chunks {
                let write_addr = addr + (chunk as u64 * chunk_size);
                self.copy_chunk_to_gpu(write_addr, cycle);
            }
        }
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
        let alloc_range = addr..(addr + num_bytes);
        self.allocations.write().insert(alloc_range, name);
    }

    /// Collect simulation statistics.
    pub fn stats(&self) -> stats::PerKernel {
        let mut stats: stats::PerKernel = self.stats.lock().clone();

        // compute on demand
        for (kernel_launch_id, kernel_stats) in stats.as_mut().iter_mut().enumerate() {
            // dbg!(&kernel_launch_id);
            // dbg!(&self
            //     .executed_kernels
            //     .lock()
            //     .iter()
            //     .map(|(i, k)| (i, k.to_string()))
            //     .collect::<Vec<_>>());

            let kernel = &self.executed_kernels.lock()[&(kernel_launch_id as u64)];
            let kernel_info = stats::KernelInfo {
                name: kernel.config.unmangled_name.clone(),
                mangled_name: kernel.config.mangled_name.clone(),
                launch_id: kernel_launch_id,
            };
            // let kernel = self
            //     .kernels
            //     .iter()
            //     .find(|k| k.id() == kernel_launch_id as u64)
            //     .unwrap();
            kernel_stats.sim.kernel_name = kernel_info.name.clone();
            kernel_stats.sim.kernel_name_mangled = kernel_info.mangled_name.clone();
            kernel_stats.sim.kernel_launch_id = kernel_info.launch_id;
            kernel_stats.sim.is_release_build = !is_debug();

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
            // for cache_stats in [&mut kernel_stats.l1i_stats]
            //     .into_iter()
            //     .flat_map(|cache| cache.iter_mut())
            // {
            //     // for cache.iter_mut()
            //     cache_stats.kernel_info = kernel_info.clone();
            //     // cache.kernel_name = kernel.config.unmangled_name.clone();
            // }

            // kernel_stats.accesses.kernel_info = stats::KernelInfo {
            //     name: kernel.config.unmangled_name.clone(),
            //     launch_id: kernel_launch_id,
            // }
        }
        //     // if kernel was launched, update stats
        //
        // let time = std::time::Instant::now();
        // *kernel.completed_time.lock() = Some(completion_time);
        // *kernel.completed_cycle.lock() = Some(cycle);

        // kernel_stats.sim.cycles = cycle - kernel.start_cycle.lock().unwrap_or(0);
        // kernel_stats.sim.elapsed_millis = kernel
        //     .start_time
        //     .lock()
        //     .map(|start_time| completion_time.duration_since(start_time).as_millis())
        //     .unwrap_or(0);
        // }

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
            .flat_map(|cluster| cluster.read().cores.clone());
        for core in cores {
            // for cluster in &self.clusters {
            //     let cluster = cluster.try_read();
            //     for core in &cluster.cores {
            let core = core.try_read();
            // let core_id = core.core_id;
            // todo:

            for (kernel_launch_id, cache_stats) in per_kernel_cache_stats!(core.instr_l1_cache) {
                //     .per_kernel_stats()
                //     .try_lock()
                //     .as_ref()
                //     .iter()
                //     .enumerate()
                // {
                let kernel_stats = stats.get_mut(kernel_launch_id);
                kernel_stats.l1i_stats[core.core_id] = cache_stats.clone();
            }
            // stats.l1i_stats[core.core_id] = core.instr_l1_cache.stats().try_lock().clone();

            let ldst_unit = &core.load_store_unit.try_lock();
            let data_l1 = ldst_unit.data_l1.as_ref().unwrap();
            for (kernel_launch_id, cache_stats) in per_kernel_cache_stats!(data_l1) {
                let kernel_stats = stats.get_mut(kernel_launch_id);
                kernel_stats.l1d_stats[core.core_id] = cache_stats.clone();
            }
            // stats.l1d_stats[core.core_id] = data_l1.stats().try_lock().clone();
            // stats.l1c_stats[core.core_id] = stats::Cache::default();
            // stats.l1t_stats[core.core_id] = stats::Cache::default();
            //     }
            // }
        }

        for sub in &self.mem_sub_partitions {
            let sub = sub.try_lock();
            let l2_cache = sub.l2_cache.as_ref().unwrap();
            for (kernel_launch_id, cache_stats) in per_kernel_cache_stats!(l2_cache) {
                let kernel_stats = stats.get_mut(kernel_launch_id);
                kernel_stats.l2d_stats[sub.id] = cache_stats.clone();
            }
            // stats.l2d_stats[sub.id] = l2_cache.stats().try_lock().clone();
        }
        stats
    }

    /// Process commands
    ///
    /// Take as many commands as possible until we have collected as many kernels to fill
    /// the `window_size` or processed every command.
    pub fn process_commands(&mut self, cycle: u64) {
        while self.kernels.len() < self.kernel_window_size && self.command_idx < self.commands.len()
        {
            let cmd = &self.commands[self.command_idx];
            match cmd {
                Command::MemcpyHtoD(trace_model::command::MemcpyHtoD {
                    allocation_name,
                    dest_device_addr,
                    num_bytes,
                }) => self.memcopy_to_gpu(
                    *dest_device_addr,
                    *num_bytes,
                    allocation_name.clone(),
                    cycle,
                ),
                Command::MemAlloc(trace_model::command::MemAlloc {
                    allocation_name,
                    device_ptr,
                    num_bytes,
                }) => {
                    self.gpu_mem_alloc(*device_ptr, *num_bytes, allocation_name.clone(), cycle);
                }
                Command::KernelLaunch(launch) => {
                    let mut kernel =
                        Kernel::from_trace(launch.clone(), self.traces_dir.as_ref().unwrap());
                    kernel.memory_only = self.config.memory_only;
                    self.kernels.push_back(Arc::new(kernel));
                }
            }
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
    }

    pub fn add_kernel(&mut self, kernel: Kernel) {
        self.kernels.push_back(Arc::new(kernel));
    }

    /// Lauch more kernels if possible.
    ///
    /// Launch all kernels within window that are on a stream that isn't already running
    pub fn launch_kernels(&mut self, cycle: u64) {
        log::trace!("launching kernels");
        let mut launch_queue: Vec<Arc<Kernel>> = Vec::new();
        for kernel in &self.kernels {
            let stream_busy = self
                .busy_streams
                .iter()
                .any(|stream_id| *stream_id == kernel.config.stream_id);
            if !stream_busy && self.can_start_kernel() && !kernel.launched() {
                self.busy_streams.push_back(kernel.config.stream_id);
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
                assert!(
                    kernel.config.id <= up_to_kernel,
                    "launching kernel {kernel}"
                );
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

    pub fn run(&mut self) -> eyre::Result<()> {
        dbg!(&self.config.parallelization);
        match self.config.parallelization {
            config::Parallelization::Serial => {
                self.run_to_completion()?;
            }
            #[cfg(feature = "parallel")]
            config::Parallelization::Deterministic => {
                self.run_to_completion_parallel_deterministic()?;
            }
            #[cfg(feature = "parallel")]
            config::Parallelization::Nondeterministic(n) => {
                self.run_to_completion_parallel_nondeterministic(n)?;
            }
        }
        Ok(())
    }

    #[tracing::instrument]
    pub fn run_to_completion(&mut self) -> eyre::Result<()> {
        let mut cycle: u64 = 0;
        let mut last_state_change: Option<(deadlock::State, u64)> = None;

        println!("serial for {} cores", self.config.total_cores());

        while (self.commands_left() || self.kernels_left()) && !self.reached_limit(cycle) {
            self.process_commands(cycle);
            self.launch_kernels(cycle);

            let mut finished_kernel = None;
            loop {
                log::info!("======== cycle {cycle} ========");
                log::info!("");

                if self.reached_limit(cycle) || !self.active() {
                    break;
                }

                self.cycle(cycle);
                cycle += 1;
                // self.set_cycle(cycle);
                // let mut stats = self.stats.lock();
                // let kernel_stats = stats.get_mut(0);
                // kernel_stats.sim.cycles += 1;

                if !self.active() {
                    finished_kernel = self.finished_kernel();
                    if finished_kernel.is_some() {
                        break;
                    }
                }

                match self.log_after_cycle {
                    Some(ref log_after_cycle) if cycle >= *log_after_cycle => {
                        use std::io::Write;

                        println!("initializing logging after cycle {cycle}");
                        let mut log_builder = env_logger::Builder::new();

                        log_builder.format(|buf, record| {
                            writeln!(
                                buf,
                                // "{} [{}] - {}",
                                "{}",
                                // Local::now().format("%Y-%m-%dT%H:%M:%S"),
                                // record.level(),
                                record.args()
                            )
                        });

                        log_builder.parse_default_env();
                        log_builder.init();

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

                    match &mut last_state_change {
                        Some((last_state, update_cycle)) if &state == last_state => {
                            panic!("deadlock after cycle {update_cycle}");
                        }
                        Some((ref mut last_state, ref mut update_cycle)) => {
                            // log::info!("deadlock check: updated state in cycle {}", cycle);
                            *last_state = state;
                            *update_cycle = cycle;
                        }
                        None => {
                            last_state_change = Some((state, cycle));
                        }
                    }
                }
            }

            if let Some(kernel) = finished_kernel {
                self.cleanup_finished_kernel(&kernel, cycle);
            }

            log::trace!(
                "commands left={} kernels left={}",
                self.commands_left(),
                self.kernels_left()
            );
        }
        if let Some(log_after_cycle) = self.log_after_cycle {
            if log_after_cycle >= cycle {
                eprintln!("WARNING: log after {log_after_cycle} cycles but simulation ended after {cycle} cycles");
            }
        }
        log::info!("exit after {cycle} cycles");
        Ok(())
    }

    fn finished_kernel(&mut self) -> Option<Arc<Kernel>> {
        // check running kernels
        // let _active = self.active();
        let mut running_kernels = self.running_kernels.try_write().clone();
        let finished_kernel: Option<&mut Option<Arc<Kernel>>> =
            running_kernels.iter_mut().find(|k| {
                if let Some(k) = k {
                    // TODO: could also check here if !self.active()
                    k.no_more_blocks_to_run() && !k.running() && k.launched()
                } else {
                    false
                }
            });
        if let Some(kernel) = finished_kernel {
            // kernel.end_cycle = self.cycle.get();
            kernel.take()
        } else {
            None
        }
    }

    fn cleanup_finished_kernel(&mut self, kernel: &Kernel, cycle: u64) {
        log::debug!(
            "cleanup finished kernel with id={}: {}",
            kernel.id(),
            kernel
        );
        // println!("completed kernel {} in cycle {}", kernel.id(), cycle);
        self.kernels.retain(|k| k.config.id != kernel.config.id);
        self.busy_streams
            .retain(|stream| *stream != kernel.config.stream_id);

        let completion_time = std::time::Instant::now();
        *kernel.completed_time.lock() = Some(completion_time);
        *kernel.completed_cycle.lock() = Some(cycle);

        let mut stats = self.stats.lock();
        let kernel_stats = stats.get_mut(kernel.id() as usize);

        // *kernel.completed_time.lock() = Some(completion_time);
        // *kernel.completed_cycle.lock() = Some(cycle);

        kernel_stats.sim.is_release_build = !is_debug();
        kernel_stats.sim.cycles = cycle - kernel.start_cycle.lock().unwrap_or(0);
        let elapsed_millis = kernel
            .start_time
            .lock()
            .map(|start_time| completion_time.duration_since(start_time).as_millis())
            .unwrap_or(0);

        kernel_stats.sim.elapsed_millis = elapsed_millis;
        // if is_debug() {
        //     kernel_stats.sim.elapsed_millis_debug = elapsed_millis;
        // } else {
        //     kernel_stats.sim.elapsed_millis_release = elapsed_millis;
        // }

        // println!(
        //     "stats len is now {} ({:#?})",
        //     stats.as_ref().len(),
        //     stats
        //         .as_ref()
        //         .iter()
        //         .map(|kernel_stats| &kernel_stats.sim)
        //         .collect::<Vec<_>>()
        // );

        // resets some statistics between kernel launches
        //   if (!silent && m_gpgpu_sim->gpu_sim_cycle > 0) {
        //   m_gpgpu_sim->update_stats();
        //   m_gpgpu_context->print_simulation_time();
        // }
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

// MockSimulator<ic::ToyInterconnect<ic::Packet<mem_fetch::MemFetch>>>

pub fn accelmain(
    traces_dir: impl AsRef<Path>,
    config: impl Into<Arc<config::GPU>>,
) -> eyre::Result<config::GTX1080> {
    #[cfg(feature = "deadlock_detection")]
    std::thread::spawn(move || loop {
        // Create a background thread which checks for deadlocks every 10s
        std::thread::sleep(std::time::Duration::from_secs(10));
        let deadlocks = parking_lot::deadlock::check_deadlock();
        if deadlocks.is_empty() {
            continue;
        }

        println!("{} deadlocks detected", deadlocks.len());
        for (i, threads) in deadlocks.iter().enumerate() {
            println!("Deadlock #{i}");
            for t in threads {
                println!("Thread Id {:#?}", t.thread_id());
                println!("{:#?}", t.backtrace());
            }
        }
    });

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

#[cfg(test)]
mod tests {}
