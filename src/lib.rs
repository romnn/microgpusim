#![allow(
    clippy::upper_case_acronyms,
    non_camel_case_types,
    clippy::too_many_arguments,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::too_many_lines
)]
// #![allow(warnings)]

pub mod addrdec;
pub mod allocation;
pub mod arbitration;
pub mod cache;
pub mod cluster;
pub mod config;
pub mod core;
pub mod deadlock;
pub mod dram;
pub mod exec;
pub mod fifo;
pub mod instruction;
pub mod interconn;
pub mod kernel;
pub mod ldst_unit;
pub mod mem_fetch;
pub mod mem_partition_unit;
pub mod mem_sub_partition;
pub mod mshr;
pub mod opcodes;
pub mod operand_collector;
pub mod register_set;
pub mod scheduler;
pub mod scoreboard;
pub mod set_index;
pub mod simd_function_unit;
pub mod sp_unit;
pub mod tag_array;
pub mod warp;

#[cfg(test)]
pub mod testing;

use self::cluster::SIMTCoreCluster;
use self::core::{
    warp_inst_complete, Packet, PipelineStage, SIMTCore, MAX_THREAD_PER_SM, PROGRAM_MEM_START,
};
use addrdec::DecodedAddress;
use allocation::Allocations;
use fifo::Queue;
use interconn as ic;
use kernel::Kernel;
use ldst_unit::LoadStoreUnit;
use mem_fetch::{AccessKind, BitString};
use sp_unit::SPUnit;
use stats::Stats;
use trace_model::Command;

use bitvec::array::BitArray;
use color_eyre::eyre::{self};
use console::style;
use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{atomic, Arc, Mutex, RwLock};
use std::time::Instant;

pub type address = u64;

pub fn parse_commands(path: impl AsRef<Path>) -> eyre::Result<Vec<Command>> {
    let reader = utils::fs::open_readable(path.as_ref())?;
    let commands = serde_json::from_reader(reader)?;
    Ok(commands)
}

pub type SubPartition = mem_sub_partition::MemorySubPartition<fifo::FifoQueue<mem_fetch::MemFetch>>;

#[derive(Default, Debug)]
pub struct TotalDuration {
    count: u32,
    dur: std::time::Duration,
}

impl TotalDuration {
    pub fn add(&mut self, dur: std::time::Duration) {
        self.count += 1;
        self.dur += dur;
    }

    #[must_use]
    pub fn mean(&self) -> std::time::Duration {
        self.dur / self.count
    }
}

use once_cell::sync::Lazy;
pub static TIMINGS: Lazy<Mutex<HashMap<&'static str, TotalDuration>>> =
    Lazy::new(|| Mutex::new(HashMap::default()));

#[macro_export]
macro_rules! timeit {
    ($name:expr, $call:expr) => {{
        let start = std::time::Instant::now();
        let res = $call;
        let dur = start.elapsed();
        let mut timings = $crate::TIMINGS.lock().unwrap();
        timings.entry($name).or_default().add(dur);
        res
    }};
    ($call:expr) => {{
        $crate::timeit!(stringify!($call), $call)
    }};
}

#[derive()]
pub struct MockSimulator<I> {
    stats: Arc<Mutex<Stats>>,
    config: Arc<config::GPUConfig>,
    mem_partition_units: Vec<mem_partition_unit::MemoryPartitionUnit>,
    mem_sub_partitions: Vec<Arc<Mutex<SubPartition>>>,
    pub running_kernels: Vec<Option<Arc<kernel::Kernel>>>,
    executed_kernels: Mutex<HashMap<u64, String>>,
    clusters: Vec<SIMTCoreCluster<I>>,
    #[allow(dead_code)]
    warp_instruction_unique_uid: Arc<atomic::AtomicU64>,
    interconn: Arc<I>,

    parallel_simulation: bool,
    last_cluster_issue: usize,
    last_issued_kernel: usize,
    allocations: allocation::Ref,

    // for main run loop
    cycle: Cycle,
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

#[derive(Clone, Debug, Default)]
#[repr(transparent)]
pub struct Cycle(Arc<std::sync::atomic::AtomicU64>);

impl Cycle {
    #[must_use]
    pub fn new(cycle: u64) -> Self {
        Self(Arc::new(std::sync::atomic::AtomicU64::new(cycle)))
    }

    pub fn set(&self, cycle: u64) {
        use std::sync::atomic::Ordering;
        self.0.store(cycle, Ordering::SeqCst);
    }

    #[must_use]
    pub fn get(&self) -> u64 {
        use std::sync::atomic::Ordering;
        self.0.load(Ordering::SeqCst)
    }
}

pub trait FromConfig {
    fn from_config(config: &config::GPUConfig) -> Self;
}

impl FromConfig for stats::Stats {
    fn from_config(config: &config::GPUConfig) -> Self {
        let num_total_cores = config.total_cores();
        let num_mem_units = config.num_memory_controllers;
        let num_sub_partitions = num_mem_units * config.num_sub_partition_per_memory_channel;
        let num_dram_banks = config.dram_timing_options.num_banks;

        Self::new(
            num_total_cores,
            num_mem_units,
            num_sub_partitions,
            num_dram_banks,
        )
    }
}

impl<I> MockSimulator<I>
where
    I: ic::Interconnect<core::Packet> + 'static,
{
    pub fn new(interconn: Arc<I>, config: Arc<config::GPUConfig>) -> Self {
        let stats = Arc::new(Mutex::new(Stats::from_config(&config)));

        let num_mem_units = config.num_memory_controllers;
        let num_sub_partitions = config.num_sub_partition_per_memory_channel;

        let cycle = Cycle::new(0);
        let mem_partition_units: Vec<_> = (0..num_mem_units)
            .map(|i| {
                mem_partition_unit::MemoryPartitionUnit::new(
                    i,
                    cycle.clone(),
                    Arc::clone(&config),
                    Arc::clone(&stats),
                )
            })
            .collect();

        let mut mem_sub_partitions = Vec::new();
        for partition in &mem_partition_units {
            for sub in 0..num_sub_partitions {
                mem_sub_partitions.push(partition.sub_partitions[sub].clone());
            }
        }

        let max_concurrent_kernels = config.max_concurrent_kernels;
        let running_kernels = (0..max_concurrent_kernels).map(|_| None).collect();

        let allocations = Arc::new(RwLock::new(Allocations::default()));

        let warp_instruction_unique_uid = Arc::new(atomic::AtomicU64::new(0));
        let clusters: Vec<_> = (0..config.num_simt_clusters)
            .map(|i| {
                SIMTCoreCluster::new(
                    i,
                    cycle.clone(),
                    Arc::clone(&warp_instruction_unique_uid),
                    Arc::clone(&allocations),
                    Arc::clone(&interconn),
                    Arc::clone(&stats),
                    Arc::clone(&config),
                )
            })
            .collect();

        let executed_kernels = Mutex::new(HashMap::new());

        assert!(config.max_threads_per_core.rem_euclid(config.warp_size) == 0);
        let _max_warps_per_shader = config.max_threads_per_core / config.warp_size;

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
        let last_cluster_issue = config.num_simt_clusters - 1;
        Self {
            config,
            stats,
            mem_partition_units,
            mem_sub_partitions,
            interconn,
            parallel_simulation: false,
            running_kernels,
            executed_kernels,
            clusters,
            warp_instruction_unique_uid,
            last_cluster_issue,
            last_issued_kernel: 0,
            allocations,
            cycle,
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

    /// Select the next kernel to run
    ///
    /// Todo: used hack to allow selecting the kernel from the shader core,
    /// but we could maybe refactor
    pub fn select_kernel(&self) -> Option<&Arc<Kernel>> {
        let mut executed_kernels = self.executed_kernels.lock().unwrap();
        if let Some(k) = &self.running_kernels[self.last_issued_kernel] {
            if !k.no_more_blocks_to_run()
            // &&!kernel.kernel_TB_latency)
            {
                let launch_id = k.id();
                executed_kernels
                    .entry(launch_id)
                    .or_insert(k.name().to_string());
                return Some(k);
            }
        }
        let num_kernels = self.running_kernels.len();
        let max_concurrent = self.config.max_concurrent_kernels;
        for n in 0..num_kernels {
            let idx = (n + self.last_issued_kernel + 1) % max_concurrent;
            if let Some(k) = &self.running_kernels[idx] {
                // &&!kernel.kernel_TB_latency)
                if !k.no_more_blocks_to_run() {
                    let launch_id = k.id();
                    assert!(!executed_kernels.contains_key(&launch_id));
                    executed_kernels.insert(launch_id, k.name().to_string());
                    return Some(k);
                }
            }
        }
        None
    }

    pub fn more_blocks_to_run(&self) -> bool {
        // if (hit_max_cta_count())
        // return false;

        self.running_kernels.iter().any(|kernel| match kernel {
            Some(kernel) => !kernel.no_more_blocks_to_run(),
            None => false,
        })
    }

    pub fn active(&self) -> bool {
        for cluster in &self.clusters {
            if cluster.not_completed() > 0 {
                return true;
            }
        }
        for unit in &self.mem_partition_units {
            if unit.busy() {
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
        self.running_kernels.iter().any(|kernel| match kernel {
            Some(kernel) => kernel.done(),
            None => true,
        })
    }

    pub fn launch(&mut self, kernel: Arc<Kernel>) -> eyre::Result<()> {
        kernel.set_launched();
        let threads_per_block = kernel.threads_per_block();
        let max_threads_per_block = self.config.max_threads_per_core;
        if threads_per_block > max_threads_per_block {
            log::error!("kernel block size is too large");
            log::error!(
                "CTA size (x*y*z) = {threads_per_block}, max supported = {max_threads_per_block}"
            );
            return Err(eyre::eyre!("kernel block size is too large"));
        }
        for running in &mut self.running_kernels {
            if running.is_none() || running.as_ref().map_or(false, |k| k.done()) {
                *running = Some(kernel);
                break;
            }
        }
        Ok(())
    }

    fn issue_block_to_core(&mut self) {
        log::debug!("===> issue block to core");
        let last_issued = self.last_cluster_issue;
        let num_clusters = self.config.num_simt_clusters;
        for cluster_idx in 0..num_clusters {
            debug_assert_eq!(cluster_idx, self.clusters[cluster_idx].cluster_id);
            let idx = (cluster_idx + last_issued + 1) % num_clusters;
            let num_blocks_issued = self.clusters[idx].issue_block_to_core(self);
            log::trace!("cluster[{}] issued {} blocks", idx, num_blocks_issued);

            if num_blocks_issued > 0 {
                self.last_cluster_issue = idx;
                // self.total_blocks_launched += num_blocks_issued;
            }
        }
    }

    pub fn set_cycle(&self, cycle: u64) {
        let mut stats = self.stats.lock().unwrap();
        stats.sim.cycles = cycle;
        self.cycle.set(cycle);
    }

    pub fn cycle(&mut self) {
        let start_total = Instant::now();
        // int clock_mask = next_clock_domain();

        // fn is_send<T: Send>(_: T) {}
        // fn is_sync<T: Sync>(_: T) {}
        // fn is_par_iter<T: rayon::iter::ParallelIterator>(_: T) {}

        // shader core loading (pop from ICNT into core)
        let start = Instant::now();
        if false && self.parallel_simulation {
            self.clusters
                .par_iter_mut()
                .for_each(cluster::SIMTCoreCluster::interconn_cycle);
        } else {
            for cluster in &mut self.clusters {
                cluster.interconn_cycle();
            }
        }
        TIMINGS
            .lock()
            .unwrap()
            .entry("icnt_cycle")
            .or_default()
            .add(start.elapsed());

        log::debug!(
            "POP from {} memory sub partitions",
            self.mem_sub_partitions.len()
        );

        // pop from memory controller to interconnect
        // if false && self.parallel_simulation {
        // self.mem_sub_partitions
        //     .par_iter_mut()
        //     .enumerate()
        //     .for_each(|(i, mem_sub)| {
        //         let mut mem_sub = mem_sub.try_lock().unwrap();
        //         if let Some(fetch) = mem_sub.top() {
        //             let response_packet_size = if fetch.is_write() {
        //                 fetch.control_size
        //             } else {
        //                 fetch.size()
        //             };
        //             let device = self.config.mem_id_to_device_id(i);
        //             if self.interconn.has_buffer(device, response_packet_size) {
        //                 let mut fetch = mem_sub.pop().unwrap();
        //                 let cluster_id = fetch.cluster_id;
        //                 fetch.set_status(mem_fetch::Status::IN_ICNT_TO_SHADER, 0);
        //                 let packet = Packet::Fetch(fetch);
        //                 // fetch.set_return_timestamp(gpu_sim_cycle + gpu_tot_sim_cycle);
        //                 // , gpu_sim_cycle + gpu_tot_sim_cycle);
        //                 // drop(fetch);
        //                 self.interconn
        //                     .push(device, cluster_id, packet, response_packet_size);
        //                 // self.partition_replies_in_parallel += 1;
        //             } else {
        //                 // self.gpu_stall_icnt2sh += 1;
        //             }
        //         }
        //     });
        // } else {
        for (i, mem_sub) in self.mem_sub_partitions.iter().enumerate() {
            // let mut mem_sub = mem_sub.try_borrow_mut().unwrap();
            let mut mem_sub = mem_sub.try_lock().unwrap();
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
                let l2_to_dram_queue = mem_sub.l2_to_dram_queue.lock().unwrap();
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
                    .dram_latency_queue
                    .iter()
                    .map(std::string::ToString::to_string)
                    .collect();
                log::debug!(
                    "\t dram latency queue ({:3}) = {:?}",
                    dram_latency_queue.len(),
                    style(&dram_latency_queue).red()
                );
                // log::debug!("");
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
                    let packet = Packet::Fetch(fetch);
                    // fetch.set_return_timestamp(gpu_sim_cycle + gpu_tot_sim_cycle);
                    // , gpu_sim_cycle + gpu_tot_sim_cycle);
                    // drop(fetch);
                    self.interconn
                        .push(device, cluster_id, packet, response_packet_size);
                    self.partition_replies_in_parallel += 1;
                } else {
                    // self.gpu_stall_icnt2sh += 1;
                }
            }
        }
        // }

        // DRAM
        let start = Instant::now();
        if false && self.parallel_simulation {
            self.mem_partition_units
                .par_iter_mut()
                .for_each(mem_partition_unit::MemoryPartitionUnit::simple_dram_cycle);
            // this pushes into sub.dram_to_l2_queue and messes up the order
        } else {
            log::debug!("cycle for {} drams", self.mem_partition_units.len());
            for (_i, unit) in self.mem_partition_units.iter_mut().enumerate() {
                unit.simple_dram_cycle();

                // if self.config.simple_dram_model {
                //     unit.simple_dram_cycle();
                // } else {
                //     // Issue the dram command (scheduler + delay model)
                //     // unit.simple_dram_cycle();
                //     unimplemented!()
                // }
            }
        }
        TIMINGS
            .lock()
            .unwrap()
            .entry("dram_cycle")
            .or_default()
            .add(start.elapsed());

        let current_cycle = self.cycle.get();

        // L2 operations
        log::debug!(
            "moving mem requests from interconn to {} mem partitions",
            self.mem_sub_partitions.len()
        );

        let start = Instant::now();
        if false && self.parallel_simulation {
            // todo!("parallel");
            self.mem_sub_partitions
                .par_iter()
                .enumerate()
                .for_each(|(i, mem_sub)| {
                    let mut mem_sub = mem_sub.try_lock().unwrap();
                    let device = self.config.mem_id_to_device_id(i);

                    // same as full with parameter overload
                    if mem_sub
                        .interconn_to_l2_can_fit(mem_sub_partition::SECTOR_CHUNCK_SIZE as usize)
                    {
                        if let Some(Packet::Fetch(fetch)) = self.interconn.pop(device) {
                            log::debug!(
                                "got new fetch {} for mem sub partition {} ({})",
                                fetch,
                                i,
                                device
                            );

                            mem_sub.push(fetch, current_cycle);
                            // self.parallel_mem_partition_reqs += 1;
                        }
                    } else {
                        log::debug!("SKIP sub partition {} ({}): DRAM full stall", i, device);
                        self.stats.lock().unwrap().stall_dram_full += 1;
                    }
                    // we borrow all of sub here, which is a problem for the cyclic reference in l2
                    // interface
                    mem_sub.cache_cycle(current_cycle);
                });

            // dbg!(self.mem_sub_partitions.len());
            // self.mem_sub_partitions
            //     .par_iter_mut()
            //     .for_each(|mem_sub| mem_sub.try_lock().unwrap().cache_cycle(current_cycle));
        } else {
            // let mut parallel_mem_partition_reqs_per_cycle = 0;
            // let mut stall_dram_full = 0;
            for (i, mem_sub) in self.mem_sub_partitions.iter_mut().enumerate() {
                // let mut mem_sub = mem_sub.try_borrow_mut().unwrap();
                let mut mem_sub = mem_sub.try_lock().unwrap();
                // move memory request from interconnect into memory partition
                // (if not backed up)
                //
                // Note:This needs to be called in DRAM clock domain if there
                // is no L2 cache in the system In the worst case, we may need
                // to push SECTOR_CHUNCK_SIZE requests, so ensure you have enough
                // buffer for them
                let device = self.config.mem_id_to_device_id(i);

                // same as full with parameter overload
                if mem_sub.interconn_to_l2_can_fit(mem_sub_partition::SECTOR_CHUNCK_SIZE as usize) {
                    if let Some(Packet::Fetch(fetch)) = self.interconn.pop(device) {
                        log::debug!(
                            "got new fetch {} for mem sub partition {} ({})",
                            fetch,
                            i,
                            device
                        );

                        mem_sub.push(fetch, current_cycle);
                        // self.parallel_mem_partition_reqs += 1;
                    }
                } else {
                    log::debug!("SKIP sub partition {} ({}): DRAM full stall", i, device);
                    self.stats.lock().unwrap().stall_dram_full += 1;
                }
                // we borrow all of sub here, which is a problem for the cyclic reference in l2
                // interface
                mem_sub.cache_cycle(current_cycle);
            }
        }
        TIMINGS
            .lock()
            .unwrap()
            .entry("l2_cycle")
            .or_default()
            .add(start.elapsed());

        //   partiton_reqs_in_parallel += partiton_reqs_in_parallel_per_cycle;
        // if (partiton_reqs_in_parallel_per_cycle > 0) {
        //   partiton_reqs_in_parallel_util += partiton_reqs_in_parallel_per_cycle;
        //   gpu_sim_cycle_parition_util++;
        // }

        // self.interconn_transfer();

        // dbg!(self.parallel_simulation);
        let start = Instant::now();

        let kernels_completed = self
            .running_kernels
            .iter()
            .filter_map(std::option::Option::as_ref)
            .all(|k| k.no_more_blocks_to_run());
        // let active_clusters: Vec<_> = self
        //     .clusters
        //     .iter_mut()
        //     .filter(|cluster| {
        //         let cores_completed = cluster.not_completed() == 0;
        //         !cores_completed || !kernels_completed
        //     })
        //     .collect();

        let mut executed_cluster_ids = std::collections::HashSet::new();

        if self.parallel_simulation {
            // let ic_before = self.interconn
            // let before = self.gather_state();
            // panic!("doing parallel for real");
            // dbg!(rayon::current_num_threads());
            let cores: Vec<_> = self
                .clusters
                .iter_mut()
                .filter(|cluster| {
                    let cores_completed = cluster.not_completed() == 0;
                    !cores_completed || !kernels_completed
                })
                .flat_map(|cluster| cluster.cores.iter())
                .collect();
            cores.par_iter().for_each(|core| {
                core.lock().unwrap().cycle();
            });
            // dbg!(cores.len());
            // let after = self.gather_state();
            // if before != after {
            // similar_asserts::assert_eq!(before: before, after: after);
            // }

            // for cluster in self.clusters.iter_mut() {
            //     // let ordering = cluster.core_sim_order
            //     for core_id in &cluster.core_sim_order {
            //         let core = cluster.cores[*core_id].lock().unwrap();
            //         let mut load_store_unit = core.inner.load_store_unit.lock().unwrap();
            //         for (dest, fetch, size) in load_store_unit.interconn_port.drain(..) {
            //             self.interconn
            //                 .push(cluster.cluster_id, dest, Packet::Fetch(fetch), size);
            //         }
            //     }
            //
            //     if let config::SchedulingOrder::RoundRobin = self.config.simt_core_sim_order {
            //         cluster.core_sim_order.rotate_left(1);
            //     }
            // }

            // self.interconn.commit();

            if false {
                for sub in &self.mem_sub_partitions {
                    let sub = sub.lock().unwrap();
                    let mem_dest = self.config.mem_id_to_device_id(sub.id);
                    // sub id: 0..15 (have 8 mem controllers, 2 sub partitions per channel)
                    // dbg!(sub.id);
                    // dbg!(mem_dest);
                    let mut dest_queue = self.interconn.dest_queue(mem_dest).lock().unwrap();
                    // todo sort the dest queue here
                    dest_queue.make_contiguous().sort_by_key(|packet| {
                        let Packet::Fetch(fetch) = packet;
                        let ordering = &self.clusters[fetch.cluster_id].core_sim_order;
                        // dbg!(&ordering);
                        let core_id = fetch.core_id
                            - (fetch.cluster_id * self.config.num_cores_per_simt_cluster);
                        // dbg!(fetch.core_id, core_id);
                        let ordering_idx = ordering.iter().position(|id| *id == core_id).unwrap();
                        // dbg!(fetch.cluster_id);
                        // dbg!(fetch.core_id);
                        let pushed_cycle = fetch.pushed_cycle.unwrap();
                        // dbg!((pushed_cycle, core_idx));
                        (pushed_cycle, fetch.cluster_id, ordering_idx)
                    });
                    let _q = dest_queue
                        .iter()
                        .map(|p| {
                            let Packet::Fetch(fetch) = p;
                            (fetch.pushed_cycle.unwrap(), fetch.cluster_id, fetch.core_id)
                        })
                        .collect::<Vec<_>>();
                    // if !q.is_empty() {
                    //     dbg!(sub.id, q);
                    // }
                }
            }

            for cluster in &mut self.clusters {
                // let cores_completed = cluster.not_completed() == 0;
                // let kernels_completed = self
                //     .running_kernels
                //     .iter()
                //     .filter_map(std::option::Option::as_ref)
                //     .all(|k| k.no_more_blocks_to_run());
                // if cores_completed && kernels_completed {
                //     continue;
                // }
                // dbg!(&executed_cluster_ids);
                // if !executed_cluster_ids.contains(&cluster.cluster_id) {
                //     continue;
                // }

                for core_id in &cluster.core_sim_order {
                    let core = cluster.cores[*core_id].lock().unwrap();
                    let mut port = core.interconn_port.lock().unwrap();
                    for (dest, fetch, size) in port.drain(..) {
                        self.interconn
                            .push(core.cluster_id, dest, Packet::Fetch(fetch), size);
                    }
                }

                if let config::SchedulingOrder::RoundRobin = self.config.simt_core_sim_order {
                    cluster.core_sim_order.rotate_left(1);
                }
            }
        } else {
            // let mut active_sms = 0;
            for cluster in &mut self.clusters {
                log::debug!("cluster {} cycle {}", cluster.cluster_id, self.cycle.get());
                let cores_completed = cluster.not_completed() == 0;
                let kernels_completed = self
                    .running_kernels
                    .iter()
                    .filter_map(std::option::Option::as_ref)
                    .all(|k| k.no_more_blocks_to_run());
                if cores_completed && kernels_completed {
                    continue;
                }
                executed_cluster_ids.insert(cluster.cluster_id);

                // cluster.cycle();
                for core_id in &cluster.core_sim_order {
                    let mut core = cluster.cores[*core_id].lock().unwrap();
                    core.cycle();

                    let mut port = core.interconn_port.lock().unwrap();
                    for (dest, fetch, size) in port.drain(..) {
                        self.interconn
                            .push(core.cluster_id, dest, Packet::Fetch(fetch), size);
                    }
                }

                if let config::SchedulingOrder::RoundRobin = self.config.simt_core_sim_order {
                    cluster.core_sim_order.rotate_left(1);
                }

                // active_sms += cluster.num_active_sms();
            }
        }

        TIMINGS
            .lock()
            .unwrap()
            .entry("core_cycle")
            .or_default()
            .add(start.elapsed());

        // if false && self.parallel_simulation {
        // log::debug!("===> issue block to core");
        // let last_issued = self.last_cluster_issue;
        // let num_clusters = self.config.num_simt_clusters;
        // for cluster_idx in 0..num_clusters {
        //     debug_assert_eq!(cluster_idx, self.clusters[cluster_idx].cluster_id);
        //     let idx = (cluster_idx + last_issued + 1) % num_clusters;
        //     let num_blocks_issued = self.clusters[idx].issue_block_to_core(self);
        //     log::trace!("cluster[{}] issued {} blocks", idx, num_blocks_issued);
        //
        //     if num_blocks_issued > 0 {
        //         self.last_cluster_issue = idx;
        //         // self.total_blocks_launched += num_blocks_issued;
        //     }
        // }
        // } else {
        self.issue_block_to_core();
        // self.decrement_kernel_latency();
        // }

        // Depending on configuration, invalidate the caches
        // once all of threads are completed.
        let mut all_threads_complete = true;
        if self.config.flush_l1_cache {
            for cluster in &mut self.clusters {
                if cluster.not_completed() == 0 {
                    cluster.cache_invalidate();
                } else {
                    all_threads_complete = false;
                }
            }
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
                    log::debug!("flushed L2 caches...");
                    if l2_config.inner.total_lines() > 0 {
                        for (i, mem_sub) in self.mem_sub_partitions.iter_mut().enumerate() {
                            // let mut mem_sub = mem_sub.try_borrow_mut().unwrap();
                            let mut mem_sub = mem_sub.try_lock().unwrap();
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

        TIMINGS
            .lock()
            .unwrap()
            .entry("total_cycle")
            .or_default()
            .add(start_total.elapsed());
    }

    pub fn gpu_mem_alloc(&mut self, addr: address, num_bytes: u64, name: Option<&str>) {
        log::info!(
            "memalloc: {:<20} {:>15} ({:>5} f32) at address {addr:>20}",
            name.unwrap_or("<unnamed>"),
            human_bytes::human_bytes(num_bytes as f64),
            num_bytes / 4,
        );
        // let alloc_range = addr..(addr + num_bytes);
        // self.allocations
        //     .try_borrow_mut()
        //     .unwrap()
        //     .insert(alloc_range.clone(), name);
    }

    fn copy_chunk_to_gpu(&self, write_addr: address, time: u64) {
        let num_sub_partitions = self.config.num_sub_partition_per_memory_channel;
        let tlx_addr = self.config.address_mapping().tlx(write_addr);
        let partition_id = tlx_addr.sub_partition / num_sub_partitions as u64;
        let sub_partition_id = tlx_addr.sub_partition % num_sub_partitions as u64;

        let partition = &self.mem_partition_units[partition_id as usize];

        let mut mask: mem_fetch::SectorMask = BitArray::ZERO;
        // Sector chunk size is 4, so we get the highest 4 bits of the address
        // to set the sector mask
        mask.set(((write_addr % 128) as u8 / 32) as usize, true);

        log::trace!(
            "memcopy to gpu: copy 32 byte chunk starting at {} to sub partition unit {} of partition unit {} ({}) (mask {})",
            write_addr,
            sub_partition_id,
            partition_id,
            tlx_addr.sub_partition,
            mask.to_bit_string()
        );

        partition.handle_memcpy_to_gpu(write_addr, tlx_addr.sub_partition as usize, mask, time);
    }

    pub fn memcopy_to_gpu(&mut self, addr: address, num_bytes: u64, name: Option<String>) {
        log::info!(
            "memcopy: {:<20} {:>15} ({:>5} f32) to address {addr:>20}",
            name.as_deref().unwrap_or("<unnamed>"),
            human_bytes::human_bytes(num_bytes as f64),
            num_bytes / 4,
        );
        let alloc_range = addr..(addr + num_bytes);
        self.allocations.write().unwrap().insert(alloc_range, name);

        if self.config.fill_l2_on_memcopy {
            let chunk_size: u64 = 32;
            let chunks = (num_bytes as f64 / chunk_size as f64).ceil() as usize;
            let time = self.cycle.get();
            for chunk in 0..chunks {
                let write_addr = addr + (chunk as u64 * chunk_size);
                self.copy_chunk_to_gpu(write_addr, time);
            }
        }
    }

    pub fn stats(&self) -> Stats {
        let mut stats: Stats = self.stats.lock().unwrap().clone();

        for cluster in &self.clusters {
            // for core in cluster.cores.lock().unwrap().iter() {
            for core in &cluster.cores {
                let core = core.lock().unwrap();
                let core_id = core.core_id;
                stats.l1i_stats[core_id] = core.instr_l1_cache.stats().lock().unwrap().clone();
                // .insert(core_id, core.instr_l1_cache.stats().lock().unwrap().clone());
                let ldst_unit = &core.load_store_unit.lock().unwrap();

                let data_l1 = ldst_unit.data_l1.as_ref().unwrap();
                stats.l1d_stats[core_id] = data_l1.stats().lock().unwrap().clone();
                // .insert(core_id, data_l1.stats().lock().unwrap().clone());
                stats.l1c_stats[core_id] = stats::Cache::default();
                // stats.l1c_stats.insert(core_id, stats::Cache::default());
                stats.l1t_stats[core_id] = stats::Cache::default();
                // stats.l1t_stats.insert(core_id, stats::Cache::default());
            }
        }

        for sub in &self.mem_sub_partitions {
            let sub = sub.try_lock().unwrap();
            let l2_cache = sub.l2_cache.as_ref().unwrap();
            stats.l2d_stats[sub.id] = l2_cache.stats().lock().unwrap().clone();
            // .insert(sub.id, l2_cache.stats().lock().unwrap().clone());
        }
        stats
    }

    /// Process commands
    ///
    /// Take as many commands as possible until we have collected as many kernels to fill
    /// the `window_size` or processed every command.
    pub fn process_commands(&mut self) {
        while self.kernels.len() < self.kernel_window_size && self.command_idx < self.commands.len()
        {
            let cmd = &self.commands[self.command_idx];
            match cmd {
                Command::MemcpyHtoD(trace_model::MemcpyHtoD {
                    allocation_name,
                    dest_device_addr,
                    num_bytes,
                }) => self.memcopy_to_gpu(*dest_device_addr, *num_bytes, allocation_name.clone()),
                Command::MemAlloc(trace_model::MemAlloc {
                    allocation_name,
                    device_ptr,
                    num_bytes,
                }) => {
                    self.gpu_mem_alloc(*device_ptr, *num_bytes, allocation_name.clone().as_deref());
                }
                Command::KernelLaunch(launch) => {
                    let kernel =
                        Kernel::from_trace(launch.clone(), self.traces_dir.as_ref().unwrap());
                    self.kernels.push_back(Arc::new(kernel));
                }
            }
            self.command_idx += 1;
        }
        let allocations = self.allocations.read().unwrap();
        log::info!(
            "allocations: {:#?}",
            allocations
                .iter()
                .map(|(_, alloc)| alloc.to_string())
                .collect::<Vec<_>>()
        );
    }

    /// Lauch more kernels if possible.
    ///
    /// Launch all kernels within window that are on a stream that isn't already running
    pub fn launch_kernels(&mut self) {
        log::trace!("launching kernels");
        let mut launch_queue: Vec<Arc<Kernel>> = Vec::new();
        for kernel in &self.kernels {
            let stream_busy = self
                .busy_streams
                .iter()
                .any(|stream_id| *stream_id == kernel.config.stream_id);
            if !stream_busy && self.can_start_kernel() && !kernel.was_launched() {
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
            self.launch(kernel).unwrap();
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

    pub fn run_to_completion(&mut self) -> eyre::Result<()> {
        let mut cycle: u64 = 0;
        let mut last_state_change: Option<(deadlock::State, u64)> = None;

        while (self.commands_left() || self.kernels_left()) && !self.reached_limit(cycle) {
            self.process_commands();
            self.launch_kernels();

            let mut finished_kernel = None;
            loop {
                log::info!("======== cycle {cycle} ========");
                log::info!("");

                if self.reached_limit(cycle) || !self.active() {
                    break;
                }

                self.cycle();
                cycle += 1;
                self.set_cycle(cycle);

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

                        // let allocations = self.allocations.try_borrow().unwrap();
                        let allocations = self.allocations.read().unwrap();
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
                self.cleanup_finished_kernel(&kernel);
            }

            log::trace!(
                "commands left={} kernels left={}",
                self.commands_left(),
                self.kernels_left()
            );
        }
        log::info!("exit after {cycle} cycles");
        Ok(())
    }

    fn finished_kernel(&mut self) -> Option<Arc<Kernel>> {
        // check running kernels
        let _active = self.active();
        let finished_kernel: Option<&mut Option<Arc<Kernel>>> =
            self.running_kernels.iter_mut().find(|k| {
                if let Some(k) = k {
                    // TODO: could also check here if !self.active()
                    k.no_more_blocks_to_run() && !k.running() && k.was_launched()
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

    fn cleanup_finished_kernel(&mut self, kernel: &Kernel) {
        log::debug!(
            "cleanup finished kernel with id={}: {}",
            kernel.id(),
            kernel
        );
        self.kernels.retain(|k| k.config.id != kernel.config.id);
        self.busy_streams
            .retain(|stream| *stream != kernel.config.stream_id);

        // resets some statistics between kernel launches
        //   if (!silent && m_gpgpu_sim->gpu_sim_cycle > 0) {
        //   m_gpgpu_sim->update_stats();
        //   m_gpgpu_context->print_simulation_time();
        // }
    }
}

pub fn save_stats_to_file(stats: &Stats, path: &Path) -> eyre::Result<()> {
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

pub fn accelmain(
    traces_dir: impl AsRef<Path>,
    config: impl Into<Arc<config::GPUConfig>>,
) -> eyre::Result<Stats> {
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

    let interconn = Arc::new(ic::ToyInterconnect::new(
        config.num_simt_clusters,
        config.num_memory_controllers * config.num_sub_partition_per_memory_channel,
    ));
    let mut sim = MockSimulator::new(interconn, Arc::clone(&config));
    sim.add_commands(commands_path, traces_dir)?;

    sim.log_after_cycle = config.log_after_cycle;
    sim.parallel_simulation = config.parallel;

    sim.run_to_completion()?;

    let stats = sim.stats();

    Ok(stats)
}

#[cfg(test)]
mod tests {}
