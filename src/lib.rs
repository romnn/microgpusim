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
pub mod arbitration;
pub mod barrier;
pub mod cache;
pub mod cluster;
pub mod config;
pub mod core;
pub mod deadlock;
pub mod dram;
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

#[cfg(test)]
pub mod testing;

use self::cluster::SIMTCoreCluster;
use self::core::{
    warp_inst_complete, Packet, PipelineStage, SIMTCore, WarpMask, MAX_THREAD_PER_SM,
    PROGRAM_MEM_START,
};
use addrdec::DecodedAddress;
use fifo::Queue;
use interconn as ic;
use kernel::Kernel;
use ldst_unit::LoadStoreUnit;
use mem_fetch::{AccessKind, BitString};
use sp_unit::SPUnit;
use stats::Stats;

use bitvec::array::BitArray;
use color_eyre::eyre::{self};
use console::style;
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{atomic, Arc, Mutex};
use std::time::Instant;
use trace_model::Command;

pub type address = u64;

pub fn parse_commands(path: impl AsRef<Path>) -> eyre::Result<Vec<Command>> {
    let reader = utils::fs::open_readable(path.as_ref())?;
    let commands = serde_json::from_reader(reader)?;
    Ok(commands)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Allocation {
    id: usize,
    name: Option<String>,
    start_addr: address,
    end_addr: Option<address>,
}

impl std::fmt::Display for Allocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let num_bytes = self.end_addr.map(|end| end - self.start_addr);
        let num_f32 = num_bytes.map(|num_bytes| num_bytes / 4);
        f.debug_struct("Allocation")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("start_addr", &self.start_addr)
            .field("end_addr", &self.end_addr)
            .field(
                "size",
                &num_bytes.map(|num_bytes| human_bytes::human_bytes(num_bytes as f64)),
            )
            .field("num_f32", &num_f32)
            .finish()
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct Allocations(rangemap::RangeMap<address, Allocation>);

impl std::ops::Deref for Allocations {
    type Target = rangemap::RangeMap<address, Allocation>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Allocations {
    pub fn insert(&mut self, range: std::ops::Range<address>, name: Option<String>) {
        // check for intersections
        assert!(
            !self.0.overlaps(&range),
            "overlapping memory allocation {:?}",
            &range
        );
        let id = self.0.len() + 1; // zero is reserved for instructions
        let start_addr = range.start;
        let end_addr = Some(range.end);
        self.0.insert(
            range,
            Allocation {
                id,
                name,
                start_addr,
                end_addr,
            },
        );
    }
}

#[derive()]
pub struct MockSimulator<I> {
    stats: Arc<Mutex<Stats>>,
    config: Arc<config::GPUConfig>,
    mem_partition_units: Vec<mem_partition_unit::MemoryPartitionUnit>,
    mem_sub_partitions: Vec<
        Rc<RefCell<mem_sub_partition::MemorySubPartition<fifo::FifoQueue<mem_fetch::MemFetch>>>>,
    >,
    running_kernels: Vec<Option<Arc<kernel::Kernel>>>,
    executed_kernels: Mutex<HashMap<u64, String>>,
    clusters: Vec<SIMTCoreCluster<I>>,
    #[allow(dead_code)]
    warp_instruction_unique_uid: Arc<atomic::AtomicU64>,
    interconn: Arc<I>,

    last_cluster_issue: usize,
    last_issued_kernel: usize,
    allocations: Rc<RefCell<Allocations>>,

    // for main run loop
    cycle: Cycle,
    traces_dir: PathBuf,
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

#[derive(Debug, Default)]
pub struct AtomicCycle(std::sync::atomic::AtomicU64);

impl AtomicCycle {
    #[must_use]
    pub fn new(cycle: u64) -> Self {
        Self(std::sync::atomic::AtomicU64::new(cycle))
    }

    pub fn set(&self, cycle: u64) {
        use std::sync::atomic::Ordering;
        self.0.store(cycle, Ordering::SeqCst);
    }

    pub fn get(&self) -> u64 {
        use std::sync::atomic::Ordering;
        self.0.load(Ordering::SeqCst)
    }
}

pub type Cycle = Rc<AtomicCycle>;

pub trait FromConfig {
    fn from_config(config: &config::GPUConfig) -> Self;
}

impl FromConfig for stats::Stats {
    fn from_config(config: &config::GPUConfig) -> Self {
        let num_total_cores = config.total_cores();
        let num_mem_units = config.num_memory_controllers;
        let num_dram_banks = config.dram_timing_options.num_banks;

        Self::new(num_total_cores, num_mem_units, num_dram_banks)
    }
}

impl<I> MockSimulator<I>
where
    I: ic::Interconnect<core::Packet> + 'static,
{
    pub fn new(
        interconn: Arc<I>,
        config: Arc<config::GPUConfig>,
        traces_dir: impl AsRef<Path>,
        commands_path: impl AsRef<Path>,
    ) -> Self {
        let _start = Instant::now();
        let traces_dir = traces_dir.as_ref();
        let stats = Arc::new(Mutex::new(Stats::from_config(&config)));

        let num_mem_units = config.num_memory_controllers;
        let num_sub_partitions = config.num_sub_partition_per_memory_channel;

        let cycle = Rc::new(AtomicCycle::new(0));
        let mem_partition_units: Vec<_> = (0..num_mem_units)
            .map(|i| {
                mem_partition_unit::MemoryPartitionUnit::new(
                    i,
                    Rc::clone(&cycle),
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
        // let finished_kernels = Rc::new(RefCell::new(VecDeque::new()));
        let running_kernels = (0..max_concurrent_kernels).map(|_| None).collect();

        let allocations = Rc::new(RefCell::new(Allocations::default()));

        let warp_instruction_unique_uid = Arc::new(atomic::AtomicU64::new(0));
        let clusters: Vec<_> = (0..config.num_simt_clusters)
            .map(|i| {
                SIMTCoreCluster::new(
                    i,
                    Rc::clone(&cycle),
                    Arc::clone(&warp_instruction_unique_uid),
                    Rc::clone(&allocations),
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

        // let command_traces_path = traces_dir.join("commands.json");
        let commands: Vec<Command> = parse_commands(commands_path.as_ref()).unwrap();

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
            // finished_kernels,
            running_kernels,
            executed_kernels,
            clusters,
            warp_instruction_unique_uid,
            last_cluster_issue,
            last_issued_kernel: 0,
            allocations,
            cycle,
            traces_dir: traces_dir.to_path_buf(),
            commands,
            command_idx: 0,
            kernels,
            kernel_window_size: window_size,
            busy_streams,
            cycle_limit,
            log_after_cycle: None,
            partition_replies_in_parallel: 0,
        }
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
        // int clock_mask = next_clock_domain();

        // shader core loading (pop from ICNT into core)
        for cluster in &mut self.clusters {
            cluster.interconn_cycle();
        }

        log::debug!(
            "POP from {} memory sub partitions",
            self.mem_sub_partitions.len()
        );

        // pop from memory controller to interconnect
        for (i, mem_sub) in self.mem_sub_partitions.iter().enumerate() {
            let mut mem_sub = mem_sub.try_borrow_mut().unwrap();
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
                    fetch.control_size
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

        // dram
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

        // L2 operations
        log::debug!(
            "moving mem requests from interconn to {} mem partitions",
            self.mem_sub_partitions.len()
        );
        // let mut parallel_mem_partition_reqs_per_cycle = 0;
        // let mut stall_dram_full = 0;
        for (i, mem_sub) in self.mem_sub_partitions.iter_mut().enumerate() {
            let mut mem_sub = mem_sub.try_borrow_mut().unwrap();
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

                    mem_sub.push(fetch);
                    // self.parallel_mem_partition_reqs += 1;
                }
            } else {
                log::debug!("SKIP sub partition {} ({}): DRAM full stall", i, device);
                self.stats.lock().unwrap().stall_dram_full += 1;
            }
            // we borrow all of sub here, which is a problem for the cyclic reference in l2
            // interface
            mem_sub.cache_cycle(self.cycle.get());
        }

        //   partiton_reqs_in_parallel += partiton_reqs_in_parallel_per_cycle;
        // if (partiton_reqs_in_parallel_per_cycle > 0) {
        //   partiton_reqs_in_parallel_util += partiton_reqs_in_parallel_per_cycle;
        //   gpu_sim_cycle_parition_util++;
        // }

        // self.interconn_transfer();

        // let mut active_sms = 0;
        for cluster in &mut self.clusters {
            let cores_completed = cluster.not_completed() == 0;
            let kernels_completed = self
                .running_kernels
                .iter()
                .filter_map(std::option::Option::as_ref)
                .all(|k| k.no_more_blocks_to_run());
            if !cores_completed || !kernels_completed {
                cluster.cycle();
                // active_sms += cluster.num_active_sms();
            }
        }

        self.issue_block_to_core();
        // self.decrement_kernel_latency();

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
                            let mut mem_sub = mem_sub.try_borrow_mut().unwrap();
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

    pub fn memcopy_to_gpu(&mut self, addr: address, num_bytes: u64, name: Option<String>) {
        log::info!(
            "memcopy: {:<20} {:>15} ({:>5} f32) to address {addr:>20}",
            name.as_deref().unwrap_or("<unnamed>"),
            human_bytes::human_bytes(num_bytes as f64),
            num_bytes / 4,
        );
        let alloc_range = addr..(addr + num_bytes);
        self.allocations
            .try_borrow_mut()
            .unwrap()
            .insert(alloc_range, name);

        if self.config.fill_l2_on_memcopy {
            let num_sub_partitions = self.config.num_sub_partition_per_memory_channel;
            let mut transfered = 0;
            while transfered < num_bytes {
                let write_addr = addr + transfered;

                let tlx_addr = self.config.address_mapping().tlx(write_addr);
                let partition_id = tlx_addr.sub_partition / num_sub_partitions as u64;
                let sub_partition_id = tlx_addr.sub_partition % num_sub_partitions as u64;

                let partition = &self.mem_partition_units[partition_id as usize];

                let mut mask: mem_fetch::MemAccessSectorMask = BitArray::ZERO;
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

                let time = self.cycle.get();
                partition.handle_memcpy_to_gpu(
                    write_addr,
                    tlx_addr.sub_partition as usize,
                    mask,
                    time,
                );
                transfered += 32;
            }
        }
    }

    pub fn stats(&self) -> Stats {
        let mut stats: Stats = self.stats.lock().unwrap().clone();

        for cluster in &self.clusters {
            for core in cluster.cores.lock().unwrap().iter() {
                let core_id = core.inner.core_id;
                stats.l1i_stats.insert(
                    core_id,
                    core.inner.instr_l1_cache.stats().lock().unwrap().clone(),
                );
                let ldst_unit = &core.inner.load_store_unit.lock().unwrap();

                let data_l1 = ldst_unit.data_l1.as_ref().unwrap();
                stats
                    .l1d_stats
                    .insert(core_id, data_l1.stats().lock().unwrap().clone());
                stats.l1c_stats.insert(core_id, stats::Cache::default());
                stats.l1t_stats.insert(core_id, stats::Cache::default());
            }
        }

        for sub in &self.mem_sub_partitions {
            let sub: &mem_sub_partition::MemorySubPartition = &sub.as_ref().try_borrow().unwrap();
            let l2_cache = sub.l2_cache.as_ref().unwrap();
            stats
                .l2d_stats
                .insert(sub.id, l2_cache.stats().lock().unwrap().clone());
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
                    let kernel = Kernel::from_trace(&self.traces_dir, launch.clone());
                    self.kernels.push_back(Arc::new(kernel));
                }
            }
            self.command_idx += 1;
        }
        let allocations = self.allocations.try_borrow().unwrap();
        log::info!(
            "allocations: {:#?}",
            allocations
                .deref()
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

    pub fn run_to_completion(
        &mut self,
        _traces_dir: impl AsRef<Path>,
        deadlock_check: bool,
    ) -> eyre::Result<()> {
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

                        let allocations = self.allocations.try_borrow().unwrap();
                        for (_, alloc) in allocations.deref().iter() {
                            log::info!("allocation: {}", alloc);
                        }
                    }
                    _ => {}
                }

                // collect state
                if deadlock_check {
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

    // pub fn set_kernel_done(&mut self, kernel: &mut Kernel) {
    //     self.finished_kernels
    //         .borrow_mut()
    //         .push_back(kernel.config.id);
    //     let running_kernel_idx = self
    //         .running_kernels
    //         .iter()
    //         .position(|k| k.as_ref().map(|k| k.config.id) == Some(kernel.config.id))
    //         .unwrap();
    //     // kernel.end_cycle = self.cycle.get();
    //     self.running_kernels.remove(running_kernel_idx);
    // }

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
    log_after_cycle: Option<u64>,
) -> eyre::Result<Stats> {
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
    let config = Arc::new(config::GPUConfig {
        num_simt_clusters: 20,                   // 20
        num_cores_per_simt_cluster: 4,           // 1
        num_schedulers_per_core: 2,              // 1
        num_memory_controllers: 8,               // 8
        num_sub_partition_per_memory_channel: 2, // 2
        fill_l2_on_memcopy: true,                // true
        ..config::GPUConfig::default()
    });

    let interconn = Arc::new(ic::ToyInterconnect::new(
        config.num_simt_clusters,
        config.num_memory_controllers * config.num_sub_partition_per_memory_channel,
    ));
    let mut sim = MockSimulator::new(interconn, Arc::clone(&config), &traces_dir, commands_path);

    sim.log_after_cycle = log_after_cycle;

    let deadlock_check = std::env::var("DEADLOCK_CHECK")
        .unwrap_or_default()
        .to_lowercase()
        == "yes";

    sim.run_to_completion(&traces_dir, deadlock_check)?;

    let stats = sim.stats();

    Ok(stats)
}

#[cfg(test)]
mod tests {
    use crate::{cache, config, fifo, interconn as ic, mem_fetch, testing, testing::diff};
    use color_eyre::eyre;
    use itertools::Itertools;
    use pretty_assertions_sorted as full_diff;
    use serde::Serialize;
    use stats::ConvertHashMap;
    use std::collections::HashSet;
    use std::io::Write;
    use std::ops::Deref;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::Instant;
    use trace_model::Command;

    #[derive(Debug, Clone, Copy)]
    enum TraceProvider {
        Native,
        Accelsim,
        Box,
    }

    #[inline]
    fn gather_simulation_state(
        box_sim: &mut super::MockSimulator<ic::ToyInterconnect<super::Packet>>,
        play_sim: &mut playground::Accelsim,
        _trace_provider: TraceProvider,
    ) -> (testing::state::Simulation, testing::state::Simulation) {
        // iterate over sub partitions
        let num_schedulers = box_sim.config.num_schedulers_per_core;
        let num_clusters = box_sim.config.num_simt_clusters;
        let cores_per_cluster = box_sim.config.num_cores_per_simt_cluster;
        assert_eq!(
            box_sim.config.total_cores(),
            num_clusters * cores_per_cluster
        );

        let num_partitions = box_sim.mem_partition_units.len();
        let num_sub_partitions = box_sim.mem_sub_partitions.len();
        let mut box_sim_state = testing::state::Simulation::new(
            num_clusters,
            cores_per_cluster,
            num_partitions,
            num_sub_partitions,
            num_schedulers,
        );

        box_sim_state.last_cluster_issue = box_sim.last_cluster_issue;

        for (cluster_id, cluster) in box_sim.clusters.iter().enumerate() {
            for (core_id, core) in cluster.cores.lock().unwrap().iter().enumerate() {
                let global_core_id =
                    cluster_id * box_sim.config.num_cores_per_simt_cluster + core_id;
                assert_eq!(core.inner.core_id, global_core_id);

                // this is the one we will use (unless the assertion is ever false)
                let core_id = core.inner.core_id;

                // core: functional units
                for (fu_id, fu) in core.functional_units.iter().enumerate() {
                    let _fu = fu.lock().unwrap();
                    let issue_port = core.issue_ports[fu_id];
                    let issue_reg: super::register_set::RegisterSet = core.inner.pipeline_reg
                        [issue_port as usize]
                        .borrow()
                        .clone();
                    assert_eq!(issue_port, issue_reg.stage);

                    box_sim_state.functional_unit_pipelines_per_core[core_id]
                        .push(issue_reg.into());
                }
                for (_fu_id, fu) in core.functional_units.iter().enumerate() {
                    let fu = fu.lock().unwrap();
                    box_sim_state.functional_unit_pipelines_per_core[core_id].push(
                        testing::state::RegisterSet {
                            name: fu.id().to_string(),
                            pipeline: fu
                                .pipeline()
                                .iter()
                                .map(|reg| reg.clone().map(Into::into))
                                .collect(),
                        },
                    );
                }
                // core: operand collector
                box_sim_state.operand_collector_per_core[core_id] =
                    Some(core.inner.operand_collector.borrow().deref().into());
                // core: schedulers
                box_sim_state.scheduler_per_core[core_id] =
                    core.schedulers.iter().map(Into::into).collect();
                // .extend(core.schedulers.iter().map(Into::into));
                // core: l2 cache
                let ldst_unit = core.inner.load_store_unit.lock().unwrap();

                // core: pending register writes
                box_sim_state.pending_register_writes_per_core[core_id] = ldst_unit
                    .pending_writes
                    .clone()
                    .into_iter()
                    .flat_map(|(warp_id, pending_registers)| {
                        pending_registers
                            .into_iter()
                            .map(
                                move |(reg_num, pending)| testing::state::PendingRegisterWrites {
                                    warp_id,
                                    reg_num,
                                    pending,
                                },
                            )
                    })
                    .collect();

                box_sim_state.pending_register_writes_per_core[core_id].sort();
                // let l1d_tag_array = ldst_unit.data_l1.unwrap().tag_array;
                // dbg!(&l1d_tag_array);
                // let l1d_tag_array = ldst_unit.data_l1.unwrap().tag_array;
            }
        }

        for (partition_id, partition) in box_sim.mem_partition_units.iter().enumerate() {
            box_sim_state.dram_latency_queue_per_partition[partition_id].extend(
                partition
                    .dram_latency_queue
                    .clone()
                    .into_iter()
                    .map(Into::into),
            );
            box_sim_state.dram_arbitration_per_partition[partition_id] =
                testing::state::Arbitration {
                    last_borrower: partition.arbitration_metadata.last_borrower,
                    shared_credit: partition.arbitration_metadata.shared_credit,
                    private_credit: partition.arbitration_metadata.private_credit.clone().into(),
                };
        }
        for (sub_id, sub) in box_sim.mem_sub_partitions.iter().enumerate() {
            let sub = sub.borrow();
            let l2_cache = sub.l2_cache.as_ref().unwrap();
            let l2_cache: &cache::DataL2<ic::L2Interface<fifo::FifoQueue<mem_fetch::MemFetch>>> =
                l2_cache.as_any().downcast_ref().unwrap();

            box_sim_state.l2_cache_per_sub[sub_id] =
                Some(l2_cache.inner.inner.tag_array.clone().into());

            for (dest_queue, src_queue) in [
                (
                    &mut box_sim_state.interconn_to_l2_queue_per_sub[sub_id],
                    &sub.interconn_to_l2_queue,
                ),
                (
                    &mut box_sim_state.l2_to_interconn_queue_per_sub[sub_id],
                    &sub.l2_to_interconn_queue,
                ),
                (
                    &mut box_sim_state.l2_to_dram_queue_per_sub[sub_id],
                    &sub.l2_to_dram_queue.lock().unwrap(),
                ),
                (
                    &mut box_sim_state.dram_to_l2_queue_per_sub[sub_id],
                    &sub.dram_to_l2_queue,
                ),
            ] {
                // dest_queue.extend(src_queue.clone().into_iter().map(Into::into));
                *dest_queue = src_queue.clone().into_iter().map(Into::into).collect();
            }
        }

        let mut play_sim_state = testing::state::Simulation::new(
            num_clusters,
            cores_per_cluster,
            num_partitions,
            num_sub_partitions,
            num_schedulers,
        );

        play_sim_state.last_cluster_issue = play_sim.last_cluster_issue() as usize;

        for (core_id, core) in play_sim.cores().enumerate() {
            for regs in core.functional_unit_issue_register_sets() {
                play_sim_state.functional_unit_pipelines_per_core[core_id].push(regs.into());
            }
            let valid_units: HashSet<_> = box_sim_state.functional_unit_pipelines_per_core[core_id]
                .iter()
                .map(|fu| fu.name.clone())
                .collect();

            for regs in core
                .functional_unit_simd_pipeline_register_sets()
                .into_iter()
                .filter(|fu| valid_units.contains(&fu.name()))
            {
                play_sim_state.functional_unit_pipelines_per_core[core_id].push(regs.into());
            }

            // core: pending register writes
            play_sim_state.pending_register_writes_per_core[core_id] = core
                .pending_register_writes()
                .into_iter()
                .map(Into::into)
                .collect();
            play_sim_state.pending_register_writes_per_core[core_id].sort();

            // core: operand collector
            let coll = core.operand_collector();
            play_sim_state.operand_collector_per_core[core_id] = Some(coll.into());
            // core: scheduler units
            let schedulers = core.schedulers();
            assert_eq!(
                schedulers.len(),
                box_sim_state.scheduler_per_core[core_id].len()
            );

            for (sched_idx, play_scheduler) in schedulers.into_iter().enumerate() {
                play_sim_state.scheduler_per_core[core_id][sched_idx] = play_scheduler.into();

                // let box_sched = &mut box_sim_state.scheduler_per_core[core_id][sched_idx];
                // let play_sched = &mut play_sim_state.scheduler_per_core[core_id][sched_idx];
                //
                // let num_box_warps = box_sched.prioritized_warp_ids.len();
                // let num_play_warps = play_sched.prioritized_warp_ids.len();
                // let limit = num_box_warps.min(num_play_warps);
                //
                // // make sure we only compare what can be compared
                // box_sched.prioritized_warp_ids.split_off(limit);
                // play_sched.prioritized_warp_ids.split_off(limit);
                //
                // assert_eq!(
                //     box_sched.prioritized_warp_ids.len(),
                //     play_sched.prioritized_warp_ids.len(),
                // );
            }
        }

        let mut partitions_added = 0;
        for (partition_id, partition) in play_sim.partition_units().enumerate() {
            assert!(partition_id < num_partitions);
            play_sim_state.dram_latency_queue_per_partition[partition_id] = partition
                .dram_latency_queue()
                .into_iter()
                .map(Into::into)
                .collect();
            // .extend(partition.dram_latency_queue().into_iter().map(Into::into));
            partitions_added += 1;

            play_sim_state.dram_arbitration_per_partition[partition_id] =
                testing::state::Arbitration {
                    last_borrower: partition.last_borrower(),
                    shared_credit: partition.shared_credit(),
                    private_credit: partition.private_credit().into(),
                };
        }
        assert_eq!(partitions_added, num_partitions);

        let mut sub_partitions_added = 0;
        for (sub_id, sub) in play_sim.sub_partitions().enumerate() {
            play_sim_state.interconn_to_l2_queue_per_sub[sub_id] = sub
                .interconn_to_l2_queue()
                .into_iter()
                .map(Into::into)
                .collect();
            // .extend(sub.interconn_to_l2_queue().into_iter().map(Into::into));
            play_sim_state.l2_to_interconn_queue_per_sub[sub_id] = sub
                .l2_to_interconn_queue()
                .into_iter()
                .map(Into::into)
                .collect();
            // .extend(sub.l2_to_interconn_queue().into_iter().map(Into::into));
            play_sim_state.dram_to_l2_queue_per_sub[sub_id] =
                sub.dram_to_l2_queue().into_iter().map(Into::into).collect();
            // .extend(sub.dram_to_l2_queue().into_iter().map(Into::into));
            play_sim_state.l2_to_dram_queue_per_sub[sub_id] =
                sub.l2_to_dram_queue().into_iter().map(Into::into).collect();
            // .extend(sub.l2_to_dram_queue().into_iter().map(Into::into));

            play_sim_state.l2_cache_per_sub[sub_id] = Some(testing::state::Cache {
                lines: sub.l2_cache().lines().into_iter().map(Into::into).collect(),
            });
            sub_partitions_added += 1;
        }
        assert_eq!(sub_partitions_added, num_sub_partitions);
        (box_sim_state, play_sim_state)
    }

    // #[deprecated]
    // #[inline]
    // fn gather_box_simulation_state(
    //     // num_clusters: usize,
    //     // cores_per_cluster: usize,
    //     // num_partitions: usize,
    //     // num_sub_partitions: usize,
    //     box_sim: &mut super::MockSimulator<ic::ToyInterconnect<super::Packet>>,
    //     box_sim_state: &mut testing::state::Simulation,
    //     _trace_provider: TraceProvider,
    // ) {
    //     // ) -> testing::state::Simulation {
    //     // let mut box_sim_state = testing::state::Simulation::new(
    //     //     num_clusters,
    //     //     cores_per_cluster,
    //     //     num_partitions,
    //     //     num_sub_partitions,
    //     // );
    //
    //     box_sim_state.last_cluster_issue = box_sim.last_cluster_issue;
    //
    //     for (cluster_id, cluster) in box_sim.clusters.iter().enumerate() {
    //         // cluster: core sim order
    //         box_sim_state.core_sim_order_per_cluster[cluster_id] =
    //             cluster.core_sim_order.iter().copied().collect();
    //
    //         for (core_id, core) in cluster.cores.lock().unwrap().iter().enumerate() {
    //             let global_core_id =
    //                 cluster_id * box_sim.config.num_cores_per_simt_cluster + core_id;
    //             assert_eq!(core.inner.core_id, global_core_id);
    //
    //             // this is the one we will use (unless the assertion is ever false)
    //             let core_id = core.inner.core_id;
    //
    //             // core: functional units
    //             let num_fus = core.functional_units.len();
    //             box_sim_state.functional_unit_pipelines_per_core[core_id]
    //                 .resize(2 * num_fus, testing::state::RegisterSet::default());
    //
    //             for (fu_id, fu) in core.functional_units.iter().enumerate() {
    //                 let _fu = fu.lock().unwrap();
    //                 let issue_port = core.issue_ports[fu_id];
    //                 let issue_reg: super::register_set::RegisterSet = core.inner.pipeline_reg
    //                     [issue_port as usize]
    //                     .borrow()
    //                     .clone();
    //                 assert_eq!(issue_port, issue_reg.stage);
    //
    //                 box_sim_state.functional_unit_pipelines_per_core[core_id][fu_id] =
    //                     issue_reg.into();
    //             }
    //
    //             // box_sim_state.functional_unit_pipelines_per_core[core_id]
    //             //     .resize(num_fus, testing::state::RegisterSet::default());
    //             for (fu_id, fu) in core.functional_units.iter().enumerate() {
    //                 let fu = fu.lock().unwrap();
    //                 box_sim_state.functional_unit_pipelines_per_core[core_id][num_fus + fu_id] =
    //                     testing::state::RegisterSet {
    //                         name: fu.id().to_string(),
    //                         pipeline: fu
    //                             .pipeline()
    //                             .iter()
    //                             .map(|reg| reg.clone().map(Into::into))
    //                             .collect(),
    //                     };
    //                 // .push();
    //             }
    //             // core: operand collector
    //             box_sim_state.operand_collector_per_core[core_id] =
    //                 Some(core.inner.operand_collector.borrow().deref().into());
    //             // core: schedulers
    //             box_sim_state.scheduler_per_core[core_id] =
    //                 core.schedulers.iter().map(Into::into).collect();
    //             // .extend(core.schedulers.iter().map(Into::into));
    //             // core: l2 cache
    //             let ldst_unit = core.inner.load_store_unit.lock().unwrap();
    //
    //             // core: pending register writes
    //             box_sim_state.pending_register_writes_per_core[core_id] = ldst_unit
    //                 .pending_writes
    //                 .clone()
    //                 .into_iter()
    //                 .flat_map(|(warp_id, pending_registers)| {
    //                     pending_registers
    //                         .into_iter()
    //                         .map(
    //                             move |(reg_num, pending)| testing::state::PendingRegisterWrites {
    //                                 warp_id,
    //                                 reg_num,
    //                                 pending,
    //                             },
    //                         )
    //                 })
    //                 .collect();
    //
    //             box_sim_state.pending_register_writes_per_core[core_id].sort();
    //             // let l1d_tag_array = ldst_unit.data_l1.unwrap().tag_array;
    //             // dbg!(&l1d_tag_array);
    //             // let l1d_tag_array = ldst_unit.data_l1.unwrap().tag_array;
    //         }
    //     }
    //
    //     for (partition_id, partition) in box_sim.mem_partition_units.iter().enumerate() {
    //         box_sim_state.dram_latency_queue_per_partition[partition_id] = partition
    //             .dram_latency_queue
    //             .clone()
    //             .into_iter()
    //             .map(Into::into)
    //             .collect();
    //     }
    //     for (sub_id, sub) in box_sim.mem_sub_partitions.iter().enumerate() {
    //         let sub = sub.borrow();
    //         let l2_cache = sub.l2_cache.as_ref().unwrap();
    //         let l2_cache: &ported::l2::DataL2<
    //             ic::L2Interface<fifo::FifoQueue<ported::mem_fetch::MemFetch>>,
    //         > = l2_cache.as_any().downcast_ref().unwrap();
    //
    //         box_sim_state.l2_cache_per_sub[sub_id] =
    //             Some(l2_cache.inner.inner.tag_array.clone().into());
    //
    //         for (dest_queue, src_queue) in [
    //             (
    //                 &mut box_sim_state.interconn_to_l2_queue_per_sub[sub_id],
    //                 &sub.interconn_to_l2_queue,
    //             ),
    //             (
    //                 &mut box_sim_state.l2_to_interconn_queue_per_sub[sub_id],
    //                 &sub.l2_to_interconn_queue,
    //             ),
    //             (
    //                 &mut box_sim_state.l2_to_dram_queue_per_sub[sub_id],
    //                 &sub.l2_to_dram_queue.lock().unwrap(),
    //             ),
    //             (
    //                 &mut box_sim_state.dram_to_l2_queue_per_sub[sub_id],
    //                 &sub.dram_to_l2_queue,
    //             ),
    //         ] {
    //             *dest_queue = src_queue.clone().into_iter().map(Into::into).collect();
    //             // dest_queue.extend(src_queue.clone().into_iter().map(Into::into));
    //         }
    //     }
    //     // box_sim_state
    // }
    //
    // #[deprecated]
    // #[inline]
    // fn gather_play_simulation_state(
    //     // num_clusters: usize,
    //     // cores_per_cluster: usize,
    //     // num_partitions: usize,
    //     // num_sub_partitions: usize,
    //     // num_schedulers: usize,
    //     play_sim: &mut playground::Accelsim,
    //     play_sim_state: &mut testing::state::Simulation,
    //     _trace_provider: TraceProvider,
    // ) {
    //     // ) -> testing::state::Simulation {
    //     // let mut play_sim_state = testing::state::Simulation::new(
    //     //     num_clusters,
    //     //     cores_per_cluster,
    //     //     num_partitions,
    //     //     num_sub_partitions,
    //     //     num_schedulers,
    //     // );
    //
    //     play_sim_state.last_cluster_issue = play_sim.last_cluster_issue() as usize;
    //
    //     for (cluster_id, cluster) in play_sim.clusters().enumerate() {
    //         // cluster: core sim order
    //         play_sim_state.core_sim_order_per_cluster[cluster_id] = cluster.core_sim_order().into();
    //     }
    //     for (core_id, core) in play_sim.cores().enumerate() {
    //         let fu_issue_regs: Vec<_> = core
    //             .functional_unit_issue_register_sets()
    //             .into_iter()
    //             .map(testing::state::RegisterSet::from)
    //             .collect();
    //         let num_fu_issue_regs = fu_issue_regs.len();
    //         // let fu_names: HashSet<_> = fu_issue_regs.iter().map(|fu| fu.get_name()).collect();
    //         // let valid_units: HashSet<_> = fu_issue_regs.iter().map(|fu| fu.name.clone()).collect();
    //
    //         play_sim_state.functional_unit_pipelines_per_core[core_id]
    //             .resize(num_fu_issue_regs, testing::state::RegisterSet::default());
    //
    //         for (reg_id, regs) in fu_issue_regs.into_iter().enumerate() {
    //             play_sim_state.functional_unit_pipelines_per_core[core_id][reg_id] = regs.into();
    //         }
    //         // let valid_units: HashSet<_> = box_sim_state.functional_unit_pipelines_per_core[core_id]
    //         // let valid_units: HashSet<_> = [
    //
    //         let fu_simd_pipelines: Vec<testing::state::RegisterSet> = core
    //             .functional_unit_simd_pipeline_register_sets()
    //             .into_iter()
    //             .map(testing::state::RegisterSet::from)
    //             // .filter(|fu| valid_units.contains(&fu.name))
    //             .filter(|fu| match fu.name.as_str() {
    //                 "SPUnit" | "LdstUnit" => true,
    //                 _ => false,
    //             })
    //             .collect();
    //         let num_fu_simd_pipelines = fu_simd_pipelines.len();
    //
    //         play_sim_state.functional_unit_pipelines_per_core[core_id].resize(
    //             num_fu_issue_regs + num_fu_simd_pipelines,
    //             testing::state::RegisterSet::default(),
    //         );
    //         // assert_eq!(num_fu_issue_regs, num_fu_simd_pipelines);
    //         for (reg_id, regs) in fu_simd_pipelines
    //             .into_iter()
    //             // .filter(|fu| valid_units.contains(&fu.name()))
    //             .enumerate()
    //         {
    //             play_sim_state.functional_unit_pipelines_per_core[core_id]
    //                 [num_fu_issue_regs + reg_id] = regs.into();
    //         }
    //
    //         // core: pending register writes
    //         play_sim_state.pending_register_writes_per_core[core_id] = core
    //             .pending_register_writes()
    //             .into_iter()
    //             .map(Into::into)
    //             .collect();
    //         play_sim_state.pending_register_writes_per_core[core_id].sort();
    //
    //         // core: operand collector
    //         let coll = core.operand_collector();
    //         play_sim_state.operand_collector_per_core[core_id] = Some(coll.into());
    //         // core: scheduler units
    //         // let schedulers = core.schedulers();
    //         // play_sim_state.scheduler_per_core[core_id]
    //         // .resize(schedulers.len(), testing::state::Scheduler::default());
    //
    //         for (sched_idx, play_scheduler) in core.schedulers().into_iter().enumerate() {
    //             play_sim_state.scheduler_per_core[core_id][sched_idx] = play_scheduler.into();
    //         }
    //     }
    //
    //     for (partition_id, partition) in play_sim.partition_units().enumerate() {
    //         play_sim_state.dram_latency_queue_per_partition[partition_id] = partition
    //             .dram_latency_queue()
    //             .into_iter()
    //             .map(Into::into)
    //             .collect();
    //     }
    //     for (sub_id, sub) in play_sim.sub_partitions().enumerate() {
    //         play_sim_state.interconn_to_l2_queue_per_sub[sub_id] = sub
    //             .interconn_to_l2_queue()
    //             .into_iter()
    //             .map(Into::into)
    //             .collect();
    //         // .extend(sub.interconn_to_l2_queue().into_iter().map(Into::into));
    //         play_sim_state.l2_to_interconn_queue_per_sub[sub_id] = sub
    //             .l2_to_interconn_queue()
    //             .into_iter()
    //             .map(Into::into)
    //             .collect();
    //         // .extend(sub.l2_to_interconn_queue().into_iter().map(Into::into));
    //         play_sim_state.dram_to_l2_queue_per_sub[sub_id] =
    //             sub.dram_to_l2_queue().into_iter().map(Into::into).collect();
    //         // .extend(sub.dram_to_l2_queue().into_iter().map(Into::into));
    //         play_sim_state.l2_to_dram_queue_per_sub[sub_id] =
    //             sub.l2_to_dram_queue().into_iter().map(Into::into).collect();
    //         // .extend(sub.l2_to_dram_queue().into_iter().map(Into::into));
    //
    //         play_sim_state.l2_cache_per_sub[sub_id] = Some(testing::state::Cache {
    //             lines: sub.l2_cache().lines().into_iter().map(Into::into).collect(),
    //         });
    //     }
    //
    //     // play_sim_state
    // }

    fn run_lockstep(trace_dir: &Path, trace_provider: TraceProvider) -> eyre::Result<()> {
        use accelsim::tracegen::reader::Command as AccelsimCommand;

        let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));

        let box_trace_dir = trace_dir.join("trace");
        let accelsim_trace_dir = trace_dir.join("accelsim-trace");
        utils::fs::create_dirs(&box_trace_dir)?;
        utils::fs::create_dirs(&accelsim_trace_dir)?;

        let native_box_commands_path = box_trace_dir.join("commands.json");
        let native_accelsim_kernelslist_path = accelsim_trace_dir.join("kernelslist.g");

        let (box_commands_path, accelsim_kernelslist_path) = match trace_provider {
            TraceProvider::Native => {
                // use native traces
                (native_box_commands_path, native_accelsim_kernelslist_path)
            }
            TraceProvider::Accelsim => {
                assert!(native_accelsim_kernelslist_path.is_file());
                let generated_box_commands_path = box_trace_dir.join("accelsim.commands.json");
                println!(
                    "generating commands {}",
                    generated_box_commands_path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                );

                let reader = utils::fs::open_readable(&native_accelsim_kernelslist_path)?;
                let accelsim_commands =
                    accelsim::tracegen::reader::read_commands(&accelsim_trace_dir, reader)?;

                let commands: Vec<_> = accelsim_commands
                    .into_iter()
                    .map(|cmd| match cmd {
                        AccelsimCommand::MemcpyHtoD(memcopy) => {
                            Ok::<_, eyre::Report>(trace_model::Command::MemcpyHtoD(memcopy))
                        }
                        AccelsimCommand::KernelLaunch((mut kernel, metadata)) => {
                            // transform kernel instruction trace
                            let kernel_trace_path = accelsim_trace_dir.join(&kernel.trace_file);
                            let reader = utils::fs::open_readable(kernel_trace_path)?;
                            let parsed_trace = accelsim::tracegen::reader::read_trace_instructions(
                                reader,
                                metadata.trace_version,
                                metadata.line_info,
                                &kernel,
                            )?;

                            let generated_kernel_trace_name =
                                format!("accelsim-kernel-{}.msgpack", kernel.id);
                            let generated_kernel_trace_path =
                                box_trace_dir.join(&generated_kernel_trace_name);

                            let mut writer =
                                utils::fs::open_writable(&generated_kernel_trace_path)?;
                            rmp_serde::encode::write(&mut writer, &parsed_trace)?;

                            // also save as json for inspection
                            let mut writer = utils::fs::open_writable(
                                generated_kernel_trace_path.with_extension("json"),
                            )?;
                            serde_json::to_writer_pretty(&mut writer, &parsed_trace)?;

                            // update the kernel trace path
                            kernel.trace_file = generated_kernel_trace_name;

                            Ok::<_, eyre::Report>(trace_model::Command::KernelLaunch(kernel))
                        }
                    })
                    .try_collect()?;

                let mut json_serializer = serde_json::Serializer::with_formatter(
                    utils::fs::open_writable(&generated_box_commands_path)?,
                    serde_json::ser::PrettyFormatter::with_indent(b"    "),
                );
                commands.serialize(&mut json_serializer)?;

                (
                    generated_box_commands_path,
                    native_accelsim_kernelslist_path,
                )
            }
            TraceProvider::Box => {
                assert!(native_box_commands_path.is_file());
                let generated_kernelslist_path = accelsim_trace_dir.join("box-kernelslist.g");
                println!(
                    "generating commands {}",
                    generated_kernelslist_path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                );
                let mut commands_writer = utils::fs::open_writable(&generated_kernelslist_path)?;
                accelsim::tracegen::writer::generate_commands(
                    &native_box_commands_path,
                    &mut commands_writer,
                )?;
                drop(commands_writer);

                let reader = utils::fs::open_readable(&native_box_commands_path)?;
                let commands: Vec<Command> = serde_json::from_reader(reader)?;

                for cmd in commands {
                    if let Command::KernelLaunch(kernel) = cmd {
                        // generate trace for kernel
                        let generated_kernel_trace_path = trace_dir.join(format!(
                            "accelsim-trace/kernel-{}.box.traceg",
                            kernel.id + 1
                        ));
                        println!(
                            "generating trace {} for kernel {}",
                            generated_kernel_trace_path
                                .file_name()
                                .unwrap_or_default()
                                .to_string_lossy(),
                            kernel.id
                        );
                        let mut trace_writer =
                            utils::fs::open_writable(generated_kernel_trace_path)?;
                        accelsim::tracegen::writer::generate_trace(
                            &box_trace_dir,
                            &kernel,
                            &mut trace_writer,
                        )?;
                    }
                }
                (native_box_commands_path, generated_kernelslist_path)
            }
        };

        dbg!(&box_commands_path);
        dbg!(&accelsim_kernelslist_path);

        let gpgpusim_config = manifest_dir.join("accelsim/gtx1080/gpgpusim.config");
        let trace_config = manifest_dir.join("accelsim/gtx1080/gpgpusim.trace.config");
        let inter_config = manifest_dir.join("accelsim/gtx1080/config_fermi_islip.icnt");

        assert!(trace_dir.is_dir());
        assert!(box_trace_dir.is_dir());
        assert!(box_commands_path.is_file());
        assert!(accelsim_kernelslist_path.is_file());
        assert!(gpgpusim_config.is_file());
        assert!(trace_config.is_file());
        assert!(inter_config.is_file());

        // debugging config
        let box_config = Arc::new(config::GPUConfig {
            num_simt_clusters: 20,                   // 20
            num_cores_per_simt_cluster: 4,           // 1
            num_schedulers_per_core: 2,              // 2
            num_memory_controllers: 8,               // 8
            num_sub_partition_per_memory_channel: 2, // 2
            fill_l2_on_memcopy: true,                // true
            ..config::GPUConfig::default()
        });

        let box_interconn = Arc::new(ic::ToyInterconnect::new(
            box_config.num_simt_clusters,
            box_config.num_memory_controllers * box_config.num_sub_partition_per_memory_channel,
        ));

        let mut box_sim = super::MockSimulator::new(
            box_interconn,
            box_config,
            &box_trace_dir,
            &box_commands_path,
        );

        let args = vec![
            "-trace",
            accelsim_kernelslist_path.as_os_str().to_str().unwrap(),
            "-config",
            gpgpusim_config.as_os_str().to_str().unwrap(),
            "-config",
            trace_config.as_os_str().to_str().unwrap(),
            "-inter_config_file",
            inter_config.as_os_str().to_str().unwrap(),
        ];
        dbg!(&args);

        let play_config = playground::Config::default();
        let mut play_sim = playground::Accelsim::new(&play_config, &args)?;

        let mut play_time_cycle = std::time::Duration::ZERO;
        let mut play_time_other = std::time::Duration::ZERO;
        let mut box_time_cycle = std::time::Duration::ZERO;
        let mut box_time_other = std::time::Duration::ZERO;

        let mut gather_state_time = std::time::Duration::ZERO;
        let mut gather_box_state_time = std::time::Duration::ZERO;
        let mut gather_play_state_time = std::time::Duration::ZERO;

        let mut last_valid_box_sim_state = None;
        let mut last_valid_play_sim_state = None;

        let mut cycle = 0;

        let use_full_diff = std::env::var("FULL_DIFF")
            .unwrap_or_default()
            .to_lowercase()
            == "yes";
        let check_after: u64 = std::env::var("CHECK_AFTER")
            .ok()
            .as_deref()
            .map(str::parse)
            .transpose()?
            .unwrap_or(0);
        let check_every: u64 = std::env::var("CHECK_EVERY")
            .ok()
            .as_deref()
            .map(str::parse)
            .transpose()?
            .unwrap_or(200);
        assert!(check_every >= 1);

        // let _num_schedulers = box_sim.config.num_schedulers_per_core;
        // let num_clusters = box_sim.config.num_simt_clusters;
        // let cores_per_cluster = box_sim.config.num_cores_per_simt_cluster;
        // assert_eq!(
        //     box_sim.config.total_cores(),
        //     num_clusters * cores_per_cluster
        // );
        // let _num_partitions = box_sim.mem_partition_units.len();
        // let _num_sub_partitions = box_sim.mem_sub_partitions.len();
        //
        // let mut box_sim_state = testing::state::Simulation::new(
        //     num_clusters,
        //     cores_per_cluster,
        //     num_partitions,
        //     num_sub_partitions,
        // );
        //
        // let mut play_sim_state = testing::state::Simulation::new(
        //     num_clusters,
        //     cores_per_cluster,
        //     num_partitions,
        //     num_sub_partitions,
        // );

        box_sim.process_commands();
        box_sim.launch_kernels();

        while play_sim.commands_left() || play_sim.kernels_left() {
            let mut start = Instant::now();
            play_sim.process_commands();
            play_sim.launch_kernels();
            play_time_other += start.elapsed();

            // check that memcopy commands were handled correctly
            start = Instant::now();
            let (box_sim_state, play_sim_state) =
                gather_simulation_state(&mut box_sim, &mut play_sim, trace_provider);
            gather_state_time += start.elapsed();

            // start = Instant::now();
            // gather_box_simulation_state(&mut box_sim, &mut box_sim_state, trace_provider);
            // box_sim_state = gather_box_simulation_state(
            //     num_clusters,
            //     cores_per_cluster,
            //     num_partitions,
            //     num_sub_partitions,
            //     &mut box_sim,
            //     trace_provider,
            // );
            // gather_box_state_time += start.elapsed();
            // gather_state_time += start.elapsed();

            // start = Instant::now();
            // gather_play_simulation_state(&mut play_sim, &mut play_sim_state, trace_provider);
            // play_sim_state = gather_play_simulation_state(
            //     num_clusters,
            //     cores_per_cluster,
            //     num_partitions,
            //     num_sub_partitions,
            //     &mut play_sim,
            //     trace_provider,
            // );
            // gather_play_state_time += start.elapsed();
            // gather_state_time += start.elapsed();

            if use_full_diff {
                full_diff::assert_eq!(&box_sim_state, &play_sim_state);
            } else {
                diff::assert_eq!(box: &box_sim_state, play: &play_sim_state);
            }

            // start = Instant::now();
            // box_sim.process_commands();
            // box_sim.launch_kernels();
            // box_time_other += start.elapsed();

            let mut finished_kernel_uid: Option<u32> = None;
            loop {
                if !play_sim.active() {
                    break;
                }

                start = Instant::now();
                play_sim.cycle();
                cycle = play_sim.get_cycle();
                play_time_cycle += start.elapsed();

                start = Instant::now();
                box_sim.cycle();
                box_sim.set_cycle(cycle);
                box_time_cycle += start.elapsed();

                if cycle >= check_after && cycle % check_every == 0 {
                    start = Instant::now();
                    let (box_sim_state, play_sim_state) =
                        gather_simulation_state(&mut box_sim, &mut play_sim, trace_provider);
                    gather_state_time += start.elapsed();

                    start = Instant::now();
                    // gather_box_simulation_state(&mut box_sim, &mut box_sim_state, trace_provider);
                    // box_sim_state = gather_box_simulation_state(
                    //     num_clusters,
                    //     cores_per_cluster,
                    //     num_partitions,
                    //     num_sub_partitions,
                    //     &mut box_sim,
                    //     trace_provider,
                    // );
                    gather_box_state_time += start.elapsed();

                    start = Instant::now();
                    // gather_play_simulation_state(&mut play_sim, &mut play_sim_state, trace_provider);
                    // play_sim_state = gather_play_simulation_state(
                    //     num_clusters,
                    //     cores_per_cluster,
                    //     num_partitions,
                    //     num_sub_partitions,
                    //     num_schedulers,
                    //     &mut play_sim,
                    //     trace_provider,
                    // );
                    gather_play_state_time += start.elapsed();

                    // sanity checks
                    // assert_eq!(
                    //     schedulers.len(),
                    //     box_sim_state.scheduler_per_core[core_id].len()
                    // );
                    // let box_sched = &mut box_sim_state.scheduler_per_core[core_id][sched_idx];
                    // let play_sched = &mut play_sim_state.scheduler_per_core[core_id][sched_idx];
                    //
                    // let num_box_warps = box_sched.prioritized_warp_ids.len();
                    // let num_play_warps = play_sched.prioritized_warp_ids.len();
                    // let limit = num_box_warps.min(num_play_warps);
                    //
                    // // make sure we only compare what can be compared
                    // box_sched.prioritized_warp_ids.split_off(limit);
                    // // box_sched.prioritized_dynamic_warp_ids.split_off(limit);
                    // play_sched.prioritized_warp_ids.split_off(limit);
                    // // play_sched.prioritized_dynamic_warp_ids.split_off(limit);
                    //
                    // assert_eq!(
                    //     box_sched.prioritized_warp_ids.len(),
                    //     play_sched.prioritized_warp_ids.len(),
                    // );
                    // // assert_eq!(
                    // //     box_sched.prioritized_dynamic_warp_ids.len(),
                    // //     play_sched.prioritized_dynamic_warp_ids.len(),
                    // // );

                    if box_sim_state != play_sim_state {
                        println!(
                            "validated play state for cycle {}: {:#?}",
                            cycle - 1,
                            &last_valid_play_sim_state
                        );

                        {
                            serde_json::to_writer_pretty(
                                utils::fs::open_writable(
                                    manifest_dir.join("debug.playground.state.json"),
                                )?,
                                &last_valid_play_sim_state,
                            )?;

                            // format!("{:#?}", ).as_bytes(),
                            serde_json::to_writer_pretty(
                                utils::fs::open_writable(
                                    manifest_dir.join("debug.box.state.json"),
                                )?,
                                &last_valid_box_sim_state,
                            )?;
                            // .write_all(format!("{:#?}", last_valid_box_sim_state).as_bytes())?;
                        };

                        // dbg!(&box_sim.allocations);
                        // for (sub_id, sub) in play_sim.sub_partitions().enumerate() {
                        //     let play_icnt_l2_queue = sub
                        //         .interconn_to_l2_queue()
                        //         .iter()
                        //         .map(|fetch| fetch.get_addr())
                        //         .collect::<Vec<_>>();
                        //     dbg!(sub_id, play_icnt_l2_queue);
                        // }
                        //
                        // for (sub_id, sub) in box_sim.mem_sub_partitions.iter().enumerate() {
                        //     let box_icnt_l2_queue = sub
                        //         .borrow()
                        //         .interconn_to_l2_queue
                        //         .iter()
                        //         .map(|fetch| fetch.addr())
                        //         .collect::<Vec<_>>();
                        //     dbg!(sub_id, box_icnt_l2_queue);
                        // }
                    }
                    println!("checking for diff after cycle {cycle}");

                    if use_full_diff {
                        full_diff::assert_eq!(&box_sim_state, &play_sim_state);
                    } else {
                        diff::assert_eq!(box: &box_sim_state, play: &play_sim_state);
                    }

                    // this should be okay performance wise (copy, no allocation)
                    last_valid_box_sim_state = Some(box_sim_state.clone());
                    last_valid_play_sim_state = Some(play_sim_state.clone());
                }

                // box out of loop
                start = Instant::now();
                if !box_sim.active() {
                    box_sim.process_commands();
                    box_sim.launch_kernels();
                }

                if let Some(kernel) = box_sim.finished_kernel() {
                    box_sim.cleanup_finished_kernel(&kernel);
                }
                box_time_other += start.elapsed();

                finished_kernel_uid = play_sim.finished_kernel_uid();
                if finished_kernel_uid.is_some() {
                    break;
                }
            }

            if let Some(uid) = finished_kernel_uid {
                play_sim.cleanup_finished_kernel(uid);
            }

            if play_sim.limit_reached() {
                println!(
                    "GPGPU-Sim: ** break due to reaching the maximum cycles (or instructions) **"
                );
                std::io::stdout().flush()?;
                break;
            }
        }

        if cycle > 0 {
            let box_time_cycle = box_time_cycle / u32::try_from(cycle).unwrap();
            let box_time_other = box_time_other / u32::try_from(cycle).unwrap();
            let play_time_cycle = play_time_cycle / u32::try_from(cycle).unwrap();
            let play_time_other = play_time_other / u32::try_from(cycle).unwrap();
            println!(
                "box time  (cycle):\t {:>3.6} ms",
                box_time_cycle.as_secs_f64() * 1000.0
            );
            println!(
                "box time  (other):\t {:>3.6} ms",
                box_time_other.as_secs_f64() * 1000.0
            );
            println!(
                "play time (cycle):\t {:>3.6} ms",
                play_time_cycle.as_secs_f64() * 1000.0
            );
            println!(
                "play time (other):\t {:>3.6} ms",
                play_time_other.as_secs_f64() * 1000.0
            );
        }

        let num_checks = u32::try_from(cycle.saturating_sub(check_after) / check_every).unwrap();
        if num_checks > 0 {
            let gather_box_state_time = gather_box_state_time / num_checks;
            let gather_play_state_time = gather_play_state_time / num_checks;
            let gather_state_time = gather_state_time / num_checks;

            dbg!(gather_box_state_time);
            dbg!(gather_play_state_time);
            dbg!(gather_box_state_time + gather_play_state_time);
            dbg!(gather_state_time);
        }
        dbg!(&cycle);

        let play_stats = play_sim.stats().clone();
        let box_stats = box_sim.stats();
        // let playground_dur = start.elapsed();

        // dbg!(&play_stats);
        // dbg!(&box_stats);

        // dbg!(&playground_dur);
        // dbg!(&box_dur);

        // compare stats here
        diff::assert_eq!(
            play: &stats::PerCache(play_stats.l1i_stats.clone().convert()),
            box: &box_stats.l1i_stats
        );
        diff::assert_eq!(
            play: &stats::PerCache(play_stats.l1d_stats.clone().convert()),
            box: &box_stats.l1d_stats,
        );
        diff::assert_eq!(
            play: &stats::PerCache(play_stats.l1t_stats.clone().convert()),
            box: &box_stats.l1t_stats,
        );
        diff::assert_eq!(
            play: &stats::PerCache(play_stats.l1c_stats.clone().convert()),
            box: &box_stats.l1c_stats,
        );
        diff::assert_eq!(
            play: &stats::PerCache(play_stats.l2d_stats.clone().convert()),
            box: &box_stats.l2d_stats,
        );

        diff::assert_eq!(
            play: play_stats.accesses,
            box: playground::stats::Accesses::from(box_stats.accesses.clone())
        );

        // dbg!(&play_stats.accesses);
        // dbg!(&box_stats.accesses);
        //
        // dbg!(&play_stats.instructions);
        // dbg!(&box_stats.instructions);
        //
        // dbg!(&play_stats.sim);
        // dbg!(&box_stats.sim);

        let box_dram_stats = playground::stats::DRAM::from(box_stats.dram.clone());

        // dbg!(&play_stats.dram);
        // dbg!(&box_dram_stats);

        diff::assert_eq!(play: &play_stats.dram, box: &box_dram_stats);

        let box_instructions =
            playground::stats::InstructionCounts::from(box_stats.instructions.clone());
        diff::assert_eq!(play: &play_stats.instructions, box: &box_instructions);

        // dbg!(&play_stats.sim, &box_stats.sim);
        diff::assert_eq!(
            play: &play_stats.sim,
            box: &playground::stats::Sim::from(box_stats.sim.clone()),
        );

        // this uses our custom PartialEq::eq implementation
        assert_eq!(&play_stats, &box_stats);
        // assert!(false);
        Ok(())
    }

    macro_rules! lockstep_checks {
        ($($name:ident: $path:expr,)*) => {
            $(
                paste::paste! {
                    #[ignore = "native traces cannot be compared"]
                    #[test]
                    fn [<lockstep_native_ $name _test>]() -> color_eyre::eyre::Result<()> {
                        let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
                        let trace_dir = manifest_dir.join($path);
                        run_lockstep(&trace_dir, TraceProvider::Native)
                    }

                    #[test]
                    fn [<lockstep_accelsim_ $name _test>]() -> color_eyre::eyre::Result<()> {
                        let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
                        let trace_dir = manifest_dir.join($path);
                        run_lockstep(&trace_dir, TraceProvider::Accelsim)
                    }

                    #[test]
                    fn [<lockstep_box_ $name _test>]() -> color_eyre::eyre::Result<()> {
                        let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
                        let trace_dir = manifest_dir.join($path);
                        run_lockstep(&trace_dir, TraceProvider::Box)
                    }
                }
            )*
        }
    }

    lockstep_checks! {
        // vectoradd
        vectoradd_32_100: "results/vectorAdd/vectorAdd-dtype-32-length-100",
        vectoradd_32_1000: "results/vectorAdd/vectorAdd-dtype-32-length-1000",
        vectoradd_32_10000: "results/vectorAdd/vectorAdd-dtype-32-length-10000",
        // simple matrixmul
        simple_matrixmul_32_32_32_32:
            "results/simple_matrixmul/simple_matrixmul-dtype-32-m-32-n-32-p-32",
        simple_matrixmul_32_32_32_64:
            "results/simple_matrixmul/simple_matrixmul-dtype-32-m-32-n-32-p-64",
        simple_matrixmul_32_64_128_128:
            "results/simple_matrixmul/simple_matrixmul-dtype-32-m-64-n-128-p-128",
        // matrixmul (shared memory)
        matrixmul_32_32: "results/matrixmul/matrixmul-dtype-32-rows-32",
        matrixmul_32_64: "results/matrixmul/matrixmul-dtype-32-rows-64",
        matrixmul_32_128: "results/matrixmul/matrixmul-dtype-32-rows-128",
        matrixmul_32_256: "results/matrixmul/matrixmul-dtype-32-rows-256",
        // transpose
        transpose_256_naive: "results/transpose/transpose-dim-256-variant-naive-repeat-1",
        transpose_256_coalesced: "results/transpose/transpose-dim-256-variant-coalesced-repeat-1",
        transpose_256_optimized: "results/transpose/transpose-dim-256-variant-optimized-repeat-1",
    }

    macro_rules! accelsim_compat_tests {
        ($($name:ident: $input:expr,)*) => {
            $(
                #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
                async fn $name() -> color_eyre::eyre::Result<()> {
                    use validate::materialize::{self, Benchmarks};

                    // load benchmark config
                    let (benchmark_name, input_idx) = $input;
                    let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));

                    let benchmarks_path = manifest_dir.join("test-apps/test-apps-materialized.yml");
                    let reader = utils::fs::open_readable(benchmarks_path)?;
                    let benchmarks = Benchmarks::from_reader(reader)?;
                    let bench_config = benchmarks.get_single_config(benchmark_name, input_idx).unwrap();

                    let traces_dir = &bench_config.accelsim_trace.traces_dir;
                    let kernelslist = traces_dir.join("kernelslist.g");

                    let materialize::AccelsimSimConfigFiles {
                        config,
                        config_dir,
                        trace_config,
                        inter_config,
                    } = bench_config.accelsim_simulate.configs.clone();

                    let sim_config = accelsim::SimConfig {
                        config: Some(config),
                        config_dir: Some(config_dir),
                        trace_config: Some(trace_config),
                        inter_config: Some(inter_config),
                    };

                    validate_playground_accelsim_compat(&traces_dir, &kernelslist, &sim_config).await
                }
            )*
        }
    }

    accelsim_compat_tests! {
        // vectoradd
        test_accelsim_compat_vectoradd_0: ("vectorAdd", 0),
        test_accelsim_compat_vectoradd_1: ("vectorAdd", 1),
        test_accelsim_compat_vectoradd_2: ("vectorAdd", 2),
        // simple matrixmul
        test_accelsim_compat_simple_matrixmul_0: ("simple_matrixmul", 0),
        test_accelsim_compat_simple_matrixmul_1: ("simple_matrixmul", 1),
        test_accelsim_compat_simple_matrixmul_17: ("simple_matrixmul", 17),
        // matrixmul (shared memory)
        test_accelsim_compat_matrixmul_0: ("matrixmul", 0),
        test_accelsim_compat_matrixmul_1: ("matrixmul", 1),
        test_accelsim_compat_matrixmul_2: ("matrixmul", 2),
        test_accelsim_compat_matrixmul_3: ("matrixmul", 3),
        // transpose
        test_accelsim_compat_transpose_0: ("transpose", 0),
        test_accelsim_compat_transpose_1: ("transpose", 1),
        test_accelsim_compat_transpose_2: ("transpose", 2),
    }

    // #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    // async fn test_playground_accelsim_compat() -> eyre::Result<()> {
    //     // load benchmark config
    //     let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
    //     use validate::materialize::{self, BenchmarkConfig, Benchmarks};
    //
    //     let reader =
    //         utils::fs::open_readable(manifest_dir.join("test-apps/test-apps-materialized.yml"))?;
    //     let benchmarks = Benchmarks::from_reader(reader)?;
    //     let bench_config = benchmarks.get_single_config("vectorAdd", 0).unwrap();
    //
    //     let traces_dir = &bench_config.accelsim_trace.traces_dir;
    //     let kernelslist = traces_dir.join("kernelslist.g");
    //
    //     let materialize::AccelsimSimConfigFiles {
    //         config,
    //         config_dir,
    //         trace_config,
    //         inter_config,
    //     } = bench_config.accelsim_simulate.configs.clone();
    //
    //     let sim_config = accelsim::SimConfig {
    //         config: Some(config),
    //         config_dir: Some(config_dir),
    //         trace_config: Some(trace_config),
    //         inter_config: Some(inter_config),
    //     };
    //
    //     validate_playground_accelsim_compat(&traces_dir, &kernelslist, &sim_config).await
    // }

    pub mod playground_sim {
        use async_process::Command;
        use color_eyre::{
            eyre::{self, WrapErr},
            Help,
        };
        use std::path::{Path, PathBuf};
        use std::time::{Duration, Instant};

        pub async fn simulate_trace(
            _traces_dir: impl AsRef<Path>,
            kernelslist: impl AsRef<Path>,
            sim_config: &accelsim::SimConfig,
            timeout: Option<Duration>,
            accelsim_compat_mode: bool,
        ) -> eyre::Result<(std::process::Output, Duration)> {
            let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
            #[cfg(debug_assertions)]
            let target = "debug";
            #[cfg(not(debug_assertions))]
            let target = "release";

            let playground_bin = manifest_dir.join("target").join(target).join("playground");
            let playground_bin = playground_bin
                .canonicalize()
                .wrap_err_with(|| {
                    eyre::eyre!(
                        "playground executable {} does not exist",
                        playground_bin.display()
                    )
                })
                .with_suggestion(|| format!("make sure to build playground with `cargo build -p playground` for the {target:?} target"))?;

            let gpgpu_sim_config = sim_config.config().unwrap();
            let trace_config = sim_config.trace_config().unwrap();
            let inter_config = sim_config.inter_config.as_ref().unwrap();

            let mut cmd = Command::new(playground_bin);
            if accelsim_compat_mode {
                cmd.env("ACCELSIM_COMPAT_MODE", "yes");
            }
            let args = vec![
                "--kernels",
                kernelslist.as_ref().as_os_str().to_str().unwrap(),
                "--config",
                gpgpu_sim_config.as_os_str().to_str().unwrap(),
                "--trace-config",
                trace_config.as_os_str().to_str().unwrap(),
                "--inter-config",
                inter_config.as_os_str().to_str().unwrap(),
            ];
            println!("{}", args.join(" "));
            cmd.args(args);

            // cmd.args(vec![
            //     "-trace",
            //     kernelslist.as_ref().as_os_str().to_str().unwrap(),
            //     "-config",
            //     gpgpu_sim_config.as_os_str().to_str().unwrap(),
            //     "-config",
            //     trace_config.as_os_str().to_str().unwrap(),
            //     "-inter_config_file",
            //     inter_config.as_os_str().to_str().unwrap(),
            // ]);
            dbg!(&cmd);

            let start = Instant::now();
            let result = match timeout {
                Some(timeout) => tokio::time::timeout(timeout, cmd.output()).await,
                None => Ok(cmd.output().await),
            };
            let result = result??;
            let dur = start.elapsed();

            if !result.status.success() {
                return Err(utils::CommandError::new(&cmd, result).into_eyre());
            }

            Ok((result, dur))
        }
    }

    async fn validate_playground_accelsim_compat(
        traces_dir: &Path,
        kernelslist: &Path,
        sim_config: &accelsim::SimConfig,
    ) -> eyre::Result<()> {
        dbg!(&traces_dir);
        dbg!(&kernelslist);
        dbg!(&sim_config);

        let parse_options = accelsim::parser::Options::default();
        let timeout = None;

        // run accelsim
        let (accelsim_stdout, accelsim_stderr, accelsim_stats) = {
            let (output, accelsim_dur) =
                accelsim_sim::simulate_trace(&traces_dir, &kernelslist, sim_config, timeout)
                    .await?;
            dbg!(&accelsim_dur);

            let stdout = utils::decode_utf8!(output.stdout);
            let stderr = utils::decode_utf8!(output.stderr);
            // println!("\nn{}\n", log);
            let log_reader = std::io::Cursor::new(&output.stdout);
            let stats = accelsim::parser::parse_stats(log_reader, &parse_options)?;
            (stdout, stderr, stats)
        };
        // dbg!(&accelsim_stats);

        // run playground in accelsim compat mode
        let accelsim_compat_mode = true;
        let (playground_stdout, playground_stderr, playground_stats) = {
            let (output, playground_dur) = playground_sim::simulate_trace(
                &traces_dir,
                &kernelslist,
                sim_config,
                timeout,
                accelsim_compat_mode,
            )
            .await?;
            dbg!(&playground_dur);

            let stdout = utils::decode_utf8!(output.stdout);
            let stderr = utils::decode_utf8!(output.stderr);
            // println!("\nn{}\n", log);
            let log_reader = std::io::Cursor::new(&output.stdout);
            let stats = accelsim::parser::parse_stats(log_reader, &parse_options)?;
            (stdout, stderr, stats)
        };
        // dbg!(&playground_stats);

        let filter_func =
            |((_name, _kernel, stat_name), _value): &((String, u16, String), f64)| -> bool {
                // we ignore rates and other stats that can vary per run
                !matches!(
                    stat_name.as_str(),
                    "gpgpu_silicon_slowdown"
                        | "gpgpu_simulation_rate"
                        | "gpgpu_simulation_time_sec"
                        | "gpu_ipc"
                        | "gpu_occupancy"
                        | "gpu_tot_ipc"
                        | "l1_inst_cache_total_miss_rate"
                        | "l2_bandwidth_gbps"
                )
            };

        let cmp_play_stats: accelsim::Stats =
            playground_stats.into_iter().filter(filter_func).collect();

        let cmp_accel_stats: accelsim::Stats = accelsim_stats
            .clone()
            .into_iter()
            .filter(filter_func)
            .collect();

        for stat in ["warp_instruction_count", "gpu_tot_sim_cycle"] {
            println!(
                "{:>15}:\t play={:.1}\t accel={:.1}",
                stat,
                cmp_play_stats.find_stat(stat).copied().unwrap_or_default(),
                cmp_accel_stats.find_stat(stat).copied().unwrap_or_default(),
            );
        }
        // diff::assert_eq!(
        //     play: cmp_play_stats.find_stat("warp_instruction_count"),
        //     accelsim: cmp_accel_stats.find_stat("warp_instruction_count"),
        // );
        // diff::assert_eq!(
        //     play: cmp_play_stats.find_stat("gpu_tot_sim_cycle"),
        //     accelsim: cmp_accel_stats.find_stat("gpu_tot_sim_cycle"),
        // );

        {
            // save the logs
            utils::fs::open_writable(traces_dir.join("debug.playground.stdout"))?
                .write_all(playground_stdout.as_bytes())?;
            utils::fs::open_writable(traces_dir.join("debug.playground.stderr"))?
                .write_all(playground_stderr.as_bytes())?;

            utils::fs::open_writable(traces_dir.join("debug.accelsim.stdout"))?
                .write_all(accelsim_stdout.as_bytes())?;
            utils::fs::open_writable(traces_dir.join("debug.accelsim.stderr"))?
                .write_all(accelsim_stderr.as_bytes())?;
        }

        // diff::assert_eq!(play: playground_stdout, accelsim: accelsim_stdout);
        diff::assert_eq!(play: cmp_play_stats, accelsim: cmp_accel_stats);
        // assert!(false);
        Ok(())
    }
}
