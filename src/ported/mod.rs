pub mod addrdec;
pub mod barrier;
pub mod cache;
pub mod cache_block;
pub mod cluster;
pub mod core;
pub mod dram;
pub mod instruction;
pub mod interconn;
pub mod l1;
pub mod l2;
pub mod ldst_unit;
pub mod mem_fetch;
pub mod mem_sub_partition;
pub mod mshr;
pub mod opcodes;
pub mod operand_collector;
pub mod register_set;
pub mod scheduler;
pub mod scoreboard;
pub mod set_index_function;
pub mod simd_function_unit;
pub mod sp_unit;
pub mod stats;
pub mod tag_array;
pub mod utils;

use self::cluster::*;
use self::core::*;
use addrdec::*;
use interconn as ic;
use ldst_unit::*;
use mem_fetch::*;
use mem_sub_partition::*;
use scheduler::*;
use set_index_function::*;
use sp_unit::*;
use stats::Stats;
use tag_array::*;
use utils::*;

use crate::config;
use bitvec::{array::BitArray, field::BitField, BitArr};
use color_eyre::eyre;
use console::style;
use itertools::Itertools;
use log::{error, info, trace, warn};
use nvbit_model::dim::{self, Dim};
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::fmt::Binary;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;
use trace_model::{Command, KernelLaunch, MemAccessTraceEntry};

pub type address = u64;

fn debug_break(msg: impl AsRef<str>) {
    let prompt = style(format!("{}: continue?", msg.as_ref()))
        .red()
        .to_string();
    // if !dialoguer::Confirm::new()
    //     .with_prompt(prompt)
    //     .wait_for_newline(true)
    //     .interact()
    //     .unwrap()
    // {
    //     panic!("debug stop: {}", msg.as_ref());
    // }
}

/// Context
#[derive(Debug)]
pub struct Context {}

impl Context {
    // gpgpu_ptx_sim_init_perf
    // GPGPUSim_Init
    // start_sim_thread
}

#[derive(Debug)]
struct FunctionInfo {
    // gpgpu_ctx = ctx;
    // m_uid = (gpgpu_ctx->function_info_sm_next_uid)++;
    // m_entry_point = (entry_point == 1) ? true : false;
    // m_extern = (entry_point == 2) ? true : false;
    // num_reconvergence_pairs = 0;
    // m_symtab = NULL;
    // m_assembled = false;
    // m_return_var_sym = NULL;
    // m_kernel_info.cmem = 0;
    // m_kernel_info.lmem = 0;
    // m_kernel_info.regs = 0;
    // m_kernel_info.smem = 0;
    // m_local_mem_framesize = 0;
    // m_args_aligned_size = -1;
    // pdom_done = false;  // initialize it to false
}

/// KernelInfo represents a kernel.
///
/// This includes its launch configuration,
/// as well as its state of execution.
///
/// TODO: rename to just kernel if this handles all the state.
#[derive()]
pub struct KernelInfo {
    // dim3 gridDim, dim3 blockDim,
    // trace_function_info *m_function_info,
    // trace_parser *parser, class trace_config *config,
    // kernel_trace_t *kernel_trace_info
    //
    // trace_config *m_tconfig;
    // const std::unordered_map<std::string, OpcodeChar> *OpcodeMap;
    pub opcodes: &'static opcodes::OpcodeMap,
    // trace_parser *m_parser;
    // kernel_trace_t *m_kernel_trace_info;
    // bool m_was_launched;
    pub config: KernelLaunch,
    pub uid: usize,
    // function_info: FunctionInfo,
    // shared_mem: bool,
    trace: Vec<MemAccessTraceEntry>,
    trace_iter: RwLock<std::vec::IntoIter<MemAccessTraceEntry>>,
    launched: Mutex<bool>,
    function_info: FunctionInfo,
    num_cores_running: usize,
    // m_kernel_entry = entry;
    // m_grid_dim = gridDim;
    // m_block_dim = blockDim;

    // next_block: Option<Dim>,
    // next_block_iter: RwLock<std::iter::Peekable<dim::Iter>>,
    next_block_iter: Mutex<std::iter::Peekable<dim::Iter>>,
    next_thread_iter: Mutex<std::iter::Peekable<dim::Iter>>,
    // next_thread_id: Option<Dim>,
    // next_thread_id_iter: dim::Iter,

    // m_next_cta.x = 0;
    // m_next_cta.y = 0;
    // m_next_cta.z = 0;
    // m_next_tid = m_next_cta;
    // m_num_cores_running = 0;
    // m_param_mem = new memory_space_impl<8192>("param", 64 * 1024);
    //
    // // Jin: parent and child kernel management for CDP
    // m_parent_kernel = NULL;
    //
    // // Jin: launch latency management
    // m_launch_latency = entry->gpgpu_ctx->device_runtime->g_kernel_launch_latency;
    //
    // m_kernel_TB_latency =
    //     entry->gpgpu_ctx->device_runtime->g_kernel_launch_latency +
    //     num_blocks() * entry->gpgpu_ctx->device_runtime->g_TB_launch_latency;
    //
    pub cache_config_set: bool,
}

impl PartialEq for KernelInfo {
    fn eq(&self, other: &Self) -> bool {
        self.uid == other.uid
    }
}

impl std::fmt::Debug for KernelInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelInfo")
            .field("name", &self.name())
            .field("id", &self.uid)
            .field("instructions", &self.trace.len())
            .field("launched", &self.launched)
            .field("grid", &self.config.grid)
            .field("block", &self.config.block)
            .field("stream", &self.config.stream_id)
            .field("shared_mem", &self.config.shared_mem_bytes)
            .field("registers", &self.config.num_registers)
            .field("block", &self.current_block())
            .field("thread", &self.next_block_iter.lock().unwrap().peek())
            .finish()
    }
}

impl std::fmt::Display for KernelInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelInfo")
            .field("name", &self.name())
            .field("id", &self.uid)
            .finish()
    }
}

pub fn read_trace(path: &Path) -> eyre::Result<Vec<MemAccessTraceEntry>> {
    use serde::Deserializer;
    let file = std::fs::OpenOptions::new().read(true).open(path)?;
    let reader = std::io::BufReader::new(file);
    let mut reader = rmp_serde::Deserializer::new(reader);
    let mut trace = vec![];
    let decoder = nvbit_io::Decoder::new(|access: MemAccessTraceEntry| {
        trace.push(access);
    });
    reader.deserialize_seq(decoder)?;
    Ok(trace)
}

impl KernelInfo {
    // gpgpu_ptx_sim_init_perf
    // GPGPUSim_Init
    // start_sim_thread
    pub fn from_trace(traces_dir: impl AsRef<Path>, config: KernelLaunch) -> Self {
        dbg!(&config);
        let trace_path = traces_dir
            .as_ref()
            .join(&config.trace_file)
            .with_extension("msgpack");
        dbg!(&trace_path);
        let mut trace = read_trace(&trace_path).unwrap();
        // trace.sort_unstable_by(|a, b| (a.block_id, a.warp_id).cmp(&(b.block_id, b.warp_id)));
        trace.sort_unstable_by(|a, b| {
            (a.block_id, a.warp_id_in_block, a.instr_offset).cmp(&(
                b.block_id,
                b.warp_id_in_block,
                b.instr_offset,
            ))
        });
        // dbg!(&trace
        //     .iter()
        //     .map(|i| (
        //         &i.block_id,
        //         &i.warp_id_in_block,
        //         &i.instr_offset,
        //         &i.instr_opcode
        //     ))
        //     .collect::<Vec<_>>());

        let mut trace_iter = trace.clone().into_iter();

        let next_block_iter = Mutex::new(config.grid.into_iter().peekable());
        let next_thread_iter = Mutex::new(config.block.into_iter().peekable());
        // let next_block = next_block_iter.next();
        // let next_thread_id = next_block;
        // dbg!(&next_block);
        // let uid = next_kernel_uid;
        let uid = 0; // todo
        let opcodes = opcodes::get_opcode_map(&config).unwrap();

        Self {
            config,
            uid,
            trace,
            trace_iter: RwLock::new(trace_iter),
            opcodes,
            launched: Mutex::new(false),
            num_cores_running: 0,
            function_info: FunctionInfo {},
            cache_config_set: false,
            next_block_iter,
            next_thread_iter,
            // next_block,
            // next_thread_id,
        }
    }

    // pub fn next_threadblock_traces(&self) -> Vec<MemAccessTraceEntry> {
    // pub fn next_threadblock_traces(&self, warps: &mut [Option<SchedulerWarp>]) {
    // pub fn next_threadblock_traces(&self, kernel: &KernelInfo, warps: &mut [SchedulerWarp]) {
    pub fn next_threadblock_traces(&self, warps: &mut [scheduler::CoreWarp]) {
        // pub fn next_threadblock_traces(&self, warps: &mut [Option<SchedulerWarp>]) {
        // debug_assert!(self.next_block.is_some());
        // todo!("next_threadblock_traces");
        // debug_assert!(self.next_block_iter.peek().is_some());
        // // let Some(next_block) = self.next_block else {
        let Some(next_block) = self.next_block_iter.lock().unwrap().next() else {

            println!("blocks done: no more threadblock traces");
            return;
        };
        // let next_block = nvbit_model::Dim { x: 0, y: 0, z: 0 };
        // dbg!(&next_block);
        // for warp in warps.iter_mut() {
        //     // warp.clear();
        //     *warp = None;
        // }
        let mut lock = self.trace_iter.write().unwrap();
        let trace_iter = lock.take_while_ref(|entry| entry.block_id == next_block);
        for trace in trace_iter {
            // dbg!(&trace.warp_id_in_block);
            // dbg!(&trace.instr_offset);
            let warp_id = trace.warp_id_in_block as usize;
            let instr = instruction::WarpInstruction::from_trace(&self, trace);
            // warps[warp_id] = Some(SchedulerWarp::default());
            let warp = warps.get_mut(warp_id).unwrap();
            let mut warp = warp.try_borrow_mut().unwrap();
            // .as_mut().unwrap();
            // warp.trace_instructions.push_back(instr);
            warp.push_trace_instruction(instr);
        }

        // set the pc from the traces and ignore the functional model
        // NOTE: set next pc is not needed so the entire block goes away
        // for warp in warps.iter_mut() {
        //     // if let Some(warp) = warp {
        //     let mut warp = warp.lock().unwrap();
        //     let num_instr = warp.instruction_count();
        //     if num_instr > 0 {
        //         println!("warp {}: {num_instr} instructions", warp.warp_id);
        //     }
        //     // for schedwarps without any instructions, there is no pc
        //     // before, we used option<schedwarp> which was in a way cleaner..
        //     if let Some(start_pc) = warp.trace_start_pc() {
        //         warp.set_next_pc(start_pc);
        //     }
        //     // }
        // }
        // println!("added {total} instructions");
        // panic!("threadblock traces: {:#?}", warp.push_trace_instruciton);

        // temp: add exit instructions to traces if not already
        // for warp in warps.iter_mut() {
        //     let mut warp = warp.lock().unwrap();
        //     match warp.trace_instructions.back() {
        //         Some(instruction::WarpInstruction { opcode, .. }) => {
        //             if opcode.category != opcodes::ArchOp::EXIT_OPS {
        //                 // add exit to back
        //             }
        //         }
        //         None => {
        //             // add exit to back
        //         }
        //     }
        // }
    }

    pub fn inc_running(&mut self) {
        self.num_cores_running += 1;
    }

    pub fn name(&self) -> &str {
        &self.config.name
    }

    pub fn was_launched(&self) -> bool {
        *self.launched.lock().unwrap()
    }

    pub fn running(&self) -> bool {
        self.num_cores_running > 0
    }

    // pub fn increment_block(&mut self) {
    //     self.next_block = self.next_block_iter.next()
    // }

    // pub fn increment_thread_id(&mut self) {
    //     // self.next_thread_id = self.next_thread_id_iter.next()
    // }

    pub fn current_block(&self) -> Option<nvbit_model::Point> {
        self.next_block_iter.lock().unwrap().peek().copied()
    }

    pub fn current_thread(&self) -> Option<nvbit_model::Point> {
        self.next_thread_iter.lock().unwrap().peek().copied()
    }

    // pub fn block_id(&self) -> u64 {
    //     // todo: make this nicer
    //     // self.next_block_iter.peek().unwrap().size()
    //     // todo!("block_id");
    //
    //     // self.next_block_iter.peek().id() as usize
    //     let mut iter = self.next_block_iter.lock().unwrap();
    //     // iter.by_ref().by_ref().id()
    //     0
    //     // .peek()
    //     // .map(|b| b.size())
    // }
    // pub fn next_block_id(&self) -> Option<usize> {
    // pub fn next_block_id(&self) -> Option<usize> {
    //     self.next_block_iter.peek().id() as usize
    // }

    pub fn done(&self) -> bool {
        self.no_more_blocks_to_run() && !self.running()
    }

    pub fn num_blocks(&self) -> usize {
        let grid = self.config.grid;
        grid.x as usize * grid.y as usize * grid.z as usize
    }

    // pub fn padded_threads_per_block(&self) -> usize {
    //     pad_to_multiple(self.threads_per_block(), self.config.warp_size)
    // }

    pub fn threads_per_block(&self) -> usize {
        let block = self.config.block;
        block.x as usize * block.y as usize * block.z as usize
    }

    pub fn no_more_blocks_to_run(&self) -> bool {
        // dbg!(&self.current_block());
        // todo!("KernelInfo: no_more_blocks_to_run");
        self.current_block().is_none()
        // self.next_block_iter.lock().unwrap().peek().is_none()
        // self.next_block.is_none()
        //     let next_block = self.next_block;
        // let grid = self.config.grid;
        // next_block.x >= grid.x || next_block.y >= grid.y || next_block.z >= grid.z
    }

    pub fn more_threads_in_block(&self) -> bool {
        self.current_thread().is_some()
        // lock().unwrap().peek().is_some()
        // todo!("more_threads_in_block");
        // self.next_thread_id.is_some()
        // return m_next_tid.z < m_block_dim.z && m_next_tid.y < m_block_dim.y &&
        //        m_next_tid.x < m_block_dim.x;
    }
}

pub fn parse_commands(path: impl AsRef<Path>) -> eyre::Result<Vec<Command>> {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .open(&path.as_ref())?;
    let reader = std::io::BufReader::new(file);
    let commands = serde_json::from_reader(reader)?;
    Ok(commands)
}

#[derive(Default)]
// pub struct MockSimulator<'a> {
pub struct MockSimulator<I> {
    stats: Arc<Mutex<Stats>>,
    config: Arc<config::GPUConfig>,
    // mem_partition_units: Vec<MemoryPartitionUnit<I>>,
    // mem_partition_units: Vec<MemoryPartitionUnit<ic::L2Interface<Packet>>>,
    mem_partition_units: Vec<MemoryPartitionUnit>,
    // Vec<MemoryPartitionUnit<ic::L2Interface<I, FifoQueue<mem_fetch::MemFetch>>>>,
    // mem_sub_partitions: Vec<Rc<RefCell<MemorySubPartition<ic::L2Interface<Packet>>>>>,
    mem_sub_partitions: Vec<Rc<RefCell<MemorySubPartition<FifoQueue<mem_fetch::MemFetch>>>>>,
    // mem_sub_partitions: Vec<Rc<RefCell<MemorySubPartition<I>>>>,
    running_kernels: Vec<Option<Arc<KernelInfo>>>,
    executed_kernels: Mutex<HashMap<usize, String>>,
    // clusters: Vec<SIMTCoreCluster>,
    // clusters: Vec<SIMTCoreCluster<'a>>,
    clusters: Vec<SIMTCoreCluster<I>>,
    interconn: Arc<I>,

    last_cluster_issue: usize,
    last_issued_kernel: usize,

    cycle: Cycle,
    // gpu_stall_icnt2sh: usize,
    // partition_replies_in_parallel: usize,
}

#[derive(Debug, Default)]
pub struct AtomicCycle(std::sync::atomic::AtomicU64);

impl AtomicCycle {
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

// impl MockSimulator {
// impl<'a> MockSimulator<'a> {
impl<I> MockSimulator<I>
where
    // I: ic::MemFetchInterface + 'static,
    I: ic::Interconnect<core::Packet> + 'static,
{
    // see new trace_gpgpu_sim
    pub fn new(interconn: Arc<I>, config: Arc<config::GPUConfig>) -> Self {
        let stats = Arc::new(Mutex::new(Stats::default()));

        let num_mem_units = config.num_mem_units;
        let num_sub_partitions = config.num_sub_partition_per_memory_channel;

        let mem_partition_units: Vec<_> = (0..num_mem_units)
            .map(|i| {
                MemoryPartitionUnit::new(
                    i,
                    // l2_port.clone(),
                    config.clone(),
                    stats.clone(),
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

        let cycle = Rc::new(AtomicCycle::new(0));
        let clusters: Vec<_> = (0..config.num_simt_clusters)
            .map(|i| {
                SIMTCoreCluster::new(
                    i,
                    Rc::clone(&cycle),
                    interconn.clone(),
                    stats.clone(),
                    config.clone(),
                )
            })
            .collect();

        let executed_kernels = Mutex::new(HashMap::new());

        Self {
            config,
            stats,
            mem_partition_units,
            mem_sub_partitions,
            interconn,
            running_kernels,
            executed_kernels,
            clusters,
            last_cluster_issue: 0,
            last_issued_kernel: 0,
            cycle,
            // gpu_stall_icnt2sh: 0,
            // partition_replies_in_parallel: 0,
        }
    }

    /// Select the next kernel to run
    ///
    /// Todo: used hack to allow selecting the kernel from the shader core,
    /// but we could maybe refactor
    pub fn select_kernel(&self) -> Option<&Arc<KernelInfo>> {
        let mut executed_kernels = self.executed_kernels.lock().unwrap();
        dbg!(&self.running_kernels.iter().filter(|k| k.is_some()).count());
        if let Some(k) = &self.running_kernels[self.last_issued_kernel] {
            dbg!(&k);
            if !k.no_more_blocks_to_run()
            // &&!kernel.kernel_TB_latency)
            {
                let launch_uid = k.uid;
                if !executed_kernels.contains_key(&launch_uid) {
                    executed_kernels.insert(launch_uid, k.name().to_string());
                }
                return Some(k);
            }
        }
        let num_kernels = self.running_kernels.len();
        let max_concurrent = self.config.max_concurrent_kernels;
        for n in 0..num_kernels {
            let idx = (n + self.last_issued_kernel + 1) % max_concurrent;
            if let Some(k) = &self.running_kernels[idx] {
                if !k.no_more_blocks_to_run()
                // &&!kernel.kernel_TB_latency)
                {
                    let launch_uid = k.uid;
                    assert!(!executed_kernels.contains_key(&launch_uid));
                    executed_kernels.insert(launch_uid, k.name().to_string());
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
        // println!("cluster done");
        for unit in &self.mem_partition_units {
            if unit.busy() {
                return true;
            }
        }
        // println!("mem done");
        if self.interconn.busy() {
            return true;
        }
        // println!("icnt done");
        if self.more_blocks_to_run() {
            return true;
        }
        // println!("no more blocks");
        // println!("done");
        false
    }

    pub fn can_start_kernel(&self) -> bool {
        self.running_kernels.iter().any(|kernel| match kernel {
            Some(kernel) => kernel.done(),
            None => true,
        })
    }

    pub fn launch(&mut self, kernel: Arc<KernelInfo>) -> eyre::Result<()> {
        *kernel.launched.lock().unwrap() = true;
        let threads_per_block = kernel.threads_per_block();
        let max_threads_per_block = self.config.max_threads_per_core;
        if threads_per_block > max_threads_per_block {
            error!("kernel block size is too large");
            error!(
                "CTA size (x*y*z) = {threads_per_block}, max supported = {max_threads_per_block}"
            );
            return Err(eyre::eyre!("kernel block size is too large"));
        }
        for running in &mut self.running_kernels {
            if running.is_none() || running.as_ref().map_or(false, |k| k.done()) {
                running.insert(kernel);
                break;
            }
        }
        Ok(())
    }

    fn issue_block_to_core(&mut self) {
        println!("issue block 2 core");
        let last_issued = self.last_cluster_issue;
        let num_clusters = self.config.num_simt_clusters;
        for cluster in &self.clusters {
            let idx = (cluster.cluster_id + last_issued + 1) % num_clusters;
            let num_blocks_issued = cluster.issue_block_to_core(self);
            if num_blocks_issued > 0 {
                self.last_cluster_issue = idx;
                // self.total_blocks_launched += num_blocks_issued;
            }
        }

        // unsigned last_issued = m_last_cluster_issue;
        // for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
        //   unsigned idx = (i + last_issued + 1) % m_shader_config->n_simt_clusters;
        //   unsigned num = m_cluster[idx]->issue_block2core();
        //   if (num) {
        //     m_last_cluster_issue = idx;
        //     m_total_cta_launched += num;
        //   }
        // }
    }

    // fn interconn_transfer(&mut self) {
    //     // not modeling the flits in the interconnect for now
    //     // todo!("sim: interconn transfer");
    // }

    pub fn set_cycle(&self, cycle: u64) {
        self.cycle.set(cycle)
    }

    pub fn cycle(&mut self) {
        // int clock_mask = next_clock_domain();

        // shader core loading (pop from ICNT into core)
        for cluster in &mut self.clusters {
            cluster.interconn_cycle();
        }

        let mut partition_replies_in_parallel_per_cycle = 0;

        println!(
            "pop from {} memory sub partitions",
            self.mem_sub_partitions.len()
        );

        // pop from memory controller to interconnect
        for (i, mem_sub) in self.mem_sub_partitions.iter().enumerate() {
            let mut mem_sub = mem_sub.borrow_mut();
            println!(
                "checking sub partition[{}]: icnt to l2 queue={} l2 to icnt queue={} l2 to dram queue={} dram to l2 queue={}",
                i,
                mem_sub.interconn_to_l2_queue,
                mem_sub.l2_to_interconn_queue,
                mem_sub.l2_to_dram_queue.lock().unwrap(),
                mem_sub.dram_to_l2_queue,
            );

            if let Some(fetch) = mem_sub.top() {
                dbg!(&fetch.addr());
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
                    partition_replies_in_parallel_per_cycle += 1;
                } else {
                    // self.gpu_stall_icnt2sh += 1;
                }
            }
            // else {
            //     mem_sub.pop();
            // }
        }
        // self.partition_replies_in_parallel += partition_replies_in_parallel_per_cycle;

        // dram
        println!("cycle for {} drams", self.mem_partition_units.len());
        for (i, unit) in self.mem_partition_units.iter_mut().enumerate() {
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
        println!(
            "moving mem requests from interconn to {} mem partitions",
            self.mem_sub_partitions.len()
        );
        let mut parallel_mem_partition_reqs_per_cycle = 0;
        let mut stall_dram_full = 0;
        for (i, mem_sub) in self.mem_sub_partitions.iter_mut().enumerate() {
            let mut mem_sub = mem_sub.borrow_mut();
            // move memory request from interconnect into memory partition
            // (if not backed up)
            //
            // Note:This needs to be called in DRAM clock domain if there
            // is no L2 cache in the system In the worst case, we may need
            // to push SECTOR_CHUNCK_SIZE requests, so ensure you have enough
            // buffer for them
            if mem_sub.full(SECTOR_CHUNCK_SIZE) {
                stall_dram_full += 1;
            } else {
                let device = self.config.mem_id_to_device_id(i);
                if let Some(Packet::Fetch(fetch)) = self.interconn.pop(device) {
                    println!(
                        "got new fetch {} for mem sub partition {} ({})",
                        fetch, i, device
                    );

                    mem_sub.push(fetch);
                    parallel_mem_partition_reqs_per_cycle += 1;
                }
            }
            // we borrow all of sub here, which is a problem for the cyclic reference in l2
            // interface
            mem_sub.cache_cycle(0); // gpu_sim_cycle + gpu_tot_sim_cycle);
                                    // mem_sub.accumulate_L2cache_stats(m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX]);
        }

        //   partiton_reqs_in_parallel += partiton_reqs_in_parallel_per_cycle;
        // if (partiton_reqs_in_parallel_per_cycle > 0) {
        //   partiton_reqs_in_parallel_util += partiton_reqs_in_parallel_per_cycle;
        //   gpu_sim_cycle_parition_util++;
        // }

        // self.interconn_transfer();

        let mut active_sms = 0;
        for cluster in &mut self.clusters {
            if cluster.not_completed() > 0 {
                // || get_more_cta_left())
                cluster.cycle();
                active_sms += cluster.num_active_sms();
            }
        }

        self.issue_block_to_core();
        // self.decrement_kernel_latency();

        // Depending on configuration, invalidate the caches
        // once all of threads are completed.
        let mut all_threads_complete = true;
        if self.config.flush_l1_cache {
            for cluster in self.clusters.iter_mut() {
                if cluster.not_completed() == 0 {
                    cluster.cache_invalidate();
                } else {
                    all_threads_complete = false;
                }
            }
        }

        if self.config.flush_l2_cache {
            if !self.config.flush_l1_cache {
                for cluster in self.clusters.iter_mut() {
                    if cluster.not_completed() > 0 {
                        all_threads_complete = false;
                        break;
                    }
                }
            }

            if let Some(l2_config) = &self.config.data_cache_l2 {
                if all_threads_complete {
                    // && !l2_config.disabled() {
                    println!("flushed L2 caches...");
                    if l2_config.total_lines() > 0 {
                        let mut dlc = 0;
                        for (i, mem_sub) in self.mem_sub_partitions.iter_mut().enumerate() {
                            let mut mem_sub = mem_sub.borrow_mut();
                            mem_sub.flush_l2();
                            // debug_assert_eq!(dlc, 0);
                            // println!("dirty lines flushed from L2 {} is {}", i, dlc);
                        }
                    }
                }
            }
        }
    }

    pub fn memcopy_to_gpu(&mut self, addr: address, num_bytes: u64) {
        if self.config.fill_l2_on_memcopy {
            println!("memcopy: {num_bytes} bytes to {addr:?}");
            let n_partitions = self.config.num_sub_partition_per_memory_channel;
            let mut transfered = 0;
            while transfered < num_bytes {
                let write_addr = addr + transfered as u64;

                let mut mask: mem_fetch::MemAccessSectorMask = BitArray::ZERO;
                mask.store((write_addr % 128 / 32) as u8);

                let tlx_addr = self.config.address_mapping().tlx(addr);
                let part_id = tlx_addr.sub_partition / n_partitions as u64;
                let part = &self.mem_partition_units[part_id as usize];
                part.handle_memcpy_to_gpu(write_addr, tlx_addr.sub_partition as usize, mask);
                transfered += 32;
            }
        }
    }
}

pub fn accelmain(traces_dir: impl AsRef<Path>) -> eyre::Result<()> {
    info!("box version {}", 0);
    let traces_dir = traces_dir.as_ref();
    let start_time = Instant::now();

    // debugging config
    let mut config = config::GPUConfig::default();
    config.num_simt_clusters = 1;
    config.num_cores_per_simt_cluster = 1;
    config.num_schedulers_per_core = 1;
    let config = Arc::new(config);

    assert!(config.max_threads_per_core.rem_euclid(config.warp_size) == 0);
    let max_warps_per_shader = config.max_threads_per_core / config.warp_size;

    let window_size = if config.concurrent_kernel_sm {
        config.max_concurrent_kernels
    } else {
        1
    };
    assert!(window_size > 0);

    // std::vector<trace_command> commandlist = tracer.parse_commandlist_file();
    let command_traces_path = traces_dir.join("commands.json");
    let mut commands: Vec<Command> = parse_commands(&command_traces_path)?;

    // todo: make this a hashset?
    let mut busy_streams: VecDeque<u64> = VecDeque::new();
    let mut kernels: VecDeque<Arc<KernelInfo>> = VecDeque::new();
    kernels.reserve_exact(window_size);

    let interconn = Arc::new(ic::ToyInterconnect::new(
        config.num_simt_clusters,
        config.num_mem_units * config.num_sub_partition_per_memory_channel,
        // config.num_simt_clusters * config.num_cores_per_simt_cluster,
        // config.num_mem_units,
        Some(9), // found by printf debugging gpgusim
    ));
    let mut sim = MockSimulator::new(interconn, config.clone());
    // let mut sim = MockSimulator::<ic::CoreMemoryInterface>::new(config.clone());

    let mut i = 0;
    while i < commands.len() || !kernels.is_empty() {
        // take as many commands as possible until we have
        // collected as many kernels to fill the window_size
        // or processed every command.

        while kernels.len() < window_size && i < commands.len() {
            let cmd = &commands[i];
            match cmd {
                Command::MemcpyHtoD {
                    dest_device_addr,
                    num_bytes,
                } => sim.memcopy_to_gpu(*dest_device_addr, *num_bytes),
                Command::KernelLaunch(launch) => {
                    let kernel = KernelInfo::from_trace(traces_dir, launch.clone());
                    kernels.push_back(Arc::new(kernel));
                }
            }
            i += 1;
        }

        // Launch all kernels within window that are on a stream
        // that isn't already running
        for kernel in &mut kernels {
            let stream_busy = busy_streams
                .iter()
                .any(|stream_id| *stream_id == kernel.config.stream_id);
            if !stream_busy && sim.can_start_kernel() && !kernel.was_launched() {
                println!("launching kernel {:#?}", kernel.name());
                busy_streams.push_back(kernel.config.stream_id);
                sim.launch(kernel.clone());
            }
        }

        dbg!(&config.num_simt_clusters);
        dbg!(&config.num_cores_per_simt_cluster);
        dbg!(&config.concurrent_kernel_sm);
        dbg!(&config.num_mem_units);
        dbg!(&config.num_sub_partition_per_memory_channel);
        dbg!(&config.num_memory_chips_per_controller);
        // return Ok(());

        // drive kernels to completion
        // while sim.active() {
        let cycle_limit: Option<u64> = std::env::var("CYCLES")
            .ok()
            .as_deref()
            .map(str::parse)
            .map(Result::ok)
            .flatten();

        let mut cycle: u64 = 0;
        let mut done = false;
        while !done {
            if let Some(cycle_limit) = cycle_limit {
                if cycle >= cycle_limit {
                    // early exit
                    break;
                }
            }

            println!("\n======== cycle {cycle} ========\n");
            sim.set_cycle(cycle);
            sim.cycle();

            done = !sim.active();
            cycle += 1;
        }

        println!("exit after {cycle} cycles");
        break;
    }
    Ok(())
}
