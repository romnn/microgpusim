pub mod addrdec;
pub mod barrier;
pub mod cache;
pub mod cache_block;
pub mod cluster;
pub mod core;
pub mod dram;
pub mod fifo;
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
pub mod tag_array;
pub mod utils;

#[cfg(test)]
pub mod testing;

use self::cluster::*;
use self::core::*;
use addrdec::*;
use fifo::Queue;
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
use std::borrow::Borrow;
use std::cell::RefCell;
use std::collections::HashSet;
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
    trace_pos: RwLock<usize>,
    // trace_iter: RwLock<Peekable<std::vec::IntoIter<MemAccessTraceEntry>>>,
    // trace_iter: RwLock<std::vec::IntoIter<MemAccessTraceEntry>>,
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
            .field(
                "shared_mem",
                &human_bytes::human_bytes(self.config.shared_mem_bytes as f64),
            )
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
    pub fn from_trace(traces_dir: impl AsRef<Path>, config: KernelLaunch) -> Self {
        println!(
            "parsing kernel for launch {:?} from {}",
            &config, &config.trace_file
        );
        let trace_path = traces_dir
            .as_ref()
            .join(&config.trace_file)
            .with_extension("msgpack");

        let mut trace = read_trace(&trace_path).unwrap();
        // trace.sort_unstable_by(|a, b| (a.block_id, a.warp_id).cmp(&(b.block_id, b.warp_id)));
        trace.sort_unstable_by(|a, b| {
            (a.block_id, a.warp_id_in_block, a.instr_offset).cmp(&(
                b.block_id,
                b.warp_id_in_block,
                b.instr_offset,
            ))
        });

        // check if grid size is equal to the number of unique blocks in the trace
        let all_blocks: HashSet<_> = trace.iter().map(|t| t.block_id).collect();
        println!(
            "kernel trace has {}/{} blocks",
            all_blocks.len(),
            config.grid.size()
        );
        assert_eq!(config.grid.size(), all_blocks.len() as u64);

        // dbg!(&trace
        //     .iter()
        //     .map(|i| (
        //         &i.block_id,
        //         &i.warp_id_in_block,
        //         &i.instr_offset,
        //         &i.instr_opcode
        //     ))
        //     .collect::<Vec<_>>());

        let mut trace_iter = trace.clone().into_iter().peekable();

        let next_block_iter = Mutex::new(config.grid.into_iter().peekable());
        let next_thread_iter = Mutex::new(config.block.into_iter().peekable());
        // let next_block = next_block_iter.next();
        // let next_thread_id = next_block;
        // let uid = next_kernel_uid;
        let uid = 0; // todo
        let opcodes = opcodes::get_opcode_map(&config).unwrap();

        Self {
            config,
            uid,
            trace,
            trace_pos: RwLock::new(0),
            // trace_iter: RwLock::new(trace_iter),
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
        println!("next thread block traces for block {}", next_block);

        // let next_block = nvbit_model::Dim { x: 0, y: 0, z: 0 };
        // for warp in warps.iter_mut() {
        //     // warp.clear();
        //     *warp = None;
        // }

        let mut trace_pos = self.trace_pos.write().unwrap();
        // let mut lock = self.trace_iter.write().unwrap();
        // let trace_iter = lock.take_while_ref(|entry| entry.block_id == next_block);

        let mut instructions = 0;
        let trace_size = self.trace.len();
        while *trace_pos < trace_size {
            let entry = &self.trace[*trace_pos];
            if entry.block_id > next_block.into() {
                // get instructions until new block
                break;
            }

            let warp_id = entry.warp_id_in_block as usize;
            let instr = instruction::WarpInstruction::from_trace(&self, entry.clone());
            let warp = warps.get_mut(warp_id).unwrap();
            let mut warp = warp.try_borrow_mut().unwrap();
            warp.push_trace_instruction(instr);

            instructions += 1;
            *trace_pos += 1;
        }

        // while
        // for trace in trace_iter {
        //     if trace.block_id > next_block
        //     let warp_id = trace.warp_id_in_block as usize;
        //     let instr = instruction::WarpInstruction::from_trace(&self, trace);
        //     // warps[warp_id] = Some(SchedulerWarp::default());
        //     let warp = warps.get_mut(warp_id).unwrap();
        //     let mut warp = warp.try_borrow_mut().unwrap();
        //     // .as_mut().unwrap();
        //     // warp.trace_instructions.push_back(instr);
        //     warp.push_trace_instruction(instr);
        //     instructions += 1;
        // }

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

        println!(
            "added {instructions} instructions ({} per thread) for block {next_block}",
            instructions / 32
        );
        debug_assert!(instructions > 0);
        // debug_assert!(instructions % 32 == 0);
        // dbg!(warps
        //     .iter()
        //     .map(|w| w.try_borrow().unwrap().trace_instructions.len())
        //     .collect::<Vec<_>>());
        // debug_assert!(
        //     warps
        //         .iter()
        //         .map(|w| w.try_borrow().unwrap().trace_instructions.len())
        //         .collect::<HashSet<_>>()
        //         .len()
        //         == 1,
        //     "all warps have the same number of instructions"
        // );
        debug_assert!(
            warps
                .iter()
                .all(|w| !w.try_borrow().unwrap().trace_instructions.is_empty()),
            "all warps have at least one instruction (need at least an EXIT)"
        );

        // println!("warps: {:#?}", warps);

        // for w in warps {
        //     let w = w.try_borrow().unwrap();
        //     if !w.done_exit() && w.trace_instructions.is_empty() {
        //         panic!("active warp {} has no instructions", w.warp_id);
        //     }
        // }

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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Allocation {
    id: usize,
    name: Option<String>,
    start_addr: address,
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
        if self.0.overlaps(&range) {
            panic!("overlapping memory allocation {:?}", &range);
        }
        let id = self.0.len() + 1; // zero is reserved for instructions
        let start_addr = range.start;
        self.0.insert(
            range,
            Allocation {
                name,
                // avoid joining of allocations using the id and range
                id,
                start_addr,
            },
        );
    }
}

#[derive()]
// pub struct MockSimulator<'a> {
pub struct MockSimulator<I> {
    stats: Arc<Mutex<Stats>>,
    config: Arc<config::GPUConfig>,
    // mem_partition_units: Vec<MemoryPartitionUnit<I>>,
    // mem_partition_units: Vec<MemoryPartitionUnit<ic::L2Interface<Packet>>>,
    mem_partition_units: Vec<MemoryPartitionUnit>,
    // Vec<MemoryPartitionUnit<ic::L2Interface<I, FifoQueue<mem_fetch::MemFetch>>>>,
    // mem_sub_partitions: Vec<Rc<RefCell<MemorySubPartition<ic::L2Interface<Packet>>>>>,
    mem_sub_partitions: Vec<Rc<RefCell<MemorySubPartition<fifo::FifoQueue<mem_fetch::MemFetch>>>>>,
    // mem_sub_partitions: Vec<Rc<RefCell<MemorySubPartition<I>>>>,
    running_kernels: Vec<Option<Arc<KernelInfo>>>,
    executed_kernels: Mutex<HashMap<usize, String>>,
    // clusters: Vec<SIMTCoreCluster>,
    // clusters: Vec<SIMTCoreCluster<'a>>,
    clusters: Vec<SIMTCoreCluster<I>>,
    interconn: Arc<I>,

    last_cluster_issue: usize,
    last_issued_kernel: usize,
    allocations: Rc<RefCell<Allocations>>,

    // for main run loop
    cycle: Cycle,
    traces_dir: PathBuf,
    commands: Vec<Command>,
    command_idx: usize,
    kernels: VecDeque<Arc<KernelInfo>>,
    kernel_window_size: usize,
    busy_streams: VecDeque<u64>,
    cycle_limit: Option<u64>,
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

pub trait FromConfig {
    fn from_config(config: &config::GPUConfig) -> Self;
}

impl FromConfig for stats::Stats {
    fn from_config(config: &config::GPUConfig) -> Self {
        let num_total_cores = config.total_cores();
        let num_mem_units = config.num_mem_units;
        let num_dram_banks = config.dram_timing_options.num_banks;

        Self::new(num_total_cores, num_mem_units, num_dram_banks)
    }
}

// impl MockSimulator {
// impl<'a> MockSimulator<'a> {
impl<I> MockSimulator<I>
where
    // I: ic::MemFetchInterface + 'static,
    I: ic::Interconnect<core::Packet> + 'static,
{
    // see new trace_gpgpu_sim
    pub fn new(
        interconn: Arc<I>,
        config: Arc<config::GPUConfig>,
        traces_dir: impl AsRef<Path>,
    ) -> Self {
        let traces_dir = traces_dir.as_ref();
        let stats = Arc::new(Mutex::new(Stats::from_config(&*config)));

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

        let allocations = Rc::new(RefCell::new(Allocations::default()));

        let cycle = Rc::new(AtomicCycle::new(0));
        let clusters: Vec<_> = (0..config.num_simt_clusters)
            .map(|i| {
                SIMTCoreCluster::new(
                    i,
                    Rc::clone(&cycle),
                    Rc::clone(&allocations),
                    Arc::clone(&interconn),
                    Arc::clone(&stats),
                    Arc::clone(&config),
                )
            })
            .collect();

        let executed_kernels = Mutex::new(HashMap::new());

        assert!(config.max_threads_per_core.rem_euclid(config.warp_size) == 0);
        let max_warps_per_shader = config.max_threads_per_core / config.warp_size;

        let window_size = if config.concurrent_kernel_sm {
            config.max_concurrent_kernels
        } else {
            1
        };
        assert!(window_size > 0);

        let command_traces_path = traces_dir.join("commands.json");
        let mut commands: Vec<Command> = parse_commands(&command_traces_path).unwrap();

        // todo: make this a hashset?
        let mut busy_streams: VecDeque<u64> = VecDeque::new();
        let mut kernels: VecDeque<Arc<KernelInfo>> = VecDeque::new();
        kernels.reserve_exact(window_size);

        let cycle_limit: Option<u64> = std::env::var("CYCLES")
            .ok()
            .as_deref()
            .map(str::parse)
            .map(Result::ok)
            .flatten();

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
            allocations,
            cycle,
            traces_dir: traces_dir.to_path_buf(),
            commands,
            command_idx: 0,
            kernels,
            kernel_window_size: window_size,
            busy_streams,
            cycle_limit,
        }
    }

    /// Select the next kernel to run
    ///
    /// Todo: used hack to allow selecting the kernel from the shader core,
    /// but we could maybe refactor
    pub fn select_kernel(&self) -> Option<&Arc<KernelInfo>> {
        let mut executed_kernels = self.executed_kernels.lock().unwrap();
        // dbg!(&self.running_kernels.iter().filter(|k| k.is_some()).count());
        if let Some(k) = &self.running_kernels[self.last_issued_kernel] {
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
        println!("issue block to core");
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
        let mut stats = self.stats.lock().unwrap();
        stats.sim.cycles = cycle;
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
            "POP from {} memory sub partitions",
            self.mem_sub_partitions.len()
        );

        // pop from memory controller to interconnect
        for (i, mem_sub) in self.mem_sub_partitions.iter().enumerate() {
            let mut mem_sub = mem_sub.borrow_mut();
            {
                println!("checking sub partition[{i}]:");
                println!(
                    "\t icnt to l2 queue ({:<3}) = {}",
                    mem_sub.interconn_to_l2_queue.len(),
                    mem_sub.interconn_to_l2_queue
                );
                println!(
                    "\t l2 to icnt queue ({:<3}) = {}",
                    mem_sub.l2_to_interconn_queue.len(),
                    mem_sub.l2_to_interconn_queue
                );
                let l2_to_dram_queue = mem_sub.l2_to_dram_queue.lock().unwrap();
                println!(
                    "\t l2 to dram queue ({:<3}) = {}",
                    l2_to_dram_queue.len(),
                    l2_to_dram_queue
                );
                println!(
                    "\t dram to l2 queue ({:<3}) = {}",
                    mem_sub.dram_to_l2_queue.len(),
                    mem_sub.dram_to_l2_queue
                );
                let partition = &self.mem_partition_units[mem_sub.partition_id];
                let dram_latency_queue: Vec<_> = partition
                    .dram_latency_queue
                    .iter()
                    .map(|f| f.to_string())
                    .collect();
                println!(
                    "\t dram latency queue ({:3}) = {:?}",
                    dram_latency_queue.len(),
                    style(&dram_latency_queue).red()
                );
                println!("");
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
            let device = self.config.mem_id_to_device_id(i);

            // same as full with parameter overload
            if mem_sub.interconn_to_l2_can_fit(SECTOR_CHUNCK_SIZE as usize) {
                if let Some(Packet::Fetch(fetch)) = self.interconn.pop(device) {
                    println!(
                        "got new fetch {} for mem sub partition {} ({})",
                        fetch, i, device
                    );

                    mem_sub.push(fetch);
                    parallel_mem_partition_reqs_per_cycle += 1;
                }
            } else {
                println!("SKIP sub partition {} ({}): DRAM full stall", i, device);
                stall_dram_full += 1;
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

    pub fn memcopy_to_gpu(&mut self, addr: address, num_bytes: u64, name: Option<String>) {
        println!(
            "memcopy: {:<20} {:>15} ({:>5} f32) to address {addr:>20}",
            name.as_deref().unwrap_or("<unnamed>"),
            human_bytes::human_bytes(num_bytes as f64),
            num_bytes / 4,
        );
        let alloc_range = addr..(addr + num_bytes);
        self.allocations
            .borrow_mut()
            .insert(alloc_range.clone(), name);
        if self.config.fill_l2_on_memcopy {
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
            let sub: &MemorySubPartition = &sub.as_ref().borrow();
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
    /// the window_size or processed every command.
    pub fn process_commands(&mut self) {
        while self.kernels.len() < self.kernel_window_size && self.command_idx < self.commands.len()
        {
            let cmd = &self.commands[self.command_idx];
            match cmd {
                Command::MemcpyHtoD {
                    allocation_name,
                    dest_device_addr,
                    num_bytes,
                } => self.memcopy_to_gpu(*dest_device_addr, *num_bytes, allocation_name.clone()),
                Command::KernelLaunch(launch) => {
                    let kernel = KernelInfo::from_trace(&self.traces_dir, launch.clone());
                    self.kernels.push_back(Arc::new(kernel));
                }
            }
            self.command_idx += 1;
        }
    }

    /// Lauch more kernels if possible.
    ///
    /// Launch all kernels within window that are on a stream that isn't already running
    pub fn lauch_kernels(&mut self) {
        let mut launch_queue: Vec<Arc<KernelInfo>> = Vec::new();
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
            println!("launching kernel {:#?}", kernel.name());
            self.launch(kernel);
        }
    }

    pub fn reached_limit(&self, cycle: u64) -> bool {
        match self.cycle_limit {
            Some(limit) if cycle >= limit => true,
            _ => false,
        }
    }

    pub fn commands_left(&self) -> bool {
        self.command_idx < self.commands.len()
    }

    pub fn kernels_left(&self) -> bool {
        !self.kernels.is_empty()
    }

    pub fn run_to_completion(&mut self, traces_dir: impl AsRef<Path>) -> eyre::Result<()> {
        let mut cycle: u64 = 0;
        while self.commands_left() || self.kernels_left() {
            self.process_commands();
            self.lauch_kernels();

            loop {
                println!("\n======== cycle {cycle} ========\n");

                if self.reached_limit(cycle) || !self.active() {
                    break;
                }

                self.cycle();

                cycle += 1;
                self.set_cycle(cycle);
            }

            // TODO:
            // self.cleanup_finished_kernel(finished_kernel_uid);

            println!("exit after {cycle} cycles");
            dbg!(self.commands_left());
            dbg!(self.kernels_left());

            // since we are not yet cleaning up launched kernels, we break here
            break;
        }
        Ok(())
    }
}

fn save_stats_to_file(stats: &Stats, out_file: &Path) -> eyre::Result<()> {
    use serde::Serialize;
    use std::fs;

    let out_file = out_file.with_extension("json");

    if let Some(parent) = &out_file.parent() {
        fs::create_dir_all(parent).ok();
    }
    let output_file = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(out_file)?;

    // write stats as json
    output_file
        .metadata()
        .unwrap()
        .permissions()
        .set_readonly(false);
    let mut writer = std::io::BufWriter::new(output_file);
    let mut json_serializer = serde_json::Serializer::with_formatter(
        writer,
        serde_json::ser::PrettyFormatter::with_indent(b"    "),
    );
    // TODO: how to serialize?
    // stats.serialize(&mut json_serializer)?;

    // let mut csv_writer = csv::WriterBuilder::new()
    //     .flexible(false)
    //     .from_writer(output_file);
    //
    // csv_writer.write_record(["kernel", "kernel_id", "stat", "value"])?;

    // sort stats before writing to csv
    // let mut sorted_stats: Vec<_> = stats.iter().collect();
    // sorted_stats.sort_by(|a, b| a.0.cmp(b.0));
    //
    // for ((kernel, kcount, stat), value) in &sorted_stats {
    //     csv_writer.write_record([kernel, &kcount.to_string(), stat, &value.to_string()])?;
    // }
    Ok(())
}

pub struct Box {}

pub fn accelmain(
    traces_dir: impl AsRef<Path>,
    stats_out_file: Option<&PathBuf>,
) -> eyre::Result<Stats> {
    info!("box version {}", 0);
    let traces_dir = traces_dir.as_ref();
    let start_time = Instant::now();

    // debugging config
    let mut config = config::GPUConfig::default();
    config.num_simt_clusters = 1;
    config.num_cores_per_simt_cluster = 1;
    config.num_schedulers_per_core = 1;
    let config = Arc::new(config);

    let interconn = Arc::new(ic::ToyInterconnect::new(
        config.num_simt_clusters,
        config.num_mem_units * config.num_sub_partition_per_memory_channel,
        // config.num_simt_clusters * config.num_cores_per_simt_cluster,
        // config.num_mem_units,
        Some(9), // found by printf debugging gpgusim
    ));
    let mut sim = MockSimulator::new(interconn, config.clone(), traces_dir);

    sim.run_to_completion(&traces_dir);

    let stats = sim.stats();
    eprintln!("STATS:\n");
    eprintln!("DRAM: total reads: {}", &stats.dram.total_reads());
    eprintln!("DRAM: total writes: {}", &stats.dram.total_writes());
    eprintln!("SIM: {:#?}", &stats.sim);
    eprintln!("INSTRUCTIONS: {:#?}", &stats.instructions);
    eprintln!("ACCESSES: {:#?}", &stats.accesses);
    eprintln!("L1I: {:#?}", &stats.l1i_stats.reduce());
    eprintln!("L1D: {:#?}", &stats.l1d_stats.reduce());
    eprintln!("L2D: {:#?}", &stats.l2d_stats.reduce());

    // save stats to file
    // let stats_file_path = stats_out_file
    //     .as_ref()
    //     .cloned()
    //     .unwrap_or(traces_dir.join("box.stats.txt"));
    // save_stats_to_file(&stats, &stats_file_path)?;

    Ok(stats)
}

#[cfg(test)]
mod tests {
    use crate::{
        config,
        ported::{fifo::Queue, interconn as ic, testing},
        Simulation,
    };
    use color_eyre::eyre;
    use pretty_assertions_sorted as diff;
    use stats::ConvertHashMap;
    use std::collections::HashMap;
    use std::io::Write;
    use std::ops::Deref;
    use std::path::PathBuf;
    use std::sync::Arc;

    // impl From<playground::mem_fetch::mf_type> for super::mem_fetch::Kind {
    //     fn from(kind: playground::mem_fetch::mf_type) -> Self {
    //         use playground::mem_fetch::mf_type;
    //         match kind {
    //             mf_type::READ_REQUEST => super::mem_fetch::Kind::READ_REQUEST,
    //             mf_type::WRITE_REQUEST => super::mem_fetch::Kind::WRITE_REQUEST,
    //             mf_type::READ_REPLY => super::mem_fetch::Kind::READ_REPLY,
    //             mf_type::WRITE_ACK => super::mem_fetch::Kind::WRITE_ACK,
    //         }
    //     }
    // }
    //
    // impl From<playground::mem_fetch::mem_access_type> for super::mem_fetch::AccessKind {
    //     fn from(kind: playground::mem_fetch::mem_access_type) -> Self {
    //         use playground::mem_fetch::mem_access_type;
    //         match kind {
    //             mem_access_type::GLOBAL_ACC_R => super::mem_fetch::AccessKind::GLOBAL_ACC_R,
    //             mem_access_type::LOCAL_ACC_R => super::mem_fetch::AccessKind::LOCAL_ACC_R,
    //             mem_access_type::CONST_ACC_R => super::mem_fetch::AccessKind::CONST_ACC_R,
    //             mem_access_type::TEXTURE_ACC_R => super::mem_fetch::AccessKind::TEXTURE_ACC_R,
    //             mem_access_type::GLOBAL_ACC_W => super::mem_fetch::AccessKind::GLOBAL_ACC_W,
    //             mem_access_type::LOCAL_ACC_W => super::mem_fetch::AccessKind::LOCAL_ACC_W,
    //             mem_access_type::L1_WRBK_ACC => super::mem_fetch::AccessKind::L1_WRBK_ACC,
    //             mem_access_type::L2_WRBK_ACC => super::mem_fetch::AccessKind::L2_WRBK_ACC,
    //             mem_access_type::INST_ACC_R => super::mem_fetch::AccessKind::INST_ACC_R,
    //             mem_access_type::L1_WR_ALLOC_R => super::mem_fetch::AccessKind::L1_WR_ALLOC_R,
    //             mem_access_type::L2_WR_ALLOC_R => super::mem_fetch::AccessKind::L2_WR_ALLOC_R,
    //             other @ mem_access_type::NUM_MEM_ACCESS_TYPE => {
    //                 panic!("bad mem access kind: {:?}", other)
    //             }
    //         }
    //     }
    // }
    //
    // #[derive(Clone, PartialEq, Eq, Hash)]
    // pub struct WarpInstructionKey {
    //     opcode: String,
    //     pc: usize,
    //     warp_id: usize,
    // }
    //
    // impl std::fmt::Debug for WarpInstructionKey {
    //     fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    //         write!(f, "{}[pc={},warp={}]", self.opcode, self.pc, self.warp_id)
    //     }
    // }
    //
    // #[derive(Clone, PartialEq, Eq, Hash)]
    // pub struct RegisterSetKey {
    //     stage: super::core::PipelineStage,
    //     pipeline: Vec<Option<WarpInstructionKey>>,
    // }
    //
    // impl std::fmt::Debug for RegisterSetKey {
    //     fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    //         write!(f, "{:?}={:?}", self.stage, self.pipeline)
    //     }
    // }
    //
    // impl From<super::register_set::RegisterSet> for RegisterSetKey {
    //     fn from(reg: super::register_set::RegisterSet) -> Self {
    //         let pipeline = reg
    //             .regs
    //             .into_iter()
    //             .map(|instr| match instr {
    //                 Some(instr) => Some(WarpInstructionKey {
    //                     opcode: instr.opcode.to_string(),
    //                     pc: instr.pc,
    //                     warp_id: instr.warp_id,
    //                 }),
    //                 None => None,
    //             })
    //             .collect();
    //         Self {
    //             stage: reg.stage,
    //             pipeline,
    //         }
    //     }
    // }
    //
    // impl<'a> From<playground::main::pipeline_stage_name_t> for super::core::PipelineStage {
    //     fn from(stage: playground::main::pipeline_stage_name_t) -> Self {
    //         use playground::main::pipeline_stage_name_t;
    //         match stage {
    //             // pipeline_stage_name_t::ID_OC_SP => Self::ID_OC_SP,
    //             // pipeline_stage_name_t::ID_OC_DP => Self::ID_OC_DP,
    //             // pipeline_stage_name_t::ID_OC_INT => Self::ID_OC_INT,
    //             // pipeline_stage_name_t::ID_OC_SFU => Self::ID_OC_SFU,
    //             // pipeline_stage_name_t::ID_OC_MEM => Self::ID_OC_MEM,
    //             pipeline_stage_name_t::OC_EX_SP => Self::OC_EX_SP,
    //             // pipeline_stage_name_t::OC_EX_DP => Self::OC_EX_DP,
    //             // pipeline_stage_name_t::OC_EX_INT => Self::OC_EX_INT,
    //             // pipeline_stage_name_t::OC_EX_SFU => Self::OC_EX_SFU,
    //             pipeline_stage_name_t::OC_EX_MEM => Self::OC_EX_MEM,
    //             pipeline_stage_name_t::EX_WB => Self::EX_WB,
    //             // pipeline_stage_name_t::ID_OC_TENSOR_CORE => Self::ID_OC_TENSOR_CORE,
    //             // pipeline_stage_name_t::OC_EX_TENSOR_CORE => Self::OC_EX_TENSOR_CORE,
    //             other => panic!("bad pipeline stage {:?}", other),
    //         }
    //     }
    // }
    //
    // impl<'a> From<playground::RegisterSet<'a>> for RegisterSetKey {
    //     fn from(reg: playground::RegisterSet<'a>) -> Self {
    //         Self {
    //             stage: reg.stage.into(),
    //             pipeline: reg
    //                 .pipeline
    //                 .into_iter()
    //                 .map(|instr| {
    //                     if instr.empty() {
    //                         None
    //                     } else {
    //                         let opcode = unsafe { std::ffi::CStr::from_ptr(instr.opcode_str()) };
    //                         let opcode = opcode
    //                             .to_str()
    //                             .unwrap()
    //                             .trim_start_matches("OP_")
    //                             .to_string();
    //                         Some(WarpInstructionKey {
    //                             opcode,
    //                             pc: instr.get_pc() as usize,
    //                             warp_id: instr.warp_id() as usize,
    //                         })
    //                     }
    //                 })
    //                 .collect(),
    //         }
    //     }
    // }
    //
    // #[derive(Clone, PartialEq, Eq, Hash)]
    // pub struct MemFetchKey {
    //     kind: super::mem_fetch::Kind,
    //     access_kind: super::mem_fetch::AccessKind,
    //     // cannot compare addr because its different between runs
    //     // addr: super::address,
    //     relative_addr: Option<(usize, super::address)>,
    // }
    //
    // impl std::fmt::Debug for MemFetchKey {
    //     fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    //         write!(f, "{:?}({:?}", self.kind, self.access_kind)?;
    //         if let Some((alloc_id, rel_addr)) = self.relative_addr {
    //             write!(f, "@{}+{}", alloc_id, rel_addr)?;
    //         }
    //         write!(f, ")")
    //     }
    // }
    //
    // impl<'a> From<playground::MemFetch<'a>> for MemFetchKey {
    //     fn from(fetch: playground::MemFetch<'a>) -> Self {
    //         let addr = fetch.get_addr();
    //         let relative_addr = fetch.get_relative_addr();
    //         Self {
    //             kind: fetch.get_type().into(),
    //             access_kind: fetch.get_access_type().into(),
    //             relative_addr: if addr == relative_addr {
    //                 None
    //             } else {
    //                 Some((fetch.get_alloc_id() as usize, relative_addr))
    //             },
    //         }
    //     }
    // }
    //
    // impl From<super::mem_fetch::MemFetch> for MemFetchKey {
    //     fn from(fetch: super::mem_fetch::MemFetch) -> Self {
    //         let addr = fetch.addr();
    //         Self {
    //             kind: fetch.kind,
    //             access_kind: *fetch.access_kind(),
    //             relative_addr: match fetch.access.allocation {
    //                 Some(alloc) => Some((alloc.id, addr - alloc.start_addr)),
    //                 None => None,
    //             },
    //         }
    //     }
    // }
    //
    // #[derive(Debug, Clone, PartialEq, Eq)]
    // pub struct SimulationState {
    //     interconn_to_l2_queue: Vec<Vec<MemFetchKey>>,
    //     l2_to_interconn_queue: Vec<Vec<MemFetchKey>>,
    //     l2_to_dram_queue: Vec<Vec<MemFetchKey>>,
    //     dram_to_l2_queue: Vec<Vec<MemFetchKey>>,
    //     dram_latency_queue: Vec<Vec<MemFetchKey>>,
    //     functional_unit_pipelines: Vec<Vec<RegisterSetKey>>,
    // }
    //
    // impl SimulationState {
    //     pub fn new(
    //         total_cores: usize,
    //         num_mem_partitions: usize,
    //         num_sub_partitions: usize,
    //     ) -> Self {
    //         Self {
    //             // per sub partition
    //             interconn_to_l2_queue: vec![vec![]; num_sub_partitions],
    //             l2_to_interconn_queue: vec![vec![]; num_sub_partitions],
    //             l2_to_dram_queue: vec![vec![]; num_sub_partitions],
    //             dram_to_l2_queue: vec![vec![]; num_sub_partitions],
    //             // per partition
    //             dram_latency_queue: vec![vec![]; num_mem_partitions],
    //             // per core
    //             functional_unit_pipelines: vec![vec![]; total_cores],
    //         }
    //     }
    // }

    #[test]
    fn test_vectoradd() -> eyre::Result<()> {
        let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
        let vec_add_trace_dir = manifest_dir.join("results/vectorAdd/vectorAdd-100-32");
        let vec_add_trace_dir = manifest_dir.join("results/vectorAdd/vectorAdd-1000-32");
        let vec_add_trace_dir = manifest_dir.join("results/vectorAdd/vectorAdd-10000-32");

        let kernelslist = vec_add_trace_dir.join("accelsim-trace/kernelslist.g");
        let gpgpusim_config = manifest_dir.join("accelsim/gtx1080/gpgpusim.config");
        let trace_config = manifest_dir.join("accelsim/gtx1080/gpgpusim.trace.config");
        let inter_config = manifest_dir.join("accelsim/gtx1080/config_fermi_islip.icnt");

        assert!(vec_add_trace_dir.is_dir());
        assert!(kernelslist.is_file());
        assert!(gpgpusim_config.is_file());
        assert!(trace_config.is_file());
        assert!(inter_config.is_file());

        // let start = std::time::Instant::now();
        // let box_stats = super::accelmain(&vec_add_trace_dir.join("trace"), None)?;

        // debugging config
        let mut box_config = config::GPUConfig::default();
        box_config.num_simt_clusters = 1;
        box_config.num_cores_per_simt_cluster = 1;
        box_config.num_schedulers_per_core = 1;
        let box_config = Arc::new(box_config);

        let box_interconn = Arc::new(ic::ToyInterconnect::new(
            box_config.num_simt_clusters,
            box_config.num_mem_units * box_config.num_sub_partition_per_memory_channel,
            // config.num_simt_clusters * config.num_cores_per_simt_cluster,
            // config.num_mem_units,
            Some(9), // found by printf debugging gpgusim
        ));

        let mut box_sim = super::MockSimulator::new(
            box_interconn,
            box_config.clone(),
            vec_add_trace_dir.join("trace"),
        );
        // let box_dur = start.elapsed();

        // let start = std::time::Instant::now();
        let mut args = vec![
            "-trace",
            kernelslist.as_os_str().to_str().unwrap(),
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

        // accelsim.run_to_completion();
        // let ref_stats = accelsim.stats().clone();
        // let ref_stats = playground::run(&config, &args)?;

        let mut cycle = 0;
        while play_sim.commands_left() || play_sim.kernels_left() {
            play_sim.process_commands();
            play_sim.launch_kernels();

            box_sim.process_commands();
            box_sim.lauch_kernels();

            let mut finished_kernel_uid: Option<u32> = None;
            loop {
                if !play_sim.active() {
                    break;
                }

                play_sim.cycle();
                cycle = play_sim.get_cycle();

                box_sim.cycle();
                box_sim.set_cycle(cycle);

                // todo: extract also l1i ready (least important)
                // todo: extract wb pipeline

                // iterate over sub partitions
                let total_cores = box_sim.config.total_cores();
                let num_partitions = box_sim.mem_partition_units.len();
                let num_sub_partitions = box_sim.mem_sub_partitions.len();
                let mut box_sim_state = testing::state::Simulation::new(
                    total_cores,
                    num_partitions,
                    num_sub_partitions,
                );

                for (cluster_id, cluster) in box_sim.clusters.iter().enumerate() {
                    for (core_id, core) in cluster.cores.lock().unwrap().iter().enumerate() {
                        let global_core_id =
                            cluster_id * box_sim.config.num_cores_per_simt_cluster + core_id;
                        assert_eq!(core.inner.core_id, global_core_id);

                        // this is the one we will use (unless the assertion is ever false)
                        let core_id = core.inner.core_id;

                        // core: functional units
                        for (fu_id, fu) in core.functional_units.iter().enumerate() {
                            let fu = fu.lock().unwrap();
                            let issue_port = core.issue_ports[fu_id];
                            let issue_reg: super::register_set::RegisterSet =
                                core.inner.pipeline_reg[issue_port as usize]
                                    .borrow()
                                    .clone();
                            assert_eq!(issue_port, issue_reg.stage);

                            box_sim_state.functional_unit_pipelines[core_id].push(issue_reg.into());
                        }
                        // core: operand collector
                        box_sim_state.operand_collectors[core_id]
                            .insert(core.inner.operand_collector.borrow().deref().into());
                        // core: schedulers
                        box_sim_state.schedulers[core_id]
                            .extend(core.schedulers.iter().map(Into::into));
                    }
                }

                for (partition_id, partition) in box_sim.mem_partition_units.iter().enumerate() {
                    box_sim_state.dram_latency_queue[partition_id].extend(
                        partition
                            .dram_latency_queue
                            .clone()
                            .into_iter()
                            .map(Into::into),
                    );
                }
                for (sub_id, sub) in box_sim.mem_sub_partitions.iter().enumerate() {
                    for (dest_queue, src_queue) in [
                        (
                            &mut box_sim_state.interconn_to_l2_queue[sub_id],
                            &sub.borrow().interconn_to_l2_queue,
                        ),
                        (
                            &mut box_sim_state.l2_to_interconn_queue[sub_id],
                            &sub.borrow().l2_to_interconn_queue,
                        ),
                        (
                            &mut box_sim_state.l2_to_dram_queue[sub_id],
                            &sub.borrow().l2_to_dram_queue.lock().unwrap(),
                        ),
                        (
                            &mut box_sim_state.dram_to_l2_queue[sub_id],
                            &sub.borrow().dram_to_l2_queue,
                        ),
                    ] {
                        dest_queue.extend(src_queue.clone().into_iter().map(Into::into));
                    }
                }

                let mut play_sim_state = testing::state::Simulation::new(
                    total_cores,
                    num_partitions,
                    num_sub_partitions,
                );
                for (core_id, core) in play_sim.cores().enumerate() {
                    for reg in core.register_sets().into_iter() {
                        play_sim_state.functional_unit_pipelines[core_id].push(reg.into());
                    }
                    // core: operand collector
                    let coll = core.operand_collector();
                    play_sim_state.operand_collectors[core_id].insert(coll.into());
                    // core: scheduler units
                    let schedulers = core.schedulers();
                    assert_eq!(schedulers.len(), box_sim_state.schedulers[core_id].len());
                    for (sched_idx, scheduler) in schedulers.into_iter().enumerate() {
                        // let scheduler = testing::state::Scheduler::from(scheduler);
                        play_sim_state.schedulers[core_id].push(scheduler.into());

                        let num_box_warps = box_sim_state.schedulers[core_id][sched_idx]
                            .prioritized_warps
                            .len();
                        let num_play_warps = play_sim_state.schedulers[core_id][sched_idx]
                            .prioritized_warps
                            .len();
                        let limit = num_box_warps.min(num_play_warps);

                        // fix: make sure we only compare what can be compared
                        box_sim_state.schedulers[core_id][sched_idx]
                            .prioritized_warps
                            .split_off(limit);
                        play_sim_state.schedulers[core_id][sched_idx]
                            .prioritized_warps
                            .split_off(limit);

                        // assert_eq!(
                        //     box_sim_state.schedulers[core_id][sched_idx]
                        //         .prioritized_warps
                        //         .len(),
                        //     play_sim_state.schedulers[core_id][sched_idx]
                        //         .prioritized_warps
                        //         .len(),
                        // );
                    }
                }

                for (partition_id, partition) in play_sim.partition_units().enumerate() {
                    play_sim_state.dram_latency_queue[partition_id]
                        .extend(partition.dram_latency_queue().into_iter().map(Into::into));
                }
                for (sub_id, sub) in play_sim.sub_partitions().enumerate() {
                    play_sim_state.interconn_to_l2_queue[sub_id]
                        .extend(sub.interconn_to_l2_queue().into_iter().map(Into::into));
                    play_sim_state.l2_to_interconn_queue[sub_id]
                        .extend(sub.l2_to_interconn_queue().into_iter().map(Into::into));
                    play_sim_state.dram_to_l2_queue[sub_id]
                        .extend(sub.dram_to_l2_queue().into_iter().map(Into::into));
                    play_sim_state.l2_to_dram_queue[sub_id]
                        .extend(sub.l2_to_dram_queue().into_iter().map(Into::into));
                }

                println!("checking for diff after cycle {}", cycle - 1);
                // dbg!(&box_sim_state.schedulers);
                // dbg!(&play_sim_state.schedulers);
                diff::assert_eq!(&box_sim_state, &play_sim_state);
                println!(
                    "validated play state for cycle {}: {:#?}",
                    cycle - 1,
                    &play_sim_state
                );

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
            &stats::PerCache(play_stats.l1i_stats.clone().convert()),
            &box_stats.l1i_stats
        );
        diff::assert_eq!(
            &stats::PerCache(play_stats.l1d_stats.clone().convert()),
            &box_stats.l1d_stats,
        );
        diff::assert_eq!(
            &stats::PerCache(play_stats.l1t_stats.clone().convert()),
            &box_stats.l1t_stats,
        );
        diff::assert_eq!(
            &stats::PerCache(play_stats.l1c_stats.clone().convert()),
            &box_stats.l1c_stats,
        );
        diff::assert_eq!(
            &stats::PerCache(play_stats.l2d_stats.clone().convert()),
            &box_stats.l2d_stats,
        );

        diff::assert_eq!(
            play_stats.accesses,
            playground::stats::Accesses::from(box_stats.accesses.clone())
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

        diff::assert_eq!(&play_stats.dram, &box_dram_stats);

        let playground_instructions =
            playground::stats::InstructionCounts::from(box_stats.instructions.clone());
        diff::assert_eq!(&play_stats.instructions, &playground_instructions);

        diff::assert_eq!(
            &play_stats.sim,
            &playground::stats::Sim::from(box_stats.sim.clone()),
        );

        // this uses our custom PartialEq::eq implementation
        assert_eq!(&play_stats, &box_stats);

        assert!(false, "all good!");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_async_vectoradd() -> eyre::Result<()> {
        let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
        let vec_add_trace_dir = manifest_dir.join("results/vectorAdd/vectorAdd-100-32");

        let kernelslist = vec_add_trace_dir.join("accelsim-trace/kernelslist.g");
        let gpgpusim_config = manifest_dir.join("accelsim/gtx1080/gpgpusim.config");
        let trace_config = manifest_dir.join("accelsim/gtx1080/gpgpusim.trace.config");
        let inter_config = manifest_dir.join("accelsim/gtx1080/config_fermi_islip.icnt");

        assert!(vec_add_trace_dir.is_dir());
        assert!(kernelslist.is_file());
        assert!(gpgpusim_config.is_file());
        assert!(trace_config.is_file());
        assert!(inter_config.is_file());

        for _ in 0..10 {
            let trace_dir = vec_add_trace_dir.join("trace");
            let stats: stats::Stats = tokio::task::spawn_blocking(move || {
                let stats = super::accelmain(trace_dir, None)?;
                Ok::<_, eyre::Report>(stats)
            })
            .await??;

            let handles = (0..1).map(|_| {
                let kernelslist = kernelslist.clone();
                let inter_config = inter_config.clone();
                let trace_config = trace_config.clone();
                let gpgpusim_config = gpgpusim_config.clone();

                tokio::task::spawn_blocking(move || {
                    // let kernelslist = kernelslist.to_string_lossy().to_string();
                    // let gpgpusim_config = gpgpusim_config.to_string_lossy().to_string();
                    // let trace_config = trace_config.to_string_lossy().to_string();
                    // let inter_config = inter_config.to_string_lossy().to_string();
                    //
                    // let mut args = vec![
                    //     "-trace",
                    //     &kernelslist,
                    //     "-config",
                    //     &gpgpusim_config,
                    //     "-config",
                    //     &trace_config,
                    //     "-inter_config_file",
                    //     &inter_config,
                    // ];

                    let mut args = vec![
                        "-trace",
                        kernelslist.as_os_str().to_str().unwrap(),
                        "-config",
                        gpgpusim_config.as_os_str().to_str().unwrap(),
                        "-config",
                        trace_config.as_os_str().to_str().unwrap(),
                        "-inter_config_file",
                        inter_config.as_os_str().to_str().unwrap(),
                    ];
                    dbg!(&args);

                    let config = playground::Config::default();
                    let ref_stats = playground::run(&config, &args)?;
                    Ok::<_, eyre::Report>(ref_stats)
                })
            });

            // wait for all to complete
            let ref_stats: Vec<Result<Result<_, _>, _>> = futures::future::join_all(handles).await;
            let ref_stats: Result<Vec<Result<_, _>>, _> = ref_stats.into_iter().collect();
            let ref_stats: Result<Vec<_>, _> = ref_stats?.into_iter().collect();
            let ref_stats: Vec<_> = ref_stats?;

            let ref_stats = ref_stats[0].clone();
            dbg!(&ref_stats);
        }

        Ok(())
    }
}
