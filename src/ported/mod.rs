#![allow(warnings)]

pub mod addrdec;
pub mod cache;
pub mod cache_block;
pub mod operand_collector;
pub mod register_set;
pub mod core;
pub mod instruction;
pub mod interconn;
pub mod l1;
pub mod l2;
pub mod ldst_unit;
pub mod mem_fetch;
pub mod mem_sub_partition;
pub mod mshr;
pub mod opcodes;
pub mod scheduler;
pub mod set_index_function;
pub mod stats;
pub mod tag_array;
pub mod utils;

use self::core::*;
use addrdec::*;
use interconn::*;
use ldst_unit::*;
use mem_fetch::*;
use mem_sub_partition::*;
use scheduler::*;
use set_index_function::*;
use stats::Stats;
use tag_array::*;
use utils::*;

use crate::config;
use color_eyre::eyre;
use itertools::Itertools;
use log::{error, info, trace, warn};
use nvbit_model::dim::{self, Dim};
use std::collections::{HashMap, VecDeque};
use std::fmt::Binary;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;
use trace_model::{Command, KernelLaunch, MemAccessTraceEntry};

pub type address = u64;

/// Shader config

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
            // next_block_iter.lock().unwrap().peek())
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
        trace.sort_unstable_by(|a, b| (a.block_id, a.warp_id).cmp(&(b.block_id, b.warp_id)));
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
    pub fn next_threadblock_traces(&self, warps: &mut [SchedulerWarp]) {
        // debug_assert!(self.next_block.is_some());
        // todo!("next_threadblock_traces");
        // debug_assert!(self.next_block_iter.peek().is_some());
        // // let Some(next_block) = self.next_block else {
        // let Some(next_block) = self.next_block_iter.next() else {
        //     return;
        // };
        let next_block = nvbit_model::Dim { x: 0, y: 0, z: 0 };
        dbg!(&next_block);
        for warp in warps.iter_mut() {
            warp.clear();
        }
        let mut lock = self.trace_iter.write().unwrap();
        let trace_iter = lock.take_while_ref(|entry| entry.block_id == next_block);
        for trace in trace_iter {
            // dbg!(&trace);
            let warp_id = trace.warp_id as usize;
            let instr = instruction::WarpInstruction::from_trace(&self, trace);
            warps[warp_id].trace_instructions.push_back(instr);
        }

        // set the pc from the traces and ignore the functional model
        for warp in warps.iter_mut() {
            let num_instr = warp.trace_instructions.len();
            if num_instr > 0 {
                println!("warp {}: {num_instr} instructions", warp.warp_id);
            }
            warp.next_pc = warp.trace_start_pc();
        }
        // println!("added {total} instructions");
    }

    // pub fn next_threadblock_traces(&self) -> impl Iterator<Item = MemAccessTraceEntry> + '_ {
    //     let mut iter = self.trace_iter.write().unwrap();
    //     iter.take_while_ref(|entry| Some(entry.block_id) == self.next_block).collect()
    //         // .cloned()
    // }

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
        // todo!("no_more_blocks_to_run");
        self.current_block().is_some()
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

#[derive(Debug, Default)]
pub struct MockSimulator {
    stats: Arc<Mutex<Stats>>,
    config: Arc<config::GPUConfig>,
    memory_partition_units: Vec<MemoryPartitionUnit>,
    memory_sub_partitions: Vec<MemorySubPartition>,
    running_kernels: Vec<Option<Arc<KernelInfo>>>,
    executed_kernels: Mutex<HashMap<usize, String>>,
    clusters: Vec<SIMTCoreCluster>,
    last_cluster_issue: usize,
    last_issued_kernel: usize,
}

impl MockSimulator {
    // see new trace_gpgpu_sim(
    //      *(m_gpgpu_context->the_gpgpusim->g_the_gpu_config),
    //      m_gpgpu_context);
    pub fn new(config: Arc<config::GPUConfig>) -> Self {
        let stats = Arc::new(Mutex::new(Stats::default()));
        //         gpgpu_ctx = ctx;
        //   m_shader_config = &m_config.m_shader_config;
        //   m_memory_config = &m_config.m_memory_config;
        //   ctx->ptx_parser->set_ptx_warp_size(m_shader_config);
        //   ptx_file_line_stats_create_exposed_latency_tracker(m_config.num_shader());
        //
        // #ifdef GPGPUSIM_POWER_MODEL
        //   m_gpgpusim_wrapper = new gpgpu_sim_wrapper(config.g_power_simulation_enabled,
        //                                              config.g_power_config_name, config.g_power_simulation_mode, config.g_dvfs_enabled);
        // #endif
        //
        //   m_shader_stats = new shader_core_stats(m_shader_config);
        //   m_memory_stats = new memory_stats_t(m_config.num_shader(), m_shader_config,
        //                                       m_memory_config, this);
        //   average_pipeline_duty_cycle = (float *)malloc(sizeof(float));
        //   active_sms = (float *)malloc(sizeof(float));
        //   m_power_stats =
        //       new power_stat_t(m_shader_config, average_pipeline_duty_cycle, active_sms,
        //                        m_shader_stats, m_memory_config, m_memory_stats);
        //
        //   gpu_sim_insn = 0;
        //   gpu_tot_sim_insn = 0;
        //   gpu_tot_issued_cta = 0;
        //   gpu_completed_cta = 0;
        //   m_total_cta_launched = 0;
        //   gpu_deadlock = false;
        //
        //   gpu_stall_dramfull = 0;
        //   gpu_stall_icnt2sh = 0;
        //   partiton_reqs_in_parallel = 0;
        //   partiton_reqs_in_parallel_total = 0;
        //   partiton_reqs_in_parallel_util = 0;
        //   partiton_reqs_in_parallel_util_total = 0;
        //   gpu_sim_cycle_parition_util = 0;
        //   gpu_tot_sim_cycle_parition_util = 0;
        //   partiton_replys_in_parallel = 0;
        //   partiton_replys_in_parallel_total = 0;
        //
        //   m_memory_partition_unit =
        //       new memory_partition_unit *[m_memory_config->m_n_mem];
        //   m_memory_sub_partition =
        //       new memory_sub_partition *[m_memory_config->m_n_mem_sub_partition];
        //   for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
        //     m_memory_partition_unit[i] =
        //         new memory_partition_unit(i, m_memory_config, m_memory_stats, this);
        //     for (unsigned p = 0;
        //          p < m_memory_config->m_n_sub_partition_per_memory_channel; p++) {
        //       unsigned submpid =
        //           i * m_memory_config->m_n_sub_partition_per_memory_channel + p;
        //       m_memory_sub_partition[submpid] =
        //           m_memory_partition_unit[i]->get_sub_partition(p);
        //     }
        //   }
        //
        //   icnt_wrapper_init();
        //   icnt_create(m_shader_config->n_simt_clusters,
        //               m_memory_config->m_n_mem_sub_partition);
        //
        //   time_vector_create(NUM_MEM_REQ_STAT);
        //   fprintf(stdout,
        //           "GPGPU-Sim uArch: performance model initialization complete.\n");
        //
        //   m_running_kernels.resize(config.max_concurrent_kernel, NULL);
        //   m_last_issued_kernel = 0;
        //   m_last_cluster_issue = m_shader_config->n_simt_clusters -
        //                          1;  // this causes first launch to use simt cluster 0
        //   *average_pipeline_duty_cycle = 0;
        //   *active_sms = 0;
        //
        //   last_liveness_message_time = 0;
        //
        //   // Jin: functional simulation for CDP
        //   m_functional_sim = false;
        //   m_functional_sim_kernel = NULL;
        let num_mem_units = config.num_mem_units;
        let num_sub_partitions = config.num_sub_partition_per_memory_channel;

        let memory_partition_units: Vec<_> = (0..num_mem_units)
            .map(|i| MemoryPartitionUnit::new(i, config.clone()))
            .collect();

        let memory_sub_partitions = Vec::new();
        // memory_sub_partitions.reserve_exact(additional)
        // for (i, mem_unit) in memory_partition_units.iter().enumerate() {
        //     for sub_mem_id in 0..num_sub_partitions {
        //         memory_sub_partitions[sub_mem_id] = mem_unit.get_sub_partition(i);
        //     }
        // }
        //
        // let memory_sub_partition: Vec<_> = (0..config.num_sub_partition_per_memory_channel)
        //     .map(|i| MemorySubPartition::new(i, config))
        //     .collect();

        // for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
        //   m_memory_partition_unit[i] =
        //       new memory_partition_unit(i, m_memory_config, m_memory_stats, this);
        //   for (unsigned p = 0;
        //        p < m_memory_config->m_n_sub_partition_per_memory_channel; p++) {
        //     unsigned submpid =
        //         i * m_memory_config->m_n_sub_partition_per_memory_channel + p;
        //     m_memory_sub_partition[submpid] =
        //         m_memory_partition_unit[i]->get_sub_partition(p);
        //   }
        // }

        let max_concurrent_kernels = config.max_concurrent_kernels;
        let running_kernels = (0..max_concurrent_kernels).map(|_| None).collect();

        // new trace_simt_core_cluster(
        // this, i, m_shader_config, m_memory_config,
        // m_shader_stats, m_memory_stats);

        let clusters: Vec<_> = (0..config.num_simt_clusters)
            .map(|i| SIMTCoreCluster::new(i, stats.clone(), config.clone()))
            .collect();

        let executed_kernels = Mutex::new(HashMap::new());

        Self {
            config,
            stats,
            memory_partition_units,
            memory_sub_partitions,
            running_kernels,
            executed_kernels,
            clusters,
            last_cluster_issue: 0,
            last_issued_kernel: 0,
        }
    }

    /// Select the next kernel to run
    ///
    /// Todo: used hack to allow selecting the kernel from the shader core,
    /// but we could maybe refactor
    pub fn select_kernel(&self) -> Option<Arc<KernelInfo>> {
        let mut executed_kernels = self.executed_kernels.lock().unwrap();
        if let Some(k) = &self.running_kernels[self.last_issued_kernel] {
            if !k.no_more_blocks_to_run()
            // &&!kernel.kernel_TB_latency)
            {
                let launch_uid = k.uid;
                if !executed_kernels.contains_key(&launch_uid) {
                    executed_kernels.insert(launch_uid, k.name().to_string());
                }
                return Some(k.clone());
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
                    return Some(k.clone());
                }
            }
        }
        None
    }

    pub fn active(&self) -> bool {
        true
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
        let max_threads_per_block = self.config.max_threads_per_shader;
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

    // fn issue_block_to_core_inner(&self, cluster: &mut SIMTCoreCluster) -> usize {
    //     let mut num_blocks_issued = 0;
    //
    //     let num_cores = cluster.cores.len();
    //     for (i, core) in cluster.cores.iter().enumerate() {
    //         let core_id = (i + cluster.block_issue_next_core + 1) % num_cores;
    //         let mut kernel = None;
    //         if self.config.concurrent_kernel_sm {
    //             // always select latest issued kernel
    //             kernel = self.select_kernel()
    //         } else {
    //             if let Some(current) = &core.current_kernel {
    //                 if !current.no_more_blocks_to_run() {
    //                     // wait until current kernel finishes
    //                     if core.active_warps() == 0 {
    //                         kernel = self.select_kernel();
    //                         core.current_kernel = kernel;
    //                     }
    //                 }
    //             }
    //         }
    //         if let Some(kernel) = kernel {
    //             if kernel.no_more_blocks_to_run() && core.can_issue_block(kernel) {
    //                 core.issue_block(kernel);
    //                 num_blocks_issued += 1;
    //                 cluster.block_issue_next_core = i;
    //                 break;
    //             }
    //         }
    //     }
    //     num_blocks_issued
    // }

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

    pub fn cycle(&mut self) {
        self.issue_block_to_core();

        // if (clock_mask & CORE) {
        // shader core loading (pop from ICNT into core) follows CORE clock
        for cluster in &mut self.clusters {
            cluster.interconn_cycle();
        }
        // }

        let mut active_sms = 0;
        // if (clock_mask & CORE) {
        for cluster in &mut self.clusters {
            if cluster.not_completed() {
                cluster.cycle();
                active_sms += cluster.num_active_sms();
            }
            // cluster.cycle();
            // if (m_cluster[i]->get_not_completed() || get_more_cta_left()) {
            //     m_cluster[i]->core_cycle();
            // }
        }

        // if (clock_mask & DRAM) {
        // for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
        // for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
        //   if (m_memory_config->simple_dram_model) {
        //     m_memory_partition_unit[i]->simple_dram_model_cycle();
        // } else {
        //     // Issue the dram command (scheduler + delay model)
        //     m_memory_partition_unit[i]->dram_cycle();
        //     }
        // }

        let mut parallel_mem_partition_reqs_per_cycle = 0;
        let mut stall_dram_full = 0;
        for mem_sub in &mut self.memory_sub_partitions {
            if mem_sub.full(SECTOR_CHUNCK_SIZE) {
                stall_dram_full + -1;
            } else {
                // mem_fetch *mf = (mem_fetch *)icnt_pop(m_shader_config->mem2device(i));
                if let Some(fetch) = None {
                    mem_sub.push(fetch);
                    parallel_mem_partition_reqs_per_cycle += 1;
                }
            }
        }
        // if (clock_mask & L2) {
        // m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].clear();
        // for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
        //   // move memory request from interconnect into memory partition (if not
        //   // backed up) Note:This needs to be called in DRAM clock domain if there
        //   // is no L2 cache in the system In the worst case, we may need to push
        //   // SECTOR_CHUNCK_SIZE requests, so ensure you have enough buffer for them
        //   if (m_memory_sub_partition[i]->full(SECTOR_CHUNCK_SIZE)) {
        //     gpu_stall_dramfull++;
        //   } else {
        //     mem_fetch *mf = (mem_fetch *)icnt_pop(m_shader_config->mem2device(i));
        //     m_memory_sub_partition[i]->push(mf, gpu_sim_cycle + gpu_tot_sim_cycle);
        //     if (mf) partiton_reqs_in_parallel_per_cycle++;
        //   }
        //   m_memory_sub_partition[i]->cache_cycle(gpu_sim_cycle + gpu_tot_sim_cycle);
        //   m_memory_sub_partition[i]->accumulate_L2cache_stats(
        //       m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX]);
        // }
    }

    pub fn memcopy_to_gpu(&mut self, addr: address, num_bytes: u64) {
        println!("memcopy: {num_bytes} bytes to {addr:?}");
        if self.config.fill_l2_on_memcopy {
            let mut transfered = 0;
            while transfered < num_bytes {
                let write_addr = addr + transfered as u64;
                let mask: mem_access_sector_mask = 0;
                // mask.set(wr_addr % 128 / 32);
                let raw_addr = addrdec::addrdec_tlx(write_addr);
                let partition_id = raw_addr.sub_partition
                    / self.config.num_sub_partition_per_memory_channel as u64;
                let partition = &self.memory_partition_units[partition_id as usize];
                partition.handle_memcpy_to_gpu(write_addr, raw_addr.sub_partition, mask);
                transfered += 32;
            }
        }
        //   if (m_memory_config->m_perf_sim_memcpy) {
        //   // if(!m_config.trace_driven_mode)    //in trace-driven mode, CUDA runtime
        //   // can start nre data structure at any position 	assert (dst_start_addr %
        //   // 32
        //   //== 0);
        //
        //   for (unsigned counter = 0; counter < count; counter += 32) {
        //     const unsigned wr_addr = dst_start_addr + counter;
        //     addrdec_t raw_addr;
        //     mem_access_sector_mask_t mask;
        //     mask.set(wr_addr % 128 / 32);
        //     m_memory_config->m_address_mapping.addrdec_tlx(wr_addr, &raw_addr);
        //     const unsigned partition_id =
        //         raw_addr.sub_partition /
        //         m_memory_config->m_n_sub_partition_per_memory_channel;
        //     m_memory_partition_unit[partition_id]->handle_memcpy_to_gpu(
        //         wr_addr, raw_addr.sub_partition, mask);
        //   }
        // }
    }

    pub fn cache_cycle() {
        // if (m_memory_sub_partition[i]->full(SECTOR_CHUNCK_SIZE)) {
        // gpu_stall_dramfull++;
        // } else {
        // mem_fetch *mf = (mem_fetch *)icnt_pop(m_shader_config->mem2device(i));
        // m_memory_sub_partition[i]->push(mf, gpu_sim_cycle + gpu_tot_sim_cycle);
        // if (mf) partiton_reqs_in_parallel_per_cycle++;
        // }
        // m_memory_sub_partition[i]->cache_cycle(gpu_sim_cycle + gpu_tot_sim_cycle);
    }
}

// void trace_shader_core_ctx::init_traces(unsigned start_warp, unsigned end_warp,
//                                         kernel_info_t &kernel) {
//   std::vector<std::vector<inst_trace_t> *> threadblock_traces;
//   for (unsigned i = start_warp; i < end_warp; ++i) {
//     trace_shd_warp_t *m_trace_warp = static_cast<trace_shd_warp_t *>(m_warp[i]);
//     m_trace_warp->clear();
//     threadblock_traces.push_back(&(m_trace_warp->warp_traces));
//   }
//   trace_kernel_info_t &trace_kernel =
//       static_cast<trace_kernel_info_t &>(kernel);
//   trace_kernel.get_next_threadblock_traces(threadblock_traces);
//
//   // set the pc from the traces and ignore the functional model
//   for (unsigned i = start_warp; i < end_warp; ++i) {
//     trace_shd_warp_t *m_trace_warp = static_cast<trace_shd_warp_t *>(m_warp[i]);
//     m_trace_warp->set_next_pc(m_trace_warp->get_start_trace_pc());
//     m_trace_warp->set_kernel(&trace_kernel);
//   }
// }

// unsigned start_warp = start_thread / m_config->warp_size;
//   unsigned end_warp = end_thread / m_config->warp_size +
//                       ((end_thread % m_config->warp_size) ? 1 : 0);
//   // set the pc from the traces and ignore the functional model
// for (unsigned i = start_warp; i < end_warp; ++i) {
//   trace_shd_warp_t *m_trace_warp = static_cast<trace_shd_warp_t *>(m_warp[i]);
//   m_trace_warp->set_next_pc(m_trace_warp->get_start_trace_pc());
//   m_trace_warp->set_kernel(&trace_kernel);
// }

pub fn accelmain(traces_dir: impl AsRef<Path>) -> eyre::Result<()> {
    info!("box version {}", 0);
    color_eyre::install()?;
    // log = "0.4"
    let traces_dir = traces_dir.as_ref();
    // reg_options registers the options from the option parser
    // init parses some more complex string options and data structures

    // m_gpgpu_context->the_gpgpusim->g_the_gpu_config->init();
    //
    // m_gpgpu_context->the_gpgpusim->g_the_gpu = new trace_gpgpu_sim(
    //   *(m_gpgpu_context->the_gpgpusim->g_the_gpu_config), m_gpgpu_context);
    //
    // m_gpgpu_context->the_gpgpusim->g_stream_manager =
    //   new stream_manager((m_gpgpu_context->the_gpgpusim->g_the_gpu),
    //                      m_gpgpu_context->func_sim->g_cuda_launch_blocking);
    //
    let start_time = Instant::now();

    // debugging config
    let mut config = config::GPUConfig::default();
    config.num_simt_clusters = 1;
    config.num_cores_per_simt_cluster = 1;
    let config = Arc::new(config);

    assert!(config.max_threads_per_shader.rem_euclid(config.warp_size) == 0);
    let max_warps_per_shader = config.max_threads_per_shader / config.warp_size;

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

    // kernel_trace_t* kernel_trace_info = tracer.parse_kernel_info(commandlist[i].command_string);
    // kernel_info = create_kernel_info(kernel_trace_info, m_gpgpu_context, &tconfig, &tracer);
    // kernels_info.push_back(kernel_info);

    let mut sim = MockSimulator::new(config.clone());

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

        // drive kernels to completion
        // while sim.active() {
        for i in 0..1 {
            println!("cycle {i}");
            sim.cycle();
            // if !sim.active() {
            //     break;
            // }
            // if sim.active() {
            //     // sim_cycles = tru
            //     // m_gpgpu_sim->deadlock_check();
            // } else {
            // }
        }
        // bool active = false;
        // bool sim_cycles = false;
        // unsigned finished_kernel_uid = 0;
        //
        // do {
        //   if (!m_gpgpu_sim->active())
        //     break;
        //
        //   // performance simulation
        //   if (m_gpgpu_sim->active()) {
        //     m_gpgpu_sim->cycle();
        //     sim_cycles = true;
        //     m_gpgpu_sim->deadlock_check();
        //   } else {
        //     if (m_gpgpu_sim->cycle_insn_cta_max_hit()) {
        //       m_gpgpu_context->the_gpgpusim->g_stream_manager
        //           ->stop_all_running_kernels();
        //       break;
        //     }
        //   }
        //
        //   active = m_gpgpu_sim->active();
        //   finished_kernel_uid = m_gpgpu_sim->finished_kernel();
        // } while (active && !finished_kernel_uid);
        break;
    }
    Ok(())
}
