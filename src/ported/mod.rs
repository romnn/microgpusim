#![allow(warnings)]

pub mod mem_fetch;
pub mod mem_sub_partition;
pub mod set_index_function;
pub mod tag_array;

use mem_fetch::*;
use mem_sub_partition::*;
use set_index_function::*;
use tag_array::*;

use super::config::GPUConfig;
use anyhow::Result;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::time::Instant;
use trace_model::KernelLaunch;


pub type address = u64;

/// Shader config

// void trace_gpgpu_sim::createSIMTCluster() {
//   m_cluster = new simt_core_cluster *[m_shader_config->n_simt_clusters];
//   for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
//     m_cluster[i] =
//         new trace_simt_core_cluster(this, i, m_shader_config, m_memory_config,
//                                     m_shader_stats, m_memory_stats);
// }

// void trace_simt_core_cluster::create_shader_core_ctx() {
//   m_core = new shader_core_ctx *[m_config->n_simt_cores_per_cluster];
//   for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++) {
//     unsigned sid = m_config->cid_to_sid(i, m_cluster_id);
//     m_core[i] = new trace_shader_core_ctx(m_gpu, this, sid, m_cluster_id,
//                                           m_config, m_mem_config, m_stats);
//     m_core_sim_order.push_back(i);
//   }
// }

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
#[derive(Debug)]
pub struct KernelInfo {
    config: KernelLaunch,
    // function_info: FunctionInfo,
    // shared_mem: bool,
    launched: bool,
    function_info: FunctionInfo,
    // m_kernel_entry = entry;
    // m_grid_dim = gridDim;
    // m_block_dim = blockDim;
    // m_next_cta.x = 0;
    // m_next_cta.y = 0;
    // m_next_cta.z = 0;
    // m_next_tid = m_next_cta;
    // m_num_cores_running = 0;
    // m_uid = (entry->gpgpu_ctx->kernel_info_m_next_uid)++;
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
    // cache_config_set = false;
    //
    // FOR TRACE DRIVEN
    //
    // m_parser = parser;
    // m_tconfig = config;
    // m_kernel_trace_info = kernel_trace_info;
    // m_was_launched = false;
    //
    // // resolve the binary version
    // if (kernel_trace_info->binary_verion == AMPERE_RTX_BINART_VERSION ||
    //     kernel_trace_info->binary_verion == AMPERE_A100_BINART_VERSION)
    //   OpcodeMap = &Ampere_OpcodeMap;
    // else if (kernel_trace_info->binary_verion == VOLTA_BINART_VERSION)
    //   OpcodeMap = &Volta_OpcodeMap;
    // else if (kernel_trace_info->binary_verion == PASCAL_TITANX_BINART_VERSION ||
    //          kernel_trace_info->binary_verion == PASCAL_P100_BINART_VERSION)
    //   OpcodeMap = &Pascal_OpcodeMap;
    // else if (kernel_trace_info->binary_verion == KEPLER_BINART_VERSION)
    //   OpcodeMap = &Kepler_OpcodeMap;
    // else if (kernel_trace_info->binary_verion == TURING_BINART_VERSION)
    //   OpcodeMap = &Turing_OpcodeMap;
    // else {
    //   printf("unsupported binary version: %d\n",
    //          kernel_trace_info->binary_verion);
    //   fflush(stdout);
    //   exit(0);
    // }
}

impl KernelInfo {
    // gpgpu_ptx_sim_init_perf
    // GPGPUSim_Init
    // start_sim_thread
    pub fn new(config: KernelLaunch) -> Self {
        Self {
            config,
            launched: false,
            function_info: FunctionInfo {},
        }
    }

    pub fn was_launched(&self) -> bool {
        self.launched
    }
}

impl std::fmt::Display for KernelInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Kernel")
    }
}

fn parse_commands(path: impl AsRef<Path>) -> Result<Vec<trace_model::Command>> {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .open(&path.as_ref())?;
    let reader = std::io::BufReader::new(file);
    let commands = serde_json::from_reader(reader)?;
    Ok(commands)
}

#[derive(Debug, Default)]
pub struct MockSimulator {
    memory_sub_partition: Vec<MemorySubPartition>,
}

impl MockSimulator {
    pub fn can_start_kernel(&self) -> bool {
        true
    }
    pub fn launch(&mut self, kernel: &mut KernelInfo) {
        kernel.launched = true;
    }
    pub fn cycle() {}
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

pub fn accelmain(traces_dir: impl AsRef<Path>) -> Result<()> {
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

    // shader config init
    let config = GPUConfig::default();

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
    let mut commands: Vec<trace_model::Command> = parse_commands(&command_traces_path)?;

    let mut busy_streams: VecDeque<u64> = VecDeque::new();
    let mut kernels: VecDeque<KernelInfo> = VecDeque::new();
    kernels.reserve_exact(window_size);

    // kernel_trace_t* kernel_trace_info = tracer.parse_kernel_info(commandlist[i].command_string);
    // kernel_info = create_kernel_info(kernel_trace_info, m_gpgpu_context, &tconfig, &tracer);
    // kernels_info.push_back(kernel_info);

    let mut i = 0;
    while i < commands.len() || !kernels.is_empty() {
        // take as many commands as possible until we have
        // collected as many kernels to fill the window_size
        // or processed every command.
        while kernels.len() < window_size && i < commands.len() {
            let cmd = &commands[i];
            println!("command {:#?}", cmd);
            match cmd {
                trace_model::Command::MemcpyHtoD { .. } => {
                    //  m_gpgpu_sim->perf_memcpy_to_gpu(addre, Bcount);
                }
                trace_model::Command::KernelLaunch(config) => {
                    let kernel = KernelInfo::new(config.clone());
                    kernels.push_back(kernel);
                    println!("launch kernel command {:#?}", cmd);
                }
            }
            i += 1;
        }

        let mut s = MockSimulator::default();

        // Launch all kernels within window that are on a stream
        // that isn't already running
        for kernel in &mut kernels {
            let stream_busy = busy_streams.iter().any(|s| *s == kernel.config.stream_id);
            if !stream_busy && s.can_start_kernel() && !kernel.was_launched() {
                println!("launching kernel {}", kernel.config.name);
                s.launch(kernel);
                busy_streams.push_back(kernel.config.stream_id);
            }
        }

        // drive kernels to completion
        //
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
