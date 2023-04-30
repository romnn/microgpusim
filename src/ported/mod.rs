#![allow(warnings)]

pub mod addrdec;
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
use log::{info, trace, warn};
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
    // dim3 gridDim, dim3 blockDim,
    // trace_function_info *m_function_info,
    // trace_parser *parser, class trace_config *config,
    // kernel_trace_t *kernel_trace_info
    //
    // trace_config *m_tconfig;
    // const std::unordered_map<std::string, OpcodeChar> *OpcodeMap;
    // trace_parser *m_parser;
    // kernel_trace_t *m_kernel_trace_info;
    // bool m_was_launched;
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
    config: GPUConfig,
    memory_partition_units: Vec<MemoryPartitionUnit>,
    memory_sub_partitions: Vec<MemorySubPartition>,
}

impl MockSimulator {
    // see new trace_gpgpu_sim(
    //      *(m_gpgpu_context->the_gpgpusim->g_the_gpu_config),
    //      m_gpgpu_context);
    pub fn new(config: GPUConfig) -> Self {
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
            .map(|i| MemoryPartitionUnit::new(i, config))
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

        Self {
            config,
            memory_partition_units,
            memory_sub_partitions,
        }
    }

    pub fn can_start_kernel(&self) -> bool {
        true
    }

    pub fn launch(&mut self, kernel: &mut KernelInfo) {
        kernel.launched = true;
    }

    pub fn cycle() {
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
    }

    pub fn memcopy_to_gpu(&mut self, addr: address, num_bytes: u32) {
        if self.config.fill_l2_on_memcopy {
            let mut transfered = 0;
            while transfered < num_bytes {
                let write_addr = addr + transfered as u64;
                let mask: mem_access_sector_mask = 0;
                // mask.set(wr_addr % 128 / 32);
                let raw_addr = addrdec::addrdec_tlx(write_addr);
                let partition_id =
                    raw_addr.sub_partition / self.config.num_sub_partition_per_memory_channel;
                let partition = &self.memory_partition_units[partition_id];
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

pub fn accelmain(traces_dir: impl AsRef<Path>) -> Result<()> {
    info!("box version {}", 0);
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

    // todo: make this a hashset?
    let mut busy_streams: VecDeque<u64> = VecDeque::new();
    let mut kernels: VecDeque<KernelInfo> = VecDeque::new();
    kernels.reserve_exact(window_size);

    // kernel_trace_t* kernel_trace_info = tracer.parse_kernel_info(commandlist[i].command_string);
    // kernel_info = create_kernel_info(kernel_trace_info, m_gpgpu_context, &tconfig, &tracer);
    // kernels_info.push_back(kernel_info);

    let mut s = MockSimulator::new(config);

    let mut i = 0;
    while i < commands.len() || !kernels.is_empty() {
        // take as many commands as possible until we have
        // collected as many kernels to fill the window_size
        // or processed every command.

        while kernels.len() < window_size && i < commands.len() {
            let cmd = &commands[i];
            println!("command {:#?}", cmd);
            match cmd {
                trace_model::Command::MemcpyHtoD {
                    dest_device_addr,
                    num_bytes,
                } => s.memcopy_to_gpu(*dest_device_addr, *num_bytes),
                trace_model::Command::KernelLaunch(config) => {
                    let kernel = KernelInfo::new(config.clone());
                    kernels.push_back(kernel);
                    println!("launch kernel command {:#?}", cmd);
                }
            }
            i += 1;
        }

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
