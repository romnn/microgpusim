use super::config::GPUConfig;
use anyhow::Result;

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

/// KernelInfo
#[derive(Debug)]
pub struct KernelInfo {}

impl KernelInfo {
    // gpgpu_ptx_sim_init_perf
    // GPGPUSim_Init
    // start_sim_thread
}

// while (!fs.eof()) {
//     getline(fs, line);
//     if (line.empty())
//       continue;
//     else if (line.substr(0, 10) == "MemcpyHtoD") {
//       trace_command command;
//       command.command_string = line;
//       command.m_type = command_type::cpu_gpu_mem_copy;
//       commandlist.push_back(command);
//     } else if (line.substr(0, 6) == "kernel") {
//       trace_command command;
//       command.m_type = command_type::kernel_launch;
//       filepath = directory + "/" + line;
//       command.command_string = filepath;
//       commandlist.push_back(command);
//     }
//     // ignore gpu_to_cpu_memory_cpy
// }

fn accelmain() -> Result<()> {
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
    // m_gpgpu_context->the_gpgpusim->g_simulation_starttime = time((time_t *)NULL);
    //
    // return m_gpgpu_context->the_gpgpusim->g_the_gpu;

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
    let mut busy_streams: Vec<usize> = Vec::new();
    let mut kernels: Vec<KernelInfo> = Vec::new();
    kernels.reserve_exact(window_size);

    // kernel_trace_t* kernel_trace_info = tracer.parse_kernel_info(commandlist[i].command_string);
    // kernel_info = create_kernel_info(kernel_trace_info, m_gpgpu_context, &tconfig, &tracer);
    // kernels_info.push_back(kernel_info);
    // std::cout << "Header info loaded for kernel command : " << commandlist[i].command_string << std::endl;

    // unsigned i = 0;
    // while i < commandlist.size() || !kernels_info.empty()) {
    Ok(())
}
