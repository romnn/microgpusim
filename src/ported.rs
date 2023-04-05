use super::config::GPUConfig;
use anyhow::Result;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};

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
///
/// TODO: rename to just kernel if this handles all the state.
#[derive(Debug)]
pub struct KernelInfo {
    launched: bool,
}

impl KernelInfo {
    // gpgpu_ptx_sim_init_perf
    // GPGPUSim_Init
    // start_sim_thread

    pub fn was_launched(&self) -> bool {
        false
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

#[derive(Debug)]
pub struct MockSimulator {}

impl MockSimulator {
    pub fn can_start_kernel(&self) -> bool {
        true
    }
    pub fn launch(&self, kernel: &KernelInfo) {
        kernel.launched = true;
    }
}

pub fn accelmain(trace_dir: impl AsRef<Path>) -> Result<()> {
    let trace_dir = trace_dir.as_ref();
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

    // todo

    let traces_dir = PathBuf::from(
        std::env::var("TRACES_DIR").unwrap_or(env!("CARGO_MANIFEST_DIR").to_string()),
    );
    // map_or_else(|_| example_dir.join("traces"), PathBuf::from)
    let command_traces_path = traces_dir.join("commands.json");
    let mut commands: Vec<trace_model::Command> = parse_commands(&command_traces_path)?;

    // std::vector<trace_command> commandlist = tracer.parse_commandlist_file();
    let mut busy_streams: VecDeque<usize> = VecDeque::new();
    let mut kernels: VecDeque<KernelInfo> = VecDeque::new();
    kernels.reserve_exact(window_size);

    // kernel_trace_t* kernel_trace_info = tracer.parse_kernel_info(commandlist[i].command_string);
    // kernel_info = create_kernel_info(kernel_trace_info, m_gpgpu_context, &tconfig, &tracer);
    // kernels_info.push_back(kernel_info);
    // std::cout << "Header info loaded for kernel command : " << commandlist[i].command_string << std::endl;

    let mut i = 0;
    while i < commands.len() || !kernels.is_empty() {
        // take as many commands as possible until we have
        // collected as many kernels to fill the window_size
        // or processed every command.
        while kernels.len() < window_size && i < commands.len() {
            match &commands[i] {
                cmd @ trace_model::Command::MemcpyHtoD { .. } => {
                    println!("memcpy command {:#?}", cmd);
                    //  m_gpgpu_sim->perf_memcpy_to_gpu(addre, Bcount);
                }
                cmd @ trace_model::Command::KernelLaunch(_) => {
                    let kernel_info = KernelInfo { launched: false };
                    kernels.push_back(kernel_info);
                    println!("launch kernel command {:#?}", cmd);
                }
            }
            i += 1;
        }

        let s = MockSimulator {};

        // Launch all kernels within window that are on a stream
        // that isn't already running
        for kernel in &kernels {
            let mut stream_busy = false;
            for stream in &busy_streams {
                if stream == kernel.stream {
                    stream_busy = true;
                }
                if !stream_busy && s.can_start_kernel() && !kernel.was_launched() {
                    println!("launching kernel {}", kernel);
                    s.launch(kernel);
                    busy_streams.push_back(kernel.stream);
                }
            }
        }
    }
    Ok(())
}

//  kernel_info = create_kernel_info(kernel_trace_info, m_gpgpu_context, &tconfig, &tracer);
//

// gpgpu_ptx_sim_info info;
//   info.smem = kernel_trace_info->shmem;
//   info.regs = kernel_trace_info->nregs;
//   dim3 gridDim(kernel_trace_info->grid_dim_x, kernel_trace_info->grid_dim_y, kernel_trace_info->grid_dim_z);
//   dim3 blockDim(kernel_trace_info->tb_dim_x, kernel_trace_info->tb_dim_y, kernel_trace_info->tb_dim_z);
//   trace_function_info *function_info =
//       new trace_function_info(info, m_gpgpu_context);
//   function_info->set_name(kernel_trace_info->kernel_name.c_str());
//   trace_kernel_info_t *kernel_info =
//       new trace_kernel_info_t(gridDim, blockDim, function_info,
//     		  parser, config, kernel_trace_info);

//   return kernel_info;
