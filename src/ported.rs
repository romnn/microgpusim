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
    Ok(())
}
