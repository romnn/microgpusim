#include "gpgpu_sim.hpp"

#include "gpgpu_context.hpp"
#include "ptx_stats.hpp"

gpgpu_sim::gpgpu_sim(const gpgpu_sim_config &config, gpgpu_context *ctx)
    : gpgpu_t(config, ctx), m_config(config) {
  gpgpu_ctx = ctx;
  m_shader_config = &m_config.m_shader_config;
  m_memory_config = &m_config.m_memory_config;
  ctx->ptx_parser->set_ptx_warp_size(m_shader_config);
  ptx_file_line_stats_create_exposed_latency_tracker(m_config.num_shader());

  // #ifdef GPGPUSIM_POWER_MODEL
  //   m_gpgpusim_wrapper = new gpgpu_sim_wrapper(
  //       config.g_power_simulation_enabled, config.g_power_config_name,
  //       config.g_power_simulation_mode, config.g_dvfs_enabled);
  // #endif

  // m_shader_stats = new shader_core_stats(m_shader_config);
  // m_memory_stats = new memory_stats_t(m_config.num_shader(), m_shader_config,
  //                                     m_memory_config, this);
  // average_pipeline_duty_cycle = (float *)malloc(sizeof(float));
  // active_sms = (float *)malloc(sizeof(float));
  // m_power_stats =
  //     new power_stat_t(m_shader_config, average_pipeline_duty_cycle,
  //     active_sms,
  //                      m_shader_stats, m_memory_config, m_memory_stats);
  //
  // gpu_sim_insn = 0;
  // gpu_tot_sim_insn = 0;
  // gpu_tot_issued_cta = 0;
  // gpu_completed_cta = 0;
  // m_total_cta_launched = 0;
  // gpu_deadlock = false;
  //
  // gpu_stall_dramfull = 0;
  // gpu_stall_icnt2sh = 0;
  // partiton_reqs_in_parallel = 0;
  // partiton_reqs_in_parallel_total = 0;
  // partiton_reqs_in_parallel_util = 0;
  // partiton_reqs_in_parallel_util_total = 0;
  // gpu_sim_cycle_parition_util = 0;
  // gpu_tot_sim_cycle_parition_util = 0;
  // partiton_replys_in_parallel = 0;
  // partiton_replys_in_parallel_total = 0;
  //
  // m_memory_partition_unit =
  //     new memory_partition_unit *[m_memory_config->m_n_mem];
  // m_memory_sub_partition =
  //     new memory_sub_partition *[m_memory_config->m_n_mem_sub_partition];
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
  //
  // icnt_wrapper_init();
  // icnt_create(m_shader_config->n_simt_clusters,
  //             m_memory_config->m_n_mem_sub_partition);
  //
  // time_vector_create(NUM_MEM_REQ_STAT);
  // fprintf(stdout,
  //         "GPGPU-Sim uArch: performance model initialization complete.\n");
  //
  // m_running_kernels.resize(config.max_concurrent_kernel, NULL);
  // m_last_issued_kernel = 0;
  // m_last_cluster_issue = m_shader_config->n_simt_clusters -
  //                        1; // this causes first launch to use simt cluster 0
  // *average_pipeline_duty_cycle = 0;
  // *active_sms = 0;
  //
  // last_liveness_message_time = 0;

  // Jin: functional simulation for CDP
  // m_functional_sim = false;
  // m_functional_sim_kernel = NULL;
}

void gpgpu_sim::hit_watchpoint(unsigned watchpoint_num, ptx_thread_info *thd,
                               const ptx_instruction *pI) {
  g_watchpoint_hits[watchpoint_num] = watchpoint_event(thd, pI);
}
