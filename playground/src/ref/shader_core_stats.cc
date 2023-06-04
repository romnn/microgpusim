#include "shader_core_stats.hpp"

void shader_core_stats::print(FILE *fout) const {
  unsigned long long thread_icount_uarch = 0;
  unsigned long long warp_icount_uarch = 0;

  for (unsigned i = 0; i < m_config->num_shader(); i++) {
    thread_icount_uarch += m_num_sim_insn[i];
    warp_icount_uarch += m_num_sim_winsn[i];
  }
  fprintf(fout, "gpgpu_n_tot_thrd_icount = %lld\n", thread_icount_uarch);
  fprintf(fout, "gpgpu_n_tot_w_icount = %lld\n", warp_icount_uarch);

  fprintf(fout, "gpgpu_n_stall_shd_mem = %d\n", gpgpu_n_stall_shd_mem);
  fprintf(fout, "gpgpu_n_mem_read_local = %d\n", gpgpu_n_mem_read_local);
  fprintf(fout, "gpgpu_n_mem_write_local = %d\n", gpgpu_n_mem_write_local);
  fprintf(fout, "gpgpu_n_mem_read_global = %d\n", gpgpu_n_mem_read_global);
  fprintf(fout, "gpgpu_n_mem_write_global = %d\n", gpgpu_n_mem_write_global);
  fprintf(fout, "gpgpu_n_mem_texture = %d\n", gpgpu_n_mem_texture);
  fprintf(fout, "gpgpu_n_mem_const = %d\n", gpgpu_n_mem_const);

  fprintf(fout, "gpgpu_n_load_insn  = %d\n", gpgpu_n_load_insn);
  fprintf(fout, "gpgpu_n_store_insn = %d\n", gpgpu_n_store_insn);
  fprintf(fout, "gpgpu_n_shmem_insn = %d\n", gpgpu_n_shmem_insn);
  fprintf(fout, "gpgpu_n_sstarr_insn = %d\n", gpgpu_n_sstarr_insn);
  fprintf(fout, "gpgpu_n_tex_insn = %d\n", gpgpu_n_tex_insn);
  fprintf(fout, "gpgpu_n_const_mem_insn = %d\n", gpgpu_n_const_insn);
  fprintf(fout, "gpgpu_n_param_mem_insn = %d\n", gpgpu_n_param_insn);

  fprintf(fout, "gpgpu_n_shmem_bkconflict = %d\n", gpgpu_n_shmem_bkconflict);
  fprintf(fout, "gpgpu_n_cache_bkconflict = %d\n", gpgpu_n_cache_bkconflict);

  fprintf(fout, "gpgpu_n_intrawarp_mshr_merge = %d\n",
          gpgpu_n_intrawarp_mshr_merge);
  fprintf(fout, "gpgpu_n_cmem_portconflict = %d\n", gpgpu_n_cmem_portconflict);

  fprintf(fout, "gpgpu_stall_shd_mem[c_mem][resource_stall] = %d\n",
          gpu_stall_shd_mem_breakdown[C_MEM][BK_CONF]);
  // fprintf(fout, "gpgpu_stall_shd_mem[c_mem][mshr_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[C_MEM][MSHR_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[c_mem][icnt_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[C_MEM][ICNT_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[c_mem][data_port_stall] = %d\n",
  // gpu_stall_shd_mem_breakdown[C_MEM][DATA_PORT_STALL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[t_mem][mshr_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[T_MEM][MSHR_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[t_mem][icnt_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[T_MEM][ICNT_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[t_mem][data_port_stall] = %d\n",
  // gpu_stall_shd_mem_breakdown[T_MEM][DATA_PORT_STALL]);
  fprintf(fout, "gpgpu_stall_shd_mem[s_mem][bk_conf] = %d\n",
          gpu_stall_shd_mem_breakdown[S_MEM][BK_CONF]);
  fprintf(
      fout, "gpgpu_stall_shd_mem[gl_mem][resource_stall] = %d\n",
      gpu_stall_shd_mem_breakdown[G_MEM_LD][BK_CONF] +
          gpu_stall_shd_mem_breakdown[G_MEM_ST][BK_CONF] +
          gpu_stall_shd_mem_breakdown[L_MEM_LD][BK_CONF] +
          gpu_stall_shd_mem_breakdown[L_MEM_ST][BK_CONF]); // coalescing stall
                                                           // at data cache
  fprintf(
      fout, "gpgpu_stall_shd_mem[gl_mem][coal_stall] = %d\n",
      gpu_stall_shd_mem_breakdown[G_MEM_LD][COAL_STALL] +
          gpu_stall_shd_mem_breakdown[G_MEM_ST][COAL_STALL] +
          gpu_stall_shd_mem_breakdown[L_MEM_LD][COAL_STALL] +
          gpu_stall_shd_mem_breakdown[L_MEM_ST]
                                     [COAL_STALL]); // coalescing stall + bank
                                                    // conflict at data cache
  fprintf(fout, "gpgpu_stall_shd_mem[gl_mem][data_port_stall] = %d\n",
          gpu_stall_shd_mem_breakdown[G_MEM_LD][DATA_PORT_STALL] +
              gpu_stall_shd_mem_breakdown[G_MEM_ST][DATA_PORT_STALL] +
              gpu_stall_shd_mem_breakdown[L_MEM_LD][DATA_PORT_STALL] +
              gpu_stall_shd_mem_breakdown[L_MEM_ST]
                                         [DATA_PORT_STALL]); // data port stall
                                                             // at data cache
  // fprintf(fout, "gpgpu_stall_shd_mem[g_mem_ld][mshr_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[G_MEM_LD][MSHR_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[g_mem_ld][icnt_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[G_MEM_LD][ICNT_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[g_mem_ld][wb_icnt_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[G_MEM_LD][WB_ICNT_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[g_mem_ld][wb_rsrv_fail] = %d\n",
  // gpu_stall_shd_mem_breakdown[G_MEM_LD][WB_CACHE_RSRV_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[g_mem_st][mshr_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[G_MEM_ST][MSHR_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[g_mem_st][icnt_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[G_MEM_ST][ICNT_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[g_mem_st][wb_icnt_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[G_MEM_ST][WB_ICNT_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[g_mem_st][wb_rsrv_fail] = %d\n",
  // gpu_stall_shd_mem_breakdown[G_MEM_ST][WB_CACHE_RSRV_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[l_mem_ld][mshr_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[L_MEM_LD][MSHR_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[l_mem_ld][icnt_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[L_MEM_LD][ICNT_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[l_mem_ld][wb_icnt_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[L_MEM_LD][WB_ICNT_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[l_mem_ld][wb_rsrv_fail] = %d\n",
  // gpu_stall_shd_mem_breakdown[L_MEM_LD][WB_CACHE_RSRV_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[l_mem_st][mshr_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[L_MEM_ST][MSHR_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[l_mem_st][icnt_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[L_MEM_ST][ICNT_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[l_mem_ld][wb_icnt_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[L_MEM_ST][WB_ICNT_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[l_mem_ld][wb_rsrv_fail] = %d\n",
  // gpu_stall_shd_mem_breakdown[L_MEM_ST][WB_CACHE_RSRV_FAIL]);

  fprintf(fout, "gpu_reg_bank_conflict_stalls = %d\n",
          gpu_reg_bank_conflict_stalls);

  fprintf(fout, "Warp Occupancy Distribution:\n");
  fprintf(fout, "Stall:%d\t", shader_cycle_distro[2]);
  fprintf(fout, "W0_Idle:%d\t", shader_cycle_distro[0]);
  fprintf(fout, "W0_Scoreboard:%d", shader_cycle_distro[1]);
  for (unsigned i = 3; i < m_config->warp_size + 3; i++)
    fprintf(fout, "\tW%d:%d", i - 2, shader_cycle_distro[i]);
  fprintf(fout, "\n");
  fprintf(fout, "single_issue_nums: ");
  for (unsigned i = 0; i < m_config->gpgpu_num_sched_per_core; i++)
    fprintf(fout, "WS%d:%d\t", i, single_issue_nums[i]);
  fprintf(fout, "\n");
  fprintf(fout, "dual_issue_nums: ");
  for (unsigned i = 0; i < m_config->gpgpu_num_sched_per_core; i++)
    fprintf(fout, "WS%d:%d\t", i, dual_issue_nums[i]);
  fprintf(fout, "\n");

  m_outgoing_traffic_stats->print(fout);
  m_incoming_traffic_stats->print(fout);
}

void shader_core_stats::event_warp_issued(unsigned s_id, unsigned warp_id,
                                          unsigned num_issued,
                                          unsigned dynamic_warp_id) {
  assert(warp_id <= m_config->max_warps_per_shader);
  for (unsigned i = 0; i < num_issued; ++i) {
    if (m_shader_dynamic_warp_issue_distro[s_id].size() <= dynamic_warp_id) {
      m_shader_dynamic_warp_issue_distro[s_id].resize(dynamic_warp_id + 1);
    }
    ++m_shader_dynamic_warp_issue_distro[s_id][dynamic_warp_id];
    if (m_shader_warp_slot_issue_distro[s_id].size() <= warp_id) {
      m_shader_warp_slot_issue_distro[s_id].resize(warp_id + 1);
    }
    ++m_shader_warp_slot_issue_distro[s_id][warp_id];
  }
}

void shader_core_stats::visualizer_print(gzFile visualizer_file) {
  // warp divergence breakdown
  gzprintf(visualizer_file, "WarpDivergenceBreakdown:");
  unsigned int total = 0;
  unsigned int cf =
      (m_config->gpgpu_warpdistro_shader == -1) ? m_config->num_shader() : 1;
  gzprintf(visualizer_file, " %d",
           (shader_cycle_distro[0] - last_shader_cycle_distro[0]) / cf);
  gzprintf(visualizer_file, " %d",
           (shader_cycle_distro[1] - last_shader_cycle_distro[1]) / cf);
  gzprintf(visualizer_file, " %d",
           (shader_cycle_distro[2] - last_shader_cycle_distro[2]) / cf);
  for (unsigned i = 0; i < m_config->warp_size + 3; i++) {
    if (i >= 3) {
      total += (shader_cycle_distro[i] - last_shader_cycle_distro[i]);
      if (((i - 3) % (m_config->warp_size / 8)) ==
          ((m_config->warp_size / 8) - 1)) {
        gzprintf(visualizer_file, " %d", total / cf);
        total = 0;
      }
    }
    last_shader_cycle_distro[i] = shader_cycle_distro[i];
  }
  gzprintf(visualizer_file, "\n");

  gzprintf(visualizer_file, "ctas_completed: %d\n", ctas_completed);
  ctas_completed = 0;
  // warp issue breakdown
  unsigned sid = m_config->gpgpu_warp_issue_shader;
  unsigned count = 0;
  unsigned warp_id_issued_sum = 0;
  gzprintf(visualizer_file, "WarpIssueSlotBreakdown:");
  if (m_shader_warp_slot_issue_distro[sid].size() > 0) {
    for (std::vector<unsigned>::const_iterator iter =
             m_shader_warp_slot_issue_distro[sid].begin();
         iter != m_shader_warp_slot_issue_distro[sid].end(); iter++, count++) {
      unsigned diff = count < m_last_shader_warp_slot_issue_distro.size()
                          ? *iter - m_last_shader_warp_slot_issue_distro[count]
                          : *iter;
      gzprintf(visualizer_file, " %d", diff);
      warp_id_issued_sum += diff;
    }
    m_last_shader_warp_slot_issue_distro = m_shader_warp_slot_issue_distro[sid];
  } else {
    gzprintf(visualizer_file, " 0");
  }
  gzprintf(visualizer_file, "\n");

#define DYNAMIC_WARP_PRINT_RESOLUTION 32
  unsigned total_issued_this_resolution = 0;
  unsigned dynamic_id_issued_sum = 0;
  count = 0;
  gzprintf(visualizer_file, "WarpIssueDynamicIdBreakdown:");
  if (m_shader_dynamic_warp_issue_distro[sid].size() > 0) {
    for (std::vector<unsigned>::const_iterator iter =
             m_shader_dynamic_warp_issue_distro[sid].begin();
         iter != m_shader_dynamic_warp_issue_distro[sid].end();
         iter++, count++) {
      unsigned diff =
          count < m_last_shader_dynamic_warp_issue_distro.size()
              ? *iter - m_last_shader_dynamic_warp_issue_distro[count]
              : *iter;
      total_issued_this_resolution += diff;
      if ((count + 1) % DYNAMIC_WARP_PRINT_RESOLUTION == 0) {
        gzprintf(visualizer_file, " %d", total_issued_this_resolution);
        dynamic_id_issued_sum += total_issued_this_resolution;
        total_issued_this_resolution = 0;
      }
    }
    if (count % DYNAMIC_WARP_PRINT_RESOLUTION != 0) {
      gzprintf(visualizer_file, " %d", total_issued_this_resolution);
      dynamic_id_issued_sum += total_issued_this_resolution;
    }
    m_last_shader_dynamic_warp_issue_distro =
        m_shader_dynamic_warp_issue_distro[sid];
    assert(warp_id_issued_sum == dynamic_id_issued_sum);
  } else {
    gzprintf(visualizer_file, " 0");
  }
  gzprintf(visualizer_file, "\n");

  // overall cache miss rates
  gzprintf(visualizer_file, "gpgpu_n_cache_bkconflict: %d\n",
           gpgpu_n_cache_bkconflict);
  gzprintf(visualizer_file, "gpgpu_n_shmem_bkconflict: %d\n",
           gpgpu_n_shmem_bkconflict);

  // instruction count per shader core
  gzprintf(visualizer_file, "shaderinsncount:  ");
  for (unsigned i = 0; i < m_config->num_shader(); i++)
    gzprintf(visualizer_file, "%u ", m_num_sim_insn[i]);
  gzprintf(visualizer_file, "\n");
  // warp instruction count per shader core
  gzprintf(visualizer_file, "shaderwarpinsncount:  ");
  for (unsigned i = 0; i < m_config->num_shader(); i++)
    gzprintf(visualizer_file, "%u ", m_num_sim_winsn[i]);
  gzprintf(visualizer_file, "\n");
  // warp divergence per shader core
  gzprintf(visualizer_file, "shaderwarpdiv: ");
  for (unsigned i = 0; i < m_config->num_shader(); i++)
    gzprintf(visualizer_file, "%u ", m_n_diverge[i]);
  gzprintf(visualizer_file, "\n");
}
