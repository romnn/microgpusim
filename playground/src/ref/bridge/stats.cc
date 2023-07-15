#include "stats.hpp"
#include "main.hpp"

#include "../cache_stats.hpp"
#include "../cache_stats.hpp"
#include "../cache_sub_stats.hpp"
#include "../memory_sub_partition.hpp"
#include "../cache_reservation_fail_reason.hpp"
#include "../trace_simt_core_cluster.hpp"
#include "../trace_shader_core_ctx.hpp"
#include "../read_only_cache.hpp"
#include "../ldst_unit.hpp"
#include "../tex_cache.hpp"
#include "../l1_cache.hpp"

void transfer_cache_stats(CacheKind cache, unsigned cache_id,
                          const cache_stats &stats, Stats &out) {
  for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
    for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
      out.add_accesses(cache, cache_id, type, status, false,
                       (stats)(type, status, false));
    }
    for (unsigned status = 0; status < NUM_CACHE_RESERVATION_FAIL_STATUS;
         ++status) {
      out.add_accesses(cache, cache_id, type, status, true,
                       (stats)(type, status, true));
    }
  }
}

void accelsim_bridge::transfer_stats(Stats &stats) const {
  m_gpgpu_sim->transfer_stats(stats);
}

void trace_gpgpu_sim_bridge::transfer_stats(Stats &stats) const {
  transfer_general_stats(stats);
  transfer_dram_stats(stats);

  // per core cache stats
  transfer_core_cache_stats(stats);

  // l2 data cache stats
  transfer_l2d_stats(stats);
}

void trace_gpgpu_sim_bridge::transfer_dram_stats(Stats &stats) const {
  // dram stats are set with
  // m_stats->memlatstat_dram_access(data);

  unsigned i, j, k, l, m;
  unsigned max_bank_accesses, min_bank_accesses, max_chip_accesses,
      min_chip_accesses;

  k = 0;
  l = 0;
  m = 0;
  max_bank_accesses = 0;
  max_chip_accesses = 0;
  min_bank_accesses = 0xFFFFFFFF;
  min_chip_accesses = 0xFFFFFFFF;
  for (i = 0; i < m_memory_config->m_n_mem; i++) {
    for (j = 0; j < m_memory_config->nbk; j++) {
      l = m_memory_stats->totalbankaccesses[i][j];
      if (l < min_bank_accesses) min_bank_accesses = l;
      if (l > max_bank_accesses) max_bank_accesses = l;
      k += l;
      m += l;
    }
    if (m < min_chip_accesses) min_chip_accesses = m;
    if (m > max_chip_accesses) max_chip_accesses = m;
    m = 0;
  }
  stats.set_total_dram_accesses(k);

  // read access
  k = 0;
  l = 0;
  m = 0;
  max_bank_accesses = 0;
  max_chip_accesses = 0;
  min_bank_accesses = 0xFFFFFFFF;
  min_chip_accesses = 0xFFFFFFFF;
  for (i = 0; i < m_memory_config->m_n_mem; i++) {
    for (j = 0; j < m_memory_config->nbk; j++) {
      l = m_memory_stats->totalbankreads[i][j];
      if (l < min_bank_accesses) min_bank_accesses = l;
      if (l > max_bank_accesses) max_bank_accesses = l;
      k += l;
      m += l;
    }
    if (m < min_chip_accesses) min_chip_accesses = m;
    if (m > max_chip_accesses) max_chip_accesses = m;
    m = 0;
  }
  stats.set_total_dram_reads(k);

  // write access
  k = 0;
  l = 0;
  m = 0;
  max_bank_accesses = 0;
  max_chip_accesses = 0;
  min_bank_accesses = 0xFFFFFFFF;
  min_chip_accesses = 0xFFFFFFFF;
  for (i = 0; i < m_memory_config->m_n_mem; i++) {
    for (j = 0; j < m_memory_config->nbk; j++) {
      l = m_memory_stats->totalbankwrites[i][j];
      if (l < min_bank_accesses) min_bank_accesses = l;
      if (l > max_bank_accesses) max_bank_accesses = l;
      k += l;
      m += l;
    }
    if (m < min_chip_accesses) min_chip_accesses = m;
    if (m > max_chip_accesses) max_chip_accesses = m;
    m = 0;
  }
  stats.set_total_dram_writes(k);
}

/// see: void trace_gpgpu_sim::gpu_print_stat() {
void trace_gpgpu_sim_bridge::transfer_general_stats(Stats &stats) const {
  stats.set_sim_cycle(gpu_tot_sim_cycle);
  stats.set_sim_instructions(gpu_tot_sim_insn);

  // gpu_sim_cycle and gpu_sim_insn are reset in between launches using
  // update_stats() stats.set_sim_cycle(gpu_sim_cycle);
  // stats.set_sim_instructions(gpu_sim_insn);

  // stats.set_global_float("gpu_ipc", (float)gpu_sim_insn / gpu_sim_cycle);
  // stats.set_global_u64("gpu_tot_sim_cycle",
  //                      gpu_tot_sim_cycle + gpu_sim_cycle);
  // stats.set_global_u64("gpu_tot_sim_insn", gpu_tot_sim_insn +
  // gpu_sim_insn);
  //
  // stats.set_global_float("gpu_tot_ipc",
  //                        (float)(gpu_tot_sim_insn + gpu_sim_insn) /
  //                            (gpu_tot_sim_cycle + gpu_sim_cycle));
  //
  // stats.set_global_u64("gpu_tot_issued_cta",
  //                      gpu_tot_issued_cta + m_total_cta_launched);

  // see: m_shader_stats->print(stdout);
  // stats.set_num_stall_shared_mem(m_shader_stats->gpgpu_n_stall_shd_mem);
  stats.set_num_mem_write(m_shader_stats->made_write_mfs);
  stats.set_num_mem_read(m_shader_stats->made_read_mfs);
  stats.set_num_mem_const(m_shader_stats->gpgpu_n_mem_const);
  stats.set_num_mem_texture(m_shader_stats->gpgpu_n_mem_texture);
  stats.set_num_mem_read_global(m_shader_stats->gpgpu_n_mem_read_global);
  stats.set_num_mem_write_global(m_shader_stats->gpgpu_n_mem_write_global);
  stats.set_num_mem_read_local(m_shader_stats->gpgpu_n_mem_read_local);
  stats.set_num_mem_write_local(m_shader_stats->gpgpu_n_mem_write_local);
  stats.set_num_mem_l2_writeback(m_shader_stats->gpgpu_n_mem_l2_writeback);
  stats.set_num_mem_l1_write_allocate(
      m_shader_stats->gpgpu_n_mem_l1_write_allocate);
  stats.set_num_mem_l2_write_allocate(
      m_shader_stats->gpgpu_n_mem_l2_write_allocate);

  stats.set_num_load_instructions(m_shader_stats->gpgpu_n_load_insn);
  stats.set_num_store_instructions(m_shader_stats->gpgpu_n_store_insn);
  stats.set_num_shared_mem_instructions(m_shader_stats->gpgpu_n_shmem_insn);
  stats.set_num_sstarr_instructions(m_shader_stats->gpgpu_n_sstarr_insn);
  stats.set_num_texture_instructions(m_shader_stats->gpgpu_n_tex_insn);
  stats.set_num_const_instructions(m_shader_stats->gpgpu_n_const_insn);
  stats.set_num_param_instructions(m_shader_stats->gpgpu_n_param_insn);

  //   fprintf(fout, "gpgpu_n_shmem_bkconflict = %d\n",
  //   gpgpu_n_shmem_bkconflict); fprintf(fout, "gpgpu_n_cache_bkconflict =
  //   %d\n", gpgpu_n_cache_bkconflict);
  //
  //   fprintf(fout, "gpgpu_n_intrawarp_mshr_merge = %d\n",
  //           gpgpu_n_intrawarp_mshr_merge);
  //   fprintf(fout, "gpgpu_n_cmem_portconflict = %d\n",
  //   gpgpu_n_cmem_portconflict);
}

void trace_gpgpu_sim_bridge::transfer_core_cache_stats(Stats &stats) const {
  for (unsigned cluster_id = 0; cluster_id < m_shader_config->n_simt_clusters;
       ++cluster_id) {
    for (unsigned core_id = 0;
         core_id < m_shader_config->n_simt_cores_per_cluster; ++core_id) {
      trace_shader_core_ctx *core = m_cluster[cluster_id]->m_core[core_id];

      unsigned global_cache_id = cluster_id * +core_id;
      assert(core->m_tpc == cluster_id);
      assert(core->m_sid == core_id);

      // L1I
      if (!m_shader_config->m_L1I_config.disabled() && core->m_L1I)
        transfer_cache_stats(CacheKind::L1I, global_cache_id,
                             core->m_L1I->get_stats(), stats);

      // L1T
      if (!m_shader_config->m_L1T_config.disabled() && core->m_ldst_unit &&
          core->m_ldst_unit->m_L1T)
        transfer_cache_stats(CacheKind::L1T, global_cache_id,
                             core->m_ldst_unit->m_L1T->get_stats(), stats);

      // L1D
      if (!m_shader_config->m_L1D_config.disabled() && core->m_ldst_unit &&
          core->m_ldst_unit->m_L1D)
        transfer_cache_stats(CacheKind::L1D, global_cache_id,
                             core->m_ldst_unit->m_L1D->get_stats(), stats);

      // L1C
      if (!m_shader_config->m_L1C_config.disabled() && core->m_ldst_unit &&
          core->m_ldst_unit->m_L1C)
        transfer_cache_stats(CacheKind::L1C, global_cache_id,
                             core->m_ldst_unit->m_L1C->get_stats(), stats);
    }
  }
}

/// L2 cache stats
void trace_gpgpu_sim_bridge::transfer_l2d_stats(Stats &stats) const {
  if (m_memory_config->m_L2_config.disabled()) {
    return;
  }

  cache_stats l2_stats;
  struct cache_sub_stats l2_css;
  struct cache_sub_stats total_l2_css;
  l2_stats.clear();
  l2_css.clear();
  total_l2_css.clear();

  for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
    if (!m_memory_sub_partition[i]->m_config->m_L2_config.disabled()) {
      class l2_cache *l2_cache = m_memory_sub_partition[i]->m_L2cache;

      transfer_cache_stats(CacheKind::L2D, i, l2_cache->get_stats(), stats);
    };
  }
}
