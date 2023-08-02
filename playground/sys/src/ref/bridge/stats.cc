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
                          const cache_stats &stats, StatsBridge &out) {
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

void accelsim_bridge::transfer_stats(StatsBridge &stats) const {
  transfer_general_stats(stats);
  transfer_dram_stats(stats);

  // per core cache stats
  transfer_core_cache_stats(stats);

  // l2 data cache stats
  transfer_l2d_stats(stats);
}

void accelsim_bridge::transfer_dram_stats(StatsBridge &stats) const {
  // dram stats are set with
  // m_stats->memlatstat_dram_access(data);

  unsigned i, j, k, l, m;
  unsigned max_bank_accesses, min_bank_accesses, max_chip_accesses,
      min_chip_accesses;

  unsigned num_mem = m_gpgpu_sim->m_memory_config->m_n_mem;
  unsigned num_banks = m_gpgpu_sim->m_memory_config->nbk;

  k = 0;
  l = 0;
  m = 0;
  max_bank_accesses = 0;
  max_chip_accesses = 0;
  min_bank_accesses = 0xFFFFFFFF;
  min_chip_accesses = 0xFFFFFFFF;
  for (i = 0; i < num_mem; i++) {
    for (j = 0; j < num_banks; j++) {
      l = m_gpgpu_sim->m_memory_stats->totalbankaccesses[i][j];
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
  for (i = 0; i < num_mem; i++) {
    for (j = 0; j < num_banks; j++) {
      l = m_gpgpu_sim->m_memory_stats->totalbankreads[i][j];
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
  for (i = 0; i < num_mem; i++) {
    for (j = 0; j < num_banks; j++) {
      l = m_gpgpu_sim->m_memory_stats->totalbankwrites[i][j];
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
void accelsim_bridge::transfer_general_stats(StatsBridge &stats) const {
  stats.set_sim_cycle(m_gpgpu_sim->gpu_tot_sim_cycle +
                      m_gpgpu_sim->gpu_sim_cycle);
  stats.set_sim_instructions(m_gpgpu_sim->gpu_tot_sim_insn +
                             m_gpgpu_sim->gpu_sim_insn);

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
  const shader_core_stats *shader_stats = m_gpgpu_sim->m_shader_stats;
  stats.set_num_mem_write(shader_stats->made_write_mfs);
  stats.set_num_mem_read(shader_stats->made_read_mfs);
  stats.set_num_mem_const(shader_stats->gpgpu_n_mem_const);
  stats.set_num_mem_texture(shader_stats->gpgpu_n_mem_texture);
  stats.set_num_mem_read_global(shader_stats->gpgpu_n_mem_read_global);
  stats.set_num_mem_write_global(shader_stats->gpgpu_n_mem_write_global);
  stats.set_num_mem_read_local(shader_stats->gpgpu_n_mem_read_local);
  stats.set_num_mem_write_local(shader_stats->gpgpu_n_mem_write_local);
  stats.set_num_mem_l2_writeback(shader_stats->gpgpu_n_mem_l2_writeback);
  stats.set_num_mem_l1_write_allocate(
      shader_stats->gpgpu_n_mem_l1_write_allocate);
  stats.set_num_mem_l2_write_allocate(
      shader_stats->gpgpu_n_mem_l2_write_allocate);

  stats.set_num_load_instructions(shader_stats->gpgpu_n_load_insn);
  stats.set_num_store_instructions(shader_stats->gpgpu_n_store_insn);
  stats.set_num_shared_mem_instructions(shader_stats->gpgpu_n_shmem_insn);
  stats.set_num_sstarr_instructions(shader_stats->gpgpu_n_sstarr_insn);
  stats.set_num_texture_instructions(shader_stats->gpgpu_n_tex_insn);
  stats.set_num_const_instructions(shader_stats->gpgpu_n_const_insn);
  stats.set_num_param_instructions(shader_stats->gpgpu_n_param_insn);

  //   fprintf(fout, "gpgpu_n_shmem_bkconflict = %d\n",
  //   gpgpu_n_shmem_bkconflict); fprintf(fout, "gpgpu_n_cache_bkconflict =
  //   %d\n", gpgpu_n_cache_bkconflict);
  //
  //   fprintf(fout, "gpgpu_n_intrawarp_mshr_merge = %d\n",
  //           gpgpu_n_intrawarp_mshr_merge);
  //   fprintf(fout, "gpgpu_n_cmem_portconflict = %d\n",
  //   gpgpu_n_cmem_portconflict);
}

void accelsim_bridge::transfer_core_cache_stats(StatsBridge &stats) const {
  const shader_core_config *shader_config = m_gpgpu_sim->m_shader_config;

  for (unsigned cluster_id = 0; cluster_id < shader_config->n_simt_clusters;
       ++cluster_id) {
    for (unsigned core_id = 0;
         core_id < shader_config->n_simt_cores_per_cluster; ++core_id) {
      trace_shader_core_ctx *core =
          m_gpgpu_sim->m_cluster[cluster_id]->m_core[core_id];

      unsigned global_core_id =
          cluster_id * shader_config->n_simt_cores_per_cluster + core_id;
      unsigned global_cache_id =
          cluster_id * shader_config->n_simt_cores_per_cluster + core_id;
      assert(core->m_tpc == cluster_id);
      assert(core->m_sid == global_core_id);

      // L1I
      if (!shader_config->m_L1I_config.disabled() && core->m_L1I)
        transfer_cache_stats(CacheKind::L1I, global_cache_id,
                             core->m_L1I->get_stats(), stats);

      // L1T
      if (!shader_config->m_L1T_config.disabled() && core->m_ldst_unit &&
          core->m_ldst_unit->m_L1T)
        transfer_cache_stats(CacheKind::L1T, global_cache_id,
                             core->m_ldst_unit->m_L1T->get_stats(), stats);

      // L1D
      if (!shader_config->m_L1D_config.disabled() && core->m_ldst_unit &&
          core->m_ldst_unit->m_L1D)
        transfer_cache_stats(CacheKind::L1D, global_cache_id,
                             core->m_ldst_unit->m_L1D->get_stats(), stats);

      // L1C
      if (!shader_config->m_L1C_config.disabled() && core->m_ldst_unit &&
          core->m_ldst_unit->m_L1C)
        transfer_cache_stats(CacheKind::L1C, global_cache_id,
                             core->m_ldst_unit->m_L1C->get_stats(), stats);
    }
  }
}

/// L2 cache stats
void accelsim_bridge::transfer_l2d_stats(StatsBridge &stats) const {
  const memory_config *mem_config = m_gpgpu_sim->m_memory_config;

  if (mem_config->m_L2_config.disabled()) {
    return;
  }

  for (unsigned i = 0; i < mem_config->m_n_mem_sub_partition; i++) {
    memory_sub_partition *sub_partition =
        m_gpgpu_sim->m_memory_sub_partition[i];
    if (!sub_partition->m_config->m_L2_config.disabled()) {
      class l2_cache *l2_cache = sub_partition->m_L2cache;

      transfer_cache_stats(CacheKind::L2D, i, l2_cache->get_stats(), stats);
    };
  }
}
