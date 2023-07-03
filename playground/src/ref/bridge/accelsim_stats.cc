#include "accelsim_stats.hpp"

#include "../cache_stats.hpp"
#include "../cache_sub_stats.hpp"
#include "../memory_sub_partition.hpp"
#include "../trace_gpgpu_sim.hpp"
#include <type_traits>

std::unique_ptr<accelsim_stats> trace_gpgpu_sim::get_accelsim_stats() {
  std::unique_ptr<accelsim_stats> stats = std::make_unique<accelsim_stats>();

  // L2 cache stats
  if (!m_memory_config->m_L2_config.disabled()) {
    cache_stats l2_stats;
    struct cache_sub_stats l2_css;
    struct cache_sub_stats total_l2_css;
    l2_stats.clear();
    l2_css.clear();
    total_l2_css.clear();

    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
      m_memory_sub_partition[i]->accumulate_L2cache_stats(l2_stats);
      m_memory_sub_partition[i]->get_L2cache_sub_stats(l2_css);

      // fprintf(stdout,
      //         "L2_cache_bank[%d]: Access = %llu, Miss = %llu, Miss_rate = "
      //         "%.3lf, Pending_hits = %llu, Reservation_fails = %llu\n",
      //         i, l2_css.accesses, l2_css.misses,
      //         (double)l2_css.misses / (double)l2_css.accesses,
      //         l2_css.pending_hits, l2_css.res_fails);

      total_l2_css += l2_css;
    }

    if (!m_memory_config->m_L2_config.disabled() &&
        m_memory_config->m_L2_config.get_num_lines()) {
      stats->l2_total_cache_accesses = total_l2_css.accesses;
      stats->l2_total_cache_misses = total_l2_css.misses;
      stats->l2_total_cache_miss_rate = 0.0;
      if (total_l2_css.accesses > 0)
        stats->l2_total_cache_miss_rate =
            (double)total_l2_css.misses / (double)total_l2_css.accesses;
      stats->l2_total_cache_pending_hits = total_l2_css.pending_hits;
      stats->l2_total_cache_reservation_fails = total_l2_css.res_fails;

      // l2_stats.print_stats(stdout, "L2_cache_stats_breakdown");
      // printf("L2_total_cache_reservation_fail_breakdown:\n");
      //
      // l2_stats.print_fail_stats(stdout, "L2_cache_stats_fail_breakdown");
      // total_l2_css.print_port_stats(stdout, "L2_cache");
    }
  }
  return stats;
}
