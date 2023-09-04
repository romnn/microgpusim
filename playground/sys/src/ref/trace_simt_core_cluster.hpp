#pragma once

#include <cstdio>
#include <list>

#include "spdlog/logger.h"
#include "shader_core_config.hpp"
#include "trace_gpgpu_sim.hpp"

class mem_fetch;
class cache_stats;
class trace_gpgpu_sim;
class memory_config;
class trace_shader_core_ctx;

class trace_simt_core_cluster {
 public:
  trace_simt_core_cluster(class trace_gpgpu_sim *gpu, unsigned cluster_id,
                          const shader_core_config *config,
                          const memory_config *mem_config,
                          class shader_core_stats *stats,
                          class memory_stats_t *mstats, FILE *stats_out)
      : logger(gpu->logger), stats_out(stats_out) {
    m_config = config;
    m_cta_issue_next_core = m_config->n_simt_cores_per_cluster -
                            1;  // this causes first launch to use hw cta 0
    m_cluster_id = cluster_id;
    m_gpu = gpu;
    m_stats = stats;
    m_memory_stats = mstats;
    m_mem_config = mem_config;

    create_shader_core_ctx();
  }

  virtual void create_shader_core_ctx();

  void core_cycle();
  void icnt_cycle();

  void reinit();
  unsigned issue_block2core();
  // void cache_flush();
  void cache_invalidate();
  bool icnt_injection_buffer_full(unsigned size, bool write);
  void icnt_inject_request_packet(class mem_fetch *mf);

  // for perfect memory interface
  bool response_queue_full() {
    return (m_response_fifo.size() >= m_config->n_simt_ejection_buffer_size);
  }
  void push_response_fifo(class mem_fetch *mf) {
    m_response_fifo.push_back(mf);
  }

  unsigned get_not_completed() const;

  void print_not_completed(FILE *fp) const;
  unsigned get_n_active_cta() const;
  unsigned get_n_active_sms() const;
  trace_gpgpu_sim *get_gpu() { return m_gpu; }

  float get_current_occupancy(unsigned long long &active,
                              unsigned long long &total) const;

  void print_cache_stats(FILE *fp, unsigned &dl1_accesses,
                         unsigned &dl1_misses) const;

  void get_cache_stats(cache_stats &cs) const;
  void get_L1I_sub_stats(struct cache_sub_stats &css) const;
  void get_L1D_sub_stats(struct cache_sub_stats &css) const;
  void get_L1C_sub_stats(struct cache_sub_stats &css) const;
  void get_L1T_sub_stats(struct cache_sub_stats &css) const;

  void get_icnt_stats(long &n_simt_to_mem, long &n_mem_to_simt) const;

  std::shared_ptr<spdlog::logger> logger;
  FILE *stats_out;

  friend class cluster_bridge;
  friend class accelsim_bridge;

 protected:
  unsigned m_cluster_id;
  trace_gpgpu_sim *m_gpu;

  const shader_core_config *m_config;
  shader_core_stats *m_stats;
  memory_stats_t *m_memory_stats;
  trace_shader_core_ctx **m_core;
  const memory_config *m_mem_config;

  unsigned m_cta_issue_next_core;
  std::list<unsigned> m_core_sim_order;
  std::list<mem_fetch *> m_response_fifo;
};
