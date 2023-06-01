#pragma once

#include "simt_core_cluster.hpp"

class gpgpu_sim;

class trace_simt_core_cluster : public simt_core_cluster {
 public:
  trace_simt_core_cluster(class gpgpu_sim *gpu, unsigned cluster_id,
                          const shader_core_config *config,
                          const memory_config *mem_config,
                          class shader_core_stats *stats,
                          class memory_stats_t *mstats)
      : simt_core_cluster(gpu, cluster_id, config, mem_config, stats, mstats) {
    create_shader_core_ctx();
  }

  virtual void create_shader_core_ctx();
};
