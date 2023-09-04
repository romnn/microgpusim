#pragma once

#include <list>
#include <map>

#include "accelsim_config.hpp"
#include "accelsim_stats.hpp"
#include "rust/cxx.h"
#include "playground-sys/src/bridge/stats.rs.h"

#include "../trace_config.hpp"
#include "../trace_command.hpp"

#include "memory_partition_unit.hpp"
#include "core.hpp"
#include "cluster.hpp"

class accelsim_bridge {
 public:
  accelsim_bridge(accelsim_config config, rust::Slice<const rust::Str> argv);
  ~accelsim_bridge();

  void run_to_completion();
  void process_commands();
  void launch_kernels();
  void cycle();
  void cleanup_finished_kernel(unsigned finished_kernel_uid);

  uint64_t get_cycle() const {
    return m_gpgpu_sim->gpu_tot_sim_cycle + m_gpgpu_sim->gpu_sim_cycle;
  }
  bool active() const { return m_gpgpu_sim->active(); };
  unsigned get_finished_kernel_uid() { return m_gpgpu_sim->finished_kernel(); };
  bool limit_reached() const { return m_gpgpu_sim->cycle_insn_cta_max_hit(); };
  bool commands_left() const { return command_idx < commandlist.size(); };
  bool active_kernels() const { return kernels_info.size(); };
  bool kernels_left() const { return !kernels_info.empty(); };

  // stats transfer
  void transfer_stats(StatsBridge &stats) const;
  void transfer_dram_stats(StatsBridge &stats) const;
  void transfer_general_stats(StatsBridge &stats) const;
  void transfer_core_cache_stats(StatsBridge &stats) const;
  void transfer_l2d_stats(StatsBridge &stats) const;

  const std::vector<memory_sub_partition_bridge> &get_sub_partitions() const {
    return sub_partitions;
  }

  const std::vector<memory_partition_unit_bridge> &get_partition_units() const {
    return partition_units;
  }

  const std::vector<core_bridge> &get_cores() const { return cores; }
  const std::vector<cluster_bridge> &get_clusters() const { return clusters; }

  unsigned get_last_cluster_issue() const {
    return m_gpgpu_sim->m_last_cluster_issue;
  }

 private:
  trace_parser *tracer;
  trace_config tconfig;
  trace_gpgpu_sim *m_gpgpu_sim;
  gpgpu_context *m_gpgpu_context;

  std::shared_ptr<spdlog::logger> logger;

  std::vector<trace_command> commandlist;
  std::vector<unsigned long> busy_streams;
  std::vector<trace_kernel_info_t *> kernels_info;

  unsigned command_idx;
  unsigned window_size;
  FILE *stats_out;
  bool print_stats;
  bool accelsim_compat_mode;
  unsigned log_after_cycle;

  // for handing out references to components
  std::vector<memory_sub_partition_bridge> sub_partitions;
  std::vector<memory_partition_unit_bridge> partition_units;
  std::vector<core_bridge> cores;
  std::vector<cluster_bridge> clusters;
};

std::unique_ptr<accelsim_bridge> new_accelsim_bridge(
    accelsim_config config, rust::Slice<const rust::Str> argv);
