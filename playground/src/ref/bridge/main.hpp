#pragma once

#include "accelsim_config.hpp"
#include "accelsim_stats.hpp"
#include "rust/cxx.h"
#include "playground/src/bridge/stats.rs.h"

#include "./stats.hpp"
#include "../trace_parser.hpp"
#include "../trace_config.hpp"
#include "../trace_command.hpp"
#include "../trace_kernel_info.hpp"
#include "../trace_gpgpu_sim.hpp"

class accelsim_bridge {
 public:
  accelsim_bridge(accelsim_config config, rust::Slice<const rust::Str> argv);

  void run_to_completion();
  void process_commands();
  void launch_kernels();
  void cycle();
  void cleanup_finished_kernel(unsigned finished_kernel_uid);
  unsigned get_finished_kernel_uid();

  bool active() const;
  bool limit_reached() const;
  bool commands_left() const { return command_idx < commandlist.size(); };
  bool active_kernels() const { return kernels_info.size(); };
  bool kernels_left() const { return !kernels_info.empty(); };

  // stats transfer
  void transfer_stats(StatsBridge &stats) const;
  void transfer_dram_stats(StatsBridge &stats) const;
  void transfer_general_stats(StatsBridge &stats) const;
  void transfer_core_cache_stats(StatsBridge &stats) const;
  void transfer_l2d_stats(StatsBridge &stats) const;

  bool sub_partitions() const;

 private:
  trace_parser *tracer;
  trace_config tconfig;
  // trace_gpgpu_sim_bridge *m_gpgpu_sim;
  trace_gpgpu_sim *m_gpgpu_sim;
  gpgpu_context *m_gpgpu_context;

  std::vector<trace_command> commandlist;
  std::vector<unsigned long> busy_streams;
  std::vector<trace_kernel_info_t *> kernels_info;

  unsigned command_idx;
  unsigned window_size;
  bool silent;

  // bool active;
  // unsigned finished_kernel_uid;
};

// int accelsim(accelsim_config config, rust::Slice<const rust::Str> argv, Stats
// &stats);
std::unique_ptr<accelsim_bridge> new_accelsim_bridge(
    accelsim_config config, rust::Slice<const rust::Str> argv);

// int accelsim_old(accelsim_config config, rust::Slice<const rust::Str> argv);
