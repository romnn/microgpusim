#pragma once

#include "accelsim_config.hpp"
#include "accelsim_stats.hpp"
#include "rust/cxx.h"
#include "playground/src/bridge/stats.rs.h"

#include "../trace_parser.hpp"
#include "../trace_config.hpp"
#include "../trace_command.hpp"
#include "../trace_kernel_info.hpp"

class trace_gpgpu_sim_bridge;

class accelsim_bridge {
 public:
  accelsim_bridge(accelsim_config config, rust::Slice<const rust::Str> argv);

  void run_to_completion();
  void transfer_stats(Stats &stats) const;
  // void transfer_dram_stats(Stats &stats);
  // void transfer_general_stats(Stats &stats);
  // void transfer_core_cache_stats(Stats &stats);
  // void transfer_l2d_stats(Stats &stats);

 private:
  trace_parser *tracer;
  trace_config tconfig;
  trace_gpgpu_sim_bridge *m_gpgpu_sim;
  gpgpu_context *m_gpgpu_context;

  std::vector<trace_command> commandlist;
  std::vector<unsigned long> busy_streams;
  std::vector<trace_kernel_info_t *> kernels_info;

  unsigned window_size;
  bool silent;
};

// int accelsim(accelsim_config config, rust::Slice<const rust::Str> argv, Stats
// &stats);
std::unique_ptr<accelsim_bridge> new_accelsim_bridge(
    accelsim_config config, rust::Slice<const rust::Str> argv);

// int accelsim_old(accelsim_config config, rust::Slice<const rust::Str> argv);
