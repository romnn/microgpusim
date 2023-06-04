#pragma once

#include <string.h>

#include "gpgpu_context.hpp"
#include "gpgpu_functional_sim_config.hpp"
#include "memory_config.hpp"
#include "option_parser.hpp"
#include "power_config.hpp"
#include "shader_core_config.hpp"
#include "trace.hpp"

// constants for statistics printouts
#define GPU_RSTAT_SHD_INFO 0x1
#define GPU_RSTAT_BW_STAT 0x2
#define GPU_RSTAT_WARP_DIS 0x4
#define GPU_RSTAT_DWF_MAP 0x8
#define GPU_RSTAT_L1MISS 0x10
#define GPU_RSTAT_PDOM 0x20
#define GPU_RSTAT_SCHED 0x40
#define GPU_MEMLATSTAT_MC 0x2

// constants for configuring merging of coalesced scatter-gather requests
#define TEX_MSHR_MERGE 0x4
#define CONST_MSHR_MERGE 0x2
#define GLOBAL_MSHR_MERGE 0x1

// clock constants
#define MhZ *1000000

#define CREATELOG 111
#define SAMPLELOG 222
#define DUMPLOG 333

class gpgpu_sim_config : public power_config,
                         public gpgpu_functional_sim_config {
public:
  gpgpu_sim_config(gpgpu_context *ctx)
      : m_shader_config(ctx), m_memory_config(ctx) {
    m_valid = false;
    gpgpu_ctx = ctx;
  }
  void reg_options(class OptionParser *opp);
  void init() {

    gpu_stat_sample_freq = 10000;
    gpu_runtime_stat_flag = 0;
    sscanf(gpgpu_runtime_stat, "%d:%x", &gpu_stat_sample_freq,
           &gpu_runtime_stat_flag);
    m_shader_config.init();
    ptx_set_tex_cache_linesize(m_shader_config.m_L1T_config.get_line_sz());
    m_memory_config.init();
    init_clock_domains();
    power_config::init();
    Trace::init();

    // initialize file name if it is not set
    time_t curr_time;
    time(&curr_time);
    char *date = ctime(&curr_time);
    char *s = date;
    while (*s) {
      if (*s == ' ' || *s == '\t' || *s == ':')
        *s = '-';
      if (*s == '\n' || *s == '\r')
        *s = 0;
      s++;
    }
    char buf[1024];
    snprintf(buf, 1024, "gpgpusim_visualizer__%s.log.gz", date);
    g_visualizer_filename = strdup(buf);

    m_valid = true;
  }

  unsigned get_core_freq() const { return core_freq; }
  unsigned num_shader() const { return m_shader_config.num_shader(); }
  unsigned num_cluster() const { return m_shader_config.n_simt_clusters; }
  unsigned get_max_concurrent_kernel() const { return max_concurrent_kernel; }
  unsigned checkpoint_option;

  size_t stack_limit() const { return stack_size_limit; }
  size_t heap_limit() const { return heap_size_limit; }
  size_t sync_depth_limit() const { return runtime_sync_depth_limit; }
  size_t pending_launch_count_limit() const {
    return runtime_pending_launch_count_limit;
  }

  bool flush_l1() const { return gpgpu_flush_l1_cache; }

private:
  void init_clock_domains(void);

  // backward pointer
  class gpgpu_context *gpgpu_ctx;
  bool m_valid;
  shader_core_config m_shader_config;
  memory_config m_memory_config;
  // clock domains - frequency
  double core_freq;
  double icnt_freq;
  double dram_freq;
  double l2_freq;
  double core_period;
  double icnt_period;
  double dram_period;
  double l2_period;

  // GPGPU-Sim timing model options
  unsigned long long gpu_max_cycle_opt;
  unsigned long long gpu_max_insn_opt;
  unsigned gpu_max_cta_opt;
  unsigned gpu_max_completed_cta_opt;

  char *gpgpu_runtime_stat;
  bool gpgpu_flush_l1_cache;
  bool gpgpu_flush_l2_cache;
  bool gpu_deadlock_detect;
  int gpgpu_frfcfs_dram_sched_queue_size;
  int gpgpu_cflog_interval;
  char *gpgpu_clock_domains;
  unsigned max_concurrent_kernel;

  // visualizer
  bool g_visualizer_enabled;
  char *g_visualizer_filename;
  int g_visualizer_zlevel;

  // statistics collection
  int gpu_stat_sample_freq;
  int gpu_runtime_stat_flag;

  // Device Limits
  size_t stack_size_limit;
  size_t heap_size_limit;
  size_t runtime_sync_depth_limit;
  size_t runtime_pending_launch_count_limit;

  // gpu compute capability options
  unsigned int gpgpu_compute_capability_major;
  unsigned int gpgpu_compute_capability_minor;
  unsigned long long liveness_message_freq;

  friend class trace_gpgpu_sim;
};
