#pragma once

// #include "playground/src/bridge/main.rs.h"
// #include "bridge/accelsim_stats.hpp"
#include "gpgpu_sim_config.hpp"
#include "icnt_wrapper.hpp"
#include "memory_partition_unit.hpp"
#include "memory_stats.hpp"
#include "occupancy_stats.hpp"
#include "shader_core_stats.hpp"
#include "visualizer.hpp"

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

// clock domains
#define CORE 0x01
#define L2 0x02
#define DRAM 0x04
#define ICNT 0x08

class gpgpu_context;

// class trace_gpgpu_sim : public gpgpu_sim {
class trace_gpgpu_sim {
 public:
  trace_gpgpu_sim(const gpgpu_sim_config &config, gpgpu_context *ctx)
      : m_config(config) {
    gpgpu_ctx = ctx;
    // m_global_mem = new memory_space_impl<8192>("global", 64 * 1024);
    // m_tex_mem = new memory_space_impl<8192>("tex", 64 * 1024);
    // m_surf_mem = new memory_space_impl<8192>("surf", 64 * 1024);

    // m_dev_malloc = GLOBAL_HEAP_START;
    // checkpoint_option = m_function_model_config.get_checkpoint_option();
    // checkpoint_kernel = m_function_model_config.get_checkpoint_kernel();
    // checkpoint_CTA = m_function_model_config.get_checkpoint_CTA();
    // resume_option = m_function_model_config.get_resume_option();
    // resume_kernel = m_function_model_config.get_resume_kernel();
    // resume_CTA = m_function_model_config.get_resume_CTA();
    // checkpoint_CTA_t = m_function_model_config.get_checkpoint_CTA_t();
    // checkpoint_insn_Y = m_function_model_config.get_checkpoint_insn_Y();

    // initialize texture mappings to empty
    // m_NameToTextureInfo.clear();
    // m_NameToCudaArray.clear();
    // m_TextureRefToName.clear();
    // m_NameToAttribute.clear();

    // if (m_function_model_config.get_ptx_inst_debug_to_file() != 0)
    //   ptx_inst_debug_file =
    //       fopen(m_function_model_config.get_ptx_inst_debug_file(), "w");

    gpu_sim_cycle = 0;
    gpu_tot_sim_cycle = 0;

    gpgpu_ctx = ctx;
    m_shader_config = &m_config.m_shader_config;
    m_memory_config = &m_config.m_memory_config;

    // REMOVE: ptx
    // ctx->ptx_parser->set_ptx_warp_size(m_shader_config);
    // ptx_file_line_stats_create_exposed_latency_tracker(m_config.num_shader());

    // REMOVE: power
    // #ifdef GPGPUSIM_POWER_MODEL
    //   m_gpgpusim_wrapper = new gpgpu_sim_wrapper(
    //       config.g_power_simulation_enabled, config.g_power_config_name,
    //       config.g_power_simulation_mode, config.g_dvfs_enabled);
    // #endif

    m_shader_stats = new shader_core_stats(m_shader_config);
    m_memory_stats = new memory_stats_t(m_config.num_shader(), m_shader_config,
                                        m_memory_config, this);
    average_pipeline_duty_cycle = (float *)malloc(sizeof(float));
    active_sms = (float *)malloc(sizeof(float));

    // REMOVE: power
    // m_power_stats =
    //     new power_stat_t(m_shader_config, average_pipeline_duty_cycle,
    //     active_sms, m_shader_stats, m_memory_config, m_memory_stats);

    gpu_sim_insn = 0;
    gpu_tot_sim_insn = 0;
    gpu_tot_issued_cta = 0;
    gpu_completed_cta = 0;
    m_total_cta_launched = 0;
    gpu_deadlock = false;

    gpu_stall_dramfull = 0;
    gpu_stall_icnt2sh = 0;
    partiton_reqs_in_parallel = 0;
    partiton_reqs_in_parallel_total = 0;
    partiton_reqs_in_parallel_util = 0;
    partiton_reqs_in_parallel_util_total = 0;
    gpu_sim_cycle_parition_util = 0;
    gpu_tot_sim_cycle_parition_util = 0;
    partiton_replys_in_parallel = 0;
    partiton_replys_in_parallel_total = 0;

    m_memory_partition_unit =
        new memory_partition_unit *[m_memory_config->m_n_mem];
    m_memory_sub_partition =
        new memory_sub_partition *[m_memory_config->m_n_mem_sub_partition];
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      m_memory_partition_unit[i] =
          new memory_partition_unit(i, m_memory_config, m_memory_stats, this);
      for (unsigned p = 0;
           p < m_memory_config->m_n_sub_partition_per_memory_channel; p++) {
        unsigned submpid =
            i * m_memory_config->m_n_sub_partition_per_memory_channel + p;
        m_memory_sub_partition[submpid] =
            m_memory_partition_unit[i]->get_sub_partition(p);
      }
    }

    fprintf(stdout,
            "GPGPU-Sim uArch: create interconnect for %u clusters with %u "
            "memory sub partitions\n",
            m_shader_config->n_simt_clusters,
            m_memory_config->m_n_mem_sub_partition);

    icnt_wrapper_init();
    icnt_create(m_shader_config->n_simt_clusters,
                m_memory_config->m_n_mem_sub_partition);

    time_vector_create(NUM_MEM_REQ_STAT);
    fprintf(stdout,
            "GPGPU-Sim uArch: performance model initialization complete.\n");

    m_running_kernels.resize(config.max_concurrent_kernel, NULL);
    m_last_issued_kernel = 0;
    m_last_cluster_issue = m_shader_config->n_simt_clusters -
                           1;  // this causes first launch to use simt cluster
    *average_pipeline_duty_cycle = 0;
    *active_sms = 0;

    last_liveness_message_time = 0;

    // Jin: functional simulation for CDP
    m_functional_sim = false;
    m_functional_sim_kernel = NULL;

    createSIMTCluster();
  }

  virtual void createSIMTCluster();

  unsigned long long gpu_sim_cycle;
  unsigned long long gpu_tot_sim_cycle;

  void init();
  void cycle();
  void simple_cycle();
  bool active();
  bool cycle_insn_cta_max_hit() {
    return (m_config.gpu_max_cycle_opt && (gpu_tot_sim_cycle + gpu_sim_cycle) >=
                                              m_config.gpu_max_cycle_opt) ||
           (m_config.gpu_max_insn_opt &&
            (gpu_tot_sim_insn + gpu_sim_insn) >= m_config.gpu_max_insn_opt) ||
           (m_config.gpu_max_cta_opt &&
            (gpu_tot_issued_cta >= m_config.gpu_max_cta_opt)) ||
           (m_config.gpu_max_completed_cta_opt &&
            (gpu_completed_cta >= m_config.gpu_max_completed_cta_opt));
  }

  // std::unique_ptr<accelsim_stats> get_accelsim_stats();
  void gpu_print_stat();
  void print_stats();
  // void transfer_stats(AccelsimStats &stats);
  void update_stats();
  void deadlock_check();

  void inc_completed_cta() { gpu_completed_cta++; }
  void decrement_kernel_latency();

  bool is_functional_sim() { return m_functional_sim; }
  trace_kernel_info_t *get_functional_kernel() {
    return m_functional_sim_kernel;
  }

  void launch(trace_kernel_info_t *kinfo);
  bool can_start_kernel();
  unsigned finished_kernel();
  void set_kernel_done(trace_kernel_info_t *kernel);
  void stop_all_running_kernels();

  void perf_memcpy_to_gpu(size_t dst_start_addr, size_t count);
  void memcpy_to_gpu(size_t dst_start_addr, const void *src, size_t count);
  void memcpy_from_gpu(void *dst, size_t src_start_addr, size_t count);
  void memcpy_gpu_to_gpu(size_t dst, size_t src, size_t count);

  FuncCache get_cache_config(std::string kernel_name);
  void set_cache_config(std::string kernel_name, FuncCache cacheConfig);
  bool has_special_cache_config(std::string kernel_name);
  void change_cache_config(FuncCache cache_config);
  void set_cache_config(std::string kernel_name);

  int shader_clock() const;
  bool get_more_cta_left() const;
  bool hit_max_cta_count() const;

  bool kernel_more_cta_left(trace_kernel_info_t *kernel) const;
  trace_kernel_info_t *select_kernel();

  //! Get shader core configuration
  /*!
   * Returning the configuration of the shader core, used by the functional
   * simulation only so far
   */
  const shader_core_config *getShaderCoreConfig();

  const gpgpu_sim_config &get_config() const { return m_config; }

  unsigned long long gpu_sim_insn;
  unsigned long long gpu_tot_sim_insn;
  unsigned long long gpu_sim_insn_last_update;
  unsigned gpu_sim_insn_last_update_sid;
  unsigned long long last_gpu_sim_insn;
  unsigned long long last_liveness_message_time;

  occupancy_stats gpu_occupancy;
  occupancy_stats gpu_tot_occupancy;

  // performance counter for stalls due to congestion.
  unsigned int gpu_stall_dramfull;
  unsigned int gpu_stall_icnt2sh;
  unsigned long long partiton_reqs_in_parallel;
  unsigned long long partiton_reqs_in_parallel_total;
  unsigned long long partiton_reqs_in_parallel_util;
  unsigned long long partiton_reqs_in_parallel_util_total;
  unsigned long long gpu_sim_cycle_parition_util;
  unsigned long long gpu_tot_sim_cycle_parition_util;
  unsigned long long partiton_replys_in_parallel;
  unsigned long long partiton_replys_in_parallel_total;

  // backward pointer
  class gpgpu_context *gpgpu_ctx;

 private:
  // clocks
  void reinit_clock_domains(void);
  int next_clock_domain(void);
  void issue_block2core();

  void shader_print_runtime_stat(FILE *fout);
  void shader_print_l1_miss_stat(FILE *fout) const;
  void shader_print_cache_stats(FILE *fout) const;
  void shader_print_scheduler_stat(FILE *fout, bool print_dynamic_info) const;
  void visualizer_printstat();
  // void gpgpu_debug();

  // set by stream operation every time a functoinal simulation is done
  bool m_functional_sim;
  trace_kernel_info_t *m_functional_sim_kernel;

 public:
  void functional_launch(trace_kernel_info_t *k) {
    m_functional_sim = true;
    m_functional_sim_kernel = k;
  }
  void finish_functional_sim(trace_kernel_info_t *k) {
    assert(m_functional_sim);
    assert(m_functional_sim_kernel == k);
    m_functional_sim = false;
    m_functional_sim_kernel = NULL;
  }

 protected:
  class trace_simt_core_cluster **m_cluster;
  class memory_partition_unit **m_memory_partition_unit;
  class memory_sub_partition **m_memory_sub_partition;

  std::vector<std::string>
      m_executed_kernel_names;  //< names of kernel for stat printout
  std::vector<unsigned>
      m_executed_kernel_uids;  //< uids of kernel launches for stat printout
  std::string executed_kernel_info_string();  //< format the kernel information
                                              // into a string for stat printout
  std::string executed_kernel_name();
  void clear_executed_kernel_info();  //< clear the kernel information after
                                      // stat printout

  std::vector<trace_kernel_info_t *> m_running_kernels;
  unsigned m_last_issued_kernel;

  std::list<unsigned> m_finished_kernel;

  const shader_core_config *m_shader_config;
  const memory_config *m_memory_config;

  class shader_core_stats *m_shader_stats;
  class memory_stats_t *m_memory_stats;

  const gpgpu_sim_config &m_config;

  unsigned long long m_total_cta_launched;
  unsigned long long gpu_tot_issued_cta;
  unsigned gpu_completed_cta;

  bool gpu_deadlock;

  unsigned m_last_cluster_issue;
  float *average_pipeline_duty_cycle;
  float *active_sms;
  // time of next rising edge
  double core_time;
  double icnt_time;
  double dram_time;
  double l2_time;
};
