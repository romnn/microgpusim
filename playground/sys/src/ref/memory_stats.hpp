#pragma once

#include "memory_config.hpp"
#include "shader_core_config.hpp"

class memory_stats_t {
 public:
  memory_stats_t(unsigned n_shader,
                 const class shader_core_config *shader_config,
                 const memory_config *mem_config,
                 const class trace_gpgpu_sim *gpu);

  unsigned memlatstat_done(class mem_fetch *mf);
  void memlatstat_read_done(class mem_fetch *mf);
  void memlatstat_dram_access(class mem_fetch *mf);
  void memlatstat_icnt2mem_pop(class mem_fetch *mf);
  void memlatstat_lat_pw();
  void memlatstat_print(unsigned n_mem, unsigned gpu_mem_n_bk);

  // void visualizer_print(gzFile visualizer_file);

  // Reset local L2 stats that are aggregated each sampling window
  void clear_L2_stats_pw();

  unsigned m_n_shader;

  const shader_core_config *m_shader_config;
  const memory_config *m_memory_config;
  const class trace_gpgpu_sim *m_gpu;

  unsigned max_mrq_latency;
  unsigned max_dq_latency;
  unsigned max_mf_latency;
  unsigned max_icnt2mem_latency;
  unsigned long long int tot_icnt2mem_latency;
  unsigned long long int tot_icnt2sh_latency;
  unsigned long long int tot_mrq_latency;
  unsigned long long int tot_mrq_num;
  unsigned max_icnt2sh_latency;
  unsigned mrq_lat_table[32];
  unsigned dq_lat_table[32];
  unsigned mf_lat_table[32];
  unsigned icnt2mem_lat_table[24];
  unsigned icnt2sh_lat_table[24];
  unsigned mf_lat_pw_table[32];  // table storing values of mf latency Per
                                 // Window
  unsigned mf_num_lat_pw;
  unsigned max_warps;
  unsigned mf_tot_lat_pw;  // total latency summed up per window. divide by
                           // mf_num_lat_pw to obtain average latency Per Window
  unsigned long long int mf_total_lat;
  unsigned long long int *
      *mf_total_lat_table;      // mf latency sums[dram chip id][bank id]
  unsigned **mf_max_lat_table;  // mf latency sums[dram chip id][bank id]
  unsigned num_mfs;
  unsigned int ***bankwrites;  // bankwrites[shader id][dram chip id][bank id]
  unsigned int ***bankreads;   // bankreads[shader id][dram chip id][bank id]
  unsigned int **totalbankwrites;    // bankwrites[dram chip id][bank id]
  unsigned int **totalbankreads;     // bankreads[dram chip id][bank id]
  unsigned int **totalbankaccesses;  // bankaccesses[dram chip id][bank id]
  unsigned int
      *num_MCBs_accessed;  // tracks how many memory controllers are accessed
                           // whenever any thread in a warp misses in cache
  unsigned int *position_of_mrq_chosen;  // position of mrq in m_queue chosen

  unsigned ***mem_access_type_stats;  // dram access type classification

  // AerialVision L2 stats
  unsigned L2_read_miss;
  unsigned L2_write_miss;
  unsigned L2_read_hit;
  unsigned L2_write_hit;

  // L2 cache stats
  unsigned int *L2_cbtoL2length;
  unsigned int *L2_cbtoL2writelength;
  unsigned int *L2_L2tocblength;
  unsigned int *L2_dramtoL2length;
  unsigned int *L2_dramtoL2writelength;
  unsigned int *L2_L2todramlength;

  // DRAM access row locality stats
  unsigned int *
      *concurrent_row_access;    // concurrent_row_access[dram chip id][bank id]
  unsigned int **num_activates;  // num_activates[dram chip id][bank id]
  unsigned int **row_access;     // row_access[dram chip id][bank id]
  unsigned int **max_conc_access2samerow;  // max_conc_access2samerow[dram chip
                                           // id][bank id]
  unsigned int **max_servicetime2samerow;  // max_servicetime2samerow[dram chip
                                           // id][bank id]

  // Power stats
  unsigned total_n_access;
  unsigned total_n_reads;
  unsigned total_n_writes;
};
