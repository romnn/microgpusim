#pragma once

#include <cstddef>

class gpgpu_context;
class ptx_instruction;

void ptx_file_line_stats_add_exec_count(const ptx_instruction *pInsn);
void ptx_file_line_stats_create_exposed_latency_tracker(int n_shader_cores);
void ptx_file_line_stats_commit_exposed_latency(int sc_id, int exposed_latency);

class ptx_stats {
public:
  ptx_stats(gpgpu_context *ctx) {
    ptx_line_stats_filename = NULL;
    gpgpu_ctx = ctx;
  }
  char *ptx_line_stats_filename;
  bool enable_ptx_file_line_stats;
  gpgpu_context *gpgpu_ctx;
  // set options
  // void ptx_file_line_stats_options(option_parser_t opp);

  // output stats to a file
  void ptx_file_line_stats_write_file();
  // stat collection interface to gpgpu-sim
  void ptx_file_line_stats_add_latency(unsigned pc, unsigned latency);
  void ptx_file_line_stats_add_dram_traffic(unsigned pc, unsigned dram_traffic);
  void ptx_file_line_stats_add_smem_bank_conflict(unsigned pc,
                                                  unsigned n_way_bkconflict);
  void ptx_file_line_stats_add_uncoalesced_gmem(unsigned pc, unsigned n_access);
  void ptx_file_line_stats_add_inflight_memory_insn(int sc_id, unsigned pc);
  void ptx_file_line_stats_sub_inflight_memory_insn(int sc_id, unsigned pc);
  void ptx_file_line_stats_add_warp_divergence(unsigned pc,
                                               unsigned n_way_divergence);
};
