#pragma once

#include <list>
#include <unordered_map>
#include <vector>

#include "dim3.hpp"
#include "inst_trace.hpp"
#include "kernel_trace.hpp"
#include "opcode_char.hpp"
#include "trace_function_info.hpp"

class trace_parser;

// class trace_kernel_info_t : public kernel_info_t {
class trace_kernel_info_t {
 public:
  trace_kernel_info_t(dim3 gridDim, dim3 blockDim,
                      trace_function_info *m_function_info,
                      trace_parser *parser, class trace_config *config,
                      kernel_trace_t *kernel_trace_info);

  void get_next_threadblock_traces(
      std::vector<std::vector<inst_trace_t> *> threadblock_traces);

  unsigned long get_cuda_stream_id() {
    return m_kernel_trace_info->cuda_stream_id;
  }

  kernel_trace_t *get_trace_info() { return m_kernel_trace_info; }

  bool was_launched() { return m_was_launched; }

  void set_launched() { m_was_launched = true; }

  // from kernel_info.hpp
  void inc_running() { m_num_cores_running++; }
  void dec_running() {
    assert(m_num_cores_running > 0);
    m_num_cores_running--;
  }
  bool running() const { return m_num_cores_running > 0; }
  bool done() const { return no_more_ctas_to_run() && !running(); }

  class trace_function_info *entry() {
    return m_kernel_entry;
  }
  const class trace_function_info *entry() const { return m_kernel_entry; }

  size_t num_blocks() const {
    return m_grid_dim.x * m_grid_dim.y * m_grid_dim.z;
  }

  size_t threads_per_cta() const {
    return m_block_dim.x * m_block_dim.y * m_block_dim.z;
  }

  dim3 get_grid_dim() const { return m_grid_dim; }
  dim3 get_cta_dim() const { return m_block_dim; }

  void increment_cta_id() {
    increment_x_then_y_then_z(m_next_cta, m_grid_dim);
    m_next_tid.x = 0;
    m_next_tid.y = 0;
    m_next_tid.z = 0;
  }

  dim3 get_next_cta_id() const { return m_next_cta; }
  unsigned get_next_cta_id_single() const {
    return m_next_cta.x + m_grid_dim.x * m_next_cta.y +
           m_grid_dim.x * m_grid_dim.y * m_next_cta.z;
  }
  bool no_more_ctas_to_run() const {
    return (m_next_cta.x >= m_grid_dim.x || m_next_cta.y >= m_grid_dim.y ||
            m_next_cta.z >= m_grid_dim.z);
  }

  void increment_thread_id() {
    increment_x_then_y_then_z(m_next_tid, m_block_dim);
  }

  dim3 get_next_thread_id_3d() const { return m_next_tid; }

  unsigned get_next_thread_id() const {
    return m_next_tid.x + m_block_dim.x * m_next_tid.y +
           m_block_dim.x * m_block_dim.y * m_next_tid.z;
  }

  bool more_threads_in_cta() const {
    return m_next_tid.z < m_block_dim.z && m_next_tid.y < m_block_dim.y &&
           m_next_tid.x < m_block_dim.x;
  }

  unsigned get_uid() const { return m_uid; }
  std::string get_name() const { return name(); }
  std::string name() const;

 private:
  trace_kernel_info_t *m_parent_kernel;
  dim3 m_parent_ctaid;
  dim3 m_parent_tid;
  std::list<trace_kernel_info_t *> m_child_kernels;

  trace_config *m_tconfig;
  const std::unordered_map<std::string, OpcodeChar> *OpcodeMap;
  trace_parser *m_parser;
  kernel_trace_t *m_kernel_trace_info;
  bool m_was_launched;

  friend class trace_shd_warp_t;

  // from kernel_info.hpp
  class trace_function_info *m_kernel_entry;

  unsigned m_uid;

  dim3 m_grid_dim;
  dim3 m_block_dim;
  dim3 m_next_cta;
  dim3 m_next_tid;

  unsigned m_num_cores_running;

 public:
  void print_parent_info();
  bool is_finished();
  bool children_all_finished();
  void notify_parent_finished();

 public:
  unsigned long long launch_cycle;
  unsigned long long start_cycle;
  unsigned long long end_cycle;
  unsigned m_launch_latency;

  mutable bool cache_config_set;

  unsigned m_kernel_TB_latency;  // this used for any CPU-GPU kernel latency and
                                 // counted in the gpu_cycle
};
