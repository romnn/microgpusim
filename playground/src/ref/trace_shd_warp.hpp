#pragma once

#include "shd_warp.hpp"

#include "inst_trace.hpp"
#include "trace_kernel_info.hpp"

class trace_warp_inst_t;

class trace_shd_warp_t : public shd_warp_t {
public:
  trace_shd_warp_t(class shader_core_ctx *shader, unsigned warp_size)
      : shd_warp_t(shader, warp_size) {
    trace_pc = 0;
    m_kernel_info = NULL;
  }

  std::vector<inst_trace_t> warp_traces;
  const trace_warp_inst_t *get_next_trace_inst();
  void clear();
  bool trace_done();
  address_type get_start_trace_pc();
  virtual address_type get_pc();
  virtual kernel_info_t *get_kernel_info() const { return m_kernel_info; }
  void set_kernel(trace_kernel_info_t *kernel_info) {
    m_kernel_info = kernel_info;
  }

private:
  unsigned trace_pc;
  trace_kernel_info_t *m_kernel_info;
};
