#pragma once

#include "function_info.hpp"

class trace_function_info : public function_info {
public:
  trace_function_info(const struct gpgpu_ptx_sim_info &info,
                      gpgpu_context *m_gpgpu_context)
      : function_info(0, m_gpgpu_context) {
    m_kernel_info = info;
  }

  virtual const struct gpgpu_ptx_sim_info *get_kernel_info() const {
    return &m_kernel_info;
  }

  virtual const void set_kernel_info(const struct gpgpu_ptx_sim_info &info) {
    m_kernel_info = info;
  }

  virtual ~trace_function_info() {}
};
