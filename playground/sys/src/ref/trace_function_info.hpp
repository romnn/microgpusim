#pragma once

// #include "function_info.hpp"
//
#include <string>

#include "gpgpu_ptx_sim_info.hpp"

class gpgpu_context;

// class trace_function_info : public function_info {
class trace_function_info {
 public:
  trace_function_info(const struct gpgpu_ptx_sim_info &info,
                      gpgpu_context *m_gpgpu_context) {
    // : function_info(0, m_gpgpu_context) {
    gpgpu_ctx = m_gpgpu_context;
    // m_uid = (gpgpu_ctx->function_info_sm_next_uid)++;
    // m_entry_point = (entry_point == 1) ? true : false;
    // m_extern = (entry_point == 2) ? true : false;
    // num_reconvergence_pairs = 0;
    // m_symtab = NULL;
    // m_assembled = false;
    // m_return_var_sym = NULL;
    // m_kernel_info.cmem = 0;
    // m_kernel_info.lmem = 0;
    // m_kernel_info.regs = 0;
    // m_kernel_info.smem = 0;
    // m_local_mem_framesize = 0;
    // m_args_aligned_size = -1;
    // pdom_done = false; // initialize it to false

    m_kernel_info = info;
  }

  void set_name(const char *name) { m_name = name; }

  virtual const struct gpgpu_ptx_sim_info *get_kernel_info() const {
    return &m_kernel_info;
  }

  virtual const void set_kernel_info(const struct gpgpu_ptx_sim_info &info) {
    m_kernel_info = info;
  }

  virtual ~trace_function_info() {}

  // from "function_info.hpp"
  std::string get_name() const { return m_name; }

  // backward pointer
  class gpgpu_context *gpgpu_ctx;

 protected:
  // Registers/shmem/etc. used (from ptxas -v), loaded from ___.ptxinfo along
  // with ___.ptx
  struct gpgpu_ptx_sim_info m_kernel_info;

 private:
  std::string m_name;
};
