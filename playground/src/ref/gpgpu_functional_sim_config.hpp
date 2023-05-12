#pragma once

#include "option_parser.hpp"

class gpgpu_functional_sim_config {
public:
  void reg_options(class OptionParser *opp);

  void ptx_set_tex_cache_linesize(unsigned linesize);

  unsigned get_forced_max_capability() const {
    return m_ptx_force_max_capability;
  }
  bool convert_to_ptxplus() const { return m_ptx_convert_to_ptxplus; }
  bool use_cuobjdump() const { return m_ptx_use_cuobjdump; }
  bool experimental_lib_support() const { return m_experimental_lib_support; }

  int get_ptx_inst_debug_to_file() const { return g_ptx_inst_debug_to_file; }
  const char *get_ptx_inst_debug_file() const { return g_ptx_inst_debug_file; }
  // const char *get_mem_debug_file() const { return g_mem_debug_file; }
  int get_ptx_inst_debug_thread_uid() const {
    return g_ptx_inst_debug_thread_uid;
  }
  unsigned get_texcache_linesize() const { return m_texcache_linesize; }
  int get_checkpoint_option() const { return checkpoint_option; }
  int get_checkpoint_kernel() const { return checkpoint_kernel; }
  int get_checkpoint_CTA() const { return checkpoint_CTA; }
  int get_resume_option() const { return resume_option; }
  int get_resume_kernel() const { return resume_kernel; }
  int get_resume_CTA() const { return resume_CTA; }
  int get_checkpoint_CTA_t() const { return checkpoint_CTA_t; }
  int get_checkpoint_insn_Y() const { return checkpoint_insn_Y; }

private:
  // PTX options
  int m_ptx_convert_to_ptxplus;
  int m_ptx_use_cuobjdump;
  int m_experimental_lib_support;
  unsigned m_ptx_force_max_capability;
  int checkpoint_option;
  int checkpoint_kernel;
  int checkpoint_CTA;
  unsigned resume_option;
  unsigned resume_kernel;
  unsigned resume_CTA;
  unsigned checkpoint_CTA_t;
  int checkpoint_insn_Y;
  int g_ptx_inst_debug_to_file;
  char *g_ptx_inst_debug_file;
  // char *g_mem_debug_file;
  int g_ptx_inst_debug_thread_uid;

  unsigned m_texcache_linesize;
};
