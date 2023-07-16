#include "gpgpu_functional_sim_config.hpp"

void gpgpu_functional_sim_config::reg_options(class OptionParser *opp) {
  option_parser_register(opp, "-gpgpu_ptx_use_cuobjdump", OPT_BOOL,
                         &m_ptx_use_cuobjdump,
                         "Use cuobjdump to extract ptx and sass from binaries",
#if (CUDART_VERSION >= 4000)
                         "1"
#else
                         "0"
#endif
  );
  option_parser_register(opp, "-gpgpu_experimental_lib_support", OPT_BOOL,
                         &m_experimental_lib_support,
                         "Try to extract code from cuda libraries [Broken "
                         "because of unknown cudaGetExportTable]",
                         "0");
  option_parser_register(opp, "-checkpoint_option", OPT_INT32,
                         &checkpoint_option,
                         " checkpointing flag (0 = no checkpoint)", "0");
  option_parser_register(
      opp, "-checkpoint_kernel", OPT_INT32, &checkpoint_kernel,
      " checkpointing during execution of which kernel (1- 1st kernel)", "1");
  option_parser_register(
      opp, "-checkpoint_CTA", OPT_INT32, &checkpoint_CTA,
      " checkpointing after # of CTA (< less than total CTA)", "0");
  option_parser_register(opp, "-resume_option", OPT_INT32, &resume_option,
                         " resume flag (0 = no resume)", "0");
  option_parser_register(opp, "-resume_kernel", OPT_INT32, &resume_kernel,
                         " Resume from which kernel (1= 1st kernel)", "0");
  option_parser_register(opp, "-resume_CTA", OPT_INT32, &resume_CTA,
                         " resume from which CTA ", "0");
  option_parser_register(opp, "-checkpoint_CTA_t", OPT_INT32, &checkpoint_CTA_t,
                         " resume from which CTA ", "0");
  option_parser_register(opp, "-checkpoint_insn_Y", OPT_INT32,
                         &checkpoint_insn_Y, " resume from which CTA ", "0");

  option_parser_register(
      opp, "-gpgpu_ptx_convert_to_ptxplus", OPT_BOOL, &m_ptx_convert_to_ptxplus,
      "Convert SASS (native ISA) to ptxplus and run ptxplus", "0");
  option_parser_register(opp, "-gpgpu_ptx_force_max_capability", OPT_UINT32,
                         &m_ptx_force_max_capability,
                         "Force maximum compute capability", "0");
  option_parser_register(
      opp, "-gpgpu_ptx_inst_debug_to_file", OPT_BOOL, &g_ptx_inst_debug_to_file,
      "Dump executed instructions' debug information to file", "0");
  option_parser_register(
      opp, "-gpgpu_ptx_inst_debug_file", OPT_CSTR, &g_ptx_inst_debug_file,
      "Executed instructions' debug output file", "inst_debug.txt");
  option_parser_register(opp, "-gpgpu_ptx_inst_debug_thread_uid", OPT_INT32,
                         &g_ptx_inst_debug_thread_uid,
                         "Thread UID for executed instructions' debug output",
                         "1");

  /* custom
  option_parser_register(
      opp, "-gpgpu_mem_debug_file", OPT_CSTR, &g_mem_debug_file,
      "Custom memory debug output file", "mem_debug.txt");
  */
}

void gpgpu_functional_sim_config::ptx_set_tex_cache_linesize(
    unsigned linesize) {
  m_texcache_linesize = linesize;
}
