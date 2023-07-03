#pragma once

#include <map>
#include <set>
#include <string>

#include "hal.hpp"
#include "option_parser.hpp"
#include "ptx_reg.hpp"
#include "rec_pts.hpp"

extern int g_debug_execution;
extern bool g_interactive_debugger_enabled;

class gpgpu_context;
class trace_function_info;
class gpgpu_t;
class trace_kernel_info_t;
class rec_pts;
class gpgpu_ptx_sim_arg_list_t;

extern void print_splash();

class cuda_sim {
 public:
  cuda_sim(gpgpu_context *ctx) {
    g_ptx_sim_num_insn = 0;
    g_ptx_kernel_count =
        -1;  // used for classification stat collection purposes
    gpgpu_param_num_shaders = 0;
    g_cuda_launch_blocking = false;
    g_inst_classification_stat = NULL;
    g_inst_op_classification_stat = NULL;
    g_assemble_code_next_pc = 0;
    g_debug_thread_uid = 0;
    g_override_embedded_ptx = false;
    ptx_tex_regs = NULL;
    g_ptx_thread_info_delete_count = 0;
    g_ptx_thread_info_uid_next = 1;
    g_debug_pc = 0xBEEF1518;
    gpgpu_ctx = ctx;
  }
  // global variables
  char *opcode_latency_int;
  char *opcode_latency_fp;
  char *opcode_latency_dp;
  char *opcode_latency_sfu;
  char *opcode_latency_tensor;
  char *opcode_initiation_int;
  char *opcode_initiation_fp;
  char *opcode_initiation_dp;
  char *opcode_initiation_sfu;
  char *opcode_initiation_tensor;
  int cp_count;
  int cp_cta_resume;
  int g_ptxinfo_error_detected;
  unsigned g_ptx_sim_num_insn;
  char *cdp_latency_str;
  int g_ptx_kernel_count;  // used for classification stat collection purposes
  std::map<const void *, std::string>
      g_global_name_lookup;  // indexed by hostVar
  std::map<const void *, std::string>
      g_const_name_lookup;  // indexed by hostVar
  int g_ptx_sim_mode;  // if non-zero run functional simulation only (i.e., no
                       // notion of a clock cycle)
  unsigned gpgpu_param_num_shaders;
  class std::map<trace_function_info *, rec_pts> g_rpts;
  bool g_cuda_launch_blocking;
  void **g_inst_classification_stat;
  void **g_inst_op_classification_stat;
  std::set<std::string> g_globals;
  std::set<std::string> g_constants;
  std::map<unsigned, trace_function_info *> g_pc_to_finfo;
  int gpgpu_ptx_instruction_classification;
  unsigned cdp_latency[5];
  unsigned g_assemble_code_next_pc;
  int g_debug_thread_uid;
  bool g_override_embedded_ptx;
  std::set<unsigned long long> g_ptx_cta_info_sm_idx_used;
  ptx_reg_t *ptx_tex_regs;
  unsigned g_ptx_thread_info_delete_count;
  unsigned g_ptx_thread_info_uid_next;
  addr_t g_debug_pc;
  // backward pointer
  class gpgpu_context *gpgpu_ctx;
  // global functions
  void ptx_opcocde_latency_options(option_parser_t opp);
  void gpgpu_cuda_ptx_sim_main_func(trace_kernel_info_t &kernel,
                                    bool openCL = false);
  void init_inst_classification_stat();

  // REMOVE: ptx
  // int gpgpu_opencl_ptx_sim_main_func(kernel_info_t *grid);
  // kernel_info_t *gpgpu_opencl_ptx_sim_init_grid(class function_info *entry,
  //                                               gpgpu_ptx_sim_arg_list_t
  //                                               args, struct dim3 gridDim,
  //                                               struct dim3 blockDim,
  //                                               gpgpu_t *gpu);
  // void gpgpu_ptx_sim_register_global_variable(void *hostVar,
  //                                             const char *deviceName,
  //                                             size_t size);
  // void gpgpu_ptx_sim_register_const_variable(void *, const char *deviceName,
  //                                            size_t size);
  // void gpgpu_ptx_sim_memcpy_symbol(const char *hostVar, const void *src,
  //                                  size_t count, size_t offset, int to,
  //                                  gpgpu_t *gpu);

  void read_sim_environment_variables();
  void set_param_gpgpu_num_shaders(int num_shaders) {
    gpgpu_param_num_shaders = num_shaders;
  }
  struct rec_pts find_reconvergence_points(trace_function_info *finfo);
  address_type get_converge_point(address_type pc);
  void ptx_print_insn(address_type pc, FILE *fp);
  std::string ptx_get_insn_str(address_type pc);
  template <int activate_level>
  bool ptx_debug_exec_dump_cond(int thd_uid, addr_t pc);
};
