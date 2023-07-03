#include "cuda_sim.hpp"

#include <cstdio>
#include <cstring>

#include "gpgpu_context.hpp"
#include "option_parser.hpp"
#include "stats_wrapper.hpp"
#include "trace_function_info.hpp"

int g_debug_execution = 0;
bool g_interactive_debugger_enabled = false;

// Output debug information to file options
void cuda_sim::ptx_opcocde_latency_options(option_parser_t opp) {
  option_parser_register(
      opp, "-ptx_opcode_latency_int", OPT_CSTR, &opcode_latency_int,
      "Opcode latencies for integers <ADD,MAX,MUL,MAD,DIV,SHFL>"
      "Default 1,1,19,25,145,32",
      "1,1,19,25,145,32");
  option_parser_register(opp, "-ptx_opcode_latency_fp", OPT_CSTR,
                         &opcode_latency_fp,
                         "Opcode latencies for single precision floating "
                         "points <ADD,MAX,MUL,MAD,DIV>"
                         "Default 1,1,1,1,30",
                         "1,1,1,1,30");
  option_parser_register(opp, "-ptx_opcode_latency_dp", OPT_CSTR,
                         &opcode_latency_dp,
                         "Opcode latencies for double precision floating "
                         "points <ADD,MAX,MUL,MAD,DIV>"
                         "Default 8,8,8,8,335",
                         "8,8,8,8,335");
  option_parser_register(opp, "-ptx_opcode_latency_sfu", OPT_CSTR,
                         &opcode_latency_sfu,
                         "Opcode latencies for SFU instructions"
                         "Default 8",
                         "8");
  option_parser_register(opp, "-ptx_opcode_latency_tesnor", OPT_CSTR,
                         &opcode_latency_tensor,
                         "Opcode latencies for Tensor instructions"
                         "Default 64",
                         "64");
  option_parser_register(
      opp, "-ptx_opcode_initiation_int", OPT_CSTR, &opcode_initiation_int,
      "Opcode initiation intervals for integers <ADD,MAX,MUL,MAD,DIV,SHFL>"
      "Default 1,1,4,4,32,4",
      "1,1,4,4,32,4");
  option_parser_register(opp, "-ptx_opcode_initiation_fp", OPT_CSTR,
                         &opcode_initiation_fp,
                         "Opcode initiation intervals for single precision "
                         "floating points <ADD,MAX,MUL,MAD,DIV>"
                         "Default 1,1,1,1,5",
                         "1,1,1,1,5");
  option_parser_register(opp, "-ptx_opcode_initiation_dp", OPT_CSTR,
                         &opcode_initiation_dp,
                         "Opcode initiation intervals for double precision "
                         "floating points <ADD,MAX,MUL,MAD,DIV>"
                         "Default 8,8,8,8,130",
                         "8,8,8,8,130");
  option_parser_register(opp, "-ptx_opcode_initiation_sfu", OPT_CSTR,
                         &opcode_initiation_sfu,
                         "Opcode initiation intervals for sfu instructions"
                         "Default 8",
                         "8");
  option_parser_register(opp, "-ptx_opcode_initiation_tensor", OPT_CSTR,
                         &opcode_initiation_tensor,
                         "Opcode initiation intervals for tensor instructions"
                         "Default 64",
                         "64");
  option_parser_register(opp, "-cdp_latency", OPT_CSTR, &cdp_latency_str,
                         "CDP API latency <cudaStreamCreateWithFlags, \
cudaGetParameterBufferV2_init_perWarp, cudaGetParameterBufferV2_perKernel, \
cudaLaunchDeviceV2_init_perWarp, cudaLaunchDevicV2_perKernel>"
                         "Default 7200,8000,100,12000,1600",
                         "7200,8000,100,12000,1600");
}

void cuda_sim::ptx_print_insn(address_type pc, FILE *fp) {
  // std::map<unsigned, trace_function_info *>::iterator f =
  //     g_pc_to_finfo.find(pc);
  // if (f == g_pc_to_finfo.end()) {
  //   fprintf(fp, "<no instruction at address 0x%lx>", pc);
  //   return;
  // }
  // trace_function_info *finfo = f->second;
  // assert(finfo);
  // finfo->print_insn(pc, fp);
}

std::string cuda_sim::ptx_get_insn_str(address_type pc) {
  return "";
  //   std::map<unsigned, trace_function_info *>::iterator f =
  //   g_pc_to_finfo.find(pc); if (f == g_pc_to_finfo.end()) {
  // #define STR_SIZE 255
  //     char buff[STR_SIZE];
  //     buff[STR_SIZE - 1] = '\0';
  //     snprintf(buff, STR_SIZE, "<no instruction at address 0x%lx>", pc);
  //     return std::string(buff);
  //   }
  //   trace_function_info *finfo = f->second;
  //   assert(finfo);
  //   return finfo->get_insn_str(pc);
}

template <int activate_level>
bool cuda_sim::ptx_debug_exec_dump_cond(int thd_uid, addr_t pc) {
  if (g_debug_execution >= activate_level) {
    // check each type of debug dump constraint to filter out dumps
    if ((g_debug_thread_uid != 0) &&
        (thd_uid != (unsigned)g_debug_thread_uid)) {
      return false;
    }
    if ((g_debug_pc != 0xBEEF1518) && (pc != g_debug_pc)) {
      return false;
    }

    return true;
  }

  return false;
}

void cuda_sim::init_inst_classification_stat() {
  static std::set<unsigned> init;
  if (init.find(g_ptx_kernel_count) != init.end()) return;
  init.insert(g_ptx_kernel_count);

#define MAX_CLASS_KER 1024
  char kernelname[MAX_CLASS_KER] = "";
  if (!g_inst_classification_stat)
    g_inst_classification_stat = (void **)calloc(MAX_CLASS_KER, sizeof(void *));
  snprintf(kernelname, MAX_CLASS_KER, "Kernel %d Classification\n",
           g_ptx_kernel_count);
  assert(g_ptx_kernel_count <
         MAX_CLASS_KER);  // a static limit on number of kernels increase it if
                          // it fails!
  g_inst_classification_stat[g_ptx_kernel_count] =
      StatCreate(kernelname, 1, 20);
  if (!g_inst_op_classification_stat)
    g_inst_op_classification_stat =
        (void **)calloc(MAX_CLASS_KER, sizeof(void *));
  snprintf(kernelname, MAX_CLASS_KER, "Kernel %d OP Classification\n",
           g_ptx_kernel_count);
  g_inst_op_classification_stat[g_ptx_kernel_count] =
      StatCreate(kernelname, 1, 100);
}

// REMOVE: ptx
// kernel_info_t *cuda_sim::gpgpu_opencl_ptx_sim_init_grid(
//     class function_info *entry, gpgpu_ptx_sim_arg_list_t args,
//     struct dim3 gridDim, struct dim3 blockDim, gpgpu_t *gpu) {
//   kernel_info_t *result =
//       new kernel_info_t(gridDim, blockDim, entry, gpu->getNameArrayMapping(),
//                         gpu->getNameInfoMapping());
//   unsigned argcount = args.size();
//   unsigned argn = 1;
//   for (gpgpu_ptx_sim_arg_list_t::iterator a = args.begin(); a != args.end();
//        a++) {
//     entry->add_param_data(argcount - argn, &(*a));
//     argn++;
//   }
//   entry->finalize(result->get_param_memory());
//   g_ptx_kernel_count++;
//   fflush(stdout);
//
//   return result;
// }

// void cuda_sim::gpgpu_ptx_sim_register_const_variable(void *hostVar,
//                                                      const char *deviceName,
//                                                      size_t size) {
//   printf("GPGPU-Sim PTX registering constant %s (%zu bytes) to name
//   mapping\n",
//          deviceName, size);
//   g_const_name_lookup[hostVar] = deviceName;
// }
//
// void cuda_sim::gpgpu_ptx_sim_register_global_variable(void *hostVar,
//                                                       const char *deviceName,
//                                                       size_t size) {
//   printf("GPGPU-Sim PTX registering global %s hostVar to name mapping\n",
//          deviceName);
//   g_global_name_lookup[hostVar] = deviceName;
// }

// void cuda_sim::gpgpu_ptx_sim_memcpy_symbol(const char *hostVar, const void
// *src,
//                                            size_t count, size_t offset, int
//                                            to, gpgpu_t *gpu) {
//   printf(
//       "GPGPU-Sim PTX: starting gpgpu_ptx_sim_memcpy_symbol with hostVar
//       0x%p\n", hostVar);
//   bool found_sym = false;
//   memory_space_t mem_region = undefined_space;
//   std::string sym_name;
//
//   std::map<const void *, std::string>::iterator c =
//       gpu->gpgpu_ctx->func_sim->g_const_name_lookup.find(hostVar);
//   if (c != gpu->gpgpu_ctx->func_sim->g_const_name_lookup.end()) {
//     found_sym = true;
//     sym_name = c->second;
//     mem_region = const_space;
//   }
//   std::map<const void *, std::string>::iterator g =
//       gpu->gpgpu_ctx->func_sim->g_global_name_lookup.find(hostVar);
//   if (g != gpu->gpgpu_ctx->func_sim->g_global_name_lookup.end()) {
//     if (found_sym) {
//       printf("Execution error: PTX symbol \"%s\" w/ hostVar=0x%Lx is declared
//       "
//              "both const and global?\n",
//              sym_name.c_str(), (unsigned long long)hostVar);
//       abort();
//     }
//     found_sym = true;
//     sym_name = g->second;
//     mem_region = global_space;
//   }
//   if (g_globals.find(hostVar) != g_globals.end()) {
//     found_sym = true;
//     sym_name = hostVar;
//     mem_region = global_space;
//   }
//   if (g_constants.find(hostVar) != g_constants.end()) {
//     found_sym = true;
//     sym_name = hostVar;
//     mem_region = const_space;
//   }
//
//   if (!found_sym) {
//     printf("Execution error: No information for PTX symbol w/
//     hostVar=0x%Lx\n",
//            (unsigned long long)hostVar);
//     abort();
//   } else
//     printf("GPGPU-Sim PTX: gpgpu_ptx_sim_memcpy_symbol: Found PTX symbol w/ "
//            "hostVar=0x%Lx\n",
//            (unsigned long long)hostVar);
//   const char *mem_name = NULL;
//   memory_space *mem = NULL;
//
//   std::map<std::string, symbol_table *>::iterator st =
//       gpgpu_ctx->ptx_parser->g_sym_name_to_symbol_table.find(sym_name.c_str());
//   assert(st != gpgpu_ctx->ptx_parser->g_sym_name_to_symbol_table.end());
//   symbol_table *symtab = st->second;
//
//   symbol *sym = symtab->lookup(sym_name.c_str());
//   assert(sym);
//   unsigned dst = sym->get_address() + offset;
//   switch (mem_region.get_type()) {
//   case const_space:
//     mem = gpu->get_global_memory();
//     mem_name = "const";
//     break;
//   case global_space:
//     mem = gpu->get_global_memory();
//     mem_name = "global";
//     break;
//   default:
//     abort();
//   }
//   printf(
//       "GPGPU-Sim PTX: gpgpu_ptx_sim_memcpy_symbol: copying %s memory %zu
//       bytes "
//       "%s symbol %s+%zu @0x%x ...\n",
//       mem_name, count, (to ? " to " : "from"), sym_name.c_str(), offset,
//       dst);
//   for (unsigned n = 0; n < count; n++) {
//     if (to)
//       mem->write(dst + n, 1, ((char *)src) + n, NULL, NULL);
//     else
//       mem->read(dst + n, 1, ((char *)src) + n);
//   }
//   fflush(stdout);
// }

extern int ptx_debug;

void cuda_sim::read_sim_environment_variables() {
  ptx_debug = 0;
  g_debug_execution = 0;
  g_interactive_debugger_enabled = false;

  char *mode = getenv("PTX_SIM_MODE_FUNC");
  if (mode) sscanf(mode, "%u", &g_ptx_sim_mode);
  printf(
      "GPGPU-Sim PTX: simulation mode %d (can change with PTX_SIM_MODE_FUNC "
      "environment variable:\n",
      g_ptx_sim_mode);
  printf(
      "               1=functional simulation only, 0=detailed performance "
      "simulator)\n");
  char *dbg_inter = getenv("GPGPUSIM_DEBUG");
  if (dbg_inter && strlen(dbg_inter)) {
    printf("GPGPU-Sim PTX: enabling interactive debugger\n");
    fflush(stdout);
    g_interactive_debugger_enabled = true;
  }
  char *dbg_level = getenv("PTX_SIM_DEBUG");
  if (dbg_level && strlen(dbg_level)) {
    printf("GPGPU-Sim PTX: setting debug level to %s\n", dbg_level);
    fflush(stdout);
    sscanf(dbg_level, "%d", &g_debug_execution);
  }
  char *dbg_thread = getenv("PTX_SIM_DEBUG_THREAD_UID");
  if (dbg_thread && strlen(dbg_thread)) {
    printf("GPGPU-Sim PTX: printing debug information for thread uid %s\n",
           dbg_thread);
    fflush(stdout);
    sscanf(dbg_thread, "%d", &g_debug_thread_uid);
  }
  char *dbg_pc = getenv("PTX_SIM_DEBUG_PC");
  if (dbg_pc && strlen(dbg_pc)) {
    printf(
        "GPGPU-Sim PTX: printing debug information for instruction with PC = "
        "%s\n",
        dbg_pc);
    fflush(stdout);
    sscanf(dbg_pc, "%ld", &g_debug_pc);
  }

#if CUDART_VERSION > 1010
  g_override_embedded_ptx = false;
  char *usefile = getenv("PTX_SIM_USE_PTX_FILE");
  if (usefile && strlen(usefile)) {
    printf(
        "GPGPU-Sim PTX: overriding embedded ptx with ptx file "
        "(PTX_SIM_USE_PTX_FILE is set)\n");
    fflush(stdout);
    g_override_embedded_ptx = true;
  }
  char *blocking = getenv("CUDA_LAUNCH_BLOCKING");
  if (blocking && !strcmp(blocking, "1")) {
    g_cuda_launch_blocking = true;
  }
#else
  g_cuda_launch_blocking = true;
  g_override_embedded_ptx = true;
#endif

  if (g_debug_execution >= 40) {
    ptx_debug = 1;
  }
}
