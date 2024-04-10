#include "cuda_sim.hpp"

#include "checkpoint.hpp"
#include "dim3.hpp"
#include "function_info.hpp"
#include "functional_core_sim.hpp"
#include "gpgpu.hpp"
#include "gpgpu_context.hpp"
#include "gpgpu_sim.hpp"
#include "gpgpusim_ctx.hpp"
#include "kernel_info.hpp"
#include "ptx_instruction.hpp"
#include "stat.hpp"
#include "stream_manager.hpp"
#include "util.hpp"

int g_debug_execution = 0;

void cuda_sim::ptx_print_insn(address_type pc, FILE *fp) {
  std::map<unsigned, function_info *>::iterator f = g_pc_to_finfo.find(pc);
  if (f == g_pc_to_finfo.end()) {
    fprintf(fp, "<no instruction at address 0x%llx>", pc);
    return;
  }
  function_info *finfo = f->second;
  assert(finfo);
  finfo->print_insn(pc, fp);
}

std::string cuda_sim::ptx_get_insn_str(address_type pc) {
  std::map<unsigned, function_info *>::iterator f = g_pc_to_finfo.find(pc);
  if (f == g_pc_to_finfo.end()) {
#define STR_SIZE 255
    char buff[STR_SIZE];
    buff[STR_SIZE - 1] = '\0';
    snprintf(buff, STR_SIZE, "<no instruction at address 0x%llx>", pc);
    return std::string(buff);
  }
  function_info *finfo = f->second;
  assert(finfo);
  return finfo->get_insn_str(pc);
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
  if (init.find(g_ptx_kernel_count) != init.end())
    return;
  init.insert(g_ptx_kernel_count);

#define MAX_CLASS_KER 1024
  char kernelname[MAX_CLASS_KER] = "";
  if (!g_inst_classification_stat)
    g_inst_classification_stat = (void **)calloc(MAX_CLASS_KER, sizeof(void *));
  snprintf(kernelname, MAX_CLASS_KER, "Kernel %d Classification\n",
           g_ptx_kernel_count);
  assert(g_ptx_kernel_count <
         MAX_CLASS_KER); // a static limit on number of kernels increase it if
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

void cuda_sim::set_param_gpgpu_num_shaders(int num_shaders) {
  gpgpu_param_num_shaders = num_shaders;
}

kernel_info_t *cuda_sim::gpgpu_opencl_ptx_sim_init_grid(
    class function_info *entry, gpgpu_ptx_sim_arg_list_t args,
    struct dim3 gridDim, struct dim3 blockDim, gpgpu_t *gpu) {
  kernel_info_t *result =
      new kernel_info_t(gridDim, blockDim, entry, gpu->getNameArrayMapping(),
                        gpu->getNameInfoMapping());
  unsigned argcount = args.size();
  unsigned argn = 1;
  for (gpgpu_ptx_sim_arg_list_t::iterator a = args.begin(); a != args.end();
       a++) {
    entry->add_param_data(argcount - argn, &(*a));
    argn++;
  }
  entry->finalize(result->get_param_memory());
  g_ptx_kernel_count++;
  fflush(stdout);

  return result;
}

void cuda_sim::gpgpu_ptx_sim_register_const_variable(void *hostVar,
                                                     const char *deviceName,
                                                     size_t size) {
  printf("GPGPU-Sim PTX registering constant %s (%zu bytes) to name mapping\n",
         deviceName, size);
  g_const_name_lookup[hostVar] = deviceName;
}

void cuda_sim::gpgpu_ptx_sim_register_global_variable(void *hostVar,
                                                      const char *deviceName,
                                                      size_t size) {
  printf("GPGPU-Sim PTX registering global %s hostVar to name mapping\n",
         deviceName);
  g_global_name_lookup[hostVar] = deviceName;
}

void cuda_sim::gpgpu_ptx_sim_memcpy_symbol(const char *hostVar, const void *src,
                                           size_t count, size_t offset, int to,
                                           gpgpu_t *gpu) {
  printf(
      "GPGPU-Sim PTX: starting gpgpu_ptx_sim_memcpy_symbol with hostVar 0x%p\n",
      hostVar);
  bool found_sym = false;
  memory_space_t mem_region = undefined_space;
  std::string sym_name;

  std::map<const void *, std::string>::iterator c =
      gpu->gpgpu_ctx->func_sim->g_const_name_lookup.find(hostVar);
  if (c != gpu->gpgpu_ctx->func_sim->g_const_name_lookup.end()) {
    found_sym = true;
    sym_name = c->second;
    mem_region = const_space;
  }
  std::map<const void *, std::string>::iterator g =
      gpu->gpgpu_ctx->func_sim->g_global_name_lookup.find(hostVar);
  if (g != gpu->gpgpu_ctx->func_sim->g_global_name_lookup.end()) {
    if (found_sym) {
      printf("Execution error: PTX symbol \"%s\" w/ hostVar=0x%llx is declared "
             "both const and global?\n",
             sym_name.c_str(), (unsigned long long)hostVar);
      abort();
    }
    found_sym = true;
    sym_name = g->second;
    mem_region = global_space;
  }
  if (g_globals.find(hostVar) != g_globals.end()) {
    found_sym = true;
    sym_name = hostVar;
    mem_region = global_space;
  }
  if (g_constants.find(hostVar) != g_constants.end()) {
    found_sym = true;
    sym_name = hostVar;
    mem_region = const_space;
  }

  if (!found_sym) {
    printf("Execution error: No information for PTX symbol w/ hostVar=0x%llx\n",
           (unsigned long long)hostVar);
    abort();
  } else
    printf("GPGPU-Sim PTX: gpgpu_ptx_sim_memcpy_symbol: Found PTX symbol w/ "
           "hostVar=0x%llx\n",
           (unsigned long long)hostVar);
  const char *mem_name = NULL;
  memory_space *mem = NULL;

  std::map<std::string, symbol_table *>::iterator st =
      gpgpu_ctx->ptx_parser->g_sym_name_to_symbol_table.find(sym_name.c_str());
  assert(st != gpgpu_ctx->ptx_parser->g_sym_name_to_symbol_table.end());
  symbol_table *symtab = st->second;

  symbol *sym = symtab->lookup(sym_name.c_str());
  assert(sym);
  unsigned dst = sym->get_address() + offset;
  switch (mem_region.get_type()) {
  case const_space:
    mem = gpu->get_global_memory();
    mem_name = "const";
    break;
  case global_space:
    mem = gpu->get_global_memory();
    mem_name = "global";
    break;
  default:
    abort();
  }
  printf(
      "GPGPU-Sim PTX: gpgpu_ptx_sim_memcpy_symbol: copying %s memory %zu bytes "
      "%s symbol %s+%zu @0x%x ...\n",
      mem_name, count, (to ? " to " : "from"), sym_name.c_str(), offset, dst);
  for (unsigned n = 0; n < count; n++) {
    if (to)
      mem->write(dst + n, 1, ((char *)src) + n, NULL, NULL);
    else
      mem->read(dst + n, 1, ((char *)src) + n);
  }
  fflush(stdout);
}

const struct gpgpu_ptx_sim_info *
ptx_sim_kernel_info(const function_info *kernel) {
  return kernel->get_kernel_info();
}

unsigned max_cta(const struct gpgpu_ptx_sim_info *kernel_info,
                 unsigned threads_per_cta, unsigned int warp_size,
                 unsigned int n_thread_per_shader,
                 unsigned int gpgpu_shmem_size,
                 unsigned int gpgpu_shader_registers,
                 unsigned int max_cta_per_core) {
  unsigned int padded_cta_size = threads_per_cta;
  if (padded_cta_size % warp_size)
    padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);
  unsigned int result_thread = n_thread_per_shader / padded_cta_size;

  unsigned int result_shmem = (unsigned)-1;
  if (kernel_info->smem > 0)
    result_shmem = gpgpu_shmem_size / kernel_info->smem;
  unsigned int result_regs = (unsigned)-1;
  if (kernel_info->regs > 0)
    result_regs = gpgpu_shader_registers /
                  (padded_cta_size * ((kernel_info->regs + 3) & ~3));
  printf("padded cta size is %d and %d and %d", padded_cta_size,
         kernel_info->regs, ((kernel_info->regs + 3) & ~3));
  // Limit by CTA
  unsigned int result_cta = max_cta_per_core;

  unsigned result = result_thread;
  result = gs_min2(result, result_shmem);
  result = gs_min2(result, result_regs);
  result = gs_min2(result, result_cta);

  printf("GPGPU-Sim uArch: CTA/core = %u, limited by:", result);
  if (result == result_thread)
    printf(" threads");
  if (result == result_shmem)
    printf(" shmem");
  if (result == result_regs)
    printf(" regs");
  if (result == result_cta)
    printf(" cta_limit");
  printf("\n");

  return result;
}

/*!
This function simulates the CUDA code functionally, it takes a kernel_info_t
parameter which holds the data for the CUDA kernel to be executed
!*/
void cuda_sim::gpgpu_cuda_ptx_sim_main_func(kernel_info_t &kernel,
                                            bool openCL) {
  printf(
      "GPGPU-Sim: Performing Functional Simulation, executing kernel %s...\n",
      kernel.name().c_str());

  // using a shader core object for book keeping, it is not needed but as most
  // function built for performance simulation need it we use it here
  // extern gpgpu_sim *g_the_gpu;
  // before we execute, we should do PDOM analysis for functional simulation
  // scenario.
  function_info *kernel_func_info = kernel.entry();
  const struct gpgpu_ptx_sim_info *kernel_info =
      ptx_sim_kernel_info(kernel_func_info);
  checkpoint *g_checkpoint;
  g_checkpoint = new checkpoint();

  if (kernel_func_info->is_pdom_set()) {
    printf("GPGPU-Sim PTX: PDOM analysis already done for %s \n",
           kernel.name().c_str());
  } else {
    printf("GPGPU-Sim PTX: finding reconvergence points for \'%s\'...\n",
           kernel.name().c_str());
    kernel_func_info->do_pdom();
    kernel_func_info->set_pdom();
  }

  unsigned max_cta_tot = max_cta(
      kernel_info, kernel.threads_per_cta(),
      gpgpu_ctx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()->warp_size,
      gpgpu_ctx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()
          ->n_thread_per_shader,
      gpgpu_ctx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()
          ->gpgpu_shmem_size,
      gpgpu_ctx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()
          ->gpgpu_shader_registers,
      gpgpu_ctx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()
          ->max_cta_per_core);
  printf("Max CTA : %d\n", max_cta_tot);

  int cp_op = gpgpu_ctx->the_gpgpusim->g_the_gpu->checkpoint_option;
  int cp_kernel = gpgpu_ctx->the_gpgpusim->g_the_gpu->checkpoint_kernel;
  cp_count = gpgpu_ctx->the_gpgpusim->g_the_gpu->checkpoint_insn_Y;
  cp_cta_resume = gpgpu_ctx->the_gpgpusim->g_the_gpu->checkpoint_CTA_t;
  int cta_launched = 0;

  // we excute the kernel one CTA (Block) at the time, as synchronization
  // functions work block wise
  while (!kernel.no_more_ctas_to_run()) {
    unsigned temp = kernel.get_next_cta_id_single();

    if (cp_op == 0 ||
        (cp_op == 1 && cta_launched < cp_cta_resume &&
         kernel.get_uid() == cp_kernel) ||
        kernel.get_uid() < cp_kernel) // just fro testing
    {
      functionalCoreSim cta(
          &kernel, gpgpu_ctx->the_gpgpusim->g_the_gpu,
          gpgpu_ctx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()->warp_size);
      cta.execute(cp_count, temp);

#if (CUDART_VERSION >= 5000)
      gpgpu_ctx->device_runtime->launch_all_device_kernels();
#endif
    } else {
      kernel.increment_cta_id();
    }
    cta_launched++;
  }

  if (cp_op == 1) {
    char f1name[2048];
    snprintf(f1name, 2048, "checkpoint_files/global_mem_%d.txt",
             kernel.get_uid());
    g_checkpoint->store_global_mem(
        gpgpu_ctx->the_gpgpusim->g_the_gpu->get_global_memory(), f1name,
        (char *)"%08x");
  }

  // registering this kernel as done

  // openCL kernel simulation calls don't register the kernel so we don't
  // register its exit
  if (!openCL) {
    // extern stream_manager *g_stream_manager;
    gpgpu_ctx->the_gpgpusim->g_stream_manager->register_finished_kernel(
        kernel.get_uid());
  }

  //******PRINTING*******
  printf("GPGPU-Sim: Done functional simulation (%u instructions simulated).\n",
         g_ptx_sim_num_insn);
  if (gpgpu_ptx_instruction_classification) {
    StatDisp(g_inst_classification_stat[g_ptx_kernel_count]);
    StatDisp(g_inst_op_classification_stat[g_ptx_kernel_count]);
  }

  // time_t variables used to calculate the total simulation time
  // the start time of simulation is hold by the global variable
  // g_simulation_starttime g_simulation_starttime is initilized by
  // gpgpu_ptx_sim_init_perf() in gpgpusim_entrypoint.cc upon starting gpgpu-sim
  time_t end_time, elapsed_time, days, hrs, minutes, sec;
  end_time = time((time_t *)NULL);
  elapsed_time =
      MAX(end_time - gpgpu_ctx->the_gpgpusim->g_simulation_starttime, 1);

  // calculating and printing simulation time in terms of days, hours, minutes
  // and seconds
  days = elapsed_time / (3600 * 24);
  hrs = elapsed_time / 3600 - 24 * days;
  minutes = elapsed_time / 60 - 60 * (hrs + 24 * days);
  sec = elapsed_time - 60 * (minutes + 60 * (hrs + 24 * days));

  fflush(stderr);
  printf(
      "\n\ngpgpu_simulation_time = %u days, %u hrs, %u min, %u sec (%u sec)\n",
      (unsigned)days, (unsigned)hrs, (unsigned)minutes, (unsigned)sec,
      (unsigned)elapsed_time);
  printf("gpgpu_simulation_rate = %u (inst/sec)\n",
         (unsigned)(g_ptx_sim_num_insn / elapsed_time));
  fflush(stdout);
}

struct rec_pts cuda_sim::find_reconvergence_points(function_info *finfo) {
  rec_pts tmp;
  std::map<function_info *, rec_pts>::iterator r = g_rpts.find(finfo);

  if (r == g_rpts.end()) {
    int num_recon = finfo->get_num_reconvergence_pairs();

    gpgpu_recon_t *kernel_recon_points =
        (struct gpgpu_recon_t *)calloc(num_recon, sizeof(struct gpgpu_recon_t));
    finfo->get_reconvergence_pairs(kernel_recon_points);
    printf("GPGPU-Sim PTX: reconvergence points for %s...\n",
           finfo->get_name().c_str());
    for (int i = 0; i < num_recon; i++) {
      printf("GPGPU-Sim PTX: %2u (potential) branch divergence @ ", i + 1);
      kernel_recon_points[i].source_inst->print_insn();
      printf("\n");
      printf("GPGPU-Sim PTX:    immediate post dominator      @ ");
      if (kernel_recon_points[i].target_inst)
        kernel_recon_points[i].target_inst->print_insn();
      printf("\n");
    }
    printf("GPGPU-Sim PTX: ... end of reconvergence points for %s\n",
           finfo->get_name().c_str());

    tmp.s_kernel_recon_points = kernel_recon_points;
    tmp.s_num_recon = num_recon;
    g_rpts[finfo] = tmp;
  } else {
    tmp = r->second;
  }
  return tmp;
}

address_type cuda_sim::get_converge_point(address_type pc) {
  // the branch could encode the reconvergence point and/or a bit that indicates
  // the reconvergence point is the return PC on the call stack in the case the
  // branch has no immediate postdominator in the function (i.e., due to
  // multiple return points).

  std::map<unsigned, function_info *>::iterator f = g_pc_to_finfo.find(pc);
  assert(f != g_pc_to_finfo.end());
  function_info *finfo = f->second;
  rec_pts tmp = find_reconvergence_points(finfo);

  int i = 0;
  for (; i < tmp.s_num_recon; ++i) {
    if (tmp.s_kernel_recon_points[i].source_pc == pc) {
      if (tmp.s_kernel_recon_points[i].target_pc == (unsigned)-2) {
        return RECONVERGE_RETURN_PC;
      } else {
        return tmp.s_kernel_recon_points[i].target_pc;
      }
    }
  }
  return NO_BRANCH_DIVERGENCE;
}
