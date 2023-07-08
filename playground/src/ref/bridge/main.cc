#include <iostream>

#include "../gpgpu_context.hpp"
#include "../icnt_wrapper.hpp"
#include "../option_parser.hpp"
#include "../stream_manager.hpp"
#include "../trace_config.hpp"
#include "../trace_gpgpu_sim.hpp"
#include "../trace_kernel_info.hpp"
#include "../trace_parser.hpp"

#include "../cache_stats.hpp"
#include "../cache_stats.hpp"
#include "../cache_sub_stats.hpp"
#include "../memory_sub_partition.hpp"
#include "../cache_reservation_fail_reason.hpp"
#include "../trace_simt_core_cluster.hpp"
#include "../trace_shader_core_ctx.hpp"
#include "../read_only_cache.hpp"
#include "../ldst_unit.hpp"
#include "../tex_cache.hpp"
#include "../l1_cache.hpp"

#include "main.hpp"
#include "playground/src/bridge/main.rs.h"

class trace_gpgpu_sim_bridge : public trace_gpgpu_sim {
 public:
  using trace_gpgpu_sim::trace_gpgpu_sim;

  void transfer_stats(AccelsimStats &stats) {
    transfer_general_stats(stats);

    // per core cache stats
    transfer_core_cache_stats(stats);
    // transfer_l1i_stats(stats);
    // transfer_l1t_stats(stats);
    // transfer_l1d_stats(stats);
    // transfer_l1c_stats(stats);

    // l2 data cache stats
    transfer_l2d_stats(stats);
  }

  void transfer_general_stats(AccelsimStats &stats) {
    // see: void trace_gpgpu_sim::gpu_print_stat() {

    // stats.set_global_u64("gpu_sim_cycle", gpu_sim_cycle);
    // stats.set_global_u64("gpu_sim_insn", gpu_sim_insn);
    // stats.set_global_float("gpu_ipc", (float)gpu_sim_insn / gpu_sim_cycle);
    // stats.set_global_u64("gpu_tot_sim_cycle",
    //                      gpu_tot_sim_cycle + gpu_sim_cycle);
    // stats.set_global_u64("gpu_tot_sim_insn", gpu_tot_sim_insn +
    // gpu_sim_insn);
    //
    // stats.set_global_float("gpu_tot_ipc",
    //                        (float)(gpu_tot_sim_insn + gpu_sim_insn) /
    //                            (gpu_tot_sim_cycle + gpu_sim_cycle));
    //
    // stats.set_global_u64("gpu_tot_issued_cta",
    //                      gpu_tot_issued_cta + m_total_cta_launched);

    // see: m_shader_stats->print(stdout);
    // stats.set_num_stall_shared_mem(m_shader_stats->gpgpu_n_stall_shd_mem);
    stats.set_num_mem_read_local(m_shader_stats->gpgpu_n_mem_read_local);
    stats.set_num_mem_write_local(m_shader_stats->gpgpu_n_mem_write_local);
    stats.set_num_mem_read_global(m_shader_stats->gpgpu_n_mem_read_global);
    stats.set_num_mem_write_global(m_shader_stats->gpgpu_n_mem_write_global);
    stats.set_num_mem_texture(m_shader_stats->gpgpu_n_mem_texture);
    stats.set_num_mem_const(m_shader_stats->gpgpu_n_mem_const);

    stats.set_num_load_instructions(m_shader_stats->gpgpu_n_load_insn);
    stats.set_num_store_instructions(m_shader_stats->gpgpu_n_store_insn);
    stats.set_num_shared_mem_instructions(m_shader_stats->gpgpu_n_shmem_insn);
    stats.set_num_sstarr_instructions(m_shader_stats->gpgpu_n_sstarr_insn);
    stats.set_num_texture_instructions(m_shader_stats->gpgpu_n_tex_insn);
    stats.set_num_const_instructions(m_shader_stats->gpgpu_n_const_insn);
    stats.set_num_param_instructions(m_shader_stats->gpgpu_n_param_insn);

    //   fprintf(fout, "gpgpu_n_shmem_bkconflict = %d\n",
    //   gpgpu_n_shmem_bkconflict); fprintf(fout, "gpgpu_n_cache_bkconflict =
    //   %d\n", gpgpu_n_cache_bkconflict);
    //
    //   fprintf(fout, "gpgpu_n_intrawarp_mshr_merge = %d\n",
    //           gpgpu_n_intrawarp_mshr_merge);
    //   fprintf(fout, "gpgpu_n_cmem_portconflict = %d\n",
    //   gpgpu_n_cmem_portconflict);
  }

  // /// L1 data stats
  // void transfer_l1d_stats(AccelsimStats &stats) {
  //   if (m_shader_config->m_L1D_config.disabled()) {
  //     return;
  //   }
  //   struct cache_sub_stats total_css;
  //   struct cache_sub_stats css;
  //
  //   total_css.clear();
  //   css.clear();
  //   // fprintf(fout, "L1D_cache:\n");
  //   // for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
  //   //   m_cluster[i]->get_L1D_sub_stats(css);
  //   //
  //   //   fprintf(stdout,
  //   //           "\tL1D_cache_core[%d]: Access = %llu, Miss = %llu, Miss_rate
  //   =
  //   //           "
  //   //           "%.3lf, Pending_hits = %llu, Reservation_fails = %llu\n",
  //   //           i, css.accesses, css.misses,
  //   //           (double)css.misses / (double)css.accesses, css.pending_hits,
  //   //           css.res_fails);
  //   //
  //   //   total_css += css;
  //   // }
  //   // fprintf(fout, "\tL1D_total_cache_accesses = %llu\n",
  //   total_css.accesses);
  //   // fprintf(fout, "\tL1D_total_cache_misses = %llu\n", total_css.misses);
  //   // if (total_css.accesses > 0) {
  //   //   fprintf(fout, "\tL1D_total_cache_miss_rate = %.4lf\n",
  //   //           (double)total_css.misses / (double)total_css.accesses);
  //   // }
  //   // fprintf(fout, "\tL1D_total_cache_pending_hits = %llu\n",
  //   //         total_css.pending_hits);
  //   // fprintf(fout, "\tL1D_total_cache_reservation_fails = %llu\n",
  //   //         total_css.res_fails);
  //   // total_css.print_port_stats(fout, "\tL1D_cache");
  // }

  // /// L1 const stats
  // void transfer_l1c_stats(AccelsimStats &stats) {
  //   if (m_shader_config->m_L1C_config.disabled()) {
  //     return;
  //   }
  //   struct cache_sub_stats total_css;
  //   struct cache_sub_stats css;
  //
  //   total_css.clear();
  //   css.clear();
  //   // fprintf(fout, "L1C_cache:\n");
  //   // for (unsigned i = 0; i < m_shader_config->n_simt_clusters; ++i) {
  //   //   m_cluster[i]->get_L1C_sub_stats(css);
  //   //   total_css += css;
  //   // }
  //   // fprintf(fout, "\tL1C_total_cache_accesses = %llu\n",
  //   total_css.accesses);
  //   // fprintf(fout, "\tL1C_total_cache_misses = %llu\n", total_css.misses);
  //   // if (total_css.accesses > 0) {
  //   //   fprintf(fout, "\tL1C_total_cache_miss_rate = %.4lf\n",
  //   //           (double)total_css.misses / (double)total_css.accesses);
  //   // }
  //   // fprintf(fout, "\tL1C_total_cache_pending_hits = %llu\n",
  //   //         total_css.pending_hits);
  //   // fprintf(fout, "\tL1C_total_cache_reservation_fails = %llu\n",
  //   //         total_css.res_fails);
  // }

  // /// L1 texture stats
  // void transfer_l1t_stats(AccelsimStats &stats) {
  //   if (m_shader_config->m_L1T_config.disabled()) {
  //     return;
  //   }
  //   struct cache_sub_stats total_css;
  //   struct cache_sub_stats css;
  //   // total_css.clear();
  //   // css.clear();
  //   // fprintf(fout, "L1T_cache:\n");
  //   // for (unsigned i = 0; i < m_shader_config->n_simt_clusters; ++i) {
  //   //   m_cluster[i]->get_L1T_sub_stats(css);
  //   //   total_css += css;
  //   // }
  //   // fprintf(fout, "\tL1T_total_cache_accesses = %llu\n",
  //   total_css.accesses);
  //   // fprintf(fout, "\tL1T_total_cache_misses = %llu\n", total_css.misses);
  //   // if (total_css.accesses > 0) {
  //   //   fprintf(fout, "\tL1T_total_cache_miss_rate = %.4lf\n",
  //   //           (double)total_css.misses / (double)total_css.accesses);
  //   // }
  //   // fprintf(fout, "\tL1T_total_cache_pending_hits = %llu\n",
  //   //         total_css.pending_hits);
  //   // fprintf(fout, "\tL1T_total_cache_reservation_fails = %llu\n",
  //   //         total_css.res_fails);
  // }

  void transfer_core_cache_stats(AccelsimStats &stats) {
    for (unsigned cluster_id = 0; cluster_id < m_shader_config->n_simt_clusters;
         ++cluster_id) {
      for (unsigned core_id = 0;
           core_id < m_shader_config->n_simt_cores_per_cluster; ++core_id) {
        trace_shader_core_ctx *core = m_cluster[cluster_id]->m_core[core_id];

        unsigned global_cache_id = cluster_id * +core_id;
        assert(core->m_tpc == cluster_id);
        assert(core->m_sid == core_id);

        // L1I
        if (!m_shader_config->m_L1I_config.disabled() && core->m_L1I)
          transfer_cache_stats(CacheKind::L1I, global_cache_id,
                               core->m_L1I->get_stats(), stats);

        // L1T
        if (!m_shader_config->m_L1T_config.disabled() && core->m_ldst_unit &&
            core->m_ldst_unit->m_L1T)
          transfer_cache_stats(CacheKind::L1T, global_cache_id,
                               core->m_ldst_unit->m_L1T->get_stats(), stats);

        // L1D
        if (!m_shader_config->m_L1D_config.disabled() && core->m_ldst_unit &&
            core->m_ldst_unit->m_L1D)
          transfer_cache_stats(CacheKind::L1D, global_cache_id,
                               core->m_ldst_unit->m_L1D->get_stats(), stats);

        // L1C
        if (!m_shader_config->m_L1C_config.disabled() && core->m_ldst_unit &&
            core->m_ldst_unit->m_L1C)
          transfer_cache_stats(CacheKind::L1C, global_cache_id,
                               core->m_ldst_unit->m_L1C->get_stats(), stats);
      }
    }
  }

  // /// L1 instructions stats
  // void transfer_l1i_stats(AccelsimStats &stats) {
  //   if (m_shader_config->m_L1I_config.disabled()) {
  //     return;
  //   }
  //
  //   // struct cache_sub_stats total_css;
  //   // struct cache_sub_stats css;
  //   // total_css.clear();
  //   // css.clear();
  //   // fprintf(fout, "\n========= Core cache stats =========\n");
  //   // fprintf(fout, "L1I_cache:\n");
  //   for (unsigned cluster_id = 0; cluster_id <
  //   m_shader_config->n_simt_clusters;
  //        ++cluster_id) {
  //     // m_cluster[i]->get_L1I_sub_stats(css);
  //     // total_css += css;
  //
  //     for (unsigned core_id = 0;
  //          core_id < m_shader_config->n_simt_cores_per_cluster; ++core_id) {
  //       trace_shader_core_ctx *core = m_cluster[cluster_id]->m_core[core_id];
  //       // m_core[i]->get_L1I_sub_stats(temp_css);
  //       // total_css += temp_css;
  //       //
  //       // if (core->m_L1I) core->m_L1I->get_sub_stats(css);
  //
  //       unsigned global_cache_id = cluster_id * +core_id;
  //       assert(core->m_tpc == cluster_id);
  //       assert(core->m_sid == core_id);
  //
  //       if (core->m_L1I)
  //         transfer_cache_stats(CacheKind::L1I, global_cache_id,
  //                              core->m_L1I->get_stats(), stats);
  //     }
  //   }
  //   // fprintf(fout, "\tL1I_total_cache_accesses = %llu\n",
  //   total_css.accesses);
  //   // fprintf(fout, "\tL1I_total_cache_misses = %llu\n", total_css.misses);
  //   // if (total_css.accesses > 0) {
  //   //   fprintf(fout, "\tL1I_total_cache_miss_rate = %.4lf\n",
  //   //           (double)total_css.misses / (double)total_css.accesses);
  //   // }
  //   // fprintf(fout, "\tL1I_total_cache_pending_hits = %llu\n",
  //   //         total_css.pending_hits);
  //   // fprintf(fout, "\tL1I_total_cache_reservation_fails = %llu\n",
  //   //         total_css.res_fails);
  // }

  void transfer_cache_stats(CacheKind cache, unsigned cache_id,
                            const cache_stats &stats, AccelsimStats &out) {
    for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
      for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
        out.add_accesses(cache, cache_id, type, status, false,
                         (stats)(type, status, false));
      }
      for (unsigned status = 0; status < NUM_CACHE_RESERVATION_FAIL_STATUS;
           ++status) {
        out.add_accesses(cache, cache_id, type, status, true,
                         (stats)(type, status, true));
      }
    }
  }

  /// L2 cache stats
  void transfer_l2d_stats(AccelsimStats &stats) {
    if (m_memory_config->m_L2_config.disabled()) {
      return;
    }

    cache_stats l2_stats;
    struct cache_sub_stats l2_css;
    struct cache_sub_stats total_l2_css;
    l2_stats.clear();
    l2_css.clear();
    total_l2_css.clear();

    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
      if (!m_memory_sub_partition[i]->m_config->m_L2_config.disabled()) {
        class l2_cache *l2_cache = m_memory_sub_partition[i]->m_L2cache;
        // l2_cache->get_stats();
        //
        transfer_cache_stats(CacheKind::L2D, i, l2_cache->get_stats(), stats);

        // for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        //   for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS;
        //        ++status) {
        //     stats.add_accesses(CacheKind::L2D, i, type, status, false,
        //                        (l2_cache->get_stats())(type, status,
        //                        false));
        //
        //     // stats.add_accesses(type, status,
        //     //                    (l2_cache->m_stats)[type][status]);
        //     // ret(type, status, false) =
        //     //     m_stats[type][status] + cs(type, status, false);
        //   }
        //   for (unsigned status = 0; status <
        //   NUM_CACHE_RESERVATION_FAIL_STATUS;
        //        ++status) {
        //     // ret(type, status, true) =
        //     //     m_fail_stats[type][status] + cs(type, status, true);
        //     stats.add_accesses(CacheKind::L2D, i, type, status, true,
        //                        (l2_cache->get_stats())(type, status,
        //                        true));
        //   }
        // }

        // this is just different counting of the same cache_stats, see
        // sub_stats() method l2_cache->get_sub_stats(css);
      };
      // if (!m_memory_config->m_L2_config.disabled() &&
      //     m_memory_config->m_L2_config.get_num_lines()) {
      //   // L2c_print_cache_stat();
      //   printf("L2_total_cache_accesses = %llu\n", total_l2_css.accesses);
      //   printf("L2_total_cache_misses = %llu\n", total_l2_css.misses);
      //   if (total_l2_css.accesses > 0)
      //     printf("L2_total_cache_miss_rate = %.4lf\n",
      //            (double)total_l2_css.misses /
      //            (double)total_l2_css.accesses);
      //   printf("L2_total_cache_pending_hits = %llu\n",
      //          total_l2_css.pending_hits);
      //   printf("L2_total_cache_reservation_fails = %llu\n",
      //          total_l2_css.res_fails);
      //   printf("L2_total_cache_breakdown:\n");
      //   l2_stats.print_stats(stdout, "L2_cache_stats_breakdown");
      //   printf("L2_total_cache_reservation_fail_breakdown:\n");
      //   l2_stats.print_fail_stats(stdout, "L2_cache_stats_fail_breakdown");
      //   total_l2_css.print_port_stats(stdout, "L2_cache");
      // }
    }
  }
};

trace_kernel_info_t *create_kernel_info(kernel_trace_t *kernel_trace_info,
                                        gpgpu_context *m_gpgpu_context,
                                        class trace_config *config,
                                        trace_parser *parser);

void cli_configure(gpgpu_context *m_gpgpu_context, trace_config &m_config,
                   const std::vector<const char *> &argv, bool silent) {
  // register cli options
  option_parser_t opp = option_parser_create();
  m_gpgpu_context->ptx_reg_options(opp);
  m_gpgpu_context->func_sim->ptx_opcocde_latency_options(opp);

  icnt_reg_options(opp);

  m_gpgpu_context->the_gpgpusim->g_the_gpu_config =
      new gpgpu_sim_config(m_gpgpu_context);
  m_gpgpu_context->the_gpgpusim->g_the_gpu_config->reg_options(
      opp);  // register GPU microrachitecture options
  m_config.reg_options(opp);

  if (!silent) {
    fprintf(stdout, "GPGPU-Sim: Registered options:\n\n");
    option_parser_print_registered(opp, stdout);
  }

  // parse configuration options
  option_parser_cmdline(opp, argv);

  if (!silent) {
    fprintf(stdout, "GPGPU-Sim: Configuration options:\n\n");
    option_parser_print(opp, stdout);
  }

  // initialize config (parse gpu config from cli values)
  m_gpgpu_context->the_gpgpusim->g_the_gpu_config->init();

  // override some values
  g_network_mode = BOX_NET;
}

trace_gpgpu_sim_bridge *gpgpu_trace_sim_init_perf_model(
    gpgpu_context *m_gpgpu_context, trace_config &m_config,
    const accelsim_config &config, const std::vector<const char *> &argv,
    bool silent) {
  // seed random
  srand(1);

  // Set the Numeric locale to a standard locale where a decimal point is a
  // "dot" not a "comma" so it does the parsing correctly independent of the
  // system environment variables
  assert(setlocale(LC_NUMERIC, "C"));

  // configure using cli
  cli_configure(m_gpgpu_context, m_config, argv, silent);

  // TODO: configure using config
  // m_gpgpu_context->the_gpgpusim->g_the_gpu_config->configure(config);

  assert(m_gpgpu_context->the_gpgpusim->g_the_gpu_config->m_shader_config
             .n_simt_clusters == 1);
  assert(m_gpgpu_context->the_gpgpusim->g_the_gpu_config->m_shader_config
             .n_simt_cores_per_cluster == 1);
  assert(m_gpgpu_context->the_gpgpusim->g_the_gpu_config->m_shader_config
             .gpgpu_num_sched_per_core == 1);

  m_gpgpu_context->the_gpgpusim->g_the_gpu = new trace_gpgpu_sim_bridge(
      *(m_gpgpu_context->the_gpgpusim->g_the_gpu_config), m_gpgpu_context);

  m_gpgpu_context->the_gpgpusim->g_stream_manager =
      new stream_manager((m_gpgpu_context->the_gpgpusim->g_the_gpu),
                         m_gpgpu_context->func_sim->g_cuda_launch_blocking);

  m_gpgpu_context->the_gpgpusim->g_simulation_starttime = time((time_t *)NULL);

  return static_cast<class trace_gpgpu_sim_bridge *>(
      m_gpgpu_context->the_gpgpusim->g_the_gpu);
}

trace_kernel_info_t *create_kernel_info(kernel_trace_t *kernel_trace_info,
                                        gpgpu_context *m_gpgpu_context,
                                        class trace_config *config,
                                        trace_parser *parser) {
  gpgpu_ptx_sim_info info;
  info.smem = kernel_trace_info->shmem;
  info.regs = kernel_trace_info->nregs;
  dim3 gridDim(kernel_trace_info->grid_dim_x, kernel_trace_info->grid_dim_y,
               kernel_trace_info->grid_dim_z);
  dim3 blockDim(kernel_trace_info->tb_dim_x, kernel_trace_info->tb_dim_y,
                kernel_trace_info->tb_dim_z);
  trace_function_info *function_info =
      new trace_function_info(info, m_gpgpu_context);
  function_info->set_name(kernel_trace_info->kernel_name.c_str());
  trace_kernel_info_t *kernel_info = new trace_kernel_info_t(
      gridDim, blockDim, function_info, parser, config, kernel_trace_info);

  return kernel_info;
}

// int accelsim(accelsim_config config, rust::Slice<const rust::Str> argv) {
// std::unique_ptr<accelsim_stats> accelsim(accelsim_config config,
// rust::Slice<const rust::Str> argv) { AccelsimStats accelsim(accelsim_config
// config, rust::Slice<const rust::Str> argv) {
int accelsim(accelsim_config config, rust::Slice<const rust::Str> argv,
             AccelsimStats &stats) {
  std::cout << "Accel-Sim [build <box>]" << std::endl;

  bool silent = false;
#ifdef BOX
  if (std::getenv("SILENT") && strcmp(std::getenv("SILENT"), "yes") == 0) {
    silent = true;
  }
#endif

  unsigned long long cycle_limit = (unsigned long long)-1;
  if (std::getenv("CYCLES") && atoi(std::getenv("CYCLES")) > 0) {
    cycle_limit = atoi(std::getenv("CYCLES"));
  }

  std::vector<std::string> valid_argv;
  for (auto arg : argv) valid_argv.push_back(std::string(arg));

  std::vector<const char *> c_argv;
  // THIS stupid &arg here is important !!!!
  for (std::string &arg : valid_argv) c_argv.push_back(arg.c_str());

  // setup the gpu
  gpgpu_context *m_gpgpu_context = new gpgpu_context();
  trace_config tconfig;

  // init trace based performance model
  trace_gpgpu_sim_bridge *m_gpgpu_sim = gpgpu_trace_sim_init_perf_model(
      m_gpgpu_context, tconfig, config, c_argv, silent);

  m_gpgpu_sim->init();

  // init trace parser
  trace_parser tracer(tconfig.get_traces_filename());

  // parse trace config
  tconfig.parse_config();
  printf("initialization complete\n");

  // setup a rolling window with size of the max concurrent kernel executions
  bool concurrent_kernel_sm =
      m_gpgpu_sim->getShaderCoreConfig()->gpgpu_concurrent_kernel_sm;
  unsigned window_size =
      concurrent_kernel_sm
          ? m_gpgpu_sim->get_config().get_max_concurrent_kernel()
          : 1;
  assert(window_size > 0);

  // parse the list of commands issued to the GPU
  std::vector<trace_command> commandlist = tracer.parse_commandlist_file();
  std::vector<unsigned long> busy_streams;
  std::vector<trace_kernel_info_t *> kernels_info;
  kernels_info.reserve(window_size);

  unsigned i = 0;
  while (i < commandlist.size() || !kernels_info.empty()) {
    // gulp up as many commands as possible - either cpu_gpu_mem_copy
    // or kernel_launch - until the vector "kernels_info" has reached
    // the window_size or we have read every command from commandlist
    while (kernels_info.size() < window_size && i < commandlist.size()) {
      trace_kernel_info_t *kernel_info = NULL;
      if (commandlist[i].m_type == command_type::cpu_gpu_mem_copy) {
        // parse memcopy command
        size_t addre, Bcount;
        tracer.parse_memcpy_info(commandlist[i].command_string, addre, Bcount);
        std::cout << "launching memcpy command : "
                  << commandlist[i].command_string << std::endl;
        m_gpgpu_sim->perf_memcpy_to_gpu(addre, Bcount);
        i++;
      } else if (commandlist[i].m_type == command_type::kernel_launch) {
        // Read trace header info for window_size number of kernels
        kernel_trace_t *kernel_trace_info =
            tracer.parse_kernel_info(commandlist[i].command_string);
        kernel_info = create_kernel_info(kernel_trace_info, m_gpgpu_context,
                                         &tconfig, &tracer);
        kernels_info.push_back(kernel_info);
        std::cout << "Header info loaded for kernel command : "
                  << commandlist[i].command_string << std::endl;
        i++;
      } else {
        // unsupported commands will fail the simulation
        throw std::runtime_error("undefined command");
      }
    }

    // Launch all kernels within window that are on a stream that isn't
    // already running
    for (auto k : kernels_info) {
      // check if stream of kernel is busy
      bool stream_busy = false;
      for (auto s : busy_streams) {
        if (s == k->get_cuda_stream_id()) stream_busy = true;
      }
      if (!stream_busy && m_gpgpu_sim->can_start_kernel() &&
          !k->was_launched()) {
        std::cout << "launching kernel name: " << k->get_name()
                  << " uid: " << k->get_uid() << std::endl;
        m_gpgpu_sim->launch(k);
        k->set_launched();
        busy_streams.push_back(k->get_cuda_stream_id());
      }
    }

    bool active = false;
    bool sim_cycles = false;
    unsigned finished_kernel_uid = 0;

    do {
      unsigned long long cycle =
          m_gpgpu_sim->gpu_tot_sim_cycle + m_gpgpu_sim->gpu_sim_cycle;
      if (!m_gpgpu_sim->active()) break;

#ifdef BOX
      if (cycle >= cycle_limit) {
        // dont wait for kernel to complete
        // m_gpgpu_context->the_gpgpusim->g_stream_manager
        //     ->stop_all_running_kernels();
        printf("early exit after %llu cycles\n", cycle);
        fflush(stdout);
        return 0;
      }
#endif

      // performance simulation
      if (m_gpgpu_sim->active()) {
#ifdef BOX
        m_gpgpu_sim->simple_cycle();
#else
        m_gpgpu_sim->cycle();
#endif
        sim_cycles = true;
        m_gpgpu_sim->deadlock_check();
      } else {
        // stop all kernels if we reached max instructions limit
        if (m_gpgpu_sim->cycle_insn_cta_max_hit()) {
          m_gpgpu_context->the_gpgpusim->g_stream_manager
              ->stop_all_running_kernels();
          break;
        }
      }

      active = m_gpgpu_sim->active();
      finished_kernel_uid = m_gpgpu_sim->finished_kernel();
    } while (active && !finished_kernel_uid);

    // cleanup finished kernel
    if (finished_kernel_uid || m_gpgpu_sim->cycle_insn_cta_max_hit() ||
        !m_gpgpu_sim->active()) {
      trace_kernel_info_t *k = NULL;
      for (unsigned j = 0; j < kernels_info.size(); j++) {
        k = kernels_info.at(j);
        if (k->get_uid() == finished_kernel_uid ||
            m_gpgpu_sim->cycle_insn_cta_max_hit() || !m_gpgpu_sim->active()) {
          for (int l = 0; l < busy_streams.size(); l++) {
            if (busy_streams.at(l) == k->get_cuda_stream_id()) {
              busy_streams.erase(busy_streams.begin() + l);
              break;
            }
          }
          tracer.kernel_finalizer(k->get_trace_info());
          delete k->entry();
          delete k;
          kernels_info.erase(kernels_info.begin() + j);
          if (!m_gpgpu_sim->cycle_insn_cta_max_hit() && m_gpgpu_sim->active())
            break;
        }
      }
      assert(k);
      if (!silent) m_gpgpu_sim->print_stats();

      m_gpgpu_sim->transfer_stats(stats);
      // stats.set_global_u32("gpu_sim_cycle", m_gpgpu_sim->gpu_sim_cycle);
      // stats.set_global_u32("gpu_sim_insn", m_gpgpu_sim->gpu_sim_inst);
      // stats.set_global_u32("gpu_ipc", (float)m_gpgpu_sim->gpu_sim_insn /
      //                                     m_gpgpu_sim->gpu_sim_cycle);
      // stats.set_global_u32("gpu_tot_sim_cycle",
      // m_gpgpu_sim->gpu_tot_sim_cycle +
      //                                               m_gpgpu_sim->gpu_sim_cycle);
      // stats.set_global_u32("gpu_tot_sim_insn",
      // m_gpgpu_sim->gpu_tot_sim_insn
      // +
      //                                              m_gpgpu_sim->gpu_sim_insn);
      //
      // stats.set_global_u32(
      //     "gpu_tot_ipc",
      //     (float)(m_gpgpu_sim->gpu_tot_sim_insn +
      //     m_gpgpu_sim->gpu_sim_insn)
      //     /
      //         (m_gpgpu_sim->gpu_tot_sim_cycle +
      //         m_gpgpu_sim->gpu_sim_cycle));
      //
      // stats.set_global_u32(
      //     "gpu_tot_ipc",
      //     m_gpgpu_sim->gpu_tot_issued_cta +
      //     m_gpgpu_sim->m_total_cta_launched);

      // total_dram_writes
      // total_dram_reads
      // const_cache_read_total
      // const_cache_write_total
      // total_core_cache_read_total
      // total_core_cache_read_hit
      // l2_cache_read_total
      // l2_cache_read_hit
      // l2_cache_read_miss
      // l2_cache_write_total
      // l2_cache_write_miss
      // l2_cache_write_hit
    }

    if (!silent && sim_cycles) {
      m_gpgpu_sim->update_stats();
      m_gpgpu_context->print_simulation_time();
    }

    if (m_gpgpu_sim->cycle_insn_cta_max_hit()) {
      printf(
          "GPGPU-Sim: ** break due to reaching the maximum cycles (or "
          "instructions) **\n");
      fflush(stdout);
      break;
    }
  }

  // we print this message to inform the gpgpu-simulation stats_collect script
  // that we are done
  printf("GPGPU-Sim: *** simulation thread exiting ***\n");
  printf("GPGPU-Sim: *** exit detected ***\n");
  fflush(stdout);

  return 0;
}
