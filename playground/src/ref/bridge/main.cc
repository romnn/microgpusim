#include <iostream>

#include "../gpgpu_context.hpp"
#include "../icnt_wrapper.hpp"
#include "../option_parser.hpp"
#include "../stream_manager.hpp"
#include "../trace_config.hpp"
#include "../trace_gpgpu_sim.hpp"
#include "../trace_kernel_info.hpp"
#include "../trace_parser.hpp"

#include "stats.hpp"
#include "main.hpp"
#include "playground/src/bridge/main.rs.h"

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

int accelsim(accelsim_config config, rust::Slice<const rust::Str> argv,
             Stats &stats) {
  std::cout << "Accel-Sim [build <box>]" << std::endl;

  bool silent = false;
#ifdef BOX
  if (std::getenv("SILENT") && strcmp(std::getenv("SILENT"), "yes") == 0) {
    silent = true;
  }
#endif

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

  gpgpu_sim_config *sim_config =
      m_gpgpu_context->the_gpgpusim->g_the_gpu_config;
  // sim_config->gpu_max_cycle_opt;
  // unsigned long long cycle_limit = (unsigned long long)-1;
  if (std::getenv("CYCLES") && atoi(std::getenv("CYCLES")) > 0) {
    sim_config->gpu_max_cycle_opt = atoi(std::getenv("CYCLES"));
  }

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

      // #ifdef BOX
      //       if (cycle >= cycle_limit) {
      //         printf("early exit after %llu cycles\n", cycle);
      //         m_gpgpu_context->the_gpgpusim->g_stream_manager
      //             ->stop_all_running_kernels();
      //         break;
      //         // fflush(stdout);
      //         // return 0;
      //       }
      // #endif

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
