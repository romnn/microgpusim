#include <iostream>
#include <cstdlib>

#include "../gpgpu_context.hpp"
#include "../icnt_wrapper.hpp"
#include "../option_parser.hpp"
#include "../stream_manager.hpp"
#include "../trace_gpgpu_sim.hpp"
#include "../trace_parser.hpp"
#include "../memory_sub_partition.hpp"
#include "../trace_simt_core_cluster.hpp"
#include "../trace_shader_core_ctx.hpp"

#include "playground-sys/src/bridge/main.rs.h"

#include "spdlog/common.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "stats.hpp"
#include "main.hpp"

trace_kernel_info_t *create_kernel_info(kernel_trace_t *kernel_trace_info,
                                        gpgpu_context *m_gpgpu_context,
                                        class trace_config *config,
                                        trace_parser *parser);

void cli_configure(gpgpu_context *m_gpgpu_context, trace_config &m_config,
                   const std::vector<const char *> &argv, bool print_stats,
                   FILE *stats_out) {
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

  // if (print_stats || m_gpgpu_context->accelsim_compat_mode) {
  //   fprintf(stats_out, "GPGPU-Sim: Registered options:\n\n");
  //   option_parser_print_registered(opp, stats_out);
  // }

  // parse configuration options
  option_parser_cmdline(opp, argv);

  if (print_stats || m_gpgpu_context->accelsim_compat_mode) {
    fprintf(stats_out, "GPGPU-Sim: Configuration options:\n\n");
    option_parser_print(opp, stats_out);
  }

  // initialize config (parse gpu config from cli values)
  m_gpgpu_context->the_gpgpusim->g_the_gpu_config->init();

  // override some values
  g_network_mode = BOX_NET;
  // if (!m_gpgpu_context->accelsim_compat_mode) {
  //   g_network_mode = BOX_NET;
  // }
}

trace_gpgpu_sim *gpgpu_trace_sim_init_perf_model(
    gpgpu_context *m_gpgpu_context, trace_config &m_config,
    const accelsim_config &config, const std::vector<const char *> &argv,
    std::shared_ptr<spdlog::logger> logger, bool print_stats, FILE *stats_out) {
  // seed random
  srand(1);

  // Set the Numeric locale to a standard locale where a decimal point is a
  // "dot" not a "comma" so it does the parsing correctly independent of the
  // system environment variables
  assert(setlocale(LC_NUMERIC, "C"));

  // configure using cli
  cli_configure(m_gpgpu_context, m_config, argv, print_stats, stats_out);

  // TODO: configure using config
  // m_gpgpu_context->the_gpgpusim->g_the_gpu_config->configure(config);

  m_gpgpu_context->the_gpgpusim->g_the_gpu =
      new trace_gpgpu_sim(*(m_gpgpu_context->the_gpgpusim->g_the_gpu_config),
                          m_gpgpu_context, logger, stats_out);

  m_gpgpu_context->the_gpgpusim->g_stream_manager =
      new stream_manager((m_gpgpu_context->the_gpgpusim->g_the_gpu),
                         m_gpgpu_context->func_sim->g_cuda_launch_blocking);

  m_gpgpu_context->the_gpgpusim->g_simulation_starttime = time((time_t *)NULL);

  // return static_cast<class trace_gpgpu_sim_bridge *>(
  return static_cast<class trace_gpgpu_sim *>(
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

bool is_env_set_to(const char *key, const char *value) {
  return (std::getenv(key) && strcmp(std::getenv(key), value) == 0);
}

std::unique_ptr<accelsim_bridge> new_accelsim_bridge(
    accelsim_config config, rust::Slice<const rust::Str> argv) {
  return std::make_unique<accelsim_bridge>(config, argv);
}

accelsim_bridge::~accelsim_bridge() {
  // logger->clear();
  // spdlog::shutdown();

  delete tracer;
  delete m_gpgpu_sim;
  delete m_gpgpu_context;
};

void configure_log_level(std::shared_ptr<spdlog::logger> &logger) {
  std::string log_level = spdlog::details::os::getenv("SPDLOG_LEVEL");
  // make lowercase
  std::transform(log_level.begin(), log_level.end(), log_level.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (log_level.compare("trace") == 0) {
    logger->set_level(spdlog::level::level_enum::trace);
  } else if (log_level.compare("debug") == 0) {
    logger->set_level(spdlog::level::level_enum::debug);
  } else if (log_level.compare("warn") == 0) {
    logger->set_level(spdlog::level::level_enum::warn);
  } else if (log_level.compare("info") == 0) {
    logger->set_level(spdlog::level::level_enum::info);
  } else {
    // disable logging fully
    logger->set_level(spdlog::level::level_enum::off);
  }
}

accelsim_bridge::accelsim_bridge(accelsim_config config,
                                 rust::Slice<const rust::Str> argv) {
  print_stats = config.print_stats;
  print_stats |= is_env_set_to("PRINT_STATS", "yes");
  accelsim_compat_mode = config.accelsim_compat_mode;
  accelsim_compat_mode |= is_env_set_to("ACCELSIM_COMPAT_MODE", "yes");

  fmt::println("accelsim compat: {}", accelsim_compat_mode);

  stats_out = stdout;
  const char *stats_file = config.stats_file;
  if (std::getenv("PLAYGROUND_STATS_FILE")) {
    stats_file = std::getenv("PLAYGROUND_STATS_FILE");
  }

  if (stats_file != NULL) {
    stats_out = std::fopen(stats_file, "w");
    // fmt::println("redirecting stdout to {} ... ", std::string(log_to));
    // freopen(log_to, "w", stdout);
  }

  std::vector<std::string> valid_argv;
  for (auto arg : argv) valid_argv.push_back(std::string(arg));

  std::vector<const char *> c_argv;
  for (std::string &arg : valid_argv) c_argv.push_back(arg.c_str());

  if (std::getenv("PLAYGROUND_USE_LOG_FILE") &&
      std::getenv("PLAYGROUND_LOG_FILE")) {
    std::string log_file = std::string(std::getenv("PLAYGROUND_LOG_FILE"));
    auto file_logger =
        std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file);
    logger = std::make_shared<spdlog::logger>("playground", file_logger);
  } else {
    auto stdout_logger =
        std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    logger = std::make_shared<spdlog::logger>("playground", stdout_logger);
  }
  // logger->set_pattern("[multi_sink_example] [%^%l%$] %v");
  logger->set_pattern("%v");
  // does not work with non-global static loggers
  // spdlog::cfg::load_env_levels();

  if (accelsim_compat_mode) {
    logger->set_level(spdlog::level::level_enum::off);
  } else {
    configure_log_level(logger);
  }

  log_after_cycle = 0;
  std::string log_after_cycle_str = spdlog::details::os::getenv("LOG_AFTER");
  if (!accelsim_compat_mode && log_after_cycle_str.length() > 0 &&
      atoi(log_after_cycle_str.c_str()) > 0) {
    logger->set_level(spdlog::level::level_enum::off);
    log_after_cycle = atoi(log_after_cycle_str.c_str());
  }

  // setup the gpu
  m_gpgpu_context = new gpgpu_context();
  m_gpgpu_context->stats_out = stats_out;
  m_gpgpu_context->accelsim_compat_mode = accelsim_compat_mode;

  if (m_gpgpu_context->accelsim_compat_mode) {
    fprintf(stats_out, "Accel-Sim [build <box>]");
  }

  // init trace based performance model
  m_gpgpu_sim = gpgpu_trace_sim_init_perf_model(
      m_gpgpu_context, tconfig, config, c_argv, logger, print_stats, stats_out);

  m_gpgpu_sim->init();

  // init trace parser
  tracer = new trace_parser(tconfig.get_traces_filename(),
                            !accelsim_compat_mode, stats_out);

  // parse trace config
  tconfig.parse_config();
  m_gpgpu_sim->logger->info("initialization complete");

  // configure max cycle opt
  gpgpu_sim_config *sim_config =
      m_gpgpu_context->the_gpgpusim->g_the_gpu_config;

  sim_config->gpu_max_cycle_opt = (unsigned long long)-1;
  if (std::getenv("CYCLES") && atoi(std::getenv("CYCLES")) > 0) {
    sim_config->gpu_max_cycle_opt = atoi(std::getenv("CYCLES"));
  }

  // setup a rolling window with size of the max concurrent kernel executions
  bool concurrent_kernel_sm =
      m_gpgpu_sim->getShaderCoreConfig()->gpgpu_concurrent_kernel_sm;
  window_size = concurrent_kernel_sm
                    ? m_gpgpu_sim->get_config().get_max_concurrent_kernel()
                    : 1;
  assert(window_size > 0);

  // parse the list of commands issued to the GPU
  commandlist = tracer->parse_commandlist_file();
  kernels_info.reserve(window_size);
  command_idx = 0;

  for (unsigned i = 0; i < m_gpgpu_sim->m_memory_config->m_n_mem_sub_partition;
       i++) {
    memory_sub_partition *sub_partition =
        m_gpgpu_sim->m_memory_sub_partition[i];
    sub_partitions.push_back(memory_sub_partition_bridge(sub_partition));
  }

  for (unsigned i = 0; i < m_gpgpu_sim->m_memory_config->m_n_mem; i++) {
    memory_partition_unit *partition = m_gpgpu_sim->m_memory_partition_unit[i];
    partition_units.push_back(memory_partition_unit_bridge(partition));
  }

  for (unsigned cluster_id = 0;
       cluster_id < m_gpgpu_sim->m_shader_config->n_simt_clusters;
       cluster_id++) {
    trace_simt_core_cluster *cluster = m_gpgpu_sim->m_cluster[cluster_id];
    clusters.push_back(cluster_bridge(cluster));
    for (unsigned core_id = 0;
         core_id < m_gpgpu_sim->m_shader_config->n_simt_cores_per_cluster;
         core_id++) {
      trace_shader_core_ctx *core = cluster->m_core[core_id];
      cores.push_back(core_bridge(core));
    }
  }
}

void accelsim_bridge::process_commands() {
  // gulp up as many commands as possible - either cpu_gpu_mem_copy
  // or kernel_launch - until the vector "kernels_info" has reached
  // the window_size or we have read every command from commandlist
  while (kernels_info.size() < window_size &&
         command_idx < commandlist.size()) {
    trace_kernel_info_t *kernel_info = NULL;
    if (commandlist[command_idx].m_type == command_type::cpu_gpu_mem_copy) {
      // parse memcopy command
      size_t addre, Bcount;
      tracer->parse_memcpy_info(commandlist[command_idx].command_string, addre,
                                Bcount);
      if (m_gpgpu_context->accelsim_compat_mode) {
        fprintf(stats_out, "launching memcpy command : %s\n",
                commandlist[command_idx].command_string.c_str());
      }
      m_gpgpu_sim->perf_memcpy_to_gpu(addre, Bcount);
      command_idx++;
    } else if (commandlist[command_idx].m_type == command_type::kernel_launch) {
      // Read trace header info for window_size number of kernels
      kernel_trace_t *kernel_trace_info =
          tracer->parse_kernel_info(commandlist[command_idx].command_string);
      kernel_info = create_kernel_info(kernel_trace_info, m_gpgpu_context,
                                       &tconfig, tracer);
      kernels_info.push_back(kernel_info);

      if (m_gpgpu_context->accelsim_compat_mode) {
        fprintf(stats_out, "Header info loaded for kernel command : %s\n",
                commandlist[command_idx].command_string.c_str());
      }
      command_idx++;
    } else {
      // unsupported commands will fail the simulation
      assert(0 && "undefined command");
    }
  }
  m_gpgpu_sim->logger->info("allocations: {}", m_gpgpu_sim->m_allocations);
}

// Launch all kernels within window that are on a stream that isn't
// already running
void accelsim_bridge::launch_kernels() {
  m_gpgpu_sim->logger->trace("launching kernels");

  for (auto k : kernels_info) {
    // check if stream of kernel is busy
    bool stream_busy = false;
    for (auto s : busy_streams) {
      if (s == k->get_cuda_stream_id()) stream_busy = true;
    }
    if (!stream_busy && m_gpgpu_sim->can_start_kernel() && !k->was_launched()) {
      m_gpgpu_sim->logger->info("launching kernel: {}",
                                trace_kernel_info_ptr(k));
      // if (k->get_uid() > 1) {
      //   assert(0 && "new kernel");
      // }
      m_gpgpu_sim->launch(k);

      k->set_launched();
      busy_streams.push_back(k->get_cuda_stream_id());
    }
  }
}

void accelsim_bridge::cycle() {
  // unsigned long long cycle =
  //     m_gpgpu_sim->gpu_tot_sim_cycle + m_gpgpu_sim->gpu_sim_cycle;

  // performance simulation
  if (active()) {
    if (accelsim_compat_mode) {
      m_gpgpu_sim->cycle();
    } else {
      m_gpgpu_sim->simple_cycle();
    }
    m_gpgpu_sim->deadlock_check();
  } else {
    // stop all kernels if we reached max instructions limit
    if (m_gpgpu_sim->cycle_insn_cta_max_hit()) {
      m_gpgpu_context->the_gpgpusim->g_stream_manager->stop_all_running_kernels(
          stats_out);
      return;
    }
  }
}

void accelsim_bridge::cleanup_finished_kernel(unsigned finished_kernel_uid) {
  logger->debug("cleanup finished kernel with id={}", finished_kernel_uid);
  if (finished_kernel_uid || limit_reached() || !active()) {
    trace_kernel_info_t *k = NULL;
    for (unsigned j = 0; j < kernels_info.size(); j++) {
      k = kernels_info.at(j);
      if (k->get_uid() == finished_kernel_uid || limit_reached() || !active()) {
        for (unsigned l = 0; l < busy_streams.size(); l++) {
          if (busy_streams.at(l) == k->get_cuda_stream_id()) {
            busy_streams.erase(busy_streams.begin() + l);
            break;
          }
        }
        tracer->kernel_finalizer(k->get_trace_info());
        delete k->entry();
        delete k;
        kernels_info.erase(kernels_info.begin() + j);
        if (!limit_reached() && active()) break;
      }
    }
    // make sure kernel was found and removed
    assert(k);
    if (print_stats || accelsim_compat_mode)
      m_gpgpu_sim->print_stats(stats_out);
  }

  if ((print_stats || accelsim_compat_mode) && m_gpgpu_sim->gpu_sim_cycle > 0) {
    // update_stats() resets some statistics between kernel launches
    m_gpgpu_sim->update_stats();
    m_gpgpu_context->print_simulation_time(stats_out);
  }
}

void accelsim_bridge::run_to_completion() {
  while (commands_left() || kernels_left()) {
    // gulp up as many commands as possible - either cpu_gpu_mem_copy
    // or kernel_launch - until the vector "kernels_info" has reached
    // the window_size or we have read every command from commandlist
    process_commands();
    launch_kernels();

    unsigned finished_kernel_uid = 0;
    do {
      if (!active()) break;

      unsigned tot_cycle =
          m_gpgpu_sim->gpu_tot_sim_cycle + m_gpgpu_sim->gpu_sim_cycle;

      if (log_after_cycle > 0 && tot_cycle >= log_after_cycle) {
        fmt::println("initializing logging after cycle {}", tot_cycle);
        configure_log_level(logger);
        log_after_cycle = 0;
      }

      cycle();
      // check for finished kernel
      finished_kernel_uid = get_finished_kernel_uid();
      if (finished_kernel_uid) break;
    } while (true);

    cleanup_finished_kernel(finished_kernel_uid);

    if (m_gpgpu_sim->cycle_insn_cta_max_hit()) {
      if (m_gpgpu_context->accelsim_compat_mode) {
        fprintf(stats_out,
                "GPGPU-Sim: ** break due to reaching the maximum cycles (or "
                "instructions) **\n");
        fflush(stats_out);
      }
      break;
    }
  }

  // we print this message to inform the gpgpu-simulation stats_collect script
  // that we are done
  if (m_gpgpu_context->accelsim_compat_mode) {
    fprintf(stats_out, "GPGPU-Sim: *** simulation thread exiting ***\n");
    fprintf(stats_out, "GPGPU-Sim: *** exit detected ***\n");
    fflush(stats_out);
  }
}
