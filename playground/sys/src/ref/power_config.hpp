#pragma once

#include <cstdio>
#include <ctime>
#include <string.h>

#include "hw_perf.hpp"
#include "option_parser.hpp"

struct power_config {
  power_config() {
    m_valid = true;

    // ROMAN: fix uninitialized variables
    g_use_nonlinear_model = false;
    g_steady_power_levels_enabled = false;
  }

  void init() {
    // initialize file name if it is not set
    time_t curr_time;
    time(&curr_time);
    char *date = ctime(&curr_time);
    char *s = date;
    while (*s) {
      if (*s == ' ' || *s == '\t' || *s == ':') *s = '-';
      if (*s == '\n' || *s == '\r') *s = 0;
      s++;
    }
    char buf1[1024];
    // snprintf(buf1, 1024, "accelwattch_power_report__%s.log", date);
    snprintf(buf1, 1024, "accelwattch_power_report.log");
    g_power_filename = strdup(buf1);
    char buf2[1024];
    snprintf(buf2, 1024, "gpgpusim_power_trace_report__%s.log.gz", date);
    g_power_trace_filename = strdup(buf2);
    char buf3[1024];
    snprintf(buf3, 1024, "gpgpusim_metric_trace_report__%s.log.gz", date);
    g_metric_trace_filename = strdup(buf3);
    char buf4[1024];
    snprintf(buf4, 1024, "gpgpusim_steady_state_tracking_report__%s.log.gz",
             date);
    g_steady_state_tracking_filename = strdup(buf4);
    // for(int i =0; i< hw_perf_t::HW_TOTAL_STATS; i++){
    //   accelwattch_hybrid_configuration[i] = 0;
    // }

    if (g_steady_power_levels_enabled) {
      sscanf(gpu_steady_state_definition, "%lf:%lf",
             &gpu_steady_power_deviation, &gpu_steady_min_period);
    }

    // NOTE: After changing the nonlinear model to only scaling idle core,
    // NOTE: The min_inc_per_active_sm is not used any more
    if (g_use_nonlinear_model)
      sscanf(gpu_nonlinear_model_config, "%lf:%lf", &gpu_idle_core_power,
             &gpu_min_inc_per_active_sm);
  }
  void reg_options(class OptionParser *opp);

  char *g_power_config_name;

  bool m_valid;
  bool g_power_simulation_enabled;
  bool g_power_trace_enabled;
  bool g_steady_power_levels_enabled;
  bool g_power_per_cycle_dump;
  bool g_power_simulator_debug;
  char *g_power_filename;
  char *g_power_trace_filename;
  char *g_metric_trace_filename;
  char *g_steady_state_tracking_filename;
  int g_power_trace_zlevel;
  char *gpu_steady_state_definition;
  double gpu_steady_power_deviation;
  double gpu_steady_min_period;

  char *g_hw_perf_file_name;
  char *g_hw_perf_bench_name;
  int g_power_simulation_mode;
  bool g_dvfs_enabled;
  bool g_aggregate_power_stats;
  bool accelwattch_hybrid_configuration[hw_perf_t::HW_TOTAL_STATS];

  // Nonlinear power model
  bool g_use_nonlinear_model;
  char *gpu_nonlinear_model_config;
  double gpu_idle_core_power;
  double gpu_min_inc_per_active_sm;
};
