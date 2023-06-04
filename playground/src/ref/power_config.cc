#include "power_config.hpp"

void power_config::reg_options(class OptionParser *opp) {
  option_parser_register(opp, "-accelwattch_xml_file", OPT_CSTR,
                         &g_power_config_name, "AccelWattch XML file",
                         "accelwattch_sass_sim.xml");

  option_parser_register(opp, "-power_simulation_enabled", OPT_BOOL,
                         &g_power_simulation_enabled,
                         "Turn on power simulator (1=On, 0=Off)", "0");

  option_parser_register(opp, "-power_per_cycle_dump", OPT_BOOL,
                         &g_power_per_cycle_dump,
                         "Dump detailed power output each cycle", "0");

  option_parser_register(opp, "-hw_perf_file_name", OPT_CSTR,
                         &g_hw_perf_file_name,
                         "Hardware Performance Statistics file", "hw_perf.csv");

  option_parser_register(
      opp, "-hw_perf_bench_name", OPT_CSTR, &g_hw_perf_bench_name,
      "Kernel Name in Hardware Performance Statistics file", "");

  option_parser_register(opp, "-power_simulation_mode", OPT_INT32,
                         &g_power_simulation_mode,
                         "Switch performance counter input for power "
                         "simulation (0=Sim, 1=HW, 2=HW-Sim Hybrid)",
                         "0");

  option_parser_register(opp, "-dvfs_enabled", OPT_BOOL, &g_dvfs_enabled,
                         "Turn on DVFS for power model", "0");
  option_parser_register(opp, "-aggregate_power_stats", OPT_BOOL,
                         &g_aggregate_power_stats,
                         "Accumulate power across all kernels", "0");

  // Accelwattch Hyrbid Configuration

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L1_RH", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L1_RH],
      "Get L1 Read Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L1_RM", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L1_RM],
      "Get L1 Read Misses for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L1_WH", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L1_WH],
      "Get L1 Write Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L1_WM", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L1_WM],
      "Get L1 Write Misses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L2_RH", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L2_RH],
      "Get L2 Read Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L2_RM", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L2_RM],
      "Get L2 Read Misses for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L2_WH", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L2_WH],
      "Get L2 Write Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L2_WM", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L2_WM],
      "Get L2 Write Misses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_CC_ACC", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_CC_ACC],
      "Get Constant Cache Acesses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_SHARED_ACC", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_SHRD_ACC],
      "Get Shared Memory Acesses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(opp, "-accelwattch_hybrid_perfsim_DRAM_RD", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_DRAM_RD],
                         "Get DRAM Reads for Accelwattch-Hybrid from Accel-Sim",
                         "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_DRAM_WR", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_DRAM_WR],
      "Get DRAM Writes for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_NOC", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_NOC],
      "Get Interconnect Acesses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_PIPE_DUTY", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_PIPE_DUTY],
      "Get Pipeline Duty Cycle Acesses for Accelwattch-Hybrid from Accel-Sim",
      "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_NUM_SM_IDLE", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_NUM_SM_IDLE],
      "Get Number of Idle SMs for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_CYCLES", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_CYCLES],
      "Get Executed Cycles for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_VOLTAGE", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_VOLTAGE],
      "Get Chip Voltage for Accelwattch-Hybrid from Accel-Sim", "0");

  // Output Data Formats
  option_parser_register(
      opp, "-power_trace_enabled", OPT_BOOL, &g_power_trace_enabled,
      "produce a file for the power trace (1=On, 0=Off)", "0");

  option_parser_register(
      opp, "-power_trace_zlevel", OPT_INT32, &g_power_trace_zlevel,
      "Compression level of the power trace output log (0=no comp, 9=highest)",
      "6");

  option_parser_register(
      opp, "-steady_power_levels_enabled", OPT_BOOL,
      &g_steady_power_levels_enabled,
      "produce a file for the steady power levels (1=On, 0=Off)", "0");

  option_parser_register(opp, "-steady_state_definition", OPT_CSTR,
                         &gpu_steady_state_definition,
                         "allowed deviation:number of samples", "8:4");
}
