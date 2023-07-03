#pragma once

#include "hal.hpp"
#include "option_parser.hpp"

class trace_config {
 public:
  trace_config();

  void set_latency(unsigned category, unsigned &latency,
                   unsigned &initiation_interval) const;
  void parse_config();
  void reg_options(option_parser_t opp);
  char *get_traces_filename() { return g_traces_filename; }

 private:
  unsigned int_latency, fp_latency, dp_latency, sfu_latency, tensor_latency;
  unsigned int_init, fp_init, dp_init, sfu_init, tensor_init;
  unsigned specialized_unit_latency[SPECIALIZED_UNIT_NUM];
  unsigned specialized_unit_initiation[SPECIALIZED_UNIT_NUM];

  char *g_traces_filename;
  char *trace_opcode_latency_initiation_int;
  char *trace_opcode_latency_initiation_sp;
  char *trace_opcode_latency_initiation_dp;
  char *trace_opcode_latency_initiation_sfu;
  char *trace_opcode_latency_initiation_tensor;
  char *trace_opcode_latency_initiation_specialized_op[SPECIALIZED_UNIT_NUM];
};
