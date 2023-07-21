#pragma once

#include "spdlog/logger.h"
#include "simd_function_unit.hpp"
#include "trace_warp_inst.hpp"

class trace_shader_core_ctx;

class pipelined_simd_unit : public simd_function_unit {
 public:
  pipelined_simd_unit(register_set *result_port,
                      const shader_core_config *config, unsigned max_latency,
                      trace_shader_core_ctx *core, unsigned issue_reg_id);

  // modifiers
  virtual void cycle();
  virtual void issue(register_set &source_reg);
  virtual unsigned get_active_lanes_in_pipeline();

  virtual void active_lanes_in_pipeline() = 0;

  // accessors
  virtual bool stallable() const { return false; }
  virtual bool can_issue(const warp_inst_t &inst) const {
    return simd_function_unit::can_issue(inst);
  }
  virtual bool is_issue_partitioned() = 0;
  unsigned get_issue_reg_id() { return m_issue_reg_id; }
  // virtual void print(FILE *fp) const {
  //   simd_function_unit::print(fp);
  //   for (int s = m_pipeline_depth - 1; s >= 0; s--) {
  //     if (!m_pipeline_reg[s]->empty()) {
  //       fprintf(fp, "      %s[%2d] ", m_name.c_str(), s);
  //       m_pipeline_reg[s]->print(fp);
  //     }
  //   }
  // }

  std::shared_ptr<spdlog::logger> logger;

 protected:
  unsigned m_pipeline_depth;
  warp_inst_t **m_pipeline_reg;
  register_set *m_result_port;
  class trace_shader_core_ctx *m_core;
  unsigned m_issue_reg_id;  // if sub_core_model is enabled we can only issue
                            // from a subset of operand collectors

  unsigned active_insts_in_pipeline;
};
