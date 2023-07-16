#pragma once

#include "pipelined_simd_unit.hpp"

class specialized_unit : public pipelined_simd_unit {
 public:
  specialized_unit(register_set *result_port, const shader_core_config *config,
                   trace_shader_core_ctx *core, unsigned supported_op,
                   char *unit_name, unsigned latency, unsigned issue_reg_id);
  virtual bool can_issue(const warp_inst_t &inst) const {
    if (inst.op != m_supported_op) {
      return false;
    }
    return pipelined_simd_unit::can_issue(inst);
  }
  virtual void active_lanes_in_pipeline();
  virtual void issue(register_set &source_reg);
  bool is_issue_partitioned() { return true; }

 private:
  unsigned m_supported_op;
};
