#pragma once

#include "pipelined_simd_unit.hpp"

class tensor_core : public pipelined_simd_unit {
 public:
  tensor_core(register_set *result_port, const shader_core_config *config,
              trace_shader_core_ctx *core, unsigned issue_reg_id);
  virtual bool can_issue(const warp_inst_t &inst) const {
    switch (inst.op) {
      case TENSOR_CORE_OP:
        break;
      default:
        return false;
    }
    return pipelined_simd_unit::can_issue(inst);
  }
  virtual void active_lanes_in_pipeline();
  virtual void issue(register_set &source_reg);
  bool is_issue_partitioned() { return true; }
};
