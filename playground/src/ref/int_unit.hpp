#pragma once

#include "pipelined_simd_unit.hpp"

class int_unit : public pipelined_simd_unit {
public:
  int_unit(register_set *result_port, const shader_core_config *config,
           trace_shader_core_ctx *core, unsigned issue_reg_id);
  virtual bool can_issue(const warp_inst_t &inst) const {
    switch (inst.op) {
    case SFU_OP:
      return false;
    case LOAD_OP:
      return false;
    case TENSOR_CORE_LOAD_OP:
      return false;
    case STORE_OP:
      return false;
    case TENSOR_CORE_STORE_OP:
      return false;
    case MEMORY_BARRIER_OP:
      return false;
    case SP_OP:
      return false;
    case DP_OP:
      return false;
    default:
      break;
    }
    return pipelined_simd_unit::can_issue(inst);
  }
  virtual void active_lanes_in_pipeline();
  virtual void issue(register_set &source_reg);
  bool is_issue_partitioned() { return true; }
};
