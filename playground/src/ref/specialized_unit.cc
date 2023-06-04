#include "specialized_unit.hpp"

#include "pipelined_simd_unit.hpp"
#include "trace_shader_core_ctx.hpp"

specialized_unit::specialized_unit(register_set *result_port,
                                   const shader_core_config *config,
                                   trace_shader_core_ctx *core,
                                   unsigned supported_op, char *unit_name,
                                   unsigned latency, unsigned issue_reg_id)
    : pipelined_simd_unit(result_port, config, latency, core, issue_reg_id) {
  m_name = unit_name;
  m_supported_op = supported_op;
}

void specialized_unit::active_lanes_in_pipeline() {
  unsigned active_count = pipelined_simd_unit::get_active_lanes_in_pipeline();
  assert(active_count <= m_core->get_config()->warp_size);
  m_core->incspactivelanes_stat(active_count);
  m_core->incfuactivelanes_stat(active_count);
  m_core->incfumemactivelanes_stat(active_count);
}

void specialized_unit::issue(register_set &source_reg) {
  warp_inst_t **ready_reg =
      source_reg.get_ready(m_config->sub_core_model, m_issue_reg_id);
  // m_core->incexecstat((*ready_reg));
  (*ready_reg)->op_pipe = SPECIALIZED__OP;
  m_core->incsp_stat(m_core->get_config()->warp_size, (*ready_reg)->latency);
  pipelined_simd_unit::issue(source_reg);
}
