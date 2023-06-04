#include "tensor_core.hpp"

#include "trace_shader_core_ctx.hpp"

tensor_core::tensor_core(register_set *result_port,
                         const shader_core_config *config,
                         trace_shader_core_ctx *core, unsigned issue_reg_id)
    : pipelined_simd_unit(result_port, config, config->max_tensor_core_latency,
                          core, issue_reg_id) {
  m_name = "TENSOR_CORE";
}

void tensor_core::issue(register_set &source_reg) {
  warp_inst_t **ready_reg =
      source_reg.get_ready(m_config->sub_core_model, m_issue_reg_id);
  // m_core->incexecstat((*ready_reg));

  (*ready_reg)->op_pipe = TENSOR_CORE__OP;
  m_core->incsfu_stat(m_core->get_config()->warp_size, (*ready_reg)->latency);
  pipelined_simd_unit::issue(source_reg);
}

void tensor_core::active_lanes_in_pipeline() {
  unsigned active_count = pipelined_simd_unit::get_active_lanes_in_pipeline();
  assert(active_count <= m_core->get_config()->warp_size);
  m_core->incsfuactivelanes_stat(active_count);
  m_core->incfuactivelanes_stat(active_count);
  m_core->incfumemactivelanes_stat(active_count);
}
