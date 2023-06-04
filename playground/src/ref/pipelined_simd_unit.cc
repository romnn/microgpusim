#include "pipelined_simd_unit.hpp"

#include "trace_gpgpu_sim.hpp"
#include "trace_shader_core_ctx.hpp"

unsigned pipelined_simd_unit::get_active_lanes_in_pipeline() {
  active_mask_t active_lanes;
  active_lanes.reset();
  if (m_core->get_gpu()->get_config().g_power_simulation_enabled) {
    for (unsigned stage = 0; (stage + 1) < m_pipeline_depth; stage++) {
      if (!m_pipeline_reg[stage]->empty())
        active_lanes |= m_pipeline_reg[stage]->get_active_mask();
    }
  }
  return active_lanes.count();
}

pipelined_simd_unit::pipelined_simd_unit(register_set *result_port,
                                         const shader_core_config *config,
                                         unsigned max_latency,
                                         trace_shader_core_ctx *core,
                                         unsigned issue_reg_id)
    : simd_function_unit(config) {
  m_result_port = result_port;
  m_pipeline_depth = max_latency;
  m_pipeline_reg = new warp_inst_t *[m_pipeline_depth];
  for (unsigned i = 0; i < m_pipeline_depth; i++)
    m_pipeline_reg[i] = new warp_inst_t(config);
  m_core = core;
  m_issue_reg_id = issue_reg_id;
  active_insts_in_pipeline = 0;
}

void pipelined_simd_unit::cycle() {
  if (!m_pipeline_reg[0]->empty()) {
    m_result_port->move_in(m_pipeline_reg[0]);
    assert(active_insts_in_pipeline > 0);
    active_insts_in_pipeline--;
  }
  if (active_insts_in_pipeline) {
    for (unsigned stage = 0; (stage + 1) < m_pipeline_depth; stage++)
      move_warp(m_pipeline_reg[stage], m_pipeline_reg[stage + 1]);
  }
  if (!m_dispatch_reg->empty()) {
    if (!m_dispatch_reg->dispatch_delay()) {
      int start_stage =
          m_dispatch_reg->latency - m_dispatch_reg->initiation_interval;
      move_warp(m_pipeline_reg[start_stage], m_dispatch_reg);
      active_insts_in_pipeline++;
    }
  }
  occupied >>= 1;
}

void pipelined_simd_unit::issue(register_set &source_reg) {
  // move_warp(m_dispatch_reg,source_reg);
  bool partition_issue =
      m_config->sub_core_model && this->is_issue_partitioned();
  warp_inst_t **ready_reg =
      source_reg.get_ready(partition_issue, m_issue_reg_id);
  m_core->incexecstat((*ready_reg));
  // source_reg.move_out_to(m_dispatch_reg);
  simd_function_unit::issue(source_reg);
}
