#include "simd_function_unit.hpp"

#include "register_set.hpp"
#include "shader_core_config.hpp"
#include "warp_instr.hpp"

simd_function_unit::simd_function_unit(const shader_core_config *config) {
  m_config = config;
  m_dispatch_reg = new warp_inst_t(config);
}

void simd_function_unit::issue(register_set &source_reg) {
  bool partition_issue =
      m_config->sub_core_model && this->is_issue_partitioned();
  std::stringstream msg;
  msg << get_name() << ": moving register to dispatch register for issue";
  source_reg.move_out_to(partition_issue, this->get_issue_reg_id(),
                         m_dispatch_reg, msg.str());
  occupied.set(m_dispatch_reg->latency);
}
