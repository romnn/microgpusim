#pragma once

#include "../trace_shader_core_ctx.hpp"
#include "register_set.hpp"
#include "scheduler_unit.hpp"
#include "operand_collector.hpp"

class core_bridge {
 public:
  core_bridge(const trace_shader_core_ctx *ptr) : ptr(ptr) {}

  std::unique_ptr<std::vector<register_set_ptr>> get_register_sets() const {
    std::vector<register_set_ptr> out;
    for (unsigned n = 0; n < ptr->m_num_function_units; n++) {
      unsigned int issue_port = ptr->m_issue_port[n];
      const register_set &issue_reg = ptr->m_pipeline_reg[issue_port];
      bool is_sp = issue_port == ID_OC_SP || issue_port == OC_EX_SP;
      bool is_mem = issue_port == ID_OC_MEM || issue_port == OC_EX_MEM;
      if (is_sp || is_mem) {
        out.push_back(register_set_ptr{std::addressof(issue_reg)});
      }
    }

    return std::make_unique<std::vector<register_set_ptr>>(out);
  }

  std::unique_ptr<std::vector<scheduler_unit_ptr>> get_scheduler_units() const {
    std::vector<scheduler_unit_ptr> out;
    std::vector<scheduler_unit *>::const_iterator iter;
    for (iter = (ptr->schedulers).begin(); iter != (ptr->schedulers).end();
         iter++) {
      out.push_back(scheduler_unit_ptr{*iter});
    }
    return std::make_unique<std::vector<scheduler_unit_ptr>>(out);
  }

  std::shared_ptr<operand_collector_bridge> get_operand_collector() const {
    return std::make_shared<operand_collector_bridge>(
        &(ptr->m_operand_collector));
  }

 private:
  const class trace_shader_core_ctx *ptr;
};
