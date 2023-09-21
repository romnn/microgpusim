#pragma once

#include "../trace_shader_core_ctx.hpp"
#include "../simd_function_unit.hpp"
#include "../pipelined_simd_unit.hpp"
#include "../ldst_unit.hpp"
#include "../l1_cache.hpp"
#include "cache.hpp"
#include "register_set.hpp"
#include "mem_fetch.hpp"
#include "scheduler_unit.hpp"
#include "operand_collector.hpp"

struct pending_register_writes {
  unsigned warp_id;
  unsigned reg_num;
  unsigned pending;
};

struct bank_latency_queue {
  const std::vector<mem_fetch_ptr_shim> &get() const { return queue; };
  const std::vector<mem_fetch_ptr_shim> queue;
};

class core_bridge {
 public:
  core_bridge(const trace_shader_core_ctx *ptr) : ptr(ptr) {}

  std::unique_ptr<std::vector<register_set_ptr>>
  get_functional_unit_issue_register_sets() const {
    std::vector<register_set_ptr> out;
    for (unsigned n = 0; n < ptr->m_num_function_units; n++) {
      unsigned int issue_port = ptr->m_issue_port[n];
      const register_set &issue_reg = ptr->m_pipeline_reg[issue_port];
      bool is_sp = issue_port == ID_OC_SP || issue_port == OC_EX_SP;
      bool is_mem = issue_port == ID_OC_MEM || issue_port == OC_EX_MEM;
      bool is_sfu = issue_port == ID_OC_SFU || issue_port == OC_EX_SFU;
      bool is_int = issue_port == ID_OC_INT || issue_port == OC_EX_INT;
      bool is_dp = issue_port == ID_OC_DP || issue_port == OC_EX_DP;
      if (!is_sp && !is_mem && !is_sfu && !is_dp && !is_int) {
        continue;
      }
      out.push_back(register_set_ptr{std::addressof(issue_reg)});
    }

    return std::make_unique<std::vector<register_set_ptr>>(out);
  }

  std::unique_ptr<std::vector<register_set_ptr>>
  get_functional_unit_simd_pipeline_register_sets() const {
    std::vector<register_set_ptr> out;

    std::vector<simd_function_unit *>::const_iterator iter;
    for (iter = (ptr->m_fu).begin(); iter != (ptr->m_fu).end(); iter++) {
      const simd_function_unit *fu = *iter;
      register_set *reg = new register_set(0, fu->get_name(), 0, NULL);

      if (fu->is_pipelined()) {
        const pipelined_simd_unit *pipe_fu =
            static_cast<const pipelined_simd_unit *>(fu);
        std::vector<warp_inst_t *> regs = std::vector<warp_inst_t *>(
            pipe_fu->m_pipeline_reg,
            pipe_fu->m_pipeline_reg + pipe_fu->m_pipeline_depth);
        // uses copy constructor
        reg->regs = regs;
      }
      out.push_back(register_set_ptr{reg});
    }

    return std::make_unique<std::vector<register_set_ptr>>(out);
  }

  std::unique_ptr<std::vector<std::string>> get_functional_unit_occupied_slots()
      const {
    std::vector<std::string> out;
    std::vector<simd_function_unit *>::const_iterator iter;
    for (iter = (ptr->m_fu).begin(); iter != (ptr->m_fu).end(); iter++) {
      const simd_function_unit *fu = *iter;
      if (fu->is_pipelined()) {
        const pipelined_simd_unit *pipe_fu =
            static_cast<const pipelined_simd_unit *>(fu);
        out.push_back(mask_to_string(pipe_fu->occupied));
      }
    }

    return std::make_unique<std::vector<std::string>>(out);
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

  std::unique_ptr<std::vector<pending_register_writes>>
  get_pending_register_writes() const {
    std::vector<pending_register_writes> out;
    std::map<unsigned int, std::map<unsigned int, unsigned int>>
        &pending_writes = (ptr->m_ldst_unit)->m_pending_writes;
    std::map<unsigned int, std::map<unsigned int, unsigned int>>::const_iterator
        warp_iter;
    std::map<unsigned int, unsigned int>::const_iterator reg_iter;

    for (warp_iter = pending_writes.begin(); warp_iter != pending_writes.end();
         warp_iter++) {
      unsigned warp_id = warp_iter->first;
      for (reg_iter = (warp_iter->second).begin();
           reg_iter != (warp_iter->second).end(); reg_iter++) {
        unsigned reg_num = reg_iter->first;
        unsigned pending = reg_iter->second;
        out.push_back(pending_register_writes{warp_id, reg_num, pending});
      }
    }

    return std::make_unique<std::vector<pending_register_writes>>(out);
  }

  std::unique_ptr<std::vector<bank_latency_queue>> get_l1_bank_latency_queue()
      const {
    std::vector<bank_latency_queue> out;
    std::vector<std::deque<mem_fetch *>> queue =
        (ptr->m_ldst_unit)->l1_latency_queue;
    std::vector<std::deque<mem_fetch *>>::const_iterator iter;
    for (iter = queue.begin(); iter != queue.end(); iter++) {
      std::deque<mem_fetch *>::const_iterator bank_iter;

      std::vector<mem_fetch_ptr_shim> bank_queue_shim;
      for (bank_iter = (*iter).begin(); bank_iter != (*iter).end();
           bank_iter++) {
        bank_queue_shim.push_back(mem_fetch_ptr_shim{*bank_iter});
      }
      out.push_back(bank_latency_queue{bank_queue_shim});
    }
    return std::make_unique<std::vector<bank_latency_queue>>(out);
  }

  std::shared_ptr<cache_bridge> get_l1_data_cache() const {
    return new_cache_bridge(ptr->m_ldst_unit->m_L1D);
  }

 private:
  const class trace_shader_core_ctx *ptr;
};
