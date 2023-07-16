#pragma once

#include "scheduler_prioritization_type.hpp"
#include "scheduler_unit.hpp"

// Static Warp Limiting Scheduler
class swl_scheduler : public scheduler_unit {
 public:
  swl_scheduler(shader_core_stats *stats, trace_shader_core_ctx *shader,
                Scoreboard *scoreboard, simt_stack **simt,
                std::vector<trace_shd_warp_t *> *warp, register_set *sp_out,
                register_set *dp_out, register_set *sfu_out,
                register_set *int_out, register_set *tensor_core_out,
                std::vector<register_set *> &spec_cores_out,
                register_set *mem_out, int id, char *config_string);
  virtual ~swl_scheduler() {}
  virtual void order_warps();
  virtual void done_adding_supervised_warps() {
    m_last_supervised_issued = m_supervised_warps.begin();
  }

  const char *name() { return "swl_scheduler"; }

 protected:
  scheduler_prioritization_type m_prioritization;
  unsigned m_num_warps_to_limit;
};
