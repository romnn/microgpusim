#pragma once

#include "scheduler_unit.hpp"

class gto_scheduler : public scheduler_unit {
 public:
  gto_scheduler(shader_core_stats *stats, trace_shader_core_ctx *shader,
                Scoreboard *scoreboard, simt_stack **simt,
                std::vector<trace_shd_warp_t *> *warp, register_set *sp_out,
                register_set *dp_out, register_set *sfu_out,
                register_set *int_out, register_set *tensor_core_out,
                std::vector<register_set *> &spec_cores_out,
                register_set *mem_out, int id)
      : scheduler_unit(stats, shader, scoreboard, simt, warp, sp_out, dp_out,
                       sfu_out, int_out, tensor_core_out, spec_cores_out,
                       mem_out, id) {}
  virtual ~gto_scheduler() {}
  virtual void order_warps();
  virtual void done_adding_supervised_warps() {
    m_last_supervised_issued = m_supervised_warps.begin();
  }

  virtual const char *name() { return "gto scheduler"; }
};
