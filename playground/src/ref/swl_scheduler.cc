#include "swl_scheduler.hpp"

#include "trace_shader_core_ctx.hpp"

swl_scheduler::swl_scheduler(shader_core_stats *stats,
                             trace_shader_core_ctx *shader,
                             Scoreboard *scoreboard, simt_stack **simt,
                             std::vector<trace_shd_warp_t *> *warp,
                             register_set *sp_out, register_set *dp_out,
                             register_set *sfu_out, register_set *int_out,
                             register_set *tensor_core_out,
                             std::vector<register_set *> &spec_cores_out,
                             register_set *mem_out, int id, char *config_string)
    : scheduler_unit(stats, shader, scoreboard, simt, warp, sp_out, dp_out,
                     sfu_out, int_out, tensor_core_out, spec_cores_out, mem_out,
                     id) {
  unsigned m_prioritization_readin;
  int ret = sscanf(config_string, "warp_limiting:%d:%d",
                   &m_prioritization_readin, &m_num_warps_to_limit);
  assert(2 == ret);
  m_prioritization = (scheduler_prioritization_type)m_prioritization_readin;
  // Currently only GTO is implemented
  assert(m_prioritization == SCHEDULER_PRIORITIZATION_GTO);
  assert(m_num_warps_to_limit <= shader->get_config()->max_warps_per_shader);
}

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

void swl_scheduler::order_warps() {
  if (SCHEDULER_PRIORITIZATION_GTO == m_prioritization) {
    order_by_priority(m_next_cycle_prioritized_warps, m_supervised_warps,
                      m_last_supervised_issued,
                      MIN(m_num_warps_to_limit, m_supervised_warps.size()),
                      ORDERING_GREEDY_THEN_PRIORITY_FUNC,
                      scheduler_unit::sort_warps_by_oldest_dynamic_id);
  } else {
    fprintf(stderr, "swl_scheduler m_prioritization = %d\n", m_prioritization);
    abort();
  }
}
