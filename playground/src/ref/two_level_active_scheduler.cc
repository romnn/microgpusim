#include "two_level_active_scheduler.hpp"

#include "shader_trace.hpp"
#include "trace_warp_inst.hpp"

void two_level_active_scheduler::do_on_warp_issued(
    unsigned warp_id, unsigned num_issued,
    const std::vector<trace_shd_warp_t *>::const_iterator &prioritized_iter) {
  scheduler_unit::do_on_warp_issued(warp_id, num_issued, prioritized_iter);
  if (SCHEDULER_PRIORITIZATION_LRR == m_inner_level_prioritization) {
    std::vector<trace_shd_warp_t *> new_active;
    order_lrr(new_active, m_next_cycle_prioritized_warps, prioritized_iter,
              m_next_cycle_prioritized_warps.size());
    m_next_cycle_prioritized_warps = new_active;
  } else {
    fprintf(stderr, "Unimplemented m_inner_level_prioritization: %d\n",
            m_inner_level_prioritization);
    abort();
  }
}

void two_level_active_scheduler::order_warps() {
  // Move waiting warps to m_pending_warps
  unsigned num_demoted = 0;
  for (std::vector<trace_shd_warp_t *>::iterator iter =
           m_next_cycle_prioritized_warps.begin();
       iter != m_next_cycle_prioritized_warps.end();) {
    bool waiting = (*iter)->waiting();
    for (int i = 0; i < MAX_INPUT_VALUES; i++) {
      const warp_inst_t *inst = (*iter)->ibuffer_next_inst();
      // Is the instruction waiting on a long operation?
      if (inst && inst->in[i] > 0 &&
          this->m_scoreboard->islongop((*iter)->get_warp_id(), inst->in[i])) {
        waiting = true;
      }
    }

    if (waiting) {
      m_pending_warps.push_back(*iter);
      iter = m_next_cycle_prioritized_warps.erase(iter);
      SCHED_DPRINTF("DEMOTED warp_id=%d, dynamic_warp_id=%d\n",
                    (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());
      ++num_demoted;
    } else {
      ++iter;
    }
  }

  // If there is space in m_next_cycle_prioritized_warps, promote the next
  // m_pending_warps
  unsigned num_promoted = 0;
  if (SCHEDULER_PRIORITIZATION_SRR == m_outer_level_prioritization) {
    while (m_next_cycle_prioritized_warps.size() < m_max_active_warps) {
      m_next_cycle_prioritized_warps.push_back(m_pending_warps.front());
      m_pending_warps.pop_front();
      SCHED_DPRINTF(
          "PROMOTED warp_id=%d, dynamic_warp_id=%d\n",
          (m_next_cycle_prioritized_warps.back())->get_warp_id(),
          (m_next_cycle_prioritized_warps.back())->get_dynamic_warp_id());
      ++num_promoted;
    }
  } else {
    fprintf(stderr, "Unimplemented m_outer_level_prioritization: %d\n",
            m_outer_level_prioritization);
    abort();
  }
  assert(num_promoted == num_demoted);
}
