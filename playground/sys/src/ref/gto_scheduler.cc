#include "gto_scheduler.hpp"

void gto_scheduler::order_warps() {
  if (m_supervised_warps.size() >= 2) {
    trace_shd_warp_t *first_warp = m_supervised_warps[0];
    trace_shd_warp_t *second_warp = m_supervised_warps[1];

    assert(first_warp != NULL && "gto: first warp is not NULL");
    assert(second_warp != NULL && "gto: second warp is not NULL");
    assert(m_shader != NULL && "gto: shader is not NULL");

    if (first_warp->get_warp_id() != (unsigned)-1 &&
        second_warp->get_warp_id() != (unsigned)-1) {
      // fmt::println(
      logger->warn(
          "first: ({},{}) done={} waiting={} at barrier={} at mem barrier={} "
          "(has barrier={} outstanding stores={})",
          first_warp->get_warp_id(), first_warp->get_dynamic_warp_id(),
          first_warp->done_exit(), first_warp->waiting(),
          m_shader->warp_waiting_at_barrier(first_warp->get_warp_id()),
          m_shader->warp_waiting_at_mem_barrier(first_warp->get_warp_id()),
          m_shader->m_warp[first_warp->get_warp_id()]->get_membar(),
          m_scoreboard->num_pending_writes(second_warp->get_warp_id()));

      // fmt::println(
      logger->warn(
          "second: ({},{}) done={} waiting={} at barrier={} at mem barrier={} "
          "(has barrier={} outstanding stores={})",
          second_warp->get_warp_id(), second_warp->get_dynamic_warp_id(),
          second_warp->done_exit(), second_warp->waiting(),
          m_shader->warp_waiting_at_barrier(second_warp->get_warp_id()),
          m_shader->warp_waiting_at_mem_barrier(second_warp->get_warp_id()),
          m_shader->m_warp[second_warp->get_warp_id()]->get_membar(),
          m_scoreboard->num_pending_writes(second_warp->get_warp_id()));
    }
  }

  order_by_priority(m_next_cycle_prioritized_warps, m_supervised_warps,
                    m_last_supervised_issued, m_supervised_warps.size(),
                    ORDERING_GREEDY_THEN_PRIORITY_FUNC,
                    scheduler_unit::sort_warps_by_oldest_dynamic_id);
}
