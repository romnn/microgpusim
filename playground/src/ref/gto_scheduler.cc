#include "gto_scheduler.hpp"

void gto_scheduler::order_warps() {
  order_by_priority(m_next_cycle_prioritized_warps, m_supervised_warps,
                    m_last_supervised_issued, m_supervised_warps.size(),
                    ORDERING_GREEDY_THEN_PRIORITY_FUNC,
                    scheduler_unit::sort_warps_by_oldest_dynamic_id);
}
