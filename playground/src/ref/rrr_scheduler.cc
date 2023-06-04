#include "rrr_scheduler.hpp"

void rrr_scheduler::order_warps() {
  order_rrr(m_next_cycle_prioritized_warps, m_supervised_warps,
            m_last_supervised_issued, m_supervised_warps.size());
}
