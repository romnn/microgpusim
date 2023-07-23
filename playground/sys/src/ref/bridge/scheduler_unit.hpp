#pragma once

#include "../scheduler_unit.hpp"

struct scheduler_unit_ptr {
  const scheduler_unit *ptr;
  const scheduler_unit *get() const { return ptr; }
};

class scheduler_unit_bridge {
 public:
  scheduler_unit_bridge(const scheduler_unit *ptr) : ptr(ptr) {}

  const scheduler_unit *inner() const { return ptr; }

  std::unique_ptr<std::vector<unsigned>> get_prioritized_warp_ids() const {
    std::vector<unsigned> out;
    const std::vector<trace_shd_warp_t *> &warps =
        ptr->m_next_cycle_prioritized_warps;
    std::vector<trace_shd_warp_t *>::const_iterator iter;
    for (iter = warps.begin(); iter != warps.end(); iter++) {
      out.push_back((*iter)->get_warp_id());
    }
    return std::make_unique<std::vector<unsigned>>(out);
  }

 private:
  const class scheduler_unit *ptr;
};

std::shared_ptr<scheduler_unit_bridge> new_scheduler_unit_bridge(
    const scheduler_unit *ptr);
