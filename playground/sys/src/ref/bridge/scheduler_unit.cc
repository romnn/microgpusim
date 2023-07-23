#include "scheduler_unit.hpp"

std::shared_ptr<scheduler_unit_bridge> new_scheduler_unit_bridge(
    const scheduler_unit *ptr) {
  return std::make_shared<scheduler_unit_bridge>(ptr);
}
