#include "warp_inst.hpp"

std::shared_ptr<warp_inst_bridge> new_warp_inst_bridge(const warp_inst_t *ptr) {
  return std::make_shared<warp_inst_bridge>(ptr);
}
