#include "register_set.hpp"

std::shared_ptr<register_set_bridge> new_register_set_bridge(
    const register_set *ptr) {
  return std::make_shared<register_set_bridge>(ptr);
}
