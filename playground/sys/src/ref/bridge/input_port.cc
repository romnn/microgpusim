#include "input_port.hpp"

std::shared_ptr<input_port_bridge> new_input_port_bridge(
    const input_port_t *ptr) {
  return std::make_shared<input_port_bridge>(ptr);
}
