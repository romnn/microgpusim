#include "mem_fetch.hpp"

std::shared_ptr<mem_fetch_bridge> new_mem_fetch_bridge(const mem_fetch *ptr) {
  return std::make_shared<mem_fetch_bridge>(ptr);
}
