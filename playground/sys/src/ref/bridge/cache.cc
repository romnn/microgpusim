#include "cache.hpp"

std::shared_ptr<cache_bridge> new_cache_bridge(const baseline_cache *ptr) {
  return std::make_shared<cache_bridge>(ptr);
}
