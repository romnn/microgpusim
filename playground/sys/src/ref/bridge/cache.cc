#include "cache.hpp"

std::shared_ptr<cache_block_bridge> new_cache_block_bridge(
    const cache_block_t *ptr) {
  return std::make_shared<cache_block_bridge>(ptr);
}

std::shared_ptr<cache_bridge> new_cache_bridge(const baseline_cache *ptr) {
  return std::make_shared<cache_bridge>(ptr);
}
