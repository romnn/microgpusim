#pragma once

#include "../baseline_cache.hpp"

struct cache_block_ptr {
  const cache_block_t *ptr;
  const cache_block_t *get() const { return ptr; }
};

class cache_bridge {
 public:
  cache_bridge(const baseline_cache *ptr) : ptr(ptr) {}

  const baseline_cache *inner() const { return ptr; }

  std::unique_ptr<std::vector<cache_block_ptr>> get_lines() const {
    std::vector<cache_block_ptr> out;
    unsigned cache_lines_num = ptr->m_tag_array->m_config.get_max_num_lines();
    for (unsigned i = 0; i < cache_lines_num; i++) {
      out.push_back(cache_block_ptr{ptr->m_tag_array->m_lines[i]});
    }
    return std::make_unique<std::vector<cache_block_ptr>>(out);
  }

 private:
  const baseline_cache *ptr;
};

std::shared_ptr<cache_bridge> new_cache_bridge(const baseline_cache *ptr);
