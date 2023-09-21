#pragma once

#include "../baseline_cache.hpp"
#include "../sector_cache_block.hpp"
#include "../line_cache_block.hpp"

struct cache_block_ptr {
  const cache_block_t *ptr;
  const cache_block_t *get() const { return ptr; }
};

class cache_block_bridge {
 public:
  cache_block_bridge(const cache_block_t *ptr) : ptr(ptr){};

  const cache_block_t *inner() const { return ptr; }

  std::unique_ptr<std::vector<unsigned>> get_last_sector_access_time() const {
    const sector_cache_block *sector_block =
        dynamic_cast<const sector_cache_block *>(ptr);
    const line_cache_block *line_block =
        dynamic_cast<const line_cache_block *>(ptr);

    if (sector_block != NULL) {
      std::vector<unsigned> v(
          sector_block->m_last_sector_access_time,
          sector_block->m_last_sector_access_time + SECTOR_CHUNCK_SIZE);
      return std::make_unique<std::vector<unsigned>>(v);

    } else if (line_block != NULL) {
      std::vector<unsigned> v{line_block->m_status};
      return std::make_unique<std::vector<unsigned>>(v);

    } else {
      assert(0 && "cache block is neither sector nor line cache");
    }
  }

  std::unique_ptr<std::vector<cache_block_state>> get_sector_status() const {
    const sector_cache_block *sector_block =
        dynamic_cast<const sector_cache_block *>(ptr);
    const line_cache_block *line_block =
        dynamic_cast<const line_cache_block *>(ptr);

    if (sector_block != NULL) {
      std::vector<cache_block_state> v(
          sector_block->m_status, sector_block->m_status + SECTOR_CHUNCK_SIZE);
      return std::make_unique<std::vector<cache_block_state>>(v);

    } else if (line_block != NULL) {
      std::vector<cache_block_state> v{line_block->m_status};
      return std::make_unique<std::vector<cache_block_state>>(v);

    } else {
      assert(0 && "cache block is neither sector nor line cache");
    }
  }

 private:
  const cache_block_t *ptr;
};

std::shared_ptr<cache_block_bridge> new_cache_block_bridge(
    const cache_block_t *ptr);

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
