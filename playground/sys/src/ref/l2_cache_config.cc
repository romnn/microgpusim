#include "l2_cache_config.hpp"

void l2_cache_config::init(linear_to_raw_address_translation *address_mapping) {
  cache_config::init(m_config_string, FuncCachePreferNone);
  m_address_mapping = address_mapping;
}

unsigned l2_cache_config::set_index(new_addr_type addr) const {
  new_addr_type part_addr = addr;

  if (m_address_mapping) {
    // Calculate set index without memory partition bits to reduce set camping
    part_addr = m_address_mapping->partition_address(addr);
    fmt::println("partition address for addr {} is {}", addr, part_addr);
  }

  return cache_config::set_index(part_addr);
}
