#pragma once

#include "hal.hpp"
#include "mem_fetch.hpp"

struct evicted_block_info {
  new_addr_type m_block_addr;
  unsigned m_modified_size;

  new_addr_type m_allocation_start_addr;
  unsigned m_allocation_id;

  mem_access_byte_mask_t m_byte_mask;
  mem_access_sector_mask_t m_sector_mask;

  evicted_block_info() {
    m_block_addr = 0;
    m_modified_size = 0;
    m_byte_mask.reset();
    m_sector_mask.reset();

    m_allocation_id = 0;
    m_allocation_start_addr = 0;
  }
  void set_info(new_addr_type block_addr, unsigned modified_size) {
    m_block_addr = block_addr;
    m_modified_size = modified_size;
  }
  void set_info(new_addr_type block_addr, unsigned modified_size,
                mem_access_byte_mask_t byte_mask,
                mem_access_sector_mask_t sector_mask, unsigned alloc_id,
                new_addr_type alloc_start_addr) {
    m_block_addr = block_addr;
    m_modified_size = modified_size;
    m_byte_mask = byte_mask;
    m_sector_mask = sector_mask;
    m_allocation_id = alloc_id;
    m_allocation_start_addr = alloc_start_addr;
  }
};

std::ostream &operator<<(std::ostream &os, const evicted_block_info &info);
