#pragma once

#include "hal.hpp"
#include "mem_fetch.hpp"

enum cache_block_state { INVALID = 0, RESERVED, VALID, MODIFIED };

static const char *cache_block_state_str[] = {
    "INVALID",
    "RESERVED",
    "VALID",
    "MODIFIED",
};

struct cache_block_t {
  cache_block_t() {
    m_tag = 0;
    m_block_addr = 0;
  }

  virtual void allocate(new_addr_type tag, new_addr_type block_addr,
                        unsigned time,
                        mem_access_sector_mask_t sector_mask) = 0;
  virtual void fill(unsigned time, mem_access_sector_mask_t sector_mask,
                    mem_access_byte_mask_t byte_mask) = 0;

  virtual bool is_invalid_line() const = 0;
  virtual bool is_valid_line() const = 0;
  virtual bool is_reserved_line() const = 0;
  virtual bool is_modified_line() const = 0;

  virtual enum cache_block_state get_status(
      mem_access_sector_mask_t sector_mask) = 0;
  virtual void set_status(enum cache_block_state m_status,
                          mem_access_sector_mask_t sector_mask) = 0;
  virtual void set_byte_mask(mem_fetch *mf) = 0;
  virtual void set_byte_mask(mem_access_byte_mask_t byte_mask) = 0;
  virtual mem_access_byte_mask_t get_dirty_byte_mask() = 0;
  virtual mem_access_sector_mask_t get_dirty_sector_mask() = 0;
  virtual unsigned long long get_last_access_time() = 0;
  virtual void set_last_access_time(unsigned long long time,
                                    mem_access_sector_mask_t sector_mask) = 0;
  virtual unsigned long long get_alloc_time() = 0;
  virtual void set_ignore_on_fill(bool m_ignore,
                                  mem_access_sector_mask_t sector_mask) = 0;
  virtual void set_modified_on_fill(bool m_modified,
                                    mem_access_sector_mask_t sector_mask) = 0;
  virtual void set_readable_on_fill(bool readable,
                                    mem_access_sector_mask_t sector_mask) = 0;
  virtual void set_byte_mask_on_fill(bool m_modified) = 0;
  virtual unsigned get_modified_size() = 0;
  virtual void set_m_readable(bool readable,
                              mem_access_sector_mask_t sector_mask) = 0;
  virtual bool is_readable(mem_access_sector_mask_t sector_mask) = 0;
  virtual void print_status() = 0;
  virtual ~cache_block_t() {}

  new_addr_type m_tag;
  new_addr_type m_block_addr;

  new_addr_type get_tag() const { return m_tag; }
  new_addr_type get_block_addr() const { return m_block_addr; }
};
