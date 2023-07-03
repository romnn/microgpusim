#pragma once

#include "cache_block.hpp"
#include "hal.hpp"

struct line_cache_block : public cache_block_t {
  line_cache_block() {
    m_alloc_time = 0;
    m_fill_time = 0;
    m_last_access_time = 0;
    m_status = INVALID;
    m_ignore_on_fill_status = false;
    m_set_modified_on_fill = false;
    m_set_readable_on_fill = false;
    m_readable = true;
  }
  void allocate(new_addr_type tag, new_addr_type block_addr, unsigned time,
                mem_access_sector_mask_t sector_mask) {
    m_tag = tag;
    m_block_addr = block_addr;
    m_alloc_time = time;
    m_last_access_time = time;
    m_fill_time = 0;
    m_status = RESERVED;
    m_ignore_on_fill_status = false;
    m_set_modified_on_fill = false;
    m_set_readable_on_fill = false;
    m_set_byte_mask_on_fill = false;
  }
  virtual void fill(unsigned time, mem_access_sector_mask_t sector_mask,
                    mem_access_byte_mask_t byte_mask) {
    // if(!m_ignore_on_fill_status)
    //	assert( m_status == RESERVED );

    m_status = m_set_modified_on_fill ? MODIFIED : VALID;

    if (m_set_readable_on_fill)
      m_readable = true;
    if (m_set_byte_mask_on_fill)
      set_byte_mask(byte_mask);

    m_fill_time = time;
  }
  virtual bool is_invalid_line() { return m_status == INVALID; }
  virtual bool is_valid_line() { return m_status == VALID; }
  virtual bool is_reserved_line() { return m_status == RESERVED; }
  virtual bool is_modified_line() { return m_status == MODIFIED; }

  virtual enum cache_block_state
  get_status(mem_access_sector_mask_t sector_mask) {
    return m_status;
  }
  virtual void set_status(enum cache_block_state status,
                          mem_access_sector_mask_t sector_mask) {
    m_status = status;
  }
  virtual void set_byte_mask(mem_fetch *mf) {
    m_dirty_byte_mask = m_dirty_byte_mask | mf->get_access_byte_mask();
  }
  virtual void set_byte_mask(mem_access_byte_mask_t byte_mask) {
    m_dirty_byte_mask = m_dirty_byte_mask | byte_mask;
  }
  virtual mem_access_byte_mask_t get_dirty_byte_mask() {
    return m_dirty_byte_mask;
  }
  virtual mem_access_sector_mask_t get_dirty_sector_mask() {
    mem_access_sector_mask_t sector_mask;
    if (m_status == MODIFIED)
      sector_mask.set();
    return sector_mask;
  }
  virtual unsigned long long get_last_access_time() {
    return m_last_access_time;
  }
  virtual void set_last_access_time(unsigned long long time,
                                    mem_access_sector_mask_t sector_mask) {
    m_last_access_time = time;
  }
  virtual unsigned long long get_alloc_time() { return m_alloc_time; }
  virtual void set_ignore_on_fill(bool m_ignore,
                                  mem_access_sector_mask_t sector_mask) {
    m_ignore_on_fill_status = m_ignore;
  }
  virtual void set_modified_on_fill(bool m_modified,
                                    mem_access_sector_mask_t sector_mask) {
    m_set_modified_on_fill = m_modified;
  }
  virtual void set_readable_on_fill(bool readable,
                                    mem_access_sector_mask_t sector_mask) {
    m_set_readable_on_fill = readable;
  }
  virtual void set_byte_mask_on_fill(bool m_modified) {
    m_set_byte_mask_on_fill = m_modified;
  }
  virtual unsigned get_modified_size() {
    return SECTOR_CHUNCK_SIZE * SECTOR_SIZE; // i.e. cache line size
  }
  virtual void set_m_readable(bool readable,
                              mem_access_sector_mask_t sector_mask) {
    m_readable = readable;
  }
  virtual bool is_readable(mem_access_sector_mask_t sector_mask) {
    return m_readable;
  }
  virtual void print_status() {
    printf("m_block_addr is %lu, status = %u\n", m_block_addr, m_status);
  }

private:
  unsigned long long m_alloc_time;
  unsigned long long m_last_access_time;
  unsigned long long m_fill_time;
  cache_block_state m_status;
  bool m_ignore_on_fill_status;
  bool m_set_modified_on_fill;
  bool m_set_readable_on_fill;
  bool m_set_byte_mask_on_fill;
  bool m_readable;
  mem_access_byte_mask_t m_dirty_byte_mask;
};
