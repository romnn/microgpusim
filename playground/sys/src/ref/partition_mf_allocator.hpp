#pragma once

#include "mem_fetch_allocator.hpp"

class memory_config;

class partition_mf_allocator : public mem_fetch_allocator {
 public:
  partition_mf_allocator(const memory_config *config) {
    m_memory_config = config;
  }
  virtual mem_fetch *alloc(const class warp_inst_t &inst,
                           const mem_access_t &access,
                           unsigned long long cycle) const {
    assert(0 && "partition mem allocator without original mf used");
    abort();
    return NULL;
  }

  // virtual mem_fetch *alloc(new_addr_type addr, mem_access_type type,
  //                          unsigned size, bool wr,
  //                          unsigned long long cycle) const;
  virtual mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                           unsigned size, bool wr,
                           unsigned long long cycle) const {
    assert(0 && "partition mem allocator without original mf used");
    assert(wr);
    mem_access_t access(type, addr, size, wr, m_memory_config->gpgpu_ctx);
    mem_fetch *mf = new mem_fetch(access, NULL, WRITE_PACKET_SIZE, -1, -1, -1,
                                  m_memory_config, cycle);
    return mf;
  }

  virtual mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                           const active_mask_t &active_mask,
                           const mem_access_byte_mask_t &byte_mask,
                           const mem_access_sector_mask_t &sector_mask,
                           unsigned size, bool wr, unsigned long long cycle,
                           unsigned wid, unsigned sid, unsigned tpc,
                           mem_fetch *original_mf) const;

 private:
  const memory_config *m_memory_config;
};
