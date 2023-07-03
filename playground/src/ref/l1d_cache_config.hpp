#pragma once

#include <assert.h>

#include "cache.hpp"
#include "cache_config.hpp"
#include "hal.hpp"
#include "utils.hpp"

class l1d_cache_config : public cache_config {
 public:
  l1d_cache_config() : cache_config() {}
  unsigned set_bank(new_addr_type addr) const {
    // For sector cache, we select one sector per bank (sector interleaving)
    // This is what was found in Volta (one sector per bank, sector
    // interleaving) otherwise, line interleaving
    return cache_config::hash_function(
        addr, l1_banks, l1_banks_byte_interleaving_log2, l1_banks_log2,
        l1_banks_hashing_function);
  }
  void init(char *config, FuncCache status) {
    l1_banks_byte_interleaving_log2 = LOGB2(l1_banks_byte_interleaving);
    l1_banks_log2 = LOGB2(l1_banks);
    cache_config::init(config, status);
  }
  unsigned l1_latency;
  unsigned l1_banks;
  unsigned l1_banks_log2;
  unsigned l1_banks_byte_interleaving;
  unsigned l1_banks_byte_interleaving_log2;
  unsigned l1_banks_hashing_function;
  unsigned m_unified_cache_size;
  virtual unsigned get_max_cache_multiplier() const {
    // set * assoc * cacheline size. Then convert Byte to KB
    // gpgpu_unified_cache_size is in KB while original_sz is in B
    if (m_unified_cache_size > 0) {
      unsigned original_size = m_nset * original_m_assoc * m_line_sz / 1024;
      assert(m_unified_cache_size % original_size == 0);
      return m_unified_cache_size / original_size;
    } else {
      return MAX_DEFAULT_CACHE_SIZE_MULTIBLIER;
    }
  }
};
