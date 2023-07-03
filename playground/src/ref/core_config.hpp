#pragma once

#include "gpgpu_context.hpp"
#include "hal.hpp"

class core_config {
public:
  core_config(gpgpu_context *ctx) {
    gpgpu_ctx = ctx;
    m_valid = false;
    num_shmem_bank = 16;
    shmem_limited_broadcast = false;
    gpgpu_shmem_sizeDefault = (unsigned)-1;
    gpgpu_shmem_sizePrefL1 = (unsigned)-1;
    gpgpu_shmem_sizePrefShared = (unsigned)-1;
  }
  virtual void init() = 0;

  bool m_valid;
  unsigned warp_size;
  // backward pointer
  class gpgpu_context *gpgpu_ctx;

  // off-chip memory request architecture parameters
  int gpgpu_coalesce_arch;

  // shared memory bank conflict checking parameters
  bool shmem_limited_broadcast;
  static const address_type WORD_SIZE = 4;
  unsigned num_shmem_bank;
  unsigned shmem_bank_func(address_type addr) const {
    return ((addr / WORD_SIZE) % num_shmem_bank);
  }
  unsigned mem_warp_parts;
  mutable unsigned gpgpu_shmem_size;
  char *gpgpu_shmem_option;
  std::vector<unsigned> shmem_opt_list;
  unsigned gpgpu_shmem_sizeDefault;
  unsigned gpgpu_shmem_sizePrefL1;
  unsigned gpgpu_shmem_sizePrefShared;
  unsigned mem_unit_ports;

  // texture and constant cache line sizes (used to determine number of memory
  // accesses)
  unsigned gpgpu_cache_texl1_linesize;
  unsigned gpgpu_cache_constl1_linesize;

  unsigned gpgpu_max_insn_issue_per_warp;
  bool gmem_skip_L1D; // on = global memory access always skip the L1 cache

  bool adaptive_cache_config;
};
