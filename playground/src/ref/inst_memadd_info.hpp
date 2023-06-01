#pragma once

#include <bitset>

#include "hal.hpp"

struct inst_memadd_info_t {
  uint64_t addrs[WARP_SIZE];
  int32_t width;

  void base_stride_decompress(unsigned long long base_address, int stride,
                              const std::bitset<WARP_SIZE> &mask);
  void base_delta_decompress(unsigned long long base_address,
                             const std::vector<long long> &deltas,
                             const std::bitset<WARP_SIZE> &mask);
};
