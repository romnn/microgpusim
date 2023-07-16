#pragma once

#include <cstdint>
#include <vector>

struct accelsim_stats {
  int number;
  uint64_t l2_total_cache_accesses;
  uint64_t l2_total_cache_misses;
  double l2_total_cache_miss_rate;
  uint64_t l2_total_cache_pending_hits;
  uint64_t l2_total_cache_reservation_fails;

  std::vector<std::vector<unsigned long long>> stats;
};
