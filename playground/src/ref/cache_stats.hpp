#pragma once

#include <stdio.h>
#include <vector>

#include "cache_request_status.hpp"
#include "mem_access_type.hpp"

///
/// Cache_stats
/// Used to record statistics for each cache.
/// Maintains a record of every 'mem_access_type' and its resulting
/// 'cache_request_status' : [mem_access_type][cache_request_status]
///
class cache_stats {
 public:
  cache_stats();
  void clear();
  // Clear AerialVision cache stats after each window
  void clear_pw();
  void inc_stats(int access_type, int access_outcome);
  // Increment AerialVision cache stats
  void inc_stats_pw(int access_type, int access_outcome);
  void inc_fail_stats(int access_type, int fail_outcome);
  enum cache_request_status select_stats_status(
      enum cache_request_status probe, enum cache_request_status access) const;
  unsigned long long &operator()(int access_type, int access_outcome,
                                 bool fail_outcome);
  unsigned long long operator()(int access_type, int access_outcome,
                                bool fail_outcome) const;
  cache_stats operator+(const cache_stats &cs);
  cache_stats &operator+=(const cache_stats &cs);
  void print_stats(FILE *fout, const char *cache_name = "Cache_stats") const;
  void print_fail_stats(FILE *fout,
                        const char *cache_name = "Cache_fail_stats") const;

  unsigned long long get_stats(enum mem_access_type *access_type,
                               unsigned num_access_type,
                               enum cache_request_status *access_status,
                               unsigned num_access_status) const;
  void get_sub_stats(struct cache_sub_stats &css) const;

  // Get per-window cache stats for AerialVision
  void get_sub_stats_pw(struct cache_sub_stats_pw &css) const;

  void sample_cache_port_utility(bool data_port_busy, bool fill_port_busy);

 private:
  bool check_valid(int type, int status) const;
  bool check_fail_valid(int type, int fail) const;

  std::vector<std::vector<unsigned long long>> m_stats;
  // AerialVision cache stats (per-window)
  std::vector<std::vector<unsigned long long>> m_stats_pw;
  std::vector<std::vector<unsigned long long>> m_fail_stats;

  unsigned long long m_cache_port_available_cycles;
  unsigned long long m_cache_data_port_busy_cycles;
  unsigned long long m_cache_fill_port_busy_cycles;
};
