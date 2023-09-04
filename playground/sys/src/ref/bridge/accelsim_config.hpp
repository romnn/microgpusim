#pragma once

#include "../cache_config.hpp"

struct accelsim_config {
  bool print_stats;
  bool accelsim_compat_mode;
  const char* stats_file;
};
