#pragma once

#include "accelsim_config.hpp"
#include "accelsim_stats.hpp"
#include "rust/cxx.h"

struct AccelsimStats;

// std::unique_ptr<accelsim_stats> accelsim(accelsim_config config,
// rust::Slice<const rust::Str> argv);
int accelsim(accelsim_config config, rust::Slice<const rust::Str> argv,
             AccelsimStats &stats);
