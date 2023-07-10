#pragma once

#include "accelsim_config.hpp"
#include "accelsim_stats.hpp"
#include "rust/cxx.h"
#include "playground/src/bridge/main.rs.h"

// struct AccelsimStats;

int accelsim(accelsim_config config, rust::Slice<const rust::Str> argv,
             Stats &stats);
