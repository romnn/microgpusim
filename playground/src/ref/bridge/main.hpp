#pragma once

#include "accelsim_config.hpp"
#include "rust/cxx.h"

// extern const char *g_accelsim_version;

// int accelsim(accelsim_config config, const std::vector<std::string> &argv);
// int accelsim(accelsim_config config, rust::Slice<const rust::Str> argv);
int accelsim(accelsim_config config, rust::Slice<const rust::Str> argv);
