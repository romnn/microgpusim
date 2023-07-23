#pragma once

#include "bindings.hpp"
#include "rust/cxx.h"

#include "./ref/bridge/mem_fetch.hpp"
#include "./ref/bridge/register_set.hpp"
#include "./ref/bridge/input_port.hpp"
#include "./ref/bridge/main.hpp"
#include "./ref/bridge/core.hpp"
#include "./ref/bridge/mem_fetch.hpp"
#include "./ref/bridge/memory_partition_unit.hpp"
#include "./ref/bridge/operand_collector.hpp"
#include "./ref/bridge/scheduler_unit.hpp"
#include "./ref/bridge/trace_parser.hpp"
#include "./ref/bridge/warp_inst.hpp"

using c_void = void;
using c_ulonglong = long long unsigned int;

// this should only be added into source files (.cc) that require rust types
// #include "playground/src/bridge.rs.h"
