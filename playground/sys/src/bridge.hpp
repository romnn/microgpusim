#pragma once

#include <stdint.h>
#include "bindings.hpp"
#include "rust/cxx.h"

// #include "./ref/bridge/mem_fetch.hpp"
// #include "./ref/bridge/register_set.hpp"
// #include "./ref/bridge/input_port.hpp"
// #include "./ref/bridge/main.hpp"
// #include "./ref/bridge/core.hpp"
// #include "./ref/bridge/mem_fetch.hpp"
// #include "./ref/bridge/memory_partition_unit.hpp"
// #include "./ref/bridge/operand_collector.hpp"
// #include "./ref/bridge/scheduler_unit.hpp"
// #include "./ref/bridge/warp_inst.hpp"

using c_void = void;
using c_ulonglong = long long unsigned int;

/*
 * For the FFI boundary, cxx static asserts do not work with all types.
 *
 * e.g. unsigned long long (on the C++ side) != u64 (on the rust side)
 *
 * therefore, we use standardized uint64_t where necessary and use the
 * following static assertions to make sure we are indeed running on a
 * modern system where
 *
 *		uint64_t == unsigned long == unsigned long long
 *
 * holds.
 */
static_assert(sizeof(unsigned long) == sizeof(uint64_t),
              "replaced unsigned long with uint64_t");
static_assert(sizeof(unsigned long long) == sizeof(uint64_t),
              "replaced unsigned long long with uint64_t");
static_assert(sizeof(unsigned long) == sizeof(unsigned long long),
              "replaced unsigned long long with unsigned long");
static_assert(sizeof(long int) == sizeof(int64_t),
              "replaced long int with int32_t");
static_assert(sizeof(unsigned int) == sizeof(uint32_t),
              "replaced unsigned int with uint32_t");
static_assert(sizeof(unsigned) == sizeof(uint32_t),
              "replaced unsigned with uint32_t");
