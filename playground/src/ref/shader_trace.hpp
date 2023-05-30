#pragma once

#include "trace.hpp"

#if TRACING_ON

#define SHADER_PRINT_STR SIM_PRINT_STR "Core %d - "
#define SCHED_PRINT_STR SHADER_PRINT_STR "Scheduler %d - "
#define SHADER_DTRACE(x)                                                       \
  (DTRACE(x) &&                                                                \
   (Trace::sampling_core == get_sid() || Trace::sampling_core == -1))

// Intended to be called from inside components of a shader core.
// Depends on a get_sid() function
#define SHADER_DPRINTF(x, ...)                                                 \
  do {                                                                         \
    if (SHADER_DTRACE(x)) {                                                    \
      printf(SHADER_PRINT_STR,                                                 \
             m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle,                  \
             Trace::trace_streams_str[Trace::x], get_sid());                   \
      printf(__VA_ARGS__);                                                     \
    }                                                                          \
  } while (0)

// Intended to be called from inside a scheduler_unit.
// Depends on a m_id member
#define SCHED_DPRINTF(...)                                                     \
  do {                                                                         \
    if (SHADER_DTRACE(WARP_SCHEDULER)) {                                       \
      printf(SCHED_PRINT_STR,                                                  \
             m_shader->get_gpu()->gpu_sim_cycle +                              \
                 m_shader->get_gpu()->gpu_tot_sim_cycle,                       \
             Trace::trace_streams_str[Trace::WARP_SCHEDULER], get_sid(),       \
             m_id);                                                            \
      printf(__VA_ARGS__);                                                     \
    }                                                                          \
  } while (0)

#else

#define SHADER_DTRACE(x) (false)
#define SHADER_DPRINTF(x, ...)                                                 \
  do {                                                                         \
  } while (0)
#define SCHED_DPRINTF(x, ...)                                                  \
  do {                                                                         \
  } while (0)

#endif
