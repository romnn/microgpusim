#pragma once

namespace Trace {

enum trace_streams_type {
  WARP_SCHEDULER,
  SCOREBOARD,
  MEMORY_PARTITION_UNIT,
  MEMORY_SUBPARTITION_UNIT,
  INTERCONNECT,
  LIVENESS,
  NUM_TRACE_STREAMS,
};

static const char *trace_streams_type_str[] = {
    "WARP_SCHEDULER",           "SCOREBOARD",   "MEMORY_PARTITION_UNIT",
    "MEMORY_SUBPARTITION_UNIT", "INTERCONNECT", "LIVENESS",
    "NUM_TRACE_STREAMS",
};

extern bool enabled;
extern int sampling_core;
extern int sampling_memory_partition;
extern const char *trace_streams_str[];
extern bool trace_streams_enabled[NUM_TRACE_STREAMS];
extern const char *config_str;

void init();

}  // namespace Trace

#if TRACING_ON

#define SIM_PRINT_STR "GPGPU-Sim Cycle %llu: %s - "
#define DTRACE(x) ((Trace::trace_streams_enabled[Trace::x]) && Trace::enabled)
#define DPRINTF(x, ...)                                                      \
  do {                                                                       \
    if (DTRACE(x)) {                                                         \
      printf(SIM_PRINT_STR, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle, \
             Trace::trace_streams_str[Trace::x]);                            \
      printf(__VA_ARGS__);                                                   \
    }                                                                        \
  } while (0)

#define DPRINTFG(x, ...)                                       \
  do {                                                         \
    if (DTRACE(x)) {                                           \
      printf(SIM_PRINT_STR, gpu_sim_cycle + gpu_tot_sim_cycle, \
             Trace::trace_streams_str[Trace::x]);              \
      printf(__VA_ARGS__);                                     \
    }                                                          \
  } while (0)

#else

#define DTRACE(x) (false)
#define DPRINTF(x, ...) \
  do {                  \
  } while (0)
#define DPRINTFG(x, ...) \
  do {                   \
  } while (0)

#endif
