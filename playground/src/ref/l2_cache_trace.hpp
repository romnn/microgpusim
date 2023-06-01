#pragma once

#if TRACING_ON

#define MEMPART_PRINT_STR SIM_PRINT_STR " %d - "
#define MEMPART_DTRACE(x)                                                      \
  (DTRACE(x) && (Trace::sampling_memory_partition == -1 ||                     \
                 Trace::sampling_memory_partition == (int)get_mpid()))

#define MEM_SUBPART_PRINT_STR SIM_PRINT_STR " %d - "
#define MEM_SUBPART_DTRACE(x)                                                  \
  (DTRACE(x) && (Trace::sampling_memory_partition == -1 ||                     \
                 Trace::sampling_memory_partition == (int)m_id))

// Intended to be called from inside components of a memory partition
// Depends on a get_mpid() function
#define MEMPART_DPRINTF(...)                                                   \
  do {                                                                         \
    if (MEMPART_DTRACE(MEMORY_PARTITION_UNIT)) {                               \
      printf(                                                                  \
          MEMPART_PRINT_STR, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle,  \
          Trace::trace_streams_str[Trace::MEMORY_PARTITION_UNIT], get_mpid()); \
      printf(__VA_ARGS__);                                                     \
    }                                                                          \
  } while (0)

#define MEM_SUBPART_DPRINTF(...)                                               \
  do {                                                                         \
    if (MEM_SUBPART_DTRACE(MEMORY_PARTITION_UNIT)) {                           \
      printf(MEM_SUBPART_PRINT_STR,                                            \
             m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle,                  \
             Trace::trace_streams_str[Trace::MEMORY_SUBPARTITION_UNIT], m_id); \
      printf(__VA_ARGS__);                                                     \
    }                                                                          \
  } while (0)

#else

#define MEMPART_DTRACE(x) (false)
#define MEMPART_DPRINTF(x, ...)                                                \
  do {                                                                         \
  } while (0)

#define MEM_SUBPART_DTRACE(x) (false)
#define MEM_SUBPART_DPRINTF(x, ...)                                            \
  do {                                                                         \
  } while (0)

#endif
