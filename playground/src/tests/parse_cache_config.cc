#include <stdio.h>

#include "parse_cache_config.hpp"

extern "C" parse_cache_config_config parse_cache_config(const char *config) {
  parse_cache_config_config dest{0};
  sscanf(config, "%c:%u:%u:%u,%c:%c:%c:%c:%c,%c:%u:%u,%u:%u,%u", &dest.ct,
         &dest.m_nset, &dest.m_line_sz, &dest.m_assoc, &dest.rp, &dest.wp,
         &dest.ap, &dest.wap, &dest.sif, &dest.mshr_type, &dest.m_mshr_entries,
         &dest.m_mshr_max_merge, &dest.m_miss_queue_size,
         &dest.m_result_fifo_entries, &dest.m_data_port_width);
  return dest;
}
