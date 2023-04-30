#include "playground.hpp"
#include <stdio.h>

extern "C" void parse_cache_config(char *config, cache_config *dest) {
  sscanf(config, "%c:%u:%u:%u,%c:%c:%c:%c:%c,%c:%u:%u,%u:%u,%u", &dest->ct,
         &dest->m_nset, &dest->m_line_sz, &dest->m_assoc, &dest->rp, &dest->wp,
         &dest->ap, &dest->wap, &dest->sif, &dest->mshr_type,
         &dest->m_mshr_entries, &dest->m_mshr_max_merge,
         &dest->m_miss_queue_size, &dest->m_result_fifo_entries,
         &dest->m_data_port_width);
  printf("test");
}

extern "C" new_addr_type addrdec_packbits(new_addr_type mask, new_addr_type val,
                                          unsigned char high,
                                          unsigned char low) {
  unsigned pos = 0;
  new_addr_type result = 0;
  for (unsigned i = low; i < high; i++) {
    if ((mask & ((unsigned long long int)1 << i)) != 0) {
      result |= ((val & ((unsigned long long int)1 << i)) >> i) << pos;
      pos++;
    }
  }
  return result;
}
