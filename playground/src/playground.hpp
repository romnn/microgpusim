#pragma once

#include <stdio.h>

typedef struct CacheConfig {
  char ct;
  unsigned m_nset;
  unsigned m_line_sz;
  unsigned m_assoc;
  //
  char rp;
  char wp;
  char ap;
  char wap;
  char sif;
  //
  char mshr_type;
  unsigned m_mshr_entries;
  unsigned m_mshr_max_merge;
  unsigned m_miss_queue_size;
  unsigned m_result_fifo_entries;
  unsigned m_data_port_width;
} cache_config;

extern "C" void parse_cache_config(char *config, cache_config *dest);

typedef unsigned long long new_addr_type;
typedef unsigned long long address_type;
typedef unsigned long long addr_t;

extern "C" new_addr_type addrdec_packbits(new_addr_type mask, new_addr_type val,
                                          unsigned char high,
                                          unsigned char low);
