#pragma once

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
} parse_cache_config_config;

extern "C" parse_cache_config_config parse_cache_config(const char *config);
