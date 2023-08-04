#pragma once

#include <map>

#include "cache_block.hpp"
#include "cache_config.hpp"
#include "cache_request_status.hpp"
#include "evicted_block_info.hpp"
#include "hal.hpp"
#include "mem_fetch.hpp"

class tag_array {
 public:
  // Use this constructor
  tag_array(cache_config &config, int core_id, int type_id,
            bool accelsim_compat_mode, std::shared_ptr<spdlog::logger> logger);
  ~tag_array();

  friend class cache_bridge;

  enum cache_request_status probe(new_addr_type addr, unsigned &idx,
                                  mem_fetch *mf, bool is_write,
                                  bool probe_mode = false) const;
  enum cache_request_status probe(new_addr_type addr, unsigned &idx,
                                  mem_access_sector_mask_t mask, bool is_write,
                                  bool probe_mode = false,
                                  mem_fetch *mf = NULL) const;
  enum cache_request_status access(new_addr_type addr, unsigned time,
                                   unsigned &idx, mem_fetch *mf);
  enum cache_request_status access(new_addr_type addr, unsigned time,
                                   unsigned &idx, bool &wb,
                                   evicted_block_info &evicted, mem_fetch *mf);

  void fill(new_addr_type addr, unsigned time, mem_fetch *mf, bool is_write);
  void fill(unsigned idx, unsigned time, mem_fetch *mf);
  void fill(new_addr_type addr, unsigned time, mem_access_sector_mask_t mask,
            mem_access_byte_mask_t byte_mask, bool is_write);

  unsigned size() const { return m_config.get_num_lines(); }
  cache_block_t *get_block(unsigned idx) { return m_lines[idx]; }

  void flush();       // flush all written entries
  void invalidate();  // invalidate all entries
  void new_window();

  void print(FILE *stream, unsigned &total_access,
             unsigned &total_misses) const;
  float windowed_miss_rate() const;
  void get_stats(unsigned &total_access, unsigned &total_misses,
                 unsigned &total_hit_res, unsigned &total_res_fail) const;

  void update_cache_parameters(cache_config &config);
  void add_pending_line(mem_fetch *mf);
  void remove_pending_line(mem_fetch *mf);
  void inc_dirty() { m_dirty++; }

  bool accelsim_compat_mode;
  std::shared_ptr<spdlog::logger> logger;

 protected:
  // This constructor is intended for use only from derived classes that
  // wish to avoid unnecessary memory allocation that takes place in the
  // other tag_array constructor
  // tag_array(cache_config &config, int core_id, int type_id,
  //           cache_block_t **new_lines);
  void init(int core_id, int type_id);

 protected:
  cache_config &m_config;

  cache_block_t **m_lines;  // nbanks x nset x assoc lines in total

  unsigned m_access;
  unsigned m_miss;
  unsigned m_pending_hit;  // number of cache miss that hit a line that is
                           // allocated but not filled
  unsigned m_res_fail;
  unsigned m_sector_miss;
  unsigned m_dirty;

  int m_core_id;  // which shader core is using this
  int m_type_id;  // what kind of cache is this (normal, texture, constant)

  bool m_is_used;  // a flag if the whole cache has ever been accessed before

  typedef std::map<new_addr_type, unsigned> line_table;
  line_table pending_lines;
};
