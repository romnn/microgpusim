#pragma once

#include <cstdio>
#include <queue>
#include <set>
#include <zlib.h>

#include "fifo.hpp"
#include "hal.hpp"
#include "l2_cache.hpp"

class memory_config;
class partition_mf_allocator;

class memory_sub_partition {
public:
  memory_sub_partition(unsigned sub_partition_id, const memory_config *config,
                       class memory_stats_t *stats, class trace_gpgpu_sim *gpu);
  ~memory_sub_partition();

  unsigned get_id() const { return m_id; }

  bool busy() const;

  void cache_cycle(unsigned cycle);

  bool full() const;
  bool full(unsigned size) const;
  void push(class mem_fetch *mf, unsigned long long clock_cycle);
  class mem_fetch *pop();
  class mem_fetch *top();
  void set_done(mem_fetch *mf);

  unsigned flushL2();
  unsigned invalidateL2();

  // interface to L2_dram_queue
  bool L2_dram_queue_empty() const;
  class mem_fetch *L2_dram_queue_top() const;
  void L2_dram_queue_pop();

  // interface to dram_L2_queue
  bool dram_L2_queue_full() const;
  void dram_L2_queue_push(class mem_fetch *mf);

  void visualizer_print(gzFile visualizer_file);
  void print_cache_stat(unsigned &accesses, unsigned &misses) const;
  void print(FILE *fp) const;

  void accumulate_L2cache_stats(class cache_stats &l2_stats) const;
  void get_L2cache_sub_stats(struct cache_sub_stats &css) const;

  // Support for getting per-window L2 stats for AerialVision
  void get_L2cache_sub_stats_pw(struct cache_sub_stats_pw &css) const;
  void clear_L2cache_stats_pw();

  void force_l2_tag_update(new_addr_type addr, unsigned time,
                           mem_access_sector_mask_t mask) {
    m_L2cache->force_tag_access(addr, m_memcpy_cycle_offset + time, mask);
    m_memcpy_cycle_offset += 1;
  }

private:
  // data
  unsigned m_id; //< the global sub partition ID
  const memory_config *m_config;
  class l2_cache *m_L2cache;
  class L2interface *m_L2interface;
  class trace_gpgpu_sim *m_gpu;
  partition_mf_allocator *m_mf_allocator;

  // model delay of ROP units with a fixed latency
  struct rop_delay_t {
    unsigned long long ready_cycle;
    class mem_fetch *req;
  };
  std::queue<rop_delay_t> m_rop;

  // these are various FIFOs between units within a memory partition
  fifo_pipeline<mem_fetch> *m_icnt_L2_queue;
  fifo_pipeline<mem_fetch> *m_L2_dram_queue;
  fifo_pipeline<mem_fetch> *m_dram_L2_queue;
  fifo_pipeline<mem_fetch> *m_L2_icnt_queue; // L2 cache hit response queue

  class mem_fetch *L2dramout;
  unsigned long long int wb_addr;

  class memory_stats_t *m_stats;

  std::set<mem_fetch *> m_request_tracker;

  friend class L2interface;

  std::vector<mem_fetch *> breakdown_request_to_sector_requests(mem_fetch *mf);

  // This is a cycle offset that has to be applied to the l2 accesses to account
  // for the cudamemcpy read/writes. We want GPGPU-Sim to only count cycles for
  // kernel execution but we want cudamemcpy to go through the L2. Everytime an
  // access is made from cudamemcpy this counter is incremented, and when the l2
  // is accessed (in both cudamemcpyies and otherwise) this value is added to
  // the gpgpu-sim cycle counters.
  unsigned m_memcpy_cycle_offset;
};
