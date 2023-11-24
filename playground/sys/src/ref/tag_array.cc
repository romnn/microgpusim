#include <assert.h>

#include "cache.hpp"
#include "cache_block.hpp"
#include "cache_request_status.hpp"
#include "line_cache_block.hpp"
#include "sector_cache_block.hpp"
#include "tag_array.hpp"

tag_array::~tag_array() {
  unsigned cache_lines_num = m_config.get_max_num_lines();
  for (unsigned i = 0; i < cache_lines_num; ++i) delete m_lines[i];
  delete[] m_lines;
}

// tag_array::tag_array(cache_config &config, int core_id, int type_id,
//                      cache_block_t **new_lines)
//     : m_config(config), m_lines(new_lines) {
//   init(core_id, type_id);
// }

void tag_array::update_cache_parameters(cache_config &config) {
  m_config = config;
}

tag_array::tag_array(cache_config &config, int core_id, int type_id,
                     bool accelsim_compat_mode,
                     std::shared_ptr<spdlog::logger> logger)
    : accelsim_compat_mode(accelsim_compat_mode),
      logger(logger),
      m_config(config) {
  // assert( m_config.m_write_policy == READ_ONLY ); Old assert
  unsigned cache_lines_num = config.get_max_num_lines();
  m_lines = new cache_block_t *[cache_lines_num];
  if (config.m_cache_type == NORMAL) {
    for (unsigned i = 0; i < cache_lines_num; ++i)
      m_lines[i] = new line_cache_block();
  } else if (config.m_cache_type == SECTOR) {
    for (unsigned i = 0; i < cache_lines_num; ++i)
      m_lines[i] = new sector_cache_block();
  } else
    assert(0);

  init(core_id, type_id);
}

void tag_array::init(int core_id, int type_id) {
  m_access = 0;
  m_miss = 0;
  m_pending_hit = 0;
  m_res_fail = 0;
  m_sector_miss = 0;
  m_core_id = core_id;
  m_type_id = type_id;
  m_is_used = false;
  m_dirty = 0;
}

void tag_array::add_pending_line(mem_fetch *mf) {
  logger->trace("tag_array::add_pending_line({})", mem_fetch_ptr(mf));
  assert(mf);
  new_addr_type addr = m_config.block_addr(mf->get_addr());
  line_table::const_iterator i = pending_lines.find(addr);
  if (i == pending_lines.end()) {
    pending_lines[addr] = mf->get_inst().get_uid();
  }
}

void tag_array::remove_pending_line(mem_fetch *mf) {
  logger->trace("tag_array::remove_pending_line({})", mem_fetch_ptr(mf));
  assert(mf);
  new_addr_type addr = m_config.block_addr(mf->get_addr());
  line_table::const_iterator i = pending_lines.find(addr);
  if (i != pending_lines.end()) {
    pending_lines.erase(addr);
  }
}

enum cache_request_status tag_array::probe(new_addr_type addr, unsigned &idx,
                                           mem_fetch *mf, bool is_write,
                                           bool probe_mode) const {
  mem_access_sector_mask_t mask = mf->get_access_sector_mask();
  return probe(addr, idx, mask, is_write, probe_mode, mf);
}

enum cache_request_status tag_array::probe(new_addr_type addr, unsigned &idx,
                                           mem_access_sector_mask_t mask,
                                           bool is_write, bool probe_mode,
                                           mem_fetch *mf) const {
  // assert( m_config.m_write_policy == READ_ONLY );
  unsigned set_index = m_config.set_index(addr);
  new_addr_type tag = m_config.tag(addr);
  logger->trace("tag_array::probe({}) set_idx = {} tag = {} assoc = {}",
                mem_fetch_ptr(mf), set_index, tag, m_config.m_assoc);

  if (accelsim_compat_mode) {
    // printf("tag_array::probe(%lu) set_idx = %d tag = %lu assoc = %d\n", addr,
    //        set_index, tag, m_config.m_assoc);
  }

  unsigned invalid_line = (unsigned)-1;
  unsigned valid_line = (unsigned)-1;
  unsigned long long valid_timestamp = (unsigned)-1;

  bool all_reserved = true;
  // check for hit or pending hit
  for (unsigned way = 0; way < m_config.m_assoc; way++) {
    unsigned index = set_index * m_config.m_assoc + way;
    cache_block_t *line = m_lines[index];

    logger->trace(
        "tag_array::probe({}) => checking cache index {} (tag={}, status={}, "
        "last_access={})",
        mem_fetch_ptr(mf), index, line->m_tag,
        cache_block_state_str[line->get_status(mask)],
        line->get_last_access_time());

    if (line->m_tag == tag) {
      if (line->get_status(mask) == RESERVED) {
        idx = index;
        return HIT_RESERVED;
      } else if (line->get_status(mask) == VALID) {
        idx = index;
        return HIT;
      } else if (line->get_status(mask) == MODIFIED) {
        if ((!is_write && line->is_readable(mask)) || is_write) {
          idx = index;
          return HIT;
        } else {
          idx = index;
          return SECTOR_MISS;
        }

      } else if (line->is_valid_line() && line->get_status(mask) == INVALID) {
        idx = index;
        return SECTOR_MISS;
      } else {
        assert(line->get_status(mask) == INVALID);
      }
    }
    if (!line->is_reserved_line()) {
      // percentage of dirty lines in the cache
      // number of dirty lines / total lines in the cache
      float dirty_line_percentage =
          ((float)m_dirty / (m_config.m_nset * m_config.m_assoc)) * 100;
      // If the cacheline is from a load op (not modified),
      // or the total dirty cacheline is above a specific value,
      // Then this cacheline is eligible to be considered for replacement
      // candidate i.e. Only evict clean cachelines until total dirty
      // cachelines reach the limit.
      // assert(m_config.m_wr_percent == 0);
      if (!line->is_modified_line() ||
          dirty_line_percentage >= m_config.m_wr_percent) {
        all_reserved = false;
        if (line->is_invalid_line()) {
          invalid_line = index;
        } else {
          // valid line : keep track of most appropriate replacement candidate
          if (m_config.m_replacement_policy == LRU) {
            if (line->get_last_access_time() < valid_timestamp) {
              valid_timestamp = line->get_last_access_time();
              valid_line = index;
            }
          } else if (m_config.m_replacement_policy == FIFO) {
            if (line->get_alloc_time() < valid_timestamp) {
              valid_timestamp = line->get_alloc_time();
              valid_line = index;
            }
          }
        }
      }
    }
  }

  logger->trace(
      "tag_array::probe({}) => all reserved={} invalid_line={} valid_line={} "
      "({} policy)",
      mem_fetch_ptr(mf), all_reserved, invalid_line, valid_line,
      replacement_policy_t_str[m_config.m_replacement_policy]);

  if (all_reserved) {
    assert(m_config.m_alloc_policy == ON_MISS);
    return RESERVATION_FAIL;  // miss and not enough space in cache to
                              // allocate on miss
  }

  if (invalid_line != (unsigned)-1) {
    idx = invalid_line;
  } else if (valid_line != (unsigned)-1) {
    idx = valid_line;
  } else
    abort();  // if an unreserved block exists, it is either invalid or
              // replaceable

  return MISS;
}

enum cache_request_status tag_array::access(new_addr_type addr, unsigned time,
                                            unsigned &idx, mem_fetch *mf) {
  bool wb = false;
  evicted_block_info evicted;
  enum cache_request_status result = access(addr, time, idx, wb, evicted, mf);
  assert(!wb);
  return result;
}

enum cache_request_status tag_array::access(new_addr_type addr, unsigned time,
                                            unsigned &idx, bool &wb,
                                            evicted_block_info &evicted,
                                            mem_fetch *mf) {
  logger->trace("tag_array::access({}, time={})", mem_fetch_ptr(mf), time);
  m_access++;
  m_is_used = true;
  // shader_cache_access_log(m_core_id, m_type_id, 0);  // log accesses to
  // cache
  enum cache_request_status status = probe(addr, idx, mf, mf->is_write());

  switch (status) {
    case HIT_RESERVED:
      m_pending_hit++;
    case HIT:
      m_lines[idx]->set_last_access_time(time, mf->get_access_sector_mask());
      break;
    case MISS:
      m_miss++;
      logger->trace(
          "tag_array::access({}, time={}) => {} cache index={} allocate "
          "policy={}",
          mem_fetch_ptr(mf), time, cache_request_status_str[status], idx,
          allocation_policy_t_str[m_config.m_alloc_policy]);

      // shader_cache_access_log(m_core_id, m_type_id, 1);  // log cache
      // misses
      if (m_config.m_alloc_policy == ON_MISS) {
        if (m_lines[idx]->is_modified_line()) {
          wb = true;
          // m_lines[idx]->set_byte_mask(mf);
          evicted.set_info(m_lines[idx]->m_block_addr,
                           m_lines[idx]->get_modified_size(),
                           m_lines[idx]->get_dirty_byte_mask(),
                           m_lines[idx]->get_dirty_sector_mask(),
                           mf->get_alloc_id(), mf->get_alloc_start_addr());
          // logger->trace("set evicted alloc start addr = {} from {}",
          //               mf->get_alloc_start_addr(), mem_fetch_ptr(mf));
          m_dirty--;
        }
        logger->trace("tag_array::allocate(cache={}, tag={}, time={})", idx,
                      m_config.tag(addr), time);
        m_lines[idx]->allocate(m_config.tag(addr), m_config.block_addr(addr),
                               time, mf->get_access_sector_mask());
      }
      break;
    case SECTOR_MISS:
      assert(m_config.m_cache_type == SECTOR);
      m_sector_miss++;
      // shader_cache_access_log(m_core_id, m_type_id, 1);  // log cache
      // misses
      if (m_config.m_alloc_policy == ON_MISS) {
        bool before = m_lines[idx]->is_modified_line();
        ((sector_cache_block *)m_lines[idx])
            ->allocate_sector(time, mf->get_access_sector_mask());
        if (before && !m_lines[idx]->is_modified_line()) {
          m_dirty--;
        }
      }
      break;
    case RESERVATION_FAIL:
      m_res_fail++;
      // shader_cache_access_log(m_core_id, m_type_id, 1);  // log cache
      // misses
      break;
    default:
      fprintf(stderr,
              "tag_array::access - Error: Unknown"
              "cache_request_status %d\n",
              status);
      abort();
  }
  return status;
}

void tag_array::fill(new_addr_type addr, unsigned time, mem_fetch *mf,
                     bool is_write) {
  fill(addr, time, mf->get_access_sector_mask(), mf->get_access_byte_mask(),
       is_write);
}

// fill on fill
void tag_array::fill(new_addr_type addr, unsigned time,
                     mem_access_sector_mask_t mask,
                     mem_access_byte_mask_t byte_mask, bool is_write) {
  // assert( m_config.m_alloc_policy == ON_FILL );
  unsigned idx;
  enum cache_request_status status = probe(addr, idx, mask, is_write);
  logger->trace(
      "tag_array::fill(cache={}, tag={}, addr={}) (on fill) status={}", idx,
      m_config.tag(addr), addr, cache_request_status_str[status]);

  if (status == RESERVATION_FAIL) return;

  bool before = m_lines[idx]->is_modified_line();
  // assert(status==MISS||status==SECTOR_MISS); // MSHR should have prevented
  // redundant memory request
  if (status == MISS) {
    logger->trace("tag_array::allocate(cache={}, tag={}, time={})", idx,
                  m_config.tag(addr), time);
    m_lines[idx]->allocate(m_config.tag(addr), m_config.block_addr(addr), time,
                           mask);
  } else if (status == SECTOR_MISS) {
    // assert(0 && "sector miss not supported");
    assert(m_config.m_cache_type == SECTOR);
    ((sector_cache_block *)m_lines[idx])->allocate_sector(time, mask);
  }
  if (before && !m_lines[idx]->is_modified_line()) {
    m_dirty--;
  }
  before = m_lines[idx]->is_modified_line();
  m_lines[idx]->fill(time, mask, byte_mask);
  if (m_lines[idx]->is_modified_line() && !before) {
    m_dirty++;
  }
}

// fill on miss
void tag_array::fill(unsigned index, unsigned time, mem_fetch *mf) {
  logger->trace("tag_array::fill(cache={}, tag={}, addr={}) (on miss)", index,
                m_config.tag(mf->get_addr()), mf->get_addr());
  assert(m_config.m_alloc_policy == ON_MISS);
  bool before = m_lines[index]->is_modified_line();
  m_lines[index]->fill(time, mf->get_access_sector_mask(),
                       mf->get_access_byte_mask());
  if (m_lines[index]->is_modified_line() && !before) {
    m_dirty++;
  }
}

// TODO: we need write back the flushed data to the upper level
void tag_array::flush() {
  logger->trace("tag_array::flush()");
  if (!m_is_used) return;

  for (unsigned i = 0; i < m_config.get_num_lines(); i++)
    if (m_lines[i]->is_modified_line()) {
      for (unsigned j = 0; j < SECTOR_CHUNCK_SIZE; j++) {
        m_lines[i]->set_status(INVALID, mem_access_sector_mask_t().set(j));
      }
    }

  m_dirty = 0;
  m_is_used = false;
}

void tag_array::invalidate() {
  logger->trace("tag_array::invalidate()");
  fmt::println("tag_array::invalidate()");
  if (!m_is_used) return;

  for (unsigned i = 0; i < m_config.get_num_lines(); i++)
    for (unsigned j = 0; j < SECTOR_CHUNCK_SIZE; j++)
      m_lines[i]->set_status(INVALID, mem_access_sector_mask_t().set(j));

  m_dirty = 0;
  m_is_used = false;
}

// float tag_array::windowed_miss_rate() const {
//   unsigned n_access = m_access - m_prev_snapshot_access;
//   unsigned n_miss = (m_miss + m_sector_miss) - m_prev_snapshot_miss;
//   // unsigned n_pending_hit = m_pending_hit - m_prev_snapshot_pending_hit;
//
//   float missrate = 0.0f;
//   if (n_access != 0) missrate = (float)(n_miss + m_sector_miss) / n_access;
//   return missrate;
// }

// void tag_array::new_window() {
//   m_prev_snapshot_access = m_access;
//   m_prev_snapshot_miss = m_miss;
//   m_prev_snapshot_miss = m_miss + m_sector_miss;
//   m_prev_snapshot_pending_hit = m_pending_hit;
// }

void tag_array::print(FILE *stream, unsigned &total_access,
                      unsigned &total_misses) const {
  m_config.print(stream);
  fprintf(stream,
          "\t\tAccess = %d, Miss = %d, Sector_Miss = %d, Total_Miss = %d "
          "(%.3g), PendingHit = %d (%.3g)\n",
          m_access, m_miss, m_sector_miss, (m_miss + m_sector_miss),
          (float)(m_miss + m_sector_miss) / m_access, m_pending_hit,
          (float)m_pending_hit / m_access);
  total_misses += (m_miss + m_sector_miss);
  total_access += m_access;
}

void tag_array::get_stats(unsigned &total_access, unsigned &total_misses,
                          unsigned &total_hit_res,
                          unsigned &total_res_fail) const {
  // Update statistics from the tag array
  total_access = m_access;
  total_misses = (m_miss + m_sector_miss);
  total_hit_res = m_pending_hit;
  total_res_fail = m_res_fail;
}
