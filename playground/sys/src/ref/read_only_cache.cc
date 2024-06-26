#include "read_only_cache.hpp"

#include <memory>

#include "cache_reservation_fail_reason.hpp"
#include "trace_gpgpu_sim.hpp"

/// Access cache for read_only_cache: returns RESERVATION_FAIL if
// request could not be accepted (for any reason)
enum cache_request_status read_only_cache::access(
    new_addr_type addr, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events) {
  assert(mf->get_data_size() <= m_config.get_atom_sz());
  assert(m_config.m_write_policy == READ_ONLY);
  assert(!mf->get_is_write());
  new_addr_type block_addr = m_config.block_addr(addr);

  logger->debug(
      "{}::read_only_cache::access({}, write = {}, data size = {}, control "
      "size = {}, block = {}, time = {})",
      name(), addr, mf->is_write(), mf->get_data_size(), mf->get_ctrl_size(),
      block_addr, time);

  unsigned cache_index = (unsigned)-1;
  enum cache_request_status status =
      m_tag_array->probe(block_addr, cache_index, mf, mf->is_write());

  enum cache_request_status cache_status = RESERVATION_FAIL;

  if (status == HIT) {
    cache_status = m_tag_array->access(block_addr, time, cache_index,
                                       mf);  // update LRU state
  } else if (status != RESERVATION_FAIL) {
    if (!miss_queue_full(0)) {
      bool do_miss = false;
      send_read_request(addr, block_addr, cache_index, mf, time, do_miss,
                        events, true, false);
      if (do_miss)
        cache_status = MISS;
      else
        cache_status = RESERVATION_FAIL;
    } else {
      cache_status = RESERVATION_FAIL;
      m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
    }
  } else {
    m_stats.inc_fail_stats(mf->get_access_type(), LINE_ALLOC_FAIL);
  }

  if (accelsim_compat_mode) {
    // printf(
    //     "inst cache access(%lu) time=%d cache index=%d block addr=%lu probe "
    //     "status=%s access status = %s\n",
    //     addr, time, cache_index, block_addr,
    //     cache_request_status_str[status],
    //     cache_request_status_str[cache_status]);
  }

  m_stats.inc_stats(mf->get_access_type(),
                    m_stats.select_stats_status(status, cache_status));
  m_stats.inc_stats_pw(mf->get_access_type(),
                       m_stats.select_stats_status(status, cache_status));
  return cache_status;
}

#include "spdlog/sinks/stdout_color_sinks.h"

std::unique_ptr<read_only_cache> new_read_only_cache(
    const std::string &name, std::unique_ptr<cache_config> config, int core_id,
    int type_id, mem_fetch_interface *memport,
    enum mem_fetch_status fetch_status) {
  auto logger = spdlog::stdout_color_mt("read_only_cache");
  return std::make_unique<read_only_cache>(name.c_str(), *config, core_id,
                                           type_id, memport, fetch_status,
                                           false, logger);
}
