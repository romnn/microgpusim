#pragma once

#include "baseline_cache.hpp"
#include "tag_array.hpp"
#include "cache_event.hpp"
#include "cache_request_status.hpp"

/// Read only cache
class read_only_cache : public baseline_cache {
public:
  read_only_cache(const char *name, cache_config &config, int core_id,
                  int type_id, mem_fetch_interface *memport,
                  enum mem_fetch_status status)
      : baseline_cache(name, config, core_id, type_id, memport, status) {}

  /// Access cache for read_only_cache: returns RESERVATION_FAIL if request
  /// could not be accepted (for any reason)
  virtual enum cache_request_status access(new_addr_type addr, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events);

  virtual ~read_only_cache() {}

protected:
  read_only_cache(const char *name, cache_config &config, int core_id,
                  int type_id, mem_fetch_interface *memport,
                  enum mem_fetch_status status, tag_array *new_tag_array)
      : baseline_cache(name, config, core_id, type_id, memport, status,
                       new_tag_array) {}
};
