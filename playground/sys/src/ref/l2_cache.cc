#include "l2_cache.hpp"

// The l2 cache access function calls the base data_cache access
// implementation.  When the L2 needs to diverge from L1, L2 specific
// changes should be made here.
enum cache_request_status l2_cache::access(new_addr_type addr, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events) {
  // return data_cache::access(addr, mf, time, events);
  cache_request_status status = data_cache::access(addr, mf, time, events);
  printf("L2 cache access(%lu) time=%d status=%s\n", addr, time,
         cache_request_status_str[status]);
  return status;
}
