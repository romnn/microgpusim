#include "cache_reservation_fail_reason.hpp"

#include <assert.h>

const char *cache_fail_status_str(enum cache_reservation_fail_reason status) {
  assert(sizeof(static_cache_reservation_fail_reason_str) /
             sizeof(const char *) ==
         NUM_CACHE_RESERVATION_FAIL_STATUS);
  assert(status < NUM_CACHE_RESERVATION_FAIL_STATUS);

  return static_cache_reservation_fail_reason_str[status];
}
