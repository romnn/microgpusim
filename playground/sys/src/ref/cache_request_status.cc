#include "cache_request_status.hpp"

#include <assert.h>

const char *get_cache_request_status_str(enum cache_request_status status) {
  assert(sizeof(cache_request_status_str) / sizeof(const char *) ==
         NUM_CACHE_REQUEST_STATUS);
  assert(status < NUM_CACHE_REQUEST_STATUS);

  return cache_request_status_str[status];
}
