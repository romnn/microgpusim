#include "mem_access.hpp"

const char *mem_access_type_str(enum mem_access_type access_type) {
  assert(access_type < NUM_MEM_ACCESS_TYPE);
  return g_mem_access_type_str[access_type];
}
