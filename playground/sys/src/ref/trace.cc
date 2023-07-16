#include "trace.hpp"

#include <cstring>

namespace Trace {

bool enabled = false;
int sampling_core = 0;
int sampling_memory_partition = -1;
bool trace_streams_enabled[NUM_TRACE_STREAMS] = {false};
const char *config_str;

void init() {
  for (unsigned i = 0; i < NUM_TRACE_STREAMS; ++i) {
    if (strstr(config_str, trace_streams_type_str[i]) != NULL) {
      trace_streams_enabled[i] = true;
    }
  }
}
}  // namespace Trace
