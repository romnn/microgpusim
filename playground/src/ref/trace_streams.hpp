#pragma once

// const char* trace_streams_str[] = {

enum trace_streams_type {
  WARP_SCHEDULER,
  SCOREBOARD,
  MEMORY_PARTITION_UNIT,
  MEMORY_SUBPARTITION_UNIT,
  INTERCONNECT,
  LIVENESS,
  NUM_TRACE_STREAMS
};
