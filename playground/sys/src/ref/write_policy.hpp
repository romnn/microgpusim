#pragma once

static const char* write_policy_t_str[] = {
    "READ_ONLY",   "WRITE_BACK",         "WRITE_THROUGH",
    "WRITE_EVICT", "LOCAL_WB_GLOBAL_WT",
};

enum write_policy_t {
  READ_ONLY,
  WRITE_BACK,
  WRITE_THROUGH,
  WRITE_EVICT,
  LOCAL_WB_GLOBAL_WT
};
