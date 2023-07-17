#pragma once

enum mem_stage_stall_type {
  NO_RC_FAIL = 0,
  BK_CONF,
  MSHR_RC_FAIL,
  ICNT_RC_FAIL,
  COAL_STALL,
  TLB_STALL,
  DATA_PORT_STALL,
  WB_ICNT_RC_FAIL,
  WB_CACHE_RSRV_FAIL,
  N_MEM_STAGE_STALL_TYPE
};

static const char *mem_stage_stall_type_str[] = {
    "NO_RC_FAIL",         "BK_CONF",
    "MSHR_RC_FAIL",       "ICNT_RC_FAIL",
    "COAL_STALL",         "TLB_STALL",
    "DATA_PORT_STALL",    "WB_ICNT_RC_FAIL",
    "WB_CACHE_RSRV_FAIL", "N_MEM_STAGE_STALL_TYPE",
};
