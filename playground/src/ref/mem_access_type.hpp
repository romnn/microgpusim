#pragma once

#include <assert.h>

static const char *access_type_str[] = {
    "GLOBAL_ACC_R", "LOCAL_ACC_R",   "CONST_ACC_R",   "TEXTURE_ACC_R",
    "GLOBAL_ACC_W", "LOCAL_ACC_W",   "L1_WRBK_ACC",   "L2_WRBK_ACC",
    "INST_ACC_R",   "L1_WR_ALLOC_R", "L2_WR_ALLOC_R", "NUM_MEM_ACCESS_TYPE",
};

enum mem_access_type {
  GLOBAL_ACC_R,
  LOCAL_ACC_R,
  CONST_ACC_R,
  TEXTURE_ACC_R,
  GLOBAL_ACC_W,
  LOCAL_ACC_W,
  L1_WRBK_ACC,
  L2_WRBK_ACC,
  INST_ACC_R,
  L1_WR_ALLOC_R,
  L2_WR_ALLOC_R,
  NUM_MEM_ACCESS_TYPE,
};

const char *mem_access_type_str(enum mem_access_type access_type);
