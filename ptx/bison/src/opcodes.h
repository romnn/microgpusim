#pragma once

enum opcode_t {
#define OP_DEF(OP, FUNC, STR, DST, CLASSIFICATION) OP,
#define OP_W_DEF(OP, FUNC, STR, DST, CLASSIFICATION) OP,
#include "./opcodes.def"
  NUM_OPCODES
#undef OP_DEF
#undef OP_W_DEF
};

static const char *g_opcode_str[NUM_OPCODES] = {
#define OP_DEF(OP, FUNC, STR, DST, CLASSIFICATION) STR,
#define OP_W_DEF(OP, FUNC, STR, DST, CLASSIFICATION) STR,
#include "./opcodes.def"
#undef OP_DEF
#undef OP_W_DEF
};

enum special_regs {
  CLOCK_REG,
  HALFCLOCK_ID,
  CLOCK64_REG,
  CTAID_REG,
  ENVREG_REG,
  GRIDID_REG,
  LANEID_REG,
  LANEMASK_EQ_REG,
  LANEMASK_LE_REG,
  LANEMASK_LT_REG,
  LANEMASK_GE_REG,
  LANEMASK_GT_REG,
  NCTAID_REG,
  NTID_REG,
  NSMID_REG,
  NWARPID_REG,
  PM_REG,
  SMID_REG,
  TID_REG,
  WARPID_REG,
  WARPSZ_REG
};

enum wmma_type {
  LOAD_A,
  LOAD_B,
  LOAD_C,
  STORE_D,
  MMA,
  ROW,
  COL,
  M16N16K16,
  M32N8K16,
  M8N32K16
};
