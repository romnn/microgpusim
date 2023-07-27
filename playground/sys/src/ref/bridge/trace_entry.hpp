#pragma once

#include "../hal.hpp"
#include "../opcode_char.hpp"
#include "../trace_instr_opcode.hpp"

struct TraceEntry {
  unsigned block_x;
  unsigned block_y;
  unsigned block_z;
  unsigned line_num;
  unsigned pc;
  unsigned mask;
  unsigned reg_dsts_num;
  unsigned reg_dest[MAX_DST];
  const char *raw_opcode;
  TraceInstrOpcode opcode;
  op_type op;
  unsigned reg_srcs_num;
  unsigned reg_src[MAX_SRC];
};
