#pragma once

#include <set>

class ptx_instruction;

extern const char *g_opcode_string[];

struct basic_block_t {
  basic_block_t(unsigned ID, ptx_instruction *begin, ptx_instruction *end,
                bool entry, bool ex) {
    bb_id = ID;
    ptx_begin = begin;
    ptx_end = end;
    is_entry = entry;
    is_exit = ex;
    immediatepostdominator_id = -1;
    immediatedominator_id = -1;
  }

  ptx_instruction *ptx_begin;
  ptx_instruction *ptx_end;
  // indices of other basic blocks in m_basic_blocks array
  std::set<int> predecessor_ids;
  std::set<int> successor_ids;
  std::set<int> postdominator_ids;
  std::set<int> dominator_ids;
  std::set<int> Tmp_ids;
  int immediatepostdominator_id;
  int immediatedominator_id;
  bool is_entry;
  bool is_exit;
  unsigned bb_id;

  // if this basic block dom B
  bool dom(const basic_block_t *B) {
    return (B->dominator_ids.find(this->bb_id) != B->dominator_ids.end());
  }

  // if this basic block pdom B
  bool pdom(const basic_block_t *B) {
    return (B->postdominator_ids.find(this->bb_id) !=
            B->postdominator_ids.end());
  }
};
