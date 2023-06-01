#include "function_info.hpp"

std::list<ptx_instruction *>::iterator
function_info::find_next_real_instruction(
    std::list<ptx_instruction *>::iterator i) {
  while ((i != m_instructions.end()) && (*i)->is_label())
    i++;
  return i;
}

void function_info::create_basic_blocks() {
  std::list<ptx_instruction *> leaders;
  std::list<ptx_instruction *>::iterator i, l;

  // first instruction is a leader
  i = m_instructions.begin();
  leaders.push_back(*i);
  i++;
  while (i != m_instructions.end()) {
    ptx_instruction *pI = *i;
    if (pI->is_label()) {
      leaders.push_back(pI);
      i = find_next_real_instruction(++i);
    } else {
      switch (pI->get_opcode()) {
      case BRA_OP:
      case RET_OP:
      case EXIT_OP:
      case RETP_OP:
      case BREAK_OP:
        i++;
        if (i != m_instructions.end())
          leaders.push_back(*i);
        i = find_next_real_instruction(i);
        break;
      case CALL_OP:
      case CALLP_OP:
        if (pI->has_pred()) {
          printf("GPGPU-Sim PTX: Warning found predicated call\n");
          i++;
          if (i != m_instructions.end())
            leaders.push_back(*i);
          i = find_next_real_instruction(i);
        } else
          i++;
        break;
      default:
        i++;
      }
    }
  }

  if (leaders.empty()) {
    printf("GPGPU-Sim PTX: Function \'%s\' has no basic blocks\n",
           m_name.c_str());
    return;
  }

  unsigned bb_id = 0;
  l = leaders.begin();
  i = m_instructions.begin();
  m_basic_blocks.push_back(
      new basic_block_t(bb_id++, *find_next_real_instruction(i), NULL, 1, 0));
  ptx_instruction *last_real_inst = *(l++);

  for (; i != m_instructions.end(); i++) {
    ptx_instruction *pI = *i;
    if (l != leaders.end() && *i == *l) {
      // found start of next basic block
      m_basic_blocks.back()->ptx_end = last_real_inst;
      if (find_next_real_instruction(i) !=
          m_instructions.end()) { // if not bogus trailing label
        m_basic_blocks.push_back(new basic_block_t(
            bb_id++, *find_next_real_instruction(i), NULL, 0, 0));
        last_real_inst = *find_next_real_instruction(i);
      }
      // start search for next leader
      l++;
    }
    pI->assign_bb(m_basic_blocks.back());
    if (!pI->is_label())
      last_real_inst = pI;
  }
  m_basic_blocks.back()->ptx_end = last_real_inst;
  m_basic_blocks.push_back(
      /*exit basic block*/ new basic_block_t(bb_id, NULL, NULL, 0, 1));
}

void function_info::print_basic_blocks() {
  printf("Printing basic blocks for function \'%s\':\n", m_name.c_str());
  std::list<ptx_instruction *>::iterator ptx_itr;
  unsigned last_bb = 0;
  for (ptx_itr = m_instructions.begin(); ptx_itr != m_instructions.end();
       ptx_itr++) {
    if ((*ptx_itr)->get_bb()) {
      if ((*ptx_itr)->get_bb()->bb_id != last_bb) {
        printf("\n");
        last_bb = (*ptx_itr)->get_bb()->bb_id;
      }
      printf("bb_%02u\t: ", (*ptx_itr)->get_bb()->bb_id);
      (*ptx_itr)->print_insn();
      printf("\n");
    }
  }
  printf("\nSummary of basic blocks for \'%s\':\n", m_name.c_str());
  std::vector<basic_block_t *>::iterator bb_itr;
  for (bb_itr = m_basic_blocks.begin(); bb_itr != m_basic_blocks.end();
       bb_itr++) {
    printf("bb_%02u\t:", (*bb_itr)->bb_id);
    if ((*bb_itr)->ptx_begin)
      printf(" first: %s\t", ((*bb_itr)->ptx_begin)->get_opcode_cstr());
    else
      printf(" first: NULL\t");
    if ((*bb_itr)->ptx_end) {
      printf(" last: %s\t", ((*bb_itr)->ptx_end)->get_opcode_cstr());
    } else
      printf(" last: NULL\t");
    printf("\n");
  }
  printf("\n");
}

void function_info::print_basic_block_links() {
  printf("Printing basic blocks links for function \'%s\':\n", m_name.c_str());
  std::vector<basic_block_t *>::iterator bb_itr;
  for (bb_itr = m_basic_blocks.begin(); bb_itr != m_basic_blocks.end();
       bb_itr++) {
    printf("ID: %d\t:", (*bb_itr)->bb_id);
    if (!(*bb_itr)->predecessor_ids.empty()) {
      printf("Predecessors:");
      std::set<int>::iterator p;
      for (p = (*bb_itr)->predecessor_ids.begin();
           p != (*bb_itr)->predecessor_ids.end(); p++) {
        printf(" %d", *p);
      }
      printf("\t");
    }
    if (!(*bb_itr)->successor_ids.empty()) {
      printf("Successors:");
      std::set<int>::iterator s;
      for (s = (*bb_itr)->successor_ids.begin();
           s != (*bb_itr)->successor_ids.end(); s++) {
        printf(" %d", *s);
      }
    }
    printf("\n");
  }
}
operand_info *function_info::find_break_target(
    ptx_instruction *p_break_insn) // find the target of a break instruction
{
  const basic_block_t *break_bb = p_break_insn->get_bb();
  // go through the dominator tree
  for (const basic_block_t *p_bb = break_bb; p_bb->immediatedominator_id != -1;
       p_bb = m_basic_blocks[p_bb->immediatedominator_id]) {
    // reverse search through instructions in basic block for breakaddr
    // instruction
    unsigned insn_addr = p_bb->ptx_end->get_m_instr_mem_index();
    while (insn_addr >= p_bb->ptx_begin->get_m_instr_mem_index()) {
      ptx_instruction *pI = m_instr_mem[insn_addr];
      insn_addr -= 1;
      if (pI == NULL)
        continue; // temporary solution for variable size instructions
      if (pI->get_opcode() == BREAKADDR_OP) {
        return &(pI->dst());
      }
    }
  }

  assert(0);

  // lazy fallback: just traverse backwards?
  for (int insn_addr = p_break_insn->get_m_instr_mem_index(); insn_addr >= 0;
       insn_addr--) {
    ptx_instruction *pI = m_instr_mem[insn_addr];
    if (pI->get_opcode() == BREAKADDR_OP) {
      return &(pI->dst());
    }
  }

  return NULL;
}
void function_info::connect_basic_blocks() // iterate across m_basic_blocks of
                                           // function, connecting basic blocks
                                           // together
{
  std::vector<basic_block_t *>::iterator bb_itr;
  std::vector<basic_block_t *>::iterator bb_target_itr;
  basic_block_t *exit_bb = m_basic_blocks.back();

  // start from first basic block, which we know is the entry point
  bb_itr = m_basic_blocks.begin();
  for (bb_itr = m_basic_blocks.begin(); bb_itr != m_basic_blocks.end();
       bb_itr++) {
    ptx_instruction *pI = (*bb_itr)->ptx_end;
    if ((*bb_itr)->is_exit) // reached last basic block, no successors to link
      continue;
    if (pI->get_opcode() == RETP_OP || pI->get_opcode() == RET_OP ||
        pI->get_opcode() == EXIT_OP) {
      (*bb_itr)->successor_ids.insert(exit_bb->bb_id);
      exit_bb->predecessor_ids.insert((*bb_itr)->bb_id);
      if (pI->has_pred()) {
        printf("GPGPU-Sim PTX: Warning detected predicated return/exit.\n");
        // if predicated, add link to next block
        unsigned next_addr = pI->get_m_instr_mem_index() + pI->inst_size();
        if (next_addr < m_instr_mem_size && m_instr_mem[next_addr]) {
          basic_block_t *next_bb = m_instr_mem[next_addr]->get_bb();
          (*bb_itr)->successor_ids.insert(next_bb->bb_id);
          next_bb->predecessor_ids.insert((*bb_itr)->bb_id);
        }
      }
      continue;
    } else if (pI->get_opcode() == BRA_OP) {
      // find successor and link that basic_block to this one
      operand_info &target = pI->dst(); // get operand, e.g. target name
      unsigned addr = labels[target.name()];
      ptx_instruction *target_pI = m_instr_mem[addr];
      basic_block_t *target_bb = target_pI->get_bb();
      (*bb_itr)->successor_ids.insert(target_bb->bb_id);
      target_bb->predecessor_ids.insert((*bb_itr)->bb_id);
    }

    if (!(pI->get_opcode() == BRA_OP && (!pI->has_pred()))) {
      // if basic block does not end in an unpredicated branch,
      // then next basic block is also successor
      // (this is better than testing for .uni)
      unsigned next_addr = pI->get_m_instr_mem_index() + pI->inst_size();
      basic_block_t *next_bb = m_instr_mem[next_addr]->get_bb();
      (*bb_itr)->successor_ids.insert(next_bb->bb_id);
      next_bb->predecessor_ids.insert((*bb_itr)->bb_id);
    } else
      assert(pI->get_opcode() == BRA_OP);
  }
}
bool function_info::connect_break_targets() // connecting break instructions
                                            // with proper targets
{
  std::vector<basic_block_t *>::iterator bb_itr;
  std::vector<basic_block_t *>::iterator bb_target_itr;
  bool modified = false;

  // start from first basic block, which we know is the entry point
  bb_itr = m_basic_blocks.begin();
  for (bb_itr = m_basic_blocks.begin(); bb_itr != m_basic_blocks.end();
       bb_itr++) {
    basic_block_t *p_bb = *bb_itr;
    ptx_instruction *pI = p_bb->ptx_end;
    if (p_bb->is_exit) // reached last basic block, no successors to link
      continue;
    if (pI->get_opcode() == BREAK_OP) {
      // backup existing successor_ids for stability check
      std::set<int> orig_successor_ids = p_bb->successor_ids;

      // erase the previous linkage with old successors
      for (std::set<int>::iterator succ_ids = p_bb->successor_ids.begin();
           succ_ids != p_bb->successor_ids.end(); ++succ_ids) {
        basic_block_t *successor_bb = m_basic_blocks[*succ_ids];
        successor_bb->predecessor_ids.erase(p_bb->bb_id);
      }
      p_bb->successor_ids.clear();

      // find successor and link that basic_block to this one
      // successor of a break is set by an preceeding breakaddr instruction
      operand_info *target = find_break_target(pI);
      unsigned addr = labels[target->name()];
      ptx_instruction *target_pI = m_instr_mem[addr];
      basic_block_t *target_bb = target_pI->get_bb();
      p_bb->successor_ids.insert(target_bb->bb_id);
      target_bb->predecessor_ids.insert(p_bb->bb_id);

      if (pI->has_pred()) {
        // predicated break - add link to next basic block
        unsigned next_addr = pI->get_m_instr_mem_index() + pI->inst_size();
        basic_block_t *next_bb = m_instr_mem[next_addr]->get_bb();
        p_bb->successor_ids.insert(next_bb->bb_id);
        next_bb->predecessor_ids.insert(p_bb->bb_id);
      }

      modified = modified || (orig_successor_ids != p_bb->successor_ids);
    }
  }

  return modified;
}
void function_info::do_pdom() {
  create_basic_blocks();
  connect_basic_blocks();
  bool modified = false;
  do {
    find_dominators();
    find_idominators();
    modified = connect_break_targets();
  } while (modified == true);

  if (g_debug_execution >= 50) {
    print_basic_blocks();
    print_basic_block_links();
    print_basic_block_dot();
  }
  if (g_debug_execution >= 2) {
    print_dominators();
  }
  find_postdominators();
  find_ipostdominators();
  if (g_debug_execution >= 50) {
    print_postdominators();
    print_ipostdominators();
  }
  printf("GPGPU-Sim PTX: pre-decoding instructions for \'%s\'...\n",
         m_name.c_str());
  for (unsigned ii = 0; ii < m_n;
       ii += m_instr_mem[ii]->inst_size()) { // handle branch instructions
    ptx_instruction *pI = m_instr_mem[ii];
    pI->pre_decode();
  }
  printf("GPGPU-Sim PTX: ... done pre-decoding instructions for \'%s\'.\n",
         m_name.c_str());
  fflush(stdout);
  m_assembled = true;
}
