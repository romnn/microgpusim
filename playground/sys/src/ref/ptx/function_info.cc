#include "function_info.hpp"

#include "gpgpu_ptx_sim_arg.hpp"

function_info::function_info(int entry_point, gpgpu_context *ctx) {
  gpgpu_ctx = ctx;
  m_uid = (gpgpu_ctx->function_info_sm_next_uid)++;
  m_entry_point = (entry_point == 1) ? true : false;
  m_extern = (entry_point == 2) ? true : false;
  num_reconvergence_pairs = 0;
  m_symtab = NULL;
  m_assembled = false;
  m_return_var_sym = NULL;
  m_kernel_info.cmem = 0;
  m_kernel_info.lmem = 0;
  m_kernel_info.regs = 0;
  m_kernel_info.smem = 0;
  m_local_mem_framesize = 0;
  m_args_aligned_size = -1;
  pdom_done = false;  // initialize it to false
}

std::list<ptx_instruction *>::iterator
function_info::find_next_real_instruction(
    std::list<ptx_instruction *>::iterator i) {
  while ((i != m_instructions.end()) && (*i)->is_label()) i++;
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
          if (i != m_instructions.end()) leaders.push_back(*i);
          i = find_next_real_instruction(i);
          break;
        case CALL_OP:
        case CALLP_OP:
          if (pI->has_pred()) {
            printf("GPGPU-Sim PTX: Warning found predicated call\n");
            i++;
            if (i != m_instructions.end()) leaders.push_back(*i);
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
          m_instructions.end()) {  // if not bogus trailing label
        m_basic_blocks.push_back(new basic_block_t(
            bb_id++, *find_next_real_instruction(i), NULL, 0, 0));
        last_real_inst = *find_next_real_instruction(i);
      }
      // start search for next leader
      l++;
    }
    pI->assign_bb(m_basic_blocks.back());
    if (!pI->is_label()) last_real_inst = pI;
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

void function_info::print_dominators() {
  printf("Printing dominators for function \'%s\':\n", m_name.c_str());
  std::vector<int>::iterator bb_itr;
  for (unsigned i = 0; i < m_basic_blocks.size(); i++) {
    printf("ID: %d\t:", i);
    for (std::set<int>::iterator j = m_basic_blocks[i]->dominator_ids.begin();
         j != m_basic_blocks[i]->dominator_ids.end(); j++)
      printf(" %d", *j);
    printf("\n");
  }
}

void function_info::print_postdominators() {
  printf("Printing postdominators for function \'%s\':\n", m_name.c_str());
  std::vector<int>::iterator bb_itr;
  for (unsigned i = 0; i < m_basic_blocks.size(); i++) {
    printf("ID: %d\t:", i);
    for (std::set<int>::iterator j =
             m_basic_blocks[i]->postdominator_ids.begin();
         j != m_basic_blocks[i]->postdominator_ids.end(); j++)
      printf(" %d", *j);
    printf("\n");
  }
}

void function_info::print_ipostdominators() {
  printf("Printing immediate postdominators for function \'%s\':\n",
         m_name.c_str());
  std::vector<int>::iterator bb_itr;
  for (unsigned i = 0; i < m_basic_blocks.size(); i++) {
    printf("ID: %d\t:", i);
    printf("%d\n", m_basic_blocks[i]->immediatepostdominator_id);
  }
}

void function_info::print_idominators() {
  printf("Printing immediate dominators for function \'%s\':\n",
         m_name.c_str());
  std::vector<int>::iterator bb_itr;
  for (unsigned i = 0; i < m_basic_blocks.size(); i++) {
    printf("ID: %d\t:", i);
    printf("%d\n", m_basic_blocks[i]->immediatedominator_id);
  }
}

unsigned function_info::get_num_reconvergence_pairs() {
  if (!num_reconvergence_pairs) {
    if (m_basic_blocks.size() == 0) return 0;
    for (unsigned i = 0; i < (m_basic_blocks.size() - 1);
         i++) {  // last basic block containing exit obviously won't have a pair
      if (m_basic_blocks[i]->ptx_end->get_opcode() == BRA_OP) {
        num_reconvergence_pairs++;
      }
    }
  }
  return num_reconvergence_pairs;
}

void function_info::get_reconvergence_pairs(gpgpu_recon_t *recon_points) {
  unsigned idx = 0;  // array index
  if (m_basic_blocks.size() == 0) return;
  for (unsigned i = 0; i < (m_basic_blocks.size() - 1);
       i++) {  // last basic block containing exit obviously won't have a pair
#ifdef DEBUG_GET_RECONVERG_PAIRS
    printf("i=%d\n", i);
    fflush(stdout);
#endif
    if (m_basic_blocks[i]->ptx_end->get_opcode() == BRA_OP) {
#ifdef DEBUG_GET_RECONVERG_PAIRS
      printf("\tbranch!\n");
      printf("\tbb_id=%d; ipdom=%d\n", m_basic_blocks[i]->bb_id,
             m_basic_blocks[i]->immediatepostdominator_id);
      printf("\tm_instr_mem index=%d\n",
             m_basic_blocks[i]->ptx_end->get_m_instr_mem_index());
      fflush(stdout);
#endif
      recon_points[idx].source_pc = m_basic_blocks[i]->ptx_end->get_PC();
      recon_points[idx].source_inst = m_basic_blocks[i]->ptx_end;
#ifdef DEBUG_GET_RECONVERG_PAIRS
      printf("\trecon_points[idx].source_pc=%d\n", recon_points[idx].source_pc);
#endif
      if (m_basic_blocks[m_basic_blocks[i]->immediatepostdominator_id]
              ->ptx_begin) {
        recon_points[idx].target_pc =
            m_basic_blocks[m_basic_blocks[i]->immediatepostdominator_id]
                ->ptx_begin->get_PC();
        recon_points[idx].target_inst =
            m_basic_blocks[m_basic_blocks[i]->immediatepostdominator_id]
                ->ptx_begin;
      } else {
        // reconverge after function return
        recon_points[idx].target_pc = -2;
        recon_points[idx].target_inst = NULL;
      }
#ifdef DEBUG_GET_RECONVERG_PAIRS
      m_basic_blocks[m_basic_blocks[i]->immediatepostdominator_id]
          ->ptx_begin->print_insn();
      printf("\trecon_points[idx].target_pc=%d\n", recon_points[idx].target_pc);
      fflush(stdout);
#endif
      idx++;
    }
  }
}

// interface with graphviz (print the graph in DOT language) for plotting
void function_info::print_basic_block_dot() {
  printf("Basic Block in DOT\n");
  printf("digraph %s {\n", m_name.c_str());
  std::vector<basic_block_t *>::iterator bb_itr;
  for (bb_itr = m_basic_blocks.begin(); bb_itr != m_basic_blocks.end();
       bb_itr++) {
    printf("\t");
    std::set<int>::iterator s;
    for (s = (*bb_itr)->successor_ids.begin();
         s != (*bb_itr)->successor_ids.end(); s++) {
      unsigned succ_bb = *s;
      printf("%d -> %d; ", (*bb_itr)->bb_id, succ_bb);
    }
    printf("\n");
  }
  printf("}\n");
}

operand_info *function_info::find_break_target(
    ptx_instruction *p_break_insn)  // find the target of a break instruction
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
        continue;  // temporary solution for variable size instructions
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

void function_info::connect_basic_blocks()  // iterate across m_basic_blocks of
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
    if ((*bb_itr)->is_exit)  // reached last basic block, no successors to link
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
      operand_info &target = pI->dst();  // get operand, e.g. target name
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
bool function_info::connect_break_targets()  // connecting break instructions
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
    if (p_bb->is_exit)  // reached last basic block, no successors to link
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
       ii += m_instr_mem[ii]->inst_size()) {  // handle branch instructions
    ptx_instruction *pI = m_instr_mem[ii];
    pI->pre_decode();
  }
  printf("GPGPU-Sim PTX: ... done pre-decoding instructions for \'%s\'.\n",
         m_name.c_str());
  fflush(stdout);
  m_assembled = true;
}

#define MAX_INST_SIZE 8 /*bytes*/

void function_info::ptx_assemble() {
  if (m_assembled) {
    return;
  }

  // get the instructions into instruction memory...
  unsigned num_inst = m_instructions.size();
  m_instr_mem_size = MAX_INST_SIZE * (num_inst + 1);
  m_instr_mem = new ptx_instruction *[m_instr_mem_size];

  printf("GPGPU-Sim PTX: instruction assembly for function \'%s\'... ",
         m_name.c_str());
  fflush(stdout);
  std::list<ptx_instruction *>::iterator i;

  addr_t PC =
      gpgpu_ctx->func_sim->g_assemble_code_next_pc;  // globally unique address
                                                     // (across functions)
  // start function on an aligned address
  for (unsigned i = 0; i < (PC % MAX_INST_SIZE); i++)
    gpgpu_ctx->s_g_pc_to_insn.push_back((ptx_instruction *)NULL);
  PC += PC % MAX_INST_SIZE;
  m_start_PC = PC;

  addr_t n = 0;  // offset in m_instr_mem
  // Why s_g_pc_to_insn.size() is needed to reserve additional memory for insts?
  // reserve is cumulative. s_g_pc_to_insn.reserve(s_g_pc_to_insn.size() +
  // MAX_INST_SIZE*m_instructions.size());
  gpgpu_ctx->s_g_pc_to_insn.reserve(MAX_INST_SIZE * m_instructions.size());
  for (i = m_instructions.begin(); i != m_instructions.end(); i++) {
    ptx_instruction *pI = *i;
    if (pI->is_label()) {
      const symbol *l = pI->get_label();
      labels[l->name()] = n;
    } else {
      gpgpu_ctx->func_sim->g_pc_to_finfo[PC] = this;
      m_instr_mem[n] = pI;
      gpgpu_ctx->s_g_pc_to_insn.push_back(pI);
      assert(pI == gpgpu_ctx->s_g_pc_to_insn[PC]);
      pI->set_m_instr_mem_index(n);
      pI->set_PC(PC);
      assert(pI->inst_size() <= MAX_INST_SIZE);
      for (unsigned i = 1; i < pI->inst_size(); i++) {
        gpgpu_ctx->s_g_pc_to_insn.push_back((ptx_instruction *)NULL);
        m_instr_mem[n + i] = NULL;
      }
      n += pI->inst_size();
      PC += pI->inst_size();
    }
  }
  gpgpu_ctx->func_sim->g_assemble_code_next_pc = PC;
  for (unsigned ii = 0; ii < n;
       ii += m_instr_mem[ii]->inst_size()) {  // handle branch instructions
    ptx_instruction *pI = m_instr_mem[ii];
    if (pI->get_opcode() == BRA_OP || pI->get_opcode() == BREAKADDR_OP ||
        pI->get_opcode() == CALLP_OP) {
      operand_info &target = pI->dst();  // get operand, e.g. target name
      if (labels.find(target.name()) == labels.end()) {
        printf(
            "GPGPU-Sim PTX: Loader error (%s:%u): Branch label \"%s\" does not "
            "appear in assembly code.",
            pI->source_file(), pI->source_line(), target.name().c_str());
        abort();
      }
      unsigned index = labels[target.name()];  // determine address from name
      unsigned PC = m_instr_mem[index]->get_PC();
      m_symtab->set_label_address(target.get_symbol(), PC);
      target.set_type(label_t);
    }
  }
  m_n = n;
  printf("  done.\n");
  fflush(stdout);
}

void function_info::add_param_name_type_size(unsigned index, std::string name,
                                             int type, size_t size, bool ptr,
                                             memory_space_t space) {
  unsigned parsed_index;
  char buffer[2048];
  snprintf(buffer, 2048, "%s_param_%%u", m_name.c_str());
  int ntokens = sscanf(name.c_str(), buffer, &parsed_index);
  if (ntokens == 1) {
    assert(m_ptx_kernel_param_info.find(parsed_index) ==
           m_ptx_kernel_param_info.end());
    m_ptx_kernel_param_info[parsed_index] =
        param_info(name, type, size, ptr, space);
  } else {
    assert(m_ptx_kernel_param_info.find(index) ==
           m_ptx_kernel_param_info.end());
    m_ptx_kernel_param_info[index] = param_info(name, type, size, ptr, space);
  }
}

void function_info::add_param_data(unsigned argn,
                                   struct gpgpu_ptx_sim_arg *args) {
  const void *data = args->m_start;

  bool scratchpad_memory_param =
      false;  // Is this parameter in CUDA shared memory or OpenCL local memory

  std::map<unsigned, param_info>::iterator i =
      m_ptx_kernel_param_info.find(argn);
  if (i != m_ptx_kernel_param_info.end()) {
    if (i->second.is_ptr_shared()) {
      assert(
          args->m_start == NULL &&
          "OpenCL parameter pointer to local memory must have NULL as value");
      scratchpad_memory_param = true;
    } else {
      param_t tmp;
      tmp.pdata = args->m_start;
      tmp.size = args->m_nbytes;
      tmp.offset = args->m_offset;
      tmp.type = 0;
      i->second.add_data(tmp);
      i->second.add_offset((unsigned)args->m_offset);
    }
  } else {
    scratchpad_memory_param = true;
  }

  if (scratchpad_memory_param) {
    // This should only happen for OpenCL:
    //
    // The LLVM PTX compiler in NVIDIA's driver (version 190.29)
    // does not generate an argument in the function declaration
    // for __constant arguments.
    //
    // The associated constant memory space can be allocated in two
    // ways. It can be explicitly initialized in the .ptx file where
    // it is declared.  Or, it can be allocated using the clCreateBuffer
    // on the host. In this later case, the .ptx file will contain
    // a global declaration of the parameter, but it will have an unknown
    // array size.  Thus, the symbol's address will not be set and we need
    // to set it here before executing the PTX.

    char buffer[2048];
    snprintf(buffer, 2048, "%s_param_%u", m_name.c_str(), argn);

    symbol *p = m_symtab->lookup(buffer);
    if (p == NULL) {
      printf(
          "GPGPU-Sim PTX: ERROR ** could not locate symbol for \'%s\' : cannot "
          "bind buffer\n",
          buffer);
      abort();
    }
    if (data)
      p->set_address((addr_t) * (size_t *)data);
    else {
      // clSetKernelArg was passed NULL pointer for data...
      // this is used for dynamically sized shared memory on NVIDIA platforms
      bool is_ptr_shared = false;
      if (i != m_ptx_kernel_param_info.end()) {
        is_ptr_shared = i->second.is_ptr_shared();
      }

      if (!is_ptr_shared and !p->is_shared()) {
        printf(
            "GPGPU-Sim PTX: ERROR ** clSetKernelArg passed NULL but arg not "
            "shared memory\n");
        abort();
      }
      unsigned num_bits = 8 * args->m_nbytes;
      printf(
          "GPGPU-Sim PTX: deferred allocation of shared region for \"%s\" from "
          "0x%lx to 0x%lx (shared memory space)\n",
          p->name().c_str(), m_symtab->get_shared_next(),
          m_symtab->get_shared_next() + num_bits / 8);
      fflush(stdout);
      assert((num_bits % 8) == 0);
      addr_t addr = m_symtab->get_shared_next();
      addr_t addr_pad =
          num_bits
              ? (((num_bits / 8) - (addr % (num_bits / 8))) % (num_bits / 8))
              : 0;
      p->set_address(addr + addr_pad);
      m_symtab->alloc_shared(num_bits / 8 + addr_pad);
    }
  }
}

unsigned function_info::get_args_aligned_size() {
  if (m_args_aligned_size >= 0) return m_args_aligned_size;

  unsigned param_address = 0;
  unsigned int total_size = 0;
  for (std::map<unsigned, param_info>::iterator i =
           m_ptx_kernel_param_info.begin();
       i != m_ptx_kernel_param_info.end(); i++) {
    param_info &p = i->second;
    std::string name = p.get_name();
    symbol *param = m_symtab->lookup(name.c_str());

    size_t arg_size = p.get_size() / 8;  // size of param in bytes
    total_size = (total_size + arg_size - 1) / arg_size * arg_size;  // aligned
    p.add_offset(total_size);
    param->set_address(param_address + total_size);
    total_size += arg_size;
  }

  m_args_aligned_size = (total_size + 3) / 4 * 4;  // final size aligned to word

  return m_args_aligned_size;
}

void function_info::finalize(memory_space *param_mem) {
  unsigned param_address = 0;
  for (std::map<unsigned, param_info>::iterator i =
           m_ptx_kernel_param_info.begin();
       i != m_ptx_kernel_param_info.end(); i++) {
    param_info &p = i->second;
    if (p.is_ptr_shared())
      continue;  // Pointer to local memory: Should we pass the allocated shared
                 // memory address to the param memory space?
    std::string name = p.get_name();
    int type = p.get_type();
    param_t param_value = p.get_value();
    param_value.type = type;
    symbol *param = m_symtab->lookup(name.c_str());
    unsigned xtype = param->type()->get_key().scalar_type();
    assert(xtype == (unsigned)type);
    size_t size;
    size = param_value.size;  // size of param in bytes
    // assert(param_value.offset == param_address);
    if (size != p.get_size() / 8) {
      printf(
          "GPGPU-Sim PTX: WARNING actual kernel paramter size = %zu bytes vs. "
          "formal size = %zu (using smaller of two)\n",
          size, p.get_size() / 8);
      size = (size < (p.get_size() / 8)) ? size : (p.get_size() / 8);
    }
    // copy the parameter over word-by-word so that parameter that crosses a
    // memory page can be copied over
    // Jin: copy parameter using aligned rules
    const type_info *paramtype = param->type();
    int align_amount = paramtype->get_key().get_alignment_spec();
    align_amount = (align_amount == -1) ? size : align_amount;
    param_address = (param_address + align_amount - 1) / align_amount *
                    align_amount;  // aligned

    const size_t word_size = 4;
    // param_address = (param_address + size - 1) / size * size; //aligned with
    // size
    for (size_t idx = 0; idx < size; idx += word_size) {
      const char *pdata = reinterpret_cast<const char *>(param_value.pdata) +
                          idx;  // cast to char * for ptr arithmetic
      param_mem->write(param_address + idx, word_size, pdata, NULL, NULL);
    }
    unsigned offset = p.get_offset();
    assert(offset == param_address);
    param->set_address(param_address);
    param_address += size;
  }
}

void function_info::param_to_shared(memory_space *shared_mem,
                                    symbol_table *symtab) {
  // TODO: call this only for PTXPlus with GT200 models
  // extern gpgpu_sim* g_the_gpu;
  // REMOVE: ptx
  // if (not
  // gpgpu_ctx->the_gpgpusim->g_the_gpu->get_config().convert_to_ptxplus())
  //   return;

  // copies parameters into simulated shared memory
  for (std::map<unsigned, param_info>::iterator i =
           m_ptx_kernel_param_info.begin();
       i != m_ptx_kernel_param_info.end(); i++) {
    param_info &p = i->second;
    if (p.is_ptr_shared())
      continue;  // Pointer to local memory: Should we pass the allocated shared
                 // memory address to the param memory space?
    std::string name = p.get_name();
    int type = p.get_type();
    param_t value = p.get_value();
    value.type = type;
    symbol *param = symtab->lookup(name.c_str());
    unsigned xtype = param->type()->get_key().scalar_type();
    assert(xtype == (unsigned)type);

    int tmp;
    size_t size;
    unsigned offset = p.get_offset();
    type_info_key::type_decode(xtype, size, tmp);

    // Write to shared memory - offset + 0x10
    shared_mem->write(offset + 0x10, size / 8, value.pdata, NULL, NULL);
  }
}

void function_info::list_param(FILE *fout) const {
  for (std::map<unsigned, param_info>::const_iterator i =
           m_ptx_kernel_param_info.begin();
       i != m_ptx_kernel_param_info.end(); i++) {
    const param_info &p = i->second;
    std::string name = p.get_name();
    symbol *param = m_symtab->lookup(name.c_str());
    addr_t param_addr = param->get_address();
    fprintf(fout, "%s: %#08lx\n", name.c_str(), param_addr);
  }
  fflush(fout);
}

void function_info::ptx_jit_config(
    std::map<unsigned long long, size_t> mallocPtr_Size,
    memory_space *param_mem, gpgpu_t *gpu, dim3 gridDim, dim3 blockDim) {
  static unsigned long long counter = 0;
  std::vector<std::pair<size_t, unsigned char *>> param_data;
  std::vector<unsigned> offsets;
  std::vector<bool> paramIsPointer;

  char *gpgpusim_path = getenv("GPGPUSIM_ROOT");
  assert(gpgpusim_path != NULL);
  char *wys_exec_path = getenv("WYS_EXEC_PATH");
  assert(wys_exec_path != NULL);
  std::string command =
      std::string("mkdir ") + gpgpusim_path + "/debug_tools/WatchYourStep/data";
  std::string filename(std::string(gpgpusim_path) +
                       "/debug_tools/WatchYourStep/data/params.config" +
                       std::to_string(counter));

  // initialize paramList
  char buff[1024];
  std::string filename_c(filename + "_c");
  snprintf(buff, 1024, "c++filt %s > %s", get_name().c_str(),
           filename_c.c_str());
  assert(system(buff) != NULL);
  FILE *fp = fopen(filename_c.c_str(), "r");
  fgets(buff, 1024, fp);
  fclose(fp);
  std::string fn(buff);
  size_t pos1, pos2;
  pos1 = fn.find_last_of("(");
  pos2 = fn.find(")", pos1);
  assert(pos2 > pos1 && pos1 > 0);
  strcpy(buff, fn.substr(pos1 + 1, pos2 - pos1 - 1).c_str());
  char *tok;
  tok = strtok(buff, ",");
  std::string tmp;
  while (tok != NULL) {
    std::string param(tok);
    if (param.find("<") != std::string::npos) {
      assert(param.find(">") == std::string::npos);
      assert(param.find("*") == std::string::npos);
      tmp = param;
    } else {
      if (tmp.length() > 0) {
        tmp = "";
        assert(param.find(">") != std::string::npos);
        assert(param.find("<") == std::string::npos);
        assert(param.find("*") == std::string::npos);
      }
      printf("%s\n", param.c_str());
      if (param.find("*") != std::string::npos) {
        paramIsPointer.push_back(true);
      } else {
        paramIsPointer.push_back(false);
      }
    }
    tok = strtok(NULL, ",");
  }

  for (std::map<unsigned, param_info>::iterator i =
           m_ptx_kernel_param_info.begin();
       i != m_ptx_kernel_param_info.end(); i++) {
    param_info &p = i->second;
    std::string name = p.get_name();
    symbol *param = m_symtab->lookup(name.c_str());
    addr_t param_addr = param->get_address();
    param_t param_value = p.get_value();
    offsets.push_back((unsigned)p.get_offset());

    if (paramIsPointer[i->first] &&
        (*(unsigned long long *)param_value.pdata != 0)) {
      // is pointer
      assert(param_value.size == sizeof(void *) &&
             "MisID'd this param as pointer");
      size_t array_size = 0;
      unsigned long long param_pointer =
          *(unsigned long long *)param_value.pdata;
      if (mallocPtr_Size.find(param_pointer) != mallocPtr_Size.end()) {
        array_size = mallocPtr_Size[param_pointer];
      } else {
        for (std::map<unsigned long long, size_t>::iterator j =
                 mallocPtr_Size.begin();
             j != mallocPtr_Size.end(); j++) {
          if (param_pointer > j->first &&
              param_pointer < j->first + j->second) {
            array_size = j->first + j->second - param_pointer;
            break;
          }
        }
        assert(array_size > 0 && "pointer was not previously malloc'd");
      }

      unsigned char *val = (unsigned char *)malloc(param_value.size);
      param_mem->read(param_addr, param_value.size, (void *)val);
      unsigned char *array_val = (unsigned char *)malloc(array_size);
      gpu->get_global_memory()->read(*(unsigned *)((void *)val), array_size,
                                     (void *)array_val);
      param_data.push_back(
          std::pair<size_t, unsigned char *>(array_size, array_val));
      paramIsPointer.push_back(true);
    } else {
      unsigned char *val = (unsigned char *)malloc(param_value.size);
      param_mem->read(param_addr, param_value.size, (void *)val);
      param_data.push_back(
          std::pair<size_t, unsigned char *>(param_value.size, val));
      paramIsPointer.push_back(false);
    }
  }

  FILE *fout = fopen(filename.c_str(), "w");
  printf("Writing data to %s ...\n", filename.c_str());
  fprintf(fout, "%s\n", get_name().c_str());
  fprintf(fout, "%u,%u,%u %u,%u,%u\n", gridDim.x, gridDim.y, gridDim.z,
          blockDim.x, blockDim.y, blockDim.z);
  size_t index = 0;
  for (std::vector<std::pair<size_t, unsigned char *>>::const_iterator i =
           param_data.begin();
       i != param_data.end(); i++) {
    if (paramIsPointer[index]) {
      fprintf(fout, "*");
    }
    fprintf(fout, "%lu :", i->first);
    for (size_t j = 0; j < i->first; j++) {
      fprintf(fout, " %u", i->second[j]);
    }
    fprintf(fout, " : %u", offsets[index]);
    free(i->second);
    fprintf(fout, "\n");
    index++;
  }
  fflush(fout);
  fclose(fout);

  // ptx config
  std::string ptx_config_fn(std::string(gpgpusim_path) +
                            "/debug_tools/WatchYourStep/data/ptx.config" +
                            std::to_string(counter));
  snprintf(buff, 1024,
           "grep -rn \".entry %s\" %s/*.ptx | cut -d \":\" -f 1-2 > %s",
           get_name().c_str(), wys_exec_path, ptx_config_fn.c_str());
  if (system(buff) != 0) {
    printf("WARNING: Failed to execute grep to find ptx source \n");
    printf("Problematic call: %s", buff);
    abort();
  }
  FILE *fin = fopen(ptx_config_fn.c_str(), "r");
  char ptx_source[256];
  unsigned line_number;
  int numscanned = fscanf(fin, "%[^:]:%u", ptx_source, &line_number);
  assert(numscanned == 2);
  fclose(fin);
  snprintf(buff, 1024,
           "grep -rn \".version\" %s | cut -d \":\" -f 1 | xargs -I \"{}\" awk "
           "\"NR>={}&&NR<={}+2\" %s > %s",
           ptx_source, ptx_source, ptx_config_fn.c_str());
  if (system(buff) != 0) {
    printf("WARNING: Failed to execute grep to find ptx header \n");
    printf("Problematic call: %s", buff);
    abort();
  }
  fin = fopen(ptx_source, "r");
  assert(fin != NULL);
  printf("Writing data to %s ...\n", ptx_config_fn.c_str());
  fout = fopen(ptx_config_fn.c_str(), "a");
  assert(fout != NULL);
  for (unsigned i = 0; i < line_number; i++) {
    assert(fgets(buff, 1024, fin) != NULL);
    assert(!feof(fin));
  }
  fprintf(fout, "\n\n");
  do {
    fprintf(fout, "%s", buff);
    assert(fgets(buff, 1024, fin) != NULL);
    if (feof(fin)) {
      break;
    }
  } while (strstr(buff, "entry") == NULL);

  fclose(fin);
  fflush(fout);
  fclose(fout);
  counter++;
}
