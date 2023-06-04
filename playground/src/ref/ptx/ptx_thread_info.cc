#include "ptx_thread_info.hpp"

void ptx_thread_info::ptx_fetch_inst(inst_t &inst) const {
  addr_t pc = get_pc();
  const ptx_instruction *pI = m_func_info->get_instruction(pc);
  inst = (const inst_t &)*pI;
  assert(inst.valid());
}

void ptx_thread_info::ptx_exec_inst(warp_inst_t &inst, unsigned lane_id) {
  bool skip = false;
  int op_classification = 0;
  addr_t pc = next_instr();
  assert(pc ==
         inst.pc); // make sure timing model and functional model are in sync
  const ptx_instruction *pI = m_func_info->get_instruction(pc);

  set_npc(pc + pI->inst_size());

  try {
    clearRPC();
    m_last_set_operand_value.u64 = 0;

    if (is_done()) {
      printf("attempted to execute instruction on a thread that is already "
             "done.\n");
      assert(0);
    }

    if (g_debug_execution >= 6 ||
        m_gpu->get_config().get_ptx_inst_debug_to_file()) {
      if ((m_gpu->gpgpu_ctx->func_sim->g_debug_thread_uid == 0) ||
          (get_uid() ==
           (unsigned)(m_gpu->gpgpu_ctx->func_sim->g_debug_thread_uid))) {
        clear_modifiedregs();
        enable_debug_trace();
      }
    }

    if (pI->has_pred()) {
      const operand_info &pred = pI->get_pred();
      ptx_reg_t pred_value = get_operand_value(pred, pred, PRED_TYPE, this, 0);
      if (pI->get_pred_mod() == -1) {
        skip = (pred_value.pred & 0x0001) ^
               pI->get_pred_neg(); // ptxplus inverts the zero flag
      } else {
        skip = !pred_lookup(pI->get_pred_mod(), pred_value.pred & 0x000F);
      }
    }
    int inst_opcode = pI->get_opcode();

    if (skip) {
      inst.set_not_active(lane_id);
    } else {
      const ptx_instruction *pI_saved = pI;
      ptx_instruction *pJ = NULL;
      if (pI->get_opcode() == VOTE_OP || pI->get_opcode() == ACTIVEMASK_OP) {
        pJ = new ptx_instruction(*pI);
        *((warp_inst_t *)pJ) = inst; // copy active mask information
        pI = pJ;
      }

      if (((inst_opcode == MMA_OP || inst_opcode == MMA_LD_OP ||
            inst_opcode == MMA_ST_OP))) {
        if (inst.active_count() != MAX_WARP_SIZE) {
          printf(
              "Tensor Core operation are warp synchronous operation. All the "
              "threads needs to be active.");
          assert(0);
        }
      }

      // Tensorcore is warp synchronous operation. So these instructions needs
      // to be executed only once. To make the simulation faster removing the
      // redundant tensorcore operation
      if (!tensorcore_op(inst_opcode) ||
          ((tensorcore_op(inst_opcode)) && (lane_id == 0))) {
        switch (inst_opcode) {
#define OP_DEF(OP, FUNC, STR, DST, CLASSIFICATION)                             \
  case OP:                                                                     \
    FUNC(pI, this);                                                            \
    op_classification = CLASSIFICATION;                                        \
    break;
#define OP_W_DEF(OP, FUNC, STR, DST, CLASSIFICATION)                           \
  case OP:                                                                     \
    FUNC(pI, get_core(), inst);                                                \
    op_classification = CLASSIFICATION;                                        \
    break;
#include "opcodes.def"
#undef OP_DEF
#undef OP_W_DEF
        default:
          printf("Execution error: Invalid opcode (0x%x)\n", pI->get_opcode());
          break;
        }
      }
      delete pJ;
      pI = pI_saved;

      // Run exit instruction if exit option included
      if (pI->is_exit())
        exit_impl(pI, this);
    }

    const gpgpu_functional_sim_config &config = m_gpu->get_config();

    // Output instruction information to file and stdout
    if (config.get_ptx_inst_debug_to_file() != 0 &&
        (config.get_ptx_inst_debug_thread_uid() == 0 ||
         config.get_ptx_inst_debug_thread_uid() == get_uid())) {
      fprintf(m_gpu->get_ptx_inst_debug_file(), "[thd=%u] : (%s:%u - %s)\n",
              get_uid(), pI->source_file(), pI->source_line(),
              pI->get_source());
      // fprintf(ptx_inst_debug_file, "has memory read=%d, has memory
      // write=%d\n", pI->has_memory_read(), pI->has_memory_write());
      fflush(m_gpu->get_ptx_inst_debug_file());
    }

    if (m_gpu->gpgpu_ctx->func_sim->ptx_debug_exec_dump_cond<5>(get_uid(),
                                                                pc)) {
      dim3 ctaid = get_ctaid();
      dim3 tid = get_tid();
      printf("%u [thd=%u][i=%u] : ctaid=(%u,%u,%u) tid=(%u,%u,%u) icount=%u "
             "[pc=%u] (%s:%u - %s)  [0x%llx]\n",
             m_gpu->gpgpu_ctx->func_sim->g_ptx_sim_num_insn, get_uid(),
             pI->uid(), ctaid.x, ctaid.y, ctaid.z, tid.x, tid.y, tid.z,
             get_icount(), pc, pI->source_file(), pI->source_line(),
             pI->get_source(), m_last_set_operand_value.u64);
      fflush(stdout);
    }

    addr_t insn_memaddr = 0xFEEBDAED;
    memory_space_t insn_space = undefined_space;
    _memory_op_t insn_memory_op = no_memory_op;
    unsigned insn_data_size = 0;
    if ((pI->has_memory_read() || pI->has_memory_write())) {
      if (!((inst_opcode == MMA_LD_OP || inst_opcode == MMA_ST_OP))) {
        insn_memaddr = last_eaddr();
        insn_space = last_space();
        unsigned to_type = pI->get_type();
        insn_data_size = datatype2size(to_type);
        insn_memory_op = pI->has_memory_read() ? memory_load : memory_store;
      }
    }

    if (pI->get_opcode() == BAR_OP && pI->barrier_op() == RED_OPTION) {
      inst.add_callback(lane_id, last_callback().function,
                        last_callback().instruction, this,
                        false /*not atomic*/);
    }

    if (pI->get_opcode() == ATOM_OP) {
      insn_memaddr = last_eaddr();
      insn_space = last_space();
      inst.add_callback(lane_id, last_callback().function,
                        last_callback().instruction, this, true /*atomic*/);
      unsigned to_type = pI->get_type();
      insn_data_size = datatype2size(to_type);
    }

    if (pI->get_opcode() == TEX_OP) {
      inst.set_addr(lane_id, last_eaddr());
      assert(inst.space == last_space());
      insn_data_size = get_tex_datasize(
          pI,
          this); // texture obtain its data granularity from the texture info
    }

    // Output register information to file and stdout
    if (config.get_ptx_inst_debug_to_file() != 0 &&
        (config.get_ptx_inst_debug_thread_uid() == 0 ||
         config.get_ptx_inst_debug_thread_uid() == get_uid())) {
      dump_modifiedregs(m_gpu->get_ptx_inst_debug_file());
      dump_regs(m_gpu->get_ptx_inst_debug_file());
    }

    if (g_debug_execution >= 6) {
      if (m_gpu->gpgpu_ctx->func_sim->ptx_debug_exec_dump_cond<6>(get_uid(),
                                                                  pc))
        dump_modifiedregs(stdout);
    }
    if (g_debug_execution >= 10) {
      if (m_gpu->gpgpu_ctx->func_sim->ptx_debug_exec_dump_cond<10>(get_uid(),
                                                                   pc))
        dump_regs(stdout);
    }
    update_pc();
    m_gpu->gpgpu_ctx->func_sim->g_ptx_sim_num_insn++;

    // not using it with functional simulation mode
    if (!(this->m_functionalSimulationMode))
      ptx_file_line_stats_add_exec_count(pI);

    if (m_gpu->gpgpu_ctx->func_sim->gpgpu_ptx_instruction_classification) {
      m_gpu->gpgpu_ctx->func_sim->init_inst_classification_stat();
      unsigned space_type = 0;
      switch (pI->get_space().get_type()) {
      case global_space:
        space_type = 10;
        break;
      case local_space:
        space_type = 11;
        break;
      case tex_space:
        space_type = 12;
        break;
      case surf_space:
        space_type = 13;
        break;
      case param_space_kernel:
      case param_space_local:
        space_type = 14;
        break;
      case shared_space:
        space_type = 15;
        break;
      case const_space:
        space_type = 16;
        break;
      default:
        space_type = 0;
        break;
      }
      StatAddSample(m_gpu->gpgpu_ctx->func_sim->g_inst_classification_stat
                        [m_gpu->gpgpu_ctx->func_sim->g_ptx_kernel_count],
                    op_classification);
      if (space_type)
        StatAddSample(m_gpu->gpgpu_ctx->func_sim->g_inst_classification_stat
                          [m_gpu->gpgpu_ctx->func_sim->g_ptx_kernel_count],
                      (int)space_type);
      StatAddSample(m_gpu->gpgpu_ctx->func_sim->g_inst_op_classification_stat
                        [m_gpu->gpgpu_ctx->func_sim->g_ptx_kernel_count],
                    (int)pI->get_opcode());
    }
    if ((m_gpu->gpgpu_ctx->func_sim->g_ptx_sim_num_insn % 100000) == 0) {
      dim3 ctaid = get_ctaid();
      dim3 tid = get_tid();
      DPRINTF(LIVENESS,
              "GPGPU-Sim PTX: %u instructions simulated : ctaid=(%u,%u,%u) "
              "tid=(%u,%u,%u)\n",
              m_gpu->gpgpu_ctx->func_sim->g_ptx_sim_num_insn, ctaid.x, ctaid.y,
              ctaid.z, tid.x, tid.y, tid.z);
      fflush(stdout);
    }

    // "Return values"
    if (!skip) {
      if (!((inst_opcode == MMA_LD_OP || inst_opcode == MMA_ST_OP))) {
        inst.space = insn_space;
        inst.set_addr(lane_id, insn_memaddr);
        inst.data_size = insn_data_size; // simpleAtomicIntrinsics
        assert(inst.memory_op == insn_memory_op);
      }
    }

  } catch (int x) {
    printf("GPGPU-Sim PTX: ERROR (%d) executing intruction (%s:%u)\n", x,
           pI->source_file(), pI->source_line());
    printf("GPGPU-Sim PTX:       '%s'\n", pI->get_source());
    abort();
  }
}

void ptx_thread_info::set_reg(const symbol *reg, const ptx_reg_t &value) {
  assert(reg != NULL);
  if (reg->name() == "_")
    return;
  assert(!m_regs.empty());
  assert(reg->uid() > 0);
  m_regs.back()[reg] = value;
  if (m_enable_debug_trace)
    m_debug_trace_regs_modified.back()[reg] = value;
  m_last_set_operand_value = value;
}

void ptx_thread_info::print_reg_thread(char *fname) {
  FILE *fp = fopen(fname, "w");
  assert(fp != NULL);

  int size = m_regs.size();

  if (size > 0) {
    reg_map_t reg = m_regs.back();

    reg_map_t::const_iterator it;
    for (it = reg.begin(); it != reg.end(); ++it) {
      const std::string &name = it->first->name();
      const std::string &dec = it->first->decl_location();
      unsigned size = it->first->get_size_in_bytes();
      fprintf(fp, "%s %llu %s %d\n", name.c_str(), it->second, dec.c_str(),
              size);
    }
    // m_regs.pop_back();
  }
  fclose(fp);
}

void ptx_thread_info::resume_reg_thread(char *fname, symbol_table *symtab) {
  FILE *fp2 = fopen(fname, "r");
  assert(fp2 != NULL);
  // m_regs.push_back( reg_map_t() );
  char line[200];
  while (fgets(line, sizeof line, fp2) != NULL) {
    symbol *reg;
    char *pch;
    pch = strtok(line, " ");
    char *name = pch;
    reg = symtab->lookup(name);
    ptx_reg_t data;
    pch = strtok(NULL, " ");
    data = atoi(pch);
    pch = strtok(NULL, " ");
    pch = strtok(NULL, " ");
    m_regs.back()[reg] = data;
  }
  fclose(fp2);
}

ptx_reg_t ptx_thread_info::get_reg(const symbol *reg) {
  static bool unfound_register_warned = false;
  assert(reg != NULL);
  assert(!m_regs.empty());
  reg_map_t::iterator regs_iter = m_regs.back().find(reg);
  if (regs_iter == m_regs.back().end()) {
    assert(reg->type()->get_key().is_reg());
    const std::string &name = reg->name();
    unsigned call_uid = m_callstack.back().m_call_uid;
    ptx_reg_t uninit_reg;
    uninit_reg.u32 = 0x0;
    set_reg(reg, uninit_reg); // give it a value since we are going to warn the
                              // user anyway
    std::string file_loc = get_location();
    if (!unfound_register_warned) {
      printf("GPGPU-Sim PTX: WARNING (%s) ** reading undefined register \'%s\' "
             "(cuid:%u). Setting to 0X00000000. This is okay if you are "
             "simulating the native ISA"
             "\n",
             file_loc.c_str(), name.c_str(), call_uid);
      unfound_register_warned = true;
    }
    regs_iter = m_regs.back().find(reg);
  }
  if (m_enable_debug_trace)
    m_debug_trace_regs_read.back()[reg] = regs_iter->second;
  return regs_iter->second;
}

ptx_reg_t ptx_thread_info::get_operand_value(const operand_info &op,
                                             operand_info dstInfo,
                                             unsigned opType,
                                             ptx_thread_info *thread,
                                             int derefFlag) {
  ptx_reg_t result, tmp;

  if (op.get_double_operand_type() == 0) {
    if (((opType != BB128_TYPE) && (opType != BB64_TYPE) &&
         (opType != FF64_TYPE)) ||
        (op.get_addr_space() != undefined_space)) {
      if (op.is_reg()) {
        result = get_reg(op.get_symbol());
      } else if (op.is_builtin()) {
        result.u32 = get_builtin(op.get_int(), op.get_addr_offset());
      } else if (op.is_immediate_address()) {
        result.u64 = op.get_addr_offset();
      } else if (op.is_memory_operand()) {
        // a few options here...
        const symbol *sym = op.get_symbol();
        const type_info *type = sym->type();
        const type_info_key &info = type->get_key();

        if (info.is_reg()) {
          const symbol *name = op.get_symbol();
          result.u64 = get_reg(name).u64 + op.get_addr_offset();
        } else if (info.is_param_kernel()) {
          result.u64 = sym->get_address() + op.get_addr_offset();
        } else if (info.is_param_local()) {
          result.u64 = sym->get_address() + op.get_addr_offset();
        } else if (info.is_global()) {
          assert(op.get_addr_offset() == 0);
          result.u64 = sym->get_address();
        } else if (info.is_local()) {
          result.u64 = sym->get_address() + op.get_addr_offset();
        } else if (info.is_const()) {
          result.u64 = sym->get_address() + op.get_addr_offset();
        } else if (op.is_shared()) {
          result.u64 = op.get_symbol()->get_address() + op.get_addr_offset();
        } else if (op.is_sstarr()) {
          result.u64 = op.get_symbol()->get_address() + op.get_addr_offset();
        } else {
          const char *name = op.name().c_str();
          printf("GPGPU-Sim PTX: ERROR ** get_operand_value : unknown memory "
                 "operand type for %s\n",
                 name);
          abort();
        }

      } else if (op.is_literal()) {
        result = op.get_literal_value();
      } else if (op.is_label()) {
        result.u64 = op.get_symbol()->get_address();
      } else if (op.is_shared()) {
        result.u64 = op.get_symbol()->get_address();
      } else if (op.is_sstarr()) {
        result.u64 = op.get_symbol()->get_address();
      } else if (op.is_const()) {
        result.u64 = op.get_symbol()->get_address();
      } else if (op.is_global()) {
        result.u64 = op.get_symbol()->get_address();
      } else if (op.is_local()) {
        result.u64 = op.get_symbol()->get_address();
      } else if (op.is_function_address()) {
        result.u64 = (size_t)op.get_symbol()->get_pc();
      } else if (op.is_param_kernel()) {
        result.u64 = op.get_symbol()->get_address();
      } else {
        const char *name = op.name().c_str();
        const symbol *sym2 = op.get_symbol();
        const type_info *type2 = sym2->type();
        const type_info_key &info2 = type2->get_key();
        if (info2.is_param_kernel()) {
          result.u64 = sym2->get_address() + op.get_addr_offset();
        } else {
          printf("GPGPU-Sim PTX: ERROR ** get_operand_value : unknown operand "
                 "type for %s\n",
                 name);
          assert(0);
        }
      }

      if (op.get_operand_lohi() == 1)
        result.u64 = result.u64 & 0xFFFF;
      else if (op.get_operand_lohi() == 2)
        result.u64 = (result.u64 >> 16) & 0xFFFF;
    } else if (opType == BB128_TYPE) {
      // b128
      result.u128.lowest = get_reg(op.vec_symbol(0)).u32;
      result.u128.low = get_reg(op.vec_symbol(1)).u32;
      result.u128.high = get_reg(op.vec_symbol(2)).u32;
      result.u128.highest = get_reg(op.vec_symbol(3)).u32;
    } else {
      // bb64 or ff64
      result.bits.ls = get_reg(op.vec_symbol(0)).u32;
      result.bits.ms = get_reg(op.vec_symbol(1)).u32;
    }
  } else if (op.get_double_operand_type() == 1) {
    ptx_reg_t firstHalf, secondHalf;
    firstHalf.u64 = get_reg(op.vec_symbol(0)).u64;
    secondHalf.u64 = get_reg(op.vec_symbol(1)).u64;
    if (op.get_operand_lohi() == 1)
      secondHalf.u64 = secondHalf.u64 & 0xFFFF;
    else if (op.get_operand_lohi() == 2)
      secondHalf.u64 = (secondHalf.u64 >> 16) & 0xFFFF;
    result.u64 = firstHalf.u64 + secondHalf.u64;
  } else if (op.get_double_operand_type() == 2) {
    // s[reg1 += reg2]
    // reg1 is incremented after value is returned: the value returned is
    // s[reg1]
    ptx_reg_t firstHalf, secondHalf;
    firstHalf.u64 = get_reg(op.vec_symbol(0)).u64;
    secondHalf.u64 = get_reg(op.vec_symbol(1)).u64;
    if (op.get_operand_lohi() == 1)
      secondHalf.u64 = secondHalf.u64 & 0xFFFF;
    else if (op.get_operand_lohi() == 2)
      secondHalf.u64 = (secondHalf.u64 >> 16) & 0xFFFF;
    result.u64 = firstHalf.u64;
    firstHalf.u64 = firstHalf.u64 + secondHalf.u64;
    set_reg(op.vec_symbol(0), firstHalf);
  } else if (op.get_double_operand_type() == 3) {
    // s[reg += immediate]
    // reg is incremented after value is returned: the value returned is s[reg]
    ptx_reg_t firstHalf;
    firstHalf.u64 = get_reg(op.get_symbol()).u64;
    result.u64 = firstHalf.u64;
    firstHalf.u64 = firstHalf.u64 + op.get_addr_offset();
    set_reg(op.get_symbol(), firstHalf);
  }

  ptx_reg_t finalResult;
  memory_space *mem = NULL;
  size_t size = 0;
  int t = 0;
  finalResult.u64 = 0;

  // complete other cases for reading from memory, such as reading from other
  // const memory
  if ((op.get_addr_space() == global_space) && (derefFlag)) {
    // global memory - g[4], g[$r0]
    mem = thread->get_global_memory();
    type_info_key::type_decode(opType, size, t);
    mem->read(result.u32, size / 8, &finalResult.u128);
    thread->m_last_effective_address = result.u32;
    thread->m_last_memory_space = global_space;

    if (opType == S16_TYPE || opType == S32_TYPE)
      sign_extend(finalResult, size, dstInfo);
  } else if ((op.get_addr_space() == shared_space) && (derefFlag)) {
    // shared memory - s[4], s[$r0]
    mem = thread->m_shared_mem;
    type_info_key::type_decode(opType, size, t);
    mem->read(result.u32, size / 8, &finalResult.u128);
    thread->m_last_effective_address = result.u32;
    thread->m_last_memory_space = shared_space;

    if (opType == S16_TYPE || opType == S32_TYPE)
      sign_extend(finalResult, size, dstInfo);
  } else if ((op.get_addr_space() == const_space) && (derefFlag)) {
    // const memory - ce0c1[4], ce0c1[$r0]
    mem = thread->get_global_memory();
    type_info_key::type_decode(opType, size, t);
    mem->read((result.u32 + op.get_const_mem_offset()), size / 8,
              &finalResult.u128);
    thread->m_last_effective_address = result.u32;
    thread->m_last_memory_space = const_space;
    if (opType == S16_TYPE || opType == S32_TYPE)
      sign_extend(finalResult, size, dstInfo);
  } else if ((op.get_addr_space() == local_space) && (derefFlag)) {
    // local memory - l0[4], l0[$r0]
    mem = thread->m_local_mem;
    type_info_key::type_decode(opType, size, t);
    mem->read(result.u32, size / 8, &finalResult.u128);
    thread->m_last_effective_address = result.u32;
    thread->m_last_memory_space = local_space;
    if (opType == S16_TYPE || opType == S32_TYPE)
      sign_extend(finalResult, size, dstInfo);
  } else {
    finalResult = result;
  }

  if ((op.get_operand_neg() == true) && (derefFlag)) {
    switch (opType) {
    // Default to f32 for now, need to add support for others
    case S8_TYPE:
    case U8_TYPE:
    case B8_TYPE:
      finalResult.s8 = -finalResult.s8;
      break;
    case S16_TYPE:
    case U16_TYPE:
    case B16_TYPE:
      finalResult.s16 = -finalResult.s16;
      break;
    case S32_TYPE:
    case U32_TYPE:
    case B32_TYPE:
      finalResult.s32 = -finalResult.s32;
      break;
    case S64_TYPE:
    case U64_TYPE:
    case B64_TYPE:
      finalResult.s64 = -finalResult.s64;
      break;
    case F16_TYPE:
      finalResult.f16 = -finalResult.f16;
      break;
    case F32_TYPE:
      finalResult.f32 = -finalResult.f32;
      break;
    case F64_TYPE:
    case FF64_TYPE:
      finalResult.f64 = -finalResult.f64;
      break;
    default:
      assert(0);
    }
  }

  return finalResult;
}

void ptx_thread_info::get_vector_operand_values(const operand_info &op,
                                                ptx_reg_t *ptx_regs,
                                                unsigned num_elements) {
  assert(op.is_vector());
  assert(num_elements <= 8);

  for (int idx = num_elements - 1; idx >= 0; --idx) {
    const symbol *sym = NULL;
    sym = op.vec_symbol(idx);
    if (strcmp(sym->name().c_str(), "_") != 0) {
      reg_map_t::iterator reg_iter = m_regs.back().find(sym);
      assert(reg_iter != m_regs.back().end());
      ptx_regs[idx] = reg_iter->second;
    }
  }
}

void ptx_thread_info::set_operand_value(const operand_info &dst,
                                        const ptx_reg_t &data, unsigned type,
                                        ptx_thread_info *thread,
                                        const ptx_instruction *pI, int overflow,
                                        int carry) {
  thread->set_operand_value(dst, data, type, thread, pI);

  if (dst.get_double_operand_type() == -2) {
    ptx_reg_t predValue;

    const symbol *sym = dst.vec_symbol(0);
    predValue.u64 = (m_regs.back()[sym].u64) & ~(0x0C);
    predValue.u64 |= ((overflow & 0x01) << 3);
    predValue.u64 |= ((carry & 0x01) << 2);

    set_reg(sym, predValue);
  } else if (dst.get_double_operand_type() == 0) {
    // intentionally do nothing
  } else {
    printf("Unexpected double destination\n");
    assert(0);
  }
}

void ptx_thread_info::set_operand_value(const operand_info &dst,
                                        const ptx_reg_t &data, unsigned type,
                                        ptx_thread_info *thread,
                                        const ptx_instruction *pI) {
  ptx_reg_t dstData;
  memory_space *mem = NULL;
  size_t size;
  int t;

  type_info_key::type_decode(type, size, t);

  /*complete this section for other cases*/
  if (dst.get_addr_space() == undefined_space) {
    ptx_reg_t setValue;
    setValue.u64 = data.u64;

    // Double destination in set instruction ($p0|$p1) - second is negation of
    // first
    if (dst.get_double_operand_type() == -1) {
      ptx_reg_t setValue2;
      const symbol *name1 = dst.vec_symbol(0);
      const symbol *name2 = dst.vec_symbol(1);

      if ((type == F16_TYPE) || (type == F32_TYPE) || (type == F64_TYPE) ||
          (type == FF64_TYPE)) {
        setValue2.f32 = (setValue.u64 == 0) ? 1.0f : 0.0f;
      } else {
        setValue2.u32 = (setValue.u64 == 0) ? 0xFFFFFFFF : 0;
      }

      set_reg(name1, setValue);
      set_reg(name2, setValue2);
    }

    // Double destination in cvt,shr,mul,etc. instruction ($p0|$r4) - second
    // register operand receives data, first predicate operand is set as
    // $p0=($r4!=0) Also for Double destination in set instruction ($p0/$r1)
    else if ((dst.get_double_operand_type() == -2) ||
             (dst.get_double_operand_type() == -3)) {
      ptx_reg_t predValue;
      const symbol *predName = dst.vec_symbol(0);
      const symbol *regName = dst.vec_symbol(1);
      predValue.u64 = 0;

      switch (type) {
      case S8_TYPE:
        if ((setValue.s8 & 0x7F) == 0)
          predValue.u64 |= 1;
        break;
      case S16_TYPE:
        if ((setValue.s16 & 0x7FFF) == 0)
          predValue.u64 |= 1;
        break;
      case S32_TYPE:
        if ((setValue.s32 & 0x7FFFFFFF) == 0)
          predValue.u64 |= 1;
        break;
      case S64_TYPE:
        if ((setValue.s64 & 0x7FFFFFFFFFFFFFFF) == 0)
          predValue.u64 |= 1;
        break;
      case U8_TYPE:
      case B8_TYPE:
        if (setValue.u8 == 0)
          predValue.u64 |= 1;
        break;
      case U16_TYPE:
      case B16_TYPE:
        if (setValue.u16 == 0)
          predValue.u64 |= 1;
        break;
      case U32_TYPE:
      case B32_TYPE:
        if (setValue.u32 == 0)
          predValue.u64 |= 1;
        break;
      case U64_TYPE:
      case B64_TYPE:
        if (setValue.u64 == 0)
          predValue.u64 |= 1;
        break;
      case F16_TYPE:
        if (setValue.f16 == 0)
          predValue.u64 |= 1;
        break;
      case F32_TYPE:
        if (setValue.f32 == 0)
          predValue.u64 |= 1;
        break;
      case F64_TYPE:
      case FF64_TYPE:
        if (setValue.f64 == 0)
          predValue.u64 |= 1;
        break;
      default:
        assert(0);
        break;
      }

      if ((type == S8_TYPE) || (type == S16_TYPE) || (type == S32_TYPE) ||
          (type == S64_TYPE) || (type == U8_TYPE) || (type == U16_TYPE) ||
          (type == U32_TYPE) || (type == U64_TYPE) || (type == B8_TYPE) ||
          (type == B16_TYPE) || (type == B32_TYPE) || (type == B64_TYPE)) {
        if ((setValue.u32 & (1 << (size - 1))) != 0)
          predValue.u64 |= 1 << 1;
      }
      if (type == F32_TYPE) {
        if (setValue.f32 < 0)
          predValue.u64 |= 1 << 1;
      }

      if (dst.get_operand_lohi() == 1) {
        setValue.u64 =
            ((m_regs.back()[regName].u64) & (~(0xFFFF))) + (data.u64 & 0xFFFF);
      } else if (dst.get_operand_lohi() == 2) {
        setValue.u64 = ((m_regs.back()[regName].u64) & (~(0xFFFF0000))) +
                       ((data.u64 << 16) & 0xFFFF0000);
      }

      set_reg(predName, predValue);
      set_reg(regName, setValue);
    } else if (type == BB128_TYPE) {
      // b128 stuff here.
      ptx_reg_t setValue2, setValue3, setValue4;
      setValue.u64 = 0;
      setValue2.u64 = 0;
      setValue3.u64 = 0;
      setValue4.u64 = 0;
      setValue.u32 = data.u128.lowest;
      setValue2.u32 = data.u128.low;
      setValue3.u32 = data.u128.high;
      setValue4.u32 = data.u128.highest;

      const symbol *name1, *name2, *name3, *name4 = NULL;

      name1 = dst.vec_symbol(0);
      name2 = dst.vec_symbol(1);
      name3 = dst.vec_symbol(2);
      name4 = dst.vec_symbol(3);

      set_reg(name1, setValue);
      set_reg(name2, setValue2);
      set_reg(name3, setValue3);
      set_reg(name4, setValue4);
    } else if (type == BB64_TYPE || type == FF64_TYPE) {
      // ptxplus version of storing 64 bit values to registers stores to two
      // adjacent registers
      ptx_reg_t setValue2;
      setValue.u32 = 0;
      setValue2.u32 = 0;

      setValue.u32 = data.bits.ls;
      setValue2.u32 = data.bits.ms;

      const symbol *name1, *name2 = NULL;

      name1 = dst.vec_symbol(0);
      name2 = dst.vec_symbol(1);

      set_reg(name1, setValue);
      set_reg(name2, setValue2);
    } else {
      if (dst.get_operand_lohi() == 1) {
        setValue.u64 = ((m_regs.back()[dst.get_symbol()].u64) & (~(0xFFFF))) +
                       (data.u64 & 0xFFFF);
      } else if (dst.get_operand_lohi() == 2) {
        setValue.u64 =
            ((m_regs.back()[dst.get_symbol()].u64) & (~(0xFFFF0000))) +
            ((data.u64 << 16) & 0xFFFF0000);
      }
      set_reg(dst.get_symbol(), setValue);
    }
  }

  // global memory - g[4], g[$r0]
  else if (dst.get_addr_space() == global_space) {
    dstData = thread->get_operand_value(dst, dst, type, thread, 0);
    mem = thread->get_global_memory();
    type_info_key::type_decode(type, size, t);

    mem->write(dstData.u32, size / 8, &data.u128, thread, pI);
    thread->m_last_effective_address = dstData.u32;
    thread->m_last_memory_space = global_space;
  }

  // shared memory - s[4], s[$r0]
  else if (dst.get_addr_space() == shared_space) {
    dstData = thread->get_operand_value(dst, dst, type, thread, 0);
    mem = thread->m_shared_mem;
    type_info_key::type_decode(type, size, t);

    mem->write(dstData.u32, size / 8, &data.u128, thread, pI);
    thread->m_last_effective_address = dstData.u32;
    thread->m_last_memory_space = shared_space;
  }

  // local memory - l0[4], l0[$r0]
  else if (dst.get_addr_space() == local_space) {
    dstData = thread->get_operand_value(dst, dst, type, thread, 0);
    mem = thread->m_local_mem;
    type_info_key::type_decode(type, size, t);

    mem->write(dstData.u32, size / 8, &data.u128, thread, pI);
    thread->m_last_effective_address = dstData.u32;
    thread->m_last_memory_space = local_space;
  }

  else {
    printf("Destination stores to unknown location.");
    assert(0);
  }
}

void ptx_thread_info::set_vector_operand_values(const operand_info &dst,
                                                const ptx_reg_t &data1,
                                                const ptx_reg_t &data2,
                                                const ptx_reg_t &data3,
                                                const ptx_reg_t &data4) {
  unsigned num_elements = dst.get_vect_nelem();
  if (num_elements > 0) {
    set_reg(dst.vec_symbol(0), data1);
    if (num_elements > 1) {
      set_reg(dst.vec_symbol(1), data2);
      if (num_elements > 2) {
        set_reg(dst.vec_symbol(2), data3);
        if (num_elements > 3) {
          set_reg(dst.vec_symbol(3), data4);
        }
      }
    }
  }

  m_last_set_operand_value = data1;
}

void ptx_thread_info::set_wmma_vector_operand_values(
    const operand_info &dst, const ptx_reg_t &data1, const ptx_reg_t &data2,
    const ptx_reg_t &data3, const ptx_reg_t &data4, const ptx_reg_t &data5,
    const ptx_reg_t &data6, const ptx_reg_t &data7, const ptx_reg_t &data8) {
  unsigned num_elements = dst.get_vect_nelem();
  if (num_elements == 8) {
    set_reg(dst.vec_symbol(0), data1);
    set_reg(dst.vec_symbol(1), data2);
    set_reg(dst.vec_symbol(2), data3);
    set_reg(dst.vec_symbol(3), data4);
    set_reg(dst.vec_symbol(4), data5);
    set_reg(dst.vec_symbol(5), data6);
    set_reg(dst.vec_symbol(6), data7);
    set_reg(dst.vec_symbol(7), data8);
  } else {
    printf("error:set_wmma_vector_operands");
  }

  m_last_set_operand_value = data8;
}

ptx_thread_info::~ptx_thread_info() {
  m_gpu->gpgpu_ctx->func_sim->g_ptx_thread_info_delete_count++;
}

ptx_thread_info::ptx_thread_info(kernel_info_t &kernel) : m_kernel(kernel) {
  m_uid = kernel.entry()->gpgpu_ctx->func_sim->g_ptx_thread_info_uid_next++;
  m_core = NULL;
  m_barrier_num = -1;
  m_at_barrier = false;
  m_valid = false;
  m_gridid = 0;
  m_thread_done = false;
  m_cycle_done = 0;
  m_PC = 0;
  m_icount = 0;
  m_last_effective_address = 0;
  m_last_memory_space = undefined_space;
  m_branch_taken = 0;
  m_shared_mem = NULL;
  m_sstarr_mem = NULL;
  m_warp_info = NULL;
  m_cta_info = NULL;
  m_local_mem = NULL;
  m_symbol_table = NULL;
  m_func_info = NULL;
  m_hw_tid = -1;
  m_hw_wid = -1;
  m_hw_sid = -1;
  m_last_dram_callback.function = NULL;
  m_last_dram_callback.instruction = NULL;
  m_regs.push_back(reg_map_t());
  m_debug_trace_regs_modified.push_back(reg_map_t());
  m_debug_trace_regs_read.push_back(reg_map_t());
  m_callstack.push_back(stack_entry());
  m_RPC = -1;
  m_RPC_updated = false;
  m_last_was_call = false;
  m_enable_debug_trace = false;
  m_local_mem_stack_pointer = 0;
  m_gpu = NULL;
  m_last_set_operand_value = ptx_reg_t();
}

const ptx_version &ptx_thread_info::get_ptx_version() const {
  return m_func_info->get_ptx_version();
}

void ptx_thread_info::set_done() {
  assert(!m_at_barrier);
  m_thread_done = true;
  m_cycle_done = m_gpu->gpu_sim_cycle;
}

unsigned ptx_thread_info::get_builtin(int builtin_id, unsigned dim_mod) {
  assert(m_valid);
  switch ((builtin_id & 0xFFFF)) {
  case CLOCK_REG:
    return (unsigned)(m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
  case CLOCK64_REG:
    abort(); // change return value to unsigned long long?
             // GPGPUSim clock is 4 times slower - multiply by 4
    return (m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) * 4;
  case HALFCLOCK_ID:
    // GPGPUSim clock is 4 times slower - multiply by 4
    // Hardware clock counter is incremented at half the shader clock
    // frequency - divide by 2 (Henry '10)
    return (m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) * 2;
  case CTAID_REG:
    assert(dim_mod < 3);
    if (dim_mod == 0)
      return m_ctaid.x;
    if (dim_mod == 1)
      return m_ctaid.y;
    if (dim_mod == 2)
      return m_ctaid.z;
    abort();
    break;
  case ENVREG_REG: {
    int index = builtin_id >> 16;
    dim3 gdim = this->get_core()->get_kernel_info()->get_grid_dim();
    switch (index) {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
      return 0;
      break;
    case 6:
      return gdim.x;
    case 7:
      return gdim.y;
    case 8:
      return gdim.z;
    case 9:
      if (gdim.z == 1 && gdim.y == 1)
        return 1;
      else if (gdim.z == 1)
        return 2;
      else
        return 3;
      break;
    default:
      break;
    }
  }
  case GRIDID_REG:
    return m_gridid;
  case LANEID_REG:
    return get_hw_tid() % m_core->get_warp_size();
  case LANEMASK_EQ_REG:
    feature_not_implemented("%lanemask_eq");
    return 0;
  case LANEMASK_LE_REG:
    feature_not_implemented("%lanemask_le");
    return 0;
  case LANEMASK_LT_REG:
    feature_not_implemented("%lanemask_lt");
    return 0;
  case LANEMASK_GE_REG:
    feature_not_implemented("%lanemask_ge");
    return 0;
  case LANEMASK_GT_REG:
    feature_not_implemented("%lanemask_gt");
    return 0;
  case NCTAID_REG:
    assert(dim_mod < 3);
    if (dim_mod == 0)
      return m_nctaid.x;
    if (dim_mod == 1)
      return m_nctaid.y;
    if (dim_mod == 2)
      return m_nctaid.z;
    abort();
    break;
  case NTID_REG:
    assert(dim_mod < 3);
    if (dim_mod == 0)
      return m_ntid.x;
    if (dim_mod == 1)
      return m_ntid.y;
    if (dim_mod == 2)
      return m_ntid.z;
    abort();
    break;
  case NWARPID_REG:
    feature_not_implemented("%nwarpid");
    return 0;
  case PM_REG:
    feature_not_implemented("%pm");
    return 0;
  case SMID_REG:
    feature_not_implemented("%smid");
    return 0;
  case TID_REG:
    assert(dim_mod < 3);
    if (dim_mod == 0)
      return m_tid.x;
    if (dim_mod == 1)
      return m_tid.y;
    if (dim_mod == 2)
      return m_tid.z;
    abort();
    break;
  case WARPSZ_REG:
    return m_core->get_warp_size();
  default:
    assert(0);
  }
  return 0;
}

void ptx_thread_info::set_info(function_info *func) {
  m_symbol_table = func->get_symtab();
  m_func_info = func;
  m_PC = func->get_start_PC();
}

void ptx_thread_info::cpy_tid_to_reg(dim3 tid) {
  // copies %tid.x, %tid.y and %tid.z into $r0
  ptx_reg_t data;
  data.s64 = 0;

  data.u32 = (tid.x + (tid.y << 16) + (tid.z << 26));

  const symbol *r0 = m_symbol_table->lookup("$r0");
  if (r0) {
    // No need to set pid if kernel doesn't use it
    set_reg(r0, data);
  }
}

void ptx_thread_info::print_insn(unsigned pc, FILE *fp) const {
  m_func_info->print_insn(pc, fp);
}

void ptx_thread_info::callstack_push(unsigned pc, unsigned rpc,
                                     const symbol *return_var_src,
                                     const symbol *return_var_dst,
                                     unsigned call_uid) {
  m_RPC = -1;
  m_RPC_updated = true;
  m_last_was_call = true;
  assert(m_func_info != NULL);
  m_callstack.push_back(stack_entry(m_symbol_table, m_func_info, pc, rpc,
                                    return_var_src, return_var_dst, call_uid));
  m_regs.push_back(reg_map_t());
  m_debug_trace_regs_modified.push_back(reg_map_t());
  m_debug_trace_regs_read.push_back(reg_map_t());
  m_local_mem_stack_pointer += m_func_info->local_mem_framesize();
}

// ptxplus version of callstack_push.
void ptx_thread_info::callstack_push_plus(unsigned pc, unsigned rpc,
                                          const symbol *return_var_src,
                                          const symbol *return_var_dst,
                                          unsigned call_uid) {
  m_RPC = -1;
  m_RPC_updated = true;
  m_last_was_call = true;
  assert(m_func_info != NULL);
  m_callstack.push_back(stack_entry(m_symbol_table, m_func_info, pc, rpc,
                                    return_var_src, return_var_dst, call_uid));
  // m_regs.push_back( reg_map_t() );
  // m_debug_trace_regs_modified.push_back( reg_map_t() );
  // m_debug_trace_regs_read.push_back( reg_map_t() );
  m_local_mem_stack_pointer += m_func_info->local_mem_framesize();
}

bool ptx_thread_info::callstack_pop() {
  const symbol *rv_src = m_callstack.back().m_return_var_src;
  const symbol *rv_dst = m_callstack.back().m_return_var_dst;
  assert(!((rv_src != NULL) ^
           (rv_dst != NULL))); // ensure caller and callee agree on whether
                               // there is a return value

  // read return value from callee frame
  arg_buffer_t buffer(m_gpu->gpgpu_ctx);
  if (rv_src != NULL)
    buffer = copy_arg_to_buffer(this, operand_info(rv_src, m_gpu->gpgpu_ctx),
                                rv_dst);

  m_symbol_table = m_callstack.back().m_symbol_table;
  m_NPC = m_callstack.back().m_PC;
  m_RPC_updated = true;
  m_last_was_call = false;
  m_RPC = m_callstack.back().m_RPC;
  m_func_info = m_callstack.back().m_func_info;
  if (m_func_info) {
    assert(m_local_mem_stack_pointer >= m_func_info->local_mem_framesize());
    m_local_mem_stack_pointer -= m_func_info->local_mem_framesize();
  }
  m_callstack.pop_back();
  m_regs.pop_back();
  m_debug_trace_regs_modified.pop_back();
  m_debug_trace_regs_read.pop_back();

  // write return value into caller frame
  if (rv_dst != NULL)
    copy_buffer_to_frame(this, buffer);

  return m_callstack.empty();
}

// ptxplus version of callstack_pop
bool ptx_thread_info::callstack_pop_plus() {
  const symbol *rv_src = m_callstack.back().m_return_var_src;
  const symbol *rv_dst = m_callstack.back().m_return_var_dst;
  assert(!((rv_src != NULL) ^
           (rv_dst != NULL))); // ensure caller and callee agree on whether
                               // there is a return value

  // read return value from callee frame
  arg_buffer_t buffer(m_gpu->gpgpu_ctx);
  if (rv_src != NULL)
    buffer = copy_arg_to_buffer(this, operand_info(rv_src, m_gpu->gpgpu_ctx),
                                rv_dst);

  m_symbol_table = m_callstack.back().m_symbol_table;
  m_NPC = m_callstack.back().m_PC;
  m_RPC_updated = true;
  m_last_was_call = false;
  m_RPC = m_callstack.back().m_RPC;
  m_func_info = m_callstack.back().m_func_info;
  if (m_func_info) {
    assert(m_local_mem_stack_pointer >= m_func_info->local_mem_framesize());
    m_local_mem_stack_pointer -= m_func_info->local_mem_framesize();
  }
  m_callstack.pop_back();
  // m_regs.pop_back();
  // m_debug_trace_regs_modified.pop_back();
  // m_debug_trace_regs_read.pop_back();

  // write return value into caller frame
  if (rv_dst != NULL)
    copy_buffer_to_frame(this, buffer);

  return m_callstack.empty();
}

void ptx_thread_info::dump_callstack() const {
  std::list<stack_entry>::const_iterator c = m_callstack.begin();
  std::list<reg_map_t>::const_iterator r = m_regs.begin();

  printf("\n\n");
  printf("Call stack for thread uid = %u (sc=%u, hwtid=%u)\n", m_uid, m_hw_sid,
         m_hw_tid);
  while (c != m_callstack.end() && r != m_regs.end()) {
    const stack_entry &c_e = *c;
    const reg_map_t &regs = *r;
    if (!c_e.m_valid) {
      printf("  <entry>                              #regs = %zu\n",
             regs.size());
    } else {
      printf("  %20s  PC=%3u RV= (callee=\'%s\',caller=\'%s\') #regs = %zu\n",
             c_e.m_func_info->get_name().c_str(), c_e.m_PC,
             c_e.m_return_var_src->name().c_str(),
             c_e.m_return_var_dst->name().c_str(), regs.size());
    }
    c++;
    r++;
  }
  if (c != m_callstack.end() || r != m_regs.end()) {
    printf("  *** mismatch in m_regs and m_callstack sizes ***\n");
  }
  printf("\n\n");
}

std::string ptx_thread_info::get_location() const {
  const ptx_instruction *pI = m_func_info->get_instruction(m_PC);
  char buf[1024];
  snprintf(buf, 1024, "%s:%u", pI->source_file(), pI->source_line());
  return std::string(buf);
}

const ptx_instruction *ptx_thread_info::get_inst() const {
  return m_func_info->get_instruction(m_PC);
}

const ptx_instruction *ptx_thread_info::get_inst(addr_t pc) const {
  return m_func_info->get_instruction(pc);
}

void ptx_thread_info::dump_regs(FILE *fp) {
  if (m_regs.empty())
    return;
  if (m_regs.back().empty())
    return;
  fprintf(fp, "Register File Contents:\n");
  fflush(fp);
  reg_map_t::const_iterator r;
  for (r = m_regs.back().begin(); r != m_regs.back().end(); ++r) {
    const symbol *sym = r->first;
    ptx_reg_t value = r->second;
    std::string name = sym->name();
    print_reg(fp, name, value, m_symbol_table);
  }
}

void ptx_thread_info::dump_modifiedregs(FILE *fp) {
  if (!(m_debug_trace_regs_modified.empty() ||
        m_debug_trace_regs_modified.back().empty())) {
    fprintf(fp, "Output Registers:\n");
    fflush(fp);
    reg_map_t::iterator r;
    for (r = m_debug_trace_regs_modified.back().begin();
         r != m_debug_trace_regs_modified.back().end(); ++r) {
      const symbol *sym = r->first;
      std::string name = sym->name();
      ptx_reg_t value = r->second;
      print_reg(fp, name, value, m_symbol_table);
    }
  }
  if (!(m_debug_trace_regs_read.empty() ||
        m_debug_trace_regs_read.back().empty())) {
    fprintf(fp, "Input Registers:\n");
    fflush(fp);
    reg_map_t::iterator r;
    for (r = m_debug_trace_regs_read.back().begin();
         r != m_debug_trace_regs_read.back().end(); ++r) {
      const symbol *sym = r->first;
      std::string name = sym->name();
      ptx_reg_t value = r->second;
      print_reg(fp, name, value, m_symbol_table);
    }
  }
}

void ptx_thread_info::push_breakaddr(const operand_info &breakaddr) {
  m_breakaddrs.push(breakaddr);
}

const operand_info &ptx_thread_info::pop_breakaddr() {
  if (m_breakaddrs.empty()) {
    printf("empty breakaddrs stack");
    assert(0);
  }
  operand_info &breakaddr = m_breakaddrs.top();
  m_breakaddrs.pop();
  return breakaddr;
}

void ptx_thread_info::set_npc(const function_info *f) {
  m_NPC = f->get_start_PC();
  m_func_info = const_cast<function_info *>(f);
  m_symbol_table = m_func_info->get_symtab();
}

void feature_not_implemented(const char *f) {
  printf("GPGPU-Sim: feature '%s' not supported\n", f);
  abort();
}
