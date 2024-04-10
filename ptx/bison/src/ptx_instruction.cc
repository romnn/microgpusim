#include "ptx_instruction.hpp"

#include "function_info.hpp"
#include "hal.hpp"
#include "ptx.parser.tab.h"

void ptx_instruction::set_fp_or_int_archop() {
  oprnd_type = UN_OP;
  if ((m_opcode == MEMBAR_OP) || (m_opcode == SSY_OP) || (m_opcode == BRA_OP) ||
      (m_opcode == BAR_OP) || (m_opcode == RET_OP) || (m_opcode == RETP_OP) ||
      (m_opcode == NOP_OP) || (m_opcode == EXIT_OP) || (m_opcode == CALLP_OP) ||
      (m_opcode == CALL_OP)) {
    // do nothing
  } else if ((m_opcode == CVT_OP || m_opcode == SET_OP ||
              m_opcode == SLCT_OP)) {
    if (get_type2() == F16_TYPE || get_type2() == F32_TYPE ||
        get_type2() == F64_TYPE || get_type2() == FF64_TYPE) {
      oprnd_type = FP_OP;
    } else
      oprnd_type = INT_OP;

  } else {
    if (get_type() == F16_TYPE || get_type() == F32_TYPE ||
        get_type() == F64_TYPE || get_type() == FF64_TYPE) {
      oprnd_type = FP_OP;
    } else
      oprnd_type = INT_OP;
  }
}

void ptx_instruction::set_mul_div_or_other_archop() {
  sp_op = OTHER_OP;
  if ((m_opcode != MEMBAR_OP) && (m_opcode != SSY_OP) && (m_opcode != BRA_OP) &&
      (m_opcode != BAR_OP) && (m_opcode != EXIT_OP) && (m_opcode != NOP_OP) &&
      (m_opcode != RETP_OP) && (m_opcode != RET_OP) && (m_opcode != CALLP_OP) &&
      (m_opcode != CALL_OP)) {
    if (get_type() == F64_TYPE || get_type() == FF64_TYPE) {
      switch (get_opcode()) {
      case MUL_OP:
      case MAD_OP:
      case FMA_OP:
        sp_op = DP_MUL_OP;
        break;
      case DIV_OP:
      case REM_OP:
        sp_op = DP_DIV_OP;
        break;
      case RCP_OP:
        sp_op = DP_DIV_OP;
        break;
      case LG2_OP:
        sp_op = FP_LG_OP;
        break;
      case RSQRT_OP:
      case SQRT_OP:
        sp_op = FP_SQRT_OP;
        break;
      case SIN_OP:
      case COS_OP:
        sp_op = FP_SIN_OP;
        break;
      case EX2_OP:
        sp_op = FP_EXP_OP;
        break;
      case MMA_OP:
        sp_op = TENSOR__OP;
        break;
      case TEX_OP:
        sp_op = TEX__OP;
        break;
      default:
        if ((op == DP_OP) || (op == ALU_OP))
          sp_op = DP___OP;
        break;
      }
    } else if (get_type() == F16_TYPE || get_type() == F32_TYPE) {
      switch (get_opcode()) {
      case MUL_OP:
      case MAD_OP:
      case FMA_OP:
        sp_op = FP_MUL_OP;
        break;
      case DIV_OP:
      case REM_OP:
        sp_op = FP_DIV_OP;
        break;
      case RCP_OP:
        sp_op = FP_DIV_OP;
        break;
      case LG2_OP:
        sp_op = FP_LG_OP;
        break;
      case RSQRT_OP:
      case SQRT_OP:
        sp_op = FP_SQRT_OP;
        break;
      case SIN_OP:
      case COS_OP:
        sp_op = FP_SIN_OP;
        break;
      case EX2_OP:
        sp_op = FP_EXP_OP;
        break;
      case MMA_OP:
        sp_op = TENSOR__OP;
        break;
      case TEX_OP:
        sp_op = TEX__OP;
        break;
      default:
        if ((op == SP_OP) || (op == ALU_OP))
          sp_op = FP__OP;
        break;
      }
    } else {
      switch (get_opcode()) {
      case MUL24_OP:
      case MAD24_OP:
        sp_op = INT_MUL24_OP;
        break;
      case MUL_OP:
      case MAD_OP:
      case FMA_OP:
        if (get_type() == U32_TYPE || get_type() == S32_TYPE ||
            get_type() == B32_TYPE)
          sp_op = INT_MUL32_OP;
        else
          sp_op = INT_MUL_OP;
        break;
      case DIV_OP:
      case REM_OP:
        sp_op = INT_DIV_OP;
        break;
      case MMA_OP:
        sp_op = TENSOR__OP;
        break;
      case TEX_OP:
        sp_op = TEX__OP;
        break;
      default:
        if ((op == INTP_OP) || (op == ALU_OP))
          sp_op = INT__OP;
        break;
      }
    }
  }
}

void ptx_instruction::set_bar_type() {
  if (m_opcode == BAR_OP) {
    switch (m_barrier_op) {
    case SYNC_OPTION:
      bar_type = SYNC;
      break;
    case ARRIVE_OPTION:
      bar_type = ARRIVE;
      break;
    case RED_OPTION:
      bar_type = RED;
      switch (m_atomic_spec) {
      case ATOMIC_POPC:
        red_type = POPC_RED;
        break;
      case ATOMIC_AND:
        red_type = AND_RED;
        break;
      case ATOMIC_OR:
        red_type = OR_RED;
        break;
      }
      break;
    default:
      abort();
    }
  } else if (m_opcode == SST_OP) {
    bar_type = SYNC;
  }
}

void ptx_instruction::set_opcode_and_latency() {
  unsigned int_latency[6];
  unsigned fp_latency[5];
  unsigned dp_latency[5];
  unsigned sfu_latency;
  unsigned tensor_latency;
  unsigned int_init[6];
  unsigned fp_init[5];
  unsigned dp_init[5];
  unsigned sfu_init;
  unsigned tensor_init;
  /*
   * [0] ADD,SUB
   * [1] MAX,Min
   * [2] MUL
   * [3] MAD
   * [4] DIV
   * [5] SHFL
   */
  sscanf(gpgpu_ctx->func_sim->opcode_latency_int, "%u,%u,%u,%u,%u,%u",
         &int_latency[0], &int_latency[1], &int_latency[2], &int_latency[3],
         &int_latency[4], &int_latency[5]);
  sscanf(gpgpu_ctx->func_sim->opcode_latency_fp, "%u,%u,%u,%u,%u",
         &fp_latency[0], &fp_latency[1], &fp_latency[2], &fp_latency[3],
         &fp_latency[4]);
  sscanf(gpgpu_ctx->func_sim->opcode_latency_dp, "%u,%u,%u,%u,%u",
         &dp_latency[0], &dp_latency[1], &dp_latency[2], &dp_latency[3],
         &dp_latency[4]);
  sscanf(gpgpu_ctx->func_sim->opcode_latency_sfu, "%u", &sfu_latency);
  sscanf(gpgpu_ctx->func_sim->opcode_latency_tensor, "%u", &tensor_latency);
  sscanf(gpgpu_ctx->func_sim->opcode_initiation_int, "%u,%u,%u,%u,%u,%u",
         &int_init[0], &int_init[1], &int_init[2], &int_init[3], &int_init[4],
         &int_init[5]);
  sscanf(gpgpu_ctx->func_sim->opcode_initiation_fp, "%u,%u,%u,%u,%u",
         &fp_init[0], &fp_init[1], &fp_init[2], &fp_init[3], &fp_init[4]);
  sscanf(gpgpu_ctx->func_sim->opcode_initiation_dp, "%u,%u,%u,%u,%u",
         &dp_init[0], &dp_init[1], &dp_init[2], &dp_init[3], &dp_init[4]);
  sscanf(gpgpu_ctx->func_sim->opcode_initiation_sfu, "%u", &sfu_init);
  sscanf(gpgpu_ctx->func_sim->opcode_initiation_tensor, "%u", &tensor_init);
  sscanf(gpgpu_ctx->func_sim->cdp_latency_str, "%u,%u,%u,%u,%u",
         &gpgpu_ctx->func_sim->cdp_latency[0],
         &gpgpu_ctx->func_sim->cdp_latency[1],
         &gpgpu_ctx->func_sim->cdp_latency[2],
         &gpgpu_ctx->func_sim->cdp_latency[3],
         &gpgpu_ctx->func_sim->cdp_latency[4]);

  if (!m_operands.empty()) {
    std::vector<operand_info>::iterator it;
    for (it = ++m_operands.begin(); it != m_operands.end(); it++) {
      num_operands++;
      if ((it->is_reg() || it->is_vector())) {
        num_regs++;
      }
    }
  }
  op = ALU_OP;
  mem_op = NOT_TEX;
  initiation_interval = latency = 1;
  switch (m_opcode) {
  case MOV_OP:
    assert(!(has_memory_read() && has_memory_write()));
    if (has_memory_read())
      op = LOAD_OP;
    if (has_memory_write())
      op = STORE_OP;
    break;
  case LD_OP:
    op = LOAD_OP;
    break;
  case MMA_LD_OP:
    op = TENSOR_CORE_LOAD_OP;
    break;
  case LDU_OP:
    op = LOAD_OP;
    break;
  case ST_OP:
    op = STORE_OP;
    break;
  case MMA_ST_OP:
    op = TENSOR_CORE_STORE_OP;
    break;
  case BRA_OP:
    op = BRANCH_OP;
    break;
  case BREAKADDR_OP:
    op = BRANCH_OP;
    break;
  case TEX_OP:
    op = LOAD_OP;
    mem_op = TEX;
    break;
  case ATOM_OP:
    op = LOAD_OP;
    break;
  case BAR_OP:
    op = BARRIER_OP;
    break;
  case SST_OP:
    op = BARRIER_OP;
    break;
  case MEMBAR_OP:
    op = MEMORY_BARRIER_OP;
    break;
  case CALL_OP: {
    if (m_is_printf || m_is_cdp) {
      op = ALU_OP;
    } else
      op = CALL_OPS;
    break;
  }
  case CALLP_OP: {
    if (m_is_printf || m_is_cdp) {
      op = ALU_OP;
    } else
      op = CALL_OPS;
    break;
  }
  case RET_OP:
  case RETP_OP:
    op = RET_OPS;
    break;
  case ADD_OP:
  case ADDP_OP:
  case ADDC_OP:
  case SUB_OP:
  case SUBC_OP:
    // ADD,SUB latency
    switch (get_type()) {
    case F32_TYPE:
      latency = fp_latency[0];
      initiation_interval = fp_init[0];
      op = SP_OP;
      break;
    case F64_TYPE:
    case FF64_TYPE:
      latency = dp_latency[0];
      initiation_interval = dp_init[0];
      op = DP_OP;
      break;
    case B32_TYPE:
    case U32_TYPE:
    case S32_TYPE:
    default: // Use int settings for default
      latency = int_latency[0];
      initiation_interval = int_init[0];
      op = INTP_OP;
      break;
    }
    break;
  case MAX_OP:
  case MIN_OP:
    // MAX,MIN latency
    switch (get_type()) {
    case F32_TYPE:
      latency = fp_latency[1];
      initiation_interval = fp_init[1];
      op = SP_OP;
      break;
    case F64_TYPE:
    case FF64_TYPE:
      latency = dp_latency[1];
      initiation_interval = dp_init[1];
      op = DP_OP;
      break;
    case B32_TYPE:
    case U32_TYPE:
    case S32_TYPE:
    default: // Use int settings for default
      latency = int_latency[1];
      initiation_interval = int_init[1];
      op = INTP_OP;
      break;
    }
    break;
  case MUL_OP:
    // MUL latency
    switch (get_type()) {
    case F32_TYPE:
      latency = fp_latency[2];
      initiation_interval = fp_init[2];
      op = SP_OP;
      break;
    case F64_TYPE:
    case FF64_TYPE:
      latency = dp_latency[2];
      initiation_interval = dp_init[2];
      op = DP_OP;
      break;
    case B32_TYPE:
    case U32_TYPE:
    case S32_TYPE:
    default: // Use int settings for default
      latency = int_latency[2];
      initiation_interval = int_init[2];
      op = INTP_OP;
      break;
    }
    break;
  case MAD_OP:
  case MADC_OP:
  case MADP_OP:
  case FMA_OP:
    // MAD latency
    switch (get_type()) {
    case F32_TYPE:
      latency = fp_latency[3];
      initiation_interval = fp_init[3];
      op = SP_OP;
      break;
    case F64_TYPE:
    case FF64_TYPE:
      latency = dp_latency[3];
      initiation_interval = dp_init[3];
      op = DP_OP;
      break;
    case B32_TYPE:
    case U32_TYPE:
    case S32_TYPE:
    default: // Use int settings for default
      latency = int_latency[3];
      initiation_interval = int_init[3];
      op = INTP_OP;
      break;
    }
    break;
  case MUL24_OP: // MUL24 is performed on mul32 units (with additional
                 // instructions for bitmasking) on devices with compute
                 // capability >1.x
    latency = int_latency[2] + 1;
    initiation_interval = int_init[2] + 1;
    op = INTP_OP;
    break;
  case MAD24_OP:
    latency = int_latency[3] + 1;
    initiation_interval = int_init[3] + 1;
    op = INTP_OP;
    break;
  case DIV_OP:
  case REM_OP:
    // Floating point only
    op = SFU_OP;
    switch (get_type()) {
    case F32_TYPE:
      latency = fp_latency[4];
      initiation_interval = fp_init[4];
      break;
    case F64_TYPE:
    case FF64_TYPE:
      latency = dp_latency[4];
      initiation_interval = dp_init[4];
      break;
    case B32_TYPE:
    case U32_TYPE:
    case S32_TYPE:
    default: // Use int settings for default
      latency = int_latency[4];
      initiation_interval = int_init[4];
      break;
    }
    break;
  case SQRT_OP:
  case SIN_OP:
  case COS_OP:
  case EX2_OP:
  case LG2_OP:
  case RSQRT_OP:
  case RCP_OP:
    latency = sfu_latency;
    initiation_interval = sfu_init;
    op = SFU_OP;
    break;
  case MMA_OP:
    latency = tensor_latency;
    initiation_interval = tensor_init;
    op = TENSOR_CORE_OP;
    break;
  case SHFL_OP:
    latency = int_latency[5];
    initiation_interval = int_init[5];
    break;
  default:
    break;
  }
  set_fp_or_int_archop();
  set_mul_div_or_other_archop();
}

static unsigned datatype2size(unsigned data_type) {
  unsigned data_size;
  switch (data_type) {
  case B8_TYPE:
  case S8_TYPE:
  case U8_TYPE:
    data_size = 1;
    break;
  case B16_TYPE:
  case S16_TYPE:
  case U16_TYPE:
  case F16_TYPE:
    data_size = 2;
    break;
  case B32_TYPE:
  case S32_TYPE:
  case U32_TYPE:
  case F32_TYPE:
    data_size = 4;
    break;
  case B64_TYPE:
  case BB64_TYPE:
  case S64_TYPE:
  case U64_TYPE:
  case F64_TYPE:
  case FF64_TYPE:
    data_size = 8;
    break;
  case BB128_TYPE:
    data_size = 16;
    break;
  default:
    assert(0);
    break;
  }
  return data_size;
}

void ptx_instruction::pre_decode() {
  pc = m_PC;
  isize = m_inst_size;
  for (unsigned i = 0; i < MAX_OUTPUT_VALUES; i++) {
    out[i] = 0;
  }
  for (unsigned i = 0; i < MAX_INPUT_VALUES; i++) {
    in[i] = 0;
  }
  incount = 0;
  outcount = 0;
  is_vectorin = 0;
  is_vectorout = 0;
  std::fill_n(arch_reg.src, MAX_REG_OPERANDS, -1);
  std::fill_n(arch_reg.dst, MAX_REG_OPERANDS, -1);
  pred = 0;
  ar1 = 0;
  ar2 = 0;
  space = m_space_spec;
  memory_op = no_memory_op;
  data_size = 0;
  if (has_memory_read() || has_memory_write()) {
    unsigned to_type = get_type();
    data_size = datatype2size(to_type);
    memory_op = has_memory_read() ? memory_load : memory_store;
  }

  bool has_dst = false;

  switch (get_opcode()) {
#define OP_DEF(OP, FUNC, STR, DST, CLASSIFICATION)                             \
  case OP:                                                                     \
    has_dst = (DST != 0);                                                      \
    break;
#define OP_W_DEF(OP, FUNC, STR, DST, CLASSIFICATION)                           \
  case OP:                                                                     \
    has_dst = (DST != 0);                                                      \
    break;
#include "opcodes.def"
#undef OP_DEF
#undef OP_W_DEF
  default:
    printf("Execution error: Invalid opcode (0x%x)\n", get_opcode());
    break;
  }

  switch (m_cache_option) {
  case CA_OPTION:
    cache_op = CACHE_ALL;
    break;
  case NC_OPTION:
    cache_op = CACHE_L1;
    break;
  case CG_OPTION:
    cache_op = CACHE_GLOBAL;
    break;
  case CS_OPTION:
    cache_op = CACHE_STREAMING;
    break;
  case LU_OPTION:
    cache_op = CACHE_LAST_USE;
    break;
  case CV_OPTION:
    cache_op = CACHE_VOLATILE;
    break;
  case WB_OPTION:
    cache_op = CACHE_WRITE_BACK;
    break;
  case WT_OPTION:
    cache_op = CACHE_WRITE_THROUGH;
    break;
  default:
    // if( m_opcode == LD_OP || m_opcode == LDU_OP )
    if (m_opcode == MMA_LD_OP || m_opcode == LD_OP || m_opcode == LDU_OP)
      cache_op = CACHE_ALL;
    // else if( m_opcode == ST_OP )
    else if (m_opcode == MMA_ST_OP || m_opcode == ST_OP)
      cache_op = CACHE_WRITE_BACK;
    else if (m_opcode == ATOM_OP)
      cache_op = CACHE_GLOBAL;
    break;
  }

  set_opcode_and_latency();
  set_bar_type();
  // Get register operands
  int n = 0, m = 0;
  ptx_instruction::const_iterator opr = op_iter_begin();
  for (; opr != op_iter_end(); opr++, n++) { // process operands
    const operand_info &o = *opr;
    if (has_dst && n == 0) {
      // Do not set the null register "_" as an architectural register
      if (o.is_reg() && !o.is_non_arch_reg()) {
        out[0] = o.reg_num();
        arch_reg.dst[0] = o.arch_reg_num();
      } else if (o.is_vector()) {
        is_vectorin = 1;
        unsigned num_elem = o.get_vect_nelem();
        if (num_elem >= 1)
          out[0] = o.reg1_num();
        if (num_elem >= 2)
          out[1] = o.reg2_num();
        if (num_elem >= 3)
          out[2] = o.reg3_num();
        if (num_elem >= 4)
          out[3] = o.reg4_num();
        if (num_elem >= 5)
          out[4] = o.reg5_num();
        if (num_elem >= 6)
          out[5] = o.reg6_num();
        if (num_elem >= 7)
          out[6] = o.reg7_num();
        if (num_elem >= 8)
          out[7] = o.reg8_num();
        for (int i = 0; i < num_elem; i++)
          arch_reg.dst[i] = o.arch_reg_num(i);
      }
    } else {
      if (o.is_reg() && !o.is_non_arch_reg()) {
        int reg_num = o.reg_num();
        arch_reg.src[m] = o.arch_reg_num();
        switch (m) {
        case 0:
          in[0] = reg_num;
          break;
        case 1:
          in[1] = reg_num;
          break;
        case 2:
          in[2] = reg_num;
          break;
        default:
          break;
        }
        m++;
      } else if (o.is_vector()) {
        // assert(m == 0); //only support 1 vector operand (for textures) right
        // now
        is_vectorout = 1;
        unsigned num_elem = o.get_vect_nelem();
        if (num_elem >= 1)
          in[m + 0] = o.reg1_num();
        if (num_elem >= 2)
          in[m + 1] = o.reg2_num();
        if (num_elem >= 3)
          in[m + 2] = o.reg3_num();
        if (num_elem >= 4)
          in[m + 3] = o.reg4_num();
        if (num_elem >= 5)
          in[m + 4] = o.reg5_num();
        if (num_elem >= 6)
          in[m + 5] = o.reg6_num();
        if (num_elem >= 7)
          in[m + 6] = o.reg7_num();
        if (num_elem >= 8)
          in[m + 7] = o.reg8_num();
        for (int i = 0; i < num_elem; i++)
          arch_reg.src[m + i] = o.arch_reg_num(i);
        m += num_elem;
      }
    }
  }

  // Setting number of input and output operands which is required for
  // scoreboard check
  for (int i = 0; i < MAX_OUTPUT_VALUES; i++)
    if (out[i] > 0)
      outcount++;

  for (int i = 0; i < MAX_INPUT_VALUES; i++)
    if (in[i] > 0)
      incount++;

  // Get predicate
  if (has_pred()) {
    const operand_info &p = get_pred();
    pred = p.reg_num();
  }

  // Get address registers inside memory operands.
  // Assuming only one memory operand per instruction,
  //  and maximum of two address registers for one memory operand.
  if (has_memory_read() || has_memory_write()) {
    ptx_instruction::const_iterator op = op_iter_begin();
    for (; op != op_iter_end(); op++, n++) { // process operands
      const operand_info &o = *op;

      if (o.is_memory_operand()) {
        // We do not support the null register as a memory operand
        assert(!o.is_non_arch_reg());

        // Check PTXPlus-type operand
        // memory operand with addressing (ex. s[0x4] or g[$r1])
        if (o.is_memory_operand2()) {
          // memory operand with one address register (ex. g[$r1+0x4] or
          // s[$r2+=0x4])
          if (o.get_double_operand_type() == 0 ||
              o.get_double_operand_type() == 3) {
            ar1 = o.reg_num();
            arch_reg.src[4] = o.arch_reg_num();
            // TODO: address register in $r2+=0x4 should be an output register
            // as well
          }
          // memory operand with two address register (ex. s[$r1+$r1] or
          // g[$r1+=$r2])
          else if (o.get_double_operand_type() == 1 ||
                   o.get_double_operand_type() == 2) {
            ar1 = o.reg1_num();
            arch_reg.src[4] = o.arch_reg_num();
            ar2 = o.reg2_num();
            arch_reg.src[5] = o.arch_reg_num();
            // TODO: first address register in $r1+=$r2 should be an output
            // register as well
          }
        } else if (o.is_immediate_address()) {
        }
        // Regular PTX operand
        else if (o.get_symbol()
                     ->type()
                     ->get_key()
                     .is_reg()) { // Memory operand contains a register
          ar1 = o.reg_num();
          arch_reg.src[4] = o.arch_reg_num();
        }
      }
    }
  }

  // get reconvergence pc
  reconvergence_pc = gpgpu_ctx->func_sim->get_converge_point(pc);

  m_decoded = true;
}

static std::list<operand_info>
check_operands(int opcode, const std::list<int> &scalar_type,
               const std::list<operand_info> &operands, gpgpu_context *ctx) {
  static int g_warn_literal_operands_two_type_inst;
  if ((opcode == CVT_OP) || (opcode == SET_OP) || (opcode == SLCT_OP) ||
      (opcode == TEX_OP) || (opcode == MMA_OP) || (opcode == DP4A_OP) ||
      (opcode == VMIN_OP) || (opcode == VMAX_OP)) {
    // just make sure these do not have have const operands...
    if (!g_warn_literal_operands_two_type_inst) {
      std::list<operand_info>::const_iterator o;
      for (o = operands.begin(); o != operands.end(); o++) {
        const operand_info &op = *o;
        if (op.is_literal()) {
          printf(
              "GPGPU-Sim PTX: PTX uses two scalar type intruction with literal "
              "operand.\n");
          g_warn_literal_operands_two_type_inst = 1;
        }
      }
    }
  } else {
    assert(scalar_type.size() < 2);
    if (scalar_type.size() == 1) {
      std::list<operand_info> result;
      int inst_type = scalar_type.front();
      std::list<operand_info>::const_iterator o;
      for (o = operands.begin(); o != operands.end(); o++) {
        const operand_info &op = *o;
        if (op.is_literal()) {
          if ((op.get_type() == double_op_t) && (inst_type == F32_TYPE)) {
            ptx_reg_t v = op.get_literal_value();
            float u = (float)v.f64;
            operand_info n(u, ctx);
            result.push_back(n);
          } else {
            result.push_back(op);
          }
        } else {
          result.push_back(op);
        }
      }
      return result;
    }
  }
  return operands;
}

ptx_instruction::ptx_instruction(
    int opcode, const symbol *pred, int neg_pred, int pred_mod, symbol *label,
    const std::list<operand_info> &operands, const operand_info &return_var,
    const std::list<int> &options, const std::list<int> &wmma_options,
    const std::list<int> &scalar_type, memory_space_t space_spec,
    const char *file, unsigned line, const char *source,
    const core_config *config, gpgpu_context *ctx)
    : warp_inst_t(config), m_return_var(ctx) {
  gpgpu_ctx = ctx;
  m_uid = ++(ctx->g_num_ptx_inst_uid);
  m_PC = 0;
  m_opcode = opcode;
  m_pred = pred;
  m_neg_pred = neg_pred;
  m_pred_mod = pred_mod;
  m_label = label;
  const std::list<operand_info> checked_operands =
      check_operands(opcode, scalar_type, operands, ctx);
  m_operands.insert(m_operands.begin(), checked_operands.begin(),
                    checked_operands.end());
  m_return_var = return_var;
  m_options = options;
  m_wmma_options = wmma_options;
  m_wide = false;
  m_hi = false;
  m_lo = false;
  m_uni = false;
  m_exit = false;
  m_abs = false;
  m_neg = false;
  m_to_option = false;
  m_cache_option = 0;
  m_rounding_mode = RN_OPTION;
  m_compare_op = -1;
  m_saturation_mode = 0;
  m_geom_spec = 0;
  m_vector_spec = 0;
  m_atomic_spec = 0;
  m_membar_level = 0;
  m_inst_size = 8; // bytes
  int rr = 0;
  std::list<int>::const_iterator i;
  unsigned n = 1;
  for (i = wmma_options.begin(); i != wmma_options.end(); i++, n++) {
    int last_ptx_inst_option = *i;
    switch (last_ptx_inst_option) {
    case SYNC_OPTION:
    case LOAD_A:
    case LOAD_B:
    case LOAD_C:
    case STORE_D:
    case MMA:
      m_wmma_type = last_ptx_inst_option;
      break;
    case ROW:
    case COL:
      m_wmma_layout[rr++] = last_ptx_inst_option;
      break;
    case M16N16K16:
    case M32N8K16:
    case M8N32K16:
      break;
    default:
      assert(0);
      break;
    }
  }
  rr = 0;
  n = 1;
  for (i = options.begin(); i != options.end(); i++, n++) {
    int last_ptx_inst_option = *i;
    switch (last_ptx_inst_option) {
    case SYNC_OPTION:
    case ARRIVE_OPTION:
    case RED_OPTION:
      m_barrier_op = last_ptx_inst_option;
      break;
    case EQU_OPTION:
    case NEU_OPTION:
    case LTU_OPTION:
    case LEU_OPTION:
    case GTU_OPTION:
    case GEU_OPTION:
    case EQ_OPTION:
    case NE_OPTION:
    case LT_OPTION:
    case LE_OPTION:
    case GT_OPTION:
    case GE_OPTION:
    case LS_OPTION:
    case HS_OPTION:
      m_compare_op = last_ptx_inst_option;
      break;
    case NUM_OPTION:
    case NAN_OPTION:
      m_compare_op = last_ptx_inst_option;
      // assert(0); // finish this
      break;
    case SAT_OPTION:
      m_saturation_mode = 1;
      break;
    case RNI_OPTION:
    case RZI_OPTION:
    case RMI_OPTION:
    case RPI_OPTION:
    case RN_OPTION:
    case RZ_OPTION:
    case RM_OPTION:
    case RP_OPTION:
      m_rounding_mode = last_ptx_inst_option;
      break;
    case HI_OPTION:
      m_compare_op = last_ptx_inst_option;
      m_hi = true;
      assert(!m_lo);
      assert(!m_wide);
      break;
    case LO_OPTION:
      m_compare_op = last_ptx_inst_option;
      m_lo = true;
      assert(!m_hi);
      assert(!m_wide);
      break;
    case WIDE_OPTION:
      m_wide = true;
      assert(!m_lo);
      assert(!m_hi);
      break;
    case UNI_OPTION:
      m_uni = true; // don't care... < now we DO care when constructing
                    // flowgraph>
      break;
    case GEOM_MODIFIER_1D:
    case GEOM_MODIFIER_2D:
    case GEOM_MODIFIER_3D:
      m_geom_spec = last_ptx_inst_option;
      break;
    case V2_TYPE:
    case V3_TYPE:
    case V4_TYPE:
      m_vector_spec = last_ptx_inst_option;
      break;
    case ATOMIC_AND:
    case ATOMIC_OR:
    case ATOMIC_XOR:
    case ATOMIC_CAS:
    case ATOMIC_EXCH:
    case ATOMIC_ADD:
    case ATOMIC_INC:
    case ATOMIC_DEC:
    case ATOMIC_MIN:
    case ATOMIC_MAX:
      m_atomic_spec = last_ptx_inst_option;
      break;
    case APPROX_OPTION:
      break;
    case FULL_OPTION:
      break;
    case ANY_OPTION:
      m_vote_mode = vote_any;
      break;
    case ALL_OPTION:
      m_vote_mode = vote_all;
      break;
    case BALLOT_OPTION:
      m_vote_mode = vote_ballot;
      break;
    case GLOBAL_OPTION:
      m_membar_level = GLOBAL_OPTION;
      break;
    case CTA_OPTION:
      m_membar_level = CTA_OPTION;
      break;
    case SYS_OPTION:
      m_membar_level = SYS_OPTION;
      break;
    case FTZ_OPTION:
      break;
    case EXIT_OPTION:
      m_exit = true;
      break;
    case ABS_OPTION:
      m_abs = true;
      break;
    case NEG_OPTION:
      m_neg = true;
      break;
    case TO_OPTION:
      m_to_option = true;
      break;
    case CA_OPTION:
    case CG_OPTION:
    case CS_OPTION:
    case LU_OPTION:
    case CV_OPTION:
    case WB_OPTION:
    case WT_OPTION:
      m_cache_option = last_ptx_inst_option;
      break;
    case HALF_OPTION:
      m_inst_size = 4; // bytes
      break;
    case EXTP_OPTION:
      break;
    case NC_OPTION:
      m_cache_option = last_ptx_inst_option;
      break;
    case UP_OPTION:
    case DOWN_OPTION:
    case BFLY_OPTION:
    case IDX_OPTION:
      m_shfl_op = last_ptx_inst_option;
      break;
    case PRMT_F4E_MODE:
    case PRMT_B4E_MODE:
    case PRMT_RC8_MODE:
    case PRMT_ECL_MODE:
    case PRMT_ECR_MODE:
    case PRMT_RC16_MODE:
      m_prmt_op = last_ptx_inst_option;
      break;
    default:
      assert(0);
      break;
    }
  }
  m_scalar_type = scalar_type;
  m_space_spec = space_spec;
  if ((opcode == ST_OP || opcode == LD_OP || opcode == LDU_OP) &&
      (space_spec == undefined_space)) {
    m_space_spec = generic_space;
  }
  for (std::vector<operand_info>::const_iterator i = m_operands.begin();
       i != m_operands.end(); ++i) {
    const operand_info &op = *i;
    if (op.get_addr_space() != undefined_space)
      // TODO: can have more than one memory
      // space for ptxplus (g8x) inst
      m_space_spec = op.get_addr_space();
  }
  if (opcode == TEX_OP)
    m_space_spec = tex_space;

  m_source_file = file ? file : "<unknown>";
  m_source_line = line;
  m_source = source;
  // Trim tabs
  m_source.erase(std::remove(m_source.begin(), m_source.end(), '\t'),
                 m_source.end());

  if (opcode == CALL_OP) {
    const operand_info &target = func_addr();
    assert(target.is_function_address());
    const symbol *func_addr = target.get_symbol();
    const function_info *target_func = func_addr->get_pc();
    std::string fname = target_func->get_name();

    if (fname == "vprintf") {
      m_is_printf = true;
    }
    if (fname == "cudaStreamCreateWithFlags")
      m_is_cdp = 1;
    if (fname == "cudaGetParameterBufferV2")
      m_is_cdp = 2;
    if (fname == "cudaLaunchDeviceV2")
      m_is_cdp = 4;
  }
}

void ptx_instruction::print_insn() const {
  print_insn(stdout);
  fflush(stdout);
}

void ptx_instruction::print_insn(FILE *fp) const {
  fprintf(fp, "%s", to_string().c_str());
}

#define STR_SIZE 1024

std::string ptx_instruction::to_string() const {
  char buf[STR_SIZE];
  unsigned used_bytes = 0;
  if (!is_label()) {
    used_bytes += snprintf(buf + used_bytes, STR_SIZE - used_bytes,
                           " PC=0x%03llx ", m_PC);
  } else {
    used_bytes +=
        snprintf(buf + used_bytes, STR_SIZE - used_bytes, "                ");
  }
  used_bytes +=
      snprintf(buf + used_bytes, STR_SIZE - used_bytes, "(%s:%d) %s",
               m_source_file.c_str(), m_source_line, m_source.c_str());
  return std::string(buf);
}
operand_info ptx_instruction::get_pred() const {
  return operand_info(m_pred, gpgpu_ctx);
}
