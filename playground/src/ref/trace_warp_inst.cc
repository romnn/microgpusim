#include "trace_warp_inst.hpp"

#include <iostream>

#include "inst_trace.hpp"
#include "kernel_trace.hpp"
#include "opcode.hpp"
#include "opcode_power_map.hpp"
#include "trace_config.hpp"
#include "trace_instr_opcode.hpp"

types_of_operands get_oprnd_type(op_type op, special_ops sp_op) {
  switch (op) {
  case SP_OP:
  case SFU_OP:
  case SPECIALIZED_UNIT_2_OP:
  case SPECIALIZED_UNIT_3_OP:
  case DP_OP:
  case LOAD_OP:
  case STORE_OP:
    return FP_OP;
  case INTP_OP:
  case SPECIALIZED_UNIT_4_OP:
    return INT_OP;
  case ALU_OP:
    if ((sp_op == FP__OP) || (sp_op == TEX__OP) || (sp_op == OTHER_OP))
      return FP_OP;
    else if (sp_op == INT__OP)
      return INT_OP;
  default:
    return UN_OP;
  }
}

const char *trace_warp_inst_t::opcode_str() const {
  if (m_opcode > 0 && m_opcode < SASS_NUM_OPCODES) {
    return g_trace_instr_opcode_str[m_opcode - 1];
  }
  return "<UNKNOWN>";
}

bool trace_warp_inst_t::parse_from_trace_struct(
    const inst_trace_t &trace,
    const std::unordered_map<std::string, OpcodeChar> *OpcodeMap,
    const class trace_config *tconfig,
    const class kernel_trace_t *kernel_trace_info) {
  // fill the inst_t and warp_inst_t params

  // fill active mask
  active_mask_t active_mask = trace.mask;
  set_active(active_mask);

  // fill and initialize common params
  m_decoded = true;
  pc = (address_type)trace.m_pc;

  isize =
      16; // starting from MAXWELL isize=16 bytes (including the control bytes)
  for (unsigned i = 0; i < MAX_OUTPUT_VALUES; i++) {
    out[i] = 0;
  }
  for (unsigned i = 0; i < MAX_INPUT_VALUES; i++) {
    in[i] = 0;
  }

  is_vectorin = 0;
  is_vectorout = 0;
  ar1 = 0;
  ar2 = 0;
  memory_op = no_memory_op;
  data_size = 0;
  op = ALU_OP;
  sp_op = OTHER_OP;
  mem_op = NOT_TEX;
  const_cache_operand = 0;
  oprnd_type = UN_OP;

  // get the opcode
  std::vector<std::string> opcode_tokens = trace.get_opcode_tokens();
  std::string opcode1 = opcode_tokens[0];

  std::unordered_map<std::string, OpcodeChar>::const_iterator it =
      OpcodeMap->find(opcode1);
  if (it != OpcodeMap->end()) {
    m_opcode = it->second.opcode;
    op = (op_type)(it->second.opcode_category);
    const std::unordered_map<unsigned, unsigned> *OpcPowerMap = &OpcodePowerMap;
    std::unordered_map<unsigned, unsigned>::const_iterator it2 =
        OpcPowerMap->find(m_opcode);
    if (it2 != OpcPowerMap->end())
      sp_op = (special_ops)(it2->second);
    oprnd_type = get_oprnd_type(op, sp_op);

    assert(m_opcode > 0);
    // printf("====> instruction %s parsed as %s (%u) [%lu active threads]\n",
    //        opcode1.c_str(), g_trace_instr_opcode_str[m_opcode - 1], m_opcode,
    //        active_mask.count());

  } else {
    std::cout << "ERROR:  undefined instruction : " << trace.opcode
              << " Opcode: " << opcode1 << std::endl;
    assert(0 && "undefined instruction");
  }
  std::string opcode = trace.opcode;
  if (opcode1 == "MUFU") { // Differentiate between different MUFU operations
                           // for power model
    if ((opcode == "MUFU.SIN") || (opcode == "MUFU.COS"))
      sp_op = FP_SIN_OP;
    if ((opcode == "MUFU.EX2") || (opcode == "MUFU.RCP"))
      sp_op = FP_EXP_OP;
    if (opcode == "MUFU.RSQ")
      sp_op = FP_SQRT_OP;
    if (opcode == "MUFU.LG2")
      sp_op = FP_LG_OP;
  }

  if (opcode1 == "IMAD") { // Differentiate between different IMAD operations
                           // for power model
    if ((opcode == "IMAD.MOV") || (opcode == "IMAD.IADD"))
      sp_op = INT__OP;
  }

  // fill regs information
  num_regs = trace.reg_srcs_num + trace.reg_dsts_num;
  num_operands = num_regs;
  outcount = trace.reg_dsts_num;
  for (unsigned m = 0; m < trace.reg_dsts_num; ++m) {
    out[m] = trace.reg_dest[m] + 1; // Increment by one because GPGPU-sim starts
                                    // from R1, while SASS starts from R0
    arch_reg.dst[m] = trace.reg_dest[m] + 1;
  }

  incount = trace.reg_srcs_num;
  for (unsigned m = 0; m < trace.reg_srcs_num; ++m) {
    in[m] = trace.reg_src[m] + 1; // Increment by one because GPGPU-sim starts
                                  // from R1, while SASS starts from R0
    arch_reg.src[m] = trace.reg_src[m] + 1;
  }

  // fill latency and initl
  tconfig->set_latency(op, latency, initiation_interval);

  // fill addresses
  if (trace.memadd_info != NULL) {
    data_size = trace.memadd_info->width;
    for (unsigned i = 0; i < warp_size(); ++i)
      set_addr(i, trace.memadd_info->addrs[i]);
  }

  // handle special cases and fill memory space
  switch (m_opcode) {
  case OP_LDC: // handle Load from Constant
    data_size = 4;
    memory_op = memory_load;
    const_cache_operand = 1;
    space.set_type(const_space);
    cache_op = CACHE_ALL;
    break;
  case OP_LDG:
  case OP_LDL:
    assert(data_size > 0);
    memory_op = memory_load;
    cache_op = CACHE_ALL;
    if (m_opcode == OP_LDL)
      space.set_type(local_space);
    else
      space.set_type(global_space);
    // check the cache scope, if its strong GPU, then bypass L1
    if (trace.check_opcode_contain(opcode_tokens, "STRONG") &&
        trace.check_opcode_contain(opcode_tokens, "GPU")) {
      cache_op = CACHE_GLOBAL;
    }
    break;
  case OP_STG:
  case OP_STL:
    assert(data_size > 0);
    memory_op = memory_store;
    cache_op = CACHE_ALL;
    if (m_opcode == OP_STL)
      space.set_type(local_space);
    else
      space.set_type(global_space);
    break;
  case OP_ATOMG:
  case OP_RED:
  case OP_ATOM:
    assert(data_size > 0);
    memory_op = memory_load;
    op = LOAD_OP;
    space.set_type(global_space);
    m_isatomic = true;
    cache_op = CACHE_GLOBAL; // all the atomics should be done at L2
    break;
  case OP_LDS:
    assert(data_size > 0);
    memory_op = memory_load;
    space.set_type(shared_space);
    break;
  case OP_STS:
    assert(data_size > 0);
    memory_op = memory_store;
    space.set_type(shared_space);
    break;
  case OP_ATOMS:
    assert(data_size > 0);
    m_isatomic = true;
    memory_op = memory_load;
    space.set_type(shared_space);
    break;
  case OP_LDSM:
    assert(data_size > 0);
    space.set_type(shared_space);
    break;
  case OP_ST:
  case OP_LD:
    assert(data_size > 0);
    if (m_opcode == OP_LD)
      memory_op = memory_load;
    else
      memory_op = memory_store;
    // resolve generic loads
    if (kernel_trace_info->shmem_base_addr == 0 ||
        kernel_trace_info->local_base_addr == 0) {
      // shmem and local addresses are not set
      // assume all the mem reqs are shared by default
      space.set_type(shared_space);
    } else {
      // check the first active address
      for (unsigned i = 0; i < warp_size(); ++i)
        if (active_mask.test(i)) {
          if (trace.memadd_info->addrs[i] >=
                  kernel_trace_info->shmem_base_addr &&
              trace.memadd_info->addrs[i] < kernel_trace_info->local_base_addr)
            space.set_type(shared_space);
          else if (trace.memadd_info->addrs[i] >=
                       kernel_trace_info->local_base_addr &&
                   trace.memadd_info->addrs[i] <
                       kernel_trace_info->local_base_addr +
                           LOCAL_MEM_SIZE_MAX) {
            space.set_type(local_space);
            cache_op = CACHE_ALL;
          } else {
            space.set_type(global_space);
            cache_op = CACHE_ALL;
          }
          break;
        }
    }

    break;
  case OP_BAR:
    // TO DO: fill this correctly
    bar_id = 0;
    bar_count = (unsigned)-1;
    bar_type = SYNC;
    // TO DO
    // if bar_type = RED;
    // set bar_type
    // barrier_type bar_type;
    // reduction_type red_type;
    break;
  case OP_HADD2:
  case OP_HADD2_32I:
  case OP_HFMA2:
  case OP_HFMA2_32I:
  case OP_HMUL2_32I:
  case OP_HSET2:
  case OP_HSETP2:
    initiation_interval =
        initiation_interval / 2; // FP16 has 2X throughput than FP32
    break;
  default:
    break;
  }

  return true;
}

void move_warp(trace_warp_inst_t *&dst, trace_warp_inst_t *&src) {
  assert(dst->empty());
  trace_warp_inst_t *temp = dst;
  dst = src;
  src = temp;
  src->clear();
}

void move_warp(warp_inst_t *&dst, warp_inst_t *&src) {
  assert(dst->empty());
  warp_inst_t *temp = dst;
  dst = src;
  src = temp;
  src->clear();
}
