#pragma once

#include <cstring>

#include "cache_operator_type.hpp"
#include "hal.hpp"
#include "memory_space.hpp"

class inst_t {
 public:
  inst_t() {
    m_decoded = false;
    pc = (address_type)-1;
    reconvergence_pc = (address_type)-1;
    op = NO_OP;
    bar_type = NOT_BAR;
    red_type = NOT_RED;
    bar_id = (unsigned)-1;
    bar_count = (unsigned)-1;
    oprnd_type = UN_OP;
    sp_op = OTHER_OP;
    op_pipe = UNKOWN_OP;
    mem_op = NOT_TEX;
    const_cache_operand = 0;
    num_operands = 0;
    num_regs = 0;
    std::memset(out, 0, sizeof(unsigned));
    std::memset(in, 0, sizeof(unsigned));
    is_vectorin = 0;
    is_vectorout = 0;
    space = memory_space_t();
    cache_op = CACHE_UNDEFINED;
    latency = 1;
    initiation_interval = 1;
    for (unsigned i = 0; i < MAX_REG_OPERANDS; i++) {
      arch_reg.src[i] = -1;
      arch_reg.dst[i] = -1;
    }
    isize = 0;

    // ROMAN: initialize variables
    pred = 0;
    ar1 = 0;
    ar2 = 0;
  }
  bool valid() const { return m_decoded; }
  virtual void print_insn(FILE *fp) const {
    fprintf(fp, " [inst @ pc=0x%04lx] ", pc);
  }
  bool is_load() const {
    return (op == LOAD_OP || op == TENSOR_CORE_LOAD_OP ||
            memory_op == memory_load);
  }
  bool is_store() const {
    return (op == STORE_OP || op == TENSOR_CORE_STORE_OP ||
            memory_op == memory_store);
  }

  bool is_fp() const { return ((sp_op == FP__OP)); }  // VIJAY
  bool is_fpdiv() const { return ((sp_op == FP_DIV_OP)); }
  bool is_fpmul() const { return ((sp_op == FP_MUL_OP)); }
  bool is_dp() const { return ((sp_op == DP___OP)); }
  bool is_dpdiv() const { return ((sp_op == DP_DIV_OP)); }
  bool is_dpmul() const { return ((sp_op == DP_MUL_OP)); }
  bool is_imul() const { return ((sp_op == INT_MUL_OP)); }
  bool is_imul24() const { return ((sp_op == INT_MUL24_OP)); }
  bool is_imul32() const { return ((sp_op == INT_MUL32_OP)); }
  bool is_idiv() const { return ((sp_op == INT_DIV_OP)); }
  bool is_sfu() const {
    return ((sp_op == FP_SQRT_OP) || (sp_op == FP_LG_OP) ||
            (sp_op == FP_SIN_OP) || (sp_op == FP_EXP_OP) ||
            (sp_op == TENSOR__OP));
  }
  bool is_alu() const { return (sp_op == INT__OP); }

  unsigned get_num_operands() const { return num_operands; }
  unsigned get_num_regs() const { return num_regs; }
  void set_num_regs(unsigned num) { num_regs = num; }
  void set_num_operands(unsigned num) { num_operands = num; }
  void set_bar_id(unsigned id) { bar_id = id; }
  void set_bar_count(unsigned count) { bar_count = count; }

  address_type pc;  // program counter address of instruction
  unsigned isize;   // size of instruction in bytes
  op_type op;       // opcode (uarch visible)

  barrier_type bar_type;
  reduction_type red_type;
  unsigned bar_id;
  unsigned bar_count;

  types_of_operands oprnd_type;  // code (uarch visible) identify if the
                                 // operation is an interger or a floating point
  special_ops
      sp_op;  // code (uarch visible) identify if int_alu, fp_alu, int_mul ....
  operation_pipeline op_pipe;  // code (uarch visible) identify the pipeline of
                               // the operation (SP, SFU or MEM)
  mem_operation mem_op;        // code (uarch visible) identify memory type
  bool const_cache_operand;    // has a load from constant memory as an operand
  _memory_op_t memory_op;      // memory_op used by ptxplus
  unsigned num_operands;
  unsigned num_regs;  // count vector operand as one register operand

  address_type reconvergence_pc;  // -1 => not a branch, -2 => use function
                                  // return address

  unsigned out[8];
  unsigned outcount;
  unsigned in[24];
  unsigned incount;
  unsigned char is_vectorin;
  unsigned char is_vectorout;
  int pred;  // predicate register number
  int ar1, ar2;
  // register number for bank conflict evaluation
  struct {
    int dst[MAX_REG_OPERANDS];
    int src[MAX_REG_OPERANDS];
  } arch_reg;
  // int arch_reg[MAX_REG_OPERANDS]; // register number for bank conflict
  // evaluation
  unsigned latency;  // operation latency
  unsigned initiation_interval;

  unsigned data_size;  // what is the size of the word being operated on?
  memory_space_t space;
  cache_operator_type cache_op;

 protected:
  bool m_decoded;
  virtual void pre_decode() {}
};
