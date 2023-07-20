#pragma once

#include <list>

#include "basic_block.hpp"
#include "opcode.hpp"
#include "operand_info.hpp"
#include "symbol.hpp"
#include "warp_instr.hpp"

// class symbol;
// class operand_info;
class gpgpu_context;

class ptx_instruction : public warp_inst_t {
 public:
  ptx_instruction(int opcode, const symbol *pred, int neg_pred, int pred_mod,
                  symbol *label, const std::list<operand_info> &operands,
                  const operand_info &return_var, const std::list<int> &options,
                  const std::list<int> &wmma_options,
                  const std::list<int> &scalar_type, memory_space_t space_spec,
                  const char *file, unsigned line, const char *source,
                  const core_config *config, gpgpu_context *ctx);

  void print_insn() const;
  virtual void print_insn(FILE *fp) const;
  std::string to_string() const;
  unsigned inst_size() const { return m_inst_size; }
  unsigned uid() const { return m_uid; }
  int get_opcode() const { return m_opcode; }
  const char *get_opcode_cstr() const {
    if (m_opcode != -1) {
      return g_opcode_string[m_opcode];
    } else {
      return "label";
    }
  }
  const char *source_file() const { return m_source_file.c_str(); }
  unsigned source_line() const { return m_source_line; }
  unsigned get_num_operands() const { return m_operands.size(); }
  bool has_pred() const { return m_pred != NULL; }
  operand_info get_pred() const;
  bool get_pred_neg() const { return m_neg_pred; }
  int get_pred_mod() const { return m_pred_mod; }
  const char *get_source() const { return m_source.c_str(); }

  const std::list<int> get_scalar_type() const { return m_scalar_type; }
  const std::list<int> get_options() const { return m_options; }

  typedef std::vector<operand_info>::const_iterator const_iterator;

  const_iterator op_iter_begin() const { return m_operands.begin(); }

  const_iterator op_iter_end() const { return m_operands.end(); }

  const operand_info &dst() const {
    assert(!m_operands.empty());
    return m_operands[0];
  }

  const operand_info &func_addr() const {
    assert(!m_operands.empty());
    if (!m_operands[0].is_return_var()) {
      return m_operands[0];
    } else {
      assert(m_operands.size() >= 2);
      return m_operands[1];
    }
  }

  operand_info &dst() {
    assert(!m_operands.empty());
    return m_operands[0];
  }

  const operand_info &src1() const {
    assert(m_operands.size() > 1);
    return m_operands[1];
  }

  const operand_info &src2() const {
    assert(m_operands.size() > 2);
    return m_operands[2];
  }

  const operand_info &src3() const {
    assert(m_operands.size() > 3);
    return m_operands[3];
  }
  const operand_info &src4() const {
    assert(m_operands.size() > 4);
    return m_operands[4];
  }
  const operand_info &src5() const {
    assert(m_operands.size() > 5);
    return m_operands[5];
  }
  const operand_info &src6() const {
    assert(m_operands.size() > 6);
    return m_operands[6];
  }
  const operand_info &src7() const {
    assert(m_operands.size() > 7);
    return m_operands[7];
  }
  const operand_info &src8() const {
    assert(m_operands.size() > 8);
    return m_operands[8];
  }

  const operand_info &operand_lookup(unsigned n) const {
    assert(n < m_operands.size());
    return m_operands[n];
  }
  bool has_return() const { return m_return_var.is_valid(); }

  memory_space_t get_space() const { return m_space_spec; }
  unsigned get_vector() const { return m_vector_spec; }
  unsigned get_atomic() const { return m_atomic_spec; }

  int get_wmma_type() const { return m_wmma_type; }
  int get_wmma_layout(int index) const {
    return m_wmma_layout[index];  // 0->Matrix D,1->Matrix C
  }
  int get_type() const {
    assert(!m_scalar_type.empty());
    return m_scalar_type.front();
  }

  int get_type2() const {
    assert(m_scalar_type.size() == 2);
    return m_scalar_type.back();
  }

  void assign_bb(
      basic_block_t *basic_block)  // assign instruction to a basic block
  {
    m_basic_block = basic_block;
  }
  basic_block_t *get_bb() { return m_basic_block; }
  void set_m_instr_mem_index(unsigned index) { m_instr_mem_index = index; }
  void set_PC(addr_t PC) { m_PC = PC; }
  addr_t get_PC() const { return m_PC; }

  unsigned get_m_instr_mem_index() { return m_instr_mem_index; }
  unsigned get_cmpop() const { return m_compare_op; }
  const symbol *get_label() const { return m_label; }
  bool is_label() const {
    if (m_label) {
      assert(m_opcode == -1);
      return true;
    }
    return false;
  }
  bool is_hi() const { return m_hi; }
  bool is_lo() const { return m_lo; }
  bool is_wide() const { return m_wide; }
  bool is_uni() const { return m_uni; }
  bool is_exit() const { return m_exit; }
  bool is_abs() const { return m_abs; }
  bool is_neg() const { return m_neg; }
  bool is_to() const { return m_to_option; }
  unsigned cache_option() const { return m_cache_option; }
  unsigned rounding_mode() const { return m_rounding_mode; }
  unsigned saturation_mode() const { return m_saturation_mode; }
  unsigned dimension() const { return m_geom_spec; }
  unsigned barrier_op() const { return m_barrier_op; }
  unsigned shfl_op() const { return m_shfl_op; }
  unsigned prmt_op() const { return m_prmt_op; }
  enum vote_mode_t { vote_any, vote_all, vote_uni, vote_ballot };
  enum vote_mode_t vote_mode() const { return m_vote_mode; }

  int membar_level() const { return m_membar_level; }

  bool has_memory_read() const {
    if (m_opcode == LD_OP || m_opcode == LDU_OP || m_opcode == TEX_OP ||
        m_opcode == MMA_LD_OP)
      return true;
    // Check PTXPlus operand type below
    // Source operands are memory operands
    ptx_instruction::const_iterator op = op_iter_begin();
    for (int n = 0; op != op_iter_end(); op++, n++) {  // process operands
      if (n > 0 && op->is_memory_operand2())           // source operands only
        return true;
    }
    return false;
  }
  bool has_memory_write() const {
    if (m_opcode == ST_OP || m_opcode == MMA_ST_OP) return true;
    // Check PTXPlus operand type below
    // Destination operand is a memory operand
    ptx_instruction::const_iterator op = op_iter_begin();
    for (int n = 0; (op != op_iter_end() && n < 1);
         op++, n++) {                          // process operands
      if (n == 0 && op->is_memory_operand2())  // source operands only
        return true;
    }
    return false;
  }

 private:
  void set_opcode_and_latency();
  void set_bar_type();
  void set_fp_or_int_archop();
  void set_mul_div_or_other_archop();

  basic_block_t *m_basic_block;
  unsigned m_uid;
  addr_t m_PC;
  std::string m_source_file;
  unsigned m_source_line;
  std::string m_source;

  const symbol *m_pred;
  bool m_neg_pred;
  int m_pred_mod;
  int m_opcode;
  const symbol *m_label;
  std::vector<operand_info> m_operands;
  operand_info m_return_var;

  std::list<int> m_options;
  std::list<int> m_wmma_options;
  bool m_wide;
  bool m_hi;
  bool m_lo;
  bool m_exit;
  bool m_abs;
  bool m_neg;
  bool m_uni;  // if branch instruction, this evaluates to true for uniform
               // branches (ie jumps)
  bool m_to_option;
  unsigned m_cache_option;
  int m_wmma_type;
  int m_wmma_layout[2];
  int m_wmma_configuration;
  unsigned m_rounding_mode;
  unsigned m_compare_op;
  unsigned m_saturation_mode;
  unsigned m_barrier_op;
  unsigned m_shfl_op;
  unsigned m_prmt_op;

  std::list<int> m_scalar_type;
  memory_space_t m_space_spec;
  int m_geom_spec;
  int m_vector_spec;
  int m_atomic_spec;
  enum vote_mode_t m_vote_mode;
  int m_membar_level;
  int m_instr_mem_index;  // index into m_instr_mem array
  unsigned m_inst_size;   // bytes

  virtual void pre_decode();
  friend class function_info;
  // backward pointer
  class gpgpu_context *gpgpu_ctx;
};
