#pragma once

#include <bitset>
#include <cstdio>
#include <cstring>
#include <list>

#include "address.hpp"
#include "cache_operator_type.hpp"
#include "core_config.hpp"
#include "dram_callback.hpp"
#include "hal.hpp"
#include "mem_access.hpp"
#include "memory_space.hpp"

enum uarch_op_t {
  NO_OP = -1,
  ALU_OP = 1,
  SFU_OP,
  TENSOR_CORE_OP,
  DP_OP,
  SP_OP,
  INTP_OP,
  ALU_SFU_OP,
  LOAD_OP,
  TENSOR_CORE_LOAD_OP,
  TENSOR_CORE_STORE_OP,
  STORE_OP,
  BRANCH_OP,
  BARRIER_OP,
  MEMORY_BARRIER_OP,
  CALL_OPS,
  RET_OPS,
  EXIT_OPS,
  SPECIALIZED_UNIT_1_OP = SPEC_UNIT_START_ID,
  SPECIALIZED_UNIT_2_OP,
  SPECIALIZED_UNIT_3_OP,
  SPECIALIZED_UNIT_4_OP,
  SPECIALIZED_UNIT_5_OP,
  SPECIALIZED_UNIT_6_OP,
  SPECIALIZED_UNIT_7_OP,
  SPECIALIZED_UNIT_8_OP
};
typedef enum uarch_op_t op_type;

enum uarch_bar_t { NOT_BAR = -1, SYNC = 1, ARRIVE, RED };
typedef enum uarch_bar_t barrier_type;

enum uarch_red_t { NOT_RED = -1, POPC_RED = 1, AND_RED, OR_RED };
typedef enum uarch_red_t reduction_type;

enum uarch_operand_type_t { UN_OP = -1, INT_OP, FP_OP };
typedef enum uarch_operand_type_t types_of_operands;

enum special_operations_t {
  OTHER_OP,
  INT__OP,
  INT_MUL24_OP,
  INT_MUL32_OP,
  INT_MUL_OP,
  INT_DIV_OP,
  FP_MUL_OP,
  FP_DIV_OP,
  FP__OP,
  FP_SQRT_OP,
  FP_LG_OP,
  FP_SIN_OP,
  FP_EXP_OP,
  DP_MUL_OP,
  DP_DIV_OP,
  DP___OP,
  TENSOR__OP,
  TEX__OP
};

typedef enum special_operations_t special_ops;

enum operation_pipeline_t {
  UNKOWN_OP,
  SP__OP,
  DP__OP,
  INTP__OP,
  SFU__OP,
  TENSOR_CORE__OP,
  MEM__OP,
  SPECIALIZED__OP,
};
typedef enum operation_pipeline_t operation_pipeline;
enum mem_operation_t { NOT_TEX, TEX };
typedef enum mem_operation_t mem_operation;

enum _memory_op_t { no_memory_op = 0, memory_load, memory_store };

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
    memset(out, 0, sizeof(unsigned));
    memset(in, 0, sizeof(unsigned));
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
  }
  bool valid() const { return m_decoded; }
  virtual void print_insn(FILE *fp) const {
    fprintf(fp, " [inst @ pc=0x%04llx] ", pc);
  }
  bool is_load() const {
    return (op == LOAD_OP || op == TENSOR_CORE_LOAD_OP ||
            memory_op == memory_load);
  }
  bool is_store() const {
    return (op == STORE_OP || op == TENSOR_CORE_STORE_OP ||
            memory_op == memory_store);
  }

  bool is_fp() const { return ((sp_op == FP__OP)); } // VIJAY
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

  address_type pc; // program counter address of instruction
  unsigned isize;  // size of instruction in bytes
  op_type op;      // opcode (uarch visible)

  barrier_type bar_type;
  reduction_type red_type;
  unsigned bar_id;
  unsigned bar_count;

  types_of_operands oprnd_type; // code (uarch visible) identify if the
                                // operation is an interger or a floating point
  special_ops
      sp_op; // code (uarch visible) identify if int_alu, fp_alu, int_mul ....
  operation_pipeline op_pipe; // code (uarch visible) identify the pipeline of
                              // the operation (SP, SFU or MEM)
  mem_operation mem_op;       // code (uarch visible) identify memory type
  bool const_cache_operand;   // has a load from constant memory as an operand
  _memory_op_t memory_op;     // memory_op used by ptxplus
  unsigned num_operands;
  unsigned num_regs; // count vector operand as one register operand

  address_type reconvergence_pc; // -1 => not a branch, -2 => use function
                                 // return address

  unsigned out[8];
  unsigned outcount;
  unsigned in[24];
  unsigned incount;
  unsigned char is_vectorin;
  unsigned char is_vectorout;
  int pred; // predicate register number
  int ar1, ar2;
  // register number for bank conflict evaluation
  struct {
    int dst[MAX_REG_OPERANDS];
    int src[MAX_REG_OPERANDS];
  } arch_reg;
  // int arch_reg[MAX_REG_OPERANDS]; // register number for bank conflict
  // evaluation
  unsigned latency; // operation latency
  unsigned initiation_interval;

  unsigned data_size; // what is the size of the word being operated on?
  memory_space_t space;
  cache_operator_type cache_op;

protected:
  bool m_decoded;
  virtual void pre_decode() {}
};

class warp_inst_t : public inst_t {
public:
  // constructors
  warp_inst_t() {
    m_uid = 0;
    m_empty = true;
    m_config = NULL;
  }
  warp_inst_t(const core_config *config) {
    m_uid = 0;
    assert(config->warp_size <= MAX_WARP_SIZE);
    m_config = config;
    m_empty = true;
    m_isatomic = false;
    m_per_scalar_thread_valid = false;
    m_mem_accesses_created = false;
    m_cache_hit = false;
    m_is_printf = false;
    m_is_cdp = 0;
    should_do_atomic = true;
  }
  virtual ~warp_inst_t() {}

  // modifiers
  void broadcast_barrier_reduction(const active_mask_t &access_mask);
  void do_atomic(bool forceDo = false);
  void do_atomic(const active_mask_t &access_mask, bool forceDo = false);
  void clear() { m_empty = true; }

  void issue(const active_mask_t &mask, unsigned warp_id,
             unsigned long long cycle, int dynamic_warp_id, int sch_id);

  const active_mask_t &get_active_mask() const { return m_warp_active_mask; }
  void completed(unsigned long long cycle)
      const; // stat collection: called when the instruction is completed

  void set_addr(unsigned n, new_addr_type addr) {
    if (!m_per_scalar_thread_valid) {
      m_per_scalar_thread.resize(m_config->warp_size);
      m_per_scalar_thread_valid = true;
    }
    m_per_scalar_thread[n].memreqaddr[0] = addr;
  }
  void set_addr(unsigned n, new_addr_type *addr, unsigned num_addrs) {
    if (!m_per_scalar_thread_valid) {
      m_per_scalar_thread.resize(m_config->warp_size);
      m_per_scalar_thread_valid = true;
    }
    assert(num_addrs <= MAX_ACCESSES_PER_INSN_PER_THREAD);
    for (unsigned i = 0; i < num_addrs; i++)
      m_per_scalar_thread[n].memreqaddr[i] = addr[i];
  }
  void print_m_accessq() {
    if (accessq_empty())
      return;
    else {
      printf("Printing mem access generated\n");
      std::list<mem_access_t>::iterator it;
      for (it = m_accessq.begin(); it != m_accessq.end(); ++it) {
        printf("MEM_TXN_GEN:%s:%llx, Size:%d \n",
               mem_access_type_str(it->get_type()), it->get_addr(),
               it->get_size());
      }
    }
  }
  struct transaction_info {
    std::bitset<4> chunks; // bitmask: 32-byte chunks accessed
    mem_access_byte_mask_t bytes;
    active_mask_t active; // threads in this transaction

    bool test_bytes(unsigned start_bit, unsigned end_bit) {
      for (unsigned i = start_bit; i <= end_bit; i++)
        if (bytes.test(i))
          return true;
      return false;
    }
  };

  void generate_mem_accesses();
  void memory_coalescing_arch(bool is_write, mem_access_type access_type);
  void memory_coalescing_arch_atomic(bool is_write,
                                     mem_access_type access_type);
  void memory_coalescing_arch_reduce_and_send(bool is_write,
                                              mem_access_type access_type,
                                              const transaction_info &info,
                                              new_addr_type addr,
                                              unsigned segment_size);

  void
  add_callback(unsigned lane_id,
               void (*function)(const class inst_t *, class ptx_thread_info *),
               const inst_t *inst, class ptx_thread_info *thread, bool atomic) {
    if (!m_per_scalar_thread_valid) {
      m_per_scalar_thread.resize(m_config->warp_size);
      m_per_scalar_thread_valid = true;
      if (atomic)
        m_isatomic = true;
    }
    m_per_scalar_thread[lane_id].callback.function = function;
    m_per_scalar_thread[lane_id].callback.instruction = inst;
    m_per_scalar_thread[lane_id].callback.thread = thread;
  }
  void set_active(const active_mask_t &active);

  void clear_active(const active_mask_t &inactive);
  void set_not_active(unsigned lane_id);

  // accessors
  virtual void print_insn(FILE *fp) const {
    fprintf(fp, " [inst @ pc=0x%04llx] ", pc);
    for (int i = (int)m_config->warp_size - 1; i >= 0; i--)
      fprintf(fp, "%c", ((m_warp_active_mask[i]) ? '1' : '0'));
  }
  bool active(unsigned thread) const { return m_warp_active_mask.test(thread); }
  unsigned active_count() const { return m_warp_active_mask.count(); }
  unsigned issued_count() const {
    assert(m_empty == false);
    return m_warp_issued_mask.count();
  } // for instruction counting
  bool empty() const { return m_empty; }
  unsigned warp_id() const {
    assert(!m_empty);
    return m_warp_id;
  }
  unsigned warp_id_func() const // to be used in functional simulations only
  {
    return m_warp_id;
  }
  unsigned dynamic_warp_id() const {
    assert(!m_empty);
    return m_dynamic_warp_id;
  }
  bool has_callback(unsigned n) const {
    return m_warp_active_mask[n] && m_per_scalar_thread_valid &&
           (m_per_scalar_thread[n].callback.function != NULL);
  }
  new_addr_type get_addr(unsigned n) const {
    assert(m_per_scalar_thread_valid);
    return m_per_scalar_thread[n].memreqaddr[0];
  }

  bool isatomic() const { return m_isatomic; }

  unsigned warp_size() const { return m_config->warp_size; }

  bool accessq_empty() const { return m_accessq.empty(); }
  unsigned accessq_count() const { return m_accessq.size(); }
  const mem_access_t &accessq_back() { return m_accessq.back(); }
  void accessq_pop_back() { m_accessq.pop_back(); }

  bool dispatch_delay() {
    if (cycles > 0)
      cycles--;
    return cycles > 0;
  }

  bool has_dispatch_delay() { return cycles > 0; }

  void print(FILE *fout) const;
  unsigned get_uid() const { return m_uid; }
  unsigned get_schd_id() const { return m_scheduler_id; }
  active_mask_t get_warp_active_mask() const { return m_warp_active_mask; }

protected:
  unsigned m_uid;
  bool m_empty;
  bool m_cache_hit;
  unsigned long long issue_cycle;
  unsigned cycles; // used for implementing initiation interval delay
  bool m_isatomic;
  bool should_do_atomic;
  bool m_is_printf;
  unsigned m_warp_id;
  unsigned m_dynamic_warp_id;
  const core_config *m_config;
  // dynamic active mask for timing model
  // (after predication)
  active_mask_t m_warp_active_mask;

  // active mask at issue (prior to predication test)
  // -- for instruction counting
  active_mask_t m_warp_issued_mask;

  struct per_thread_info {
    per_thread_info() {
      for (unsigned i = 0; i < MAX_ACCESSES_PER_INSN_PER_THREAD; i++)
        memreqaddr[i] = 0;
    }
    dram_callback_t callback;
    new_addr_type
        memreqaddr[MAX_ACCESSES_PER_INSN_PER_THREAD]; // effective address,
                                                      // upto 8 different
                                                      // requests (to support
                                                      // 32B access in 8 chunks
                                                      // of 4B each)
  };
  bool m_per_scalar_thread_valid;
  std::vector<per_thread_info> m_per_scalar_thread;
  bool m_mem_accesses_created;
  std::list<mem_access_t> m_accessq;

  unsigned m_scheduler_id; // the scheduler that issues this inst

  // Jin: cdp support
public:
  int m_is_cdp;
};
