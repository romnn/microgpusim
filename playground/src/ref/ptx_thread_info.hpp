#pragma once

#include <list>
#include <stack>
#include <unordered_map>

#include "core.hpp"
#include "gpgpu.hpp"
#include "kernel_info.hpp"
#include "ptx_warp_info.hpp"
#include "ptx_cta_info.hpp"
#include "ptx_instruction.hpp"
#include "ptx_reg.hpp"
#include "symbol.hpp"
#include "stack_entry.hpp"

class ptx_thread_info {
public:
  ~ptx_thread_info();
  ptx_thread_info(kernel_info_t &kernel);

  void init(gpgpu_t *gpu, core_t *core, unsigned sid, unsigned cta_id,
            unsigned wid, unsigned tid, bool fsim) {
    m_gpu = gpu;
    m_core = core;
    m_hw_sid = sid;
    m_hw_ctaid = cta_id;
    m_hw_wid = wid;
    m_hw_tid = tid;
    m_functionalSimulationMode = fsim;
  }

  void ptx_fetch_inst(inst_t &inst) const;
  void ptx_exec_inst(warp_inst_t &inst, unsigned lane_id);

  const ptx_version &get_ptx_version() const;
  void set_reg(const symbol *reg, const ptx_reg_t &value);
  void print_reg_thread(char *fname);
  void resume_reg_thread(char *fname, symbol_table *symtab);
  ptx_reg_t get_reg(const symbol *reg);
  ptx_reg_t get_operand_value(const operand_info &op, operand_info dstInfo,
                              unsigned opType, ptx_thread_info *thread,
                              int derefFlag);
  void set_operand_value(const operand_info &dst, const ptx_reg_t &data,
                         unsigned type, ptx_thread_info *thread,
                         const ptx_instruction *pI);
  void set_operand_value(const operand_info &dst, const ptx_reg_t &data,
                         unsigned type, ptx_thread_info *thread,
                         const ptx_instruction *pI, int overflow, int carry);
  void get_vector_operand_values(const operand_info &op, ptx_reg_t *ptx_regs,
                                 unsigned num_elements);
  void set_vector_operand_values(const operand_info &dst,
                                 const ptx_reg_t &data1, const ptx_reg_t &data2,
                                 const ptx_reg_t &data3,
                                 const ptx_reg_t &data4);
  void set_wmma_vector_operand_values(
      const operand_info &dst, const ptx_reg_t &data1, const ptx_reg_t &data2,
      const ptx_reg_t &data3, const ptx_reg_t &data4, const ptx_reg_t &data5,
      const ptx_reg_t &data6, const ptx_reg_t &data7, const ptx_reg_t &data8);

  function_info *func_info() { return m_func_info; }
  void print_insn(unsigned pc, FILE *fp) const;
  void set_info(function_info *func);
  unsigned get_uid() const { return m_uid; }

  dim3 get_ctaid() const { return m_ctaid; }
  dim3 get_tid() const { return m_tid; }
  dim3 get_ntid() const { return m_ntid; }
  class gpgpu_sim *get_gpu() { return (gpgpu_sim *)m_gpu; }
  unsigned get_hw_tid() const { return m_hw_tid; }
  unsigned get_hw_ctaid() const { return m_hw_ctaid; }
  unsigned get_hw_wid() const { return m_hw_wid; }
  unsigned get_hw_sid() const { return m_hw_sid; }
  core_t *get_core() { return m_core; }

  unsigned get_icount() const { return m_icount; }
  void set_valid() { m_valid = true; }
  addr_t last_eaddr() const { return m_last_effective_address; }
  memory_space_t last_space() const { return m_last_memory_space; }
  dram_callback_t last_callback() const { return m_last_dram_callback; }
  unsigned long long get_cta_uid() { return m_cta_info->get_sm_idx(); }

  void set_single_thread_single_block() {
    m_ntid.x = 1;
    m_ntid.y = 1;
    m_ntid.z = 1;
    m_ctaid.x = 0;
    m_ctaid.y = 0;
    m_ctaid.z = 0;
    m_tid.x = 0;
    m_tid.y = 0;
    m_tid.z = 0;
    m_nctaid.x = 1;
    m_nctaid.y = 1;
    m_nctaid.z = 1;
    m_gridid = 0;
    m_valid = true;
  }
  void set_tid(dim3 tid) { m_tid = tid; }
  void cpy_tid_to_reg(dim3 tid);
  void set_ctaid(dim3 ctaid) { m_ctaid = ctaid; }
  void set_ntid(dim3 tid) { m_ntid = tid; }
  void set_nctaid(dim3 cta_size) { m_nctaid = cta_size; }

  unsigned get_builtin(int builtin_id, unsigned dim_mod);

  void set_done();
  bool is_done() { return m_thread_done; }
  unsigned donecycle() const { return m_cycle_done; }

  unsigned next_instr() {
    m_icount++;
    m_branch_taken = false;
    return m_PC;
  }
  bool branch_taken() const { return m_branch_taken; }
  unsigned get_pc() const { return m_PC; }
  void set_npc(unsigned npc) { m_NPC = npc; }
  void set_npc(const function_info *f);
  void callstack_push(unsigned npc, unsigned rpc, const symbol *return_var_src,
                      const symbol *return_var_dst, unsigned call_uid);
  bool callstack_pop();
  void callstack_push_plus(unsigned npc, unsigned rpc,
                           const symbol *return_var_src,
                           const symbol *return_var_dst, unsigned call_uid);
  bool callstack_pop_plus();
  void dump_callstack() const;
  std::string get_location() const;
  const ptx_instruction *get_inst() const;
  const ptx_instruction *get_inst(addr_t pc) const;
  bool rpc_updated() const { return m_RPC_updated; }
  bool last_was_call() const { return m_last_was_call; }
  unsigned get_rpc() const { return m_RPC; }
  void clearRPC() {
    m_RPC = -1;
    m_RPC_updated = false;
    m_last_was_call = false;
  }
  unsigned get_return_PC() { return m_callstack.back().m_PC; }
  void update_pc() { m_PC = m_NPC; }
  void dump_regs(FILE *fp);
  void dump_modifiedregs(FILE *fp);
  void clear_modifiedregs() {
    m_debug_trace_regs_modified.back().clear();
    m_debug_trace_regs_read.back().clear();
  }
  function_info *get_finfo() { return m_func_info; }
  const function_info *get_finfo() const { return m_func_info; }
  void push_breakaddr(const operand_info &breakaddr);
  const operand_info &pop_breakaddr();
  void enable_debug_trace() { m_enable_debug_trace = true; }
  unsigned get_local_mem_stack_pointer() const {
    return m_local_mem_stack_pointer;
  }

  memory_space *get_global_memory() { return m_gpu->get_global_memory(); }
  memory_space *get_tex_memory() { return m_gpu->get_tex_memory(); }
  memory_space *get_surf_memory() { return m_gpu->get_surf_memory(); }
  memory_space *get_param_memory() { return m_kernel.get_param_memory(); }
  const gpgpu_functional_sim_config &get_config() const {
    return m_gpu->get_config();
  }
  bool isInFunctionalSimulationMode() { return m_functionalSimulationMode; }
  void exitCore() {
    // m_core is not used in case of functional simulation mode
    if (!m_functionalSimulationMode)
      m_core->warp_exit(m_hw_wid);
  }

  void registerExit() { m_cta_info->register_thread_exit(this); }
  unsigned get_reduction_value(unsigned ctaid, unsigned barid) {
    return m_core->get_reduction_value(ctaid, barid);
  }
  void and_reduction(unsigned ctaid, unsigned barid, bool value) {
    m_core->and_reduction(ctaid, barid, value);
  }
  void or_reduction(unsigned ctaid, unsigned barid, bool value) {
    m_core->or_reduction(ctaid, barid, value);
  }
  void popc_reduction(unsigned ctaid, unsigned barid, bool value) {
    m_core->popc_reduction(ctaid, barid, value);
  }

  // Jin: get corresponding kernel grid for CDP purpose
  kernel_info_t &get_kernel() { return m_kernel; }

public:
  addr_t m_last_effective_address;
  bool m_branch_taken;
  memory_space_t m_last_memory_space;
  dram_callback_t m_last_dram_callback;
  memory_space *m_shared_mem;
  memory_space *m_sstarr_mem;
  memory_space *m_local_mem;
  ptx_warp_info *m_warp_info;
  ptx_cta_info *m_cta_info;
  ptx_reg_t m_last_set_operand_value;

private:
  bool m_functionalSimulationMode;
  unsigned m_uid;
  kernel_info_t &m_kernel;
  core_t *m_core;
  gpgpu_t *m_gpu;
  bool m_valid;
  dim3 m_ntid;
  dim3 m_tid;
  dim3 m_nctaid;
  dim3 m_ctaid;
  unsigned m_gridid;
  bool m_thread_done;
  unsigned m_hw_sid;
  unsigned m_hw_tid;
  unsigned m_hw_wid;
  unsigned m_hw_ctaid;

  unsigned m_icount;
  unsigned m_PC;
  unsigned m_NPC;
  unsigned m_RPC;
  bool m_RPC_updated;
  bool m_last_was_call;
  unsigned m_cycle_done;

  int m_barrier_num;
  bool m_at_barrier;

  symbol_table *m_symbol_table;
  function_info *m_func_info;

  std::list<stack_entry> m_callstack;
  unsigned m_local_mem_stack_pointer;

  // typedef tr1_hash_map<const symbol *, ptx_reg_t> reg_map_t;
  typedef std::unordered_map<const symbol *, ptx_reg_t> reg_map_t;
  std::list<reg_map_t> m_regs;
  std::list<reg_map_t> m_debug_trace_regs_modified;
  std::list<reg_map_t> m_debug_trace_regs_read;
  bool m_enable_debug_trace;

  std::stack<class operand_info, std::vector<operand_info>> m_breakaddrs;
};
