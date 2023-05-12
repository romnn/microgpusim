#pragma once

#include <list>
#include <set>
#include <string>

// #include "gpgpu_context.hpp"
#include "ptx_instruction.hpp"
#include "ptx_version.hpp"
#include "symbol_table.hpp"
#include "gpgpu.hpp"
#include "gpu_recon.hpp"
#include "gpgpu_ptx_sim_info.hpp"
#include "param.hpp"
#include "param_info.hpp"

class gpgpu_context;

class function_info {
public:
  function_info(int entry_point, gpgpu_context *ctx);
  const ptx_version &get_ptx_version() const {
    return m_symtab->get_ptx_version();
  }
  unsigned get_sm_target() const { return m_symtab->get_sm_target(); }
  bool is_extern() const { return m_extern; }
  void set_name(const char *name) { m_name = name; }
  void set_symtab(symbol_table *symtab) { m_symtab = symtab; }
  std::string get_name() const { return m_name; }
  unsigned print_insn(unsigned pc, FILE *fp) const;
  std::string get_insn_str(unsigned pc) const;
  void add_inst(const std::list<ptx_instruction *> &instructions) {
    m_instructions = instructions;
  }
  std::list<ptx_instruction *>::iterator
  find_next_real_instruction(std::list<ptx_instruction *>::iterator i);
  void create_basic_blocks();

  void print_basic_blocks();

  void print_basic_block_links();
  void print_basic_block_dot();

  operand_info *find_break_target(
      ptx_instruction *p_break_insn); // find the target of a break instruction
  void connect_basic_blocks(); // iterate across m_basic_blocks of function,
                               // connecting basic blocks together
  bool
  connect_break_targets(); // connecting break instructions with proper targets

  // iterate across m_basic_blocks of function,
  // finding dominator blocks, using algorithm of
  // Muchnick's Adv. Compiler Design & Implemmntation Fig 7.14
  void find_dominators();
  void print_dominators();
  void find_idominators();
  void print_idominators();

  // iterate across m_basic_blocks of function,
  // finding postdominator blocks, using algorithm of
  // Muchnick's Adv. Compiler Design & Implemmntation Fig 7.14
  void find_postdominators();
  void print_postdominators();

  // iterate across m_basic_blocks of function,
  // finding immediate postdominator blocks, using algorithm of
  // Muchnick's Adv. Compiler Design & Implemmntation Fig 7.15
  void find_ipostdominators();
  void print_ipostdominators();
  void do_pdom(); // function to call pdom analysis

  unsigned get_num_reconvergence_pairs();

  void get_reconvergence_pairs(gpgpu_recon_t *recon_points);

  unsigned get_function_size() { return m_instructions.size(); }

  void ptx_assemble();

  unsigned ptx_get_inst_op(ptx_thread_info *thread);
  void add_param(const char *name, struct param_t value) {
    m_kernel_params[name] = value;
  }
  void add_param_name_type_size(unsigned index, std::string name, int type,
                                size_t size, bool ptr, memory_space_t space);
  void add_param_data(unsigned argn, struct gpgpu_ptx_sim_arg *args);
  void add_return_var(const symbol *rv) { m_return_var_sym = rv; }
  void add_arg(const symbol *arg) {
    assert(arg != NULL);
    m_args.push_back(arg);
  }
  void remove_args() { m_args.clear(); }
  unsigned num_args() const { return m_args.size(); }
  unsigned get_args_aligned_size();

  const symbol *get_arg(unsigned n) const {
    assert(n < m_args.size());
    return m_args[n];
  }
  bool has_return() const { return m_return_var_sym != NULL; }
  const symbol *get_return_var() const { return m_return_var_sym; }
  const ptx_instruction *get_instruction(unsigned PC) const {
    unsigned index = PC - m_start_PC;
    if (index < m_instr_mem_size)
      return m_instr_mem[index];
    return NULL;
  }
  addr_t get_start_PC() const { return m_start_PC; }

  void finalize(memory_space *param_mem);
  void param_to_shared(memory_space *shared_mem, symbol_table *symtab);
  void list_param(FILE *fout) const;
  void ptx_jit_config(std::map<unsigned long long, size_t> mallocPtr_Size,
                      memory_space *param_mem, gpgpu_t *gpu, dim3 gridDim,
                      dim3 blockDim);

  virtual const struct gpgpu_ptx_sim_info *get_kernel_info() const {
    assert(m_kernel_info.maxthreads == maxnt_id);
    return &m_kernel_info;
  }

  virtual const void set_kernel_info(const struct gpgpu_ptx_sim_info &info) {
    m_kernel_info = info;
    m_kernel_info.ptx_version = 10 * get_ptx_version().ver();
    m_kernel_info.sm_target = get_ptx_version().target();
    // THIS DEPENDS ON ptxas being called after the PTX is parsed.
    m_kernel_info.maxthreads = maxnt_id;
  }
  symbol_table *get_symtab() { return m_symtab; }

  unsigned local_mem_framesize() const { return m_local_mem_framesize; }
  void set_framesize(unsigned sz) { m_local_mem_framesize = sz; }
  bool is_entry_point() const { return m_entry_point; }
  bool is_pdom_set() const { return pdom_done; } // return pdom flag
  void set_pdom() { pdom_done = true; }          // set pdom flag

  void add_config_param(size_t size, unsigned alignment) {
    unsigned offset = 0;
    if (m_param_configs.size() > 0) {
      unsigned offset_nom =
          m_param_configs.back().first + m_param_configs.back().second;
      // ensure offset matches alignment requirements
      offset = offset_nom % alignment ? (offset_nom / alignment + 1) * alignment
                                      : offset_nom;
    }
    m_param_configs.push_back(std::pair<size_t, unsigned>(size, offset));
  }

  std::pair<size_t, unsigned> get_param_config(unsigned param_num) const {
    return m_param_configs[param_num];
  }

  void set_maxnt_id(unsigned maxthreads) { maxnt_id = maxthreads; }
  unsigned get_maxnt_id() { return maxnt_id; }
  // backward pointer
  class gpgpu_context *gpgpu_ctx;

protected:
  // Registers/shmem/etc. used (from ptxas -v), loaded from ___.ptxinfo along
  // with ___.ptx
  struct gpgpu_ptx_sim_info m_kernel_info;

private:
  unsigned maxnt_id;
  unsigned m_uid;
  unsigned m_local_mem_framesize;
  bool m_entry_point;
  bool m_extern;
  bool m_assembled;
  bool pdom_done; // flag to check whether pdom is completed or not
  std::string m_name;
  ptx_instruction **m_instr_mem;
  unsigned m_start_PC;
  unsigned m_instr_mem_size;
  std::map<std::string, param_t> m_kernel_params;
  std::map<unsigned, param_info> m_ptx_kernel_param_info;
  std::vector<std::pair<size_t, unsigned>> m_param_configs;
  const symbol *m_return_var_sym;
  std::vector<const symbol *> m_args;
  std::list<ptx_instruction *> m_instructions;
  std::vector<basic_block_t *> m_basic_blocks;
  std::list<std::pair<unsigned, unsigned>> m_back_edges;
  std::map<std::string, unsigned> labels;
  unsigned num_reconvergence_pairs;

  // Registers/shmem/etc. used (from ptxas -v), loaded from ___.ptxinfo along
  // with ___.ptx
  // with ___.ptx

  symbol_table *m_symtab;

  // parameter size for device kernels
  int m_args_aligned_size;

  addr_t m_n; // offset in m_instr_mem (used in do_pdom)
};
