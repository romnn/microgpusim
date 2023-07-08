#pragma once

#include <assert.h>
#include <list>
#include <vector>

#include "core_config.hpp"
#include "dram_callback.hpp"
#include "instr.hpp"
#include "mem_access.hpp"

class warp_inst_t : public inst_t {
 public:
  // constructors
  warp_inst_t() {
    m_opcode = 0;
    m_uid = 0;
    m_empty = true;
    m_config = NULL;
  }
  warp_inst_t(const core_config *config) {
    m_opcode = 0;
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
  active_mask_t get_warp_active_mask_copy() const { return m_warp_active_mask; }

  void completed(unsigned long long cycle)
      const;  // stat collection: called when the instruction is completed

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
        printf("MEM_TXN_GEN:%s:%lx, Size:%d \n",
               get_mem_access_type_str(it->get_type()), it->get_addr(),
               it->get_size());
      }
    }
  }
  struct transaction_info {
    std::bitset<4> chunks;  // bitmask: 32-byte chunks accessed
    mem_access_byte_mask_t bytes;
    active_mask_t active;  // threads in this transaction

    bool test_bytes(unsigned start_bit, unsigned end_bit) {
      for (unsigned i = start_bit; i <= end_bit; i++)
        if (bytes.test(i)) return true;
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

  void add_callback(unsigned lane_id,
                    void (*function)(const class inst_t *,
                                     class ptx_thread_info *),
                    const inst_t *inst, class ptx_thread_info *thread,
                    bool atomic) {
    if (!m_per_scalar_thread_valid) {
      m_per_scalar_thread.resize(m_config->warp_size);
      m_per_scalar_thread_valid = true;
      if (atomic) m_isatomic = true;
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
    fprintf(fp, " [inst @ pc=0x%04lx] ", pc);
    for (int i = (int)m_config->warp_size - 1; i >= 0; i--)
      fprintf(fp, "%c", ((m_warp_active_mask[i]) ? '1' : '0'));
  }
  bool active(unsigned thread) const { return m_warp_active_mask.test(thread); }
  unsigned active_count() const { return m_warp_active_mask.count(); }
  unsigned issued_count() const {
    assert(m_empty == false);
    return m_warp_issued_mask.count();
  }  // for instruction counting
  bool empty() const { return m_empty; }
  unsigned warp_id() const {
    assert(!m_empty);
    return m_warp_id;
  }
  unsigned warp_id_func() const  // to be used in functional simulations only
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
    if (cycles > 0) cycles--;
    return cycles > 0;
  }

  bool has_dispatch_delay() { return cycles > 0; }

  void print(std::ostream &os) const;
  std::string display() const;

  friend std::ostream &operator<<(std::ostream &os, const warp_inst_t &inst);
  friend std::ostream &operator<<(std::ostream &os, const warp_inst_t *inst);
  // void print(FILE *fout) const;

  unsigned get_uid() const { return m_uid; }
  unsigned get_schd_id() const { return m_scheduler_id; }

  unsigned opcode() const { return m_opcode; }
  const char *opcode_str() const;

 protected:
  unsigned m_opcode;
  unsigned m_uid;
  bool m_empty;
  bool m_cache_hit;
  unsigned long long issue_cycle;
  unsigned cycles;  // used for implementing initiation interval delay
  bool m_isatomic;
  bool should_do_atomic;
  bool m_is_printf;
  unsigned m_warp_id;
  unsigned m_dynamic_warp_id;
  const core_config *m_config;
  active_mask_t m_warp_active_mask;  // dynamic active mask for timing model
                                     // (after predication)
  active_mask_t
      m_warp_issued_mask;  // active mask at issue (prior to predication test)
                           // -- for instruction counting

  struct per_thread_info {
    per_thread_info() {
      for (unsigned i = 0; i < MAX_ACCESSES_PER_INSN_PER_THREAD; i++)
        memreqaddr[i] = 0;
    }
    dram_callback_t callback;
    new_addr_type
        memreqaddr[MAX_ACCESSES_PER_INSN_PER_THREAD];  // effective address,
                                                       // upto 8 different
                                                       // requests (to support
                                                       // 32B access in 8 chunks
                                                       // of 4B each)
  };
  bool m_per_scalar_thread_valid;
  std::vector<per_thread_info> m_per_scalar_thread;
  bool m_mem_accesses_created;
  std::list<mem_access_t> m_accessq;

  unsigned m_scheduler_id;  // the scheduler that issues this inst

  // Jin: cdp support
 public:
  int m_is_cdp;
};
