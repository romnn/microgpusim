#pragma once

#include <memory>

#include "inst_trace.hpp"
#include "trace_kernel_info.hpp"
#include "trace_warp_inst.hpp"
#include "warp_instr.hpp"

class trace_shader_core_ctx;

class trace_shd_warp_t {
 public:
  trace_shd_warp_t(class trace_shader_core_ctx *shader, unsigned warp_size)
      : m_shader(shader), m_warp_size(warp_size) {
    m_stores_outstanding = 0;
    m_inst_in_pipeline = 0;
    reset();
    trace_pc = 0;
    m_kernel_info = NULL;

    // parsed_trace_pc = 0;
  }

  void reset() {
    assert(m_stores_outstanding == 0);
    assert(m_inst_in_pipeline == 0);
    m_imiss_pending = false;
    // ROMAN: use 0 as the default warp id for uninitialized
    // m_warp_id = 0;
    // m_dynamic_warp_id = 0;
    m_warp_id = (unsigned)-1;
    m_dynamic_warp_id = (unsigned)-1;
    n_completed = m_warp_size;
    m_n_atomic = 0;
    m_membar = false;
    m_done_exit = true;
    m_last_fetch = 0;
    m_next = 0;

    // Jin: cdp support
    m_cdp_latency = 0;
    m_cdp_dummy = false;
  }

  void init(address_type start_pc, unsigned cta_id, unsigned wid,
            const std::bitset<MAX_WARP_SIZE> &active,
            unsigned dynamic_warp_id) {
    m_cta_id = cta_id;
    m_warp_id = wid;
    m_dynamic_warp_id = dynamic_warp_id;
    // m_next_pc = start_pc;
    assert(n_completed >= active.count());
    assert(n_completed <= m_warp_size);
    n_completed -= active.count();  // active threads are not yet completed
    m_active_threads = active;
    m_done_exit = false;

    // Jin: cdp support
    m_cdp_latency = 0;
    m_cdp_dummy = false;
  }

  unsigned long instruction_count();

  std::vector<inst_trace_t> warp_traces;
  std::vector<const trace_warp_inst_t *> parsed_warp_traces_cache;
  // unsigned parsed_trace_pc;

  void print_trace_instructions(bool all,
                                std::shared_ptr<spdlog::logger> &logger);
  const warp_inst_t *get_next_trace_inst();
  const trace_warp_inst_t *get_current_trace_inst();
  const trace_warp_inst_t *get_cached_trace_instruction(unsigned trace_pc);
  void clear();
  bool trace_done() const;

  address_type get_start_trace_pc();
  virtual address_type get_pc();
  virtual trace_kernel_info_t *get_kernel_info() const { return m_kernel_info; }
  void set_kernel(trace_kernel_info_t *kernel_info) {
    m_kernel_info = kernel_info;
  }

  bool functional_done() const;
  bool waiting();  // not const due to membar
  bool hardware_done() const;

  bool done_exit() const { return m_done_exit; }
  void set_done_exit() { m_done_exit = true; }

  void set_next_pc(address_type pc) {
    // NOTE:
    // trace-driven version does not use this pc but their own trace_pc
    // set_next_pc is called but does not effect execution as m_next_pc is not
    // used.

    // m_next_pc = pc;
  }
  void store_info_of_last_inst_at_barrier(const warp_inst_t *pI) {
    m_inst_at_barrier = *pI;
  }
  warp_inst_t *restore_info_of_last_inst_at_barrier() {
    return &m_inst_at_barrier;
  }

  unsigned get_n_completed() const { return n_completed; }
  void set_completed(unsigned lane) {
    assert(m_active_threads.test(lane));
    m_active_threads.reset(lane);
    n_completed++;
  }

  void set_last_fetch(unsigned long long sim_cycle) {
    m_last_fetch = sim_cycle;
  }

  void set_membar() { m_membar = true; }
  void clear_membar() { m_membar = false; }
  bool get_membar() const { return m_membar; }

  unsigned get_n_atomic() const { return m_n_atomic; }
  void inc_n_atomic() { m_n_atomic++; }
  void dec_n_atomic(unsigned n) { m_n_atomic -= n; }

  bool stores_done() const { return m_stores_outstanding == 0; }
  void inc_store_req() { m_stores_outstanding++; }
  void dec_store_req() {
    assert(m_stores_outstanding > 0);
    m_stores_outstanding--;
  }

  bool inst_in_pipeline() const { return m_inst_in_pipeline > 0; }
  void inc_inst_in_pipeline() { m_inst_in_pipeline++; }
  void dec_inst_in_pipeline() {
    assert(m_inst_in_pipeline > 0);
    // printf("inst in pipeline: %d\n", m_inst_in_pipeline);
    // if (m_warp_id == 7)
    //   assert(0 && "warp 7 dec instr count");
    m_inst_in_pipeline--;
  }

  void ibuffer_fill(unsigned slot, const warp_inst_t *pI) {
    assert(slot < IBUFFER_SIZE);
    m_ibuffer[slot].m_inst = pI;
    m_ibuffer[slot].m_valid = true;
    m_next = 0;
  }
  bool ibuffer_empty() const {
    for (unsigned i = 0; i < IBUFFER_SIZE; i++)
      if (m_ibuffer[i].m_valid) return false;
    return true;
  }
  void ibuffer_flush() {
    for (unsigned i = 0; i < IBUFFER_SIZE; i++) {
      if (m_ibuffer[i].m_valid) dec_inst_in_pipeline();
      m_ibuffer[i].m_inst = NULL;
      m_ibuffer[i].m_valid = false;
    }
  }
  const warp_inst_t *ibuffer_next_inst() const {
    // printf("ibuffer next instruction (m_next = %d)\n", m_next);
    // throw std::runtime_error("ibuffer next inst");
    return m_ibuffer[m_next].m_inst;
  }
  bool ibuffer_next_valid() const { return m_ibuffer[m_next].m_valid; }
  void ibuffer_free() {
    m_ibuffer[m_next].m_inst = NULL;
    m_ibuffer[m_next].m_valid = false;
  }
  void ibuffer_step() { m_next = (m_next + 1) % IBUFFER_SIZE; }

  bool imiss_pending() const { return m_imiss_pending; }
  void set_imiss_pending() { m_imiss_pending = true; }
  void clear_imiss_pending() { m_imiss_pending = false; }

  unsigned get_cta_id() const { return m_cta_id; }

  unsigned get_dynamic_warp_id() const { return m_dynamic_warp_id; }
  unsigned get_warp_id() const { return m_warp_id; }

  class trace_shader_core_ctx *get_shader() const { return m_shader; }

  unsigned int m_cdp_latency;
  bool m_cdp_dummy;

  unsigned trace_pc;
  unsigned m_warp_size;
  std::bitset<MAX_WARP_SIZE> m_active_threads;

  static const unsigned IBUFFER_SIZE = 2;
  struct ibuffer_entry {
    ibuffer_entry() {
      m_valid = false;
      m_inst = NULL;
    }
    const warp_inst_t *m_inst;
    bool m_valid;
  };
  ibuffer_entry m_ibuffer[IBUFFER_SIZE];
  unsigned m_next;

 private:
  const trace_warp_inst_t *parse_trace_instruction(
      const inst_trace_t &trace) const;

  // unsigned trace_pc;
  trace_kernel_info_t *m_kernel_info;

  class trace_shader_core_ctx *m_shader;
  unsigned m_cta_id;
  unsigned m_warp_id;
  // unsigned m_warp_size;
  unsigned m_dynamic_warp_id;

  // static const unsigned IBUFFER_SIZE = 2;
  // struct ibuffer_entry {
  //   ibuffer_entry() {
  //     m_valid = false;
  //     m_inst = NULL;
  //   }
  //   const warp_inst_t *m_inst;
  //   bool m_valid;
  // };
  // ibuffer_entry m_ibuffer[IBUFFER_SIZE];
  // unsigned m_next;

  warp_inst_t m_inst_at_barrier;

  // address_type m_next_pc;
  unsigned n_completed;  // number of threads in warp completed
  // std::bitset<MAX_WARP_SIZE> m_active_threads;

  bool m_imiss_pending;

  unsigned m_n_atomic;  // number of outstanding atomic operations
  bool m_done_exit;  // true once thread exit has been registered for threads in
                     // this warp

  bool m_membar;  // if true, warp is waiting at memory barrier

  unsigned long long m_last_fetch;

 public:
  unsigned m_stores_outstanding;  // number of store requests sent but not yet
                                  // acknowledged
  unsigned m_inst_in_pipeline;
};

std::unique_ptr<trace_shd_warp_t> new_trace_shd_warp(
    class trace_shader_core_ctx *shader, unsigned warp_size);
