#pragma once

#include <assert.h>
#include <bitset>
#include <memory>

#include "hal.hpp"
#include "warp_instr.hpp"

class kernel_info_t;

class shd_warp_t {
public:
  shd_warp_t(class shader_core_ctx *shader, unsigned warp_size)
      : m_shader(shader), m_warp_size(warp_size) {
    m_stores_outstanding = 0;
    m_inst_in_pipeline = 0;
    reset();
  }
  void reset() {
    assert(m_stores_outstanding == 0);
    assert(m_inst_in_pipeline == 0);
    m_imiss_pending = false;
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
    m_next_pc = start_pc;
    assert(n_completed >= active.count());
    assert(n_completed <= m_warp_size);
    n_completed -= active.count(); // active threads are not yet completed
    m_active_threads = active;
    m_done_exit = false;

    // Jin: cdp support
    m_cdp_latency = 0;
    m_cdp_dummy = false;
  }

  bool functional_done() const;
  bool waiting(); // not const due to membar
  bool hardware_done() const;

  bool done_exit() const { return m_done_exit; }
  void set_done_exit() { m_done_exit = true; }

  void print(FILE *fout) const;
  void print_ibuffer(FILE *fout) const;

  unsigned get_n_completed() const { return n_completed; }
  void set_completed(unsigned lane) {
    assert(m_active_threads.test(lane));
    m_active_threads.reset(lane);
    n_completed++;
  }

  void set_last_fetch(unsigned long long sim_cycle) {
    m_last_fetch = sim_cycle;
  }

  unsigned get_n_atomic() const { return m_n_atomic; }
  void inc_n_atomic() { m_n_atomic++; }
  void dec_n_atomic(unsigned n) { m_n_atomic -= n; }

  void set_membar() { m_membar = true; }
  void clear_membar() { m_membar = false; }
  bool get_membar() const { return m_membar; }
  virtual address_type get_pc() const { return m_next_pc; }
  virtual kernel_info_t *get_kernel_info() const;
  void set_next_pc(address_type pc) { m_next_pc = pc; }

  void store_info_of_last_inst_at_barrier(const warp_inst_t *pI) {
    m_inst_at_barrier = *pI;
  }
  warp_inst_t *restore_info_of_last_inst_at_barrier() {
    return &m_inst_at_barrier;
  }

  void ibuffer_fill(unsigned slot, const warp_inst_t *pI) {
    assert(slot < IBUFFER_SIZE);
    m_ibuffer[slot].m_inst = pI;
    m_ibuffer[slot].m_valid = true;
    m_next = 0;
  }
  bool ibuffer_empty() const {
    for (unsigned i = 0; i < IBUFFER_SIZE; i++)
      if (m_ibuffer[i].m_valid)
        return false;
    return true;
  }
  void ibuffer_flush() {
    for (unsigned i = 0; i < IBUFFER_SIZE; i++) {
      if (m_ibuffer[i].m_valid)
        dec_inst_in_pipeline();
      m_ibuffer[i].m_inst = NULL;
      m_ibuffer[i].m_valid = false;
    }
  }
  const warp_inst_t *ibuffer_next_inst() const {
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

  bool stores_done() const { return m_stores_outstanding == 0; }
  void inc_store_req() { m_stores_outstanding++; }
  void dec_store_req() {
    assert(m_stores_outstanding > 0);
    m_stores_outstanding--;
  }

  unsigned num_inst_in_buffer() const {
    unsigned count = 0;
    for (unsigned i = 0; i < IBUFFER_SIZE; i++) {
      if (m_ibuffer[i].m_valid)
        count++;
    }
    return count;
  }
  unsigned num_inst_in_pipeline() const { return m_inst_in_pipeline; }
  unsigned num_issued_inst_in_pipeline() const {
    return (num_inst_in_pipeline() - num_inst_in_buffer());
  }
  bool inst_in_pipeline() const { return m_inst_in_pipeline > 0; }
  void inc_inst_in_pipeline() { m_inst_in_pipeline++; }
  void dec_inst_in_pipeline() {
    assert(m_inst_in_pipeline > 0);
    m_inst_in_pipeline--;
  }

  unsigned get_cta_id() const { return m_cta_id; }

  unsigned get_dynamic_warp_id() const { return m_dynamic_warp_id; }
  unsigned get_warp_id() const { return m_warp_id; }

  class shader_core_ctx *get_shader() {
    return m_shader;
  }

private:
  static const unsigned IBUFFER_SIZE = 2;
  class shader_core_ctx *m_shader;
  unsigned m_cta_id;
  unsigned m_warp_id;
  unsigned m_warp_size;
  unsigned m_dynamic_warp_id;

  address_type m_next_pc;
  unsigned n_completed; // number of threads in warp completed
  std::bitset<MAX_WARP_SIZE> m_active_threads;

  bool m_imiss_pending;

  struct ibuffer_entry {
    ibuffer_entry() {
      m_valid = false;
      m_inst = NULL;
    }
    const warp_inst_t *m_inst;
    bool m_valid;
  };

  warp_inst_t m_inst_at_barrier;
  ibuffer_entry m_ibuffer[IBUFFER_SIZE];
  unsigned m_next;

  unsigned m_n_atomic; // number of outstanding atomic operations
  bool m_membar;       // if true, warp is waiting at memory barrier

  bool m_done_exit; // true once thread exit has been registered for threads in
                    // this warp

  unsigned long long m_last_fetch;

  unsigned m_stores_outstanding; // number of store requests sent but not yet
                                 // acknowledged
  unsigned m_inst_in_pipeline;

  // Jin: cdp support
public:
  unsigned int m_cdp_latency;
  bool m_cdp_dummy;
};

std::unique_ptr<shd_warp_t> new_shd_warp(class shader_core_ctx *shader,
                                         unsigned warp_size);
