#pragma once

#include <cassert>
#include <cstddef>

#include "hal.hpp"

class gpgpu_sim;
class kernel_info_t;
class warp_inst_t;
class simt_stack;

/*
 * This abstract class used as a base for functional and performance and
 * simulation, it has basic functional simulation data structures and
 * procedures.
 */
class core_t {
public:
  core_t(gpgpu_sim *gpu, kernel_info_t *kernel, unsigned warp_size,
         unsigned threads_per_shader)
      : m_gpu(gpu), m_kernel(kernel), m_simt_stack(NULL), m_thread(NULL),
        m_warp_size(warp_size) {
    m_warp_count = threads_per_shader / m_warp_size;
    // Handle the case where the number of threads is not a
    // multiple of the warp size
    if (threads_per_shader % m_warp_size != 0) {
      m_warp_count += 1;
    }
    assert(m_warp_count * m_warp_size > 0);
    m_thread = (ptx_thread_info **)calloc(m_warp_count * m_warp_size,
                                          sizeof(ptx_thread_info *));
    initilizeSIMTStack(m_warp_count, m_warp_size);

    for (unsigned i = 0; i < MAX_CTA_PER_SHADER; i++) {
      for (unsigned j = 0; j < MAX_BARRIERS_PER_CTA; j++) {
        reduction_storage[i][j] = 0;
      }
    }
  }
  virtual ~core_t() { free(m_thread); }
  virtual void warp_exit(unsigned warp_id) = 0;
  virtual bool warp_waiting_at_barrier(unsigned warp_id) const = 0;
  virtual void checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t,
                                             unsigned tid) = 0;
  class gpgpu_sim *get_gpu() { return m_gpu; }
  void execute_warp_inst_t(warp_inst_t &inst, unsigned warpId = (unsigned)-1);
  bool ptx_thread_done(unsigned hw_thread_id) const;
  virtual void updateSIMTStack(unsigned warpId, warp_inst_t *inst);
  void initilizeSIMTStack(unsigned warp_count, unsigned warps_size);
  void deleteSIMTStack();
  warp_inst_t getExecuteWarp(unsigned warpId);
  void get_pdom_stack_top_info(unsigned warpId, unsigned *pc,
                               unsigned *rpc) const;
  kernel_info_t *get_kernel_info() { return m_kernel; }
  class ptx_thread_info **get_thread_info() { return m_thread; }
  unsigned get_warp_size() const { return m_warp_size; }
  void and_reduction(unsigned ctaid, unsigned barid, bool value) {
    reduction_storage[ctaid][barid] &= value;
  }
  void or_reduction(unsigned ctaid, unsigned barid, bool value) {
    reduction_storage[ctaid][barid] |= value;
  }
  void popc_reduction(unsigned ctaid, unsigned barid, bool value) {
    reduction_storage[ctaid][barid] += value;
  }
  unsigned get_reduction_value(unsigned ctaid, unsigned barid) {
    return reduction_storage[ctaid][barid];
  }

protected:
  class gpgpu_sim *m_gpu;
  kernel_info_t *m_kernel;
  simt_stack **m_simt_stack; // pdom based reconvergence context for each warp
  class ptx_thread_info **m_thread;
  unsigned m_warp_size;
  unsigned m_warp_count;
  unsigned reduction_storage[MAX_CTA_PER_SHADER][MAX_BARRIERS_PER_CTA];
};
