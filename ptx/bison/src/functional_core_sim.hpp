#pragma once

#include "core.hpp"

#include "kernel_info.hpp"
#include "ptx_thread_info.hpp"

/*!
 * This class functionally executes a kernel. It uses the basic data structures
 * and procedures in core_t
 */
class functionalCoreSim : public core_t {
public:
  functionalCoreSim(kernel_info_t *kernel, gpgpu_sim *g, unsigned warp_size)
      : core_t(g, kernel, warp_size, kernel->threads_per_cta()) {
    m_warpAtBarrier = new bool[m_warp_count];
    m_liveThreadCount = new unsigned[m_warp_count];
  }
  virtual ~functionalCoreSim() {
    warp_exit(0);
    delete[] m_liveThreadCount;
    delete[] m_warpAtBarrier;
  }
  //! executes all warps till completion
  void execute(int inst_count, unsigned ctaid_cp);
  virtual void warp_exit(unsigned warp_id);
  virtual bool warp_waiting_at_barrier(unsigned warp_id) const {
    return (m_warpAtBarrier[warp_id] || !(m_liveThreadCount[warp_id] > 0));
  }

private:
  void executeWarp(unsigned, bool &, bool &);
  // initializes threads in the CTA block which we are executing
  void initializeCTA(unsigned ctaid_cp);
  virtual void checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t,
                                             unsigned tid) {
    if (m_thread[tid] == NULL || m_thread[tid]->is_done()) {
      m_liveThreadCount[tid / m_warp_size]--;
    }
  }

  // lunches the stack and set the threads count
  void createWarp(unsigned warpId);

  // each warp live thread count and barrier indicator
  unsigned *m_liveThreadCount;
  bool *m_warpAtBarrier;
};
