#pragma once

#include <set>

// #include "gpgpu_context.hpp"
// #include "ptx_thread_info.hpp"

class gpgpu_context;
class ptx_thread_info;

class ptx_cta_info {
 public:
  ptx_cta_info(unsigned sm_idx, gpgpu_context *ctx);
  void add_thread(ptx_thread_info *thd);
  unsigned num_threads() const;
  void check_cta_thread_status_and_reset();
  void register_thread_exit(ptx_thread_info *thd);
  void register_deleted_thread(ptx_thread_info *thd);
  unsigned get_sm_idx() const;
  unsigned get_bar_threads() const;
  void inc_bar_threads();
  void reset_bar_threads();

 private:
  // backward pointer
  class gpgpu_context *gpgpu_ctx;
  unsigned m_bar_threads;
  unsigned long long m_uid;
  unsigned m_sm_idx;
  std::set<ptx_thread_info *> m_threads_in_cta;
  std::set<ptx_thread_info *> m_threads_that_have_exited;
  std::set<ptx_thread_info *> m_dangling_pointers;
};
