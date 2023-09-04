#pragma once

#include <cstdio>
#include <list>
#include <map>
#include <pthread.h>

#include "cu_stream.hpp"
#include "stream_operation.hpp"

class trace_gpgpu_sim;

class stream_manager {
 public:
  stream_manager(trace_gpgpu_sim *gpu, bool cuda_launch_blocking);
  bool register_finished_kernel(unsigned grid_uid);
  bool check_finished_kernel();
  stream_operation front();
  void add_stream(CUstream_st *stream);
  void destroy_stream(CUstream_st *stream);
  bool concurrent_streams_empty();
  bool empty_protected();
  bool empty();
  void print(FILE *fp);
  void push(stream_operation op);
  void pushCudaStreamWaitEventToAllStreams(CUevent_st *e, unsigned int flags);
  bool operation(bool *sim);
  void stop_all_running_kernels(FILE *fp);
  unsigned size() { return m_streams.size(); };
  bool is_blocking() { return m_cuda_launch_blocking; };

 private:
  void print_impl(FILE *fp);

  bool m_cuda_launch_blocking;
  trace_gpgpu_sim *m_gpu;
  std::list<CUstream_st *> m_streams;
  std::map<unsigned, CUstream_st *> m_grid_id_to_stream;
  CUstream_st m_stream_zero;
  bool m_service_stream_zero;
  pthread_mutex_t m_lock;
  std::list<struct CUstream_st *>::iterator m_last_stream;
};
