#pragma once

#include <list>
#include <pthread.h>

#include "stream_operation.hpp"

struct CUstream_st {
public:
  CUstream_st();
  bool empty();
  bool busy();
  void synchronize();
  void push(const stream_operation &op);
  void record_next_done();
  stream_operation next();
  void cancel_front(); // front operation fails, cancle the pending status
  stream_operation &front() { return m_operations.front(); }
  void print(FILE *fp);
  unsigned get_uid() const { return m_uid; }

private:
  unsigned m_uid;
  static unsigned sm_next_stream_uid;

  std::list<stream_operation> m_operations;
  bool m_pending; // front operation has started but not yet completed

  pthread_mutex_t m_lock; // ensure only one host or gpu manipulates stream
                          // operation at one time
};

typedef struct CUstream_st *CUstream;
