#include <assert.h>

#include "cu_stream.hpp"
#include "stream_operation.hpp"

unsigned CUstream_st::sm_next_stream_uid = 0;

CUstream_st::CUstream_st() {
  m_pending = false;
  m_uid = sm_next_stream_uid++;
  pthread_mutex_init(&m_lock, NULL);
}

bool CUstream_st::empty() {
  pthread_mutex_lock(&m_lock);
  bool empty = m_operations.empty();
  pthread_mutex_unlock(&m_lock);
  return empty;
}

bool CUstream_st::busy() {
  pthread_mutex_lock(&m_lock);
  bool pending = m_pending;
  pthread_mutex_unlock(&m_lock);
  return pending;
}

void CUstream_st::synchronize() {
  // called by host thread
  bool done = false;
  do {
    pthread_mutex_lock(&m_lock);
    done = m_operations.empty();
    pthread_mutex_unlock(&m_lock);
  } while (!done);
}

void CUstream_st::push(const stream_operation &op) {
  // called by host thread
  pthread_mutex_lock(&m_lock);
  m_operations.push_back(op);
  pthread_mutex_unlock(&m_lock);
}

void CUstream_st::record_next_done() {
  // called by gpu thread
  pthread_mutex_lock(&m_lock);
  assert(m_pending);
  m_operations.pop_front();
  m_pending = false;
  pthread_mutex_unlock(&m_lock);
}

stream_operation CUstream_st::next() {
  // called by gpu thread
  pthread_mutex_lock(&m_lock);
  m_pending = true;
  stream_operation result = m_operations.front();
  pthread_mutex_unlock(&m_lock);
  return result;
}

void CUstream_st::cancel_front() {
  pthread_mutex_lock(&m_lock);
  assert(m_pending);
  m_pending = false;
  pthread_mutex_unlock(&m_lock);
}

void CUstream_st::print(FILE *fp) {
  pthread_mutex_lock(&m_lock);
  fprintf(fp, "GPGPU-Sim API:    stream %u has %zu operations\n", m_uid,
          m_operations.size());
  std::list<stream_operation>::iterator i;
  unsigned n = 0;
  for (i = m_operations.begin(); i != m_operations.end(); i++) {
    stream_operation &op = *i;
    fprintf(fp, "GPGPU-Sim API:       %u : ", n++);
    op.print(fp);
    fprintf(fp, "\n");
  }
  pthread_mutex_unlock(&m_lock);
}
