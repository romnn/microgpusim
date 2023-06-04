#pragma once

#include <cstddef>
#include <cstdio>

#include "cu_event.hpp"

class trace_kernel_info_t;
class CUstream_t;
class trace_gpgpu_sim;

enum stream_operation_type {
  stream_no_op,
  stream_memcpy_host_to_device,
  stream_memcpy_device_to_host,
  stream_memcpy_device_to_device,
  stream_memcpy_to_symbol,
  stream_memcpy_from_symbol,
  stream_kernel_launch,
  stream_event,
  stream_wait_event
};

class stream_operation {
public:
  stream_operation() {
    m_kernel = NULL;
    m_type = stream_no_op;
    m_stream = NULL;
    m_done = true;
  }
  stream_operation(const void *src, const char *symbol, size_t count,
                   size_t offset, struct CUstream_st *stream) {
    m_kernel = NULL;
    m_stream = stream;
    m_type = stream_memcpy_to_symbol;
    m_host_address_src = src;
    m_symbol = symbol;
    m_cnt = count;
    m_offset = offset;
    m_done = false;
  }
  stream_operation(const char *symbol, void *dst, size_t count, size_t offset,
                   struct CUstream_st *stream) {
    m_kernel = NULL;
    m_stream = stream;
    m_type = stream_memcpy_from_symbol;
    m_host_address_dst = dst;
    m_symbol = symbol;
    m_cnt = count;
    m_offset = offset;
    m_done = false;
  }
  stream_operation(trace_kernel_info_t *kernel, bool sim_mode,
                   struct CUstream_st *stream) {
    m_type = stream_kernel_launch;
    m_kernel = kernel;
    m_sim_mode = sim_mode;
    m_stream = stream;
    m_done = false;
  }
  stream_operation(struct CUevent_st *e, struct CUstream_st *stream) {
    m_kernel = NULL;
    m_type = stream_event;
    m_event = e;
    m_stream = stream;
    m_done = false;
  }
  stream_operation(struct CUstream_st *stream, class CUevent_st *e,
                   unsigned int flags) {
    m_kernel = NULL;
    m_type = stream_wait_event;
    m_event = e;
    m_cnt = m_event->num_issued();
    m_stream = stream;
    m_done = false;
  }
  stream_operation(const void *host_address_src, size_t device_address_dst,
                   size_t cnt, struct CUstream_st *stream) {
    m_kernel = NULL;
    m_type = stream_memcpy_host_to_device;
    m_host_address_src = host_address_src;
    m_device_address_dst = device_address_dst;
    m_host_address_dst = NULL;
    m_device_address_src = 0;
    m_cnt = cnt;
    m_stream = stream;
    m_sim_mode = false;
    m_done = false;
  }
  stream_operation(size_t device_address_src, void *host_address_dst,
                   size_t cnt, struct CUstream_st *stream) {
    m_kernel = NULL;
    m_type = stream_memcpy_device_to_host;
    m_device_address_src = device_address_src;
    m_host_address_dst = host_address_dst;
    m_device_address_dst = 0;
    m_host_address_src = NULL;
    m_cnt = cnt;
    m_stream = stream;
    m_sim_mode = false;
    m_done = false;
  }
  stream_operation(size_t device_address_src, size_t device_address_dst,
                   size_t cnt, struct CUstream_st *stream) {
    m_kernel = NULL;
    m_type = stream_memcpy_device_to_device;
    m_device_address_src = device_address_src;
    m_device_address_dst = device_address_dst;
    m_host_address_src = NULL;
    m_host_address_dst = NULL;
    m_cnt = cnt;
    m_stream = stream;
    m_sim_mode = false;
    m_done = false;
  }

  bool is_kernel() const { return m_type == stream_kernel_launch; }
  bool is_mem() const {
    return m_type == stream_memcpy_host_to_device ||
           m_type == stream_memcpy_device_to_host ||
           m_type == stream_memcpy_host_to_device;
  }
  bool is_noop() const { return m_type == stream_no_op; }
  bool is_done() const { return m_done; }
  trace_kernel_info_t *get_kernel() { return m_kernel; }
  bool do_operation(trace_gpgpu_sim *gpu);
  void print(FILE *fp) const;
  struct CUstream_st *get_stream() { return m_stream; }
  void set_stream(CUstream_st *stream) { m_stream = stream; }

private:
  struct CUstream_st *m_stream;

  bool m_done;

  stream_operation_type m_type;
  size_t m_device_address_dst;
  size_t m_device_address_src;
  void *m_host_address_dst;
  const void *m_host_address_src;
  size_t m_cnt;

  const char *m_symbol;
  size_t m_offset;

  bool m_sim_mode;
  trace_kernel_info_t *m_kernel;
  struct CUevent_st *m_event;
};
