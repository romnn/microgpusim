#pragma once

#include <cstddef>
#include <list>

struct gpgpu_ptx_sim_arg {
  gpgpu_ptx_sim_arg() { m_start = NULL; }
  gpgpu_ptx_sim_arg(const void *arg, size_t size, size_t offset) {
    m_start = arg;
    m_nbytes = size;
    m_offset = offset;
  }
  const void *m_start;
  size_t m_nbytes;
  size_t m_offset;
};

typedef std::list<gpgpu_ptx_sim_arg> gpgpu_ptx_sim_arg_list_t;
