#include "mem_access.hpp"

#include <iostream>

class gpgpu_context;

void mem_access_t::init(gpgpu_context *ctx) {
  // gpgpu_ctx = ctx;
  // m_uid = ++(gpgpu_ctx->sm_next_access_uid);
  m_uid = ++(ctx->sm_next_access_uid);
  m_addr = 0;
  m_allocation_start_addr = 0;
  m_allocation_id = 0;
  m_req_size = 0;
}

std::ostream &operator<<(std::ostream &os, const mem_access_t &access) {
  os << access.get_type_str() << "@";
  new_addr_type addr = access.get_addr();
  new_addr_type rel_addr = access.get_relative_addr();
  if (addr == rel_addr) {
    os << addr;
  } else {
    os << access.get_alloc_id() << "+" << rel_addr;
  }
  return os;
}
