#include "mem_access.hpp"

class gpgpu_context;

void mem_access_t::init(gpgpu_context *ctx) {
  gpgpu_ctx = ctx;
  m_uid = ++(gpgpu_ctx->sm_next_access_uid);
  m_addr = 0;
  m_req_size = 0;
}