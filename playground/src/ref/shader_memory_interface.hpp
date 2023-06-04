#pragma once

#include "mem_fetch_interface.hpp"
#include "trace_shader_core_ctx.hpp"
#include "trace_simt_core_cluster.hpp"

class shader_memory_interface : public mem_fetch_interface {
public:
  shader_memory_interface(trace_shader_core_ctx *core,
                          trace_simt_core_cluster *cluster) {
    m_core = core;
    m_cluster = cluster;
  }
  virtual bool full(unsigned size, bool write) const {
    return m_cluster->icnt_injection_buffer_full(size, write);
  }
  virtual void push(mem_fetch *mf) {
    m_core->inc_simt_to_mem(mf->get_num_flits(true));
    m_cluster->icnt_inject_request_packet(mf);
  }

private:
  trace_shader_core_ctx *m_core;
  trace_simt_core_cluster *m_cluster;
};
