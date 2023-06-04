#pragma once

#include "mem_fetch.hpp"
#include "mem_fetch_interface.hpp"
#include "trace_shader_core_ctx.hpp"
#include "trace_simt_core_cluster.hpp"

class perfect_memory_interface : public mem_fetch_interface {
public:
  perfect_memory_interface(trace_shader_core_ctx *core,
                           trace_simt_core_cluster *cluster) {
    m_core = core;
    m_cluster = cluster;
  }
  virtual bool full(unsigned size, bool write) const {
    return m_cluster->response_queue_full();
  }
  virtual void push(mem_fetch *mf) {
    if (mf && mf->isatomic())
      mf->do_atomic(); // execute atomic inside the "memory subsystem"
    m_core->inc_simt_to_mem(mf->get_num_flits(true));
    m_cluster->push_response_fifo(mf);
  }

private:
  trace_shader_core_ctx *m_core;
  trace_simt_core_cluster *m_cluster;
};
