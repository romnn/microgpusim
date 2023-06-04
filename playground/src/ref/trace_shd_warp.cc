#include "trace_shd_warp.hpp"

#include "trace_kernel_info.hpp"
#include "trace_shader_core_ctx.hpp"
#include "trace_warp_inst.hpp"

const warp_inst_t *trace_shd_warp_t::get_next_trace_inst() {
  if (trace_pc < warp_traces.size()) {
    trace_warp_inst_t *new_inst =
        new trace_warp_inst_t(get_shader()->get_config());
    new_inst->parse_from_trace_struct(
        warp_traces[trace_pc], m_kernel_info->OpcodeMap,
        m_kernel_info->m_tconfig, m_kernel_info->m_kernel_trace_info);
    trace_pc++;
    return new_inst;
  } else
    return NULL;
}

void trace_shd_warp_t::clear() {
  trace_pc = 0;
  warp_traces.clear();
}

// functional_done
bool trace_shd_warp_t::trace_done() { return trace_pc == (warp_traces.size()); }

address_type trace_shd_warp_t::get_start_trace_pc() {
  assert(warp_traces.size() > 0);
  return warp_traces[0].m_pc;
}

address_type trace_shd_warp_t::get_pc() {
  assert(warp_traces.size() > 0);
  assert(trace_pc < warp_traces.size());
  return warp_traces[trace_pc].m_pc;
}

bool trace_shd_warp_t::waiting() {
  if (functional_done()) {
    // waiting to be initialized with a kernel
    return true;
  } else if (m_shader->warp_waiting_at_barrier(m_warp_id)) {
    // waiting for other warps in CTA to reach barrier
    return true;
  } else if (m_shader->warp_waiting_at_mem_barrier(m_warp_id)) {
    // waiting for memory barrier
    return true;
  } else if (m_n_atomic > 0) {
    // waiting for atomic operation to complete at memory:
    // this stall is not required for accurate timing model, but rather we
    // stall here since if a call/return instruction occurs in the meantime
    // the functional execution of the atomic when it hits DRAM can cause
    // the wrong register to be read.
    return true;
  }
  return false;
}

bool trace_shd_warp_t::functional_done() const {
  return get_n_completed() == m_warp_size;
}

bool trace_shd_warp_t::hardware_done() const {
  return functional_done() && stores_done() && !inst_in_pipeline();
}
