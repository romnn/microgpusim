#include "core.hpp"

#include "gpgpu_sim.hpp"
// #include "ptx_thread_info.hpp"

// REMOVE: ptx
// void core_t::execute_warp_inst_t(warp_inst_t &inst, unsigned warpId) {
//   for (unsigned t = 0; t < m_warp_size; t++) {
//     if (inst.active(t)) {
//       if (warpId == (unsigned(-1)))
//         warpId = inst.warp_id();
//       unsigned tid = m_warp_size * warpId + t;
//       m_thread[tid]->ptx_exec_inst(inst, t);
//
//       // virtual function
//       checkExecutionStatusAndUpdate(inst, t, tid);
//     }
//   }
// }

// bool core_t::ptx_thread_done(unsigned hw_thread_id) const {
//   return ((m_thread[hw_thread_id] == NULL) ||
//           m_thread[hw_thread_id]->is_done());
// }

// void core_t::updateSIMTStack(unsigned warpId, warp_inst_t *inst) {
//   simt_mask_t thread_done;
//   addr_vector_t next_pc;
//   unsigned wtid = warpId * m_warp_size;
//   for (unsigned i = 0; i < m_warp_size; i++) {
//     if (ptx_thread_done(wtid + i)) {
//       thread_done.set(i);
//       next_pc.push_back((address_type)-1);
//     } else {
//       if (inst->reconvergence_pc == RECONVERGE_RETURN_PC)
//         inst->reconvergence_pc = get_return_pc(m_thread[wtid + i]);
//       next_pc.push_back(m_thread[wtid + i]->get_pc());
//     }
//   }
//   m_simt_stack[warpId]->update(thread_done, next_pc, inst->reconvergence_pc,
//                                inst->op, inst->isize, inst->pc);
// }

//! Get the warp to be executed using the data taken form the SIMT stack
warp_inst_t core_t::getExecuteWarp(unsigned warpId) {
  unsigned pc, rpc;
  m_simt_stack[warpId]->get_pdom_stack_top_info(&pc, &rpc);
  // REMOVE: ptx
  fprintf(stderr, "do not have ptx fetch inst");
  abort();
  // warp_inst_t wi = *(m_gpu->gpgpu_ctx->ptx_fetch_inst(pc));
  // wi.set_active(m_simt_stack[warpId]->get_active_mask());
  // return wi;
}

void core_t::deleteSIMTStack() {
  if (m_simt_stack) {
    for (unsigned i = 0; i < m_warp_count; ++i) delete m_simt_stack[i];
    delete[] m_simt_stack;
    m_simt_stack = NULL;
  }
}

void core_t::initilizeSIMTStack(unsigned warp_count, unsigned warp_size) {
  m_simt_stack = new simt_stack *[warp_count];
  for (unsigned i = 0; i < warp_count; ++i)
    m_simt_stack[i] = new simt_stack(i, warp_size, m_gpu);
  m_warp_size = warp_size;
  m_warp_count = warp_count;
}

void core_t::get_pdom_stack_top_info(unsigned warpId, unsigned *pc,
                                     unsigned *rpc) const {
  m_simt_stack[warpId]->get_pdom_stack_top_info(pc, rpc);
}
