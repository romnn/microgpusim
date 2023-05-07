use crate::ported::mem_fetch::AccessKind;

use super::address;
use nvbit_model::MemorySpace;
use std::collections::VecDeque;
use trace_model::MemAccessTraceEntry;

#[derive(Debug)]
pub struct LoadStoreUnit {}

// bool ldst_unit::shared_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
//                              mem_stage_access_type &fail_type) {
//   if (inst.space.get_type() != shared_space) return true;
//
//   if (inst.active_count() == 0) return true;
//
//   if (inst.has_dispatch_delay()) {
//     m_stats->gpgpu_n_shmem_bank_access[m_sid]++;
//   }
//
//   bool stall = inst.dispatch_delay();
//   if (stall) {
//     fail_type = S_MEM;
//     rc_fail = BK_CONF;
//   } else
//     rc_fail = NO_RC_FAIL;
//   return !stall;
// }
//
// mem_stage_stall_type ldst_unit::process_cache_access(
//     cache_t *cache, new_addr_type address, warp_inst_t &inst,
//     std::list<cache_event> &events, mem_fetch *mf,
//     enum cache_request_status status) {
//   mem_stage_stall_type result = NO_RC_FAIL;
//   bool write_sent = was_write_sent(events);
//   bool read_sent = was_read_sent(events);
//   if (write_sent) {
//     unsigned inc_ack = (m_config->m_L1D_config.get_mshr_type() == SECTOR_ASSOC)
//                            ? (mf->get_data_size() / SECTOR_SIZE)
//                            : 1;
//
//     for (unsigned i = 0; i < inc_ack; ++i)
//       m_core->inc_store_req(inst.warp_id());
//   }
//   if (status == HIT) {
//     assert(!read_sent);
//     inst.accessq_pop_back();
//     if (inst.is_load()) {
//       for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++)
//         if (inst.out[r] > 0) m_pending_writes[inst.warp_id()][inst.out[r]]--;
//     }
//     if (!write_sent) delete mf;
//   } else if (status == RESERVATION_FAIL) {
//     result = BK_CONF;
//     assert(!read_sent);
//     assert(!write_sent);
//     delete mf;
//   } else {
//     assert(status == MISS || status == HIT_RESERVED);
//     // inst.clear_active( access.get_warp_mask() ); // threads in mf writeback
//     // when mf returns
//     inst.accessq_pop_back();
//   }
//   if (!inst.accessq_empty() && result == NO_RC_FAIL) result = COAL_STALL;
//   return result;
// }
//
// mem_stage_stall_type ldst_unit::process_memory_access_queue(cache_t *cache,
//                                                             warp_inst_t &inst) {
//   mem_stage_stall_type result = NO_RC_FAIL;
//   if (inst.accessq_empty()) return result;
//
//   if (!cache->data_port_free()) return DATA_PORT_STALL;
//
//   // const mem_access_t &access = inst.accessq_back();
//   mem_fetch *mf = m_mf_allocator->alloc(
//       inst, inst.accessq_back(),
//       m_core->get_gpu()->gpu_sim_cycle + m_core->get_gpu()->gpu_tot_sim_cycle);
//   std::list<cache_event> events;
//   enum cache_request_status status = cache->access(
//       mf->get_addr(), mf,
//       m_core->get_gpu()->gpu_sim_cycle + m_core->get_gpu()->gpu_tot_sim_cycle,
//       events);
//   return process_cache_access(cache, mf->get_addr(), inst, events, mf, status);
// }
//
// mem_stage_stall_type ldst_unit::process_memory_access_queue_l1cache(
//     l1_cache *cache, warp_inst_t &inst) {
//   mem_stage_stall_type result = NO_RC_FAIL;
//   if (inst.accessq_empty()) return result;
//
//   if (m_config->m_L1D_config.l1_latency > 0) {
//     for (int j = 0; j < m_config->m_L1D_config.l1_banks;
//          j++) {  // We can handle at max l1_banks reqs per cycle
//
//       if (inst.accessq_empty()) return result;
//
//       mem_fetch *mf =
//           m_mf_allocator->alloc(inst, inst.accessq_back(),
//                                 m_core->get_gpu()->gpu_sim_cycle +
//                                     m_core->get_gpu()->gpu_tot_sim_cycle);
//       unsigned bank_id = m_config->m_L1D_config.set_bank(mf->get_addr());
//       assert(bank_id < m_config->m_L1D_config.l1_banks);
//
//       if ((l1_latency_queue[bank_id][m_config->m_L1D_config.l1_latency - 1]) ==
//           NULL) {
//         l1_latency_queue[bank_id][m_config->m_L1D_config.l1_latency - 1] = mf;
//
//         if (mf->get_inst().is_store()) {
//           unsigned inc_ack =
//               (m_config->m_L1D_config.get_mshr_type() == SECTOR_ASSOC)
//                   ? (mf->get_data_size() / SECTOR_SIZE)
//                   : 1;
//
//           for (unsigned i = 0; i < inc_ack; ++i)
//             m_core->inc_store_req(inst.warp_id());
//         }
//
//         inst.accessq_pop_back();
//       } else {
//         result = BK_CONF;
//         delete mf;
//         break;  // do not try again, just break from the loop and try the next
//                 // cycle
//       }
//     }
//     if (!inst.accessq_empty() && result != BK_CONF) result = COAL_STALL;
//
//     return result;
//   } else {
//     mem_fetch *mf =
//         m_mf_allocator->alloc(inst, inst.accessq_back(),
//                               m_core->get_gpu()->gpu_sim_cycle +
//                                   m_core->get_gpu()->gpu_tot_sim_cycle);
//     std::list<cache_event> events;
//     enum cache_request_status status = cache->access(
//         mf->get_addr(), mf,
//         m_core->get_gpu()->gpu_sim_cycle + m_core->get_gpu()->gpu_tot_sim_cycle,
//         events);
//     return process_cache_access(cache, mf->get_addr(), inst, events, mf,
//                                 status);
//   }
// }
//
// void ldst_unit::L1_latency_queue_cycle() {
//   for (int j = 0; j < m_config->m_L1D_config.l1_banks; j++) {
//     if ((l1_latency_queue[j][0]) != NULL) {
//       mem_fetch *mf_next = l1_latency_queue[j][0];
//       std::list<cache_event> events;
//       enum cache_request_status status =
//           m_L1D->access(mf_next->get_addr(), mf_next,
//                         m_core->get_gpu()->gpu_sim_cycle +
//                             m_core->get_gpu()->gpu_tot_sim_cycle,
//                         events);
//
//       bool write_sent = was_write_sent(events);
//       bool read_sent = was_read_sent(events);
//
//       if (status == HIT) {
//         assert(!read_sent);
//         l1_latency_queue[j][0] = NULL;
//         if (mf_next->get_inst().is_load()) {
//           for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++)
//             if (mf_next->get_inst().out[r] > 0) {
//               assert(m_pending_writes[mf_next->get_inst().warp_id()]
//                                      [mf_next->get_inst().out[r]] > 0);
//               unsigned still_pending =
//                   --m_pending_writes[mf_next->get_inst().warp_id()]
//                                     [mf_next->get_inst().out[r]];
//               if (!still_pending) {
//                 m_pending_writes[mf_next->get_inst().warp_id()].erase(
//                     mf_next->get_inst().out[r]);
//                 m_scoreboard->releaseRegister(mf_next->get_inst().warp_id(),
//                                               mf_next->get_inst().out[r]);
//                 m_core->warp_inst_complete(mf_next->get_inst());
//               }
//             }
//         }
//
//         // For write hit in WB policy
//         if (mf_next->get_inst().is_store() && !write_sent) {
//           unsigned dec_ack =
//               (m_config->m_L1D_config.get_mshr_type() == SECTOR_ASSOC)
//                   ? (mf_next->get_data_size() / SECTOR_SIZE)
//                   : 1;
//
//           mf_next->set_reply();
//
//           for (unsigned i = 0; i < dec_ack; ++i) m_core->store_ack(mf_next);
//         }
//
//         if (!write_sent) delete mf_next;
//
//       } else if (status == RESERVATION_FAIL) {
//         assert(!read_sent);
//         assert(!write_sent);
//       } else {
//         assert(status == MISS || status == HIT_RESERVED);
//         l1_latency_queue[j][0] = NULL;
//         if (m_config->m_L1D_config.get_write_policy() != WRITE_THROUGH &&
//             mf_next->get_inst().is_store() &&
//             (m_config->m_L1D_config.get_write_allocate_policy() ==
//                  FETCH_ON_WRITE ||
//              m_config->m_L1D_config.get_write_allocate_policy() ==
//                  LAZY_FETCH_ON_READ) &&
//             !was_writeallocate_sent(events)) {
//           unsigned dec_ack =
//               (m_config->m_L1D_config.get_mshr_type() == SECTOR_ASSOC)
//                   ? (mf_next->get_data_size() / SECTOR_SIZE)
//                   : 1;
//           mf_next->set_reply();
//           for (unsigned i = 0; i < dec_ack; ++i) m_core->store_ack(mf_next);
//           if (!write_sent && !read_sent) delete mf_next;
//         }
//       }
//     }
//
//     for (unsigned stage = 0; stage < m_config->m_L1D_config.l1_latency - 1;
//          ++stage)
//       if (l1_latency_queue[j][stage] == NULL) {
//         l1_latency_queue[j][stage] = l1_latency_queue[j][stage + 1];
//         l1_latency_queue[j][stage + 1] = NULL;
//       }
//   }
// }
//
// bool ldst_unit::constant_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
//                                mem_stage_access_type &fail_type) {
//   if (inst.empty() || ((inst.space.get_type() != const_space) &&
//                        (inst.space.get_type() != param_space_kernel)))
//     return true;
//   if (inst.active_count() == 0) return true;
//
//   mem_stage_stall_type fail;
//   if (m_config->perfect_inst_const_cache) {
//     fail = NO_RC_FAIL;
//     unsigned access_count = inst.accessq_count();
//     while (inst.accessq_count() > 0) inst.accessq_pop_back();
//     if (inst.is_load()) {
//       for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++)
//         if (inst.out[r] > 0) m_pending_writes[inst.warp_id()][inst.out[r]] -= access_count;
//     }
//   } else {
//     fail = process_memory_access_queue(m_L1C, inst);
//   }
//
//   if (fail != NO_RC_FAIL) {
//     rc_fail = fail;  // keep other fails if this didn't fail.
//     fail_type = C_MEM;
//     if (rc_fail == BK_CONF or rc_fail == COAL_STALL) {
//       m_stats->gpgpu_n_cmem_portconflict++;  // coal stalls aren't really a bank
//                                              // conflict, but this maintains
//                                              // previous behavior.
//     }
//   }
//   return inst.accessq_empty();  // done if empty.
// }
//
// bool ldst_unit::texture_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
//                               mem_stage_access_type &fail_type) {
//   if (inst.empty() || inst.space.get_type() != tex_space) return true;
//   if (inst.active_count() == 0) return true;
//   mem_stage_stall_type fail = process_memory_access_queue(m_L1T, inst);
//   if (fail != NO_RC_FAIL) {
//     rc_fail = fail;  // keep other fails if this didn't fail.
//     fail_type = T_MEM;
//   }
//   return inst.accessq_empty();  // done if empty.
// }

//
// bool ldst_unit::response_buffer_full() const {
//   return m_response_fifo.size() >= m_config->ldst_unit_response_queue_size;
// }
//
// void ldst_unit::fill(mem_fetch *mf) {
//   mf->set_status(
//       IN_SHADER_LDST_RESPONSE_FIFO,
//       m_core->get_gpu()->gpu_sim_cycle + m_core->get_gpu()->gpu_tot_sim_cycle);
//   m_response_fifo.push_back(mf);
// }
//
// void ldst_unit::flush() {
//   // Flush L1D cache
//   m_L1D->flush();
// }
//
// void ldst_unit::invalidate() {
//   // Flush L1D cache
//   m_L1D->invalidate();
// }
