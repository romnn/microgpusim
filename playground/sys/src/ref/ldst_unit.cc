#include "ldst_unit.hpp"

#include <stdio.h>

#include "mem_access.hpp"
#include "l1_cache.hpp"
#include "opndcoll_rfu.hpp"
#include "read_only_cache.hpp"
#include "scoreboard.hpp"
#include "shader_core_mem_fetch_allocator.hpp"
#include "shader_core_stats.hpp"
#include "stats/tool.hpp"
#include "tex_cache.hpp"
#include "trace_gpgpu_sim.hpp"
#include "trace_shader_core_ctx.hpp"

void ldst_unit::print_cache_stats(FILE *fp, unsigned &dl1_accesses,
                                  unsigned &dl1_misses) {
  if (m_L1D) {
    m_L1D->print(fp, dl1_accesses, dl1_misses);
  }
}

void ldst_unit::get_cache_stats(cache_stats &cs) {
  // Adds stats to 'cs' from each cache
  if (m_L1D) cs += m_L1D->get_stats();
  if (m_L1C) cs += m_L1C->get_stats();
  if (m_L1T) cs += m_L1T->get_stats();
}

void ldst_unit::get_L1D_sub_stats(struct cache_sub_stats &css) const {
  if (m_L1D) m_L1D->get_sub_stats(css);
}
void ldst_unit::get_L1C_sub_stats(struct cache_sub_stats &css) const {
  if (m_L1C) m_L1C->get_sub_stats(css);
}
void ldst_unit::get_L1T_sub_stats(struct cache_sub_stats &css) const {
  if (m_L1T) m_L1T->get_sub_stats(css);
}

bool ldst_unit::shared_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
                             mem_stage_access_type &fail_type) {
  if (inst.space.get_type() != shared_space) return true;

  if (inst.active_count() == 0) return true;

  if (inst.has_dispatch_delay()) {
    m_stats->gpgpu_n_shmem_bank_access[m_sid]++;
  }

  bool stall = inst.dispatch_delay();
  if (stall) {
    fail_type = S_MEM;
    rc_fail = BK_CONF;
  } else
    rc_fail = NO_RC_FAIL;
  return !stall;
}

mem_stage_stall_type ldst_unit::process_cache_access(
    cache_t *cache, new_addr_type address, warp_inst_t &inst,
    std::list<cache_event> &events, mem_fetch *mf,
    enum cache_request_status status) {
  mem_stage_stall_type result = NO_RC_FAIL;
  bool write_sent = was_write_sent(events);
  bool read_sent = was_read_sent(events);
  if (write_sent) {
    unsigned inc_ack = (m_config->m_L1D_config.get_mshr_type() == SECTOR_ASSOC)
                           ? (mf->get_data_size() / SECTOR_SIZE)
                           : 1;

    for (unsigned i = 0; i < inc_ack; ++i)
      m_core->inc_store_req(inst.warp_id());
  }
  if (status == HIT) {
    assert(!read_sent);
    inst.accessq_pop_back();
    if (inst.is_load()) {
      for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++)
        if (inst.out[r] > 0) {
          m_pending_writes[inst.warp_id()][inst.out[r]]--;
          logger->trace("warp {} register {}: decrement from {} to {}",
                        inst.warp_id(), inst.out[r],
                        m_pending_writes[inst.warp_id()][inst.out[r]] + 1,
                        m_pending_writes[inst.warp_id()][inst.out[r]]);
        }
    }
    if (!write_sent) delete mf;
  } else if (status == RESERVATION_FAIL) {
    result = BK_CONF;
    assert(!read_sent);
    assert(!write_sent);
    delete mf;
  } else {
    assert(status == MISS || status == HIT_RESERVED);
    // inst.clear_active( access.get_warp_mask() ); // threads in mf writeback
    // when mf returns
    inst.accessq_pop_back();
  }
  if (!inst.accessq_empty() && result == NO_RC_FAIL) result = COAL_STALL;
  return result;
}

mem_stage_stall_type ldst_unit::process_memory_access_queue(cache_t *cache,
                                                            warp_inst_t &inst) {
  mem_stage_stall_type result = NO_RC_FAIL;
  if (inst.accessq_empty()) return result;

  if (!cache->data_port_free()) return DATA_PORT_STALL;

  // const mem_access_t &access = inst.accessq_back();
  mem_fetch *mf = m_mf_allocator->alloc(
      inst, inst.accessq_back(),
      m_core->get_gpu()->gpu_sim_cycle + m_core->get_gpu()->gpu_tot_sim_cycle);
  std::list<cache_event> events;
  enum cache_request_status status = cache->access(
      mf->get_addr(), mf,
      m_core->get_gpu()->gpu_sim_cycle + m_core->get_gpu()->gpu_tot_sim_cycle,
      events);
  return process_cache_access(cache, mf->get_addr(), inst, events, mf, status);
}

mem_stage_stall_type ldst_unit::process_memory_access_queue_l1cache(
    l1_cache *cache, warp_inst_t &inst) {
  mem_stage_stall_type result = NO_RC_FAIL;
  if (inst.accessq_empty()) return result;

  new_addr_type dbg_addr = 0;
  if (m_config->m_L1D_config.l1_latency > 0) {
    for (unsigned j = 0; j < m_config->m_L1D_config.l1_banks;
         j++) {  // We can handle at max l1_banks reqs per cycle

      if (inst.accessq_empty()) return result;

      mem_fetch *mf =
          m_mf_allocator->alloc(inst, inst.accessq_back(),
                                m_core->get_gpu()->gpu_sim_cycle +
                                    m_core->get_gpu()->gpu_tot_sim_cycle);
      dbg_addr = mf->get_addr();
      unsigned bank_id =
          m_config->m_L1D_config.set_bank(mf->get_addr(), logger);
      assert(bank_id < m_config->m_L1D_config.l1_banks);

      logger->trace(
          "computed bank id {} for address {} (access queue=[{}], l1 latency "
          "queue=[{}])",
          bank_id, mf->get_addr(), fmt::join(inst.mem_access_queue(), ", "),
          fmt::join(l1_latency_queue[bank_id], ", "));

      if ((l1_latency_queue[bank_id][m_config->m_L1D_config.l1_latency - 1]) ==
          NULL) {
        l1_latency_queue[bank_id][m_config->m_L1D_config.l1_latency - 1] = mf;

        if (mf->get_inst().is_store()) {
          unsigned inc_ack =
              (m_config->m_L1D_config.get_mshr_type() == SECTOR_ASSOC)
                  ? (mf->get_data_size() / SECTOR_SIZE)
                  : 1;

          for (unsigned i = 0; i < inc_ack; ++i)
            m_core->inc_store_req(inst.warp_id());
        }

        inst.accessq_pop_back();
      } else {
        result = BK_CONF;
        delete mf;
        break;  // do not try again, just break from the loop and try the next
                // cycle
      }
    }
    if (!inst.accessq_empty() && result != BK_CONF) result = COAL_STALL;
    logger->trace(
        "process_memory_access_queue_l1cache stall cond {} for addresss {} "
        "(access queue=[{}])",
        mem_stage_stall_type_str[result], dbg_addr,
        fmt::join(inst.mem_access_queue(), ", "));

    return result;
  } else {
    mem_fetch *mf =
        m_mf_allocator->alloc(inst, inst.accessq_back(),
                              m_core->get_gpu()->gpu_sim_cycle +
                                  m_core->get_gpu()->gpu_tot_sim_cycle);
    std::list<cache_event> events;
    enum cache_request_status status = cache->access(
        mf->get_addr(), mf,
        m_core->get_gpu()->gpu_sim_cycle + m_core->get_gpu()->gpu_tot_sim_cycle,
        events);
    return process_cache_access(cache, mf->get_addr(), inst, events, mf,
                                status);
  }
}

void ldst_unit::L1_latency_queue_cycle() {
  for (unsigned j = 0; j < m_config->m_L1D_config.l1_banks; j++) {
    if ((l1_latency_queue[j][0]) != NULL) {
      mem_fetch *mf_next = l1_latency_queue[j][0];
      std::list<cache_event> events;
      enum cache_request_status status =
          m_L1D->access(mf_next->get_addr(), mf_next,
                        m_core->get_gpu()->gpu_sim_cycle +
                            m_core->get_gpu()->gpu_tot_sim_cycle,
                        events);

      bool write_sent = was_write_sent(events);
      bool read_sent = was_read_sent(events);

      if (status == HIT) {
        assert(!read_sent);
        l1_latency_queue[j][0] = NULL;
        const warp_inst_t &inst = mf_next->get_inst();
        if (inst.is_load()) {
          for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++)
            if (mf_next->get_inst().out[r] > 0) {
              assert(m_pending_writes[inst.warp_id()][inst.out[r]] > 0);
              unsigned before = m_pending_writes[inst.warp_id()][inst.out[r]];
              unsigned still_pending =
                  --m_pending_writes[inst.warp_id()][inst.out[r]];
              unsigned after = m_pending_writes[inst.warp_id()][inst.out[r]];
              assert(before == after + 1);
              logger->trace("warp {} register {}: decrement from {} to {}",
                            inst.warp_id(), inst.out[r],
                            m_pending_writes[inst.warp_id()][inst.out[r]] + 1,
                            m_pending_writes[inst.warp_id()][inst.out[r]]);

              if (!still_pending) {
                m_pending_writes[inst.warp_id()].erase(inst.out[r]);

                logger->trace("l1 latency queue release registers");
                m_scoreboard->releaseRegister(inst.warp_id(), inst.out[r]);
                m_core->warp_inst_complete(inst);
              }
            }
        }

        // For write hit in WB policy
        if (mf_next->get_inst().is_store() && !write_sent) {
          unsigned dec_ack =
              (m_config->m_L1D_config.get_mshr_type() == SECTOR_ASSOC)
                  ? (mf_next->get_data_size() / SECTOR_SIZE)
                  : 1;

          mf_next->set_reply();

          for (unsigned i = 0; i < dec_ack; ++i) m_core->store_ack(mf_next);
        }

        if (!write_sent) delete mf_next;

      } else if (status == RESERVATION_FAIL) {
        assert(!read_sent);
        assert(!write_sent);
      } else {
        assert(status == MISS || status == HIT_RESERVED);
        l1_latency_queue[j][0] = NULL;
        if (m_config->m_L1D_config.get_write_policy() != WRITE_THROUGH &&
            mf_next->get_inst().is_store() &&
            (m_config->m_L1D_config.get_write_allocate_policy() ==
                 FETCH_ON_WRITE ||
             m_config->m_L1D_config.get_write_allocate_policy() ==
                 LAZY_FETCH_ON_READ) &&
            !was_writeallocate_sent(events)) {
          unsigned dec_ack =
              (m_config->m_L1D_config.get_mshr_type() == SECTOR_ASSOC)
                  ? (mf_next->get_data_size() / SECTOR_SIZE)
                  : 1;
          mf_next->set_reply();
          for (unsigned i = 0; i < dec_ack; ++i) m_core->store_ack(mf_next);
          if (!write_sent && !read_sent) delete mf_next;
        }
      }
    }

    for (unsigned stage = 0; stage < m_config->m_L1D_config.l1_latency - 1;
         ++stage)
      if (l1_latency_queue[j][stage] == NULL) {
        l1_latency_queue[j][stage] = l1_latency_queue[j][stage + 1];
        l1_latency_queue[j][stage + 1] = NULL;
      }
  }
}

bool ldst_unit::constant_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
                               mem_stage_access_type &fail_type) {
  if (inst.empty() || ((inst.space.get_type() != const_space) &&
                       (inst.space.get_type() != param_space_kernel)))
    return true;
  if (inst.active_count() == 0) return true;

  assert(0 && "constant cycle");

  mem_stage_stall_type fail;
  if (m_config->perfect_inst_const_cache) {
    fail = NO_RC_FAIL;
    unsigned access_count = inst.accessq_count();
    while (inst.accessq_count() > 0) inst.accessq_pop_back();
    if (inst.is_load()) {
      for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++)
        if (inst.out[r] > 0) {
          m_pending_writes[inst.warp_id()][inst.out[r]] -= access_count;
          logger->trace(
              "warp {} register {}: decrement from {} to {}", inst.warp_id(),
              inst.out[r],
              m_pending_writes[inst.warp_id()][inst.out[r]] + access_count,
              m_pending_writes[inst.warp_id()][inst.out[r]]);
        }
    }
  } else {
    fail = process_memory_access_queue(m_L1C, inst);
  }

  if (fail != NO_RC_FAIL) {
    rc_fail = fail;  // keep other fails if this didn't fail.
    fail_type = C_MEM;
    if (rc_fail == BK_CONF or rc_fail == COAL_STALL) {
      m_stats->gpgpu_n_cmem_portconflict++;  // coal stalls aren't really a bank
                                             // conflict, but this maintains
                                             // previous behavior.
    }
  }
  return inst.accessq_empty();  // done if empty.
}

bool ldst_unit::texture_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
                              mem_stage_access_type &fail_type) {
  if (inst.empty() || inst.space.get_type() != tex_space) return true;
  if (inst.active_count() == 0) return true;
  mem_stage_stall_type fail = process_memory_access_queue(m_L1T, inst);
  if (fail != NO_RC_FAIL) {
    rc_fail = fail;  // keep other fails if this didn't fail.
    fail_type = T_MEM;
  }
  return inst.accessq_empty();  // done if empty.
}

bool ldst_unit::memory_cycle(warp_inst_t &inst,
                             mem_stage_stall_type &stall_reason,
                             mem_stage_access_type &access_type) {
  if (inst.empty() || ((inst.space.get_type() != global_space) &&
                       (inst.space.get_type() != local_space) &&
                       (inst.space.get_type() != param_space_local)))
    return true;
  if (inst.active_count() == 0) return true;
  if (inst.accessq_empty()) return true;

  logger->debug("memory cycle for instruction: {}", inst);

  mem_stage_stall_type stall_cond = NO_RC_FAIL;
  const mem_access_t &access = inst.accessq_back();

  bool bypassL1D = false;
  if (CACHE_GLOBAL == inst.cache_op || (m_L1D == NULL)) {
    bypassL1D = true;
  } else if (inst.space.is_global()) {  // global memory access
    // skip L1 cache if the option is enabled
    if (m_core->get_config()->gmem_skip_L1D && (CACHE_L1 != inst.cache_op))
      bypassL1D = true;
  }
  logger->debug("memory cycle for instruction: {} => access: {} (bypass l1={})",
                inst, access, bypassL1D);

  if (bypassL1D) {
    // bypass L1 cache
    unsigned control_size =
        inst.is_store() ? WRITE_PACKET_SIZE : READ_PACKET_SIZE;
    unsigned size = access.get_size() + control_size;
    // printf("Interconnect:Addr: %x, size=%d\n",access.get_addr(),size);
    if (m_icnt->full(size, inst.is_store() || inst.isatomic())) {
      stall_cond = ICNT_RC_FAIL;
    } else {
      mem_fetch *mf =
          m_mf_allocator->alloc(inst, access,
                                m_core->get_gpu()->gpu_sim_cycle +
                                    m_core->get_gpu()->gpu_tot_sim_cycle);

      m_icnt->push(mf);
      inst.accessq_pop_back();
      // inst.clear_active( access.get_warp_mask() );
      if (inst.is_load()) {
        for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++)
          if (inst.out[r] > 0)
            assert(m_pending_writes[inst.warp_id()][inst.out[r]] > 0);
      } else if (inst.is_store())
        m_core->inc_store_req(inst.warp_id());
    }
  } else {
    assert(CACHE_UNDEFINED != inst.cache_op);
    stall_cond = process_memory_access_queue_l1cache(m_L1D, inst);
  }

  if (!inst.accessq_empty() && stall_cond == NO_RC_FAIL) {
    stall_cond = COAL_STALL;
  }

  logger->debug("memory instruction stall cond: {}",
                mem_stage_stall_type_str[stall_cond]);
  if (stall_cond != NO_RC_FAIL) {
    stall_reason = stall_cond;
    bool iswrite = inst.is_store();
    if (inst.space.is_local())
      access_type = (iswrite) ? L_MEM_ST : L_MEM_LD;
    else
      access_type = (iswrite) ? G_MEM_ST : G_MEM_LD;
  }
  return inst.accessq_empty();
}

bool ldst_unit::response_buffer_full() const {
  return m_response_fifo.size() >= m_config->ldst_unit_response_queue_size;
}

void ldst_unit::fill(mem_fetch *mf) {
  mf->set_status(
      IN_SHADER_LDST_RESPONSE_FIFO,
      m_core->get_gpu()->gpu_sim_cycle + m_core->get_gpu()->gpu_tot_sim_cycle);
  m_response_fifo.push_back(mf);
}

void ldst_unit::flush() {
  // Flush L1D cache
  m_L1D->flush();
}

void ldst_unit::invalidate() {
  // Flush L1D cache
  m_L1D->invalidate();
}

ldst_unit::ldst_unit(mem_fetch_interface *icnt,
                     shader_core_mem_fetch_allocator *mf_allocator,
                     trace_shader_core_ctx *core,
                     opndcoll_rfu_t *operand_collector, Scoreboard *scoreboard,
                     const shader_core_config *config,
                     const memory_config *mem_config, shader_core_stats *stats,
                     unsigned sid, unsigned tpc)
    : pipelined_simd_unit(NULL, config, config->smem_latency, core, 0),
      m_next_wb(config) {
  assert(config->smem_latency > 1);
  init(icnt, mf_allocator, core, operand_collector, scoreboard, config,
       mem_config, stats, sid, tpc);
  if (!m_config->m_L1D_config.disabled()) {
    char L1D_name[1024];
    snprintf(L1D_name, 1024, "L1D_%03d", m_sid);
    m_L1D = new l1_cache(L1D_name, m_config->m_L1D_config, m_sid,
                         get_shader_normal_cache_id(), m_icnt, m_mf_allocator,
                         IN_L1D_MISS_QUEUE, logger, core->get_gpu());

    l1_latency_queue.resize(m_config->m_L1D_config.l1_banks);
    assert(m_config->m_L1D_config.l1_latency > 0);

    for (unsigned j = 0; j < m_config->m_L1D_config.l1_banks; j++)
      l1_latency_queue[j].resize(m_config->m_L1D_config.l1_latency,
                                 (mem_fetch *)NULL);
  }
  m_name = "LdstUnit";
}

// ldst_unit::ldst_unit(mem_fetch_interface *icnt,
//                      shader_core_mem_fetch_allocator *mf_allocator,
//                      trace_shader_core_ctx *core,
//                      opndcoll_rfu_t *operand_collector, Scoreboard
//                      *scoreboard, const shader_core_config *config, const
//                      memory_config *mem_config, shader_core_stats *stats,
//                      unsigned sid, unsigned tpc, l1_cache *new_l1d_cache)
//     : pipelined_simd_unit(NULL, config, 3, core, 0),
//       m_L1D(new_l1d_cache),
//       m_next_wb(config) {
//   init(icnt, mf_allocator, core, operand_collector, scoreboard, config,
//        mem_config, stats, sid, tpc);
// }

void ldst_unit::issue(register_set &reg_set) {
  warp_inst_t *inst = *(reg_set.get_ready());

  // record how many pending register writes/memory accesses there are for this
  // instruction
  assert(inst->empty() == false);
  if (inst->is_load() and inst->space.get_type() != shared_space) {
    unsigned warp_id = inst->warp_id();
    unsigned n_accesses = inst->accessq_count();
    for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
      unsigned reg_id = inst->out[r];
      if (reg_id > 0) {
        m_pending_writes[warp_id][reg_id] += n_accesses;
        logger->trace("warp {} register {}: increment from {} to {}", warp_id,
                      reg_id, m_pending_writes[warp_id][reg_id] - n_accesses,
                      m_pending_writes[warp_id][reg_id]);
      }
    }
  }

  inst->op_pipe = MEM__OP;
  // stat collection
  m_core->mem_instruction_stats(*inst);
  m_core->incmem_stat(m_core->get_config()->warp_size, 1);
  pipelined_simd_unit::issue(reg_set);
}

static const char *writeback_client_str[] = {
    "SharedMemory", "L1T", "L1C", "GlobalLocal", "L1D",
};

void ldst_unit::writeback() {
  logger->debug(
      "load store unit: cycle {} writeback (arb={}, writeback clients={})",
      m_core->get_gpu()->gpu_sim_cycle, m_writeback_arb,
      m_num_writeback_clients);

  // process next instruction that is going to writeback
  if (!m_next_wb.empty()) {
    logger->trace("load store unit: cycle {} writeback: next_wb={}",
                  m_core->get_gpu()->gpu_sim_cycle, m_next_wb);

    if (m_operand_collector->writeback(m_next_wb)) {
      bool insn_completed = false;
      for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
        if (m_next_wb.out[r] > 0) {
          if (m_next_wb.space.get_type() != shared_space) {
            assert(m_pending_writes[m_next_wb.warp_id()][m_next_wb.out[r]] > 0);
            unsigned still_pending =
                --m_pending_writes[m_next_wb.warp_id()][m_next_wb.out[r]];
            logger->trace(
                "load store unit: cycle {} writeback: next_wb={}, dest "
                "register {}: pending writes={}",
                m_core->get_gpu()->gpu_sim_cycle, m_next_wb, m_next_wb.out[r],
                still_pending);

            if (!still_pending) {
              m_pending_writes[m_next_wb.warp_id()].erase(m_next_wb.out[r]);
              m_scoreboard->releaseRegister(m_next_wb.warp_id(),
                                            m_next_wb.out[r]);
              insn_completed = true;
            }
          } else {  // shared
            m_scoreboard->releaseRegister(m_next_wb.warp_id(),
                                          m_next_wb.out[r]);
            insn_completed = true;
          }
        }
      }
      if (insn_completed) {
        m_core->warp_inst_complete(m_next_wb);
      }
      m_next_wb.clear();
      m_last_inst_gpu_sim_cycle = m_core->get_gpu()->gpu_sim_cycle;
      m_last_inst_gpu_tot_sim_cycle = m_core->get_gpu()->gpu_tot_sim_cycle;
    }
  } else {
    logger->debug("load store unit: cycle {} writeback: next_wb=NULL (arb={})",
                  m_core->get_gpu()->gpu_sim_cycle, m_writeback_arb);
  }

  unsigned serviced_client = -1;
  for (unsigned c = 0; m_next_wb.empty() && (c < m_num_writeback_clients);
       c++) {
    unsigned next_client = (c + m_writeback_arb) % m_num_writeback_clients;

    logger->trace("checking writeback client {}",
                  writeback_client_str[next_client]);
    switch (next_client) {
      case 0:  // shared memory
        if (!m_pipeline_reg[0]->empty()) {
          m_next_wb = *m_pipeline_reg[0];
          if (m_next_wb.isatomic()) {
            m_next_wb.do_atomic();
            m_core->decrement_atomic_count(m_next_wb.warp_id(),
                                           m_next_wb.active_count());
          }
          m_core->dec_inst_in_pipeline(m_pipeline_reg[0]->warp_id());
          m_pipeline_reg[0]->clear();
          serviced_client = next_client;
        }
        break;
      case 1:  // texture response
        if (m_L1T->access_ready()) {
          mem_fetch *mf = m_L1T->next_access();
          m_next_wb = mf->get_inst();
          delete mf;
          serviced_client = next_client;
        }
        break;
      case 2:  // const cache response
        if (m_L1C->access_ready()) {
          mem_fetch *mf = m_L1C->next_access();
          m_next_wb = mf->get_inst();
          delete mf;
          serviced_client = next_client;
        }
        break;
      case 3:  // global/local
        if (m_next_global) {
          m_next_wb = m_next_global->get_inst();
          logger->debug("has global {}", m_next_wb);
          if (m_next_global->isatomic()) {
            m_core->decrement_atomic_count(
                m_next_global->get_wid(),
                m_next_global->get_access_warp_mask().count());
          }
          delete m_next_global;
          m_next_global = NULL;
          serviced_client = next_client;
        }
        break;
      case 4:
        assert("have l1d" && m_L1D != NULL);
        if (m_L1D && m_L1D->access_ready()) {
          mem_fetch *mf = m_L1D->next_access();
          m_next_wb = mf->get_inst();
          delete mf;
          serviced_client = next_client;
        }
        break;
      default:
        abort();
    }
  }
  // update arbitration priority only if:
  // 1. the writeback buffer was available
  // 2. a client was serviced
  if (serviced_client != (unsigned)-1) {
    logger->debug(
        "load store unit writeback serviced client {} ({}) => next "
        "writeback={}",
        writeback_client_str[serviced_client], serviced_client, m_next_wb);
    m_writeback_arb = (serviced_client + 1) % m_num_writeback_clients;
  }
}

unsigned ldst_unit::clock_multiplier() const {
  // to model multiple read port, we give multiple cycles for the memory units
  if (m_config->mem_unit_ports)
    return m_config->mem_unit_ports;
  else
    return m_config->mem_warp_parts;
}

void ldst_unit::cycle() {
  logger->debug(
      "LdstUnit cycle() \tpipeline=[{}] ({}/{} active) \tresponse "
      "fifo=[{}]",
      fmt::join(m_pipeline_reg, m_pipeline_reg + m_pipeline_depth, ","),
      active_insts_in_pipeline, m_pipeline_depth,
      fmt::join(m_response_fifo, ", "));

  writeback();

  for (unsigned stage = 0; (stage + 1) < m_pipeline_depth; stage++) {
    if (m_pipeline_reg[stage]->empty() && !m_pipeline_reg[stage + 1]->empty()) {
      // std::stringstream msg;
      // msg << "load store unit: move warp from stage " << stage + 1 << " to "
      //     << stage;
      move_warp(m_pipeline_reg[stage],
                m_pipeline_reg[stage + 1]);  // , msg.str(), logger);
    } else {
      logger->trace("LdstUnit: skip moving {} to {}", stage + 1, stage);
    }
  }

  if (!m_response_fifo.empty()) {
    mem_fetch *mf = m_response_fifo.front();
    if (mf->get_access_type() == TEXTURE_ACC_R) {
      if (m_L1T->fill_port_free()) {
        m_L1T->fill(mf, m_core->get_gpu()->gpu_sim_cycle +
                            m_core->get_gpu()->gpu_tot_sim_cycle);
        m_response_fifo.pop_front();
      }
    } else if (mf->get_access_type() == CONST_ACC_R) {
      if (m_L1C->fill_port_free()) {
        mf->set_status(IN_SHADER_FETCHED,
                       m_core->get_gpu()->gpu_sim_cycle +
                           m_core->get_gpu()->gpu_tot_sim_cycle);
        m_L1C->fill(mf, m_core->get_gpu()->gpu_sim_cycle +
                            m_core->get_gpu()->gpu_tot_sim_cycle);
        m_response_fifo.pop_front();
      }
    } else {
      if (mf->get_type() == WRITE_ACK ||
          (m_config->gpgpu_perfect_mem && mf->get_is_write())) {
        m_core->store_ack(mf);
        m_response_fifo.pop_front();
        delete mf;
      } else {
        assert(!mf->get_is_write());  // L1 cache is write evict, allocate line
                                      // on load miss only

        bool bypassL1D = false;
        if (CACHE_GLOBAL == mf->get_inst().cache_op || (m_L1D == NULL)) {
          bypassL1D = true;
        } else if (mf->get_access_type() == GLOBAL_ACC_R ||
                   mf->get_access_type() ==
                       GLOBAL_ACC_W) {  // global memory access
          if (m_core->get_config()->gmem_skip_L1D) bypassL1D = true;
        }
        if (bypassL1D) {
          if (m_next_global == NULL) {
            mf->set_status(IN_SHADER_FETCHED,
                           m_core->get_gpu()->gpu_sim_cycle +
                               m_core->get_gpu()->gpu_tot_sim_cycle);
            m_response_fifo.pop_front();
            m_next_global = mf;
          }
        } else {
          if (m_L1D->fill_port_free()) {
            m_L1D->fill(mf, m_core->get_gpu()->gpu_sim_cycle +
                                m_core->get_gpu()->gpu_tot_sim_cycle);
            m_response_fifo.pop_front();
          }
        }
      }
    }
  }

  m_L1T->cycle();
  m_L1C->cycle();
  if (m_L1D) {
    m_L1D->cycle();
    if (m_config->m_L1D_config.l1_latency > 0) L1_latency_queue_cycle();
  }

  warp_inst_t &pipe_reg = *m_dispatch_reg;
  enum mem_stage_stall_type rc_fail = NO_RC_FAIL;
  mem_stage_access_type type;
  bool done = true;
  done &= shared_cycle(pipe_reg, rc_fail, type);
  done &= constant_cycle(pipe_reg, rc_fail, type);
  done &= texture_cycle(pipe_reg, rc_fail, type);
  done &= memory_cycle(pipe_reg, rc_fail, type);
  m_mem_rc = rc_fail;

  if (!done) {  // log stall types and return
    assert(rc_fail != NO_RC_FAIL);
    m_stats->gpgpu_n_stall_shd_mem++;
    m_stats->gpu_stall_shd_mem_breakdown[type][rc_fail]++;
    return;
  }

  if (!pipe_reg.empty()) {
    // ldst unit got instr from dispatch reg
    unsigned warp_id = pipe_reg.warp_id();
    if (pipe_reg.is_load()) {
      if (pipe_reg.space.get_type() == shared_space) {
        if (m_pipeline_reg[m_config->smem_latency - 1]->empty()) {
          // new shared memory request

          unsigned pipe_slot_idx = m_config->smem_latency - 1;
          // std::stringstream msg;
          // msg << "load store unit: move warp from dispatch register to "
          //        "pipeline["
          //     << pipe_slot_idx << "]",
          move_warp(m_pipeline_reg[m_config->smem_latency - 1],
                    m_dispatch_reg);  // , msg.str(), logger);
          m_dispatch_reg->clear();
        }
      } else {
        // if( pipe_reg.active_count() > 0 ) {
        //    if( !m_operand_collector->writeback(pipe_reg) )
        //        return;
        //}

        bool pending_requests = false;
        for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
          unsigned reg_id = pipe_reg.out[r];
          if (reg_id > 0) {
            if (m_pending_writes[warp_id].find(reg_id) !=
                m_pending_writes[warp_id].end()) {
              if (m_pending_writes[warp_id][reg_id] > 0) {
                pending_requests = true;
                break;
              } else {
                // this instruction is done already
                m_pending_writes[warp_id].erase(reg_id);
              }
            }
          }
        }
        if (!pending_requests) {
          logger->trace("ldst unit: load instruction {} completed",
                        warp_instr_ptr(m_dispatch_reg));
          m_core->warp_inst_complete(*m_dispatch_reg);
          m_scoreboard->releaseRegisters(m_dispatch_reg);
        }
        m_core->dec_inst_in_pipeline(warp_id);
        m_dispatch_reg->clear();
      }
    } else {
      // stores exit pipeline here
      logger->trace("ldst unit: store instruction {} completed",
                    warp_instr_ptr(m_dispatch_reg));
      m_core->dec_inst_in_pipeline(warp_id);
      m_core->warp_inst_complete(*m_dispatch_reg);
      m_dispatch_reg->clear();
    }
  }
}

// void ldst_unit::print(FILE *fout) const {
//   fprintf(fout, "LD/ST unit  = ");
//   m_dispatch_reg->print(fout);
//   if (m_mem_rc != NO_RC_FAIL) {
//     fprintf(fout, "              LD/ST stall condition: ");
//     switch (m_mem_rc) {
//       case BK_CONF:
//         fprintf(fout, "BK_CONF");
//         break;
//       case MSHR_RC_FAIL:
//         fprintf(fout, "MSHR_RC_FAIL");
//         break;
//       case ICNT_RC_FAIL:
//         fprintf(fout, "ICNT_RC_FAIL");
//         break;
//       case COAL_STALL:
//         fprintf(fout, "COAL_STALL");
//         break;
//       case WB_ICNT_RC_FAIL:
//         fprintf(fout, "WB_ICNT_RC_FAIL");
//         break;
//       case WB_CACHE_RSRV_FAIL:
//         fprintf(fout, "WB_CACHE_RSRV_FAIL");
//         break;
//       case N_MEM_STAGE_STALL_TYPE:
//         fprintf(fout, "N_MEM_STAGE_STALL_TYPE");
//         break;
//       default:
//         abort();
//     }
//     fprintf(fout, "\n");
//   }
//   fprintf(fout, "LD/ST wb    = ");
//   m_next_wb.print(fout);
//   fprintf(
//       fout,
//       "Last LD/ST writeback @ %llu + %llu
//       (gpu_sim_cycle+gpu_tot_sim_cycle)\n", m_last_inst_gpu_sim_cycle,
//       m_last_inst_gpu_tot_sim_cycle);
//   fprintf(fout, "Pending register writes:\n");
//   std::map<unsigned /*warp_id*/,
//            std::map<unsigned /*regnum*/, unsigned /*count*/>>::const_iterator
//            w;
//   for (w = m_pending_writes.begin(); w != m_pending_writes.end(); w++) {
//     unsigned warp_id = w->first;
//     const std::map<unsigned /*regnum*/, unsigned /*count*/> &warp_info =
//         w->second;
//     if (warp_info.empty()) continue;
//     fprintf(fout, "  w%2u : ", warp_id);
//     std::map<unsigned /*regnum*/, unsigned /*count*/>::const_iterator r;
//     for (r = warp_info.begin(); r != warp_info.end(); ++r) {
//       fprintf(fout, "  %u(%u)", r->first, r->second);
//     }
//     fprintf(fout, "\n");
//   }
//   m_L1C->display_state(fout);
//   m_L1T->display_state(fout);
//   if (!m_config->m_L1D_config.disabled()) m_L1D->display_state(fout);
//   fprintf(fout, "LD/ST response FIFO (occupancy = %zu):\n",
//           m_response_fifo.size());
//   for (std::list<mem_fetch *>::const_iterator i = m_response_fifo.begin();
//        i != m_response_fifo.end(); i++) {
//     const mem_fetch *mf = *i;
//     (std::ostream &)fout << mf;
//     // mf->print(fout);
//   }
// }

void ldst_unit::init(mem_fetch_interface *icnt,
                     shader_core_mem_fetch_allocator *mf_allocator,
                     trace_shader_core_ctx *core,
                     opndcoll_rfu_t *operand_collector, Scoreboard *scoreboard,
                     const shader_core_config *config,
                     const memory_config *mem_config, shader_core_stats *stats,
                     unsigned sid, unsigned tpc) {
  m_memory_config = mem_config;
  m_icnt = icnt;
  m_mf_allocator = mf_allocator;
  m_core = core;
  m_operand_collector = operand_collector;
  m_scoreboard = scoreboard;
  m_stats = stats;
  m_sid = sid;
  m_tpc = tpc;
#define STRSIZE 1024
  char L1T_name[STRSIZE];
  char L1C_name[STRSIZE];
  snprintf(L1T_name, STRSIZE, "L1T_%03d", m_sid);
  snprintf(L1C_name, STRSIZE, "L1C_%03d", m_sid);
  bool accelsim_compat_mode = core->get_gpu()->gpgpu_ctx->accelsim_compat_mode;
  m_L1T = new tex_cache(L1T_name, m_config->m_L1T_config, m_sid,
                        get_shader_texture_cache_id(), icnt, IN_L1T_MISS_QUEUE,
                        IN_SHADER_L1T_ROB, accelsim_compat_mode, logger);
  m_L1C = new read_only_cache(L1C_name, m_config->m_L1C_config, m_sid,
                              get_shader_constant_cache_id(), icnt,
                              IN_L1C_MISS_QUEUE, accelsim_compat_mode, logger);
  m_L1D = NULL;
  m_mem_rc = NO_RC_FAIL;
  m_num_writeback_clients =
      5;  // = shared memory, global/local (uncached), L1D, L1T, L1C
  m_writeback_arb = 0;
  m_next_global = NULL;
  m_last_inst_gpu_sim_cycle = 0;
  m_last_inst_gpu_tot_sim_cycle = 0;
}

void ldst_unit::active_lanes_in_pipeline() {
  unsigned active_count = pipelined_simd_unit::get_active_lanes_in_pipeline();
  assert(active_count <= m_core->get_config()->warp_size);
  m_core->incfumemactivelanes_stat(active_count);
}
