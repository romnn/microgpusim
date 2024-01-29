#include "trace_shader_core_ctx.hpp"

#include <chrono>
#include <csignal>

#include "fmt/format.h"
#include "gpgpu_context.hpp"
#include "timeit.hpp"
#include "concrete_scheduler.hpp"
#include "cuda_sim.hpp"
#include "dp_unit.hpp"
#include "hal.hpp"
#include "io.hpp"
#include "gto_scheduler.hpp"
#include "int_unit.hpp"
#include "ldst_unit.hpp"
#include "lrr_scheduler.hpp"
#include "oldest_scheduler.hpp"
#include "perfect_memory_interface.hpp"
#include "ptx_thread_info.hpp"
#include "read_only_cache.hpp"
#include "rrr_scheduler.hpp"
#include "scoreboard.hpp"
#include "shader_core_config.hpp"
#include "shader_core_mem_fetch_allocator.hpp"
#include "shader_memory_interface.hpp"
#include "shader_trace.hpp"
#include "sp_unit.hpp"
#include "special_function_unit.hpp"
#include "specialized_unit.hpp"
#include "stats/tool.hpp"
#include "swl_scheduler.hpp"
#include "tensor_core.hpp"
#include "thread_ctx.hpp"
#include "trace_gpgpu_sim.hpp"
#include "trace_kernel_info.hpp"
#include "trace_shd_warp.hpp"
#include "warp_instr.hpp"
#include "trace_warp_inst.hpp"
#include "two_level_active_scheduler.hpp"

void trace_shader_core_ctx::create_shd_warp() {
  m_warp.resize(m_config->max_warps_per_shader);
  for (unsigned k = 0; k < m_config->max_warps_per_shader; ++k) {
    m_warp[k] = new trace_shd_warp_t(this, m_config->warp_size);
  }
}

void trace_shader_core_ctx::get_pdom_stack_top_info(unsigned warp_id,
                                                    const warp_inst_t *pI,
                                                    unsigned *pc,
                                                    unsigned *rpc) {
  // In trace-driven mode, we assume no control hazard
  *pc = pI->pc;
  *rpc = pI->pc;
}

const active_mask_t &trace_shader_core_ctx::get_active_mask(
    unsigned warp_id, const warp_inst_t *pI) {
  // For Trace-driven, the active mask already set in traces, so
  // just read it from the inst
  return pI->get_active_mask();
}

unsigned trace_shader_core_ctx::sim_init_thread(
    trace_kernel_info_t &kernel, ptx_thread_info **thread_info, int sid,
    unsigned tid, unsigned threads_left, unsigned num_threads,
    trace_shader_core_ctx *core, unsigned hw_cta_id, unsigned hw_warp_id,
    trace_gpgpu_sim *gpu) {
  if (kernel.no_more_ctas_to_run()) {
    return 0;  // finished!
  }

  if (kernel.more_threads_in_cta()) {
    kernel.increment_thread_id();
  }

  if (!kernel.more_threads_in_cta()) kernel.increment_cta_id();

  return 1;
}

// return the next pc of a thread
address_type trace_shader_core_ctx::next_pc(int tid) const {
  if (tid == -1) return -1;
  ptx_thread_info *the_thread = m_thread[tid];
  if (the_thread == NULL) return -1;
  return the_thread
      ->get_pc();  // PC should already be updatd to next PC at this point
                   // (was set in shader_decode() last time thread ran)
}

void trace_shader_core_ctx::init_warps(unsigned cta_id, unsigned start_thread,
                                       unsigned end_thread, unsigned ctaid,
                                       int cta_size,
                                       trace_kernel_info_t &kernel) {
  address_type start_pc = next_pc(start_thread);
  // unsigned kernel_id = kernel.get_uid();

  unsigned start_warp = start_thread / m_config->warp_size;
  unsigned end_warp = end_thread / m_config->warp_size +
                      ((end_thread % m_config->warp_size) ? 1 : 0);

  // unsigned start_warp = start_thread / m_config->warp_size;
  // unsigned end_warp = end_thread / m_config->warp_size +
  //                     ((end_thread % m_config->warp_size) ? 1 : 0);

  if (m_config->model == POST_DOMINATOR) {
    // post dominator is the default
    // assert(0 && "post dominator PTX model");

    // unsigned warp_per_cta = cta_size / m_config->warp_size;
    for (unsigned i = start_warp; i < end_warp; ++i) {
      unsigned n_active = 0;
      simt_mask_t active_threads;
      for (unsigned t = 0; t < m_config->warp_size; t++) {
        unsigned hwtid = i * m_config->warp_size + t;
        if (hwtid < end_thread) {
          n_active++;
          assert(!m_active_threads.test(hwtid));
          m_active_threads.set(hwtid);
          active_threads.set(t);
        }
      }
      // m_simt_stack[i]->launch(start_pc, active_threads);

      // REMOVE: resume
      // if (m_gpu->resume_option == 1 && kernel_id == m_gpu->resume_kernel
      // &&
      //     ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t)
      //     {
      //   char fname[2048];
      //   snprintf(fname, 2048, "checkpoint_files/warp_%d_%d_simt.txt",
      //            i % warp_per_cta, ctaid);
      //   unsigned pc, rpc;
      //   m_simt_stack[i]->resume(fname);
      //   m_simt_stack[i]->get_pdom_stack_top_info(&pc, &rpc);
      //   for (unsigned t = 0; t < m_config->warp_size; t++) {
      //     if (m_thread != NULL) {
      //       m_thread[i * m_config->warp_size + t]->set_npc(pc);
      //       m_thread[i * m_config->warp_size + t]->update_pc();
      //     }
      //   }
      //   start_pc = pc;
      // }

      m_warp[i]->init(start_pc, cta_id, i, active_threads, m_dynamic_warp_id);
      ++m_dynamic_warp_id;
      m_not_completed += n_active;
      ++m_active_warps;
    }
  }

  m_gpu->logger->debug("initialized warps {}..{} of {} warps", start_warp,
                       end_warp, m_warp.size());

  // now init traces
  init_traces(start_warp, end_warp, kernel);
}

const warp_inst_t *trace_shader_core_ctx::get_next_inst(unsigned warp_id,
                                                        address_type pc) {
  // read the inst from the traces
  trace_shd_warp_t *m_trace_warp =
      static_cast<trace_shd_warp_t *>(m_warp[warp_id]);
  return m_trace_warp->get_next_trace_inst();
}

void trace_shader_core_ctx::updateSIMTStack(unsigned warpId,
                                            warp_inst_t *inst) {
  // No SIMT-stack in trace-driven  mode
}

void trace_shader_core_ctx::init_traces(unsigned start_warp, unsigned end_warp,
                                        trace_kernel_info_t &kernel) {
  std::vector<std::vector<inst_trace_t> *> threadblock_traces;
  for (unsigned i = start_warp; i < end_warp; ++i) {
    trace_shd_warp_t *m_trace_warp = static_cast<trace_shd_warp_t *>(m_warp[i]);
    m_trace_warp->clear();
    threadblock_traces.push_back(&(m_trace_warp->warp_traces));
  }
  trace_kernel_info_t &trace_kernel =
      static_cast<trace_kernel_info_t &>(kernel);
  trace_kernel.get_next_threadblock_traces(threadblock_traces);

  if (m_gpu->gpgpu_ctx->accelsim_compat_mode) {
    // printf("====== INIT TRACES %d-%d \n", start_warp, end_warp);
    for (unsigned i = start_warp; i < end_warp; ++i) {
      trace_shd_warp_t *m_trace_warp =
          static_cast<trace_shd_warp_t *>(m_warp[i]);
      const std::vector<inst_trace_t> &instructions = m_trace_warp->warp_traces;
      std::vector<inst_trace_t>::const_iterator iter;
      // printf("====== WARP %d \n", i);
      for (iter = instructions.begin(); iter != instructions.end(); iter++) {
        // printf("\t => instruction %s pc = %d \n", iter->opcode.c_str(),
        //        iter->m_pc);
      }
    }
    // printf("====== INIT TRACES %d-%d DONE \n", start_warp, end_warp);
  }

  // set the pc from the traces and ignore the functional model
  for (unsigned i = start_warp; i < end_warp; ++i) {
    trace_shd_warp_t *m_trace_warp = static_cast<trace_shd_warp_t *>(m_warp[i]);
    // m_trace_warp->set_next_pc(m_trace_warp->get_start_trace_pc());
    m_trace_warp->set_kernel(&trace_kernel);
  }

  logger->debug("initialized traces for warps {}..{} of {} warps\n", start_warp,
                end_warp, m_warp.size());
}

void trace_shader_core_ctx::checkExecutionStatusAndUpdate(warp_inst_t &inst,
                                                          unsigned t,
                                                          unsigned tid) {
  if (inst.isatomic()) m_warp[inst.warp_id()]->inc_n_atomic();

  if (inst.space.is_local() && (inst.is_load() || inst.is_store())) {
    new_addr_type localaddrs[MAX_ACCESSES_PER_INSN_PER_THREAD];
    unsigned num_addrs;
    num_addrs = translate_local_memaddr(
        inst.get_addr(t), tid,
        m_config->n_simt_clusters * m_config->n_simt_cores_per_cluster,
        inst.data_size, (new_addr_type *)localaddrs);
    inst.set_addr(t, (new_addr_type *)localaddrs, num_addrs);
  }

  if (inst.op == EXIT_OPS) {
    m_warp[inst.warp_id()]->set_completed(t);
  }
}

void trace_shader_core_ctx::func_exec_inst(warp_inst_t &inst) {
  for (unsigned t = 0; t < m_warp_size; t++) {
    if (inst.active(t)) {
      unsigned warpId = inst.warp_id();
      unsigned tid = m_warp_size * warpId + t;

      // virtual function
      checkExecutionStatusAndUpdate(inst, t, tid);
    }
  }

  // here, we generate memory acessess and set the status if thread (done?)
  if (inst.is_load() || inst.is_store()) {
    inst.generate_mem_accesses(logger);
  }

  // update allocation start addresses here
  std::list<mem_access_t>::iterator access_iter;
  for (access_iter = inst.m_accessq.begin();
       access_iter != inst.m_accessq.end(); ++access_iter) {
    mem_access_t &access = *access_iter;

    std::set<Allocation>::const_iterator alloc_iter;
    for (alloc_iter = m_gpu->m_allocations.begin();
         alloc_iter != m_gpu->m_allocations.end(); ++alloc_iter) {
      const Allocation &alloc = *alloc_iter;
      if (alloc.contains(access.get_addr())) {
        access.set_alloc_start_addr(alloc.start_addr);
        access.set_alloc_id(alloc.id);
        break;
      }
    }
  }

  if (inst.is_load() || inst.is_store()) {
    logger->trace("generated mem accesses: [{}]",
                  fmt::join(inst.m_accessq, ","));
  }

  trace_shd_warp_t *m_trace_warp =
      static_cast<trace_shd_warp_t *>(m_warp[inst.warp_id()]);
  assert(inst.warp_id() == m_trace_warp->get_warp_id());

  logger->debug("warp={}", m_trace_warp->get_warp_id());

  if (m_trace_warp->trace_done() && m_trace_warp->functional_done()) {
    logger->debug("completed fully");
    m_trace_warp->ibuffer_flush();
    m_barriers.warp_exit(inst.warp_id());
  } else {
    if (m_trace_warp->trace_done()) {
      logger->debug(" completed trace");
    } else {
      logger->debug(" executed pc={}", m_trace_warp->get_pc());
    }
  }
  logger->debug("\t(trace done={} ({}/{}) functional done={})",
                m_trace_warp->trace_done(), m_trace_warp->trace_pc,
                m_trace_warp->warp_traces.size(),
                m_trace_warp->functional_done());
}

void trace_shader_core_ctx::issue_warp(register_set &pipe_reg_set,
                                       const warp_inst_t *next_inst,
                                       const active_mask_t &active_mask,
                                       unsigned warp_id, unsigned sch_id) {
  warp_inst_t **pipe_reg =
      pipe_reg_set.get_free(m_config->sub_core_model, sch_id);
  assert(pipe_reg);

  logger->debug(
      "cycle {} issue {} for warp {} by scheduler {} to pipeline[?][?] {}",
      m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, next_inst->display(),
      warp_id, sch_id, pipe_reg_set, (*pipe_reg)->display());

  assert(next_inst->empty());

  // logger->debug("warp={} executed {}", warp_id,
  //               // static_cast<const warp_inst_t *>(next_inst));
  //               static_cast<const warp_inst_t &>(*next_inst));
  // create a copy
  warp_inst_t next_trace_inst_copy = *next_inst;

  m_warp[warp_id]->ibuffer_free();
  assert(next_inst->valid());

  **pipe_reg = next_trace_inst_copy;  // static instruction information

  // this sets all the info for warp inst* in pipe reg
  (*pipe_reg)->issue(active_mask, warp_id,
                     m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle,
                     m_warp[warp_id]->get_dynamic_warp_id(),
                     sch_id);  // dynamic instruction information

  assert(warp_id == (*pipe_reg)->warp_id());

  m_stats->shader_cycle_distro[2 + (*pipe_reg)->active_count()]++;

  logger->debug("warp={} executed {} pc={} instr in pipe={}",
                (*pipe_reg)->warp_id(), next_inst->opcode_str(),
                (*pipe_reg)->pc, m_warp[warp_id]->m_inst_in_pipeline);

  func_exec_inst(**pipe_reg);

  if (next_inst->op == BARRIER_OP) {
    m_warp[warp_id]->store_info_of_last_inst_at_barrier(*pipe_reg);
    m_barriers.warp_reaches_barrier(m_warp[warp_id]->get_cta_id(), warp_id,
                                    const_cast<warp_inst_t *>(next_inst));

  } else if (next_inst->op == MEMORY_BARRIER_OP) {
    m_warp[warp_id]->set_membar();
  }

  updateSIMTStack(warp_id, *pipe_reg);

  // NOTE: need display because the instruction has been emptied
  logger->debug("reserving {} registers ({},{},{},{}) for instr {}: {}",
                (*pipe_reg)->outcount, (*pipe_reg)->out[0], (*pipe_reg)->out[1],
                (*pipe_reg)->out[2], (*pipe_reg)->out[3], next_inst->display(),
                pipe_reg_set);

  m_scoreboard->reserveRegisters(*pipe_reg);
  m_warp[warp_id]->set_next_pc(next_inst->pc + next_inst->isize);

  // delete warp_inst_t class here, it is not required anymore by gpgpu-sim
  // after issue
  // zero out memory
  // std::memset((void *)next_inst, 0, sizeof(const trace_warp_inst_t));
  delete next_inst;
}

void trace_shader_core_ctx::create_front_pipeline() {
  // pipeline_stages is the sum of normal pipeline stages and
  // specialized_unit stages * 2 (for ID and EX)
  unsigned total_pipeline_stages =
      N_PIPELINE_STAGES + m_config->m_specialized_unit.size() * 2;
  m_pipeline_reg.reserve(total_pipeline_stages);
  for (int j = 0; j < N_PIPELINE_STAGES; j++) {
    logger->trace("pipeline stage {} has width {}",
                  pipeline_stage_name_t_str[j], m_config->pipe_widths[j]);
    m_pipeline_reg.push_back(register_set(
        m_config->pipe_widths[j], pipeline_stage_name_t_str[j], j, logger));
  }
  for (unsigned j = 0; j < m_config->m_specialized_unit.size(); j++) {
    m_pipeline_reg.push_back(
        register_set(m_config->m_specialized_unit[j].id_oc_spec_reg_width,
                     m_config->m_specialized_unit[j].name, j, logger));
    m_config->m_specialized_unit[j].ID_OC_SPEC_ID = m_pipeline_reg.size() - 1;
    m_specilized_dispatch_reg.push_back(
        &m_pipeline_reg[m_pipeline_reg.size() - 1]);
  }
  for (unsigned j = 0; j < m_config->m_specialized_unit.size(); j++) {
    m_pipeline_reg.push_back(
        register_set(m_config->m_specialized_unit[j].oc_ex_spec_reg_width,
                     m_config->m_specialized_unit[j].name, j, logger));
    m_config->m_specialized_unit[j].OC_EX_SPEC_ID = m_pipeline_reg.size() - 1;
  }

  if (m_config->sub_core_model) {
    // in subcore model, each scheduler should has its own issue register,
    // so ensure num scheduler = reg width
    assert(m_config->gpgpu_num_sched_per_core ==
           m_pipeline_reg[ID_OC_SP].get_size());
    assert(m_config->gpgpu_num_sched_per_core ==
           m_pipeline_reg[ID_OC_SFU].get_size());
    assert(m_config->gpgpu_num_sched_per_core ==
           m_pipeline_reg[ID_OC_MEM].get_size());
    if (m_config->gpgpu_tensor_core_avail)
      assert(m_config->gpgpu_num_sched_per_core ==
             m_pipeline_reg[ID_OC_TENSOR_CORE].get_size());
    if (m_config->gpgpu_num_dp_units > 0)
      assert(m_config->gpgpu_num_sched_per_core ==
             m_pipeline_reg[ID_OC_DP].get_size());
    if (m_config->gpgpu_num_int_units > 0)
      assert(m_config->gpgpu_num_sched_per_core ==
             m_pipeline_reg[ID_OC_INT].get_size());
    for (unsigned j = 0; j < m_config->m_specialized_unit.size(); j++) {
      if (m_config->m_specialized_unit[j].num_units > 0)
        assert(m_config->gpgpu_num_sched_per_core ==
               m_config->m_specialized_unit[j].id_oc_spec_reg_width);
    }
  }

  m_threadState = (thread_ctx_t *)calloc(sizeof(thread_ctx_t),
                                         m_config->n_thread_per_shader);

  m_not_completed = 0;
  m_active_threads.reset();
  m_n_active_cta = 0;
  for (unsigned i = 0; i < MAX_CTA_PER_SHADER; i++) m_cta_status[i] = 0;
  for (unsigned i = 0; i < m_config->n_thread_per_shader; i++) {
    m_thread[i] = NULL;
    m_threadState[i].m_cta_id = -1;
    m_threadState[i].m_active = false;
  }

  // m_icnt = new shader_memory_interface(this,cluster);
  if (m_config->gpgpu_perfect_mem) {
    m_icnt = new perfect_memory_interface(this, m_cluster);
  } else {
    m_icnt = new shader_memory_interface(this, m_cluster);
  }
  m_mem_fetch_allocator =
      new shader_core_mem_fetch_allocator(m_sid, m_tpc, m_memory_config);

  // fetch
  m_last_warp_fetched = 0;

#define STRSIZE 1024
  char name[STRSIZE];
  snprintf(name, STRSIZE, "L1I_%03d", m_sid);
  m_L1I = new read_only_cache(name, m_config->m_L1I_config, m_sid,
                              get_shader_instruction_cache_id(), m_icnt,
                              IN_L1I_MISS_QUEUE,
                              m_gpu->gpgpu_ctx->accelsim_compat_mode, logger);
}

void trace_shader_core_ctx::create_schedulers() {
  m_scoreboard = new Scoreboard(m_sid, m_config->max_warps_per_shader, m_gpu);

  // scedulers
  // must currently occur after all inputs have been initialized.
  std::string sched_config = m_config->gpgpu_scheduler_string;
  const concrete_scheduler scheduler =
      sched_config.find("lrr") != std::string::npos ? CONCRETE_SCHEDULER_LRR
      : sched_config.find("two_level_active") != std::string::npos
          ? CONCRETE_SCHEDULER_TWO_LEVEL_ACTIVE
      : sched_config.find("gto") != std::string::npos ? CONCRETE_SCHEDULER_GTO
      : sched_config.find("rrr") != std::string::npos ? CONCRETE_SCHEDULER_RRR
      : sched_config.find("old") != std::string::npos
          ? CONCRETE_SCHEDULER_OLDEST_FIRST
      : sched_config.find("warp_limiting") != std::string::npos
          ? CONCRETE_SCHEDULER_WARP_LIMITING
          : NUM_CONCRETE_SCHEDULERS;
  logger->debug("using {} scheduler", g_concrete_scheduler_str[scheduler]);
  assert(scheduler != NUM_CONCRETE_SCHEDULERS);

  for (unsigned i = 0; i < m_config->gpgpu_num_sched_per_core; i++) {
    switch (scheduler) {
      case CONCRETE_SCHEDULER_LRR:
        schedulers.push_back(new lrr_scheduler(
            m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
            &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
            &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
            &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
            &m_pipeline_reg[ID_OC_MEM], i));
        break;
      case CONCRETE_SCHEDULER_TWO_LEVEL_ACTIVE:
        schedulers.push_back(new two_level_active_scheduler(
            m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
            &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
            &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
            &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
            &m_pipeline_reg[ID_OC_MEM], i, m_config->gpgpu_scheduler_string));
        break;
      case CONCRETE_SCHEDULER_GTO:
        schedulers.push_back(new gto_scheduler(
            m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
            &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
            &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
            &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
            &m_pipeline_reg[ID_OC_MEM], i));
        break;
      case CONCRETE_SCHEDULER_RRR:
        schedulers.push_back(new rrr_scheduler(
            m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
            &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
            &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
            &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
            &m_pipeline_reg[ID_OC_MEM], i));
        break;
      case CONCRETE_SCHEDULER_OLDEST_FIRST:
        schedulers.push_back(new oldest_scheduler(
            m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
            &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
            &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
            &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
            &m_pipeline_reg[ID_OC_MEM], i));
        break;
      case CONCRETE_SCHEDULER_WARP_LIMITING:
        schedulers.push_back(new swl_scheduler(
            m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
            &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
            &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
            &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
            &m_pipeline_reg[ID_OC_MEM], i, m_config->gpgpu_scheduler_string));
        break;
      default:
        abort();
    };
  }

  for (unsigned i = 0; i < m_warp.size(); i++) {
    // distribute i's evenly though schedulers;
    schedulers[i % m_config->gpgpu_num_sched_per_core]->add_supervised_warp_id(
        i);
  }
  for (unsigned i = 0; i < m_config->gpgpu_num_sched_per_core; ++i) {
    schedulers[i]->done_adding_supervised_warps();
  }
}

// enum {
//   SP_CUS,
//   DP_CUS,
//   SFU_CUS,
//   TENSOR_CORE_CUS,
//   INT_CUS,
//   MEM_CUS,
//   GEN_CUS
// } operand_collector_unit_kind;

void trace_shader_core_ctx::create_exec_pipeline() {
  // op collector configuration
  // enum { SP_CUS, DP_CUS, SFU_CUS, TENSOR_CORE_CUS, INT_CUS, MEM_CUS, GEN_CUS
  // };

  // opndcoll_rfu_t::port_vector_t in_ports;
  // opndcoll_rfu_t::port_vector_t out_ports;
  // opndcoll_rfu_t::uint_vector_t cu_sets;

  port_vector_t in_ports;
  port_vector_t out_ports;
  uint_vector_t cu_sets;

  bool accelsim_compat_mode = get_gpu()->gpgpu_ctx->accelsim_compat_mode;

  // configure generic collectors
  m_operand_collector.add_cu_set(
      GEN_CUS, m_config->gpgpu_operand_collector_num_units_gen,
      m_config->gpgpu_operand_collector_num_out_ports_gen);

  for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_gen;
       i++) {
    in_ports.push_back(&m_pipeline_reg[ID_OC_SP]);
    in_ports.push_back(&m_pipeline_reg[ID_OC_SFU]);
    in_ports.push_back(&m_pipeline_reg[ID_OC_MEM]);
    out_ports.push_back(&m_pipeline_reg[OC_EX_SP]);
    out_ports.push_back(&m_pipeline_reg[OC_EX_SFU]);
    out_ports.push_back(&m_pipeline_reg[OC_EX_MEM]);
    if (m_config->gpgpu_tensor_core_avail) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_TENSOR_CORE]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_TENSOR_CORE]);
    }
    if (m_config->gpgpu_num_dp_units > 0) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_DP]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_DP]);
    }
    if (m_config->gpgpu_num_int_units > 0) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_INT]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_INT]);
    }
    if (m_config->m_specialized_unit.size() > 0) {
      for (unsigned j = 0; j < m_config->m_specialized_unit.size(); ++j) {
        in_ports.push_back(
            &m_pipeline_reg[m_config->m_specialized_unit[j].ID_OC_SPEC_ID]);
        out_ports.push_back(
            &m_pipeline_reg[m_config->m_specialized_unit[j].OC_EX_SPEC_ID]);
      }
    }
    cu_sets.push_back((unsigned)GEN_CUS);
    m_operand_collector.add_port(in_ports, out_ports, cu_sets);
    in_ports.clear(), out_ports.clear(), cu_sets.clear();
  }

  if (m_config->enable_specialized_operand_collector) {
    m_operand_collector.add_cu_set(
        SP_CUS, m_config->gpgpu_operand_collector_num_units_sp,
        m_config->gpgpu_operand_collector_num_out_ports_sp);

    m_operand_collector.add_cu_set(
        DP_CUS, m_config->gpgpu_operand_collector_num_units_dp,
        m_config->gpgpu_operand_collector_num_out_ports_dp);
    if (accelsim_compat_mode) {
      m_operand_collector.add_cu_set(
          TENSOR_CORE_CUS,
          m_config->gpgpu_operand_collector_num_units_tensor_core,
          m_config->gpgpu_operand_collector_num_out_ports_tensor_core);
    }
    m_operand_collector.add_cu_set(
        SFU_CUS, m_config->gpgpu_operand_collector_num_units_sfu,
        m_config->gpgpu_operand_collector_num_out_ports_sfu);

    m_operand_collector.add_cu_set(
        MEM_CUS, m_config->gpgpu_operand_collector_num_units_mem,
        m_config->gpgpu_operand_collector_num_out_ports_mem);

    m_operand_collector.add_cu_set(
        INT_CUS, m_config->gpgpu_operand_collector_num_units_int,
        m_config->gpgpu_operand_collector_num_out_ports_int);

    for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_sp;
         i++) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_SP]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_SP]);
      cu_sets.push_back((unsigned)SP_CUS);
      cu_sets.push_back((unsigned)GEN_CUS);
      m_operand_collector.add_port(in_ports, out_ports, cu_sets);
      in_ports.clear(), out_ports.clear(), cu_sets.clear();
    }

    for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_dp;
         i++) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_DP]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_DP]);
      cu_sets.push_back((unsigned)DP_CUS);
      cu_sets.push_back((unsigned)GEN_CUS);
      m_operand_collector.add_port(in_ports, out_ports, cu_sets);
      in_ports.clear(), out_ports.clear(), cu_sets.clear();
    }

    for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_sfu;
         i++) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_SFU]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_SFU]);
      cu_sets.push_back((unsigned)SFU_CUS);
      cu_sets.push_back((unsigned)GEN_CUS);
      m_operand_collector.add_port(in_ports, out_ports, cu_sets);
      in_ports.clear(), out_ports.clear(), cu_sets.clear();
    }

    if (accelsim_compat_mode) {
      for (unsigned i = 0;
           i < m_config->gpgpu_operand_collector_num_in_ports_tensor_core;
           i++) {
        in_ports.push_back(&m_pipeline_reg[ID_OC_TENSOR_CORE]);
        out_ports.push_back(&m_pipeline_reg[OC_EX_TENSOR_CORE]);
        cu_sets.push_back((unsigned)TENSOR_CORE_CUS);
        cu_sets.push_back((unsigned)GEN_CUS);
        m_operand_collector.add_port(in_ports, out_ports, cu_sets);
        in_ports.clear(), out_ports.clear(), cu_sets.clear();
      }
    }

    for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_mem;
         i++) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_MEM]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_MEM]);
      cu_sets.push_back((unsigned)MEM_CUS);
      cu_sets.push_back((unsigned)GEN_CUS);
      m_operand_collector.add_port(in_ports, out_ports, cu_sets);
      in_ports.clear(), out_ports.clear(), cu_sets.clear();
    }

    for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_int;
         i++) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_INT]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_INT]);
      cu_sets.push_back((unsigned)INT_CUS);
      cu_sets.push_back((unsigned)GEN_CUS);
      m_operand_collector.add_port(in_ports, out_ports, cu_sets);
      in_ports.clear(), out_ports.clear(), cu_sets.clear();
    }
  }

  m_operand_collector.init(m_config->gpgpu_num_reg_banks, this);

  m_num_function_units =
      m_config->gpgpu_num_sp_units + m_config->gpgpu_num_dp_units +
      m_config->gpgpu_num_sfu_units + m_config->gpgpu_num_tensor_core_units +
      m_config->gpgpu_num_int_units + m_config->m_specialized_unit_num +
      1;  // sp_unit, sfu, dp, tensor, int, ldst_unit

  for (int k = 0; k < m_config->gpgpu_num_sp_units; k++) {
    m_fu.push_back(new sp_unit(&m_pipeline_reg[EX_WB], m_config, this, k));
    m_dispatch_port.push_back(ID_OC_SP);
    m_issue_port.push_back(OC_EX_SP);
  }

  for (int k = 0; k < m_config->gpgpu_num_dp_units; k++) {
    m_fu.push_back(new dp_unit(&m_pipeline_reg[EX_WB], m_config, this, k));
    m_dispatch_port.push_back(ID_OC_DP);
    m_issue_port.push_back(OC_EX_DP);
  }
  for (int k = 0; k < m_config->gpgpu_num_int_units; k++) {
    m_fu.push_back(new int_unit(&m_pipeline_reg[EX_WB], m_config, this, k));
    m_dispatch_port.push_back(ID_OC_INT);
    m_issue_port.push_back(OC_EX_INT);
  }

  for (int k = 0; k < m_config->gpgpu_num_sfu_units; k++) {
    m_fu.push_back(new sfu(&m_pipeline_reg[EX_WB], m_config, this, k));
    m_dispatch_port.push_back(ID_OC_SFU);
    m_issue_port.push_back(OC_EX_SFU);
  }

  for (int k = 0; k < m_config->gpgpu_num_tensor_core_units; k++) {
    m_fu.push_back(new tensor_core(&m_pipeline_reg[EX_WB], m_config, this, k));
    m_dispatch_port.push_back(ID_OC_TENSOR_CORE);
    m_issue_port.push_back(OC_EX_TENSOR_CORE);
  }

  for (unsigned j = 0; j < m_config->m_specialized_unit.size(); j++) {
    for (unsigned k = 0; k < m_config->m_specialized_unit[j].num_units; k++) {
      m_fu.push_back(new specialized_unit(
          &m_pipeline_reg[EX_WB], m_config, this, SPEC_UNIT_START_ID + j,
          m_config->m_specialized_unit[j].name,
          m_config->m_specialized_unit[j].latency, k));
      m_dispatch_port.push_back(m_config->m_specialized_unit[j].ID_OC_SPEC_ID);
      m_issue_port.push_back(m_config->m_specialized_unit[j].OC_EX_SPEC_ID);
    }
  }

  m_ldst_unit = new ldst_unit(m_icnt, m_mem_fetch_allocator, this,
                              &m_operand_collector, m_scoreboard, m_config,
                              m_memory_config, m_stats, m_sid, m_tpc);
  m_fu.push_back(m_ldst_unit);
  m_dispatch_port.push_back(ID_OC_MEM);
  m_issue_port.push_back(OC_EX_MEM);

  assert(m_num_function_units == m_fu.size() and
         m_fu.size() == m_dispatch_port.size() and
         m_fu.size() == m_issue_port.size());

  // there are as many result buses as the width of the EX_WB stage
  num_result_bus = m_config->pipe_widths[EX_WB];
  for (unsigned i = 0; i < num_result_bus; i++) {
    this->m_result_bus.push_back(new std::bitset<MAX_ALU_LATENCY>());
  }
}

bool trace_shader_core_ctx::fetch_unit_response_buffer_full() const {
  return false;
}

void trace_shader_core_ctx::accept_fetch_response(mem_fetch *mf) {
  mf->set_status(IN_SHADER_FETCHED,
                 m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
  m_L1I->fill(mf, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
}

bool trace_shader_core_ctx::ldst_unit_response_buffer_full() const {
  return m_ldst_unit->response_buffer_full();
}

void trace_shader_core_ctx::accept_ldst_unit_response(mem_fetch *mf) {
  m_ldst_unit->fill(mf);
}

void trace_shader_core_ctx::store_ack(class mem_fetch *mf) {
  assert(mf->get_type() == WRITE_ACK ||
         (m_config->gpgpu_perfect_mem && mf->get_is_write()));
  unsigned warp_id = mf->get_wid();
  m_warp[warp_id]->dec_store_req();
}

#include <mutex>

std::mutex mtx;

void trace_shader_core_ctx::cycle() {
  logger->debug(
      "cycle {} core ({}, {}): core cycle \tactive={}, not completed={}",
      m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, get_tpc(), get_sid(),
      isactive(), get_not_completed());

  // if (!isactive() && get_not_completed() == 0) {
  //   logger->debug("cycle {} core ({}, {}): core done",
  //                 m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, get_tpc(),
  //                 get_sid());
  //   return;
  // }

  m_stats->shader_cycles[m_sid]++;
  writeback();
  execute();
  read_operands();
  issue();
  for (unsigned i = 0; i < m_config->inst_fetch_throughput; ++i) {
    Instant start;
    std::chrono::nanoseconds dur_ns;

    start = now();
    decode();
    dur_ns = duration(now() - start);
    mtx.lock();
    increment_timing(m_gpu->m_timings, "core::decode", dur_ns);
    mtx.unlock();

    start = now();
    fetch();
    dur_ns = duration(now() - start);
    mtx.lock();
    increment_timing(m_gpu->m_timings, "core::fetch", dur_ns);
    mtx.unlock();
  }
}

void trace_shader_core_ctx::writeback() {
  logger->debug("cycle {} core ({}, {}): writeback: ex wb pipeline={}",
                m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, get_tpc(),
                get_sid(), m_pipeline_reg[EX_WB]);
  unsigned max_committed_thread_instructions =
      m_config->warp_size *
      (m_config->pipe_widths[EX_WB]);  // from the functional units
  m_stats->m_pipeline_duty_cycle[m_sid] =
      ((float)(m_stats->m_num_sim_insn[m_sid] -
               m_stats->m_last_num_sim_insn[m_sid])) /
      max_committed_thread_instructions;

  m_stats->m_last_num_sim_insn[m_sid] = m_stats->m_num_sim_insn[m_sid];
  m_stats->m_last_num_sim_winsn[m_sid] = m_stats->m_num_sim_winsn[m_sid];

  warp_inst_t **preg = m_pipeline_reg[EX_WB].get_ready();
  warp_inst_t *pipe_reg = (preg == NULL) ? NULL : *preg;
  while (preg and !pipe_reg->empty()) {
    // ready for writeback instruction

    logger->debug("instruction {} (pc={}) ready for writeback",
                  pipe_reg->opcode_str(), pipe_reg->pc);
    /*
     * Right now, the writeback stage drains all waiting instructions
     * assuming there are enough ports in the register file or the
     * conflicts are resolved at issue.
     */
    /*
     * The operand collector writeback can generally generate a stall
     * However, here, the pipelines should be un-stallable. This is
     * guaranteed because this is the first time the writeback function
     * is called after the operand collector's step function, which
     * resets the allocations. There is one case which could result in
     * the writeback function returning false (stall), which is when
     * an instruction tries to modify two registers (GPR and predicate)
     * To handle this case, we ignore the return value (thus allowing
     * no stalling).
     */

    logger->debug("{}", warp_instr_ptr(pipe_reg));

    m_operand_collector.writeback(*pipe_reg);
    unsigned warp_id = pipe_reg->warp_id();
    m_scoreboard->releaseRegisters(pipe_reg);
    m_warp[warp_id]->dec_inst_in_pipeline();
    warp_inst_complete(*pipe_reg);
    m_gpu->gpu_sim_insn_last_update_sid = m_sid;
    m_gpu->gpu_sim_insn_last_update = m_gpu->gpu_sim_cycle;
    m_last_inst_gpu_sim_cycle = m_gpu->gpu_sim_cycle;
    m_last_inst_gpu_tot_sim_cycle = m_gpu->gpu_tot_sim_cycle;
    pipe_reg->clear();
    preg = m_pipeline_reg[EX_WB].get_ready();
    pipe_reg = (preg == NULL) ? NULL : *preg;
  }
}

void trace_shader_core_ctx::execute() {
  logger->debug("cycle {} core ({}, {}): execute: ",
                m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, get_tpc(),
                get_sid());
  for (unsigned i = 0; i < num_result_bus; i++) {
    *(m_result_bus[i]) >>= 1;
  }
  for (unsigned n = 0; n < m_num_function_units; n++) {
    unsigned issue_port = m_issue_port[n];

    register_set &issue_inst = m_pipeline_reg[issue_port];
    // if (issue_port == OC_EX_SP || issue_port == OC_EX_MEM) {
    if (true) {
      // print the state of the issue unit BEFORE
      logger->debug("fu[{}] {}\tcycle {} before \t{}", n, m_fu[n]->get_name(),
                    m_gpu->gpu_sim_cycle, issue_inst);
    }

    unsigned multiplier = m_fu[n]->clock_multiplier();
    assert(multiplier == 1);
    for (unsigned c = 0; c < multiplier; c++) m_fu[n]->cycle();
    m_fu[n]->active_lanes_in_pipeline();

    // if (issue_port == OC_EX_SP || issue_port == OC_EX_MEM) {
    if (true) {
      // print the state of the issue unit AFTER
      logger->debug("fu[{}] {}\tcycle {} after \t{}", n, m_fu[n]->get_name(),
                    m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle,
                    issue_inst);
    }

    unsigned reg_id = (unsigned)-1;
    bool partition_issue =
        m_config->sub_core_model && m_fu[n]->is_issue_partitioned();
    if (partition_issue) {
      reg_id = m_fu[n]->get_issue_reg_id();
    }

    warp_inst_t **ready_reg = issue_inst.get_ready(partition_issue, reg_id);

    if (logger->should_log(spdlog::level::trace)) {
      logger->trace("occupied: {}", mask_to_string(m_fu[n]->occupied));
      if (ready_reg != NULL) {
        logger->trace(
            "cycle {} core ({}, {}): execute: checking {} fu[{:<03}] can "
            "issue={} latency = {}",
            m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, get_tpc(),
            get_sid(), warp_instr_ptr(*ready_reg), n,
            m_fu[n]->can_issue(**ready_reg), (*ready_reg)->latency);
      }
    }

    if (issue_inst.has_ready(partition_issue, reg_id) &&
        m_fu[n]->can_issue(**ready_reg)) {
      bool schedule_wb_now = !m_fu[n]->stallable();
      int resbus = -1;
      resbus = test_res_bus((*ready_reg)->latency);

      logger->debug(
          "cycle {} core ({}, {}): execute: {} (partition issue={}, "
          "schedule wb now={}, resbus={} latency={} reg_id={}) ready for issue "
          "to fu[{:<03}]={}",
          m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, get_tpc(), get_sid(),
          warp_instr_ptr(*ready_reg), partition_issue, schedule_wb_now, resbus,
          (*ready_reg)->latency, reg_id, n, m_fu[n]->get_name());

      bool issued = true;
      if (schedule_wb_now && resbus != -1) {
        assert((*ready_reg)->latency < MAX_ALU_LATENCY);
        m_result_bus[resbus]->set((*ready_reg)->latency);
        m_fu[n]->issue(issue_inst);
      } else if (!schedule_wb_now) {
        m_fu[n]->issue(issue_inst);
      } else {
        // stall issue (cannot reserve result bus)
        issued = false;
      }

      logger->debug("execute: issue={}", issued);
    }
  }
}

void trace_shader_core_ctx::read_operands() {
  for (unsigned i = 0; i < m_config->reg_file_port_throughput; ++i)
    m_operand_collector.step();
}

void trace_shader_core_ctx::issue() {
  // Ensure fair round robin issu between schedulers
  unsigned j;
  for (unsigned i = 0; i < schedulers.size(); i++) {
    j = (Issue_Prio + i) % schedulers.size();
    schedulers[j]->cycle();
  }
  Issue_Prio = (Issue_Prio + 1) % schedulers.size();

  // really is issue;
  // for (unsigned i = 0; i < schedulers.size(); i++) {
  //    schedulers[i]->cycle();
  //}
}

void trace_shader_core_ctx::decode() {
  logger->debug("trace_shader_core_ctx::decode() valid={}",
                m_inst_fetch_buffer.m_valid);

  if (m_inst_fetch_buffer.m_valid) {
    // decode 1 or 2 instructions and place them into ibuffer
    address_type pc = m_inst_fetch_buffer.m_pc;
    unsigned warp_id = m_inst_fetch_buffer.m_warp_id;

    if (!get_gpu()->gpgpu_ctx->accelsim_compat_mode) {
      // debug: print all valid instructions in this warp
      m_warp[warp_id]->print_trace_instructions(false, logger);
    }

    const warp_inst_t *pI1 = get_next_inst(warp_id, pc);
    logger->debug(
        "ibuffer fill for warp {} at slot {} with instruction {} trace pc={}",
        warp_id, 0, pI1->display(),
        static_cast<trace_shd_warp_t *>(m_warp[warp_id])->trace_pc);

    // TODO: roman could this be in the if pI1 block?
    m_warp[warp_id]->ibuffer_fill(0, pI1);
    m_warp[warp_id]->inc_inst_in_pipeline();
    if (pI1) {
      m_stats->m_num_decoded_insn[m_sid]++;
      if ((pI1->oprnd_type == INT_OP) ||
          (pI1->oprnd_type == UN_OP)) {  // these counters get added up in
                                         // mcPat to compute scheduler power
        m_stats->m_num_INTdecoded_insn[m_sid]++;
      } else if (pI1->oprnd_type == FP_OP) {
        m_stats->m_num_FPdecoded_insn[m_sid]++;
      }
      const warp_inst_t *pI2 = get_next_inst(warp_id, pc + pI1->isize);
      if (pI2) {
        logger->debug(
            "ibuffer fill for warp {} at slot {} with instruction {} trace "
            "pc={}",
            warp_id, 1, pI2->display(),
            static_cast<trace_shd_warp_t *>(m_warp[warp_id])->trace_pc);

        m_warp[warp_id]->ibuffer_fill(1, pI2);
        m_warp[warp_id]->inc_inst_in_pipeline();
        m_stats->m_num_decoded_insn[m_sid]++;
        if ((pI1->oprnd_type == INT_OP) ||
            (pI1->oprnd_type == UN_OP)) {  // these counters get added up in
                                           // mcPat to compute scheduler power
          m_stats->m_num_INTdecoded_insn[m_sid]++;
        } else if (pI2->oprnd_type == FP_OP) {
          m_stats->m_num_FPdecoded_insn[m_sid]++;
        }
      }
    }
    m_inst_fetch_buffer.m_valid = false;
  }
}

void trace_shader_core_ctx::fetch() {
  bool before = m_L1I->access_ready();
  size_t accesses_before = m_L1I->ready_accesses().size();
  logger->debug("trace_shader_core_ctx::fetch() (valid={} l1i ready=[{}])",
                m_inst_fetch_buffer.m_valid,
                fmt::join(m_L1I->ready_accesses(), ","));

  // sanity check that we are not changing anything
  assert(before == m_L1I->access_ready());
  assert(accesses_before == m_L1I->ready_accesses().size());

  if (!m_inst_fetch_buffer.m_valid) {
    if (m_L1I->access_ready()) {
      mem_fetch *mf = m_L1I->next_access();
      m_warp[mf->get_wid()]->clear_imiss_pending();
      m_inst_fetch_buffer =
          ifetch_buffer_t(m_warp[mf->get_wid()]->get_pc(),
                          mf->get_access_size(), mf->get_wid());
      assert(m_warp[mf->get_wid()]->get_pc() ==
             (mf->get_addr() -
              PROGRAM_MEM_START));  // Verify that we got the instruction
                                    // we were expecting.
      m_inst_fetch_buffer.m_valid = true;
      m_warp[mf->get_wid()]->set_last_fetch(m_gpu->gpu_sim_cycle);
      delete mf;
    } else {
      if (logger->should_log(spdlog::level::debug)) {
        for (unsigned warp_id = 0; warp_id < m_config->max_warps_per_shader;
             warp_id++) {
          // if (m_warp[warp_id]->instruction_count() == 0) {
          if (m_warp[warp_id]->get_warp_id() == (unsigned)-1) {
            // consider empty
            continue;
          }
          assert(warp_id == m_warp[warp_id]->get_warp_id());
          // if (m_warp[warp_id]->functional_done() &&
          //     m_warp[warp_id]->hardware_done() &&
          //     m_warp[warp_id]->done_exit())
          //     {
          //   continue;
          // }
          logger->debug(
              "\tchecking warp id = {warp_id} dyn warp id = {dynamic_warp_id} "
              "(instruction count={instruction_count}, "
              "trace pc={trace_pc}, hardware_done={hardware_done}, "
              "functional_done={functional_done}, instr in "
              "pipe={instr_in_pipe}, stores={stores}, done_exit={done_exit}, "
              "has "
              "pending writes=[{pending_writes}])",
              fmt::arg("warp_id", warp_id),
              fmt::arg("dynamic_warp_id",
                       m_warp[warp_id]->get_dynamic_warp_id()),
              fmt::arg("instruction_count",
                       m_warp[warp_id]->instruction_count()),
              fmt::arg("trace_pc", m_warp[warp_id]->trace_pc),
              fmt::arg("hardware_done", m_warp[warp_id]->hardware_done()),
              fmt::arg("functional_done", m_warp[warp_id]->functional_done()),
              fmt::arg("instr_in_pipe", m_warp[warp_id]->m_inst_in_pipeline),
              fmt::arg("stores", m_warp[warp_id]->m_stores_outstanding),
              fmt::arg("done_exit", m_warp[warp_id]->done_exit()),
              fmt::arg(
                  "pending_writes",
                  fmt::join(m_scoreboard->get_pending_writes(warp_id), ",")));
        }

        logger->debug("");
        logger->debug("");
      }

      // find an active warp with space in instruction buffer that is not
      // already waiting on a cache miss and get next 1-2 instructions
      // from i-cache...
      size_t num_checked = 0;
      for (unsigned i = 0; i < m_config->max_warps_per_shader; i++) {
        num_checked++;
        unsigned warp_id =
            (m_last_warp_fetched + 1 + i) % m_config->max_warps_per_shader;
        assert(m_warp[warp_id]->get_warp_id() == warp_id ||
               m_warp[warp_id]->get_warp_id() == (unsigned)-1);

        // this code checks if this warp has finished executing and can
        // be reclaimed
        if (m_warp[warp_id]->hardware_done() &&
            !m_scoreboard->has_pending_writes(warp_id) &&
            !m_warp[warp_id]->done_exit()) {
          logger->debug("\tchecking if warp_id = {} did complete", warp_id);

          // check each thread of the warp for completion
          bool did_exit = false;
          for (unsigned t = 0; t < m_config->warp_size; t++) {
            unsigned tid = warp_id * m_config->warp_size + t;

            if (m_threadState[tid].m_active == true) {
              m_threadState[tid].m_active = false;
              unsigned cta_id = m_warp[warp_id]->get_cta_id();
              logger->debug("thread {} of block {} completed ({} left)", tid,
                            cta_id, m_cta_status[cta_id]);

              if (m_thread[tid] == NULL) {
                register_cta_thread_exit(cta_id,
                                         m_warp[warp_id]->get_kernel_info());
              } else {
                register_cta_thread_exit(cta_id,
                                         &(m_thread[tid]->get_kernel()));
              }
              m_not_completed -= 1;
              m_active_threads.reset(tid);
              did_exit = true;
            }
          }
          if (did_exit) {
            m_warp[warp_id]->set_done_exit();
          }
          --m_active_warps;
          assert(m_active_warps >= 0);
        }

        // this code fetches instructions from the i-cache or generates
        // memory
        if (!m_warp[warp_id]->functional_done() &&
            !m_warp[warp_id]->imiss_pending() &&
            m_warp[warp_id]->ibuffer_empty()) {
          if (!get_gpu()->gpgpu_ctx->accelsim_compat_mode) {
            const warp_inst_t *current_inst =
                m_warp[warp_id]->get_current_trace_inst();

            assert(current_inst != NULL);
            logger->debug("\t fetching instr {} for warp id = {}",
                          current_inst->display(), warp_id);
          }

          address_type pc;
          pc = m_warp[warp_id]->get_pc();
          address_type ppc = pc + PROGRAM_MEM_START;
          unsigned nbytes = 16;
          unsigned offset_in_block =
              pc & (m_config->m_L1I_config.get_line_sz() - 1);
          if ((offset_in_block + nbytes) > m_config->m_L1I_config.get_line_sz())
            nbytes = (m_config->m_L1I_config.get_line_sz() - offset_in_block);

          // TODO: replace with use of allocator
          // mem_fetch *mf = m_mem_fetch_allocator->alloc()
          mem_access_t acc(INST_ACC_R, ppc, nbytes, false, m_gpu->gpgpu_ctx);
          acc.set_alloc_start_addr(PROGRAM_MEM_START);

          mem_fetch *mf = new mem_fetch(
              acc, NULL /*we don't have an instruction yet*/, READ_PACKET_SIZE,
              warp_id, m_sid, m_tpc, m_memory_config,
              m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle);
          std::list<cache_event> events;
          enum cache_request_status status;

          if (m_config->perfect_inst_const_cache) {
            status = HIT;
            shader_cache_access_log(m_sid, INSTRUCTION, 0);
          } else {
            if (get_gpu()->gpgpu_ctx->accelsim_compat_mode) {
              // printf(
              //     "core %d-%d fetch inst cache access(%lu) time=%llu warp "
              //     "id=%d "
              //     "pc=%lu\n",
              //     m_tpc, m_sid, (new_addr_type)ppc,
              //     m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle, warp_id,
              //     m_warp[warp_id]->get_pc());
            }

            status = m_L1I->access(
                (new_addr_type)ppc, mf,
                m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle, events);

            logger->debug("L1I->access(addr={}) -> status = {}", ppc,
                          get_cache_request_status_str(status));
          }
          if (status == MISS) {
            m_last_warp_fetched = warp_id;
            m_warp[warp_id]->set_imiss_pending();
            m_warp[warp_id]->set_last_fetch(m_gpu->gpu_sim_cycle);
          } else if (status == HIT) {
            m_last_warp_fetched = warp_id;
            m_inst_fetch_buffer = ifetch_buffer_t(pc, nbytes, warp_id);
            m_warp[warp_id]->set_last_fetch(m_gpu->gpu_sim_cycle);
            delete mf;
          } else {
            m_last_warp_fetched = warp_id;
            assert(status == RESERVATION_FAIL);
            delete mf;
          }
          break;
        }
      }
      // fmt::println("play fetch: num checked={}", num_checked);
    }
  }

  m_L1I->cycle();
}

int trace_shader_core_ctx::test_res_bus(int latency) {
  for (unsigned i = 0; i < num_result_bus; i++) {
    if (!m_result_bus[i]->test(latency)) {
      return i;
    }
  }
  return -1;
}

void trace_shader_core_ctx::register_cta_thread_exit(
    unsigned cta_num, trace_kernel_info_t *kernel) {
  assert(m_cta_status[cta_num] > 0);
  m_cta_status[cta_num]--;
  if (!m_cta_status[cta_num]) {
    // Increment the completed CTAs
    m_stats->ctas_completed++;
    m_gpu->inc_completed_cta();
    m_n_active_cta--;
    m_barriers.deallocate_barrier(cta_num);
    shader_CTA_count_unlog(m_sid, 1);

    SHADER_DPRINTF(LIVENESS,
                   "GPGPU-Sim uArch: Finished CTA #%u (%lld,%lld), %u "
                   "CTAs running\n",
                   cta_num, m_gpu->gpu_sim_cycle, m_gpu->gpu_tot_sim_cycle,
                   m_n_active_cta);

    if (m_n_active_cta == 0) {
      SHADER_DPRINTF(
          LIVENESS,
          "GPGPU-Sim uArch: Empty (last released kernel %u \'%s\').\n",
          kernel->get_uid(), kernel->name().c_str());
      fflush(stdout);

      // Shader can only be empty when no more cta are dispatched
      if (kernel != m_kernel) {
        assert(m_kernel == NULL || !m_gpu->kernel_more_cta_left(m_kernel));
      }
      m_kernel = NULL;
    }

    // Jin: for concurrent kernels on sm
    release_shader_resource_1block(cta_num, *kernel);
    kernel->dec_running();
    if (!m_gpu->kernel_more_cta_left(kernel)) {
      if (!kernel->running()) {
        SHADER_DPRINTF(LIVENESS,
                       "GPGPU-Sim uArch: GPU detected kernel %u \'%s\' "
                       "finished on shader %u.\n",
                       kernel->get_uid(), kernel->name().c_str(), m_sid);

        if (m_kernel == kernel) m_kernel = NULL;
        m_gpu->set_kernel_done(kernel);
      }
    }
  }
}

void trace_shader_core_ctx::release_shader_resource_1block(
    unsigned hw_ctaid, trace_kernel_info_t &k) {
  if (m_config->gpgpu_concurrent_kernel_sm) {
    unsigned threads_per_cta = k.threads_per_cta();
    const class trace_function_info *kernel = k.entry();
    unsigned int padded_cta_size = threads_per_cta;
    unsigned int warp_size = m_config->warp_size;
    if (padded_cta_size % warp_size)
      padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);

    assert(m_occupied_n_threads >= padded_cta_size);
    m_occupied_n_threads -= padded_cta_size;

    int start_thread = m_occupied_cta_to_hwtid[hw_ctaid];

    for (unsigned hwtid = start_thread; hwtid < start_thread + padded_cta_size;
         hwtid++)
      m_occupied_hwtid.reset(hwtid);
    m_occupied_cta_to_hwtid.erase(hw_ctaid);

    const struct gpgpu_ptx_sim_info *kernel_info = kernel->get_kernel_info();

    assert(m_occupied_shmem >= (unsigned int)kernel_info->smem);
    m_occupied_shmem -= kernel_info->smem;

    unsigned int used_regs = padded_cta_size * ((kernel_info->regs + 3) & ~3);
    assert(m_occupied_regs >= used_regs);
    m_occupied_regs -= used_regs;

    assert(m_occupied_ctas >= 1);
    m_occupied_ctas--;
  }
}

void trace_shader_core_ctx::cache_invalidate() { m_ldst_unit->invalidate(); }

float trace_shader_core_ctx::get_current_occupancy(
    unsigned long long &active, unsigned long long &total) const {
  // To match the achieved_occupancy in nvprof, only SMs that are active are
  // counted toward the occupancy.
  if (m_active_warps > 0) {
    total += m_warp.size();
    active += m_active_warps;
    return float(active) / float(total);
  } else {
    return 0;
  }
}

void trace_shader_core_ctx::get_cache_stats(cache_stats &cs) {
  // Adds stats from each cache to 'cs'
  cs += m_L1I->get_stats();          // Get L1I stats
  m_ldst_unit->get_cache_stats(cs);  // Get L1D, L1C, L1T stats
}

void trace_shader_core_ctx::reinit(unsigned start_thread, unsigned end_thread,
                                   bool reset_not_completed) {
  if (reset_not_completed) {
    m_not_completed = 0;
    m_active_threads.reset();

    // Jin: for concurrent kernels on a SM
    m_occupied_n_threads = 0;
    m_occupied_shmem = 0;
    m_occupied_regs = 0;
    m_occupied_ctas = 0;
    m_occupied_hwtid.reset();
    m_occupied_cta_to_hwtid.clear();
    m_active_warps = 0;
  }
  for (unsigned i = start_thread; i < end_thread; i++) {
    m_threadState[i].n_insn = 0;
    m_threadState[i].m_cta_id = -1;
  }
  unsigned start_warp = start_thread / m_config->warp_size;
  unsigned end_warp = end_thread / m_config->warp_size;
  logger->debug("reset warps {}..{} (threads {}..{})", start_warp, end_warp,
                start_thread, end_thread);
  for (unsigned i = start_warp; i < end_warp; ++i) {
    m_warp[i]->reset();
    // m_simt_stack[i]->reset();
  }
}

void trace_shader_core_ctx::get_L1I_sub_stats(
    struct cache_sub_stats &css) const {
  if (m_L1I) m_L1I->get_sub_stats(css);
}

void trace_shader_core_ctx::get_L1D_sub_stats(
    struct cache_sub_stats &css) const {
  m_ldst_unit->get_L1D_sub_stats(css);
}

void trace_shader_core_ctx::get_L1C_sub_stats(
    struct cache_sub_stats &css) const {
  m_ldst_unit->get_L1C_sub_stats(css);
}

void trace_shader_core_ctx::get_L1T_sub_stats(
    struct cache_sub_stats &css) const {
  m_ldst_unit->get_L1T_sub_stats(css);
}

void trace_shader_core_ctx::get_icnt_power_stats(long &n_simt_to_mem,
                                                 long &n_mem_to_simt) const {
  n_simt_to_mem += m_stats->n_simt_to_mem[m_sid];
  n_mem_to_simt += m_stats->n_mem_to_simt[m_sid];
}

bool trace_shader_core_ctx::can_issue_1block(trace_kernel_info_t &kernel) {
  // Jin: concurrent kernels on one SM
  if (m_config->gpgpu_concurrent_kernel_sm) {
    if (m_config->max_cta(kernel) < 1) return false;

    return occupy_shader_resource_1block(kernel, false);
  } else {
    return (get_n_active_cta() < m_config->max_cta(kernel));
  }
}

int trace_shader_core_ctx::find_available_hwtid(unsigned int cta_size,
                                                bool occupy) {
  unsigned int step;
  for (step = 0; step < m_config->n_thread_per_shader; step += cta_size) {
    unsigned int hw_tid;
    for (hw_tid = step; hw_tid < step + cta_size; hw_tid++) {
      if (m_occupied_hwtid.test(hw_tid)) break;
    }
    if (hw_tid == step + cta_size)  // consecutive non-active
      break;
  }
  if (step >= m_config->n_thread_per_shader)  // didn't find
    return -1;
  else {
    if (occupy) {
      for (unsigned hw_tid = step; hw_tid < step + cta_size; hw_tid++)
        m_occupied_hwtid.set(hw_tid);
    }
    return step;
  }
}

bool trace_shader_core_ctx::occupy_shader_resource_1block(
    trace_kernel_info_t &k, bool occupy) {
  unsigned threads_per_cta = k.threads_per_cta();
  const class trace_function_info *kernel = k.entry();
  unsigned int padded_cta_size = threads_per_cta;
  unsigned int warp_size = m_config->warp_size;
  if (padded_cta_size % warp_size)
    padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);

  if (m_occupied_n_threads + padded_cta_size > m_config->n_thread_per_shader)
    return false;

  if (find_available_hwtid(padded_cta_size, false) == -1) return false;

  const struct gpgpu_ptx_sim_info *kernel_info = kernel->get_kernel_info();

  if (m_occupied_shmem + kernel_info->smem > m_config->gpgpu_shmem_size)
    return false;

  unsigned int used_regs = padded_cta_size * ((kernel_info->regs + 3) & ~3);
  if (m_occupied_regs + used_regs > m_config->gpgpu_shader_registers)
    return false;

  if (m_occupied_ctas + 1 > m_config->max_cta_per_core) return false;

  if (occupy) {
    m_occupied_n_threads += padded_cta_size;
    m_occupied_shmem += kernel_info->smem;
    m_occupied_regs += (padded_cta_size * ((kernel_info->regs + 3) & ~3));
    m_occupied_ctas++;

    SHADER_DPRINTF(LIVENESS,
                   "GPGPU-Sim uArch: Occupied %u threads, %u shared mem, %u "
                   "registers, %u ctas, on shader %d\n",
                   m_occupied_n_threads, m_occupied_shmem, m_occupied_regs,
                   m_occupied_ctas, m_sid);
  }

  return true;
}

void trace_shader_core_ctx::set_max_cta(const trace_kernel_info_t &kernel) {
  // calculate the max cta count and cta size for local memory address
  // mapping
  kernel_max_cta_per_shader = m_config->max_cta(kernel);
  unsigned int gpu_cta_size = kernel.threads_per_cta();
  kernel_padded_threads_per_cta =
      (gpu_cta_size % m_config->warp_size)
          ? m_config->warp_size * ((gpu_cta_size / m_config->warp_size) + 1)
          : gpu_cta_size;
}

void trace_shader_core_ctx::issue_block2core(trace_kernel_info_t &kernel) {
  logger->debug("core ({}, {}): issue block", get_tpc(), get_sid());
  if (!m_config->gpgpu_concurrent_kernel_sm) {
    set_max_cta(kernel);
  } else {
    assert(occupy_shader_resource_1block(kernel, true));
  }
  logger->trace("max cta: kernel={} config={} sm={}", kernel_max_cta_per_shader,
                m_config->max_cta_per_core, MAX_CTA_PER_SHADER);

  kernel.inc_running();

  // find a free CTA context
  unsigned free_cta_hw_id = (unsigned)-1;

  unsigned max_cta_per_core;
  if (!m_config->gpgpu_concurrent_kernel_sm) {
    max_cta_per_core = kernel_max_cta_per_shader;
  } else {
    max_cta_per_core = m_config->max_cta_per_core;
  }
  assert(max_cta_per_core <= MAX_CTA_PER_SHADER &&
         "max cta per shader is smaller than maximum");

  logger->trace("core ({}, {}): free block status: [{}]", get_tpc(), get_sid(),
                fmt::join(m_cta_status, m_cta_status + 32, ","));

  for (unsigned i = 0; i < max_cta_per_core; i++) {
    if (m_cta_status[i] == 0) {
      free_cta_hw_id = i;
      break;
    }
  }
  assert(free_cta_hw_id != (unsigned)-1);

  // determine hardware threads and warps that will be used for this CTA
  int cta_size = kernel.threads_per_cta();
  logger->debug("cta size = {}", cta_size);

  // hw warp id = hw thread id mod warp size, so we need to find a range
  // of hardware thread ids corresponding to an integral number of hardware
  // thread ids
  int padded_cta_size = cta_size;
  if (cta_size % m_config->warp_size) {
    padded_cta_size =
        ((cta_size / m_config->warp_size) + 1) * (m_config->warp_size);
  }

  unsigned int start_thread, end_thread;

  if (!m_config->gpgpu_concurrent_kernel_sm) {
    start_thread = free_cta_hw_id * padded_cta_size;
    end_thread = start_thread + cta_size;
  } else {
    start_thread = find_available_hwtid(padded_cta_size, true);
    // logger->debug("padded cta size: {}", padded_cta_size);
    assert((int)start_thread != -1);
    end_thread = start_thread + cta_size;
    assert(m_occupied_cta_to_hwtid.find(free_cta_hw_id) ==
           m_occupied_cta_to_hwtid.end());
    m_occupied_cta_to_hwtid[free_cta_hw_id] = start_thread;
  }

  // reset the microarchitecture state of the selected hardware thread and
  // warp contexts
  reinit(start_thread, end_thread, false);

  // NOTE: assertion does not always hold (we only reset start to end)
  // for (auto &w : m_warp) {
  //   assert(w->done_exit());
  // }

  // initalize scalar threads and determine which hardware warps they are
  // allocated to bind functional simulation state of threads to hardware
  // resources (simulation)
  warp_set_t warps;
  unsigned nthreads_in_block = 0;
  // trace_function_info *kernel_func_info = kernel.entry();
  logger->debug("core[{}][{}]: issue block {} from kernel {}", m_tpc, m_sid,
                kernel.get_next_cta_id(), kernel);

  unsigned ctaid = kernel.get_next_cta_id_single();
  // REMOVE: checkpoint
  // symbol_table *symtab = kernel_func_info->get_symtab();
  // checkpoint *g_checkpoint = new checkpoint();
  for (unsigned i = start_thread; i < end_thread; i++) {
    m_threadState[i].m_cta_id = free_cta_hw_id;
    unsigned warp_id = i / m_config->warp_size;
    nthreads_in_block += sim_init_thread(
        kernel, &m_thread[i], m_sid, i, cta_size - (i - start_thread),
        m_config->n_thread_per_shader, this, free_cta_hw_id, warp_id,
        m_cluster->get_gpu());
    m_threadState[i].m_active = true;
    // load thread local memory and register file
    // REMOVE: resume
    // if (m_gpu->resume_option == 1 && kernel.get_uid() ==
    // m_gpu->resume_kernel
    // &&
    //     ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t)
    //     {
    //   char fname[2048];
    //   snprintf(fname, 2048, "checkpoint_files/thread_%d_%d_reg.txt",
    //            i % cta_size, ctaid);
    //   m_thread[i]->resume_reg_thread(fname, symtab);
    //   char f1name[2048];
    //   snprintf(f1name, 2048,
    //   "checkpoint_files/local_mem_thread_%d_%d_reg.txt",
    //            i % cta_size, ctaid);
    //   g_checkpoint->load_global_mem(m_thread[i]->m_local_mem, f1name);
    // }
    //
    warps.set(warp_id);
  }
  assert(nthreads_in_block > 0 &&
         nthreads_in_block <=
             m_config->n_thread_per_shader);  // should be at least one, but
                                              // less than max
  m_cta_status[free_cta_hw_id] = nthreads_in_block;
  logger->debug("num threads in block {} (hw {}) = {}\n", ctaid, free_cta_hw_id,
                nthreads_in_block);
  // assert(0);

  // REMOVE: resume
  // if (m_gpu->resume_option == 1 && kernel.get_uid() ==
  // m_gpu->resume_kernel
  // &&
  //     ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
  //   char f1name[2048];
  //   snprintf(f1name, 2048, "checkpoint_files/shared_mem_%d.txt", ctaid);
  //
  //   g_checkpoint->load_global_mem(m_thread[start_thread]->m_shared_mem,
  //   f1name);
  // }
  // now that we know which warps are used in this CTA, we can allocate
  // resources for use in CTA-wide barrier operations
  m_barriers.allocate_barrier(free_cta_hw_id, warps);

  // initialize the SIMT stacks and fetch hardware
  init_warps(free_cta_hw_id, start_thread, end_thread, ctaid, cta_size, kernel);
  m_n_active_cta++;

  shader_CTA_count_log(m_sid, 1);
  SHADER_DPRINTF(LIVENESS,
                 "GPGPU-Sim uArch: cta:%2u, start_tid:%4u, end_tid:%4u, "
                 "initialized @(%lld,%lld), kernel_uid:%u, kernel_name:%s\n",
                 free_cta_hw_id, start_thread, end_thread, m_gpu->gpu_sim_cycle,
                 m_gpu->gpu_tot_sim_cycle, kernel.get_uid(),
                 kernel.get_name().c_str());
}

void trace_shader_core_ctx::print_cache_stats(FILE *fp, unsigned &dl1_accesses,
                                              unsigned &dl1_misses) {
  m_ldst_unit->print_cache_stats(fp, dl1_accesses, dl1_misses);
}

// Returns numbers of addresses in translated_addrs, each addr points to a 4B
// (32-bit) word
unsigned trace_shader_core_ctx::translate_local_memaddr(
    address_type localaddr, unsigned tid, unsigned num_shader,
    unsigned datasize, new_addr_type *translated_addrs) {
  // During functional execution, each thread sees its own memory space for
  // local memory, but these need to be mapped to a shared address space for
  // timing simulation.  We do that mapping here.

  address_type thread_base = 0;
  unsigned max_concurrent_threads = 0;
  if (m_config->gpgpu_local_mem_map) {
    // assert(0 && "gpgpu local mem map");
    // Dnew = D*N + T%nTpC + nTpC*C
    // N = nTpC*nCpS*nS (max concurent threads)
    // C = nS*K + S (hw cta number per gpu)
    // K = T/nTpC   (hw cta number per core)
    // D = data index
    // T = thread
    // nTpC = number of threads per CTA
    // nCpS = number of CTA per shader
    //
    // for a given local memory address threads in a CTA map to
    // contiguous addresses, then distribute across memory space by CTAs
    // from successive shader cores first, then by successive CTA in same
    // shader core
    thread_base =
        4 * (kernel_padded_threads_per_cta *
                 (m_sid + num_shader * (tid / kernel_padded_threads_per_cta)) +
             tid % kernel_padded_threads_per_cta);
    max_concurrent_threads =
        kernel_padded_threads_per_cta * kernel_max_cta_per_shader * num_shader;
  } else {
    // legacy mapping that maps the same address in the local memory
    // space of all threads to a single contiguous address region
    thread_base = 4 * (m_config->n_thread_per_shader * m_sid + tid);
    max_concurrent_threads = num_shader * m_config->n_thread_per_shader;
  }
  assert(thread_base < 4 /*word size*/ * max_concurrent_threads);

  // If requested datasize > 4B, split into multiple 4B accesses
  // otherwise do one sub-4 byte memory access
  unsigned num_accesses = 0;

  if (datasize >= 4) {
    // >4B access, split into 4B chunks
    assert(datasize % 4 == 0);  // Must be a multiple of 4B
    num_accesses = datasize / 4;
    assert(num_accesses <= MAX_ACCESSES_PER_INSN_PER_THREAD);  // max 32B
    assert(localaddr % 4 == 0);  // Address must be 4B aligned - required if
                                 // accessing 4B per request, otherwise access
                                 // will overflow into next thread's space
    for (unsigned i = 0; i < num_accesses; i++) {
      address_type local_word = localaddr / 4 + i;
      address_type linear_address = local_word * max_concurrent_threads * 4 +
                                    thread_base + LOCAL_GENERIC_START;
      translated_addrs[i] = linear_address;
    }
  } else {
    // Sub-4B access, do only one access
    assert(datasize > 0);
    num_accesses = 1;
    address_type local_word = localaddr / 4;
    address_type local_word_offset = localaddr % 4;
    assert((localaddr + datasize - 1) / 4 ==
           local_word);  // Make sure access doesn't overflow into next 4B
                         // chunk
    address_type linear_address = local_word * max_concurrent_threads * 4 +
                                  local_word_offset + thread_base +
                                  LOCAL_GENERIC_START;
    translated_addrs[0] = linear_address;
  }
  return num_accesses;
}

void trace_shader_core_ctx::warp_inst_complete(const warp_inst_t &inst) {
  logger->debug(
      "core {}: warp_inst_complete [uid={}, core={}, warp={}, pc={} @ time={}]",
      m_sid, inst.get_uid(), m_sid, inst.warp_id(), inst.pc,
      m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle);

  if (inst.op_pipe == SP__OP)
    m_stats->m_num_sp_committed[m_sid]++;
  else if (inst.op_pipe == SFU__OP)
    m_stats->m_num_sfu_committed[m_sid]++;
  else if (inst.op_pipe == MEM__OP)
    m_stats->m_num_mem_committed[m_sid]++;

  if (m_config->gpgpu_clock_gated_lanes == false)
    m_stats->m_num_sim_insn[m_sid] += m_config->warp_size;
  else
    m_stats->m_num_sim_insn[m_sid] += inst.active_count();

  m_stats->m_num_sim_winsn[m_sid]++;
  m_gpu->gpu_sim_insn += inst.active_count();
  inst.completed(m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle);
}

void trace_shader_core_ctx::decrement_atomic_count(unsigned wid, unsigned n) {
  assert(m_warp[wid]->get_n_atomic() >= n);
  m_warp[wid]->dec_n_atomic(n);
}

void trace_shader_core_ctx::mem_instruction_stats(const warp_inst_t &inst) {
  unsigned active_count = inst.active_count();
  // this breaks some encapsulation: the is_[space] functions, if you change
  // those, change this.
  switch (inst.space.get_type()) {
    case undefined_space:
    case reg_space:
      break;
    case shared_space:
      m_stats->gpgpu_n_shmem_insn += active_count;
      break;
    case sstarr_space:
      m_stats->gpgpu_n_sstarr_insn += active_count;
      break;
    case const_space:
      m_stats->gpgpu_n_const_insn += active_count;
      break;
    case param_space_kernel:
    case param_space_local:
      m_stats->gpgpu_n_param_insn += active_count;
      break;
    case tex_space:
      m_stats->gpgpu_n_tex_insn += active_count;
      break;
    case global_space:
    case local_space:
      if (inst.is_store())
        m_stats->gpgpu_n_store_insn += active_count;
      else
        m_stats->gpgpu_n_load_insn += active_count;
      break;
    default:
      abort();
  }
}

bool trace_shader_core_ctx::warp_waiting_at_barrier(unsigned warp_id) const {
  return m_barriers.warp_waiting_at_barrier(warp_id);
}

bool trace_shader_core_ctx::warp_waiting_at_mem_barrier(unsigned warp_id) {
  if (!m_warp[warp_id]->get_membar()) return false;
  if (!m_scoreboard->has_pending_writes(warp_id)) {
    m_warp[warp_id]->clear_membar();
    if (m_gpu->get_config().flush_l1()) {
      // Mahmoud fixed this on Nov 2019
      // Invalidate L1 cache
      // Based on Nvidia Doc, at MEM barrier, we have to
      //(1) wait for all pending writes till they are acked
      //(2) invalidate L1 cache to ensure coherence and avoid reading
      // stall
      // data
      cache_invalidate();
      // TO DO: you need to stall the SM for 5k cycles.
    }
    return false;
  }
  return true;
}

std::list<unsigned> trace_shader_core_ctx::get_regs_written(
    const inst_t &fvt) const {
  std::list<unsigned> result;
  for (unsigned op = 0; op < MAX_REG_OPERANDS; op++) {
    int reg_num = fvt.arch_reg.dst[op];  // this math needs to match that used
                                         // in function_info::ptx_decode_inst
    if (reg_num >= 0)                    // valid register
      result.push_back(reg_num);
  }
  return result;
}

void trace_shader_core_ctx::initializeSIMTStack(unsigned warp_count,
                                                unsigned warp_size) {
  m_simt_stack = new simt_stack *[warp_count];
  for (unsigned i = 0; i < warp_count; ++i) {
    m_simt_stack[i] = NULL;
    // m_simt_stack[i] = new simt_stack(i, warp_size, m_gpu);
  }
  m_warp_size = warp_size;
  m_warp_count = warp_count;
}

void trace_shader_core_ctx::broadcast_barrier_reduction(unsigned cta_id,
                                                        unsigned bar_id,
                                                        warp_set_t warps) {
  for (unsigned i = 0; i < m_config->max_warps_per_shader; i++) {
    if (warps.test(i)) {
      const warp_inst_t *inst =
          m_warp[i]->restore_info_of_last_inst_at_barrier();
      const_cast<warp_inst_t *>(inst)->broadcast_barrier_reduction(
          inst->get_active_mask());
    }
  }
}

void trace_shader_core_ctx::incexecstat(warp_inst_t *&inst) {
  // REMOVE: power
  // Latency numbers for next operations are used to scale the power values
  // for special operations, according observations from microbenchmarking
  // TODO: put these numbers in the xml configuration
  // if (get_gpu()->get_config().g_power_simulation_enabled) {
  //   switch (inst->sp_op) {
  //   case INT__OP:
  //     incialu_stat(inst->active_count(), scaling_coeffs->int_coeff);
  //     break;
  //   case INT_MUL_OP:
  //     incimul_stat(inst->active_count(), scaling_coeffs->int_mul_coeff);
  //     break;
  //   case INT_MUL24_OP:
  //     incimul24_stat(inst->active_count(),
  //     scaling_coeffs->int_mul24_coeff); break;
  //   case INT_MUL32_OP:
  //     incimul32_stat(inst->active_count(),
  //     scaling_coeffs->int_mul32_coeff); break;
  //   case INT_DIV_OP:
  //     incidiv_stat(inst->active_count(), scaling_coeffs->int_div_coeff);
  //     break;
  //   case FP__OP:
  //     incfpalu_stat(inst->active_count(), scaling_coeffs->fp_coeff);
  //     break;
  //   case FP_MUL_OP:
  //     incfpmul_stat(inst->active_count(), scaling_coeffs->fp_mul_coeff);
  //     break;
  //   case FP_DIV_OP:
  //     incfpdiv_stat(inst->active_count(), scaling_coeffs->fp_div_coeff);
  //     break;
  //   case DP___OP:
  //     incdpalu_stat(inst->active_count(), scaling_coeffs->dp_coeff);
  //     break;
  //   case DP_MUL_OP:
  //     incdpmul_stat(inst->active_count(), scaling_coeffs->dp_mul_coeff);
  //     break;
  //   case DP_DIV_OP:
  //     incdpdiv_stat(inst->active_count(), scaling_coeffs->dp_div_coeff);
  //     break;
  //   case FP_SQRT_OP:
  //     incsqrt_stat(inst->active_count(), scaling_coeffs->sqrt_coeff);
  //     break;
  //   case FP_LG_OP:
  //     inclog_stat(inst->active_count(), scaling_coeffs->log_coeff);
  //     break;
  //   case FP_SIN_OP:
  //     incsin_stat(inst->active_count(), scaling_coeffs->sin_coeff);
  //     break;
  //   case FP_EXP_OP:
  //     incexp_stat(inst->active_count(), scaling_coeffs->exp_coeff);
  //     break;
  //   case TENSOR__OP:
  //     inctensor_stat(inst->active_count(), scaling_coeffs->tensor_coeff);
  //     break;
  //   case TEX__OP:
  //     inctex_stat(inst->active_count(), scaling_coeffs->tex_coeff);
  //     break;
  //   default:
  //     break;
  //   }
  //   if (inst->const_cache_operand) // warp has const address space load
  //   as one
  //                                  // operand
  //     inc_const_accesses(1);
  // }
}
