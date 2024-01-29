#pragma once

#include "barrier_set.hpp"
#include "ifetch_buffer.hpp"
#include "opndcoll_rfu.hpp"
#include "register_set.hpp"
#include "shader_core_config.hpp"
#include "shader_core_stats.hpp"
#include "trace_gpgpu_sim.hpp"
#include "trace_shd_warp.hpp"

#include "spdlog/logger.h"

class Scoreboard;
class ldst_unit;
class simd_function_unit;
class scheduler_unit;
class mem_fetch_interface;
class shader_core_mem_fetch_allocator;
class read_only_cache;
class thread_ctx_t;
class cache_stats;
class simt_stack;
class memory_config;
class shader_core_stats;

// should be distinct from other memory spaces...
// check ptx_ir.h to verify this does not overlap
// other memory spaces
#define PROGRAM_MEM_START 0xF0000000

class trace_shader_core_ctx {
 public:
  trace_shader_core_ctx(class trace_gpgpu_sim *gpu,
                        class trace_simt_core_cluster *cluster,
                        unsigned shader_id, unsigned tpc_id,
                        const shader_core_config *config,
                        const memory_config *mem_config,
                        shader_core_stats *stats, FILE *stats_out)

      : logger(gpu->logger),
        m_gpu(gpu),
        m_kernel(NULL),
        m_simt_stack(NULL),
        m_thread(NULL),
        m_warp_size(config->warp_size),
        m_barriers(this, config->max_warps_per_shader, config->max_cta_per_core,
                   config->max_barriers_per_cta, config->warp_size),
        m_operand_collector(gpu->logger),
        m_active_warps(0),
        m_dynamic_warp_id(0),
        stats_out(stats_out) {
    // core
    m_warp_count = config->n_thread_per_shader / m_warp_size;
    // Handle the case where the number of threads is not a
    // multiple of the warp size
    if (config->n_thread_per_shader % m_warp_size != 0) {
      m_warp_count += 1;
    }
    assert(m_warp_count * m_warp_size > 0);
    m_thread = (ptx_thread_info **)calloc(m_warp_count * m_warp_size,
                                          sizeof(ptx_thread_info *));
    initializeSIMTStack(m_warp_count, m_warp_size);

    for (unsigned i = 0; i < MAX_CTA_PER_SHADER; i++) {
      for (unsigned j = 0; j < MAX_BARRIERS_PER_CTA; j++) {
        reduction_storage[i][j] = 0;
      }
    }

    // shader core
    m_cluster = cluster;
    m_config = config;
    m_memory_config = mem_config;
    m_stats = stats;
    // unsigned warp_size = config->warp_size;
    Issue_Prio = 0;

    m_sid = shader_id;
    m_tpc = tpc_id;

    // REMOVE: power
    // if (get_gpu()->get_config().g_power_simulation_enabled) {
    //   scaling_coeffs = get_gpu()->get_scaling_coeffs();
    // }

    m_last_inst_gpu_sim_cycle = 0;
    m_last_inst_gpu_tot_sim_cycle = 0;

    // Jin: for concurrent kernels on a SM
    m_occupied_n_threads = 0;
    m_occupied_shmem = 0;
    m_occupied_regs = 0;
    m_occupied_ctas = 0;
    m_occupied_hwtid.reset();
    m_occupied_cta_to_hwtid.clear();

    // trace core
    create_front_pipeline();
    create_shd_warp();
    create_schedulers();
    create_exec_pipeline();
  }

  virtual void checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t,
                                             unsigned tid);
  virtual void init_warps(unsigned cta_id, unsigned start_thread,
                          unsigned end_thread, unsigned ctaid, int cta_size,
                          trace_kernel_info_t &kernel);
  virtual void func_exec_inst(warp_inst_t &inst);
  virtual unsigned sim_init_thread(trace_kernel_info_t &kernel,
                                   ptx_thread_info **thread_info, int sid,
                                   unsigned tid, unsigned threads_left,
                                   unsigned num_threads,
                                   trace_shader_core_ctx *core,
                                   unsigned hw_cta_id, unsigned hw_warp_id,
                                   trace_gpgpu_sim *gpu);
  virtual void create_shd_warp();
  virtual const warp_inst_t *get_next_inst(unsigned warp_id, address_type pc);
  virtual void updateSIMTStack(unsigned warpId, warp_inst_t *inst);
  virtual void get_pdom_stack_top_info(unsigned warp_id, const warp_inst_t *pI,
                                       unsigned *pc, unsigned *rpc);
  virtual const active_mask_t &get_active_mask(unsigned warp_id,
                                               const warp_inst_t *pI);
  virtual void issue_warp(register_set &pipe_reg_set,
                          const warp_inst_t *next_inst,
                          const active_mask_t &active_mask, unsigned warp_id,
                          unsigned sch_id);

  // from "core.hpp"
  class trace_gpgpu_sim *get_gpu() { return m_gpu; }

  // from "shader_core_ctx.hpp"
  void cycle();
  void reinit(unsigned start_thread, unsigned end_thread,
              bool reset_not_completed);
  void cache_invalidate();
  void issue_block2core(class trace_kernel_info_t &kernel);
  void initializeSIMTStack(unsigned warp_count, unsigned warps_size);

  int find_available_hwtid(unsigned int cta_size, bool occupy);
  bool can_issue_1block(trace_kernel_info_t &kernel);
  bool occupy_shader_resource_1block(trace_kernel_info_t &kernel, bool occupy);
  void release_shader_resource_1block(unsigned hw_ctaid,
                                      trace_kernel_info_t &kernel);

  void set_max_cta(const trace_kernel_info_t &kernel);
  unsigned get_not_completed() const { return m_not_completed; }
  unsigned get_n_active_cta() const { return m_n_active_cta; }

  bool fetch_unit_response_buffer_full() const;
  bool ldst_unit_response_buffer_full() const;
  void accept_fetch_response(mem_fetch *mf);
  void accept_ldst_unit_response(class mem_fetch *mf);

  unsigned isactive() const {
    if (m_n_active_cta > 0)
      return 1;
    else
      return 0;
  }
  trace_kernel_info_t *get_kernel() { return m_kernel; }
  void set_kernel(trace_kernel_info_t *k) {
    assert(k);
    m_kernel = k;
    //        k->inc_running();
    logger->debug("GPGPU-Sim uArch: Shader {} bind to kernel {} \'{}\'\n",
                  m_sid, m_kernel->get_uid(), m_kernel->name());
  }

  unsigned current_cycle() {
    return get_gpu()->gpu_sim_cycle + get_gpu()->gpu_tot_sim_cycle;
  }

  void get_L1I_sub_stats(struct cache_sub_stats &css) const;
  void get_L1D_sub_stats(struct cache_sub_stats &css) const;
  void get_L1C_sub_stats(struct cache_sub_stats &css) const;
  void get_L1T_sub_stats(struct cache_sub_stats &css) const;

  void get_cache_stats(cache_stats &cs);
  void print_cache_stats(FILE *fp, unsigned &dl1_accesses,
                         unsigned &dl1_misses);

  void get_icnt_power_stats(long &n_simt_to_mem, long &n_mem_to_simt) const;

  virtual bool warp_waiting_at_barrier(unsigned warp_id) const;
  bool warp_waiting_at_mem_barrier(unsigned warp_id);
  float get_current_occupancy(unsigned long long &active,
                              unsigned long long &total) const;

  void broadcast_barrier_reduction(unsigned cta_id, unsigned bar_id,
                                   warp_set_t warps);

  void mem_instruction_stats(const warp_inst_t &inst);
  void decrement_atomic_count(unsigned wid, unsigned n);
  void dec_inst_in_pipeline(unsigned warp_id) {
    m_warp[warp_id]->dec_inst_in_pipeline();
  }  // also used in writeback()

  void incsfu_stat(unsigned active_count, double latency) {
    m_stats->m_num_sfu_acesses[m_sid] =
        m_stats->m_num_sfu_acesses[m_sid] + (double)active_count * latency;
  }
  void incsp_stat(unsigned active_count, double latency) {
    m_stats->m_num_sp_acesses[m_sid] =
        m_stats->m_num_sp_acesses[m_sid] + (double)active_count * latency;
  }
  void incmem_stat(unsigned active_count, double latency) {
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_mem_acesses[m_sid] =
          m_stats->m_num_mem_acesses[m_sid] + (double)active_count * latency +
          inactive_lanes_accesses_nonsfu(active_count, latency);
    } else {
      m_stats->m_num_mem_acesses[m_sid] =
          m_stats->m_num_mem_acesses[m_sid] + (double)active_count * latency;
    }
  }
  void incexecstat(warp_inst_t *&inst);
  void incregfile_reads(unsigned active_count) {
    m_stats->m_read_regfile_acesses[m_sid] =
        m_stats->m_read_regfile_acesses[m_sid] + active_count;
  }
  void incregfile_writes(unsigned active_count) {
    m_stats->m_write_regfile_acesses[m_sid] =
        m_stats->m_write_regfile_acesses[m_sid] + active_count;
  }

  void incspactivelanes_stat(unsigned active_count) {
    m_stats->m_active_sp_lanes[m_sid] =
        m_stats->m_active_sp_lanes[m_sid] + active_count;
  }
  void incsfuactivelanes_stat(unsigned active_count) {
    m_stats->m_active_sfu_lanes[m_sid] =
        m_stats->m_active_sfu_lanes[m_sid] + active_count;
  }
  void incfuactivelanes_stat(unsigned active_count) {
    m_stats->m_active_fu_lanes[m_sid] =
        m_stats->m_active_fu_lanes[m_sid] + active_count;
  }
  void incfumemactivelanes_stat(unsigned active_count) {
    m_stats->m_active_fu_mem_lanes[m_sid] =
        m_stats->m_active_fu_mem_lanes[m_sid] + active_count;
  }
  void incnon_rf_operands(unsigned active_count) {
    m_stats->m_non_rf_operands[m_sid] =
        m_stats->m_non_rf_operands[m_sid] + active_count;
  }

  void inc_simt_to_mem(unsigned n_flits) {
    m_stats->n_simt_to_mem[m_sid] += n_flits;
  }

  void inc_store_req(unsigned warp_id) { m_warp[warp_id]->inc_store_req(); }
  void store_ack(class mem_fetch *mf);
  void warp_inst_complete(const warp_inst_t &inst);
  std::list<unsigned> get_regs_written(const inst_t &fvt) const;

  unsigned get_sid() const { return m_sid; }
  unsigned get_tpc() const { return m_tpc; }

  const shader_core_config *get_config() const { return m_config; }

  std::shared_ptr<spdlog::logger> logger;
  std::vector<trace_shd_warp_t *> m_warp;  // per warp information array

 protected:
  class trace_gpgpu_sim *m_gpu;
  trace_kernel_info_t *m_kernel;
  simt_stack **m_simt_stack;  // pdom based reconvergence context for each warp
  class ptx_thread_info **m_thread;

  unsigned m_warp_size;
  unsigned m_warp_count;
  unsigned reduction_storage[MAX_CTA_PER_SHADER][MAX_BARRIERS_PER_CTA];

  friend class scheduler_unit;
  friend class TwoLevelScheduler;
  friend class LooseRoundRobbinScheduler;
  friend class accelsim_bridge;
  friend class core_bridge;

  unsigned long long m_last_inst_gpu_sim_cycle;
  unsigned long long m_last_inst_gpu_tot_sim_cycle;

  // general information
  unsigned m_sid;  // shader id
  unsigned m_tpc;  // texture processor cluster id (aka, node id when using
                   // interconnect concentration)
  const shader_core_config *m_config;
  const memory_config *m_memory_config;
  class trace_simt_core_cluster *m_cluster;

  // statistics
  shader_core_stats *m_stats;

  // CTA scheduling / hardware thread allocation
  unsigned m_n_active_cta;  // number of Cooperative Thread Arrays (blocks)
                            // currently running on this shader.
  unsigned m_cta_status[MAX_CTA_PER_SHADER];  // CTAs status
  unsigned m_not_completed;  // number of threads to be completed (==0 when all
                             // thread on this core completed)
  std::bitset<MAX_THREAD_PER_SM> m_active_threads;

  // thread contexts
  thread_ctx_t *m_threadState;

  // interconnect interface
  mem_fetch_interface *m_icnt;
  shader_core_mem_fetch_allocator *m_mem_fetch_allocator;

  // fetch
  read_only_cache *m_L1I;  // instruction cache
  int m_last_warp_fetched;

  // decode/dispatch
  barrier_set_t m_barriers;
  ifetch_buffer_t m_inst_fetch_buffer;
  std::vector<register_set> m_pipeline_reg;
  Scoreboard *m_scoreboard;
  opndcoll_rfu_t m_operand_collector;
  int m_active_warps;
  std::vector<register_set *> m_specilized_dispatch_reg;

  // schedule
  std::vector<scheduler_unit *> schedulers;

  // issue
  unsigned int Issue_Prio;

  // execute
  unsigned m_num_function_units;
  std::vector<unsigned> m_dispatch_port;
  std::vector<unsigned> m_issue_port;
  std::vector<simd_function_unit *>
      m_fu;  // stallable pipelines should be last in this array
  ldst_unit *m_ldst_unit;
  static const unsigned MAX_ALU_LATENCY = 512;
  unsigned num_result_bus;
  std::vector<std::bitset<MAX_ALU_LATENCY> *> m_result_bus;

  // used for local address mapping with single kernel launch
  unsigned kernel_max_cta_per_shader;
  unsigned kernel_padded_threads_per_cta;
  // Used for handing out dynamic warp_ids to new warps.
  // the differnece between a warp_id and a dynamic_warp_id
  // is that the dynamic_warp_id is a running number unique to every warp
  // run on this shader, where the warp_id is the static warp slot.
  unsigned m_dynamic_warp_id;

  // int m_active_warps;
  // unsigned m_n_active_cta;  // number of Cooperative Thread Arrays (blocks)
  //                           // currently running on this shader.
  // unsigned m_not_completed; // number of threads to be completed (==0 when
  // all
  //
  // // Used for handing out dynamic warp_ids to new warps.
  // // the differnece between a warp_id and a dynamic_warp_id
  // // is that the dynamic_warp_id is a running number unique to every warp
  // // run on this shader, where the warp_id is the static warp slot.
  // unsigned m_dynamic_warp_id;
  //
  // const shader_core_config *m_config;

  FILE *stats_out;

 protected:
  void create_front_pipeline();
  void create_schedulers();
  void create_exec_pipeline();

  void writeback();
  void execute();
  void read_operands();
  void issue();
  void decode();
  void fetch();

  int test_res_bus(int latency);
  address_type next_pc(int tid) const;
  void register_cta_thread_exit(unsigned cta_num, trace_kernel_info_t *kernel);

  // Returns numbers of addresses in translated_addrs
  unsigned translate_local_memaddr(address_type localaddr, unsigned tid,
                                   unsigned num_shader, unsigned datasize,
                                   new_addr_type *translated_addrs);

  unsigned inactive_lanes_accesses_sfu(unsigned active_count, double latency) {
    return (((32 - active_count) >> 1) * latency) +
           (((32 - active_count) >> 3) * latency) +
           (((32 - active_count) >> 3) * latency);
  }
  unsigned inactive_lanes_accesses_nonsfu(unsigned active_count,
                                          double latency) {
    return (((32 - active_count) >> 1) * latency);
  }

 private:
  void init_traces(unsigned start_warp, unsigned end_warp,
                   trace_kernel_info_t &kernel);

  unsigned int m_occupied_n_threads;
  unsigned int m_occupied_shmem;
  unsigned int m_occupied_regs;
  unsigned int m_occupied_ctas;
  std::bitset<MAX_THREAD_PER_SM> m_occupied_hwtid;
  std::map<unsigned int, unsigned int> m_occupied_cta_to_hwtid;
};
