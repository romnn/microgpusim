#pragma once

#include <list>
#include <map>

#include "accelsim_config.hpp"
#include "accelsim_stats.hpp"
#include "rust/cxx.h"
#include "playground-sys/src/bridge/stats.rs.h"

#include "./stats.hpp"
#include "../scheduler_unit.hpp"
#include "../trace_parser.hpp"
#include "../trace_config.hpp"
#include "../trace_command.hpp"
#include "../trace_kernel_info.hpp"
#include "../opndcoll_rfu.hpp"
#include "../memory_partition_unit.hpp"
#include "../memory_sub_partition.hpp"
#include "../trace_shader_core_ctx.hpp"
#include "../trace_gpgpu_sim.hpp"

struct mem_fetch_ptr {
  const mem_fetch *ptr;
  const mem_fetch *get() const { return ptr; }
};

class mem_fetch_bridge {
 public:
  mem_fetch_bridge(const mem_fetch *ptr) : ptr(ptr) {}

  const mem_fetch *inner() const { return ptr; }

 private:
  const class mem_fetch *ptr;
};
std::shared_ptr<mem_fetch_bridge> new_mem_fetch_bridge(const mem_fetch *ptr);

struct warp_inst_ptr {
  const warp_inst_t *ptr;
  const warp_inst_t *get() const { return ptr; }
};

class warp_inst_bridge {
 public:
  warp_inst_bridge(const warp_inst_t *ptr) : ptr(ptr){};

  const warp_inst_t *inner() const { return ptr; }

 private:
  const warp_inst_t *ptr;
};
std::shared_ptr<warp_inst_bridge> new_warp_inst_bridge(const warp_inst_t *ptr);

struct register_set_ptr {
  const register_set *ptr;
  const register_set *get() const { return ptr; }
};

class register_set_bridge {
 public:
  register_set_bridge(const register_set *ptr) : ptr(ptr) {}

  const register_set *inner() const { return ptr; };

  std::unique_ptr<std::vector<warp_inst_ptr>> get_registers() const {
    std::vector<warp_inst_ptr> out;
    std::vector<warp_inst_t *>::const_iterator iter;
    for (iter = (ptr->regs).begin(); iter != (ptr->regs).end(); iter++) {
      out.push_back(warp_inst_ptr{*iter});
    }
    return std::make_unique<std::vector<warp_inst_ptr>>(out);
  }

 private:
  const register_set *ptr;
};

std::shared_ptr<register_set_bridge> new_register_set_bridge(
    const register_set *ptr);

class input_port_bridge {
 public:
  input_port_bridge(const input_port_t *ptr) : ptr(ptr) {}

  const input_port_t *inner() const { return ptr; }

  std::unique_ptr<std::vector<register_set_ptr>> get_in_ports() const {
    std::vector<register_set_ptr> out;
    std::vector<register_set *>::const_iterator iter;
    for (iter = (ptr->m_in).begin(); iter != (ptr->m_in).end(); iter++) {
      out.push_back(register_set_ptr{*iter});
    }
    return std::make_unique<std::vector<register_set_ptr>>(out);
  }

  std::unique_ptr<std::vector<register_set_ptr>> get_out_ports() const {
    std::vector<register_set_ptr> out;
    std::vector<register_set *>::const_iterator iter;
    for (iter = (ptr->m_out).begin(); iter != (ptr->m_out).end(); iter++) {
      out.push_back(register_set_ptr{*iter});
    }
    return std::make_unique<std::vector<register_set_ptr>>(out);
  }

  const uint_vector_t &get_cu_sets() const { return ptr->m_cu_sets; }

 private:
  const class input_port_t *ptr;
};

std::shared_ptr<input_port_bridge> new_input_port_bridge(
    const input_port_t *ptr);

struct collector_unit_set {
  unsigned collector_set;
  const collector_unit_t &unit;
  const collector_unit_t &get_unit() const { return unit; }
  unsigned get_set() const { return collector_set; }
};

class operand_collector_bridge {
 public:
  operand_collector_bridge(const opndcoll_rfu_t *ptr) : ptr(ptr) {}

  const opndcoll_rfu_t *inner() const { return ptr; }

  const std::vector<input_port_t> &get_input_ports() const {
    return ptr->m_in_ports;
  }

  const std::vector<dispatch_unit_t> &get_dispatch_units() const {
    return ptr->m_dispatch_units;
  }

  std::unique_ptr<std::vector<collector_unit_set>> get_collector_units() const {
    // collector set, collector sets
    std::vector<collector_unit_set> out;
    std::map<unsigned, std::vector<collector_unit_t>>::const_iterator iter;
    for (iter = (ptr->m_cus).begin(); iter != (ptr->m_cus).end(); iter++) {
      const std::vector<collector_unit_t> &cus = iter->second;
      std::vector<collector_unit_t>::const_iterator cu;
      for (cu = cus.begin(); cu != cus.end(); cu++) {
        out.push_back(collector_unit_set{iter->first, *cu});
      }
    }
    return std::make_unique<std::vector<collector_unit_set>>(out);
  }

 private:
  const class opndcoll_rfu_t *ptr;
};

class memory_partition_unit_bridge {
 public:
  memory_partition_unit_bridge(memory_partition_unit *ptr) : ptr(ptr) {}

  std::unique_ptr<std::vector<mem_fetch_ptr>> get_dram_latency_queue() const {
    std::vector<mem_fetch_ptr> q;
    std::list<memory_partition_unit::dram_delay_t>::const_iterator iter;
    for (iter = (ptr->m_dram_latency_queue).begin();
         iter != (ptr->m_dram_latency_queue).end(); iter++) {
      q.push_back(mem_fetch_ptr{iter->req});
    }
    return std::make_unique<std::vector<mem_fetch_ptr>>(q);
  }

 private:
  class memory_partition_unit *ptr;
};

class memory_sub_partition_bridge {
 public:
  memory_sub_partition_bridge(const memory_sub_partition *ptr) : ptr(ptr) {}

  std::unique_ptr<std::vector<mem_fetch_ptr>> get_queue(
      fifo_pipeline<mem_fetch> *fifo) const {
    std::vector<mem_fetch_ptr> q;
    if (fifo != NULL) {
      fifo_data<mem_fetch> *ddp = fifo->m_head;
      while (ddp) {
        q.push_back(mem_fetch_ptr{ddp->m_data});
        ddp = ddp->m_next;
      }
    }
    return std::make_unique<std::vector<mem_fetch_ptr>>(q);
  }

  std::unique_ptr<std::vector<mem_fetch_ptr>> get_icnt_L2_queue() const {
    return get_queue(ptr->m_icnt_L2_queue);
  }
  std::unique_ptr<std::vector<mem_fetch_ptr>> get_L2_dram_queue() const {
    return get_queue(ptr->m_L2_dram_queue);
  }
  std::unique_ptr<std::vector<mem_fetch_ptr>> get_dram_L2_queue() const {
    return get_queue(ptr->m_dram_L2_queue);
  }
  std::unique_ptr<std::vector<mem_fetch_ptr>> get_L2_icnt_queue() const {
    return get_queue(ptr->m_L2_icnt_queue);
  }

 private:
  const class memory_sub_partition *ptr;
};

struct scheduler_unit_ptr {
  const scheduler_unit *ptr;
  const scheduler_unit *get() const { return ptr; }
};

class scheduler_unit_bridge {
 public:
  scheduler_unit_bridge(const scheduler_unit *ptr) : ptr(ptr) {}

  const scheduler_unit *inner() const { return ptr; }

  std::unique_ptr<std::vector<unsigned>> get_prioritized_warp_ids() const {
    std::vector<unsigned> out;
    const std::vector<trace_shd_warp_t *> &warps =
        ptr->m_next_cycle_prioritized_warps;
    std::vector<trace_shd_warp_t *>::const_iterator iter;
    for (iter = warps.begin(); iter != warps.end(); iter++) {
      out.push_back((*iter)->get_warp_id());
    }
    return std::make_unique<std::vector<unsigned>>(out);
  }

 private:
  const class scheduler_unit *ptr;
};

std::shared_ptr<scheduler_unit_bridge> new_scheduler_unit_bridge(
    const scheduler_unit *ptr);

class core_bridge {
 public:
  core_bridge(const trace_shader_core_ctx *ptr) : ptr(ptr) {}

  std::unique_ptr<std::vector<register_set_ptr>> get_register_sets() const {
    std::vector<register_set_ptr> out;
    for (unsigned n = 0; n < ptr->m_num_function_units; n++) {
      unsigned int issue_port = ptr->m_issue_port[n];
      const register_set &issue_reg = ptr->m_pipeline_reg[issue_port];
      bool is_sp = issue_port == ID_OC_SP || issue_port == OC_EX_SP;
      bool is_mem = issue_port == ID_OC_MEM || issue_port == OC_EX_MEM;
      if (is_sp || is_mem) {
        out.push_back(register_set_ptr{std::addressof(issue_reg)});
      }
    }

    return std::make_unique<std::vector<register_set_ptr>>(out);
  }

  std::unique_ptr<std::vector<scheduler_unit_ptr>> get_scheduler_units() const {
    std::vector<scheduler_unit_ptr> out;
    std::vector<scheduler_unit *>::const_iterator iter;
    for (iter = (ptr->schedulers).begin(); iter != (ptr->schedulers).end();
         iter++) {
      out.push_back(scheduler_unit_ptr{*iter});
    }
    return std::make_unique<std::vector<scheduler_unit_ptr>>(out);
  }

  std::shared_ptr<operand_collector_bridge> get_operand_collector() const {
    return std::make_shared<operand_collector_bridge>(
        &(ptr->m_operand_collector));
  }

 private:
  const class trace_shader_core_ctx *ptr;
};

class accelsim_bridge {
 public:
  accelsim_bridge(accelsim_config config, rust::Slice<const rust::Str> argv);

  void run_to_completion();
  void process_commands();
  void launch_kernels();
  void cycle();
  void cleanup_finished_kernel(unsigned finished_kernel_uid);

  uint64_t get_cycle() const { return m_gpgpu_sim->gpu_sim_cycle; }
  bool active() const { return m_gpgpu_sim->active(); };
  unsigned get_finished_kernel_uid() { return m_gpgpu_sim->finished_kernel(); };
  bool limit_reached() const { return m_gpgpu_sim->cycle_insn_cta_max_hit(); };
  bool commands_left() const { return command_idx < commandlist.size(); };
  bool active_kernels() const { return kernels_info.size(); };
  bool kernels_left() const { return !kernels_info.empty(); };

  // stats transfer
  void transfer_stats(StatsBridge &stats) const;
  void transfer_dram_stats(StatsBridge &stats) const;
  void transfer_general_stats(StatsBridge &stats) const;
  void transfer_core_cache_stats(StatsBridge &stats) const;
  void transfer_l2d_stats(StatsBridge &stats) const;

  const std::vector<memory_sub_partition_bridge> &get_sub_partitions() const {
    return sub_partitions;
  }

  const std::vector<memory_partition_unit_bridge> &get_partition_units() const {
    return partition_units;
  }

  const std::vector<core_bridge> &get_cores() const { return cores; }

 private:
  trace_parser *tracer;
  trace_config tconfig;
  trace_gpgpu_sim *m_gpgpu_sim;
  gpgpu_context *m_gpgpu_context;

  std::vector<trace_command> commandlist;
  std::vector<unsigned long> busy_streams;
  std::vector<trace_kernel_info_t *> kernels_info;

  unsigned command_idx;
  unsigned window_size;
  bool silent;

  // for handing out references to components
  std::vector<memory_sub_partition_bridge> sub_partitions;
  std::vector<memory_partition_unit_bridge> partition_units;
  std::vector<core_bridge> cores;
};

std::unique_ptr<accelsim_bridge> new_accelsim_bridge(
    accelsim_config config, rust::Slice<const rust::Str> argv);
