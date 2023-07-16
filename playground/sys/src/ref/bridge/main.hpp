#pragma once

// #include <utility>
#include <list>

#include "accelsim_config.hpp"
#include "accelsim_stats.hpp"
#include "rust/cxx.h"
#include "playground-sys/src/bridge/stats.rs.h"

#include "./stats.hpp"
#include "../trace_parser.hpp"
#include "../trace_config.hpp"
#include "../trace_command.hpp"
#include "../trace_kernel_info.hpp"
#include "../memory_partition_unit.hpp"
#include "../memory_sub_partition.hpp"
#include "../trace_gpgpu_sim.hpp"

class mem_fetch_bridge {
 public:
  mem_fetch_bridge(mem_fetch *ptr) : ptr(ptr) {}

  mem_fetch *get_mem_fetch() const { return ptr; }

 private:
  class mem_fetch *ptr;
};

class memory_partition_unit_bridge {
 public:
  memory_partition_unit_bridge(memory_partition_unit *ptr) : ptr(ptr) {}

  std::unique_ptr<std::vector<mem_fetch_bridge>> get_dram_latency_queue()
      const {
    std::vector<mem_fetch_bridge> q;
    std::list<memory_partition_unit::dram_delay_t>::const_iterator iter;
    for (iter = (ptr->m_dram_latency_queue).begin();
         iter != (ptr->m_dram_latency_queue).end(); iter++) {
      q.push_back(mem_fetch_bridge(iter->req));
    }
    return std::make_unique<std::vector<mem_fetch_bridge>>(q);
  }

 private:
  class memory_partition_unit *ptr;
};

class memory_sub_partition_bridge {
 public:
  memory_sub_partition_bridge(memory_sub_partition *ptr) : ptr(ptr) {}

  std::unique_ptr<std::vector<mem_fetch_bridge>> get_queue(
      fifo_pipeline<mem_fetch> *fifo) const {
    std::vector<mem_fetch_bridge> q;
    if (fifo != NULL) {
      fifo_data<mem_fetch> *ddp = fifo->m_head;
      while (ddp) {
        q.push_back(mem_fetch_bridge(ddp->m_data));
        ddp = ddp->m_next;
      }
    }
    return std::make_unique<std::vector<mem_fetch_bridge>>(q);
  }

  std::unique_ptr<std::vector<mem_fetch_bridge>> get_icnt_L2_queue() const {
    return get_queue(ptr->m_icnt_L2_queue);
  }
  std::unique_ptr<std::vector<mem_fetch_bridge>> get_L2_dram_queue() const {
    return get_queue(ptr->m_L2_dram_queue);
  }
  std::unique_ptr<std::vector<mem_fetch_bridge>> get_dram_L2_queue() const {
    return get_queue(ptr->m_dram_L2_queue);
  }
  std::unique_ptr<std::vector<mem_fetch_bridge>> get_L2_icnt_queue() const {
    return get_queue(ptr->m_L2_icnt_queue);
  }

 private:
  class memory_sub_partition *ptr;
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

  const std::vector<memory_sub_partition_bridge> &get_sub_partitions_vec()
      const {
    return sub_partitions;
  }

  const std::vector<memory_partition_unit_bridge> &get_partition_units_vec()
      const {
    return partition_units;
  }

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
};

std::unique_ptr<accelsim_bridge> new_accelsim_bridge(
    accelsim_config config, rust::Slice<const rust::Str> argv);
