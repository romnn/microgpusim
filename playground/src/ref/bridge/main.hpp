#pragma once

#include <utility>

#include "accelsim_config.hpp"
#include "accelsim_stats.hpp"
#include "rust/cxx.h"
#include "playground/src/bridge/stats.rs.h"

#include "./stats.hpp"
#include "../trace_parser.hpp"
#include "../trace_config.hpp"
#include "../trace_command.hpp"
#include "../trace_kernel_info.hpp"
#include "../memory_sub_partition.hpp"
#include "../trace_gpgpu_sim.hpp"

// struct SharedMemorySubPartition;
// struct MemorySubPartitionShim;

// fifo_pipeline<mem_fetch> *m_icnt_L2_queue;
// fifo_pipeline<mem_fetch> *m_L2_dram_queue;
// fifo_pipeline<mem_fetch> *m_dram_L2_queue;
// fifo_pipeline<mem_fetch> *m_L2_icnt_queue;  // L2 cache hit response queue

class mem_fetch_bridge {
 public:
  mem_fetch_bridge(mem_fetch *ptr) : ptr(ptr) {}

  mem_fetch *get_mem_fetch() const { return ptr; }

 private:
  class mem_fetch *ptr;
};

class memory_sub_partition_bridge {
 public:
  memory_sub_partition_bridge(memory_sub_partition *ptr) : ptr(ptr) {}

  // std::vector<mem_fetch *> get_icnt_L2_queue() const {
  //   return ptr->m_icnt_L2_queue->to_vector();
  // }

  // std::vector<mem_fetch_bridge> get_icnt_L2_queue() const {
  std::unique_ptr<std::vector<mem_fetch_bridge>> get_icnt_L2_queue() const {
    // rust::Vec<mem_fetch_bridge> get_icnt_L2_queue() const {
    fifo_pipeline<mem_fetch> *fifo = ptr->m_icnt_L2_queue;
    // std::vector<mem_fetch_bridge> q;
    std::vector<mem_fetch_bridge> q;
    // std::unique_ptr<std::vector<mem_fetch_bridge>> q;
    // rust::Vec<mem_fetch_bridge> q;
    if (fifo != NULL) {
      // return std::unique_ptr<std::vector<mem_fetch_bridge>>(q);
      // return std::make_unique<std::vector<mem_fetch_bridge>>(q);
      // return q;
      fifo_data<mem_fetch> *ddp = fifo->m_head;
      while (ddp) {
        // q.push_back(mem_fetch_bridge{ddp->m_data});
        // q.get()->push_back(mem_fetch_bridge(ddp->m_data));
        q.push_back(mem_fetch_bridge(ddp->m_data));
        ddp = ddp->m_next;
      }
    }
    // return q;
    return std::make_unique<std::vector<mem_fetch_bridge>>(q);
    // return std::unique_ptr<std::vector<mem_fetch_bridge>>(q);
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

  const memory_sub_partition *const *get_sub_partitions() const {
    return const_cast<const memory_sub_partition *const *>(
        m_gpgpu_sim->m_memory_sub_partition);
  }
  // const rust::Vec<MemorySubPartitionShim> &get_sub_partitions_vec() const;
  const std::vector<memory_sub_partition_bridge> &get_sub_partitions_vec()
      const {
    return sub_partitions;
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
  // rust::Vec<MemorySubPartitionShim> sub_partitions;
  // rust::Vec<memory_sub_partition_shim> sub_partitions;
  std::vector<memory_sub_partition_bridge> sub_partitions;
};

std::unique_ptr<accelsim_bridge> new_accelsim_bridge(
    accelsim_config config, rust::Slice<const rust::Str> argv);
