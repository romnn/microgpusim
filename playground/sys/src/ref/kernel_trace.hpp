#pragma once

#include <string>

struct kernel_trace_t {
  kernel_trace_t();

  std::string kernel_name;
  unsigned kernel_id;
  unsigned grid_dim_x;
  unsigned grid_dim_y;
  unsigned grid_dim_z;
  unsigned tb_dim_x;
  unsigned tb_dim_y;
  unsigned tb_dim_z;
  unsigned shmem;
  unsigned nregs;
  unsigned long cuda_stream_id;
  unsigned binary_verion;
  unsigned enable_lineinfo;
  unsigned trace_verion;
  std::string nvbit_verion;
  unsigned long long shmem_base_addr;
  unsigned long long local_base_addr;
  // Reference to open filestream
  std::ifstream *ifs;

  const std::string &get_kernel_name() const { return kernel_name; }
  unsigned get_kernel_id() const { return kernel_id; }
  unsigned get_grid_dim_x() const { return grid_dim_x; }
  unsigned get_grid_dim_y() const { return grid_dim_y; }
  unsigned get_grid_dim_z() const { return grid_dim_z; }
  unsigned get_block_dim_x() const { return tb_dim_x; }
  unsigned get_block_dim_y() const { return tb_dim_y; }
  unsigned get_block_dim_z() const { return tb_dim_z; }
  unsigned get_shared_mem_bytes() const { return shmem; }
  unsigned get_num_registers() const { return nregs; }
  unsigned long get_cuda_stream_id() const { return cuda_stream_id; }
  // unsigned binary_verion;
  // unsigned enable_lineinfo;
  // unsigned trace_verion;
  // std::string nvbit_verion;
  // unsigned long long shmem_base_addr;
  // unsigned long long local_base_addr;
  // Reference to open filestream
  // std::ifstream *ifs;
};
