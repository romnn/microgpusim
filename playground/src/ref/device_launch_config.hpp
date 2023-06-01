#pragma once

#include "dim3.hpp"

class function_info;

class device_launch_config_t {
public:
  device_launch_config_t() {}

  device_launch_config_t(dim3 _grid_dim, dim3 _block_dim,
                         unsigned int _shared_mem, function_info *_entry)
      : grid_dim(_grid_dim), block_dim(_block_dim), shared_mem(_shared_mem),
        entry(_entry) {}

  dim3 grid_dim;
  dim3 block_dim;
  unsigned int shared_mem;
  function_info *entry;
};
