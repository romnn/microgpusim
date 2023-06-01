#pragma once

class CUstream_st;
class kernel_info_t;

class device_launch_operation_t {
public:
  device_launch_operation_t() {}
  device_launch_operation_t(kernel_info_t *_grid, CUstream_st *_stream)
      : grid(_grid), stream(_stream) {}

  kernel_info_t *grid; // a new child grid

  CUstream_st *stream;
};
