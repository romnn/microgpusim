// every nvbit tool must include this once to initialize tracing
#include "assert.h"
#include "nvbit_tool.h"
#include <cstdio>

#include "utils/channel.hpp"

#include "common.h"

#define CUDA_CHECK(A) check_cuda_error((cudaError)(A), __FILE__, __LINE__);

void check_cuda_error(cudaError error, const char *file, int line) {
  if (cudaSuccess != error) {
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", file, line,
            cudaGetErrorString(error));
    fflush(stderr);
  }
  assert(error == cudaSuccess);
}

__global__ __noinline__ void flush_channel_kernel(ChannelDev *ch_dev) {
  // set a CTA id = -1 as termination flag for the receiver thread
  mem_access_t ma;
  ma.block_id_x = -1;
  ch_dev->push(&ma, sizeof(mem_access_t));
  ch_dev->flush();
}

extern "C" __noinline__ void flush_channel(void *channel_dev) {
  CUDA_CHECK(cudaGetLastError());
  flush_channel_kernel<<<1, 1>>>((ChannelDev *)channel_dev);
  CUDA_CHECK(cudaGetLastError());
  cudaDeviceSynchronize();
  CUDA_CHECK(cudaGetLastError());
}

extern "C" __noinline__ void cuda_free(void *dev_ptr) {
  CUDA_CHECK(cudaFree(dev_ptr));
}

// extern "C" __noinline__ void deallocate_reg_info(reg_info_t *dev_info) {
//   CUDA_CHECK(cudaFree(dev_info));
// }
//
extern "C" __noinline__ reg_info_t *allocate_reg_info(reg_info_t host_info) {
  reg_info_t *dev_info;
  size_t bytes = sizeof(reg_info_t);
  CUDA_CHECK(cudaMalloc(&dev_info, bytes));
  CUDA_CHECK(cudaMemcpy(dev_info, &host_info, bytes, cudaMemcpyHostToDevice));
  return dev_info;
}
