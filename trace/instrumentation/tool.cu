// every nvbit tool must include this once to initialize tracing
#include "nvbit_tool.h"

#include "utils/channel.hpp"

#include "common.h"

__global__ __noinline__ void flush_channel_kernel(ChannelDev *ch_dev) {
  /* set a CTA id = -1 to indicate communication thread that this is the
   * termination flag */
  mem_access_t ma;
  ma.cta_id_x = -1;
  ch_dev->push(&ma, sizeof(mem_access_t));
  ch_dev->flush();
}

extern "C" __noinline__ void flush_channel(void *channel_dev) {
  flush_channel_kernel<<<1, 1>>>((ChannelDev *)channel_dev);
  cudaDeviceSynchronize();
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__,
            __LINE__, cudaGetErrorString(err));
    fflush(stderr);
  }
  assert(err == cudaSuccess);
}
