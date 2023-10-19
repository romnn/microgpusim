#include "common.hpp"

__global__ __noinline__ void
global_measure_clock_overhead(unsigned int *clock_cycles) {
  const size_t iterations = 200;
  __shared__ unsigned int s_overhead[2 * iterations];

  for (size_t i = 0; i < 2 * iterations; i++) {
    volatile unsigned int start_time = clock();
    unsigned int end_time = clock();
    s_overhead[i] = (end_time - start_time);
  }

  unsigned int sum = 0;
  for (size_t i = 0; i < iterations; i++) {
    sum += s_overhead[iterations + i];
  }
  *clock_cycles = float(sum) / float(iterations);
}

unsigned int measure_clock_overhead() {
  uint32_t *h_clock_overhead = (uint32_t *)malloc(sizeof(uint32_t) * 1);

  uint32_t *d_clock_overhead;
  CUDA_SAFECALL(cudaMalloc((void **)&d_clock_overhead, sizeof(uint32_t) * 1));

  // launch kernel
  dim3 block_dim = dim3(1);
  dim3 grid_dim = dim3(1, 1, 1);

  CUDA_SAFECALL((global_measure_clock_overhead<<<grid_dim, block_dim>>>(
      d_clock_overhead)));

  CUDA_SAFECALL(cudaMemcpy((void *)h_clock_overhead, (void *)d_clock_overhead,
                           sizeof(uint32_t) * 1, cudaMemcpyDeviceToHost));

  fprintf(stderr, "clock overhead is %u cycles\n", *h_clock_overhead);

  return *h_clock_overhead;
}
