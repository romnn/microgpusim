#include <cstdlib>
#include <stdint.h>
#include <stdio.h>

#include "cuda_runtime.h"

#define CUDA_SAFECALL(call)                                                    \
  {                                                                            \
    call;                                                                      \
    cudaError err = cudaGetLastError();                                        \
    if (cudaSuccess != err) {                                                  \
      fprintf(stderr,                                                          \
              "Cuda error in function '%s' file '%s' in line %i : %s.\n",      \
              #call, __FILE__, __LINE__, cudaGetErrorString(err));             \
      fflush(stderr);                                                          \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

// have 96 KB shared memory, sizeof(unsigned)
// const static size_t SHMEM_SIZE_BYTES = 0xC000;
// const static size_t NUM_LOADS = (SHMEM_SIZE_BYTES / 2) / sizeof(unsigned
// int);
// const static int NUM_LOADS = 512;
const static int NUM_LOADS = 4096;

__global__ void global_latency(unsigned int *my_array, int array_length,
                               int iterations, unsigned int *duration,
                               unsigned int *index) {
  unsigned int start_time, end_time;
  unsigned int j = 0;

  __shared__ unsigned int s_tvalue[NUM_LOADS];
  __shared__ unsigned int s_index[NUM_LOADS];

  for (size_t k = 0; k < NUM_LOADS; k++) {
    s_index[k] = 0;
    s_tvalue[k] = 0;
  }

  // first round
  //	for (k = 0; k < iterations*256; k++)
  //		j = my_array[j];

  // printf("k=%d..%d\n", -iterations * NUM_LOADS, iterations * NUM_LOADS);

  // second round
  for (int k = -iterations * NUM_LOADS; k < iterations * NUM_LOADS; k++) {
    if (k >= 0) {
      start_time = clock();
      j = my_array[j];
      s_index[k] = j;
      end_time = clock();

      s_tvalue[k] = end_time - start_time;

    } else
      j = my_array[j];
  }

  my_array[array_length] = j;
  my_array[array_length + 1] = my_array[j];

  for (size_t k = 0; k < NUM_LOADS; k++) {
    index[k] = s_index[k];
    duration[k] = s_tvalue[k];
  }
}

void parametric_measure_global(size_t N, size_t stride, size_t iterations) {
  cudaDeviceReset();

  // print CSV header
  fprintf(stdout, "index,latency\n");

  // allocate arrays on CPU
  unsigned int *h_a;
  h_a = (unsigned int *)malloc(sizeof(unsigned int) * (N + 2));

  // allocate arrays on GPU
  unsigned int *d_a;
  CUDA_SAFECALL(cudaMalloc((void **)&d_a, sizeof(unsigned int) * (N + 2)));

  // initialize array elements on CPU with pointers into d_a
  for (size_t i = 0; i < N; i++) {
    // original:
    h_a[i] = (i + stride) % N;
  }

  h_a[N] = 0;
  h_a[N + 1] = 0;

  // copy array elements from CPU to GPU
  CUDA_SAFECALL(
      cudaMemcpy(d_a, h_a, sizeof(unsigned int) * N, cudaMemcpyHostToDevice));

  unsigned int *h_index =
      (unsigned int *)malloc(sizeof(unsigned int) * NUM_LOADS);
  unsigned int *h_timeinfo =
      (unsigned int *)malloc(sizeof(unsigned int) * NUM_LOADS);

  unsigned int *duration;
  CUDA_SAFECALL(
      cudaMalloc((void **)&duration, sizeof(unsigned int) * NUM_LOADS));

  unsigned int *d_index;
  CUDA_SAFECALL(
      cudaMalloc((void **)&d_index, sizeof(unsigned int) * NUM_LOADS));

  cudaDeviceSynchronize();
  // cudaThreadSynchronize();
  // launch kernel
  dim3 block_dim = dim3(1);
  dim3 grid_dim = dim3(1, 1, 1);

  CUDA_SAFECALL((global_latency<<<grid_dim, block_dim>>>(d_a, N, iterations,
                                                         duration, d_index)));

  // cudaThreadSynchronize();
  cudaDeviceSynchronize();

  CUDA_SAFECALL(cudaGetLastError());

  // copy results from GPU to CPU
  // cudaThreadSynchronize();
  cudaDeviceSynchronize();

  CUDA_SAFECALL(cudaMemcpy((void *)h_timeinfo, (void *)duration,
                           sizeof(unsigned int) * NUM_LOADS,
                           cudaMemcpyDeviceToHost));
  CUDA_SAFECALL(cudaMemcpy((void *)h_index, (void *)d_index,
                           sizeof(unsigned int) * NUM_LOADS,
                           cudaMemcpyDeviceToHost));

  // cudaThreadSynchronize();
  cudaDeviceSynchronize();

  for (size_t i = 0; i < NUM_LOADS; i++) {
    // print as CSV to stdout
    fprintf(stdout, "%4d,%4d\n", h_index[i], h_timeinfo[i]);
  }

  // free memory on GPU
  cudaFree(d_a);
  cudaFree(d_index);
  cudaFree(duration);

  // free memory on CPU
  free(h_a);
  free(h_index);
  free(h_timeinfo);

  cudaDeviceReset();
}

int main(int argc, char *argv[]) {
  cudaSetDevice(0);
  size_t size_bytes, iterations, stride_bytes;

  // parse arguments
  if (argc > 2) {
    size_bytes = atoi(argv[1]);
    stride_bytes = atoi(argv[2]);
    iterations = atoi(argv[3]);
  } else {
    fprintf(stderr,
            "usage: p_chase_l1 <SIZE_BYTES> <STRIDE_BYTES> <ITERATIONS> \n");
    return EXIT_FAILURE;
  }

  // the number of resulting patterns P (full iterations through size) is
  // P = NUM_LOADS / stride

  size_t size = size_bytes / sizeof(uint32_t);
  size_t stride = stride_bytes / sizeof(uint32_t);

  fprintf(stderr, "\tSIZE       = %10lu bytes (%10lu uint32, %10.4f KB)\n",
          size_bytes, size, (float)size_bytes / 1024.0);
  fprintf(stderr, "\tSTRIDE     = %10lu bytes (%10lu uint32)\n ", stride_bytes,
          stride);
  fprintf(stderr, "\tITERATIONS = %lu\n", iterations);

  // validate parameters
  if (size < stride) {
    fprintf(stderr, "ERROR: size (%lu) is smaller than stride (%lu)\n", size,
            stride);
    fflush(stderr);
    return EXIT_FAILURE;
  }
  if (size % stride != 0) {
    fprintf(stderr,
            "ERROR: size (%lu) is not an exact multiple of stride (%lu)\n",
            size, stride);
    fflush(stderr);
    return EXIT_FAILURE;
  }
  if (size < 1) {
    fprintf(stderr, "ERROR: size is < 1 (%lu)\n", size);
    fflush(stderr);
    return EXIT_FAILURE;
  }
  if (stride < 1) {
    fprintf(stderr, "ERROR: stride is < 1 (%lu)\n", stride);
    fflush(stderr);
    return EXIT_FAILURE;
  }

  // printf("\n=====%10.4f KB array, warm TLB, read NUM_LOADS element====\n",
  //          sizeof(unsigned int) * (float)N / 1024);
  //   printf("Stride = %d element, %d byte\n", stride,
  //          stride * sizeof(unsigned int));

  // The `cudaDeviceSetCacheConfig` function can be used to set preference for
  // shared memory or L1 cache globally for all CUDA kernels in your code and
  // even those used by Thrust.
  // The option cudaFuncCachePreferShared prefers shared memory, that is,
  // it sets 48 KB for shared memory and 16 KB for L1 cache.
  //
  // `cudaFuncCachePreferL1` prefers L1, that is, it sets 16 KB for
  // shared memory and 48 KB for L1 cache.
  //
  // `cudaFuncCachePreferNone` uses the preference set for the device or thread.

  parametric_measure_global(size, stride, iterations);

  // stride in element
  // iterations = 1;

  // 1. overflow cache with 1 element. stride=1, N=4097
  // 2. overflow cache with cache lines. stride=32, N_min=16*256, N_max=24*256
  // stride = 128 / sizeof(unsigned int);
  //
  // for (N = 16 * 256; N <= 24 * 256; N += stride) {
  //   printf("\n=====%10.4f KB array, warm TLB, read NUM_LOADS element====\n",
  //          sizeof(unsigned int) * (float)N / 1024);
  //   printf("Stride = %d element, %d byte\n", stride,
  //          stride * sizeof(unsigned int));
  //   parametric_measure_global(N, iterations, stride);
  //   printf("===============================================\n\n");
  // }

  cudaDeviceReset();
  fflush(stdout);
  fflush(stderr);
  return EXIT_SUCCESS;
}
