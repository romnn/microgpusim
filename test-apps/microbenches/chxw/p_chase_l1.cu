#include <assert.h>
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

// have only 48KB shared mem because we test the 16KB L1?
// const size_t SHMEM_SIZE_BYTES = 16 * (1 << 10);
// const int NUM_LOADS = 512;
// const int NUM_LOADS = 4 * 1024;
// const int NUM_LOADS = 8 * 1024;
const int NUM_LOADS = 6 * 1024;
// const int ITER_SIZE = (SHMEM_SIZE_BYTES / 2) / sizeof(unsigned int);
const int ITER_SIZE = NUM_LOADS;

__global__ void global_latency_l1_data(unsigned int *my_array, int array_length,
                                       unsigned int *duration,
                                       unsigned int *index, size_t warmup) {
  unsigned int start_time, end_time;
  uint32_t j = 0;

  __shared__ uint32_t s_tvalue[ITER_SIZE];
  __shared__ uint32_t s_index[ITER_SIZE];

  for (size_t k = 0; k < ITER_SIZE; k++) {
    s_index[k] = 0;
    s_tvalue[k] = 0;
  }

  for (int k = (int)warmup * -NUM_LOADS; k < NUM_LOADS; k++) {
    if (k >= 0) {
      start_time = clock();
      j = my_array[j];
      s_index[k] = j;
      end_time = clock();

      s_tvalue[k] = end_time - start_time;
    } else {
      j = my_array[j];
    }
  }

  my_array[array_length] = j;
  my_array[array_length + 1] = my_array[j];

  for (size_t k = 0; k < NUM_LOADS; k++) {
    index[k] = s_index[k];
    duration[k] = s_tvalue[k];
  }
}

__global__ void
global_latency_l1_readonly(const unsigned int *__restrict__ my_array,
                           int array_length, unsigned int *duration,
                           unsigned int *index, size_t warmup) {
  unsigned int start_time, end_time;
  uint32_t j = threadIdx.x;

  __shared__ uint32_t s_tvalue[ITER_SIZE];
  __shared__ uint32_t s_index[ITER_SIZE];

  for (size_t k = 0; k < ITER_SIZE; k++) {
    s_index[k] = 0;
    s_tvalue[k] = 0;
  }

  for (int it = (int)warmup * -NUM_LOADS; it < NUM_LOADS; it++) {
    if (it >= 0) {
      int k = it * blockDim.x + threadIdx.x;
      start_time = clock();
      j = __ldg(&my_array[j]);
      s_index[k] = j;
      end_time = clock();

      s_tvalue[k] = end_time - start_time;
    } else {
      j = __ldg(&my_array[j]);
    }
  }

  // my_array[array_length] = j;
  // my_array[array_length + 1] = my_array[j];

  for (size_t it = 0; it < NUM_LOADS; it++) {
    int k = it * blockDim.x + threadIdx.x;
    index[k] = s_index[k];
    duration[k] = s_tvalue[k];
  }
}

void parametric_measure_global(size_t N, size_t stride, size_t warmup) {
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
  // launch kernel
  dim3 block_dim = dim3(1);
  dim3 grid_dim = dim3(1, 1, 1);

  CUDA_SAFECALL((global_latency<<<grid_dim, block_dim>>>(d_a, N, duration,
                                                         d_index, warmup)));

  cudaDeviceSynchronize();

  CUDA_SAFECALL(cudaGetLastError());

  // copy results from GPU to CPU
  cudaDeviceSynchronize();

  CUDA_SAFECALL(cudaMemcpy((void *)h_timeinfo, (void *)duration,
                           sizeof(unsigned int) * NUM_LOADS,
                           cudaMemcpyDeviceToHost));
  CUDA_SAFECALL(cudaMemcpy((void *)h_index, (void *)d_index,
                           sizeof(unsigned int) * NUM_LOADS,
                           cudaMemcpyDeviceToHost));

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
  size_t size_bytes, stride_bytes, warmup;

  // parse arguments
  if (argc > 2) {
    size_bytes = atoi(argv[1]);
    stride_bytes = atoi(argv[2]);
    warmup = atoi(argv[3]);
  } else {
    fprintf(stderr,
            "usage: p_chase_l1 <SIZE_BYTES> <STRIDE_BYTES> <WARMUP> \n");
    return EXIT_FAILURE;
  }

  // the number of resulting patterns P (full iterations through size) is
  // P = NUM_LOADS / stride
  float one_round = (float)size_bytes / (float)stride_bytes;
  float num_rounds = (float)NUM_LOADS / one_round;

  size_t size = size_bytes / sizeof(uint32_t);
  size_t stride = stride_bytes / sizeof(uint32_t);

  fprintf(stderr, "\tSIZE       = %10lu bytes (%10lu uint32, %10.4f KB)\n",
          size_bytes, size, (float)size_bytes / 1024.0);
  fprintf(stderr, "\tSTRIDE     = %10lu bytes (%10lu uint32)\n ", stride_bytes,
          stride);
  fprintf(stderr, "\tROUNDS     = %3.3f\n", num_rounds);
  fprintf(stderr, "\tONE ROUND  = %3.3f (have %5d)\n", one_round, NUM_LOADS);
  fprintf(stderr, "\tWARMUP     = %lu\n", warmup);

  // assert(num_rounds > 1 &&
  //        "array size is too big (rounds should be at least two)");
  // assert(NUM_LOADS > size / stride);

  // validate parameters
  if (size < stride) {
    fprintf(stderr, "ERROR: size (%lu) is smaller than stride (%lu)\n", size,
            stride);
    fflush(stderr);
    return EXIT_FAILURE;
  }
  // if (size % stride != 0) {
  //   fprintf(stderr,
  //           "ERROR: size (%lu) is not an exact multiple of stride (%lu)\n",
  //           size, stride);
  //   fflush(stderr);
  //   return EXIT_FAILURE;
  // }
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

  assert(ITER_SIZE >= NUM_LOADS);

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

  cudaFuncCache want_cache_config = cudaFuncCachePreferShared;
  // cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  cudaDeviceSetCacheConfig(want_cache_config);
  cudaFuncCache have_cache_config;
  CUDA_SAFECALL(cudaDeviceGetCacheConfig(&have_cache_config));
  assert(want_cache_config == have_cache_config);

  parametric_measure_global(size, stride, warmup);

  cudaDeviceReset();
  fflush(stdout);
  fflush(stderr);
  return EXIT_SUCCESS;
}
