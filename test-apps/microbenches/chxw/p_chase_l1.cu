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

const bool USE_COMPRESSION = false;

const int KB = 1024;

// have 96 KB shared memory, sizeof(unsigned)
// const static size_t SHMEM_SIZE_BYTES = 0xC000;

// have only 48KB shared mem because we test the 16KB L1?
// const size_t SHMEM_SIZE_BYTES = 16 * (1 << 10);
// const int ITER_SIZE = 512;
// const int ITER_SIZE = 4 * 1024;
// const int ITER_SIZE = 8 * 1024;
// const int ITER_SIZE = 6 * 1024;
// const int ITER_SIZE = (SHMEM_SIZE_BYTES / 2) / sizeof(unsigned int);

// const int ITER_SIZE = 6 * 1024;
const int ITER_SIZE = ((48 * KB) / 2) / sizeof(uint32_t);

__global__ void global_latency_l1_data(unsigned int *array, int array_length,
                                       unsigned int *duration,
                                       unsigned int *index,
                                       size_t warmup_iterations) {
  unsigned int start_time, end_time;
  uint32_t warmup_j = 0;
  uint32_t j = 0;

  __shared__ uint32_t s_tvalue[ITER_SIZE];
  __shared__ uint32_t s_index[ITER_SIZE];

  for (size_t k = 0; k < ITER_SIZE; k++) {
    s_index[k] = 0;
    s_tvalue[k] = 0;
  }

  for (int k = (int)warmup_iterations * -ITER_SIZE; k < ITER_SIZE; k++) {
    if (k >= 0) {
      start_time = clock();
      j = array[j];
      s_index[k] = j;
      end_time = clock();

      s_tvalue[k] = end_time - start_time;
    } else {
      j = array[j];
      // warmup_j = array[warmup_j];
    }
  }

  array[array_length] = j;
  array[array_length + 1] = array[j];

  for (size_t k = 0; k < ITER_SIZE; k++) {
    index[k] = s_index[k];
    duration[k] = s_tvalue[k];
  }
}

// can store latencies 0-100, 100-200, .. 1500-1600
// const static unsigned int LATENCY_BIN_COUNT = 16;
const static unsigned int LATENCY_BIN_SIZE = 16;
// const static unsigned int LATENCY_BIN_BITS = 4; //
// const static unsigned int LATENCIES_PER_TVALUE = 8;
//
// constexpr unsigned floorlog2(unsigned x) {
//   return x == 1 ? 0 : 1 + floorlog2(x >> 1);
// }
//
// constexpr unsigned ceillog2(unsigned x) {
//   return x == 1 ? 0 : floorlog2(x - 1) + 1;
// }
//
// static_assert(ceillog2(LATENCY_BIN_COUNT) == LATENCY_BIN_BITS,
//               "correct latency bin bits");
// static_assert(sizeof(uint32_t) * 8 / LATENCY_BIN_BITS ==
// LATENCIES_PER_TVALUE,
//               "correct latencies per tvalue");
// static_assert(sizeof(uint32_t) * 8 == 32, "uint32 is 32 bits");
// static_assert(CHAR_BIT == 8, "have 8 bits per byte");

// const int ITER_SIZE_COMPRESSED = 12 * 1024;
const int ITER_SIZE_COMPRESSED = (48 * KB) / sizeof(uint8_t);

__global__ void global_latency_compressed(unsigned int *array, int array_length,
                                          unsigned int *duration,
                                          unsigned int *index,
                                          size_t warmup_iterations) {
  unsigned int start_time, end_time, dur;
  uint32_t j = 0;

  __shared__ uint8_t s_tvalue[ITER_SIZE_COMPRESSED];
  // __shared__ uint32_t s_index[ITER_SIZE];

  for (size_t k = 0; k < ITER_SIZE_COMPRESSED; k++) {
    // s_index[k] = 0;
    s_tvalue[k] = 0;
  }

  for (int k = (int)warmup_iterations * -ITER_SIZE_COMPRESSED;
       k < ITER_SIZE_COMPRESSED; k++) {
    if (k >= 0) {
      start_time = clock();
      j = array[j];
      // s_index[k] = j;
      s_tvalue[k] = j;
      end_time = clock();

      dur = (end_time - start_time) / LATENCY_BIN_SIZE;
      dur = dur < 256 ? dur : 255;
      s_tvalue[k] = (uint8_t)dur;

      // s_tvalue[ITER_SIZE - 1] = end_time - start_time;
      //
      // // 4 bit latency bin
      // unsigned int latency_bin = s_tvalue[ITER_SIZE - 1];
      // // unsigned int latency_bin = end_time - start_time;
      // latency_bin = (latency_bin / LATENCY_BIN_SIZE) % LATENCY_BIN_COUNT;
      // // assert(latency_bin >= 1);
      // // assert(((135 / LATENCY_BIN_SIZE) % LATENCY_BIN_COUNT) == 1);
      // const size_t tvalue_idx = k / LATENCIES_PER_TVALUE;
      // const size_t tvalue_offset = k % LATENCIES_PER_TVALUE;
      // const size_t latency_mask = (1 << LATENCY_BIN_BITS) - 1;
      // // printf("k=%u t_idx=%lu t_offset=%lu\n", k, tvalue_idx,
      // tvalue_offset);
      // // assert(latency_mask == 0xF);
      // // clear out the old bits
      // s_tvalue[tvalue_idx] &=
      //     ~(latency_mask << (tvalue_offset * LATENCY_BIN_BITS));
      // // set the new bits
      // // assert((latency_bin & ~latency_mask) == 0);
      // s_tvalue[tvalue_idx] |= (latency_bin & latency_mask)
      //                         << (tvalue_offset * LATENCY_BIN_BITS);
    } else {
      j = array[j];
    }
  }

  array[array_length] = j;
  array[array_length + 1] = array[j];

  for (size_t k = 0; k < ITER_SIZE_COMPRESSED; k++) {
    // index[k] = s_index[k];
    duration[k] = s_tvalue[k];
  }
}

__global__ void
global_latency_l1_readonly(const unsigned int *__restrict__ array,
                           int array_length, unsigned int *duration,
                           unsigned int *index, size_t warmup_iterations) {
  unsigned int start_time, end_time;
  uint32_t j = threadIdx.x;

  __shared__ uint32_t s_tvalue[ITER_SIZE];
  __shared__ uint32_t s_index[ITER_SIZE];

  for (size_t k = 0; k < ITER_SIZE; k++) {
    s_index[k] = 0;
    s_tvalue[k] = 0;
  }

  for (int it = (int)warmup_iterations * -ITER_SIZE; it < ITER_SIZE; it++) {
    if (it >= 0) {
      int k = it * blockDim.x + threadIdx.x;
      start_time = clock();
      j = __ldg(&array[j]);
      s_index[k] = j;
      end_time = clock();

      s_tvalue[k] = end_time - start_time;
    } else {
      j = __ldg(&array[j]);
    }
  }

  // array[array_length] = j;
  // array[array_length + 1] = array[j];

  for (size_t it = 0; it < ITER_SIZE; it++) {
    int k = it * blockDim.x + threadIdx.x;
    index[k] = s_index[k];
    duration[k] = s_tvalue[k];
  }
}

void parametric_measure_global(size_t N, size_t stride,
                               size_t warmup_iterations) {
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

  size_t iter_size = USE_COMPRESSION ? ITER_SIZE_COMPRESSED : ITER_SIZE;

  unsigned int *h_index =
      (unsigned int *)malloc(sizeof(unsigned int) * iter_size);
  unsigned int *h_timeinfo =
      (unsigned int *)malloc(sizeof(unsigned int) * iter_size);

  unsigned int *duration;
  CUDA_SAFECALL(
      cudaMalloc((void **)&duration, sizeof(unsigned int) * iter_size));

  unsigned int *d_index;
  CUDA_SAFECALL(
      cudaMalloc((void **)&d_index, sizeof(unsigned int) * iter_size));

  cudaDeviceSynchronize();
  // launch kernel
  dim3 block_dim = dim3(1);
  dim3 grid_dim = dim3(1, 1, 1);

  if (USE_COMPRESSION) {
    CUDA_SAFECALL((global_latency_compressed<<<grid_dim, block_dim>>>(
        d_a, N, duration, d_index, warmup_iterations)));
  } else {
    CUDA_SAFECALL((global_latency_l1_data<<<grid_dim, block_dim>>>(
        d_a, N, duration, d_index, warmup_iterations)));
  }

  cudaDeviceSynchronize();

  CUDA_SAFECALL(cudaGetLastError());

  // copy results from GPU to CPU
  cudaDeviceSynchronize();

  CUDA_SAFECALL(cudaMemcpy((void *)h_timeinfo, (void *)duration,
                           sizeof(unsigned int) * iter_size,
                           cudaMemcpyDeviceToHost));
  CUDA_SAFECALL(cudaMemcpy((void *)h_index, (void *)d_index,
                           sizeof(unsigned int) * iter_size,
                           cudaMemcpyDeviceToHost));

  cudaDeviceSynchronize();

  if (USE_COMPRESSION) {
    unsigned int j = 0;
    // fprintf(stdout, "%4d to %4d\n", (int)warmup_iterations * -(int)iter_size,
    //         (int)iter_size);
    for (int k = (int)warmup_iterations * -(int)iter_size; k < (int)iter_size;
         k++) {
      if (k >= 0) {
        j = h_a[j];
        unsigned int index = j;
        unsigned int binned_latency = h_timeinfo[k] * LATENCY_BIN_SIZE;
        fprintf(stdout, "%4d,%4d\n", index, binned_latency);
      } else {
        j = h_a[j];
      }
    }
  } else {
    for (size_t k = 0; k < iter_size; k++) {
      unsigned int index = h_index[k];
      unsigned int latency = h_timeinfo[k];
      fprintf(stdout, "%4d,%4d\n", index, latency);
    }
  }
  // unsigned int j = 0;
  // for (size_t k = 0; k < iter_size ; k++) {
  //   // print as CSV to stdout
  //   unsigned int index = h_index[k];
  //   unsigned int latency;
  //   if (USE_COMPRESSION) {
  //     latency = h_timeinfo[k];
  //     j = d_a[j];
  //     index = j;
  //
  //     // size_t tvalue_idx = i / LATENCIES_PER_TVALUE;
  //     // size_t tvalue_offset = i % LATENCIES_PER_TVALUE;
  //     // size_t latency_mask = (1 << LATENCY_BIN_BITS) - 1;
  //     // assert(latency_mask == 0xF);
  //     // latency = h_timeinfo[tvalue_idx] & (latency_mask << tvalue_offset);
  //     // latency = latency >> tvalue_offset;
  //     // latency = latency * LATENCY_BIN_SIZE;
  //   } else {
  //     latency = h_timeinfo[k];
  //   }
  //
  //   fprintf(stdout, "%4d,%4d\n", index, latency);
  // }

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
  size_t size_bytes, stride_bytes, warmup_iterations;

  // parse arguments
  if (argc > 2) {
    size_bytes = atoi(argv[1]);
    stride_bytes = atoi(argv[2]);
    warmup_iterations = atoi(argv[3]);
  } else {
    fprintf(stderr,
            "usage: p_chase_l1 <SIZE_BYTES> <STRIDE_BYTES> <WARMUP> \n");
    return EXIT_FAILURE;
  }

  size_t iter_size = USE_COMPRESSION ? ITER_SIZE_COMPRESSED : ITER_SIZE;

  // the number of resulting patterns P (full iterations through size) is
  // P = iter_size / stride
  float one_round = (float)size_bytes / (float)stride_bytes;
  float num_rounds = (float)iter_size / one_round;

  size_t size = size_bytes / sizeof(uint32_t);
  size_t stride = stride_bytes / sizeof(uint32_t);

  fprintf(stderr,
          "\tSIZE               = %10lu bytes (%10lu uint32, %10.4f KB)\n",
          size_bytes, size, (float)size_bytes / 1024.0);
  fprintf(stderr, "\tSTRIDE             = %10lu bytes (%10lu uint32)\n ",
          stride_bytes, stride);
  fprintf(stderr, "\tROUNDS             = %3.3f\n", num_rounds);
  fprintf(stderr, "\tONE ROUND          = %3.3f (have %5lu)\n", one_round,
          iter_size);
  fprintf(stderr, "\tITERATIONS         = %lu\n", iter_size);
  fprintf(stderr, "\tWARMUP ITERATIONS  = %lu\n", warmup_iterations);

  // assert(num_rounds > 1 &&
  //        "array size is too big (rounds should be at least two)");
  // assert(iter_size > size / stride);

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

  // printf("\n=====%10.4f KB array, warm TLB, read iter_size element====\n",
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
  // `cudaFuncCachePreferNone` uses the preference set for the device or
  // thread.

  cudaFuncCache want_cache_config = cudaFuncCachePreferShared;
  // cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  CUDA_SAFECALL(cudaDeviceSetCacheConfig(want_cache_config));
  CUDA_SAFECALL(
      cudaFuncSetCacheConfig(global_latency_compressed, want_cache_config));
  cudaFuncCache have_cache_config;
  CUDA_SAFECALL(cudaDeviceGetCacheConfig(&have_cache_config));
  assert(want_cache_config == have_cache_config);

  // CUDA_SAFECALL(cudaFuncSetAttribute(
  //     global_latency_compressed, cudaFuncAttributeMaxDynamicSharedMemorySize,
  //     12 * 1024));

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  fprintf(stderr, "\tSHMEM PER BLOCK   = %lu\n", prop.sharedMemPerBlock);
  fprintf(stderr, "\tSHMEM PER SM      = %lu\n",
          prop.sharedMemPerMultiprocessor);
  fprintf(stderr, "\tL2 size           = %u\n", prop.l2CacheSize);

  parametric_measure_global(size, stride, warmup_iterations);

  cudaDeviceReset();
  fflush(stdout);
  fflush(stderr);
  return EXIT_SUCCESS;
}
