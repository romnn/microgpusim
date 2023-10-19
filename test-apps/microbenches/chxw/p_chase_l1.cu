#include <algorithm>
#include <assert.h>
#include <cstdlib>
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>

#include "cuda_runtime.h"

#include "common.hpp"

// #define CUDA_SAFECALL(call)                                                    \
//   {                                                                            \
//     call;                                                                      \
//     cudaError err = cudaGetLastError();                                        \
//     if (cudaSuccess != err) {                                                  \
//       fprintf(stderr,                                                          \
//               "Cuda error in function '%s' file '%s' in line %i : %s.\n",      \
//               #call, __FILE__, __LINE__, cudaGetErrorString(err));             \
//       fflush(stderr);                                                          \
//       exit(EXIT_FAILURE);                                                      \
//     }                                                                          \
//   }

const bool USE_COMPRESSION = false;

const size_t ITER_SIZE = ((48 * KB) / 2) / sizeof(uint32_t);

// __global__ __noinline__ void
// global_measure_clock_overhead(unsigned int *clock_cycles) {
//   const size_t iterations = 200;
//   __shared__ unsigned int s_overhead[2 * iterations];
//
//   for (size_t i = 0; i < 2 * iterations; i++) {
//     volatile unsigned int start_time = clock();
//     unsigned int end_time = clock();
//     s_overhead[i] = (end_time - start_time);
//   }
//
//   unsigned int sum = 0;
//   for (size_t i = 0; i < iterations; i++) {
//     sum += s_overhead[iterations + i];
//   }
//   *clock_cycles = float(sum) / float(iterations);
// }
//
// unsigned int measure_clock_overhead() {
//   uint32_t *h_clock_overhead = (uint32_t *)malloc(sizeof(uint32_t) * 1);
//
//   uint32_t *d_clock_overhead;
//   CUDA_SAFECALL(cudaMalloc((void **)&d_clock_overhead, sizeof(uint32_t) *
//   1));
//
//   // launch kernel
//   dim3 block_dim = dim3(1);
//   dim3 grid_dim = dim3(1, 1, 1);
//
//   CUDA_SAFECALL((global_measure_clock_overhead<<<grid_dim, block_dim>>>(
//       d_clock_overhead)));
//
//   CUDA_SAFECALL(cudaMemcpy((void *)h_clock_overhead, (void
//   *)d_clock_overhead,
//                            sizeof(uint32_t) * 1, cudaMemcpyDeviceToHost));
//
//   fprintf(stderr, "clock overhead is %u cycles\n", *h_clock_overhead);
//
//   return *h_clock_overhead;
// }

__global__ __noinline__ void
global_latency_l1_data(unsigned int *array, int array_length,
                       unsigned int *duration, unsigned int *index,
                       int iter_size, size_t warmup_iterations) {
  const int max_iter_size = ITER_SIZE;
  assert(iter_size <= max_iter_size);

  unsigned int start_time, end_time;
  uint32_t j = 0;

  __shared__ uint32_t s_tvalue[max_iter_size];
  __shared__ uint32_t s_index[max_iter_size];

  for (size_t k = 0; k < iter_size; k++) {
    s_index[k] = 0;
    s_tvalue[k] = 0;
  }

  for (int k = (int)warmup_iterations * -iter_size; k < iter_size; k++) {
    if (k >= 0) {
      start_time = clock();
      j = array[j];
      s_index[k] = j;
      end_time = clock();

      s_tvalue[k] = end_time - start_time;
    } else {
      j = array[j];
    }
  }

  // store to avoid caching in readonly?
  array[array_length] = j;
  array[array_length + 1] = array[j];

  for (size_t k = 0; k < iter_size; k++) {
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
const size_t ITER_SIZE_COMPRESSED = (48 * KB) / sizeof(uint8_t);

__global__ __noinline__ void
global_latency_compressed(unsigned int *array, int array_length,
                          unsigned int *duration, unsigned int *index,
                          int iter_size, size_t warmup_iterations) {
  const int max_iter_size = ITER_SIZE_COMPRESSED;
  assert(iter_size <= max_iter_size);

  unsigned int start_time, end_time, dur;
  uint32_t j = 0;

  __shared__ uint8_t s_tvalue[max_iter_size];
  // __shared__ uint32_t s_index[max_iter_size];

  for (size_t k = 0; k < iter_size; k++) {
    // s_index[k] = 0;
    s_tvalue[k] = 0;
  }

  for (int k = (int)warmup_iterations * -iter_size; k < iter_size; k++) {
    if (k >= 0) {
      start_time = clock();
      j = array[j];
      // s_index[k] = j;
      s_tvalue[k] = j;
      end_time = clock();

      dur = (end_time - start_time) / LATENCY_BIN_SIZE;
      dur = dur < 256 ? dur : 255;
      s_tvalue[k] = (uint8_t)dur;

      // s_tvalue[iter_size - 1] = end_time - start_time;
      //
      // // 4 bit latency bin
      // unsigned int latency_bin = s_tvalue[iter_size - 1];
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

  for (size_t k = 0; k < iter_size; k++) {
    // index[k] = s_index[k];
    duration[k] = s_tvalue[k];
  }
}

const size_t READONLY_ITER_SIZE = (48 * KB) / sizeof(uint32_t);

__global__ __noinline__ void
global_latency_l1_readonly(const unsigned int *__restrict__ array,
                           int array_length, unsigned int *duration,
                           unsigned int *index, int iter_size,
                           size_t warmup_iterations) {
  const int max_iter_size = READONLY_ITER_SIZE;
  assert(iter_size <= max_iter_size);
  unsigned int start_time, end_time;
  size_t it;
  // uint32_t j = threadIdx.x;
  uint32_t j = 0;

  __shared__ uint32_t s_tvalue[max_iter_size];
  // __shared__ uint32_t s_index[max_iter_size];

  for (size_t k = 0; k < iter_size; k++) {
    // s_index[k] = 0;
    s_tvalue[k] = 0;
  }

  // no-timing iterations, for large arrays
  for (it = 0; it < warmup_iterations * iter_size; it++) {
    j = __ldg(&array[j]);
  }

  // for (int it = (int)warmup_iterations * -iter_size; it < iter_size; it++) {
  for (it = 0; it < iter_size / 32; it++) {
    int k = it * blockDim.x + threadIdx.x;
    start_time = clock();
    j = __ldg(&array[j]);
    // s_index[k] = j;
    // s_tvalue[k] = j;
    s_tvalue[iter_size - 1] = j;
    end_time = clock();

    s_tvalue[k] = end_time - start_time;
  }

  // cannot no longer write to it!
  // array[array_length] = j;
  // array[array_length + 1] = array[j];

  for (it = 0; it < iter_size / 32; it++) {
    int k = it * blockDim.x + threadIdx.x;
    // index[k] = s_index[k];
    duration[k] = s_tvalue[k];
  }
}

// declare the texture
// texture<int, 1, cudaReadModeElementType> tex_ref;

__global__ __noinline__ void
global_latency_l1_texture(unsigned int *array, cudaTextureObject_t tex,
                          int array_length, unsigned int *duration,
                          unsigned int *index, int iter_size,
                          size_t warmup_iterations) {
  const int max_iter_size = ITER_SIZE;
  assert(iter_size <= max_iter_size);

  unsigned int start_time, end_time;
  uint32_t j = 0;
  // uint32_t j = threadIdx.x;

  __shared__ uint32_t s_tvalue[max_iter_size];
  __shared__ uint32_t s_index[max_iter_size];

  for (size_t k = 0; k < iter_size; k++) {
    s_index[k] = 0;
    s_tvalue[k] = 0;
  }

  for (int it = (int)warmup_iterations * -iter_size; it < iter_size; it++) {
    // int k = it * blockDim.x + threadIdx.x;
    int k = it;
    if (it >= 0) {
      start_time = clock();
      // j = tex1Dfetch(tex_ref, j);
      j = tex1Dfetch<unsigned int>(tex, j);
      // j = __ldg(&array[j]);
      s_index[k] = j;
      end_time = clock();

      s_tvalue[k] = end_time - start_time;
    } else {
      j = tex1Dfetch<unsigned int>(tex, j);
      // j = tex1Dfetch(tex_ref, j);
      // j = __ldg(&array[j]);
    }
  }

  // array[array_length] = j;
  // array[array_length + 1] = array[j];

  size_t it = 0;
  for (; it < iter_size; it++) {
    // int k = it * blockDim.x + threadIdx.x;
    // int k = it;
    index[it] = s_index[it];
    duration[it] = s_tvalue[it];
  }

  // why is this so different?
  array[array_length] = it;
  array[array_length + 1] = s_tvalue[it - 1];
}

int parametric_measure_global(unsigned int *h_a, unsigned int *d_a, memory mem,
                              size_t N, size_t stride, size_t iter_size,
                              size_t warmup_iterations,
                              unsigned int clock_overhead) {
  // cudaDeviceReset();

  // if (true) {
  //   size_t size = (N + 2) * sizeof(unsigned int);
  //
  //   CUmemAllocationProp prop = {};
  //   prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  //   prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  //   // prop.location.id = currentDe;
  //   size_t granularity = 0;
  //   cuMemGetAllocationGranularity(&granularity, &prop,
  //                                 CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  //
  //   size_t padded_size = ROUND_UP_TO_MULTIPLE(size, granularity);
  //   CUmemGenericAllocationHandle allocHandle;
  //   cuMemCreate(&allocHandle, padded_size, &prop, 0);
  //
  //   /* Reserve a virtual address range */
  //   CUdeviceptr ptr;
  //   cuMemAddressReserve(&ptr, padded_size, 0, 0, 0);
  //   /* Map the virtual address range to the physical allocation */
  //   cuMemMap(ptr, padded_size, 0, allocHandle, 0);
  // } else {

  // // allocate arrays on CPU
  // unsigned int *h_a;
  // h_a = (unsigned int *)malloc((N + 2) * sizeof(unsigned int));
  //
  // // allocate arrays on GPU
  // unsigned int *d_a;
  // CUDA_SAFECALL(cudaMalloc((void **)&d_a, (N + 2) * sizeof(unsigned int)));
  // // }

  // initialize array elements on CPU with pointers into d_a
  for (size_t i = 0; i < N; i++) {
    // original:
    h_a[i] = (i + stride) % N;
  }

  h_a[N] = 0;
  h_a[N + 1] = 0;

  // copy array elements from CPU to GPU
  fprintf(stderr, "copy %lu elements\n", N * sizeof(uint32_t));
  CUDA_SAFECALL(
      cudaMemcpy(d_a, h_a, N * sizeof(uint32_t), cudaMemcpyHostToDevice));

  unsigned int *h_index =
      (unsigned int *)malloc(iter_size * sizeof(unsigned int));
  unsigned int *h_timeinfo =
      (unsigned int *)malloc(iter_size * sizeof(unsigned int));

  unsigned int *duration;
  CUDA_SAFECALL(
      cudaMalloc((void **)&duration, iter_size * sizeof(unsigned int)));

  unsigned int *d_index;
  CUDA_SAFECALL(
      cudaMalloc((void **)&d_index, iter_size * sizeof(unsigned int)));

  cudaTextureObject_t texObj = 0;

  cudaDeviceSynchronize();
  // launch kernel
  dim3 block_dim = dim3(1);
  dim3 grid_dim = dim3(1, 1, 1);

  switch (mem) {
  case L1Texture:
    // bind texture
    // cudaBindTexture(0, tex_ref, d_a, N * sizeof(int));

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    // resDesc.resType = cudaResourceTypeArray;
    // resDesc.res.array.array = d_a;

    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_a;
    resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
    resDesc.res.linear.desc.x = 32; // bits per channel
    resDesc.res.linear.sizeInBytes = N * sizeof(unsigned int);

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    // texDesc.addressMode[0] = cudaAddressModeWrap;
    // texDesc.addressMode[1] = cudaAddressModeWrap;
    // texDesc.filterMode = cudaFilterModeLinear;
    // texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    // Create texture object
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    cudaDeviceSynchronize();

    CUDA_SAFECALL((global_latency_l1_texture<<<grid_dim, block_dim>>>(
        d_a, texObj, N, duration, d_index, iter_size, warmup_iterations)));
    break;
  case L1ReadOnly:
    block_dim = dim3(32, 1, 1);
    CUDA_SAFECALL((global_latency_l1_readonly<<<grid_dim, block_dim>>>(
        d_a, N, duration, d_index, iter_size, warmup_iterations)));
    break;
  case L2:
  case L1Data:
    if (USE_COMPRESSION) {
      CUDA_SAFECALL((global_latency_compressed<<<grid_dim, block_dim>>>(
          d_a, N, duration, d_index, iter_size, warmup_iterations)));
    } else {
      CUDA_SAFECALL((global_latency_l1_data<<<grid_dim, block_dim>>>(
          d_a, N, duration, d_index, iter_size, warmup_iterations)));
    }
    break;
  default:
    assert(false && "error dispatching to memory");
    break;
  };
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

  unsigned int j = 0;

  switch (mem) {
  case L1ReadOnly:
    for (size_t it = 0; it < warmup_iterations * iter_size; it++) {
      j = h_a[j];
    }

    // for (it = 0; it < ITER_SIZE / 32; it++) {
    //   int k = it * blockDim.x + threadIdx.x;
    //   // index[k] = s_index[k];
    //   duration[k] = s_tvalue[k];
    // }

    for (size_t k = 0; k < iter_size / 32; k++) {
      unsigned int index = j;
      j = h_a[j];
      // unsigned int index = h_index[k * 32];
      float mean_latency = 0;
      for (size_t t = 0; t < 32; t++) {
        // if (h_index[k * 32 + t] != index) {
        //   fprintf(
        //       stderr,
        //       "threads %lu and %lu accessed different indices (%4d !=
        //       %4d)\n", k * 32, k * 32 + t, index, h_index[k * 32 + t]);
        //   return EXIT_FAILURE;
        // }

        mean_latency += (float)h_timeinfo[k * 32];
      }
      mean_latency /= 32.0;
      fprintf(stdout, "%8lu,%4lu,%10llu,%4.10f\n", N * sizeof(uint32_t),
              index * sizeof(uint32_t),
              (unsigned long long)d_a +
                  (unsigned long long)index *
                      (unsigned long long)sizeof(uint32_t),
              (float)mean_latency - (float)clock_overhead);
    }
    break;
  default:
    if (USE_COMPRESSION) {
      // unsigned int j = 0;
      for (int k = (int)warmup_iterations * -(int)iter_size; k < (int)iter_size;
           k++) {
        if (k >= 0) {
          unsigned int index = j;
          j = h_a[j];
          unsigned int binned_latency = h_timeinfo[k] * LATENCY_BIN_SIZE;
          fprintf(stdout, "%8lu,%4lu,%10llu,%4d\n", N * sizeof(uint32_t),
                  index * sizeof(uint32_t),
                  (unsigned long long)d_a +
                      (unsigned long long)index *
                          (unsigned long long)sizeof(uint32_t),
                  (int)binned_latency - (int)clock_overhead);
        } else {
          j = h_a[j];
        }
      }
    } else {
      for (size_t k = 0; k < iter_size; k++) {
        unsigned int index = (N + h_index[k] - stride) % N;
        unsigned int latency = h_timeinfo[k];
        fprintf(stdout, "%8lu,%4lu,%10llu,%4d\n", N * sizeof(uint32_t),
                index * sizeof(uint32_t),
                (unsigned long long)d_a +
                    (unsigned long long)index *
                        (unsigned long long)sizeof(uint32_t),
                (int)latency - (int)clock_overhead);
      }
    }
    break;
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

  // destroy texture object
  if (texObj != 0) {
    cudaDestroyTextureObject(texObj);
  }

  // free memory on GPU
  // cudaFree(d_a);
  cudaFree(d_index);
  cudaFree(duration);

  // free memory on CPU
  // free(h_a);
  free(h_index);
  free(h_timeinfo);

  // cudaDeviceReset();

  return EXIT_SUCCESS;
}

int main(int argc, char *argv[]) {
  cudaSetDevice(0);
  memory mem;
  size_t stride_bytes, warmup_iterations;
  size_t start_size_bytes, end_size_bytes, step_size_bytes;
  size_t iter_size = (size_t)-1;

  // parse arguments
  if (argc >= 5) {
    char *mem_name = argv[1];

    // make mem name lowercase
    for (int i = 0; mem_name[i]; i++) {
      mem_name[i] = tolower(mem_name[i]);
    }
    int k = 0;
    int *mem_found = NULL;
    for (; k < NUM_MEMORIES; k++) {
      if (!strcmp(mem_name, memory_str[k])) {
        mem_found = &k;
        break;
      }
    }

    if (mem_found == NULL) {
      fprintf(stderr, "unknown memory name %s\n", mem_name);
      return EXIT_FAILURE;
    }

    mem = (memory)(*mem_found);

    if (argc >= 7) {
      start_size_bytes = atoi(argv[2]);
      end_size_bytes = atoi(argv[3]);
      step_size_bytes = atoi(argv[4]);
      stride_bytes = atoi(argv[5]);
      warmup_iterations = atoi(argv[6]);
      if (argc >= 8) {
        iter_size = atoi(argv[7]);
      }
    } else if (argc >= 5) {
      start_size_bytes = atoi(argv[2]);
      end_size_bytes = start_size_bytes;
      step_size_bytes = 1;
      stride_bytes = atoi(argv[3]);
      warmup_iterations = atoi(argv[4]);
      if (argc >= 6) {
        iter_size = atoi(argv[5]);
      }
    }
  } else {
    fprintf(stderr, "usage:  p_chase_l1 <MEM> <SIZE_BYTES> <STRIDE_BYTES> "
                    "<WARMUP> <ITER_SIZE?>\n");
    fprintf(stderr,
            "        p_chase_l1 <MEM> <START_SIZE_BYTES> <END_SIZE_BYTES> "
            "<STEP_SIZE_BYTES> <STRIDE_BYTES> <WARMUP> <ITER_SIZE?>\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "    <MEM>:\t");
    for (int k = 0; k < NUM_MEMORIES; k++) {
      fprintf(stderr, "%s", memory_str[k]);
      if (k + 1 < NUM_MEMORIES) {
        fprintf(stderr, " | ");
      }
    }
    fprintf(stderr, "\n\n\n");
    return EXIT_FAILURE;
  }

  if (step_size_bytes < 1) {
    fprintf(stderr, "ERROR: step size < 1 (%lu) will lead to infinite loop\n",
            step_size_bytes);
    fflush(stderr);
    return EXIT_FAILURE;
  }

  if (end_size_bytes < start_size_bytes) {
    fprintf(stderr, "ERROR: end size (%lu) is smaller than start size (%lu)\n",
            end_size_bytes, start_size_bytes);
    fflush(stderr);
    return EXIT_FAILURE;
  }

  unsigned int clock_overhead = measure_clock_overhead();
  // return EXIT_SUCCESS;

  // print CSV header
  fprintf(stdout, "n,index,virt_addr,latency\n");

  size_t max_iter_size;
  switch (mem) {
  case L1ReadOnly:
    max_iter_size = READONLY_ITER_SIZE;
    break;
  case L1Data:
  case L1Texture:
  case L2:
    max_iter_size = USE_COMPRESSION ? ITER_SIZE_COMPRESSED : ITER_SIZE;
    break;
  case NUM_MEMORIES:
    assert(false && "panic dispatching to memory");
  };

  iter_size = std::min(iter_size, max_iter_size);

  size_t end_size = end_size_bytes / sizeof(uint32_t);
  fprintf(stderr, "alloc %lu elements\n", (end_size + 2) * sizeof(uint32_t));

  // allocate arrays on CPU
  unsigned int *h_a = (unsigned int *)malloc((end_size + 2) * sizeof(uint32_t));

  // allocate arrays on GPU
  unsigned int *d_a;
  CUDA_SAFECALL(cudaMalloc((void **)&d_a, (end_size + 2) * sizeof(uint32_t)));

  int exit_code = EXIT_SUCCESS;

  for (size_t size_bytes = start_size_bytes; size_bytes <= end_size_bytes;
       size_bytes += step_size_bytes) {
    // the number of resulting patterns P (full iterations through size) is
    // P = iter_size / stride
    float one_round = (float)size_bytes / (float)stride_bytes;
    float num_rounds = (float)iter_size / one_round;

    size_t size = size_bytes / sizeof(uint32_t);
    size_t stride = stride_bytes / sizeof(uint32_t);

    if (size == 0) {
      continue;
    }

    fprintf(stderr, "======================\n");
    fprintf(stderr, "\tMEMORY             = %s\n", memory_str[mem]);
    fprintf(stderr,
            "\tSIZE               = %10lu bytes (%10lu uint32, %10.4f KB)\n",
            size_bytes, size, (float)size_bytes / 1024.0);
    fprintf(stderr, "\tSTRIDE             = %10lu bytes (%10lu uint32)\n",
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
    // if (stride < 1) {
    //   fprintf(stderr, "ERROR: stride is < 1 (%lu)\n", stride);
    //   fflush(stderr);
    //   return EXIT_FAILURE;
    // }

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
    //     global_latency_compressed,
    //     cudaFuncAttributeMaxDynamicSharedMemorySize, 12 * 1024));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    fprintf(stderr, "\tSHMEM PER BLOCK   = %lu\n", prop.sharedMemPerBlock);
    fprintf(stderr, "\tSHMEM PER SM      = %lu\n",
            prop.sharedMemPerMultiprocessor);
    fprintf(stderr, "\tL2 size           = %u\n", prop.l2CacheSize);

    exit_code =
        parametric_measure_global(h_a, d_a, mem, size, stride, iter_size,
                                  warmup_iterations, clock_overhead);
    if (exit_code != EXIT_SUCCESS) {
      break;
    }
  }

  cudaFree(d_a);
  free(h_a);

  cudaDeviceReset();
  fflush(stdout);
  fflush(stderr);
  return exit_code;
}
