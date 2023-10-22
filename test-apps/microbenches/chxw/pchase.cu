#include <algorithm>
#include <assert.h>
#include <cstdlib>
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>

#include "cuda_runtime.h"

#include "common.hpp"

// const bool USE_COMPRESSION = false;
const bool USE_HOST_MAPPED_MEMORY = true;

const size_t SHARED_MEMORY_MAX_ITER_SIZE = ((48 * KB) / 2) / sizeof(uint32_t);

__global__ __noinline__ void
global_latency_l1_data_shared_memory(unsigned int *array, int array_length,
                                     unsigned int *latency, unsigned int *index,
                                     int iter_size, size_t warmup_iterations) {
  const int max_iter_size = SHARED_MEMORY_MAX_ITER_SIZE;
  assert(iter_size <= max_iter_size);

  unsigned int start_time, end_time;
  uint32_t j = 0;

  __shared__ uint32_t s_latency[max_iter_size];
  __shared__ uint32_t s_index[max_iter_size];

  for (size_t k = 0; k < iter_size; k++) {
    s_index[k] = 0;
    s_latency[k] = 0;
  }

  for (int k = (int)warmup_iterations * -iter_size; k < iter_size; k++) {
    if (k >= 0) {
      start_time = clock();
      j = array[j];
      s_index[k] = j;
      end_time = clock();

      s_latency[k] = end_time - start_time;
    } else {
      j = array[j];
    }
  }

  // store to avoid caching in readonly?
  array[array_length] = j;
  array[array_length + 1] = array[j];

  for (size_t k = 0; k < iter_size; k++) {
    index[k] = s_index[k];
    latency[k] = s_latency[k];
  }
}

__global__ __noinline__ void
global_latency_l1_data_host_mapped(unsigned int *array, int array_length,
                                   unsigned int *latency, unsigned int *index,
                                   int iter_size, size_t warmup_iterations) {
  unsigned int start_time, end_time;
  uint32_t j = 0;

  for (int k = (int)warmup_iterations * -iter_size; k < iter_size; k++) {
    if (k >= 0) {
      start_time = clock();
      j = array[j];
      index[k] = j;
      end_time = clock();

      latency[k] = end_time - start_time;
    } else {
      j = array[j];
    }
  }

  // store to avoid caching in readonly?
  array[array_length] = j;
  array[array_length + 1] = array[j];
}

__global__ __noinline__ void
global_latency_l2_data_host_mapped(unsigned int *array, int array_length,
                                   unsigned int *latency, unsigned int *index,
                                   int iter_size, size_t warmup_iterations) {
  unsigned int start_time, end_time;
  uint32_t j = 0;

  for (int k = (int)warmup_iterations * -iter_size; k < iter_size; k++) {
    if (k >= 0) {
      start_time = clock();
      j = __ldcg(&array[j]);
      index[k] = j;
      end_time = clock();

      latency[k] = end_time - start_time;
    } else {
      j = __ldcg(&array[j]);
    }
  }

  // store to avoid caching in readonly?
  array[array_length] = j;
  array[array_length + 1] = array[j];
}

// const static unsigned int LATENCY_BIN_SIZE = 16;
// const size_t ITER_SIZE_COMPRESSED = (48 * KB) / sizeof(uint8_t);
//
// __global__ __noinline__ void
// global_latency_compressed(unsigned int *array, int array_length,
//                           unsigned int *duration, unsigned int *index,
//                           int iter_size, size_t warmup_iterations) {
//   const int max_iter_size = ITER_SIZE_COMPRESSED;
//   assert(iter_size <= max_iter_size);
//
//   unsigned int start_time, end_time, dur;
//   uint32_t j = 0;
//
//   __shared__ uint8_t s_tvalue[max_iter_size];
//
//   for (size_t k = 0; k < iter_size; k++) {
//     s_tvalue[k] = 0;
//   }
//
//   for (int k = (int)warmup_iterations * -iter_size; k < iter_size; k++) {
//     if (k >= 0) {
//       start_time = clock();
//       j = array[j];
//       s_tvalue[k] = j;
//       end_time = clock();
//
//       dur = (end_time - start_time) / LATENCY_BIN_SIZE;
//       dur = dur < 256 ? dur : 255;
//       s_tvalue[k] = (uint8_t)dur;
//     } else {
//       j = array[j];
//     }
//   }
//
//   array[array_length] = j;
//   array[array_length + 1] = array[j];
//
//   for (size_t k = 0; k < iter_size; k++) {
//     duration[k] = s_tvalue[k];
//   }
// }

const size_t SHARED_MEMORY_MAX_READONLY_ITER_SIZE =
    (48 * KB) / sizeof(uint32_t);

__global__ __noinline__ void global_latency_l1_readonly_shared_memory(
    const unsigned int *__restrict__ array, int array_length,
    unsigned int *latency, unsigned int *index, int iter_size,
    size_t warmup_iterations) {
  const int max_iter_size = SHARED_MEMORY_MAX_READONLY_ITER_SIZE;
  assert(iter_size <= max_iter_size);
  unsigned int start_time, end_time;
  size_t it;
  // uint32_t j = threadIdx.x;
  uint32_t j = 0;

  __shared__ uint32_t s_latency[max_iter_size];
  // __shared__ uint32_t s_index[max_iter_size];

  for (size_t k = 0; k < iter_size; k++) {
    // s_index[k] = 0;
    s_latency[k] = 0;
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
    s_latency[iter_size - 1] = j;
    end_time = clock();

    s_latency[k] = end_time - start_time;
  }

  // cannot no longer write to it!
  // array[array_length] = j;
  // array[array_length + 1] = array[j];

  for (it = 0; it < iter_size / 32; it++) {
    int k = it * blockDim.x + threadIdx.x;
    // index[k] = s_index[k];
    latency[k] = s_latency[k];
  }
}

// declare the texture
// texture<int, 1, cudaReadModeElementType> tex_ref;

__global__ __noinline__ void global_latency_l1_texture_shared_memory(
    unsigned int *array, cudaTextureObject_t tex, int array_length,
    unsigned int *latency, unsigned int *index, int iter_size,
    size_t warmup_iterations) {
  const int max_iter_size = SHARED_MEMORY_MAX_ITER_SIZE;
  assert(iter_size <= max_iter_size);

  unsigned int start_time, end_time;
  uint32_t j = 0;
  // uint32_t j = threadIdx.x;

  __shared__ uint32_t s_latency[max_iter_size];
  __shared__ uint32_t s_index[max_iter_size];

  for (size_t k = 0; k < iter_size; k++) {
    s_index[k] = 0;
    s_latency[k] = 0;
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

      s_latency[k] = end_time - start_time;
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
    latency[it] = s_latency[it];
  }

  // why is this so different?
  array[array_length] = it;
  array[array_length + 1] = s_latency[it - 1];
}

int parametric_measure_global(unsigned int *h_a, unsigned int *d_a, memory mem,
                              size_t N, size_t stride, size_t iter_size,
                              size_t warmup_iterations, size_t repetition,
                              unsigned int clock_overhead) {
  assert(iter_size < (size_t)-1);
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
  // CUDA_CHECK(cudaMalloc((void **)&d_a, (N + 2) * sizeof(unsigned int)));
  // // }

  // initialize array elements on CPU with pointers into d_a
  for (size_t i = 0; i < N; i++) {
    // original:
    h_a[i] = (i + stride) % N;
  }

  h_a[N] = 0;
  h_a[N + 1] = 0;

  // copy array elements from CPU to GPU
  CUDA_CHECK(
      cudaMemcpy(d_a, h_a, N * sizeof(uint32_t), cudaMemcpyHostToDevice));

  unsigned int *h_index, *h_latency, *d_index, *d_latency;
  if (USE_HOST_MAPPED_MEMORY) {
    CUDA_CHECK(cudaHostAlloc(&h_index, iter_size * sizeof(unsigned int),
                             cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&h_latency, iter_size * sizeof(unsigned int),
                             cudaHostAllocMapped));

    CUDA_CHECK(cudaHostGetDevicePointer(&d_index, h_index, 0));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_latency, h_latency, 0));
  } else {
    h_index = (unsigned int *)malloc(iter_size * sizeof(unsigned int));
    h_latency = (unsigned int *)malloc(iter_size * sizeof(unsigned int));

    CUDA_CHECK(
        cudaMalloc((void **)&d_latency, iter_size * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc((void **)&d_index, iter_size * sizeof(unsigned int)));
  }

  for (size_t k = 0; k < iter_size; k++) {
    h_index[k] = 0;
    h_latency[k] = 0;
  }

  // unsigned int *h_index =
  //     (unsigned int *)malloc(iter_size * sizeof(unsigned int));
  // unsigned int *h_timeinfo =
  //     (unsigned int *)malloc(iter_size * sizeof(unsigned int));
  //
  // unsigned int *duration;
  // CUDA_CHECK(cudaMalloc((void **)&duration, iter_size * sizeof(unsigned
  // int)));
  //
  // unsigned int *d_index;
  // CUDA_CHECK(cudaMalloc((void **)&d_index, iter_size * sizeof(unsigned
  // int)));

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

    CUDA_CHECK(
        (global_latency_l1_texture_shared_memory<<<grid_dim, block_dim>>>(
            d_a, texObj, N, d_latency, d_index, iter_size, warmup_iterations)));
    break;
  case L1ReadOnly:
    block_dim = dim3(32, 1, 1);
    CUDA_CHECK(
        (global_latency_l1_readonly_shared_memory<<<grid_dim, block_dim>>>(
            d_a, N, d_latency, d_index, iter_size, warmup_iterations)));
    break;
  case L2:
    assert(USE_HOST_MAPPED_MEMORY);
    CUDA_CHECK((global_latency_l2_data_host_mapped<<<grid_dim, block_dim>>>(
        d_a, N, d_latency, d_index, iter_size, warmup_iterations)));
    break;

  case L1Data:
    if (USE_HOST_MAPPED_MEMORY) {
      CUDA_CHECK((global_latency_l1_data_host_mapped<<<grid_dim, block_dim>>>(
          d_a, N, d_latency, d_index, iter_size, warmup_iterations)));
    } else {
      CUDA_CHECK((global_latency_l1_data_shared_memory<<<grid_dim, block_dim>>>(
          d_a, N, d_latency, d_index, iter_size, warmup_iterations)));
    }
    break;
  default:
    assert(false && "error dispatching to memory");
    break;
  };
  cudaDeviceSynchronize();

  CUDA_CHECK(cudaGetLastError());

  // copy results from GPU to CPU
  cudaDeviceSynchronize();

  if (!USE_HOST_MAPPED_MEMORY) {
    CUDA_CHECK(cudaMemcpy((void *)h_latency, (void *)d_latency,
                          sizeof(unsigned int) * iter_size,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy((void *)h_index, (void *)d_index,
                          sizeof(unsigned int) * iter_size,
                          cudaMemcpyDeviceToHost));
  }

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

        mean_latency += (float)h_latency[k * 32];
      }
      mean_latency /= 32.0;
      mean_latency -= (float)clock_overhead;
      unsigned long long virt_addr =
          (unsigned long long)d_a +
          (unsigned long long)index * (unsigned long long)sizeof(uint32_t);

      // r,n,k,index,virt_addr,latency
      fprintf(stdout, "%3lu,%8lu,%4lu,%4lu,%10llu,%4.10f\n", repetition,
              N * sizeof(uint32_t), k, index * sizeof(uint32_t), virt_addr,
              mean_latency);
    }
    break;
  default:
    // if (USE_COMPRESSION) {
    //   // unsigned int j = 0;
    //   for (int k = (int)warmup_iterations * -(int)iter_size; k <
    //   (int)iter_size;
    //        k++) {
    //     if (k >= 0) {
    //       unsigned int index = j;
    //       j = h_a[j];
    //       unsigned int binned_latency = h_timeinfo[k] * LATENCY_BIN_SIZE;
    //       fprintf(stdout, "%8lu,%4lu,%10llu,%4d\n", N * sizeof(uint32_t),
    //               index * sizeof(uint32_t),
    //               (unsigned long long)d_a +
    //                   (unsigned long long)index *
    //                       (unsigned long long)sizeof(uint32_t),
    //               (int)binned_latency - (int)clock_overhead);
    //     } else {
    //       j = h_a[j];
    //     }
    //   }
    // } else {
    for (size_t k = 0; k < iter_size; k++) {
      // unsigned int index = (N + h_index[k] - stride) % N;
      unsigned int index = indexof(h_a, N, h_index[k]);
      unsigned int latency = (int)h_latency[k] - (int)clock_overhead;
      unsigned long long virt_addr =
          (unsigned long long)d_a +
          (unsigned long long)index * (unsigned long long)sizeof(uint32_t);

      // r,n,k,index,virt_addr,latency
      fprintf(stdout, "%3lu,%8lu,%4lu,%4lu,%10llu,%4d\n", repetition,
              N * sizeof(uint32_t), k, index * sizeof(uint32_t), virt_addr,
              latency);
    }
    // }
    break;
  }

  // destroy texture object
  if (texObj != 0) {
    cudaDestroyTextureObject(texObj);
  }

  if (USE_HOST_MAPPED_MEMORY) {
    CUDA_CHECK(cudaFreeHost(h_index));
    CUDA_CHECK(cudaFreeHost(h_latency));
  } else {
    CUDA_CHECK(cudaFree(d_index));
    CUDA_CHECK(cudaFree(d_latency));
    free(h_index);
    free(h_latency);
  }

  return EXIT_SUCCESS;
}

int main(int argc, char *argv[]) {
  cudaSetDevice(0);
  memory mem;
  size_t stride_bytes, warmup_iterations;
  size_t start_size_bytes, end_size_bytes, step_size_bytes;
  size_t repetitions = 1;
  size_t max_rounds = (size_t)-1;
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

    if (argc >= 8) {
      start_size_bytes = atoi(argv[2]);
      end_size_bytes = atoi(argv[3]);
      step_size_bytes = atoi(argv[4]);
      stride_bytes = atoi(argv[5]);
      warmup_iterations = atoi(argv[6]);
      repetitions = atoi(argv[7]);
      if (argc >= 9) {
        sscanf(argv[8], "R%lu", &max_rounds);
        if (max_rounds == (size_t)-1) {
          iter_size = atoi(argv[8]);
        }
      }
    } else if (argc >= 6) {
      start_size_bytes = atoi(argv[2]);
      end_size_bytes = start_size_bytes;
      step_size_bytes = 1;
      stride_bytes = atoi(argv[3]);
      warmup_iterations = atoi(argv[4]);
      repetitions = atoi(argv[5]);
      if (argc >= 7) {
        sscanf(argv[6], "R%lu", &max_rounds);
        if (max_rounds == (size_t)-1) {
          iter_size = atoi(argv[6]);
        }
      }
    }
  } else {
    fprintf(stderr, "usage:  p_chase_l1 <MEM> <SIZE_BYTES> <STRIDE_BYTES> "
                    "<WARMUP> <REPETITIONS> <ITER_SIZE?>\n");
    fprintf(stderr,
            "        p_chase_l1 <MEM> <START_SIZE_BYTES> <END_SIZE_BYTES> "
            "<STEP_SIZE_BYTES> <STRIDE_BYTES> <WARMUP> <REPETITIONS> "
            "<ITER_SIZE?>\n");
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
  fprintf(stdout, "r,n,k,index,virt_addr,latency\n");

  size_t end_size = end_size_bytes / sizeof(uint32_t);
  fprintf(stderr, "alloc %lu elements\n", (end_size + 2) * sizeof(uint32_t));

  // allocate arrays on CPU
  unsigned int *h_a = (unsigned int *)malloc((end_size + 2) * sizeof(uint32_t));

  // allocate arrays on GPU
  unsigned int *d_a;
  CUDA_CHECK(cudaMalloc((void **)&d_a, (end_size + 2) * sizeof(uint32_t)));

  int exit_code = EXIT_SUCCESS;

  for (size_t size_bytes = start_size_bytes; size_bytes <= end_size_bytes;
       size_bytes += step_size_bytes) {

    float one_round_size = (float)size_bytes / (float)stride_bytes;

    size_t per_config_iter_size = iter_size;
    if (USE_HOST_MAPPED_MEMORY) {
      if (per_config_iter_size == (size_t)-1) {
        size_t per_config_max_rounds = max_rounds;
        if (per_config_max_rounds == (size_t)-1) {
          // default to 4 rounds through N
          per_config_max_rounds = 4;
        }
        per_config_iter_size =
            std::ceil(per_config_max_rounds * (size_t)one_round_size);
      }
    } else {
      switch (mem) {
      case L1ReadOnly:
        per_config_iter_size = std::min(per_config_iter_size,
                                        SHARED_MEMORY_MAX_READONLY_ITER_SIZE);
        break;
      case L1Data:
      case L1Texture:
      case L2:
        per_config_iter_size =
            std::min(per_config_iter_size, SHARED_MEMORY_MAX_ITER_SIZE);
        break;
      case NUM_MEMORIES:
        assert(false && "panic dispatching to memory");
      };

      per_config_iter_size =
          std::min(per_config_iter_size, max_rounds * (size_t)one_round_size);
    }

    float num_rounds = (float)per_config_iter_size / one_round_size;

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
    fprintf(stderr, "\tROUNDS             = %3.3f (%3.1f uint32 per round)\n",
            num_rounds, one_round_size);
    fprintf(stderr, "\tITERATIONS         = %lu\n", per_config_iter_size);
    fprintf(stderr, "\tMAX ROUNDS         = %d\n",
            (max_rounds == (size_t)-1) ? -1 : (int)max_rounds);
    fprintf(stderr, "\tREPETITIONS        = %lu\n", repetitions);
    fprintf(stderr, "\tWARMUP ITERATIONS  = %lu\n", warmup_iterations);
    fprintf(stderr, "\tBACKING MEM        = %s\n",
            USE_HOST_MAPPED_MEMORY ? "HOST MAPPED" : "SHARED MEM");

    // assert(num_rounds > 1 &&
    //        "array size is too big (rounds should be at least two)");
    // assert(per_config_iter_size > size / stride);

    // validate parameters
    if (size < stride) {
      fprintf(stderr, "ERROR: size (%lu) is smaller than stride (%lu)\n", size,
              stride);
      fflush(stderr);
      return EXIT_FAILURE;
    }
    // if (size % stride != 0) {
    //   fprintf(stderr,
    //           "ERROR: size (%lu) is not an exact multiple of stride
    //           (%lu)\n", size, stride);
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

    // The `cudaDeviceSetCacheConfig` function can be used to set preference
    // for shared memory or L1 cache globally for all CUDA kernels in your
    // code and even those used by Thrust. The option
    // cudaFuncCachePreferShared prefers shared memory, that is, it sets 48 KB
    // for shared memory and 16 KB for L1 cache.
    //
    // `cudaFuncCachePreferL1` prefers L1, that is, it sets 16 KB for
    // shared memory and 48 KB for L1 cache.
    //
    // `cudaFuncCachePreferNone` uses the preference set for the device or
    // thread.

    cudaFuncCache prefer_shared_mem_config = cudaFuncCachePreferShared;
    cudaFuncCache prefer_l1_config = cudaFuncCachePreferL1;
    CUDA_CHECK(cudaFuncSetCacheConfig(global_latency_l1_data_host_mapped,
                                      prefer_shared_mem_config));
    CUDA_CHECK(cudaFuncSetCacheConfig(global_latency_l1_data_shared_memory,
                                      prefer_shared_mem_config));

    CUDA_CHECK(cudaDeviceSetCacheConfig(prefer_shared_mem_config));
    cudaFuncCache have_cache_config;
    CUDA_CHECK(cudaDeviceGetCacheConfig(&have_cache_config));
    assert(have_cache_config == prefer_shared_mem_config);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    fprintf(stderr, "\tSHMEM PER BLOCK    = %10lu (%10.4f KB)\n",
            prop.sharedMemPerBlock, (float)prop.sharedMemPerBlock / 1024.0);
    fprintf(stderr, "\tSHMEM PER SM       = %10lu (%10.4f KB)\n",
            prop.sharedMemPerMultiprocessor,
            (float)prop.sharedMemPerMultiprocessor / 1024.0);
    fprintf(stderr, "\tL2 size            = %10u (%10.4f KB)\n",
            prop.l2CacheSize, (float)prop.l2CacheSize / 1024.0);

    for (size_t r = 0; r < repetitions; r++) {
      exit_code = parametric_measure_global(
          h_a, d_a, mem, size, stride, per_config_iter_size, warmup_iterations,
          r, clock_overhead);
      if (exit_code != EXIT_SUCCESS) {
        break;
      }
    }

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
