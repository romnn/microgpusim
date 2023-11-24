#include <algorithm>
#include <assert.h>
#include <cstdlib>
#include <ctype.h>
#include <random>
#include <stdint.h>
#include <stdio.h>

#include "common.hpp"
#include "cuda_runtime.h"

// When using a random pointer chase, we are no longer guaranteed to cause
// misses of a single set.
// const bool FIRST_STAGE_USE_RANDOM_CHASE = false;

// increase cache size
const bool INC_SIZE = false;

// Use host mapped memory instead of shared memory to store measurements.
// As host mapped memory is not cached on the GPU, this should not disturb
// measurements just like shared memory.
//
// Pro: works for larger cache sizes (e.g. multiple rounds of L2)
// Con: slower
const bool USE_HOST_MAPPED_MEMORY = true;

// Maximum iter size when shared memory (48KB) is used.
const size_t MAX_SHARED_MEM_ITER_SIZE = ((48 * KB) / 2) / sizeof(uint32_t);

__global__ __noinline__ void global_latency_l1_set_mapping_shared_memory(
    unsigned int *array, int array_length, unsigned int *latency,
    unsigned int *index, int iter_size, size_t warmup_iterations,
    unsigned int overflow_index) {
  const int max_iter_size = MAX_SHARED_MEM_ITER_SIZE;
  assert(iter_size <= max_iter_size);

  unsigned int start_time, end_time;
  volatile uint32_t j = 0;

  __shared__ volatile uint32_t s_latency[max_iter_size];
  __shared__ volatile uint32_t s_index[max_iter_size];

  for (size_t k = 0; k < iter_size; k++) {
    s_index[k] = 0;
    s_latency[k] = 0;
  }

  for (int k = (int)warmup_iterations * -iter_size; k < iter_size; k++) {
    if (k >= 0 && j == 0) {
      // overflow the cache now
      s_index[k] = array[array_length + overflow_index];
    }
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

__global__ __noinline__ void global_latency_l1_set_mapping_host_mapped(
    unsigned int *array, int array_length, unsigned int *latency,
    unsigned int *index, int iter_size, size_t warmup_iterations,
    unsigned int overflow_index) {
  unsigned int start_time, end_time;
  volatile uint32_t j = 0;

  for (int k = (int)warmup_iterations * -iter_size; k < iter_size; k++) {
    if (k >= 0 && j == 0) {
      // overflow the cache now
      index[k] = array[array_length + overflow_index];
    }
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

// version for cc 8.6 ampere
/*
__global__ __noinline__ void global_latency_l1_set_mapping_cc86_host_mapped(
    unsigned int *array, int array_length, unsigned int *latency,
    unsigned int *index, int iter_size, size_t warmup_iterations,
    size_t round_size, unsigned int overflow_index) {
  unsigned int start_time, end_time;
  volatile uint32_t j = 0;

  // int first_round_k = 1 * round_size;
  // int first_round_k = (warmup_iterations + 1) * iter_size;
  for (int k = (int)warmup_iterations * -iter_size; k < iter_size; k++) {
    // if (k == first_round_k) {
    // skip here
    // j += overflow_index;
    // }
    // if (k >= 0 && j == 0) {
    //   // skip here
    //   j = overflow_index;
    // }
    if (k >= 0) {
      start_time = clock();
      j = array[j];
      index[k+2] = j;
      index[k+1] = j;
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

  __threadfence_system();
}
*/

__global__ __noinline__ void global_latency_l1_set_mapping_cc86_host_mapped(
    unsigned int *array, int array_length, unsigned int *latency,
    unsigned int *index, int iter_size, size_t warmup_iterations,
    size_t round_size, unsigned int overflow_index) {
  unsigned int start_time, end_time;
  uint32_t j = 0;

  for (int k = (int)warmup_iterations * -iter_size; k < iter_size; k++) {
    if (k >= 0 && k == round_size) {
      // overflow the cache now
      // j += overflow_index;
      // assert(0 && "this should not happen");
      // index[k] = array[array_length + overflow_index];
    }
    // if (k >= 0 && j == 0) {
    // overflow the cache now
    // index[k] = array[array_length + overflow_index];
    // }
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

__global__ __noinline__ void global_latency_l1_set_mapping_pchase_host_mapped(
    unsigned int *array, int array_length, unsigned int *latency,
    unsigned int *index, int iter_size, size_t warmup_iterations,
    size_t round_size, unsigned int overflow_index) {
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

__global__ __noinline__ void global_latency_l1_set_mapping_random_host_mapped(
    unsigned int *array, int array_length, unsigned int *latency,
    unsigned int *index, int iter_size, int stride, size_t warmup_iterations,
    size_t round_size, unsigned int overflow_index) {
  unsigned int start_time, end_time;
  uint32_t j = 0;

  // first pass: linear loading
  // bool no_hits = true;
  for (int k = (int)warmup_iterations * -(int)round_size;
       k < 1 * (int)round_size; k++) {
    if (k >= 0) {
      start_time = clock();
      j = array[j];
      index[k] = j;
      end_time = clock();

      latency[k] = end_time - start_time;
      // if (latency[k] <= 100) {
      //	 no_hits = false;
      //}
    } else {
      j = array[j];
    }
  }
  // assert(no_hits && "no hits during loading");

  // accesses until cache miss starting at overflow index
  /*
          cache_miss = false;
  size_t attempt = 0;
  while (!cache_miss) {
          float index = curand_uniform(overflow_index * iter_size + attempt);
          index
          attempt++;
  }
  */
  // int latency2;
  bool have_miss = false;
  /*
    for (int i = 0; i < overflow_index / stride; i++) {
          j = array[j];
    }
    */
  j = overflow_index;
  for (int k = 1 * (int)round_size + (int)overflow_index / stride;
       k < 3 * (int)round_size; k++) {
    start_time = clock();
    j = array[j];
    index[k] = j;
    end_time = clock();

    // latency2 = end_time - start_time;
    latency[k] = end_time - start_time;
    if (latency[k] > 100) {
      have_miss = true;
      break;
    }
  }

  // assert(have_miss && "have miss");
  if (!have_miss) {
    // return;
  }

  // second pass: linear loading
  j = 0;
  for (int k = 3 * (int)round_size; k < (int)iter_size; k++) {
    // if (k >= 0) {
    start_time = clock();
    j = array[j];
    index[k] = j;
    end_time = clock();

    latency[k] = end_time - start_time;
    // }
  }

  // store to avoid caching in readonly?
  array[array_length] = j;
  array[array_length + 1] = array[j];
}

__global__ __noinline__ void global_latency_l2_set_mapping_host_mapped(
    unsigned int *array, int array_length, unsigned int *latency,
    unsigned int *index, int iter_size, size_t warmup_iterations,
    unsigned int overflow_index) {
  unsigned int start_time, end_time;
  volatile uint32_t j = 0;

  for (int k = (int)warmup_iterations * -iter_size; k < iter_size; k++) {
    if (k >= 0 && j == 0) {
      // overflow the cache now
      index[k] = array[array_length + overflow_index];
    }
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

int parametric_measure_global(unsigned int *h_a, unsigned int *d_a, memory mem,
                              size_t N, size_t stride, size_t iter_size,
                              size_t warmup_iterations, size_t repetition,
                              size_t compute_capability, bool random,
                              unsigned int clock_overhead,
                              unsigned int overflow_index) {
  // initialize array elements on CPU with pointers into d_a
  for (size_t i = 0; i < N; i++) {
    h_a[i] = (i + stride) % N;
  }

  /*
  if (random) {
    const unsigned long seed = 0;
    shuffle(h_a, h_a + N, std::default_random_engine(seed));
  }
  */

  overflow_index = overflow_index % N;
  assert(N % stride == 0);
  size_t round_size = N / stride;

  if (random) {
    assert(iter_size >= 4 * round_size);
    assert(overflow_index / stride <= round_size);
  }

  h_a[N] = 0;
  h_a[N + 1] = 0;

  CUDA_CHECK(cudaMemset(d_a, 0, N * sizeof(uint32_t)));

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

    // CUDA_CHECK((global_latency_l1_texture<<<grid_dim, block_dim>>>(
    //     d_a, texObj, N, duration, d_index, iter_size, warmup_iterations)));
    assert(0 && "todo");
    break;
  case L1ReadOnly:
    block_dim = dim3(32, 1, 1);
    assert(0 && "todo");
    // CUDA_CHECK((global_latency_l1_readonly<<<grid_dim, block_dim>>>(
    //     d_a, N, duration, d_index, iter_size, warmup_iterations)));
    break;
  case L2:
    CUDA_CHECK(
        (global_latency_l2_set_mapping_host_mapped<<<grid_dim, block_dim>>>(
            d_a, N, d_latency, d_index, iter_size, warmup_iterations,
            overflow_index)));
    break;
  case L1Data:
    if (USE_HOST_MAPPED_MEMORY && compute_capability == 0) {
      CUDA_CHECK(
          (global_latency_l1_set_mapping_host_mapped<<<grid_dim, block_dim>>>(
              d_a, N, d_latency, d_index, iter_size, warmup_iterations,
              overflow_index)));
    } else if (USE_HOST_MAPPED_MEMORY && compute_capability == 86) {
      if (random) {
        CUDA_CHECK(
            (global_latency_l1_set_mapping_random_host_mapped<<<grid_dim,
                                                                block_dim>>>(
                d_a, N, d_latency, d_index, iter_size, stride,
                warmup_iterations, round_size, overflow_index)));
      } else if (INC_SIZE) {
        CUDA_CHECK((global_latency_l1_set_mapping_pchase_host_mapped<<<
                        grid_dim, block_dim>>>(d_a, N, d_latency, d_index,
                                               iter_size, warmup_iterations,
                                               round_size, overflow_index)));
      } else {
        CUDA_CHECK((global_latency_l1_set_mapping_cc86_host_mapped<<<
                        grid_dim, block_dim>>>(d_a, N, d_latency, d_index,
                                               iter_size, warmup_iterations,
                                               round_size, overflow_index)));
      }
    } else {
      CUDA_CHECK(
          (global_latency_l1_set_mapping_shared_memory<<<grid_dim, block_dim>>>(
              d_a, N, d_latency, d_index, iter_size, warmup_iterations,
              overflow_index)));
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

  // unsigned int j = 0;

  switch (mem) {
  case L1ReadOnly:
    // for (size_t it = 0; it < warmup_iterations * iter_size; it++) {
    //   j = h_a[j];
    // }
    //
    // for (size_t k = 0; k < iter_size / 32; k++) {
    //   j = h_a[j];
    //   unsigned int index = j;
    //   float mean_latency = 0;
    //   for (size_t t = 0; t < 32; t++) {
    //     mean_latency += (float)h_timeinfo[k * 32];
    //   }
    //   mean_latency /= 32.0;
    //   fprintf(stdout, "%8lu,%4lu,%10llu,%4.10f\n", N * sizeof(uint32_t),
    //           index * sizeof(uint32_t),
    //           (unsigned long long)d_a +
    //               (unsigned long long)index *
    //                   (unsigned long long)sizeof(uint32_t),
    //           (float)mean_latency - (float)clock_overhead);
    // }
    assert(0 && "todo");
    break;
  default:
    for (size_t k = 0; k < iter_size; k++) {
      unsigned int index = indexof(h_a, N, h_index[k]);
      assert(index < N);
      unsigned int latency = (int)h_latency[k] - (int)clock_overhead;
      unsigned long long virt_addr =
          (unsigned long long)d_a +
          (unsigned long long)index * (unsigned long long)sizeof(uint32_t);

      // r,n,overflow_index,k,index,virt_addr,latency
      fprintf(stdout, "%3lu,%8lu,%4u,%4lu,%4lu,%10llu,%4d\n", repetition,
              N * sizeof(uint32_t), overflow_index, k, index * sizeof(uint32_t),
              virt_addr, latency);
    }
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
  size_t size_bytes, stride_bytes, warmup_iterations;
  size_t repetitions = 1;
  size_t iter_size = (size_t)-1;

  char *compute_capability_env = getenv("COMPUTE_CAPABILITY");
  char *random_env = getenv("RANDOM");

  size_t compute_capability = 0;
  if (compute_capability_env != NULL) {
    compute_capability = (size_t)atoi(compute_capability_env);
  }
  bool random = false;
  if (random_env != NULL) {
    random = true;
  }

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
    size_bytes = atoi(argv[2]);
    stride_bytes = atoi(argv[3]);
    warmup_iterations = atoi(argv[4]);
    if (argc >= 6) {
      repetitions = atoi(argv[5]);
    }
    if (argc >= 7) {
      iter_size = atoi(argv[6]);
    }
  } else {
    fprintf(stderr,
            "usage:  p_chase_set_mapping <MEM> <SIZE_BYTES> <STRIDE_BYTES> "
            "<WARMUP> <REPETITIONS?> <ITER_SIZE?>\n");
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

  unsigned int clock_overhead = measure_clock_overhead();

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  if (!prop.canMapHostMemory) {
    fprintf(stderr, "ERROR: device does not support host-mapped memory\n");
    fflush(stderr);
    return EXIT_FAILURE;
  }

  // use smaller 24 KB l1 data cache config on Pascal
  cudaFuncCache prefer_shared_mem_config = cudaFuncCachePreferShared;
  cudaFuncCache prefer_l1_config = cudaFuncCachePreferL1;
  CUDA_CHECK(cudaFuncSetCacheConfig(global_latency_l1_set_mapping_shared_memory,
                                    prefer_shared_mem_config));
  CUDA_CHECK(cudaFuncSetCacheConfig(global_latency_l1_set_mapping_host_mapped,
                                    prefer_shared_mem_config));

  // use maximum L1 data cache on volta+
  // int shared_mem_carveout_percent = cudaSharedmemCarveoutMaxL1;
  int shared_mem_carveout_percent = 75;
  CUDA_CHECK(
      cudaFuncSetAttribute(global_latency_l1_set_mapping_shared_memory,
                           cudaFuncAttributePreferredSharedMemoryCarveout,
                           shared_mem_carveout_percent));
  CUDA_CHECK(
      cudaFuncSetAttribute(global_latency_l1_set_mapping_host_mapped,
                           cudaFuncAttributePreferredSharedMemoryCarveout,
                           shared_mem_carveout_percent));

  // fetch full 128B cache lines
  // CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 128))
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 32))

  CUDA_CHECK(cudaDeviceSetCacheConfig(prefer_shared_mem_config));
  cudaFuncCache have_cache_config;
  CUDA_CHECK(cudaDeviceGetCacheConfig(&have_cache_config));
  assert(have_cache_config == prefer_shared_mem_config);

  size_t start_size_bytes = size_bytes;
  size_t end_size_bytes = size_bytes;
  if (INC_SIZE && compute_capability == 86) {
    end_size_bytes = 3 * size_bytes;
  }
  size_t start_size = start_size_bytes / sizeof(uint32_t);
  size_t end_size = end_size_bytes / sizeof(uint32_t);
  size_t size = end_size;

  if (USE_HOST_MAPPED_MEMORY) {
    if (iter_size == (size_t)-1) {
      // default to 4 rounds through N
      iter_size = end_size_bytes * 4;
    }
  } else {
    iter_size = std::min(iter_size, MAX_SHARED_MEM_ITER_SIZE);
  }

  if (size_bytes < 1) {
    fprintf(stderr, "ERROR: size is too small (%lu)\n", size_bytes);
    fflush(stderr);
    return EXIT_FAILURE;
  }

  // allocate arrays on CPU
  // we fill the cache with 0..size and overflow by a single index in
  // size..(2*size)
  unsigned int *h_a =
      (unsigned int *)malloc(((size * 2) + 2) * sizeof(uint32_t));

  // allocate arrays on GPU
  unsigned int *d_a;
  CUDA_CHECK(cudaMalloc((void **)&d_a, ((size * 2) + 2) * sizeof(uint32_t)));

  int exit_code = EXIT_SUCCESS;

  float one_round_size = (float)size_bytes / (float)stride_bytes;
  float num_rounds = (float)iter_size / one_round_size;
  size_t stride = stride_bytes / sizeof(uint32_t);

  fprintf(stderr, "======================\n");
  fprintf(stderr, "\tMEMORY             = %s\n", memory_str[mem]);
  fprintf(stderr,
          "\tSIZE               = %10lu bytes (%10lu uint32, %10.4f KB)\n",
          size_bytes, size, (float)size_bytes / 1024.0);
  fprintf(stderr, "\tSTRIDE             = %10lu bytes (%10lu uint32)\n",
          stride_bytes, stride);
  fprintf(stderr, "\tROUNDS             = %3.3f (%3.1f uint32 per round)\n",
          num_rounds, one_round_size);
  fprintf(stderr, "\tITERATIONS         = %lu\n", iter_size);
  fprintf(stderr, "\tREPETITIONS        = %lu\n", repetitions);
  fprintf(stderr, "\tCOMPUTE CAP        = %lu\n", compute_capability);
  fprintf(stderr, "\tWARMUP ITERATIONS  = %lu\n", warmup_iterations);
  fprintf(stderr, "\tBACKING MEM        = %s\n",
          USE_HOST_MAPPED_MEMORY ? "HOST MAPPED" : "SHARED MEM");

  // print CSV header
  fprintf(stdout, "r,n,overflow_index,k,index,virt_addr,latency\n");

  if (INC_SIZE && compute_capability == 86) {
    size_t overflow_index = 0;
    for (unsigned int size = start_size; size < end_size; size += stride) {
      for (size_t r = 0; r < repetitions; r++) {
        exit_code = parametric_measure_global(
            h_a, d_a, mem, size, stride, iter_size, warmup_iterations, r,
            compute_capability, random, clock_overhead, overflow_index);
        if (exit_code != EXIT_SUCCESS) {
          break;
        }
      }
      overflow_index += stride;
    }

  } else {
    for (unsigned int overflow_index = 0; overflow_index < size;
         overflow_index += stride) {
      for (size_t r = 0; r < repetitions; r++) {
        exit_code = parametric_measure_global(
            h_a, d_a, mem, size, stride, iter_size, warmup_iterations, r,
            compute_capability, random, clock_overhead, overflow_index);
        if (exit_code != EXIT_SUCCESS) {
          break;
        }
      }
    }
  }

  cudaFree(d_a);
  free(h_a);

  cudaDeviceReset();
  fflush(stdout);
  fflush(stderr);
  return exit_code;
}
