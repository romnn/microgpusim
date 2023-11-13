#include <algorithm>
#include <assert.h>
#include <cstdlib>
#include <ctype.h>
#include <random>
#include <stdint.h>
#include <stdio.h>
#include <unordered_set>

#include "common.hpp"
#include "cuda_runtime.h"

__global__ __noinline__ void global_latency_l1_random_set_mapping_host_mapped(
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

__global__ __noinline__ void global_latency_l2_random_set_mapping_host_mapped(
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

#include <random>

/* simple class for a pseudo-random generator producing
   uniformely distributed integers */
class UniformIntDistribution {
public:
  UniformIntDistribution() : engine(std::random_device()()) {}
  /* return number in the range of [0..upper_limit) */
  unsigned int draw(unsigned int upper_limit) {
    return std::uniform_int_distribution<unsigned int>(0,
                                                       upper_limit - 1)(engine);
  }

private:
  std::mt19937 engine;
};

// template <typename T> T *compute_random_pointer_chain(T *memory, size_t size)
// {
template <typename T>
void compute_random_pointer_chain(T *memory, size_t size, size_t stride) {
  // size_t len = size;
  size_t len = size / stride;
  // T *memory = new T[len];

  // shuffle indices
  size_t *indices = new std::size_t[len];
  for (std::size_t i = 0; i < len; ++i) {
    indices[i] = i;
  }

  if (false) {
    UniformIntDistribution uniform;
    for (std::size_t i = 0; i < len - 1; ++i) {
      std::size_t j = i + uniform.draw(len - i);
      if (i != j) {
        std::swap(indices[i], indices[j]);
      }
    }
  } else {
    const unsigned long seed = 0;
    shuffle(indices, indices + len, std::default_random_engine(seed));
  }

  // fill memory with pointer references
  for (std::size_t i = 1; i < len; ++i) {
    memory[indices[i - 1] * stride] = (T)indices[i] * stride;
    // memory[(indices[i - 1] + stride) % size] = (T)indices[i];
    // memory[indices[i - 1]] = ((T)indices[i] + stride) % size;
    // (T)((size_t)&memory[indices[i]] - (size_t)&memory);
  }
  memory[indices[len - 1] * stride] = (T)indices[0] * stride;
  // memory[indices[len - 1]] = (T)indices[0];
  // (T)((size_t)&memory[indices[0]] - (size_t)&memory);
  delete[] indices;
}

int parametric_measure_global(unsigned int *h_a, unsigned int *d_a, memory mem,
                              size_t N, size_t stride, size_t iter_size,
                              size_t warmup_iterations, size_t repetition,
                              size_t compute_capability,
                              unsigned int clock_overhead,
                              unsigned int overflow_index) {

  // unsigned int *h_a_offsets = (unsigned int *)malloc(N * sizeof(uint32_t));
  // for (size_t i = 0; i < N; i++) {
  //   h_a_offsets[i] = i;
  // }
  //
  // const unsigned long seed = repetition;
  // shuffle(h_a_offsets, h_a_offsets + N, std::default_random_engine(seed));
  //
  // // initialize array elements on CPU with pointers into d_a
  for (size_t i = 0; i < N; i++) {
    h_a[i] = (i + stride) % N;
    h_a[i] = (unsigned int)-1;
  }

  compute_random_pointer_chain<unsigned int>(h_a, N, stride);
  for (size_t i = 0; i < N; i++) {
    fprintf(stderr, "HAVE: h_a[%lu] = %u\n", i, h_a[i]);
    // h_a[i] = (h_a_offsets[i] + stride) % N;
    // h_a[i] = (i + stride) % N;
  }

  // h_a[N] = 0;
  // h_a[N + 1] = 0;

  // free(h_a_offsets);
  // shuffle(h_a, h_a + N, std::default_random_engine(seed));
  // for (size_t k = 0; k < N

  size_t round_size = N / stride;

  std::unordered_set<unsigned int> unique_indices{};

  unsigned int j = 0;
  for (size_t i = 0; i < N; i++) {
    fprintf(stderr, "h_a[%u] => %u\n", j, h_a[j]);
    assert((long int)j - (long int)h_a[j] >= stride);
    j = h_a[j];
    unique_indices.insert(j);
  }
  fprintf(stderr, "unique indices: %lu\n", unique_indices.size());
  assert(unique_indices.size() == round_size);

  j = 0;
  unique_indices.clear();
  for (size_t i = 0; i < 4 * N; i++) {
    assert((long int)j - (long int)h_a[j] >= stride);
    j = h_a[j];
    unique_indices.insert(j);
  }
  assert(unique_indices.size() == round_size);

  // std::unordered_set<unsigned int> unique_indices{};
  // unsigned int j = 0;
  // for (size_t i = 0; i < N; i++) {
  //   fprintf(stderr, "h_a[%u] => %u\n", j, h_a[j]);
  //   j = h_a[j];
  //   unique_indices.insert(j);
  // }
  // fprintf(stderr, "unique indices: %lu\n", unique_indices.size());
  // assert(unique_indices.size() == round_size);

  return 0;

  // todo: seed is repetition, use same shuffle per round

  overflow_index = overflow_index % N;
  assert(N % stride == 0);

  CUDA_CHECK(cudaMemset(d_a, 0, N * sizeof(uint32_t)));

  CUDA_CHECK(
      cudaMemcpy(d_a, h_a, N * sizeof(uint32_t), cudaMemcpyHostToDevice));

  unsigned int *h_index, *h_latency, *d_index, *d_latency;
  CUDA_CHECK(cudaHostAlloc(&h_index, iter_size * sizeof(unsigned int),
                           cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc(&h_latency, iter_size * sizeof(unsigned int),
                           cudaHostAllocMapped));

  CUDA_CHECK(cudaHostGetDevicePointer(&d_index, h_index, 0));
  CUDA_CHECK(cudaHostGetDevicePointer(&d_latency, h_latency, 0));

  for (size_t k = 0; k < iter_size; k++) {
    h_index[k] = 0;
    h_latency[k] = 0;
  }

  cudaDeviceSynchronize();

  // launch kernel
  dim3 block_dim = dim3(1);
  dim3 grid_dim = dim3(1, 1, 1);

  switch (mem) {
  case L1Texture:
    assert(0 && "todo: texture");
    break;
  case L1ReadOnly:
    assert(0 && "todo: readonly");
    break;
  case L2:
    CUDA_CHECK((global_latency_l2_random_set_mapping_host_mapped<<<grid_dim,
                                                                   block_dim>>>(
        d_a, N, d_latency, d_index, iter_size, warmup_iterations,
        overflow_index)));
    break;
  case L1Data:
    CUDA_CHECK((global_latency_l1_random_set_mapping_host_mapped<<<grid_dim,
                                                                   block_dim>>>(
        d_a, N, d_latency, d_index, iter_size, warmup_iterations,
        overflow_index)));
    break;
  default:
    assert(false && "error dispatching to memory");
    break;
  };
  cudaDeviceSynchronize();

  CUDA_CHECK(cudaGetLastError());

  // copy results from GPU to CPU
  cudaDeviceSynchronize();

  cudaDeviceSynchronize();

  // unsigned int j = 0;

  switch (mem) {
  case L1ReadOnly:
    assert(0 && "todo: readonly");
    break;
  default:
    for (size_t k = 0; k < iter_size; k++) {
      unsigned int index;
      if (compute_capability == 86) {
        index = indexof(h_a, N, h_index[k]);
      } else {
        index = indexof(h_a, N, h_index[k]);
      }
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

  CUDA_CHECK(cudaFreeHost(h_index));
  CUDA_CHECK(cudaFreeHost(h_latency));

  return EXIT_SUCCESS;
}

int main(int argc, char *argv[]) {
  cudaSetDevice(0);
  memory mem;
  size_t size_bytes, stride_bytes, warmup_iterations;
  size_t repetitions = 1;
  size_t iter_size = (size_t)-1;

  char *compute_capability_env = getenv("COMPUTE_CAPABILITY");

  size_t compute_capability = 0;
  if (compute_capability_env != NULL) {
    compute_capability = (size_t)atoi(compute_capability_env);
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
  CUDA_CHECK(
      cudaFuncSetCacheConfig(global_latency_l1_random_set_mapping_host_mapped,
                             prefer_shared_mem_config));
  CUDA_CHECK(
      cudaFuncSetCacheConfig(global_latency_l2_random_set_mapping_host_mapped,
                             prefer_shared_mem_config));

  // use smallest L1 data cache on volta+
  int shared_mem_carveout_percent = 75;
  CUDA_CHECK(
      cudaFuncSetAttribute(global_latency_l1_random_set_mapping_host_mapped,
                           cudaFuncAttributePreferredSharedMemoryCarveout,
                           shared_mem_carveout_percent));
  CUDA_CHECK(
      cudaFuncSetAttribute(global_latency_l2_random_set_mapping_host_mapped,
                           cudaFuncAttributePreferredSharedMemoryCarveout,
                           shared_mem_carveout_percent));

  // fetch full 128B cache lines
  // CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 128))
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 32))

  CUDA_CHECK(cudaDeviceSetCacheConfig(prefer_shared_mem_config));
  cudaFuncCache have_cache_config;
  CUDA_CHECK(cudaDeviceGetCacheConfig(&have_cache_config));
  assert(have_cache_config == prefer_shared_mem_config);

  if (iter_size == (size_t)-1) {
    // default to 4 rounds through N
    iter_size = size_bytes * 4;
  }
  size_t size = size_bytes / sizeof(uint32_t);

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
  fprintf(stderr, "\tBACKING MEM        = HOST MAPPED");

  // print CSV header
  fprintf(stdout, "r,n,overflow_index,k,index,virt_addr,latency\n");

  for (size_t r = 0; r < repetitions; r++) {
    for (unsigned int overflow_index = 0; overflow_index < size;
         overflow_index += stride) {
      exit_code = parametric_measure_global(
          h_a, d_a, mem, size, stride, iter_size, warmup_iterations, r,
          compute_capability, clock_overhead, overflow_index);
      if (exit_code != EXIT_SUCCESS) {
        break;
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
