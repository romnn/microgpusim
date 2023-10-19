#include <algorithm>
#include <assert.h>
#include <cstdlib>
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>

#include "common.hpp"
#include "cuda_runtime.h"

const size_t ITER_SIZE = ((48 * KB) / 2) / sizeof(uint32_t);

__global__ __noinline__ void
global_latency_l1_set_mapping(unsigned int *array, int array_length,
                              unsigned int *duration, unsigned int *index,
                              int iter_size, size_t warmup_iterations,
                              unsigned int overflow_index) {
  const int max_iter_size = ITER_SIZE;
  assert(iter_size <= max_iter_size);

  unsigned int start_time, end_time;
  volatile uint32_t j = 0;

  __shared__ volatile uint32_t s_tvalue[max_iter_size];
  __shared__ volatile uint32_t s_index[max_iter_size];

  for (size_t k = 0; k < iter_size; k++) {
    s_index[k] = 0;
    s_tvalue[k] = 0;
  }

  // // stage 1: pre-fill the entire cache
  // for (int w = 0; w < warmup_iterations; w++) {
  //   // j = 0;
  //   for (int k = 0; k < iter_size; k++) {
  //     // printf("warmup %d: load j=%u\n", w, j);
  //     j = array[j];
  //     s_index[k] = j;
  //   }
  // }
  // // for (int k = 0; k < warmup_iterations * iter_size; k++) {
  // //   j = array[j];
  // //   s_index[k % iter_size] = j;
  // // }
  //
  // // stage 2: overflow by a single cache line
  // // unsigned int temp = array[overflow_index];
  //
  // // stage 3: measure the latency of the access
  // // j = 0;
  // for (int k = 0; k < iter_size; k++) {
  //   // printf("timing: load j=%u\n", j);
  //   start_time = clock();
  //   j = array[j];
  //   s_index[k] = j;
  //   end_time = clock();
  //   s_tvalue[k] = end_time - start_time;
  // }

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

int parametric_measure_global(unsigned int *h_a, unsigned int *d_a, memory mem,
                              size_t N, size_t stride, size_t iter_size,
                              size_t warmup_iterations,
                              unsigned int clock_overhead, int repetition,
                              unsigned int overflow_index) {
  // initialize array elements on CPU with pointers into d_a
  for (size_t i = 0; i < N; i++) {
    h_a[i] = (i + stride) % N;
  }

  overflow_index = overflow_index % N;

  h_a[N] = 0;
  h_a[N + 1] = 0;

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

    // CUDA_SAFECALL((global_latency_l1_texture<<<grid_dim, block_dim>>>(
    //     d_a, texObj, N, duration, d_index, iter_size, warmup_iterations)));
    assert(0 && "todo");
    break;
  case L1ReadOnly:
    block_dim = dim3(32, 1, 1);
    assert(0 && "todo");
    // CUDA_SAFECALL((global_latency_l1_readonly<<<grid_dim, block_dim>>>(
    //     d_a, N, duration, d_index, iter_size, warmup_iterations)));
    break;
  case L2:
  case L1Data:
    CUDA_SAFECALL((global_latency_l1_set_mapping<<<grid_dim, block_dim>>>(
        d_a, N, duration, d_index, iter_size, warmup_iterations,
        overflow_index)));
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
      unsigned int index = (N + h_index[k] - stride) % N;
      unsigned int latency = (int)h_timeinfo[k] - (int)clock_overhead;
      unsigned long long virt_addr =
          (unsigned long long)d_a +
          (unsigned long long)index * (unsigned long long)sizeof(uint32_t);

      fprintf(stdout, "%8lu,%3d,%4u,%4lu,%4lu,%10llu,%4d\n",
              N * sizeof(uint32_t), repetition, overflow_index, k,
              index * sizeof(uint32_t), virt_addr, latency);
    }
    break;
  }

  // destroy texture object
  if (texObj != 0) {
    cudaDestroyTextureObject(texObj);
  }

  // free memory on GPU
  cudaFree(d_index);
  cudaFree(duration);

  // free memory on CPU
  free(h_index);
  free(h_timeinfo);

  return EXIT_SUCCESS;
}

int main(int argc, char *argv[]) {
  cudaSetDevice(0);
  memory mem;
  size_t size_bytes, stride_bytes, warmup_iterations;
  size_t repetitions = 1;
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

  size_t max_iter_size = ITER_SIZE;
  iter_size = std::min(iter_size, max_iter_size);

  size_t size = size_bytes / sizeof(uint32_t);

  if (size_bytes < 1) {
    fprintf(stderr, "ERROR: size is too small (%lu)\n", size_bytes);
    fflush(stderr);
    return EXIT_FAILURE;
  }

  // allocate arrays on CPU
  // unsigned int *h_a = (unsigned int *)malloc((size + 2) * sizeof(uint32_t));
  unsigned int *h_a =
      (unsigned int *)malloc(((size * 2) + 2) * sizeof(uint32_t));

  // allocate arrays on GPU
  unsigned int *d_a;
  CUDA_SAFECALL(cudaMalloc((void **)&d_a, ((size * 2) + 2) * sizeof(uint32_t)));

  int exit_code = EXIT_SUCCESS;

  float one_round = (float)size_bytes / (float)stride_bytes;
  float num_rounds = (float)iter_size / one_round;
  size_t stride = stride_bytes / sizeof(uint32_t);

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
  fprintf(stderr, "\tREPETITIONS        = %lu\n", repetitions);
  fprintf(stderr, "\tWARMUP ITERATIONS  = %lu\n", warmup_iterations);

  // print CSV header
  fprintf(stdout, "n,r,overflow_index,i,index,virt_addr,latency\n");

  // size_t overflow_index = 10 * stride;
  // size_t start_overflow_index = 130; //  * sizeof(uint32_t);
  // size_t start_overflow_index = 130; //  * sizeof(uint32_t);
  for (int r = 0; r < repetitions; r++) {
    for (unsigned int overflow_index = 0; overflow_index < iter_size;
         overflow_index += stride) {
      // size_t overflow_index = 130; //  * sizeof(uint32_t);
      exit_code = parametric_measure_global(h_a, d_a, mem, size, stride,
                                            iter_size, warmup_iterations,
                                            clock_overhead, r, overflow_index);
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
