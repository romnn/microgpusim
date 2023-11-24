#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Number of threads in each thread block
const unsigned BLOCK_SIZE = 1024;

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

// simple copy kernel for stressing the L1 and L2 cache
__global__ void gpucachesim_skip_copy(float *a, float *b, float *dest,
                                      unsigned count, unsigned loops) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  size_t start = id * count;

  for (int l = 0; l < loops; l++) {
    // for (int i = threadIdx.x+blockDim.x*blockIdx.x; i<len;
    // i+=gridDim.x*blockDim.x)
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += gridDim.x * blockDim.x) {
      // for (int i = start; i < count; i++) {
      // float temp = dest[i];
      // dest[i] = a[i] + b[i];
      dest[i] = __ldg(&a[i]) + __ldg(&b[i]);

      // dest[i] = __ldca(src + i);
      // dest[i] = __ldca(src[i]);
      // dest[i] = __ldcg(src + i);
    }
  }
  // float4 val;
  // const float4* myinput = input+i;
  // asm("ld.global.cv.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(val.x),
  // "=f"(val.y), "=f"(val.z), "=f"(val.w) : "l"(myinput));

  // __threadfence_system();
}

// find largest power of 2
unsigned flp2(unsigned x) {
  x = x | (x >> 1);
  x = x | (x >> 2);
  x = x | (x >> 4);
  x = x | (x >> 8);
  x = x | (x >> 16);
  return x - (x >> 1);
}

void invalidate_caches() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  const unsigned num_threads_per_sm = prop.maxThreadsPerMultiProcessor;
  const unsigned num_sm = prop.multiProcessorCount;
  const unsigned l2_size = prop.l2CacheSize;
  const unsigned buffer_size_bytes = flp2(l2_size) * 10; // double the l2 size
  const unsigned buffer_size = buffer_size_bytes / sizeof(float);

  const unsigned block_size = 512;
  const unsigned num_blocks = (num_sm * num_threads_per_sm) / block_size;
  const unsigned loops = 2;

  printf("grid: (%d,1,1)\n", num_blocks);
  printf("threads: (%d,1,1)\n", block_size);

  float *a, *b, *dest;
  CUDA_SAFECALL(cudaMalloc(&a, buffer_size * sizeof(float)));
  CUDA_SAFECALL(cudaMalloc(&b, buffer_size * sizeof(float)));
  CUDA_SAFECALL(cudaMalloc(&dest, buffer_size * sizeof(float)));
  CUDA_SAFECALL((gpucachesim_skip_copy<<<num_blocks, block_size>>>(
      a, b, dest, buffer_size, loops)));
  CUDA_SAFECALL(cudaDeviceSynchronize());
  CUDA_SAFECALL(cudaFree(a));
  CUDA_SAFECALL(cudaFree(b));
  CUDA_SAFECALL(cudaFree(dest));
}

void flush_l2_cache() {
  int dev_id = 0;
  CUDA_SAFECALL(cudaGetDevice(&dev_id));
  int l2_size = 0;
  CUDA_SAFECALL(
      cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, dev_id));

  if (l2_size > 0) {
    for (int i = 0; i < 20; i++) {
      int *l2_buffer;
      CUDA_SAFECALL(cudaMalloc(&l2_buffer, l2_size * 2));

      char *host_l2_buffer = (char *)malloc(l2_size * 2);
      CUDA_SAFECALL(cudaMemcpy(l2_buffer, host_l2_buffer, l2_size * 2,
                               cudaMemcpyHostToDevice));

      // CUDA_SAFECALL(cudaMemsetAsync(l2_buffer, 0, l2_size, stream));
      // CUDA_SAFECALL(cudaMemset(l2_buffer, 0, l2_size * 2));
      CUDA_SAFECALL(cudaDeviceSynchronize());
      CUDA_SAFECALL(cudaFree(l2_buffer));
    }
  }
}

// CUDA kernel. Each thread takes care of one element of c
template <typename T>
// __global__ void vecAdd(const T *__restrict__ a, T *b, T *c, int n) {
__global__ void vecAdd(T *a, T *b, T *c, int n) {
  // __shared__ int dummy_shared[BLOCK_SIZE * 32];
  // dummy_shared[threadIdx.x] = a[0];

  // Get our global thread ID
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  // __volatile__ size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  // __volatile__ size_t id = threadIdx.x;

  // Make sure we do not go out of bounds
  if (id < n) {
    // __volatile__ size_t test = a[id] + b[id];
    // printf("c[%lu] = a[%lu] + b[%lu] = %f\n", id, id, id, a[id] + b[id]);
    c[id] = a[id] + b[id];
    // c[id] = a[id] + 32.0;
  }

  // __threadfence_system();
}

template <typename T> int vectoradd(int n) {
  // Host input vectors
  T *h_a;
  T *h_b;
  // Host output vector
  T *h_c;

  // Device input vectors
  T *d_a;
  T *d_b;
  // Device output vector
  T *d_c;

  // Size, in bytes, of each vector
  size_t bytes = n * sizeof(T);

  // Allocate memory for each vector on host
  h_a = (T *)malloc(bytes);
  h_b = (T *)malloc(bytes);
  h_c = (T *)malloc(bytes);

  // Allocate memory for each vector on GPU
  CUDA_SAFECALL(cudaMalloc(&d_a, bytes));
  CUDA_SAFECALL(cudaMalloc(&d_b, bytes));
  CUDA_SAFECALL(cudaMalloc(&d_c, bytes));

  int i;
  // Initialize vectors on host
  for (i = 0; i < n; i++) {
    h_a[i] = sin(i) * sin(i);
    h_b[i] = cos(i) * cos(i);
    h_c[i] = 0;
  }

  // Copy host vectors to device
  CUDA_SAFECALL(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
  CUDA_SAFECALL(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
  // CUDA_SAFECALL(cudaMemcpy(d_c, h_c, bytes, cudaMemcpyHostToDevice));

  // invalidate all caches
  // invalidate_caches();

  // Number of thread blocks in grid
  unsigned gridSize = (unsigned)ceil((float)n / BLOCK_SIZE);
  printf("grid: (%d,1,1)\n", gridSize);
  printf("threads: (%d,1,1)\n", BLOCK_SIZE);

  // Execute the kernel
  size_t repetitions = 1;
  for (size_t i = 0; i < repetitions; i++) {
    CUDA_SAFECALL((vecAdd<T><<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_c, n)));

    // reset the result vector
    // CUDA_SAFECALL(cudaMemset(d_c, 0, bytes));
    // CUDA_SAFECALL(cudaDeviceSynchronize());

    // flush_l2_cache();

    // Copy array back to host
    // CUDA_SAFECALL(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
  }

  // Copy array back to host
  CUDA_SAFECALL(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

  // Sum up vector c and print result divided by n, this should equal 1 within
  // error
  T sum = 0;
  for (i = 0; i < n; i++)
    sum += h_c[i];
  printf("Final sum = %f; sum/n = %f (should be ~1)\n", sum, sum / n);

  // Release device memory
  CUDA_SAFECALL(cudaFree(d_a));
  CUDA_SAFECALL(cudaFree(d_b));
  CUDA_SAFECALL(cudaFree(d_c));

  // Release host memory
  free(h_a);
  free(h_b);
  free(h_c);

  return 0;
}

int main(int argc, char *argv[]) {
  // Size of vectors
  int n = 100; // used to be 100 000
  bool use_double = false;
  if (argc > 2) {
    n = atoi(argv[1]);
    if (atoi(argv[2]) == 64)
      use_double = true;
  } else {
    fprintf(stderr, "usage: vectoradd <n> <datatype>\n");
    return 1;
  }
  // 1M floats * 4 bytes * 2 = 8MB (vs. 2MB l2 cache)
  // n = 1000000;

  if (use_double) {
    return vectoradd<double>(n);
  } else {
    return vectoradd<float>(n);
  }
}
