#include <stdio.h>

// Number of threads in each thread block
const unsigned BLOCK_SIZE = 32;

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

__device__ __forceinline__ int ldg(void *ptr) {
  int ret;
  asm volatile("ld.global.b32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}

// __device__ __forceinline__ int ldg_cs(const void *ptr) {
//   int ret;
//   asm volatile("ld.global.cs.b32 %0, [%1];" : "=r"(ret) : "l"(ptr));
//   return ret;
// }

__global__ void two_level_nested_if_imbalanced_kernel(float *arr) {
  size_t tid = threadIdx.x;
  __volatile__ float value1 = arr[100];
  if (tid < 16) {
    __volatile__ float value2 = arr[1];
    if (tid < 8) {
      __volatile__ float value3 = arr[2];
    }
  }
  __volatile__ float value4 = arr[100];
}

__global__ void two_level_nested_if_balanced_kernel(float *arr) {
  size_t tid = threadIdx.x;
  __volatile__ float value1 = arr[100];
  if (tid < 16) {
    __volatile__ float value2 = arr[10];
    if (tid < 8) {
      __volatile__ float value3 = arr[11];
    }
  } else {
    __volatile__ float value4 = arr[20];
    if (tid >= 24) {
      __volatile__ float value5 = arr[21];
    }
  }
  __volatile__ float value6 = arr[100];
}

__global__ void single_if(float *arr) {
  size_t tid = threadIdx.x;
  if (tid < 16) {
    ldg(&arr[1]);
    // __volatile__ float value1 = arr[1];
  }
}

__global__ void single_for_loop(float *arr) {
  size_t tid = threadIdx.x;
  if (tid < 16) {
    __volatile__ float value1 = arr[1];
  }
  bool is_even = tid % 2 == 0;
  size_t num_iterations = is_even ? 3 : 1;
  for (size_t i = 0; i < num_iterations; i++) {
    __volatile__ float value1 = arr[1];
  }
  __volatile__ float value2 = arr[100];
}

int main() {
  size_t bytes = 500 * sizeof(float);
  float *h_arr = (float *)malloc(bytes);
  float *d_arr;
  CUDA_SAFECALL(cudaMalloc(&d_arr, bytes));
  CUDA_SAFECALL(cudaMemcpy(d_arr, h_arr, bytes, cudaMemcpyHostToDevice));
  size_t gridSize = 1;

  // imbalanced
  CUDA_SAFECALL(
      (two_level_nested_if_imbalanced_kernel<<<gridSize, BLOCK_SIZE>>>(d_arr)));
  // balanced
  CUDA_SAFECALL(
      (two_level_nested_if_balanced_kernel<<<gridSize, BLOCK_SIZE>>>(d_arr)));
  // single if
  CUDA_SAFECALL((single_if<<<gridSize, BLOCK_SIZE>>>(d_arr)));
  // single for loop
  CUDA_SAFECALL((single_for_loop<<<gridSize, BLOCK_SIZE>>>(d_arr)));

  // cleanup
  CUDA_SAFECALL(cudaFree(d_arr));
  free(h_arr);
  return 0;
}
