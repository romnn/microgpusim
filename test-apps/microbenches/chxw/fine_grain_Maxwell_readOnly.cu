#include <stdint.h>
#include <stdio.h>

#include "cuda_runtime.h"

__global__ void global_latency(const unsigned int *__restrict__ my_array,
                               int array_length, int iterations,
                               unsigned int *duration, unsigned int *index);

void parametric_measure_global(int N, int iterations, int stride);

void measure_global();

int main() {

  cudaSetDevice(0);

  measure_global();

  cudaDeviceReset();
  return 0;
}

void measure_global() {

  int N, iterations, stride;
  // stride in element
  iterations = 64;
  stride = 32;

  // N_min =24, N_max=60
  N = 24 * 256;

  parametric_measure_global(N, iterations, stride);
}

void parametric_measure_global(int N, int iterations, int stride) {
  cudaDeviceReset();

  cudaError_t error_id;

  int i;
  unsigned int *h_a;
  /* allocate arrays on CPU */
  h_a = (unsigned int *)malloc(sizeof(unsigned int) * (N));
  unsigned int *d_a;
  /* allocate arrays on GPU */
  error_id = cudaMalloc((void **)&d_a, sizeof(unsigned int) * (N));
  if (error_id != cudaSuccess) {
    printf("Error 1.0 is %s\n", cudaGetErrorString(error_id));
  }

  /* initialize array elements on CPU with pointers into d_a. */

  for (i = 0; i < N; i++) {
    // original:
    h_a[i] = (i + stride) % N;
  }

  /* copy array elements from CPU to GPU */
  error_id =
      cudaMemcpy(d_a, h_a, sizeof(unsigned int) * N, cudaMemcpyHostToDevice);
  if (error_id != cudaSuccess) {
    printf("Error 1.1 is %s\n", cudaGetErrorString(error_id));
  }

  int s_num = 32 * iterations;
  unsigned int *h_index = (unsigned int *)malloc(sizeof(unsigned int) * s_num);
  unsigned int *h_timeinfo =
      (unsigned int *)malloc(sizeof(unsigned int) * s_num);

  unsigned int *duration;
  error_id = cudaMalloc((void **)&duration, sizeof(unsigned int) * s_num);
  if (error_id != cudaSuccess) {
    printf("Error 1.2 is %s\n", cudaGetErrorString(error_id));
  }

  unsigned int *d_index;
  error_id = cudaMalloc((void **)&d_index, sizeof(unsigned int) * s_num);
  if (error_id != cudaSuccess) {
    printf("Error 1.3 is %s\n", cudaGetErrorString(error_id));
  }

  cudaThreadSynchronize();
  /* launch kernel*/
  dim3 Db = dim3(32, 1, 1);
  dim3 Dg = dim3(1, 1, 1);

  global_latency<<<Dg, Db>>>(d_a, N, iterations, duration, d_index);

  cudaThreadSynchronize();

  error_id = cudaGetLastError();
  if (error_id != cudaSuccess) {
    printf("Error kernel is %s\n", cudaGetErrorString(error_id));
  }

  /* copy results from GPU to CPU */
  cudaThreadSynchronize();

  error_id = cudaMemcpy((void *)h_timeinfo, (void *)duration,
                        sizeof(unsigned int) * s_num, cudaMemcpyDeviceToHost);
  if (error_id != cudaSuccess) {
    printf("Error 2.0 is %s\n", cudaGetErrorString(error_id));
  }
  error_id = cudaMemcpy((void *)h_index, (void *)d_index,
                        sizeof(unsigned int) * s_num, cudaMemcpyDeviceToHost);
  if (error_id != cudaSuccess) {
    printf("Error 2.1 is %s\n", cudaGetErrorString(error_id));
  }

  cudaThreadSynchronize();

  for (i = 0; i < s_num; i += 32) {
    printf("%d \n", h_timeinfo[i]);
  }

  /* free memory on GPU */
  cudaFree(d_a);
  cudaFree(d_index);
  cudaFree(duration);

  /*free memory on CPU */
  free(h_a);
  free(h_index);
  free(h_timeinfo);

  cudaDeviceReset();
}

__global__ void global_latency(const unsigned int *__restrict__ my_array,
                               int array_length, int iterations,
                               unsigned int *duration, unsigned int *index) {

  unsigned int start_time, end_time;
  unsigned int j = threadIdx.x;

  __shared__ unsigned int s_tvalue[2048];
  __shared__ unsigned int s_index[2048];

  int it, k;

  for (k = 0; k < 2048; k++) {
    s_index[k] = 0;
    s_tvalue[k] = 0;
  }

  // no-timing iterations, for large arrays
  for (it = 0; it < 512; it++)
    j = __ldg(&my_array[j]);

  // second round
  for (it = 0; it < iterations; it++) {
    k = it * blockDim.x + threadIdx.x;

    start_time = clock();
    j = __ldg(&my_array[j]);
    s_index[k] = j;
    end_time = clock();
    s_tvalue[k] = end_time - start_time;
  }

  // copy the indices and memory latencies back to global memory
  for (it = 0; it < iterations; it++) {
    k = it * blockDim.x + threadIdx.x;
    index[k] = s_index[k];
    duration[k] = s_tvalue[k];
  }
}
