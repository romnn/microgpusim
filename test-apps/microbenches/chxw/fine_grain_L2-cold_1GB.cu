#include <stdint.h>
#include <stdio.h>

#include "cuda_runtime.h"

// compile nvcc *.cu -o test

__global__ void global_latency(unsigned int *my_array, int array_length,
                               int iterations, unsigned int *duration,
                               unsigned int *index);

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
  iterations = 1;

  N = 1024 * 1024 * 1024 / sizeof(unsigned int); // in element
  for (stride = 1; stride <= N / 2; stride *= 2) {
    printf("\n=====%d GB array, cold cache miss, read 256 element====\n",
           N / 1024 / 1024 / 1024);
    printf("Stride = %d element, %d bytes\n", stride,
           stride * sizeof(unsigned int));
    parametric_measure_global(N, iterations, stride);
    printf("===============================================\n\n");
  }
}

void parametric_measure_global(int N, int iterations, int stride) {
  cudaDeviceReset();

  int i;
  unsigned int *h_a;
  /* allocate arrays on CPU */
  h_a = (unsigned int *)malloc(sizeof(unsigned int) * (N + 2));
  unsigned int *d_a;
  /* allocate arrays on GPU */
  cudaMalloc((void **)&d_a, sizeof(unsigned int) * (N + 2));

  /* initialize array elements on CPU with pointers into d_a. */

  for (i = 0; i < N; i++) {
    // original:
    h_a[i] = (i + stride) % N;
  }

  h_a[N] = 0;
  h_a[N + 1] = 0;
  /* copy array elements from CPU to GPU */
  cudaMemcpy(d_a, h_a, sizeof(unsigned int) * N, cudaMemcpyHostToDevice);

  unsigned int *h_index = (unsigned int *)malloc(sizeof(unsigned int) * 256);
  unsigned int *h_timeinfo = (unsigned int *)malloc(sizeof(unsigned int) * 256);

  unsigned int *duration;
  cudaMalloc((void **)&duration, sizeof(unsigned int) * 256);

  unsigned int *d_index;
  cudaMalloc((void **)&d_index, sizeof(unsigned int) * 256);

  cudaThreadSynchronize();
  /* launch kernel*/
  dim3 Db = dim3(1);
  dim3 Dg = dim3(1, 1, 1);

  global_latency<<<Dg, Db>>>(d_a, N, iterations, duration, d_index);

  cudaThreadSynchronize();

  cudaError_t error_id = cudaGetLastError();
  if (error_id != cudaSuccess) {
    printf("Error kernel is %s\n", cudaGetErrorString(error_id));
  }

  /* copy results from GPU to CPU */
  cudaThreadSynchronize();

  cudaMemcpy((void *)h_timeinfo, (void *)duration, sizeof(unsigned int) * 256,
             cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)h_index, (void *)d_index, sizeof(unsigned int) * 256,
             cudaMemcpyDeviceToHost);

  cudaThreadSynchronize();

  for (i = 0; i < 256; i++)
    printf("%d\t %d\n", h_index[i], h_timeinfo[i]);

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

__global__ void global_latency(unsigned int *my_array, int array_length,
                               int iterations, unsigned int *duration,
                               unsigned int *index) {

  unsigned int start_time, end_time;
  unsigned int j = 0;

  __shared__ unsigned int s_tvalue[256];
  __shared__ unsigned int s_index[256];

  int k;

  for (k = 0; k < 256; k++) {
    s_index[k] = 0;
    s_tvalue[k] = 0;
  }

  for (k = 0; k < iterations * 256; k++) {

    start_time = clock();

    j = my_array[j];
    s_index[k] = j;
    end_time = clock();

    s_tvalue[k] = end_time - start_time;
  }

  my_array[array_length] = j;
  my_array[array_length + 1] = my_array[j];

  for (k = 0; k < 256; k++) {
    index[k] = s_index[k];
    duration[k] = s_tvalue[k];
  }
}
