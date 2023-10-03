#include <stdio.h>
#include <stdlib.h>

// declare the texture
texture<int, 1, cudaReadModeElementType> tex_ref;

__global__ void texture_latency(int *my_array, int size, unsigned int *duration,
                                int *index, int iter, unsigned int INNER_ITS) {

  const int it = 4096;

  __shared__ unsigned int s_tvalue[it];
  __shared__ int s_value[it];

  unsigned int start, end;
  int i, j;

  // initilize j
  j = 0;

  for (i = 0; i < it; i++) {
    s_value[i] = -1;
    s_tvalue[i] = 0;
  }

  for (int k = 0; k <= iter; k++) {

    for (int cnt = 0; cnt < it; cnt++) {

      start = clock();
      j = tex1Dfetch(tex_ref, j);
      s_value[cnt] = j;

      end = clock();
      s_tvalue[cnt] = (end - start);
    }
  }

  for (i = 0; i < it; i++) {
    duration[i] = s_tvalue[i];
    index[i] = s_value[i];
  }

  my_array[size] = i;
  my_array[size + 1] = s_tvalue[i - 1];
}

void parametric_measure_texture(int N, int iterations, int stride) {

  cudaError_t error_id;

  int *h_a, *d_a;
  int size = N * sizeof(int);
  h_a = (int *)malloc(size);
  // initialize array
  for (int i = 0; i < (N - 2); i++) {
    h_a[i] = (i + stride) % (N - 2);
  }
  h_a[N - 2] = 0;
  h_a[N - 1] = 0;
  cudaMalloc((void **)&d_a, size);
  // copy it to device array
  cudaMemcpy((void *)d_a, (void *)h_a, size, cudaMemcpyHostToDevice);

  // here to change the iteration numbers
  unsigned int INNER_ITS = 16;
  const int it = INNER_ITS * 256;

  // the time ivformation array and index array
  unsigned int *h_duration = (unsigned int *)malloc(it * sizeof(unsigned int));
  int *h_index = (int *)malloc(it * sizeof(int));

  int *d_index;
  error_id = cudaMalloc(&d_index, it * sizeof(int));
  if (error_id != cudaSuccess) {
    printf("Error 1.1 is %s\n", cudaGetErrorString(error_id));
  }

  unsigned int *d_duration;
  error_id = cudaMalloc(&d_duration, it * sizeof(unsigned int));
  if (error_id != cudaSuccess) {
    printf("Error 1.2 is %s\n", cudaGetErrorString(error_id));
  }

  // bind texture
  cudaBindTexture(0, tex_ref, d_a, size);

  cudaThreadSynchronize();

  error_id = cudaGetLastError();
  if (error_id != cudaSuccess) {
    printf("Error 2 is %s\n", cudaGetErrorString(error_id));
  }

  // for (int l=0; l < 20; l++) {

  // launch kernel
  dim3 Db = dim3(1);
  dim3 Dg = dim3(1, 1, 1);
  texture_latency<<<Dg, Db>>>(d_a, size, d_duration, d_index, iterations,
                              INNER_ITS);

  cudaThreadSynchronize();

  error_id = cudaGetLastError();
  if (error_id != cudaSuccess) {
    printf("Error 3 is %s\n", cudaGetErrorString(error_id));
  }

  cudaThreadSynchronize();

  /* copy results from GPU to CPU */
  cudaMemcpy((void *)h_index, (void *)d_index, it * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)h_duration, (void *)d_duration, it * sizeof(unsigned int),
             cudaMemcpyDeviceToHost);

  //}

  // print the result
  printf("\n=====Visting the %f KB array, loop %d*%d times======\n",
         (float)(N - 2) * sizeof(int) / 1024.0f, it, 1);
  for (int i = 0; i < it; i++) {
    printf("%10d\t %10f\n", h_index[i], (float)h_duration[i]);
  }

  // unbind texture
  cudaUnbindTexture(tex_ref);

  // free memory on GPU
  cudaFree(d_a);
  cudaFree(d_duration);
  cudaFree(d_index);
  cudaThreadSynchronize();

  // free memory on CPU
  free(h_a);
  free(h_duration);
  free(h_index);
}

int main() {

  cudaSetDevice(1); // 0 for Kepler, 1 for Fermi

  int N, iterations, stride;

  iterations = 1;

  // measure L1, should be 12 KB

  // stage1: overflow with 1 element
  stride = 1; // in element
  for (N = 3073; N <= 3073; N += stride) {
    parametric_measure_texture(N + 2, iterations, stride);
  }
  /*
          // stage2: overflow with cache lines
          stride = 8; // in element
          for (N = 3072; N <=3200; N += stride) {
                  parametric_measure_texture(N+2, iterations, stride);
          }
  */

  cudaDeviceReset();
  return 0;
}
