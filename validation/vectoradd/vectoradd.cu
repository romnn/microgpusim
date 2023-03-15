#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_SAFECALL(call)                                                {\
        call;                                                               \
        cudaError err = cudaGetLastError();                                 \
        if (cudaSuccess != err) {                                           \
            fprintf(                                                        \
                stderr,                                                     \
                "Cuda error in function '%s' file '%s' in line %i : %s.\n", \
                #call, __FILE__, __LINE__, cudaGetErrorString(err));        \
            fflush(stderr);                                                 \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }


// CUDA kernel. Each thread takes care of one element of c
template<typename T>
__global__ void vecAdd(T *a, T *b, T *c, int n) {
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n) c[id] = a[id] + b[id];
}

template<typename T>
int vectoradd(int n) {
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
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    int i;
    // Initialize vectors on host
    for (i = 0; i < n; i++) {
        h_a[i] = sin(i) * sin(i);
        h_b[i] = cos(i) * cos(i);
        h_c[i] = 0;
    }

    // Copy host vectors to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, bytes, cudaMemcpyHostToDevice);

    int blockSize, gridSize;

    // Number of threads in each thread block
    blockSize = 1024;

    // Number of thread blocks in grid
    gridSize = (int)ceil((float)n / blockSize);

    // Execute the kernel
    CUDA_SAFECALL((vecAdd<T><<<gridSize, blockSize>>>(d_a, d_b, d_c, n)));

    // Copy array back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Sum up vector c and print result divided by n, this should equal 1 within
    // error
    T sum = 0;
    for (i = 0; i < n; i++) sum += h_c[i];
    printf("Final sum = %f; sum/n = %f (should be ~1)\n", sum, sum / n);

    // Release device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

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
      if (atoi(argv[2]) == 64) use_double = true;
    } else {
      fprintf(stderr, "usage: vectoradd <n> <datatype>\n");
      return 1;
    }

    if (use_double) {
      return vectoradd<double>(n);
    } else {
      return vectoradd<float>(n);
    }
}
