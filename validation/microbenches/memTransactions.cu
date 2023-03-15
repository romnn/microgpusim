__device__ void store(float4 *values, int id, int index) {
  // Generate 32 different store instructions, the first done only on the first
  // thread, the next done only on the first two threads, etc. and the last done
  // on all threads.
  //    #pragma unroll
  for (int numThreads = 1; numThreads <= 32; ++numThreads)
    if (id < numThreads)
      values[index] = float4(); // Store a dummy value
}

__global__ void sameAddress(float4 *values) { store(values, threadIdx.x, 0); }
__global__ void sequentialAddresses(float4 *values) {
  store(values, threadIdx.x, threadIdx.x);
}
__global__ void separateCacheLines(float4 *values) {
  store(values, threadIdx.x, threadIdx.x * 128 / sizeof(float4));
}

int main() {
  // Allocate enough for worst case example: all 32 threads in the warp access a
  // different 128-byte cache line.
  float4 *values = 0;
  cudaMalloc((void **)&values, 32 * 128);

  // Launch example kernels with one warp.

  cudaDeviceSynchronize();
  // All threads access same element
  sameAddress<<<1, 32>>>(values);

  cudaDeviceSynchronize();
  // Threads access sequential elements ("ideal")
  sequentialAddresses<<<1, 32>>>(values);

  cudaDeviceSynchronize();
  // Each thread accesses a different 128-byte sector
  separateCacheLines<<<1, 32>>>(values);

  cudaDeviceSynchronize();
  return 0;
}
