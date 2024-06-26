/*
 *
 * globalWrite.cu
 *
 * Microbenchmark for write bandwidth to global memory.
 *
 * Build with: nvcc -I ../chLib --gpu-architecture sm_11 <options>
 * globalWrite.cu Requires: No minimum SM requirement.
 *
 * Copyright (c) 2011-2012, Archaea Software, LLC.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <stdio.h>

#include <chCommandLine.h>
#include <chError.h>

template <class T> __device__ __host__ T plus(const T &a, const T &b) {
  T ret = a;
  ret += b;
  return ret;
}

struct myInt2 {
  int2 i2;

  __host__ __device__ myInt2() {}
  __host__ __device__ myInt2(int i) { i2.x = i2.y = i; }
};

template <> __device__ __host__ myInt2 plus(const myInt2 &a, const myInt2 &b) {
  myInt2 ret;
  ret.i2.x = a.i2.x + b.i2.x;
  ret.i2.y = a.i2.y + b.i2.y;
  return ret;
}

struct myInt4 {
  int4 i4;

  __host__ __device__ myInt4() {}
  __host__ __device__ myInt4(int i) { i4.x = i4.y = i4.z = i4.w = i; }
};

template <> __device__ __host__ myInt4 plus(const myInt4 &a, const myInt4 &b) {
  myInt4 ret;
  ret.i4.x = a.i4.x + b.i4.x;
  ret.i4.y = a.i4.y + b.i4.y;
  ret.i4.z = a.i4.z + b.i4.z;
  ret.i4.w = a.i4.w + b.i4.w;
  return ret;
}

template <class T, const int n>
__global__ void GlobalWrites(T *out, T value, size_t N) {
  size_t i;
  for (i = n * blockIdx.x * blockDim.x + threadIdx.x;
       i < N - n * blockDim.x * gridDim.x; i += n * blockDim.x * gridDim.x) {
    for (int j = 0; j < n; j++) {
      size_t index = i + j * blockDim.x;
      out[index] = value;
    }
  }
  // to avoid the (index<N) conditional in the inner loop,
  // we left off some work at the end
  for (int j = 0; j < n; j++) {
    size_t index = i + j * blockDim.x;
    if (index < N)
      out[index] = value;
  }
}

template <class T, const int n, bool bOffset>
double BandwidthWrites(size_t N, int cBlocks, int cThreads) {
  T *out = 0;
  double ret = 0.0;
  double elapsedTime;
  float ms;
  int cIterations;
  cudaError_t status;
  cudaEvent_t evStart = 0;
  cudaEvent_t evStop = 0;

  cuda(Malloc(&out, N * sizeof(T)));

  cuda(EventCreate(&evStart));
  cuda(EventCreate(&evStop));

  cIterations = 10;
  cudaEventRecord(evStart);
  for (int i = 0; i < cIterations; i++) {
    GlobalWrites<T, n>
        <<<cBlocks, cThreads>>>(out + bOffset, (T)0xcc, N - bOffset);
  }
  cudaEventRecord(evStop);
  cuda(DeviceSynchronize());
  // make configurations that cannot launch error-out with 0 bandwidth
  cuda(GetLastError());
  cuda(EventElapsedTime(&ms, evStart, evStop));
  elapsedTime = ms / 1000.0f;

  // bytes per second
  ret = ((double)N * cIterations * sizeof(T)) / elapsedTime;
  // gigabytes per second
  ret /= 1024.0 * 1048576.0;

Error:
  cudaEventDestroy(evStart);
  cudaEventDestroy(evStop);
  cudaFree(out);
  return ret;
}

template <class T, const int n, bool bOffset>
double ReportRow(size_t N, size_t threadStart, size_t threadStop,
                 size_t cBlocks) {
  int maxThreads = 0;
  double maxBW = 0.0;
  printf("%d\t", n);
  for (int cThreads = threadStart; cThreads <= threadStop; cThreads *= 2) {
    double bw;
    bw = BandwidthWrites<T, n, bOffset>(N, cBlocks, cThreads);
    if (bw > maxBW) {
      maxBW = bw;
      maxThreads = cThreads;
    }
    printf("%.2f\t", bw);
  }
  printf("%.2f\t%d\n", maxBW, maxThreads);
  return maxBW;
}

template <class T, bool bCoalesced>
void Shmoo(size_t N, size_t threadStart, size_t threadStop, size_t cBlocks) {
  printf("Operand size: %d byte%c\n", (int)sizeof(T),
         sizeof(T) == 1 ? '\0' : 's');
  printf("Input size: %dM operands\n", (int)(N >> 20));
  printf("Unroll\t");
  for (int cThreads = threadStart; cThreads <= threadStop; cThreads *= 2) {
    printf("%d\t", cThreads);
  }
  printf("maxBW\tmaxThreads\n");
  ReportRow<T, 1, !bCoalesced>(N, threadStart, threadStop, cBlocks);
  ReportRow<T, 2, !bCoalesced>(N, threadStart, threadStop, cBlocks);
  ReportRow<T, 3, !bCoalesced>(N, threadStart, threadStop, cBlocks);
  ReportRow<T, 4, !bCoalesced>(N, threadStart, threadStop, cBlocks);
  ReportRow<T, 5, !bCoalesced>(N, threadStart, threadStop, cBlocks);
  ReportRow<T, 6, !bCoalesced>(N, threadStart, threadStop, cBlocks);
  ReportRow<T, 7, !bCoalesced>(N, threadStart, threadStop, cBlocks);
  ReportRow<T, 8, !bCoalesced>(N, threadStart, threadStop, cBlocks);
  ReportRow<T, 9, !bCoalesced>(N, threadStart, threadStop, cBlocks);
  ReportRow<T, 10, !bCoalesced>(N, threadStart, threadStop, cBlocks);
  ReportRow<T, 11, !bCoalesced>(N, threadStart, threadStop, cBlocks);
  ReportRow<T, 12, !bCoalesced>(N, threadStart, threadStop, cBlocks);
  ReportRow<T, 13, !bCoalesced>(N, threadStart, threadStop, cBlocks);
  ReportRow<T, 14, !bCoalesced>(N, threadStart, threadStop, cBlocks);
  ReportRow<T, 15, !bCoalesced>(N, threadStart, threadStop, cBlocks);
  ReportRow<T, 16, !bCoalesced>(N, threadStart, threadStop, cBlocks);
}

int main(int argc, char *argv[]) {
  int device = 0;
  int size = 16;
  if (chCommandLineGet(&device, "device", argc, argv)) {
    printf("Using device %d...\n", device);
  }
  cudaSetDevice(device);
  if (chCommandLineGet(&size, "size", argc, argv)) {
    printf("Using %dM operands ...\n", size);
  }

  if (chCommandLineGetBool("uncoalesced", argc, argv)) {
    printf("Using uncoalesced memory transactions\n");
    Shmoo<char, false>((size_t)size * 1048576, 32, 512, 150);
    Shmoo<short, false>((size_t)size * 1048576, 32, 512, 150);
    Shmoo<int, false>((size_t)size * 1048576, 32, 512, 150);
    Shmoo<myInt2, false>((size_t)size * 1048576, 32, 512, 150);
    Shmoo<myInt4, false>((size_t)size * 1048576, 32, 512, 150);
  } else {
    printf("Using coalesced memory transactions\n");
    Shmoo<char, true>((size_t)size * 1048576, 32, 512, 150);
    Shmoo<short, true>((size_t)size * 1048576, 32, 512, 150);
    Shmoo<int, true>((size_t)size * 1048576, 32, 512, 150);
    Shmoo<myInt2, true>((size_t)size * 1048576, 32, 512, 150);
    Shmoo<myInt4, true>((size_t)size * 1048576, 32, 512, 150);
  }
  return 0;
}
