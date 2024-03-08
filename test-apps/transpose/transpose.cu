/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// -----------------------------------------------------------------------------
// Transpose
//
// This file contains both device and host code for transposing a floating-point
// matrix.  It performs several transpose kernels, which incrementally improve
// performance through coalescing, removing shared memory bank conflicts, and
// eliminating partition camping.  Several of the kernels perform a copy, used
// to represent the best case performance that a transpose can achieve.
//
// Please see the whitepaper in the docs folder of the transpose project for a
// detailed description of this performance study.
// -----------------------------------------------------------------------------

#include <cooperative_groups.h>

namespace cg = cooperative_groups;
// Utilities and system includes
#include <helper_cuda.h>   // helper for cuda error checking functions
#include <helper_image.h>  // helper for image and data comparison
#include <helper_string.h> // helper for string parsing

const char *sSDKsample = "Transpose";

// Each block transposes/copies a tile of TILE_DIM x TILE_DIM elements
// using TILE_DIM x BLOCK_ROWS threads, so that each thread transposes
// TILE_DIM/BLOCK_ROWS elements.  TILE_DIM must be an integral multiple of
// BLOCK_ROWS

#define TILE_DIM 16
#define BLOCK_ROWS 16
// this config is used in the pdf documentation but is less efficient
// because one thread transposes 32/8=4 elements.
// #define TILE_DIM 32
// #define BLOCK_ROWS 8

// This sample assumes that MATRIX_SIZE_X = MATRIX_SIZE_Y
int MATRIX_SIZE_X = 1024;
int MATRIX_SIZE_Y = 1024;
int MUL_FACTOR = TILE_DIM;

#define FLOOR(a, b) (a - (a % b))

// Compute the tile size necessary to illustrate performance cases for SM20+
// hardware
int MAX_TILES = (FLOOR(MATRIX_SIZE_X, 512) * FLOOR(MATRIX_SIZE_Y, 512)) /
                (TILE_DIM * TILE_DIM);

// Number of repetitions used for timing.  Two sets of repetitions are
// performed: 1) over kernel launches and 2) inside the kernel over just the
// loads and stores

// -------------------------------------------------------
// Copies
// width and height must be integral multiples of TILE_DIM
// -------------------------------------------------------

__global__ void copy(float *odata, float *idata, int width, int height) {
  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

  int index = xIndex + width * yIndex;

  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    odata[index + i * width] = idata[index + i * width];
  }

  // __threadfence_system();
}

__global__ void copySharedMem(float *odata, float *idata, int width,
                              int height) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  __shared__ float tile[TILE_DIM][TILE_DIM];

  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

  int index = xIndex + width * yIndex;

  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    if (xIndex < width && yIndex < height) {
      tile[threadIdx.y][threadIdx.x] = idata[index];
    }
  }

  cg::sync(cta);

  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    if (xIndex < height && yIndex < width) {
      odata[index] = tile[threadIdx.y][threadIdx.x];
    }
  }

  // __threadfence_system();
}

// -------------------------------------------------------
// Transposes
// width and height must be integral multiples of TILE_DIM
// -------------------------------------------------------

__global__ void transposeNaive(float *odata, float *idata, int width,
                               int height) {
  /*
  Each thread executing the kernel
  transposes four elements from one column of the input
  matrix to their transposed locations in one row of the
  output matrix.
    */
  assert(TILE_DIM == blockDim.x);
  assert(BLOCK_ROWS == blockDim.y);
  // assert(TILE_DIM == BLOCK_ROWS);
  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

  int index_in = xIndex + width * yIndex;
  int index_out = yIndex + height * xIndex;

  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //   printf("thread (%3d, %3d, %3d): odata[%d] = idata[%d]\n", threadIdx.x,
    //          threadIdx.y, threadIdx.z, index_out + i, index_in + i * width);
    // }
    odata[index_out + i] = idata[index_in + i * width];
  }

  // __threadfence_system();
}

// coalesced transpose (with bank conflicts)

__global__ void transposeCoalesced(float *odata, float *idata, int width,
                                   int height) {
  assert(TILE_DIM == blockDim.x);
  assert(BLOCK_ROWS == blockDim.y);
  assert(TILE_DIM == BLOCK_ROWS);

  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  __shared__ float tile[TILE_DIM][TILE_DIM];

  // this is the same as for naive
  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
  int index_in = xIndex + (yIndex)*width;

  // this reverses
  xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
  yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
  int index_out = xIndex + (yIndex)*height;

  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    // load a tile to shared memory
    // each thread computes one column in the tile matrix
    // therefore there must exist TILE_DIM threads
    // printf("block (%d, %d, %d) warp %d thread %d (%d, %d, %d) => load %d\n",
    //        blockIdx.x, blockIdx.y, blockIdx.z, -1, -1, threadIdx.x,
    //        threadIdx.y, threadIdx.z, index_in + i * width);

    tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
  }

  // wait for the entire tile to be computed
  cg::sync(cta);

  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
  }

  // __threadfence_system();
}

// Coalesced transpose with no bank conflicts

__global__ void transposeNoBankConflicts(float *odata, float *idata, int width,
                                         int height) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
  int index_in = xIndex + (yIndex)*width;

  xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
  yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
  int index_out = xIndex + (yIndex)*height;

  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
  }

  cg::sync(cta);

  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
  }

  // __threadfence_system();
}

// Transpose that effectively reorders execution of thread blocks along
// diagonals of the matrix (also coalesced and has no bank conflicts)
//
// Here blockIdx.x is interpreted as the distance along a diagonal and
// blockIdx.y as corresponding to different diagonals
//
// blockIdx_x and blockIdx_y expressions map the diagonal coordinates to the
// more commonly used cartesian coordinates so that the only changes to the code
// from the coalesced version are the calculation of the blockIdx_x and
// blockIdx_y and replacement of blockIdx.x and bloclIdx.y with the subscripted
// versions in the remaining code

__global__ void transposeDiagonal(float *odata, float *idata, int width,
                                  int height) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  int blockIdx_x, blockIdx_y;

  // do diagonal reordering
  if (width == height) {
    blockIdx_y = blockIdx.x;
    blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
  } else {
    int bid = blockIdx.x + gridDim.x * blockIdx.y;
    blockIdx_y = bid % gridDim.y;
    blockIdx_x = ((bid / gridDim.y) + blockIdx_y) % gridDim.x;
  }

  // from here on the code is same as previous kernel except blockIdx_x replaces
  // blockIdx.x and similarly for y

  int xIndex = blockIdx_x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx_y * TILE_DIM + threadIdx.y;
  int index_in = xIndex + (yIndex)*width;

  xIndex = blockIdx_y * TILE_DIM + threadIdx.x;
  yIndex = blockIdx_x * TILE_DIM + threadIdx.y;
  int index_out = xIndex + (yIndex)*height;

  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
  }

  cg::sync(cta);

  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
  }

  // __threadfence_system();
}

// --------------------------------------------------------------------
// Partial transposes
// NB: the coarse- and fine-grained routines only perform part of a
//     transpose and will fail the test against the reference solution
//
//     They are used to assess performance characteristics of different
//     components of a full transpose
// --------------------------------------------------------------------

__global__ void transposeFineGrained(float *odata, float *idata, int width,
                                     int height) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  __shared__ float block[TILE_DIM][TILE_DIM + 1];

  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
  int index = xIndex + (yIndex)*width;

  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    block[threadIdx.y + i][threadIdx.x] = idata[index + i * width];
  }

  cg::sync(cta);

  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    odata[index + i * height] = block[threadIdx.x][threadIdx.y + i];
  }

  // __threadfence_system();
}

__global__ void transposeCoarseGrained(float *odata, float *idata, int width,
                                       int height) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  __shared__ float block[TILE_DIM][TILE_DIM + 1];

  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
  int index_in = xIndex + (yIndex)*width;

  xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
  yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
  int index_out = xIndex + (yIndex)*height;

  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    block[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
  }

  cg::sync(cta);

  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    odata[index_out + i * height] = block[threadIdx.y + i][threadIdx.x];
  }

  // __threadfence_system();
}

// ---------------------
// host utility routines
// ---------------------

void computeTransposeGold(float *gold, float *idata, const int size_x,
                          const int size_y) {
  for (int y = 0; y < size_y; ++y) {
    for (int x = 0; x < size_x; ++x) {
      gold[(x * size_y) + y] = idata[(y * size_x) + x];
    }
  }
}

void getParams(int argc, char **argv, cudaDeviceProp &deviceProp, int &repeat,
               int &size_x, int &size_y, char **variant, int max_tile_dim) {
  // set matrix size (if (x,y) dim of matrix is not square, then this will have
  // to be modified
  if (checkCmdLineFlag(argc, (const char **)argv, "dimX")) {
    size_x = getCmdLineArgumentInt(argc, (const char **)argv, "dimX");

    if (size_x > max_tile_dim) {
      printf("> MatrixSize X = %d is greater than the recommended size = %d\n",
             size_x, max_tile_dim);
    } else {
      printf("> MatrixSize X = %d\n", size_x);
    }
  } else {
    size_x = max_tile_dim;
    size_x = FLOOR(size_x, 512);
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "dimY")) {
    size_y = getCmdLineArgumentInt(argc, (const char **)argv, "dimY");

    if (size_y > max_tile_dim) {
      printf("> MatrixSize Y = %d is greater than the recommended size = %d\n",
             size_y, max_tile_dim);
    } else {
      printf("> MatrixSize Y = %d\n", size_y);
    }
  } else {
    size_y = max_tile_dim;
    size_y = FLOOR(size_y, 512);
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "repeat")) {
    repeat = getCmdLineArgumentInt(argc, (const char **)argv, "repeat");
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "variant")) {
    getCmdLineArgumentString(argc, (const char **)argv, "variant", variant);
  }
}

void showHelp() {
  printf("\n%s : Command line options\n", sSDKsample);
  printf("\t-device=n          (where n=0,1,2.... for the GPU device)\n\n");
  printf("> The default matrix size can be overridden with these parameters\n");
  printf("\t-dimX=row_dim_size (matrix row    dimensions)\n");
  printf("\t-dimY=col_dim_size (matrix column dimensions)\n");
}

void run_variant( // void (*kernel)(float *, float *, int, int),
                  // const char *kernelName,
    int variant, int reps, float *h_idata, float *h_odata, float *d_idata,
    float *d_odata, float *transposeGold, float *gold, dim3 grid, dim3 threads,
    int size_x, int size_y, size_t mem_size, bool &success) {
  // kernel pointer and descriptor
  void (*kernel)(float *, float *, int, int);
  const char *kernelName;

  // set kernel pointer
  switch (variant) {
  case 0:
    kernel = &copy;
    kernelName = "simple copy       ";
    break;

  case 1:
    kernel = &copySharedMem;
    kernelName = "shared memory copy";
    break;

  case 2:
    kernel = &transposeNaive;
    kernelName = "naive             ";
    break;

  case 3:
    kernel = &transposeCoalesced;
    kernelName = "coalesced         ";
    break;

  case 4:
    kernel = &transposeNoBankConflicts;
    kernelName = "optimized         ";
    break;

  case 5:
    kernel = &transposeCoarseGrained;
    kernelName = "coarse-grained    ";
    break;

  case 6:
    kernel = &transposeFineGrained;
    kernelName = "fine-grained      ";
    break;

  case 7:
    kernel = &transposeDiagonal;
    kernelName = "diagonal          ";
    break;
  }
  // set reference solution
  if (kernel == &copy || kernel == &copySharedMem) {
    gold = h_idata;
  } else if (kernel == &transposeCoarseGrained ||
             kernel == &transposeFineGrained) {
    gold = h_odata; // fine- and coarse-grained kernels are not full
                    // transposes, so bypass check
  } else {
    gold = transposeGold;
  }

  // Clear error status
  checkCudaErrors(cudaGetLastError());

  // warmup to avoid timing startup
  kernel<<<grid, threads>>>(d_odata, d_idata, size_x, size_y);
  checkCudaErrors(cudaGetLastError());

  // CUDA events
  cudaEvent_t start, stop;
  // initialize events
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  // take measurements for loop over kernel launches
  checkCudaErrors(cudaEventRecord(start, 0));

  for (int i = 0; i < reps; i++) {
    kernel<<<grid, threads>>>(d_odata, d_idata, size_x, size_y);
    // Ensure no launch failure
    checkCudaErrors(cudaGetLastError());
  }

  checkCudaErrors(cudaEventRecord(stop, 0));
  checkCudaErrors(cudaEventSynchronize(stop));
  float kernelTime;
  checkCudaErrors(cudaEventElapsedTime(&kernelTime, start, stop));

  checkCudaErrors(
      cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost));
  bool res = compareData(gold, h_odata, size_x * size_y, 0.01f, 0.0f);

  if (res == false) {
    printf("*** %s kernel FAILED ***\n", kernelName);
    success = false;
  }

  // take measurements for loop inside kernel
  checkCudaErrors(
      cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost));
  res = compareData(gold, h_odata, size_x * size_y, 0.01f, 0.0f);

  if (res == false) {
    printf("*** %s kernel FAILED ***\n", kernelName);
    success = false;
  }

  // report effective bandwidths
  float kernelBandwidth =
      2.0f * 1000.0f * mem_size / (1024 * 1024 * 1024) / (kernelTime / reps);
  printf("transpose %s, Throughput = %.4f GB/s, Time = %.5f ms, Size = %u fp32 "
         "elements, NumDevsUsed = %u, Workgroup = %u\n",
         kernelName, kernelBandwidth, kernelTime / reps, (size_x * size_y), 1,
         TILE_DIM * BLOCK_ROWS);

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
}
// ----
// main
// ----

int main(int argc, char **argv) {
  // Start logs
  printf("%s Starting...\n\n", sSDKsample);

  if (checkCmdLineFlag(argc, (const char **)argv, "help")) {
    showHelp();
    return 0;
  }

  int devID = findCudaDevice(argc, (const char **)argv);
  cudaDeviceProp deviceProp;

  // get number of SMs on this GPU
  checkCudaErrors(cudaGetDevice(&devID));
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

  // compute the scaling factor (for GPUs with fewer MPs)
  float scale_factor, total_tiles;
  scale_factor = std::max(
      (192.0f / (_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
                 (float)deviceProp.multiProcessorCount)),
      1.0f);

  printf("> Device %d: \"%s\"\n", devID, deviceProp.name);
  printf("> SM Capability %d.%d detected:\n", deviceProp.major,
         deviceProp.minor);

  // Calculate number of tiles we will run for the Matrix Transpose performance
  // tests
  int size_x, size_y, max_matrix_dim, matrix_size_test;

  matrix_size_test = 512; // we round down max_matrix_dim for this perf test
  total_tiles = (float)MAX_TILES / scale_factor;

  max_matrix_dim =
      FLOOR((int)(floor(sqrt(total_tiles)) * TILE_DIM), matrix_size_test);

  // This is the minimum size allowed
  if (max_matrix_dim == 0) {
    max_matrix_dim = matrix_size_test;
  }

  printf("> [%s] has %d MP(s) x %d (Cores/MP) = %d (Cores)\n", deviceProp.name,
         deviceProp.multiProcessorCount,
         _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
         _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
             deviceProp.multiProcessorCount);

  printf("> Compute performance scaling factor = %4.2f\n", scale_factor);

  // Extract parameters if there are any, command line -dimx and -dimy can
  // override any of these settings
  int reps = 100;
  char *variant_arg = NULL;
  getParams(argc, argv, deviceProp, reps, size_x, size_y, &variant_arg,
            max_matrix_dim);

  if (size_x != size_y) {
    printf("\n[%s] does not support non-square matrices (row_dim_size(%d) != "
           "col_dim_size(%d))\nExiting...\n\n",
           sSDKsample, size_x, size_y);
    exit(EXIT_FAILURE);
  }

  if (size_x % TILE_DIM != 0 || size_y % TILE_DIM != 0) {
    printf("[%s] Matrix size must be integral multiple of tile "
           "size\nExiting...\n\n",
           sSDKsample);
    exit(EXIT_FAILURE);
  }

  // execution configuration parameters
  dim3 grid(size_x / TILE_DIM, size_y / TILE_DIM),
      threads(TILE_DIM, BLOCK_ROWS);

  if (grid.x < 1 || grid.y < 1) {
    printf("[%s] grid size computation incorrect in test \nExiting...\n\n",
           sSDKsample);
    exit(EXIT_FAILURE);
  }

  // size of memory required to store the matrix
  size_t mem_size = static_cast<size_t>(sizeof(float) * size_x * size_y);

  if (2 * mem_size > deviceProp.totalGlobalMem) {
    printf("Input matrix size is larger than the available device memory!\n");
    printf("Please choose a smaller size matrix\n");
    exit(EXIT_FAILURE);
  }

  // allocate host memory
  float *h_idata = (float *)malloc(mem_size);
  float *h_odata = (float *)malloc(mem_size);
  float *transposeGold = (float *)malloc(mem_size);
  float *gold;

  // allocate device memory
  float *d_idata, *d_odata;
  checkCudaErrors(cudaMalloc((void **)&d_idata, mem_size));
  checkCudaErrors(cudaMalloc((void **)&d_odata, mem_size));

  // initialize host data
  for (int i = 0; i < (size_x * size_y); ++i) {
    h_idata[i] = (float)i;
  }

  // copy host data to device
  checkCudaErrors(
      cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));

  // Compute reference transpose solution
  computeTransposeGold(transposeGold, h_idata, size_x, size_y);

  // print out common data for all kernels
  printf("\nMatrix size: %dx%d (%dx%d tiles), tile size: %dx%d, block size: "
         "%dx%d\n\n",
         size_x, size_y, size_x / TILE_DIM, size_y / TILE_DIM, TILE_DIM,
         TILE_DIM, TILE_DIM, BLOCK_ROWS);

  //
  // loop over different kernels
  //

  bool success = true;

  const int NUM_VARIANTS = 8;
  const char *variants[NUM_VARIANTS] = {
      "simple copy", "shared memory copy", "naive",        "coalesced",
      "optimized",   "coarse-grained",     "fine-grained", "diagonal"};

  if (variant_arg == NULL) {
    for (int k = 0; k < NUM_VARIANTS; k++) {
      run_variant(k, reps, h_idata, h_odata, d_idata, d_odata, transposeGold,
                  gold, grid, threads, size_x, size_y, mem_size, success);
    }
  } else {
    int k = 0;
    int *found = NULL;
    for (; k < NUM_VARIANTS; k++) {
      if (!strcmp(variant_arg, variants[k])) {
        found = &k;
        break;
      }
    }

    if (found == NULL) {
      printf("unknown variant %s\n", variant_arg);
      success = false;
    } else {
      printf("running variant %s (%d)\n", variant_arg, *found);
      run_variant(*found, reps, h_idata, h_odata, d_idata, d_odata,
                  transposeGold, gold, grid, threads, size_x, size_y, mem_size,
                  success);
    }
  }

  // cleanup
  free(h_idata);
  free(h_odata);
  free(transposeGold);
  cudaFree(d_idata);
  cudaFree(d_odata);

  if (!success) {
    printf("Test failed!\n");
    exit(EXIT_FAILURE);
  }

  printf("Test passed\n");
  exit(EXIT_SUCCESS);
}
