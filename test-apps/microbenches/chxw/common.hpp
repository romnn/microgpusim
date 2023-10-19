#include <algorithm>
#include <assert.h>
#include <cstdlib>
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>

const int KB = 1024;

#define ROUND_UP_TO_MULTIPLE(value, multipleof)                                \
  ((unsigned int)std::ceil((float)(value) / (float)(multipleof)) * value)

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

enum memory { L1Data, L1ReadOnly, L1Texture, L2, NUM_MEMORIES };
static const char *memory_str[NUM_MEMORIES] = {
    "l1data",
    "l1readonly",
    "l1texture",
    "l2",
};

unsigned int measure_clock_overhead();
