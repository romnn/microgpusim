#include <algorithm>
#include <assert.h>
#include <cstdlib>
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>

const int KB = 1024;

#define ROUND_UP_TO_MULTIPLE(value, multipleof)                                \
  ((unsigned int)std::ceil((float)(value) / (float)(multipleof)) * value)

#define CUDA_CHECK(call)                                                       \
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

template <typename T> int indexof(std::vector<T> const &v, T needle) {
  typename std::vector<T>::const_iterator it = find(v.begin(), v.end(), needle);
  if (it != v.end()) {
    return it - v.begin();
  } else {
    return -1;
  }
}

template <typename T> int indexof(T const *arr, size_t len, T needle) {
  for (size_t i = 0; i < len; i++) {
    if (arr[i] == needle) {
      return i;
    }
  }
  return -1;
}
