#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#define BLOCK_DIM 32

#define checkCuda(expr) check((expr), #expr, __FILE__, __LINE__)

void check(cudaError_t err, const char *const func, const char *const file,
           const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

template <typename T> std::vector<T> create_rand_vector(size_t n) {
  std::random_device r;
  std::default_random_engine engine(r());
  std::uniform_int_distribution<int> uniform_dist(-256, 256);
  // std::uniform_int_distribution<int> uniform_dist(1, 5);

  std::vector<T> vec(n);
  for (size_t i{0}; i < n; ++i) {
    vec.at(i) = static_cast<T>(uniform_dist(engine));
  }

  return vec;
}

template <typename T> void print_matrix(T const *mat, size_t m, size_t n) {
  for (size_t mi = 0; mi < m; mi++) {
    for (size_t ni = 0; ni < n; ni++) {
      std::cout << mat[mi * m + ni] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

template <typename T>
void mult(T const *A, T const *B, T *C, size_t m, size_t n, size_t p) {
  size_t mi, ni, pi;

  for (mi = 0; mi < m; mi++) {
    for (pi = 0; pi < p; pi++) {
      T sum = 0.0;
      for (ni = 0; ni < n; ni++) {
        // sum += A[mi][ni] * B[ni][pi]
        sum += A[mi * n + ni] * B[ni * p + pi];
      }

      C[mi * p + pi] = sum;
    }
  }
}

template <typename T>
__global__ void mult_gpu(T const *mat_a, T const *mat_b, T *mat_c, size_t m,
                         size_t n, size_t p) {
  // 2D block and 2D thread
  // Each thread computes one cell in mat_3.
  // the grid + thradidx
  size_t i = blockIdx.y * blockDim.y + threadIdx.y;
  size_t j = blockIdx.x * blockDim.x + threadIdx.x;

  // printf("thread idx = (%u, %u, %u)\n", threadIdx.x, threadIdx.y,
  // threadIdx.z);

  // do not process outside the matrix.
  // do not forget the equal sign!
  if ((i >= m) || (j >= p)) {
    return;
  }

  float acc_sum{0};
  for (size_t k = 0; k < n; k++) {
    acc_sum += mat_a[i * n + k] * mat_b[k * p + j];
  }
  mat_c[i * p + j] = acc_sum;
}

template <typename T> int matrixmul(size_t m, size_t n, size_t p) {
  std::vector<T> const a_vec{create_rand_vector<T>(m * n)};
  std::vector<T> const b_vec{create_rand_vector<T>(n * p)};
  std::vector<T> c_gpu_vec(m * p);
  std::vector<T> c_cpu_vec(m * p);

  T const *a{a_vec.data()};
  T const *b{b_vec.data()};
  T *c_gpu{c_gpu_vec.data()};
  T *c_cpu{c_cpu_vec.data()};

  printf("(%lu x %lu) x (%lu x %lu) = (%lu x %lu)\n\n", m, n, n, p, m, p);
  // std::cout << "A:" << std::endl;
  // print_matrix<T>(a, m, n);
  //
  // std::cout << "B:" << std::endl;
  // print_matrix<T>(b, n, p);

  T *dev_a, *dev_b, *dev_c;

  // allocate device buffer
  checkCuda(cudaMalloc(&dev_a, sizeof(T) * a_vec.size()));
  checkCuda(cudaMalloc(&dev_b, sizeof(T) * b_vec.size()));
  checkCuda(cudaMalloc(&dev_c, sizeof(T) * c_gpu_vec.size()));

  // copy data from host to device
  checkCuda(
      cudaMemcpy(dev_a, a, sizeof(T) * a_vec.size(), cudaMemcpyHostToDevice));
  checkCuda(
      cudaMemcpy(dev_b, b, sizeof(T) * b_vec.size(), cudaMemcpyHostToDevice));

  // run matrix multiplication on GPU
  dim3 block_size(BLOCK_DIM, BLOCK_DIM);
  printf("block=(%d, %d, %d)\n", block_size.x, block_size.y, block_size.z);

  size_t grid_x =
      std::ceil(static_cast<double>(p) / static_cast<double>(block_size.x));
  size_t grid_y =
      std::ceil(static_cast<double>(m) / static_cast<double>(block_size.y));
  dim3 grid(grid_x, grid_y);
  printf("grid=(%d, %d, %d)\n", grid.x, grid.y, grid.z);

  mult_gpu<T><<<grid, block_size>>>(dev_a, dev_b, dev_c, m, n, p);

  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA Matrix Multiplication kernel failed to execute."
              << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // copy data from device to host
  checkCuda(cudaMemcpy(c_gpu, dev_c, sizeof(T) * c_gpu_vec.size(),
                       cudaMemcpyDeviceToHost));

  // free device buffers
  checkCuda(cudaFree(dev_a));
  checkCuda(cudaFree(dev_b));
  checkCuda(cudaFree(dev_c));

  mult<T>(a, b, c_cpu, m, n, p);

  // std::cout << "C: (GPU)" << std::endl;
  // print_matrix<T>(c_gpu, m, p);
  //
  // std::cout << "C: (CPU)" << std::endl;
  // print_matrix<T>(c_cpu, m, p);

  // verification
  printf("verifying... \n");
  bool correct = true;
  for (int mi = 0; mi < m; mi++) {
    for (int pi = 0; pi < p; pi++) {
      T have = c_gpu[mi * p + pi];
      T want = c_cpu[mi * p + pi];
      if (abs(have - want) > 1e-2) {
        printf("Error: have %f but want %f at (%d, %d)\n", have, want, mi, pi);
        correct = false;
        break;
      }
    }
  }

  if (correct) {
    printf("PASS\n");
    return 0;
  }
  return 1;
}

int main(int argc, char *argv[]) {
  if (argc < 5) {
    fprintf(stderr, "usage: matrixmul <m> <n> <p> <datatype>\n");
    return 1;
  }
  size_t m = (size_t)atoi(argv[1]);
  size_t n = (size_t)atoi(argv[2]);
  size_t p = (size_t)atoi(argv[3]);

  bool use_double = (atoi(argv[4]) == 64);
  if (use_double) {
    return matrixmul<double>(m, n, p);
  } else {
    return matrixmul<float>(m, n, p);
  }
}
