#pragma once

#include <chrono>

#ifdef __CUDACC__
inline void checkCuda(cudaError_t result, const char *file, const int line) {
  if (result != cudaSuccess) {
    fprintf(stderr, "%s@%d: CUDA Runtime Error(%d): %s\n", file, line,
            int(result), cudaGetErrorString(result));
    exit(-1);
  }
}

#define CUDA_RUNTIME(stmt) checkCuda(stmt, __FILE__, __LINE__);
#endif

/* NOTE: A and C are column major, B is row major
 */
inline void cpu_gemm(float *c,       //<! [out] and MxN matrix
                     const float *a, //<! [in] an MxK matrix
                     const float *b, //<! [in] an KxN matrix
                     const int M, const int N, const int K) {

#define A(_i, _j) a[(_i) + (_j)*M]
#define B(_i, _j) b[(_i)*N + (_j)]
#define C(_i, _j) c[(_i) + (_j)*M]

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float acc = 0;
      for (int k = 0; k < K; ++k) {
        acc += A(i, k) * B(k, j);
      }
      C(i, j) = acc;
    }
  }

#undef A
#undef B
#undef C
}

inline bool equal(float x, float y, float eps) {
  return std::abs(x - y) <=
         eps * std::max(std::max(1.0f, std::abs(x)), std::abs(y));
}

inline int random_int() { return (std::rand() % 100); }

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<float> Duration;