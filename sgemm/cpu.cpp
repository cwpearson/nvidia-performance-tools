#include <iostream>
#include <vector>
#include <algorithm>

#include "common.hpp"

int main(int argc, char **argv) {

  int M = 2;
  int N = 2;
  int K = 2;

  // initialize host data
  std::vector<float> a(M * K), b(K * N), c(M * N);
  std::generate(a.begin(), a.end(), random_int);
  std::generate(b.begin(), b.end(), random_int);

  cpu_gemm(c.data(), a.data(), b.data(), M, N, K);

#define A(_i, _j) a[(_i) + (_j)*M]
#define B(_i, _j) b[(_i)*N + (_j)]
#define C(_i, _j) c[(_i) + (_j)*M]

  float acc = 0;
  for (int k = 0; k < K; ++k) {
    acc += A(0, k) * B(k, 0);
  }
  
  if (equal(C(0, 0), acc, 1e-6)) {
    return 0;
  } else {
    return 1;
  }

#undef A
#undef B
#undef C
}