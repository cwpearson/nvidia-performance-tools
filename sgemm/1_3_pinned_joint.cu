#include <algorithm>

#include <nvToolsExt.h>

#include <argparse/argparse.hpp>

#include "common.hpp"

#define TILE_SZ_A 64
#define TILE_SZ_B 16
#define TILE_SZ_RATIO (TILE_SZ_A / TILE_SZ_B)

/* NOTE: A and C are column major, B is row major
 */
__global__ void mygemm(float * __restrict__ c,       //<! [out] and MxN matrix
                       const float *a, //<! [in] an MxK matrix
                       const float *b, //<! [in] an KxN matrix
                       const int M, const int N, const int K) {

// Macros for accessing flattened matrices
#define A(_i, _j) a[(_i) + (_j)*M]
#define B(_i, _j) b[(_i)*N + (_j)]
#define C(_i, _j) c[(_i) + (_j)*M]

  // Shared memory for tiling input B array
  __shared__ float B_s[TILE_SZ_RATIO][TILE_SZ_B];

  // Index variables
  const unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int col = blockIdx.y * TILE_SZ_B;

  // Privatization of output variables
  float c_reg[TILE_SZ_B];

  // Initialize output values
  for (unsigned int outIdx = 0; outIdx < TILE_SZ_B; ++outIdx) {
    c_reg[outIdx] = 0;
  }

  // Loop over the input tiles
  for (unsigned int tileIdx = 0; tileIdx < (K - 1) / TILE_SZ_RATIO + 1;
       ++tileIdx) {
    // Load the tile of B into shared memory
    const unsigned int i = threadIdx.x / TILE_SZ_B;
    const unsigned int j = threadIdx.x % TILE_SZ_B;
    if (tileIdx * TILE_SZ_RATIO + i < K && col + j < N) {
      B_s[i][j] = B(tileIdx * TILE_SZ_RATIO + i, col + j);
    } else {
      B_s[i][j] = 0;
    }
    __syncthreads();
    // Loop over elements inside the tile
    for (unsigned int idx = 0; idx < TILE_SZ_RATIO; ++idx) {
      // Load tile of A matrix into register
      float a_reg;
      if (row < M && tileIdx * TILE_SZ_RATIO + idx < K) {
        a_reg = A(row, tileIdx * TILE_SZ_RATIO + idx);
      } else {
        a_reg = 0;
      }
      // Loop over and update the output elements assigned to the thread
      for (unsigned int outIdx = 0; outIdx < TILE_SZ_B; ++outIdx) {
        c_reg[outIdx] += a_reg * B_s[idx][outIdx];
      }
    }
    __syncthreads();
  }

  for (unsigned int outIdx = 0; outIdx < TILE_SZ_B; ++outIdx) {
    if (row < M && col + outIdx < N) {
      C(row, col + outIdx) = c_reg[outIdx];
    }
  }

#undef A
#undef B
#undef C
}

int main(int argc, char **argv) {

  argparse::Parser parser;

  // default matrix sizes:
  // A: 1489 x 1493
  // B: 1493 x 1499
  // C: 1489 x 1499
  int m = 1489;
  int n = 1499;
  int k = 1493;

  int nIters = 5;
  int nWarmup = 5;
  bool check = false;
  parser.add_positional(m);
  parser.add_positional(n);
  parser.add_positional(k);
  parser.add_option(nIters, "--iters");
  parser.add_option(nWarmup, "--warmup");
  parser.add_flag(check, "--check");

  if (!parser.parse(argc, argv)) {
    parser.help();
    exit(EXIT_FAILURE);
  }

  const int64_t flop = int64_t(m) * int64_t(n) * int64_t(k) * 2;

  // initialize host data
  std::cerr << "generate data\n";
  nvtxRangePush("generate data");
  float *aHost, *bHost, *cHost, *cExpected;
  CUDA_RUNTIME(cudaHostAlloc(&aHost, m * k * sizeof(float), 0));
  CUDA_RUNTIME(cudaHostAlloc(&bHost, k * n * sizeof(float), 0));
  CUDA_RUNTIME(cudaHostAlloc(&cHost, m * n * sizeof(float), 0));
  CUDA_RUNTIME(cudaHostAlloc(&cExpected, m * n * sizeof(float), 0));
  std::generate(aHost, aHost + m * k, random_int);
  std::generate(bHost, bHost + k * n, random_int);
  nvtxRangePop();

  // allocate device data
  float *aDev, *bDev, *cDev;
  CUDA_RUNTIME(cudaMalloc(&aDev, m * k * sizeof(float)));
  CUDA_RUNTIME(cudaMalloc(&bDev, k * n * sizeof(float)));
  CUDA_RUNTIME(cudaMalloc(&cDev, m * n * sizeof(float)));

  // copy data to device
  std::cerr << "transfer to GPU\n";
  nvtxRangePush("host-to-device");
  CUDA_RUNTIME(
      cudaMemcpy(aDev, aHost, m * k * sizeof(float), cudaMemcpyDefault));
  CUDA_RUNTIME(
      cudaMemcpy(bDev, bHost, k * n * sizeof(float), cudaMemcpyDefault));
  nvtxRangePop();

  // create events to time GPU kernel
  cudaEvent_t start, stop;
  CUDA_RUNTIME(cudaEventCreate(&start));
  CUDA_RUNTIME(cudaEventCreate(&stop));

  // GPU kernel launch parameters
  dim3 dimGrid((m + TILE_SZ_A - 1) / TILE_SZ_A, (n +TILE_SZ_B - 1) / TILE_SZ_B);
  dim3 dimBlock(TILE_SZ_A, 1);

  // total elapsed time
  float elapsed = 0;

  /* Launch the kernel nIters + nWarmup times
     Check for correctness on the first time.
     Record the time after nWarmup runs complete.
  */
  for (int i = 0; i < nIters + nWarmup; ++i) {
    CUDA_RUNTIME(cudaEventRecord(start));
    mygemm<<<dimGrid, dimBlock>>>(cDev, aDev, bDev, m, n, k);
    CUDA_RUNTIME(cudaEventRecord(stop));
    CUDA_RUNTIME(cudaEventSynchronize(stop));

    // check result once
    if (check && 0 == i) {
      // copy result to host
      CUDA_RUNTIME(
          cudaMemcpy(cHost, cDev, m * n * sizeof(float), cudaMemcpyDefault));

      // check result on host
      cpu_gemm(cExpected, aHost, bHost, m, n, k);

      for (size_t i = 0; i < m * n; ++i) {
        if (!equal(cExpected[i], cHost[i], 1e-6)) {
          std::cerr << "Error!\n";
          exit(EXIT_FAILURE);
        }
      }
    }

    float millis;
    CUDA_RUNTIME(cudaEventElapsedTime(&millis, start, stop));
    std::cerr << i << ": " << millis << (i >= nWarmup ? " *" : " ") << "\n";

    // record time after warmup runs
    if (i >= nWarmup) {
      elapsed += millis;
    }
  }

  // print results
  double gflops = flop / ((elapsed / nIters) / 1000) / 1e9;
  std::cerr << "kernel " << gflops << "GFLOPS (" << flop << " flop, "
            << (elapsed / nIters) / 1000 << "s)\n";

  // release resources
  CUDA_RUNTIME(cudaEventDestroy(start));
  CUDA_RUNTIME(cudaEventDestroy(stop));
  CUDA_RUNTIME(cudaFree(aDev));
  CUDA_RUNTIME(cudaFree(bDev));
  CUDA_RUNTIME(cudaFree(cDev));
  CUDA_RUNTIME(cudaFreeHost(aHost));
  CUDA_RUNTIME(cudaFreeHost(bHost));
  CUDA_RUNTIME(cudaFreeHost(cHost));
  CUDA_RUNTIME(cudaFreeHost(cExpected));
  return 0;
}
