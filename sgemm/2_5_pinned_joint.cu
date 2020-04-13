#include <algorithm>
#include <chrono>

#include <nvToolsExt.h>

#include <argparse/argparse.hpp>

#include "common.hpp"

#define TILE_SZ_A 64
#define TILE_SZ_B 16
#define TILE_SZ_RATIO (TILE_SZ_A / TILE_SZ_B)

/* NOTE: A and C are column major, B is row major
 */
__global__ void mygemm(float *__restrict__ c, //<! [out] and MxN matrix
                       const float *a,        //<! [in] an MxK matrix
                       const float *b,        //<! [in] an KxN matrix
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

/* Time the total transfer & matrix-multiplication time
 */
 int main(int argc, char **argv) {

  argparse::Parser parser;

  // default matrix sizes:
  // A: 1600 x 1500
  // B: 1500 x 1400
  // C: 1600 x 1400
  int m = 1600;
  int n = 1400;
  int k = 1500;

  int nIters = 10;
  int nWarmup = 5;
  parser.add_positional(m);
  parser.add_positional(n);
  parser.add_positional(k);
  parser.add_option(nIters, "--iters");
  parser.add_option(nWarmup, "--warmup");

  if (!parser.parse(argc, argv)) {
    parser.help();
    exit(EXIT_FAILURE);
  }

  const int64_t flop = int64_t(m) * int64_t(n) * int64_t(k) * 2 * nIters;

  // initialize host data
  std::cout << "generate data\n";
  nvtxRangePush("generate data");
  float *aHost, *bHost, *cHost;
  CUDA_RUNTIME(cudaHostAlloc(&aHost, m * k * sizeof(float), 0));
  CUDA_RUNTIME(cudaHostAlloc(&bHost, k * n * sizeof(float), 0));
  CUDA_RUNTIME(cudaHostAlloc(&cHost, m * n * sizeof(float), 0));
  std::generate(aHost, aHost + m * k, random_int);
  std::generate(bHost, bHost + k * n, random_int);
  nvtxRangePop();

  // allocate device data
  float *aDev, *bDev, *cDev;
  CUDA_RUNTIME(cudaMalloc(&aDev, m * k * sizeof(float)));
  CUDA_RUNTIME(cudaMalloc(&bDev, k * n * sizeof(float)));
  CUDA_RUNTIME(cudaMalloc(&cDev, m * n * sizeof(float)));

  // create events to time GPU kernel
  cudaEvent_t start, stop;
  CUDA_RUNTIME(cudaEventCreate(&start));
  CUDA_RUNTIME(cudaEventCreate(&stop));

  // GPU kernel launch parameters
  dim3 dimGrid((m + TILE_SZ_A - 1) / TILE_SZ_A, (n +TILE_SZ_B - 1) / TILE_SZ_B);
  dim3 dimBlock(TILE_SZ_A, 1);

  float kernelTime = 0;
  float wallTime = 0;

  for (int iter = 0; iter < nWarmup + nIters; ++iter) {

    auto wallStart = Clock::now();

    // copy data to device
    nvtxRangePush("host-to-device");
    CUDA_RUNTIME(
        cudaMemcpy(aDev, aHost, m * k * sizeof(float), cudaMemcpyDefault));
    CUDA_RUNTIME(
        cudaMemcpy(bDev, bHost, k * n * sizeof(float), cudaMemcpyDefault));
    nvtxRangePop();

    // kernel time
    float kernelElapsed;
    CUDA_RUNTIME(cudaEventRecord(start));
    mygemm<<<dimGrid, dimBlock>>>(cDev, aDev, bDev, m, n, k);
    CUDA_RUNTIME(cudaEventRecord(stop));
    CUDA_RUNTIME(cudaEventSynchronize(stop));
    CUDA_RUNTIME(cudaEventElapsedTime(&kernelElapsed, start, stop));
    kernelElapsed /= 1000; // seconds

    // copy data back to host
    nvtxRangePush("device-to-host");
    CUDA_RUNTIME(
        cudaMemcpy(cHost, cDev, m * n * sizeof(float), cudaMemcpyDefault));
    nvtxRangePop();
    CUDA_RUNTIME(cudaDeviceSynchronize());

    Duration wallElapsed = Clock::now() - wallStart;

    std::cout << iter << " kernel=" << kernelElapsed
              << " wall=" << wallElapsed.count()
              << (iter >= nWarmup ? " *" : "  ") << "\n";

    // track time if no longer during warmup
    if (iter >= nWarmup) {
      wallTime += wallElapsed.count();
      kernelTime += kernelElapsed; // seconds
    }
  }

  // print results
  double kernelGflops = flop / 1e9 / kernelTime;
  std::cout << "kernel " << kernelGflops << "GFLOPS (" << flop << " flop, "
            << kernelTime << "s)\n";
  double wallGflops = flop / 1e9 / wallTime;
  std::cout << "wall " << wallGflops << "GFLOPS (" << flop << " flop, "
            << wallTime << "s)\n";
  // release resources
  CUDA_RUNTIME(cudaEventDestroy(start));
  CUDA_RUNTIME(cudaEventDestroy(stop));
  CUDA_RUNTIME(cudaFree(aDev));
  CUDA_RUNTIME(cudaFree(bDev));
  CUDA_RUNTIME(cudaFree(cDev));
  CUDA_RUNTIME(cudaFreeHost(aHost));
  CUDA_RUNTIME(cudaFreeHost(bHost));
  CUDA_RUNTIME(cudaFreeHost(cHost));
  return 0;
}