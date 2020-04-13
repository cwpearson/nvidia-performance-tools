#include <algorithm>
#include <chrono>

#include <nvToolsExt.h>

#include <argparse/argparse.hpp>

#include "common.hpp"

#define TILE_WIDTH 32

/* NOTE: A and C are column major, B is row major
 */
__global__ void mygemm(float *__restrict__ c, //<! [out] and MxN matrix
                       const float *a,        //<! [in] an MxK matrix
                       const float *b,        //<! [in] an KxN matrix
                       const int M, const int N, const int K) {

  __shared__ float aSh[TILE_WIDTH][TILE_WIDTH];
  __shared__ float bSh[TILE_WIDTH][TILE_WIDTH];
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int i = by * TILE_WIDTH + ty;
  int j = bx * TILE_WIDTH + tx;
  float acc = 0;

#define A(_i, _j) a[(_i) + (_j)*M]
#define B(_i, _j) b[(_i)*N + (_j)]
#define C(_i, _j) c[(_i) + (_j)*M]

  for (int m = 0; m < (K - 1) / TILE_WIDTH + 1; ++m) {
    if (i < M && m * TILE_WIDTH + tx < K) {
      aSh[ty][tx] = A(i, m * TILE_WIDTH + tx);
    } else {
      aSh[ty][tx] = 0;
    }
    if (j < N && m * TILE_WIDTH + ty < K) {
      bSh[ty][tx] = B(m * TILE_WIDTH + ty, j);
    } else {
      bSh[ty][tx] = 0;
    }

    __syncthreads();
    for (int k = 0; k < TILE_WIDTH; ++k) {
      acc += aSh[ty][k] * bSh[k][tx];
    }
    __syncthreads();
  }
  if (i < M && j < N) {
    C(i, j) = acc;
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

  int nIters = 5;
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
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  dim3 dimGrid;
  dimGrid.x = (n + dimBlock.x - 1) / dimBlock.x;
  dimGrid.y = (m + dimBlock.y - 1) / dimBlock.y;

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
    float millis;
    CUDA_RUNTIME(cudaEventRecord(start));
    mygemm<<<dimGrid, dimBlock>>>(cDev, aDev, bDev, m, n, k);
    CUDA_RUNTIME(cudaEventRecord(stop));
    CUDA_RUNTIME(cudaEventSynchronize(stop));
    CUDA_RUNTIME(cudaEventElapsedTime(&millis, start, stop));

    // copy data back to host
    nvtxRangePush("device-to-host");
    CUDA_RUNTIME(
        cudaMemcpy(cHost, cDev, m * n * sizeof(float), cudaMemcpyDefault));
    nvtxRangePop();
    CUDA_RUNTIME(cudaDeviceSynchronize());

    Duration wallElapsed = Clock::now() - wallStart;

    std::cout << iter << " kernel=" << millis / 1000
              << " wall=" << wallElapsed.count()
              << (iter >= nWarmup ? " *" : "  ") << "\n";

    // track time if no longer during warmup
    if (iter >= nWarmup) {
      wallTime += wallElapsed.count();
      kernelTime += millis / 1000; // seconds
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
