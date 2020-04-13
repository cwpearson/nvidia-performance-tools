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

  // 4 muls of m/2, n/2, k
  const int64_t flop = int64_t(m) / 2 * int64_t(n) / 2 * int64_t(k) * 2 * 4 * nIters;

  // initialize host data
  std::cerr << "generate data\n";
  nvtxRangePush("generate data");
  float *aHost[2], *bHost[2], *cHost[2][2];
  CUDA_RUNTIME(cudaHostAlloc(&aHost[0], m / 2 * k * sizeof(float), 0));
  CUDA_RUNTIME(cudaHostAlloc(&aHost[1], m / 2 * k * sizeof(float), 0));
  CUDA_RUNTIME(cudaHostAlloc(&bHost[0], k * n / 2 * sizeof(float), 0));
  CUDA_RUNTIME(cudaHostAlloc(&bHost[1], k * n / 2 * sizeof(float), 0));
  CUDA_RUNTIME(cudaHostAlloc(&cHost[0][0], m / 2 * n / 2 * sizeof(float), 0));
  CUDA_RUNTIME(cudaHostAlloc(&cHost[0][1], m / 2 * n / 2 * sizeof(float), 0));
  CUDA_RUNTIME(cudaHostAlloc(&cHost[1][0], m / 2 * n / 2 * sizeof(float), 0));
  CUDA_RUNTIME(cudaHostAlloc(&cHost[1][1], m / 2 * n / 2 * sizeof(float), 0));
  std::generate(aHost[0], aHost[0] + m / 2 * k, random_int);
  std::generate(aHost[1], aHost[1] + m / 2 * k, random_int);
  std::generate(bHost[0], bHost[0] + k * n / 2, random_int);
  std::generate(bHost[1], bHost[1] + k * n / 2, random_int);
  nvtxRangePop();

  // allocate device data
  std::cerr << "allocate data\n";
  float *aDev[2], *bDev[2], *cDev[2][2];
  CUDA_RUNTIME(cudaMalloc(&aDev[0], m / 2 * k * sizeof(float)));
  CUDA_RUNTIME(cudaMalloc(&aDev[1], m / 2 * k * sizeof(float)));
  CUDA_RUNTIME(cudaMalloc(&bDev[0], k * n / 2 * sizeof(float)));
  CUDA_RUNTIME(cudaMalloc(&bDev[1], k * n / 2 * sizeof(float)));
  CUDA_RUNTIME(cudaMalloc(&cDev[0][0], m / 2 * n / 2 * sizeof(float)));
  CUDA_RUNTIME(cudaMalloc(&cDev[0][1], m / 2 * n / 2 * sizeof(float)));
  CUDA_RUNTIME(cudaMalloc(&cDev[1][0], m / 2 * n / 2 * sizeof(float)));
  CUDA_RUNTIME(cudaMalloc(&cDev[1][1], m / 2 * n / 2 * sizeof(float)));

  // create streams for copy and kernels
  cudaStream_t copyStream, kernelStream;
  CUDA_RUNTIME(cudaStreamCreate(&copyStream));
  CUDA_RUNTIME(cudaStreamCreate(&kernelStream));

  cudaEvent_t waitForA0B0, waitForA1, waitForB1, waitC[2][2];
  CUDA_RUNTIME(cudaEventCreate(&waitForA0B0));
  CUDA_RUNTIME(cudaEventCreate(&waitForA1));
  CUDA_RUNTIME(cudaEventCreate(&waitForB1));
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      CUDA_RUNTIME(cudaEventCreate(&waitC[i][j]));
    }
  }

  // GPU kernel launch parameters
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  dim3 dimGrid;
  dimGrid.x = (n/2 + dimBlock.x - 1) / dimBlock.x;
  dimGrid.y = (m/2 + dimBlock.y - 1) / dimBlock.y;

  float kernelTime = 0;
  float wallTime = 0;
  for (int iter = 0; iter < nIters + nWarmup; ++iter) {

    nvtxRangePush("wall time");
    auto wallStart = Clock::now();

    // copy a0 and b0
    CUDA_RUNTIME(cudaMemcpyAsync(aDev[0], aHost[0], m / 2 * k * sizeof(float),
                                 cudaMemcpyDefault, copyStream));
    CUDA_RUNTIME(cudaMemcpyAsync(bDev[0], bHost[0], k * n / 2 * sizeof(float),
                                 cudaMemcpyDefault, copyStream));
    CUDA_RUNTIME(cudaEventRecord(waitForA0B0, copyStream));

    // have the kernelStream wait for the transfers to complete
    CUDA_RUNTIME(cudaStreamWaitEvent(kernelStream, waitForA0B0, 0));

    // launch c[0][0] = a[0] * b[0]
    mygemm<<<dimGrid, dimBlock, 0, kernelStream>>>(cDev[0][0], aDev[0], bDev[0],
                                                   m / 2, n / 2, k);
    CUDA_RUNTIME(cudaEventRecord(waitC[0][0], kernelStream));

    // copy a1
    CUDA_RUNTIME(cudaMemcpyAsync(aDev[1], aHost[1], m / 2 * k * sizeof(float),
                                 cudaMemcpyDefault, copyStream));
    CUDA_RUNTIME(cudaEventRecord(waitForA1, kernelStream));

    // launch c[1][0] = a[1] * b[0] after a[1] is on the GPU
    CUDA_RUNTIME(cudaStreamWaitEvent(kernelStream, waitForA1, 0));
    mygemm<<<dimGrid, dimBlock, 0, kernelStream>>>(cDev[1][0], aDev[1], bDev[0],
                                                   m / 2, n / 2, k);
    CUDA_RUNTIME(cudaEventRecord(waitC[1][0], kernelStream));

    // copy b1
    CUDA_RUNTIME(cudaMemcpyAsync(bDev[1], bHost[1], k * n / 2 * sizeof(float),
                                 cudaMemcpyDefault, copyStream));
    CUDA_RUNTIME(cudaEventRecord(waitForB1, kernelStream));

    // launch c[0][1] = a[0] * b[1] after B1 is on the GPU
    CUDA_RUNTIME(cudaStreamWaitEvent(kernelStream, waitForB1, 0));
    mygemm<<<dimGrid, dimBlock, 0, kernelStream>>>(cDev[0][1], aDev[0], bDev[1],
                                                   m / 2, n / 2, k);
    CUDA_RUNTIME(cudaEventRecord(waitC[0][1], kernelStream));

    // launch c[1][1] = a[1] * b[1]
    mygemm<<<dimGrid, dimBlock, 0, kernelStream>>>(cDev[1][1], aDev[1], bDev[1],
                                                   m / 2, n / 2, k);
    CUDA_RUNTIME(cudaEventRecord(waitC[1][1], kernelStream));

    // copy c back to CPU as kernels finish
    CUDA_RUNTIME(cudaStreamWaitEvent(copyStream, waitC[0][0], 0));
    CUDA_RUNTIME(cudaMemcpyAsync(cHost[0][0], cDev[0][0],
                                 m / 2 * n / 2 * sizeof(float),
                                 cudaMemcpyDefault, copyStream));
    CUDA_RUNTIME(cudaStreamWaitEvent(copyStream, waitC[1][0], 0));
    CUDA_RUNTIME(cudaMemcpyAsync(cHost[1][0], cDev[1][0],
                                 m / 2 * n / 2 * sizeof(float),
                                 cudaMemcpyDefault, copyStream));
    CUDA_RUNTIME(cudaStreamWaitEvent(copyStream, waitC[0][1], 0));
    CUDA_RUNTIME(cudaMemcpyAsync(cHost[0][1], cDev[0][1],
                                 m / 2 * n / 2 * sizeof(float),
                                 cudaMemcpyDefault, copyStream));
    CUDA_RUNTIME(cudaStreamWaitEvent(copyStream, waitC[1][1], 0));
    CUDA_RUNTIME(cudaMemcpyAsync(cHost[1][1], cDev[1][1],
                                 m / 2 * n / 2 * sizeof(float),
                                 cudaMemcpyDefault, copyStream));

    CUDA_RUNTIME(cudaDeviceSynchronize());
    nvtxRangePop(); // wall time
    Duration wallElapsed = Clock::now() - wallStart;

    // kernel time
    float kernelElapsed;
    CUDA_RUNTIME(cudaEventSynchronize(waitC[1][1]));
    CUDA_RUNTIME(cudaEventElapsedTime(&kernelElapsed, waitForA0B0, waitC[1][1]));
    kernelElapsed /= 1000; // seconds

    std::cerr << iter << " kernel=" << kernelElapsed
              << " wall=" << wallElapsed.count()
              << (iter >= nWarmup ? " *" : "  ") << "\n";

    if (iter >= nWarmup) {
      wallTime += wallElapsed.count();
      kernelTime += kernelElapsed;
    }
  }

  // print results
  double kernelGflops = flop / 1e9 / kernelTime;
  std::cerr << "kernel " << kernelGflops << "GFLOPS (" << flop << " flop, "
            << kernelTime << "s)\n";
  double wallGflops = flop / 1e9 / wallTime;
  std::cerr << "wall " << wallGflops << "GFLOPS (" << flop << " flop, "
            << wallTime << "s)\n";
  // release resources

  CUDA_RUNTIME(cudaFree(aDev[0]));
  CUDA_RUNTIME(cudaFree(aDev[1]));
  CUDA_RUNTIME(cudaFree(bDev[0]));
  CUDA_RUNTIME(cudaFree(bDev[1]));
  return 0;
}
