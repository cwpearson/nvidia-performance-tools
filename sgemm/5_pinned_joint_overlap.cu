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
  parser.add_positional(m);
  parser.add_positional(n);
  parser.add_positional(k);
  parser.add_option(nIters, "--iters");

  if (!parser.parse(argc, argv)) {
    parser.help();
    exit(EXIT_FAILURE);
  }

  const int64_t flop = int64_t(m) * int64_t(n) * int64_t(k) * 2;

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

  // create events to time GPU kernel
  cudaEvent_t start;
  CUDA_RUNTIME(cudaEventCreate(&start));

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
  dim3 dimGrid((m / 2 + TILE_SZ_A - 1) / TILE_SZ_A,
               (n / 2 + TILE_SZ_B - 1) / TILE_SZ_B);
  dim3 dimBlock(TILE_SZ_A, 1);

  nvtxRangePush("wall time");
  auto wallStart = std::chrono::system_clock::now();

  // copy a0 and b0
  CUDA_RUNTIME(cudaMemcpyAsync(aDev[0], aHost[0], m / 2 * k * sizeof(float),
                               cudaMemcpyDefault, copyStream));
  CUDA_RUNTIME(cudaMemcpyAsync(bDev[0], bHost[0], k * n / 2 * sizeof(float),
                               cudaMemcpyDefault, copyStream));
  CUDA_RUNTIME(cudaEventRecord(waitForA0B0, copyStream));

  // have the kernelStream wait for the transfers to complete
  CUDA_RUNTIME(cudaStreamWaitEvent(kernelStream, waitForA0B0, 0));

  // launch c[0][0] = a[0] * b[0]
  CUDA_RUNTIME(cudaEventRecord(start, kernelStream));
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
                               m / 2 * n / 2 * sizeof(float), cudaMemcpyDefault,
                               copyStream));
  CUDA_RUNTIME(cudaStreamWaitEvent(copyStream, waitC[1][0], 0));
  CUDA_RUNTIME(cudaMemcpyAsync(cHost[1][0], cDev[1][0],
                               m / 2 * n / 2 * sizeof(float), cudaMemcpyDefault,
                               copyStream));
  CUDA_RUNTIME(cudaStreamWaitEvent(copyStream, waitC[0][1], 0));
  CUDA_RUNTIME(cudaMemcpyAsync(cHost[0][1], cDev[0][1],
                               m / 2 * n / 2 * sizeof(float), cudaMemcpyDefault,
                               copyStream));
  CUDA_RUNTIME(cudaStreamWaitEvent(copyStream, waitC[1][1], 0));
  CUDA_RUNTIME(cudaMemcpyAsync(cHost[1][1], cDev[1][1],
                               m / 2 * n / 2 * sizeof(float), cudaMemcpyDefault,
                               copyStream));

  CUDA_RUNTIME(cudaDeviceSynchronize());
  auto wallStop = std::chrono::system_clock::now();
  nvtxRangePop(); // wall time
  float wallElapsed = std::chrono::duration_cast<std::chrono::duration<float>>(
                          wallStop - wallStart)
                          .count();

  // kernel time
  float kernelElapsed;
  CUDA_RUNTIME(cudaEventSynchronize(waitC[1][1]));
  CUDA_RUNTIME(cudaEventElapsedTime(&kernelElapsed, start, waitC[1][1]));
  kernelElapsed /= 1000; // seconds

  // print results
  double kernelGflops = flop / 1e9 / kernelElapsed;
  std::cerr << "kernel " << kernelGflops << "GFLOPS (" << flop << " flop, "
            << kernelElapsed << "s)\n";
  double wallGflops = flop / 1e9 / wallElapsed;
  std::cerr << "wall " << wallGflops << "GFLOPS (" << flop << " flop, "
            << wallElapsed << "s)\n";
  // release resources

  CUDA_RUNTIME(cudaEventDestroy(start));
  CUDA_RUNTIME(cudaFree(aDev[0]));
  CUDA_RUNTIME(cudaFree(aDev[1]));
  CUDA_RUNTIME(cudaFree(bDev[0]));
  CUDA_RUNTIME(cudaFree(bDev[1]));
  return 0;
}
