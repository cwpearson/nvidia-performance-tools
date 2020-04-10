#include <algorithm>

#include <nvToolsExt.h>

#include <argparse/argparse.hpp>

#include "common.hpp"

/* NOTE: A and C are column major, B is row major
 */
__global__ void mygemm(float *__restrict__ c, //<! [out] and MxN matrix
                       const float *a,        //<! [in] an MxK matrix
                       const float *b,        //<! [in] an KxN matrix
                       const int M, const int N, const int K) {

#define A(_i, _j) a[(_i) + (_j)*M]
#define B(_i, _j) b[(_i)*N + (_j)]
#define C(_i, _j) c[(_i) + (_j)*M]

  int gidx = blockDim.x * blockIdx.x + threadIdx.x;
  int gidy = blockDim.y * blockIdx.y + threadIdx.y;

  for (int i = gidy; i < M; i += gridDim.y * blockDim.y) {
    for (int j = gidx; j < N; j += gridDim.x * blockDim.x) {
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
  dim3 dimBlock(32, 32);
  dim3 dimGrid;
  dimGrid.x = (n + dimBlock.x - 1) / dimBlock.x;
  dimGrid.y = (m + dimBlock.y - 1) / dimBlock.y;

  // total elapsed time
  float elapsed = 0;

  /* Launch the kernel nIters + nWarmup times
     Check for correctness on the first time.
     Record the time after nWarmup runs complete.
  */
  for (int i = 0; i < nIters + nWarmup; ++i) {
    nvtxRangePush("kernel");
    CUDA_RUNTIME(cudaEventRecord(start));
    mygemm<<<dimGrid, dimBlock>>>(cDev, aDev, bDev, m, n, k);
    CUDA_RUNTIME(cudaEventRecord(stop));
    CUDA_RUNTIME(cudaEventSynchronize(stop));
    nvtxRangePop();

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