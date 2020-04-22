#include <algorithm>
#include <numeric>
#include <random>

#include <argparse/argparse.hpp>

#include "common.hpp"


__global__ void indirect(float *p, int *off, const size_t n) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  for (int i = tid; i < n; i += gridDim.x * blockDim.x) {
    int idx = off[i];
    float f = p[idx];
    f += 1;
    p[idx] = f;
  }

}



int main(int argc, char **argv) {

  argparse::Parser parser;

  int n = 10000;
  int nIters = 5;
  int nWarmup = 5;
  parser.add_positional(n);

  if (!parser.parse(argc, argv)) {
    parser.help();
    exit(EXIT_FAILURE);
  }

  // generate access patterns
  std::vector<int> cHost(n), uHost(n);
  std::iota(cHost.begin(), cHost.end(), 0);
  std::iota(uHost.begin(), uHost.end(), 0);
  std::shuffle(uHost.begin(), uHost.end(), std::mt19937{std::random_device{}()});


  // allocate device data
  float *aDev;
  int *cDev, *uDev;
  CUDA_RUNTIME(cudaMalloc(&aDev, n * sizeof(float)));
  CUDA_RUNTIME(cudaMalloc(&cDev, n * sizeof(int)));
  CUDA_RUNTIME(cudaMalloc(&uDev, n * sizeof(int)));

  // copy indices to device
  CUDA_RUNTIME(cudaMemcpy(cDev, cHost.data(), cHost.size() * sizeof(int), cudaMemcpyDefault));
  CUDA_RUNTIME(cudaMemcpy(uDev, uHost.data(), uHost.size() * sizeof(int), cudaMemcpyDefault));

  // GPU kernel launch parameters
  dim3 dimBlock(512,1,1);
  dim3 dimGrid(1,1,1);
  dimGrid.x = (n + dimBlock.x - 1) / dimBlock.x;

  for (int i = 0; i < nIters + nWarmup; ++i) {
    indirect<<<dimGrid, dimBlock>>>(aDev, cDev, n);
    CUDA_RUNTIME(cudaDeviceSynchronize());
    indirect<<<dimGrid, dimBlock>>>(aDev, uDev, n);
    CUDA_RUNTIME(cudaDeviceSynchronize());
  }

  CUDA_RUNTIME(cudaFree(aDev));
  CUDA_RUNTIME(cudaFree(cDev));
  CUDA_RUNTIME(cudaFree(uDev));

  return 0;
}