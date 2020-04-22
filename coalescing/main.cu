#include <algorithm>

#include <curand_kernel.h>

#include <argparse/argparse.hpp>

#include "common.hpp"


__global__ void coalesced(float *p, const size_t n) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < n) {
    unsigned int off = tid;
    float f = p[off];
    f += 1;
    p[off] = f;
  }

}

__global__ void uncoalesced(float *p, const size_t n) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned long long seed = 0;
  unsigned long long sequence = tid;
  unsigned long long offset = 0;

  curandState_t state;
  curand_init ( seed, sequence, offset, &state);

  if (tid < n) {
    unsigned int off = curand(state);
    float f = p[off];
    f += 1;
    p[off] = f;
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

  // allocate device data
  float *aDev;
  CUDA_RUNTIME(cudaMalloc(&aDev, n * sizeof(float)));

  // GPU kernel launch parameters
  dim3 dimBlock(512,1,1);
  dim3 dimGrid(1,1,1);
  dimGrid.x = (n + dimBlock.x - 1) / dimBlock.x;

  for (int i = 0; i < nIters + nWarmup; ++i) {
    coalesced<<<dimGrid, dimBlock>>>(aDev, n);
    CUDA_RUNTIME(cudaDeviceSynchronize());
    uncoalesced<<<dimGrid, dimBlock>>>(aDev, n);
    CUDA_RUNTIME(cudaDeviceSynchronize());
  }

  CUDA_RUNTIME(cudaFree(aDev));

  return 0;
}