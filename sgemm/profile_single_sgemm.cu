#include <iostream>
#include <memory>
#include <stdio.h>
#include <string>
#include <nvToolsExt.h>

#include "sgemm.cuh"
#include "naive_sgemm.cuh"
#include "coalesced_sgemm.cuh"
#include "cache_blocking_sgemm.cuh"
#include "blocktiling_1d_sgemm.cuh"

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#define CUDA_CHECK(call) { \
  cudaError_t err = call; \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
  } \
}

void profileKernel(
  SgemmParams ps,
  std::unique_ptr<SgemmKernelLauncher> launcher,
  const int warmup_runs = 10,
  const int test_runs = 100
) {
  // Allocate device memory
  float *d_A, *d_B, *d_C, *d_C_copy;
  size_t size_A = ps.M * ps.K * sizeof(float);
  size_t size_B = ps.K * ps.N * sizeof(float);
  size_t size_C = ps.M * ps.N * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_A, size_A);)
  CUDA_CHECK(cudaMalloc(&d_B, size_B));
  CUDA_CHECK(cudaMalloc(&d_C, size_C));
  CUDA_CHECK(cudaMalloc(&d_C_copy, size_C));

  // Copy data from host to device
  CUDA_CHECK(cudaMemcpy(d_A, ps.A, size_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, ps.B, size_B, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_C, ps.C, size_C, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_C_copy, ps.C, size_C, cudaMemcpyHostToDevice));

  SgemmParams device_ps = ps;
  device_ps.A = d_A;
  device_ps.B = d_B;
  device_ps.C = d_C;

  // Warm-up phase
  nvtxRangePushA((launcher->getKernelName() + " Warm-Up Phase").c_str());
  for (int i = 0; i < warmup_runs; i++) {
    // Reset C before each run
    CUDA_CHECK(cudaMemcpy(d_C, d_C_copy, size_C, cudaMemcpyDeviceToDevice));
    launcher->launch(device_ps);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  nvtxRangePop();

  // Test phase
  nvtxRangePushA((launcher->getKernelName() + " Test Phase").c_str());
  for (int i = 0; i < test_runs; i++) {
    // Reset C before each run
    CUDA_CHECK(cudaMemcpy(d_C, d_C_copy, size_C, cudaMemcpyDeviceToDevice));
    launcher->launch(device_ps);
    CUDA_CHECK(cudaDeviceSynchronize()); // does this need to be here?
  }
  nvtxRangePop();

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  CUDA_CHECK(cudaFree(d_C_copy));
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <kernel_name>" << std::endl;
    std::cerr << "Available kernels: naive, coalesced, cache_blocking, blocktiling_1d" << std::endl;
    return 1;
  }
  const std::string kernel_name = argv[1];

  const int M = 512;
  const int N = 512;
  const int K = 512;
  const size_t matrix_size = M * N * sizeof(float); 

  float *A, *B, *C;

  // Allocate host memory
  A = (float*)malloc(matrix_size);
  B = (float*)malloc(matrix_size);
  C = (float*)malloc(matrix_size);

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      A[i*N + j] = i * N + j;
      B[i*N + j] = i * N + j;
      C[i*N + j] = 0;
    }
  }

  SgemmParams params = {.M = M, .N = N, .K = K, .alpha = 1.0f, .beta = 0.0f, .A = A, .B = B, .C = C};
  if (kernel_name == "naive") {
    profileKernel(params, std::make_unique<NaiveSgemmLauncher>());
  } else if (kernel_name == "coalesced") {
    profileKernel(params, std::make_unique<CoalescedSgemmLauncher>());
  } else if (kernel_name == "cache_blocking") {
    profileKernel(params, std::make_unique<CacheBlockingSgemmLauncher<32>>());
  } else if (kernel_name == "blocktiling_1d") {
    profileKernel(params, std::make_unique<Blocktiling1dSgemmLauncher<64, 8, 64, 8>>());
  } else {
    std::cerr << "Unknown kernel name: " << kernel_name << std::endl;
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 1;
  }

  std::cout << kernel_name << " executed" << std::endl;

  free(A);
  free(B);
  free(C);

}