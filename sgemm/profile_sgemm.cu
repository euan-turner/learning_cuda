#include <iostream>
#include <stdio.h>
#include <nvToolsExt.h>

#include "sgemm.cuh"
#include "naive_sgemm.cuh"
#include "coalesced_sgemm.cuh"

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define CUDA_CHECK(call) { \
  cudaError_t err = call; \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
  } \
}
// TODO: Use this

const int BLOCK_SIZE = 32;

/**

 * @brief Profile a kernel function for SGEMM (Single-precision General Matrix Multiply)
 * 
 *
 * Each kernel computes \f$ C = \alpha AB + \beta C \f$
  * where \f$ A \in \mathbb{R}^{M \times K} \f$, \f$ B \in \mathbb{R}^{K \times N} \f$, and \f$ C \in \mathbb{R}^{M \times N} \f$.
 */
void profileKernel(
  void (*kernel)(SgemmParams),
  const std::string& kernel_name,
  const dim3 dimGrid,
  const dim3 dimBlock,
  SgemmParams ps,
  const int warmup_runs = 10,
  const int test_runs = 100
) {
  float* C_original;
  cudaMalloc(&C_original, ps.M * ps.N * sizeof(float));
  cudaMemcpy(C_original, ps.C, ps.M * ps.N * sizeof(float), cudaMemcpyDeviceToDevice);

  nvtxRangePushA((kernel_name + " Warm-Up Phase").c_str());
  for (int i = 0; i < warmup_runs; i++) {
      // Reset C before each run
      cudaMemcpy(ps.C, C_original, ps.M * ps.N * sizeof(float), cudaMemcpyDeviceToDevice);
      kernel<<<dimGrid, dimBlock>>>(ps);
  }
  cudaDeviceSynchronize();
  nvtxRangePop();

  nvtxRangePushA((kernel_name + " Test Phase").c_str());
  for (int i = 0; i < test_runs; i++) {
      // Reset C before each run
      cudaMemcpy(ps.C, C_original, ps.M * ps.N * sizeof(float), cudaMemcpyDeviceToDevice);
      kernel<<<dimGrid, dimBlock>>>(ps);
  }
  cudaDeviceSynchronize();
  nvtxRangePop();

  cudaFree(C_original);
}

int main(void) {
  const int M = 512;
  const int N = 512;
  const int K = 512;
  const size_t matrix_size = M * N * sizeof(float); 

  float *A, *B, *C;

  cudaMallocManaged(&A, matrix_size);
  cudaMallocManaged(&B, matrix_size);
  cudaMallocManaged(&C, matrix_size);

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      A[i*N + j] = i * N + j;
      B[i*N + j] = i * N + j;
      C[i*N + j] = 0;
    }
  }

  SgemmParams params = {.M = M, .N = N, .K = K, .alpha = 1.0f, .beta = 0.0f, .A = A, .B = B, .C = C};
  
  profileKernel(naive_sgemm, "Naive SGEMM Kernel", dim3(CEIL_DIV(M, BLOCK_SIZE), CEIL_DIV(N, BLOCK_SIZE), 1), dim3(BLOCK_SIZE, BLOCK_SIZE, 1), 
    params);
  std::cout << "SGEMM Naive Kernel executed" << std::endl;

  profileKernel(
    coalesced_sgemm<BLOCK_SIZE>,
    "Coalesced SGEMM Kernel", dim3(CEIL_DIV(M, BLOCK_SIZE), CEIL_DIV(N, BLOCK_SIZE), 1), dim3(BLOCK_SIZE * BLOCK_SIZE, 1, 1), 
    params);
  std::cout << "SGEMM Coalesced Kernel executed" << std::endl;

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

}