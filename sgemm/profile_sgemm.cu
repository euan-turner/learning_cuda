#include <iostream>
#include <memory>
#include <vector>

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


/**

 * @brief Profile a kernel function for SGEMM (Single-precision General Matrix Multiply)
 * 
 *
 * Each kernel computes \f$ C = \alpha AB + \beta C \f$
  * where \f$ A \in \mathbb{R}^{M \times K} \f$, \f$ B \in \mathbb{R}^{K \times N} \f$, and \f$ C \in \mathbb{R}^{M \times N} \f$.
 */
void profileKernels(
  SgemmParams ps,
  const int warmup_runs = 10,
  const int test_runs = 100
) {

  // Allocate device memory
  float *d_A, *d_B, *d_C, *d_C_copy;
  size_t size_A = ps.M * ps.K * sizeof(float);
  size_t size_B = ps.K * ps.N * sizeof(float);
  size_t size_C = ps.M * ps.N * sizeof(float);
  cudaMalloc(&d_A, size_A);
  cudaMalloc(&d_B, size_B);
  cudaMalloc(&d_C, size_C);
  cudaMalloc(&d_C_copy, size_C);

  // Copy data from host to device
  cudaMemcpy(d_A, ps.A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, ps.B, size_B, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, ps.C, size_C, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C_copy, ps.C, size_C, cudaMemcpyHostToDevice);


  std::vector<std::unique_ptr<SgemmKernelLauncher>> launchers;
  launchers.push_back(std::make_unique<NaiveSgemmLauncher>());
  launchers.push_back(std::make_unique<CoalescedSgemmLauncher<32>>());

  SgemmParams device_ps = ps;
  device_ps.A = d_A;
  device_ps.B = d_B;
  device_ps.C = d_C;

  for (const auto& launcher : launchers) {
    std::cout << "Profiling kernel: " << launcher->getKernelName() << std::endl;

    // Warm-up phase
    nvtxRangePushA((launcher->getKernelName() + " Warm-Up Phase").c_str());
    for (int i = 0; i < warmup_runs; i++) {
      // Reset C before each run
      cudaMemcpy(d_C, d_C_copy, size_C, cudaMemcpyDeviceToDevice);
      launcher->launch(device_ps);
      cudaDeviceSynchronize(); // does this need to be here?
    }
    nvtxRangePop();

    // Test phase
    nvtxRangePushA((launcher->getKernelName() + " Test Phase").c_str());
    for (int i = 0; i < test_runs; i++) {
      // Reset C before each run
      cudaMemcpy(d_C, d_C_copy, size_C, cudaMemcpyDeviceToDevice);
      launcher->launch(device_ps);
      cudaDeviceSynchronize(); // does this need to be here?
    }
    nvtxRangePop();

  }
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_C_copy);
}

int main(void) {
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
  
  profileKernels(params, 10, 100);

  free(A);
  free(B);
  free(C);

}