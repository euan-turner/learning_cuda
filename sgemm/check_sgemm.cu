// filepath: /homes/et422/Documents/learn_cuda/sgemm/check_sgemm.cu
#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <cassert>

#include "sgemm.cuh"
#include "naive_sgemm.cuh"
#include "coalesced_sgemm.cuh"
#include "cache_blocking_sgemm.cuh"
#include "blocktiling_1d_sgemm.cuh"

#define CUDA_CHECK(call) { \
  cudaError_t err = call; \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
  } \
}

void host_sgemm(const SgemmParams& ps) {
  // C = alpha * A * B + beta * C
  for (int i = 0; i < ps.M; ++i) {
    for (int j = 0; j < ps.N; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < ps.K; ++k) {
        sum += ps.A[i * ps.K + k] * ps.B[k * ps.N + j];
      }
      ps.C[i * ps.N + j] = ps.alpha * sum + ps.beta * ps.C[i * ps.N + j];
    }
  }
}

bool compare_matrices(const float* ref, const float* test, int M, int N, float atol = 1e-3f, float rtol = 1e-3f) {
  for (int i = 0; i < M * N; ++i) {
    float diff = std::fabs(ref[i] - test[i]);
    float tol = atol + rtol * std::fabs(ref[i]);
    if (diff > tol) {
      std::cerr << "Mismatch at index " << i << ": ref=" << ref[i] << ", test=" << test[i] << ", diff=" << diff << std::endl;
      return false;
    }
  }
  return true;
}

int main() {
  const int M = 512;
  const int N = 512;
  const int K = 512;
  const size_t size_A = M * K * sizeof(float);
  const size_t size_B = K * N * sizeof(float);
  const size_t size_C = M * N * sizeof(float);

  // Allocate and initialize host matrices
  std::vector<float> A(M * K), B(K * N), C(M * N), C_ref(M * N);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      A[i * K + (j % K)] = static_cast<float>(i * N + j);
      B[(i % K) * N + j] = static_cast<float>(i * N + j);
      C[i * N + j] = 0.0f;
      C_ref[i * N + j] = 0.0f;
    }
  }

  SgemmParams params = {.M = M, .N = N, .K = K, .alpha = 1.0f, .beta = 0.0f, .A = A.data(), .B = B.data(), .C = C.data()};
  SgemmParams ref_params = params;
  ref_params.C = C_ref.data();

  // Compute reference result on host
  host_sgemm(ref_params);

  // Prepare device memory
  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, size_A));
  CUDA_CHECK(cudaMalloc(&d_B, size_B));
  CUDA_CHECK(cudaMalloc(&d_C, size_C));

  CUDA_CHECK(cudaMemcpy(d_A, A.data(), size_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, B.data(), size_B, cudaMemcpyHostToDevice));

  std::vector<std::unique_ptr<SgemmKernelLauncher>> launchers;
  launchers.push_back(std::make_unique<NaiveSgemmLauncher>());
  launchers.push_back(std::make_unique<CoalescedSgemmLauncher>());
  launchers.push_back(std::make_unique<CacheBlockingSgemmLauncher<32>>());
  launchers.push_back(std::make_unique<Blocktiling1dSgemmLauncher<64, 8, 64, 8>>());

  for (const auto& launcher : launchers) {
    // Reset C on device
    CUDA_CHECK(cudaMemcpy(d_C, C.data(), size_C, cudaMemcpyHostToDevice));

    SgemmParams device_ps = params;
    device_ps.A = d_A;
    device_ps.B = d_B;
    device_ps.C = d_C;

    launcher->launch(device_ps);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> C_out(M * N);
    CUDA_CHECK(cudaMemcpy(C_out.data(), d_C, size_C, cudaMemcpyDeviceToHost));

    bool ok = compare_matrices(C_ref.data(), C_out.data(), M, N);
    std::cout << "Kernel " << launcher->getKernelName() << ": " << (ok ? "PASS" : "FAIL") << std::endl;
    assert(ok && "SGEMM kernel output does not match reference!");
  }

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  return 0;
}