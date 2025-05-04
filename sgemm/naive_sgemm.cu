#include "sgemm.cuh"
#include "naive_sgemm.cuh"

__global__
void naive_sgemm(SgemmParams ps) {
  // assign unique entry in C for this thread
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // cover case where M,N aren't multiples of 32
  if (x < ps.M && y < ps.N) {
    float tmp = 0.0;
    // perform dot product
    for (int i = 0; i < ps.K; i++) {
      tmp += ps.A[x * ps.K + i] * ps.B[i * ps.N + y];
    }
    ps.C[x * ps.N + y] = ps.alpha * tmp + ps.beta * ps.C[x * ps.N + y];
  }
}

std::string NaiveSgemmLauncher::getKernelName() const {
  return "naive_sgemm";
}

cudaError_t NaiveSgemmLauncher::launch(SgemmParams params) {
  dim3 gridDim(CEIL_DIV(params.M, 32), CEIL_DIV(params.N, 32));
  dim3 blockDim(32, 32);
  naive_sgemm<<<gridDim, blockDim>>>(params);
  return checkLaunchError(cudaGetLastError(), getKernelName());
}