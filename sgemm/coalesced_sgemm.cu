#include "sgemm.cuh"
#include "coalesced_sgemm.cuh"

__global__
void coalesced_sgemm(SgemmParams ps) {
  // ensure adjacent threads in a warp are assigned adjacent entries in C along a row
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;
  // thread is responsible for C[y,x]

  // cover case where M,N aren't multiples of 32
  if (x < ps.N && y < ps.M) {
    float tmp = 0.0;
    // perform dot product
    for (int i = 0; i < ps.K; i++) {
      tmp += ps.A[y * ps.K + i] * ps.B[i * ps.N + x];
    }
    ps.C[y * ps.N + x] = ps.alpha * tmp + ps.beta * ps.C[y * ps.N +  x];
  }
}

cudaError_t CoalescedSgemmLauncher::launch(SgemmParams params) {
  dim3 gridDim(CEIL_DIV(params.N, 32), CEIL_DIV(params.M, 32));
  dim3 blockDim(32, 32);
  coalesced_sgemm<<<gridDim, blockDim>>>(params);
  return checkLaunchError(cudaGetLastError(), getKernelName());
}

std::string CoalescedSgemmLauncher::getKernelName() const {
  return "coalesced_sgemm";
}