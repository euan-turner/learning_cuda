#ifndef COALESCED_SGEMM_CUH
#define COALESCED_SGEMM_CUH

#include "sgemm.cuh"

template <const uint BLOCKSIZE>
__global__
void coalesced_sgemm(SgemmParams ps) {
  // assign unique entry in C for this thread
  const uint x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const uint y = blockIdx.y * BLOCKSIZE + (threadIdx.y % BLOCKSIZE);

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

template <const uint BLOCKSIZE>
class CoalescedSgemmLauncher : public SgemmKernelLauncher {
public:
  cudaError_t launch(SgemmParams params) override {
    dim3 gridDim(CEIL_DIV(params.M, BLOCKSIZE), CEIL_DIV(params.N, BLOCKSIZE));
    dim3 blockDim(BLOCKSIZE * BLOCKSIZE);
    coalesced_sgemm<BLOCKSIZE><<<gridDim, blockDim>>>(params);
    return checkLaunchError(cudaGetLastError(), getKernelName());
  }

  std::string getKernelName() const override {
    return "coalesced_sgemm";
  }
};

#endif // COALESCED_SGEMM_CUH