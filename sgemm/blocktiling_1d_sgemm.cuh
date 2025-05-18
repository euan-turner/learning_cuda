#include "sgemm.cuh"

// A cache is BM * BK, B cache is BK * BN
// Each thread calculates TM results
// One thread block is responsible for a BM * BN chunk
// Thread blocks are 1-dimensional as they handle transposed tiles of A and B
template <int BM, int BK, int BN, int TM>
__global__
void blocktiling_1d_sgemm(SgemmParams ps) {
  // assign output block in C
  const uint x = blockIdx.x;
  const uint y = blockIdx.y;

  const uint threadCol = threadIdx.x % BN;
  const uint threadRow = threadIdx.x / BN;

  // shared memory buffers
  __shared__ float A[BM][BK];
  __shared__ float B[BK][BN];

  // advance pointers to starting positions
  const float *A_ptr = ps.A + y * BM * ps.K;
  const float *B_ptr = ps.B + x * BN;
  float *C_ptr = ps.C + y * BM * ps.N + x * BN;

  // Assign thread indices within A_ptr and B_ptr to coalesce loads
  const uint innerColA = threadIdx.x % BK;
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColB = threadIdx.x % BN;
  const uint innerRowB = threadIdx.x / BN;

  float threadResults[TM] = {0.0};

  for (int blkIdx = 0; blkIdx < ps.K; blkIdx += BK) {
    // load blocks into shared memory
    A[innerRowA][innerColA] = A_ptr[innerRowA * ps.K + innerColA];
    B[innerRowB][innerColB] = B_ptr[innerRowB * ps.N + innerColB];
    __syncthreads();

    // advance blocks
    A_ptr += BK;
    B_ptr += BK * ps.N;

    // compute multiple partial dot products
    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // cache B entry
      float B_tmp = B[dotIdx][threadCol];
      for (int resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] += A[threadRow * TM + resIdx][dotIdx] * B_tmp;
      }
    }
    __syncthreads();
  }
  for (int resIdx = 0; resIdx < TM; ++resIdx) {
    C_ptr[(threadRow * TM + resIdx) * ps.N + threadCol] = ps.alpha * threadResults[resIdx] + ps.beta * C_ptr[(threadRow * TM + resIdx) * ps.N + threadCol];
  }
}

class Blocktiling1dSgemmLauncher : public SgemmKernelLauncher {
public:
  std::string getKernelName() const override {
    return "1d_blocktiling_sgemm";
  }

  cudaError_t launch(SgemmParams params) {
    const int BK = 8;
    const int TM = 8;
    if (params.M >= 64 && params.N >= 64) {
      const uint BM = 64;
      const uint BN = 64;
      dim3 gridDim(CEIL_DIV(params.N, BN), CEIL_DIV(params.M, BM));
      dim3 blockDim((BM * BN) / TM);
      blocktiling_1d_sgemm<BM, BK, BN, TM><<<gridDim, blockDim>>>(params);
      return checkLaunchError(cudaGetLastError(), getKernelName());
    } else {
      return cudaErrorInvalidValue;
    }
    
  }
};
