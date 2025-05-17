#include "sgemm.cuh"

// A cache is BM * BK, B cache is BK * BN
// Each thread calculates TM  * TN results
// One thread block is responsble for a BM * BN chunk
// Thread blocks are 1-dimensional as they handle transposed tiles of A and B
template <int BM, int BK, int BN, int TM, int TN>
__global__
void blocktiling_2d_sgemm(SgemmParams ps) {
  // assign output block in C
  const uint x = blockIdx.x;
  const uint y = blockIdx.y;

  const int resultsFromBlocktile = BM * BN;
  const int threadsInBlocktile = resultsFromBlocktile / (TM * TN); // == blockDim.x

  // Assign thread index within result block
  // BN / TN threads span a row of the output block
  const uint threadCol = threadIdx.x % (BN / TN);
  const uint threadRow = threadIdx.x / (BN / TN);

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
  // Assign strides for SMEM caching - there are fewer threads/block than elements to be cached
  // so caching must iterate over the blocks in A_ptr and B_ptr
  const uint strideA = threadsInBlocktile / BK;
  const uint strideB = threadsInBlocktile / BN;

  float threadResults[TM][TN] = {0.0};
  // Register caches from SMEM
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  for (int blkIdx = 0; blkIdx < ps.K; blkIdx += BK) {
    // cache A_ptr[0:BM][0:BK] and B_ptr[0:BK][0:BN]
    for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      A[loadOffset + innerRowA][innerColA] = A_ptr[(loadOffset + innerRowA) * ps.K + innerColA];
    }
    for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      B[loadOffset + innerRowB][innerColB] = B_ptr[(loadOffset + innerRowB) * ps.N + innerColB];
    }
    __syncthreads();

    // advance blocks
    A_ptr += BK;
    B_ptr += BK * ps.N;

    // per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // cache A and B in registers
      for (uint i = 0; i < TM; ++i) {
        regM[i] = A[threadRow * TM + i][dotIdx];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = B[dotIdx][threadCol * TN + i];
      }

      // outer product on register cache for partial results
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM][resIdxN] += regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }
  // Write back TM * TN results per thread
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      uint idx = (threadRow * TM + resIdxM) * ps.N + threadCol * TN + resIdxN;
      C_ptr[idx] = ps.alpha * threadResults[resIdxM][resIdxN] + ps.beta * C_ptr[idx];
    }
  }
  
}

class Blocktiling2dSgemmLauncher : public SgemmKernelLauncher {
public:
  std::string getKernelName() const override {
    return "2d_blocktiling_sgemm";
  }

  cudaError_t launch(SgemmParams ps) {
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    // Require blockDim = (BM * BN) / (TM * TN)
    if (ps.M >= 128 && ps.N >= 128) {
      const uint BM = 128;
      const uint BN = 128;
      dim3 gridDim(CEIL_DIV(ps.N, BN), CEIL_DIV(ps.M, BM));
      dim3 blockDim((BM * BN) / (TM * TN));
      blocktiling_2d_sgemm<BM, BK, BN, TM, TN><<<gridDim, blockDim>>>(ps);
      return checkLaunchError(cudaGetLastError(), getKernelName());
    } else {
      return cudaErrorInvalidValue;
    }
  }
};