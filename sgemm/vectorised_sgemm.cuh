#include "sgemm.cuh"

// Vectorises accesses to GMEM and SMEM
// A cache is BM * BK, B cache is BK * BN
// Each thread calculates TM  * TN results
// One thread block is responsble for a BM * BN chunk
// Thread blocks are 1-dimensional as they handle transposed tiles of A and B
template <int BM, int BK, int BN, int TM, int TN>
__global__
void vectorised_sgemm(SgemmParams ps) {
  // assign output block in C
  const uint x = blockIdx.x;
  const uint y = blockIdx.y;

  // Assign thread index within result block
  // BN / TN threads span a row of the output block
  const uint threadCol = threadIdx.x % (BN / TN);
  const uint threadRow = threadIdx.x / (BN / TN);

  // shared memory buffers
  __shared__ float A[BK][BM]; // transposed
  __shared__ float B[BK][BN];

  // advance pointers to starting positions
  const float *A_ptr = ps.A + y * BM * ps.K;
  const float *B_ptr = ps.B + x * BN;
  float *C_ptr = ps.C + y * BM * ps.N + x * BN;

  // Assign thread indices within A_ptr and B_ptr to vectorise loads
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  const uint innerRowB = threadIdx.x / (BN / 4);

  float threadResults[TM][TN] = {0.0};
  // Register caches from SMEM
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  for (int blkIdx = 0; blkIdx < ps.K; blkIdx += BK) {
    // each thread loads a vector of 4 elements at a time from A_ptr and B_ptr
    float4 tmp = reinterpret_cast<const float4 *>(&A_ptr[innerRowA * ps.K + innerColA * 4])[0];
    // transpose into SMEM
    A[innerColA * 4 + 0][innerRowA] = tmp.x;
    A[innerColA * 4 + 1][innerRowA] = tmp.y;
    A[innerColA * 4 + 2][innerRowA] = tmp.z;
    A[innerColA * 4 + 3][innerRowA] = tmp.w;

    reinterpret_cast<float4 *>(&B[innerRowB][innerColB * 4])[0] =
      reinterpret_cast<const float4 *>(&B_ptr[innerRowB * ps.N + innerColB * 4])[0];
    __syncthreads();

    // advance blocks
    A_ptr += BK;
    B_ptr += BK * ps.N;

    // per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // cache A and B in registers
      for (uint i = 0; i < TM; ++i) {
        regM[i] = A[dotIdx][threadRow * TM + i];
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
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
      // load vector of C in
      float4 tmp = reinterpret_cast<float4 *>(&C_ptr[(threadRow * TM + resIdxM) * ps.N + threadCol * TN + resIdxN])[0];

      // sgemm
      tmp.x = ps.alpha * threadResults[resIdxM][resIdxN] + ps.beta * tmp.x;
      tmp.y = ps.alpha * threadResults[resIdxM][resIdxN + 1] + ps.beta * tmp.y;
      tmp.z = ps.alpha * threadResults[resIdxM][resIdxN + 2] + ps.beta * tmp.z;
      tmp.w = ps.alpha * threadResults[resIdxM][resIdxN + 3] + ps.beta * tmp.w;

      // write vector of C back
      reinterpret_cast<float4 *>(&C_ptr[(threadRow * TM + resIdxM) * ps.N + threadCol * TN + resIdxN])[0] = tmp;
    }
  }
  
}

class VectorisedSgemmLauncher : public SgemmKernelLauncher {
public:
  std::string getKernelName() const override {
    return "vectorised_sgemm";
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
      vectorised_sgemm<BM, BK, BN, TM, TN><<<gridDim, blockDim>>>(ps);
      return checkLaunchError(cudaGetLastError(), getKernelName());
    } else {
      return cudaErrorInvalidValue;
    }
  }
};