#include "sgemm.cuh"

template <int BLOCK_SIZE>
__global__
void cache_blocking_sgemm(SgemmParams ps) {
  // assign output block in C, to be computed by this thread
  const uint x = blockIdx.x;
  const uint y = blockIdx.y;

  // shared memory buffers
  __shared__ float A[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float B[BLOCK_SIZE][BLOCK_SIZE];

  // advance pointers to block starting positions
  // A_ptr is first block on the yth block row of A
  const float *A_ptr = ps.A + y * BLOCK_SIZE * ps.K;
  // B_ptr is first block on the xth block column of B
  const float *B_ptr = ps.B + x * BLOCK_SIZE;
  // C_ptr is the xth block column of the yth block row of C
  float *C_ptr = ps.C + y * BLOCK_SIZE * ps.N + x * BLOCK_SIZE;

  float tmp = 0.0;
  for (int blkIdx = 0; blkIdx < ps.K; blkIdx += BLOCK_SIZE) {
    // load A and B blocks into shared memory
    // accesses and stores are coalesced
    A[threadIdx.y][threadIdx.x] = A_ptr[threadIdx.y * ps.K + threadIdx.x];
    B[threadIdx.y][threadIdx.x] = B_ptr[threadIdx.y * ps.N + threadIdx.x];
    __syncthreads();

    // advance blocks
    A_ptr += BLOCK_SIZE;
    B_ptr += BLOCK_SIZE * ps.N;

    // compute partial dot product
    // thread is responsible for partial product into C_ptr[threadIdx.y][threadIdx.x]
    for (int idx = 0; idx < BLOCK_SIZE; ++idx) {
      tmp += A[threadIdx.y][idx] * B[idx][threadIdx.x];
    }
    // prevent faster blocks from loading the next block into the cache
    __syncthreads();
  }
  C_ptr[threadIdx.y * ps.N + threadIdx.x] = ps.alpha * tmp + ps.beta * ps.C[threadIdx.y * ps.N + threadIdx.x];
}

class CacheBlockingSgemmLauncher : public SgemmKernelLauncher {
public:
  std::string getKernelName() const override {
    return "cache_blocking_sgemm";
  }

  cudaError_t launch(SgemmParams params) {
    const int BLOCK_SIZE = 32;
    dim3 gridDim(CEIL_DIV(params.N, BLOCK_SIZE), CEIL_DIV(params.M, BLOCK_SIZE));
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    cache_blocking_sgemm<BLOCK_SIZE><<<gridDim, blockDim>>>(params);
    return checkLaunchError(cudaGetLastError(), getKernelName());
  }
};