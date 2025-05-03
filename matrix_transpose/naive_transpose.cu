#include <iostream>
#include <stdio.h>
#include <nvToolsExt.h>

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;

__global__
void copy(float *odata, const float *idata) {
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    odata[(y+j)*width + x] = idata[(y+j)*width + x];
  }
}

__global__
void transposeNaive(float *odata, const float *idata) {
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    odata[x*width + (y+j)] = idata[(y+j)*width + x];
  }
}

__global__
void transposeCoalesced(float *odata, const float *idata) {
  __shared__ float tile[TILE_DIM][TILE_DIM];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;
  
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*gridDim.x*TILE_DIM + x];
  }

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
  }
}

__global__
void transposeCoalescedNoBankConflicts(float *odata, const float *idata) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;
  
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*gridDim.x*TILE_DIM + x];
  }

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
  }
}

void profileKernel(
  void (*kernel)(float*, const float*),
  const std::string& kernel_name,
  float *odata,
  const float *idata,
  const int DIM,
  const int warmup_runs = 10,
  const int test_runs = 100
) {
  dim3 dimGrid(DIM/TILE_DIM, DIM/TILE_DIM, 1);
  dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

  nvtxRangePushA((kernel_name + " Warm-Up Phase").c_str());
  for (int i = 0; i < warmup_runs; i++) {
    kernel<<<dimGrid, dimBlock>>>(odata, idata);
  }
  cudaDeviceSynchronize();
  nvtxRangePop();
  nvtxRangePushA((kernel_name + " Test Phase").c_str());
  for (int i = 0; i < test_runs; i++) {
    kernel<<<dimGrid, dimBlock>>>(odata, idata);
  }
  cudaDeviceSynchronize();
  nvtxRangePop();
}

int main(void) {
  const int DIM = 512;
  const size_t matrix_size = DIM * DIM * sizeof(float); 

  float *src, *dst;

  cudaMallocManaged(&src, matrix_size);
  cudaMallocManaged(&dst, matrix_size);

  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < DIM; j++) {
      src[i*DIM + j] = i * DIM + j;
      dst[i*DIM + j] = 0;
    }
  }

  profileKernel(copy, "Copy Kernel", dst, src, DIM);
  std::cout << "Copy Kernel executed" << std::endl;
  profileKernel(transposeNaive, "Naive Transpose Kernel", dst, src, DIM);
  std::cout << "Naive Transpose Kernel executed" << std::endl;
  profileKernel(transposeCoalesced, "Coalesced Transpose Kernel", dst, src, DIM);
  std::cout << "Coalesced Transpose Kernel executed" << std::endl;
  profileKernel(transposeCoalescedNoBankConflicts, "Coalesced No Bank Conflicts Transpose Kernel", dst, src, DIM);
  std::cout << "Coalesced No Bank Conflicts Transpose Kernel executed" << std::endl;

  cudaFree(src);
  cudaFree(dst);

  return 0;
}