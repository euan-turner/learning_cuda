#include "copy_kernel.cuh"

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