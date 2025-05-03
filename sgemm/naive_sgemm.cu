#include "naive_sgemm.cuh"

__global__
void naive_sgemm(int M, int N, int K, 
  float alpha, const float *A, const float *B,
  float beta, float *C) {
  // assign unique entry in C for this thread
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // cover case where M,N aren't multiples of 32
  if (x < M && y < N) {
    float tmp = 0.0;
    // perform dot product
    for (int i = 0; i < K; i++) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}