#ifndef NAIVE_SGEMM_CUH
#define NAIVE_SGEMM_CUH

__global__
void naive_sgemm(int M, int N, int K, float alpha, const float *A,
  const float *B, float beta, float *C);

#endif // NAIVE_SGEMM_CUH