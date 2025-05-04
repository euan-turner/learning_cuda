#ifndef SGEMM_CUH
#define SGEMM_CUH

struct SgemmParams {
  int M, N, K;
  float alpha, beta;
  const float *A, *B;
  float *C;
};

#endif // SGEMM_CUH