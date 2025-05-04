#ifndef NAIVE_SGEMM_CUH
#define NAIVE_SGEMM_CUH

#include "sgemm.cuh"

__global__
void naive_sgemm(SgemmParams ps);

#endif // NAIVE_SGEMM_CUH