#ifndef NAIVE_SGEMM_CUH
#define NAIVE_SGEMM_CUH

#include "sgemm.cuh"

__global__
void naive_sgemm(SgemmParams ps);


class NaiveSgemmLauncher : public SgemmKernelLauncher {
public:
  std::string getKernelName() const override;
  cudaError_t launch(SgemmParams params) override;
};
#endif // NAIVE_SGEMM_CUH