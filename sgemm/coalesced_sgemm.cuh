#ifndef COALESCED_SGEMM_CUH
#define COALESCED_SGEMM_CUH

#include "sgemm.cuh"

__global__
void coalesced_sgemm(SgemmParams ps);

class CoalescedSgemmLauncher : public SgemmKernelLauncher {
public:
  std::string getKernelName() const override;
  cudaError_t launch(SgemmParams params) override;
};

#endif // COALESCED_SGEMM_CUH