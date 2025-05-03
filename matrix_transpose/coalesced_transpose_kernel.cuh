#ifndef COALESCED_TRANSPOSE_KERNEL_CUH
#define COALESCED_TRANSPOSE_KERNEL_CUH

__global__
void transposeCoalesced(float *odata, const float *idata);

#endif // COALESCED_TRANSPOSE_KERNEL_CUH