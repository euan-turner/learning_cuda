#ifndef NAIVE_TRANSPOSE_KERNEL_CUH
#define NAIVE_TRANSPOSE_KERNEL_CUH

__global__
void transposeNaive(float *odata, const float *idata);

#endif // NAIVE_TRANSPOSE_KERNEL_CUH