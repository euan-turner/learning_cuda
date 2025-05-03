#ifndef COPY_KERNEL_CUH
#define COPY_KERNEL_CUH

__global__
void copy(float *odata, const float *idata);

#endif // COPY_KERNEL_CUH