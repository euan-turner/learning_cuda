#ifndef SGEMM_CUH
#define SGEMM_CUH

#include <string>
#include <iostream>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))


struct SgemmParams {
  int M, N, K;
  float alpha, beta;
  const float *A, *B;
  float *C;
};



class SgemmKernelLauncher {
public:
  virtual ~SgemmKernelLauncher() = default;

  // Custom launch configuration for each kernel
  virtual cudaError_t launch(SgemmParams params) = 0;

  virtual std::string getKernelName() const = 0;

protected:
  cudaError_t checkLaunchError(cudaError_t launchStatus, const std::string& kernelName) {
    if (launchStatus != cudaSuccess) {
      std::cerr << "Error launching " << kernelName << ": " << cudaGetErrorString(launchStatus) << std::endl;
      return launchStatus;
    }
    cudaError_t asyncError = cudaGetLastError();
    if (asyncError != cudaSuccess) {
      std::cerr << "Error in " << kernelName << ": " << cudaGetErrorString(asyncError) << std::endl;
      return asyncError;
    }
    return cudaSuccess;
  }
};

#endif // SGEMM_CUH