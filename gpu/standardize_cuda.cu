#include <cuda_runtime.h>

extern "C" __global__
void normalizeAtomsKernel(float *coords, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) coords[i] = coords[i] * 0.5f; // dummy normalization
}
