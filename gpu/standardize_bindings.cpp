#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

void launch_normalize(pybind11::array_t<float> arr);

void launch_normalize(pybind11::array_t<float> arr) {
    auto buf = arr.request();
    float* ptr = (float*)buf.ptr;
    int n = buf.size;

    float *d;
    cudaMalloc(&d, n*sizeof(float));
    cudaMemcpy(d, ptr, n*sizeof(float), cudaMemcpyHostToDevice);

    normalizeAtomsKernel<<<(n+255)/256,256>>>(d,n);
    cudaDeviceSynchronize();

    cudaMemcpy(ptr, d, n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d);
}

PYBIND11_MODULE(rdkit_gpu_std, m) {
    m.def("normalize_atoms", &launch_normalize);
}
