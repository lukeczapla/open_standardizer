cmake_minimum_required(VERSION 3.20)

project(open_standardizer_gpu LANGUAGES CXX CUDA)

# You probably already do this at a higher level, but just in case:
find_package(pybind11 REQUIRED)
find_package(RDKit REQUIRED)
find_package(CUDAToolkit REQUIRED)   # gives CUDA::cudart

# IMPORTANT: Name here must match PYBIND11_MODULE(...) and your Python import
#   PYBIND11_MODULE(standardize_gpu, m) { ... }
#   import standardize_gpu
pybind11_add_module(standardize_gpu
    src/gpu/standardize_bindings.cpp

    # existing CUDA sources
    src/gpu/kernels.cu
    src/gpu/stereo.cu
    src/gpu/aromaticity.cu
    src/gpu/mesomerizer.cu
    src/gpu/bond_order.cu
    src/gpu/charge.cu
    src/gpu/final_clean.cu

    # new CUDA / C++ sources that make core_ops actually usable
    src/gpu/core_ops.cu
    src/gpu/tautomerize.cu      # ← include this if you created tautomerize.cu
    src/gpu/core_ops_batch.cpp  # CPU-side batch builder you already had
)

target_link_libraries(standardize_gpu
    PRIVATE
        RDKit::RDKit
        CUDA::cudart
)

# Optional but usually nice:
target_include_directories(standardize_gpu
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu
)

set_target_properties(standardize_gpu PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)