pybind11_add_module(rdkit_standardizer_gpu
    gpu/standardize_bindings.cpp
    src/gpu/kernels.cu
    src/gpu/stereo.cu
    src/gpu/aromaticity.cu
    src/gpu/mesomerizer.cu
    src/gpu/bond_order.cu
    src/gpu/charge.cu
    src/gpu/final_clean.cu
)

target_link_libraries(rdkit_standardizer_gpu
    PRIVATE
        RDKit::RDKit
        CUDA::cudart
)