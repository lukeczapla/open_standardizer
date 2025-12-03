"""
Auto-register GPU kernels if pybind11 / CUDA modules are importable.
"""

from .standardize_gpu import register_gpu_kernel

try:
    # These come from pybind11-wrapped CUDA objects
    from rdkit_standardizer_gpu import (
        gpu_stereo,
        gpu_charge,
        gpu_bond_order,
        gpu_aromaticity,
        gpu_mesomerizer,
        gpu_final_clean,
    )

    register_gpu_kernel("stereo", gpu_stereo)
    register_gpu_kernel("charge", gpu_charge)
    register_gpu_kernel("bond_order", gpu_bond_order)
    register_gpu_kernel("aromaticity", gpu_aromaticity)
    register_gpu_kernel("mesomerizer", gpu_mesomerizer)
    register_gpu_kernel("final_clean", gpu_final_clean)

except Exception as e:
    # No CUDA module installed — this is fine.
    print(f"[INFO] GPU kernels unavailable: {e}")
