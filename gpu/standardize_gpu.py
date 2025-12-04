"""
GPU-accelerated per-op standardization wrappers for open-standardizer.

Each function here has the signature:

    gpu_<op_name>(smiles: str, pipeline: List[str]) -> str

Under the hood they delegate to `_gpu_driver`, which talks to the
compiled pybind11+CUDA module `standardize_gpu` (built from src/gpu).

C++/CUDA exports in `standardize_gpu`:

    gpu_stereo(smiles)               -> smiles
    gpu_charge(smiles)               -> smiles
    gpu_bond_order(smiles)           -> smiles
    gpu_aromaticity(smiles)          -> smiles
    gpu_mesomerizer(smiles)          -> smiles
    gpu_tautomerize(smiles)          -> smiles
    gpu_final_clean(smiles)          -> smiles
    gpu_clear_isotopes(smiles)       -> smiles
    gpu_clear_stereo_core(smiles)    -> smiles
    gpu_keep_largest_fragment(smiles)-> smiles
    gpu_remove_explicit_h(smiles)    -> smiles

The wrappers here:

  - Accept and return SMILES strings.
  - Accept a `pipeline` parameter to match the Policy signature
    (currently unused by the CUDA functions but preserved for future
     multi-step kernels).
  - Propagate GPUNotAvailable / GPUStepFailed so the Policy layer
    can decide when to fall back to CPU.
"""

from __future__ import annotations

from typing import List

from ..gpu_exceptions import GPUNotAvailable, GPUStepFailed
from ._gpu_driver import (
    gpu_stereo_run,
    gpu_charge_run,
    gpu_bond_order_run,
    gpu_aromatize_run,
    gpu_mesomerize_run,
    gpu_final_clean_run,
    gpu_clear_isotopes_run,
    gpu_clear_stereo_run,
    gpu_keep_largest_fragment_run,
    gpu_remove_explicit_h_run,
    gpu_tautomerize_run,
)


# -------------------------------------------------------------------------
# Core stereo / charge / bond-order / aromaticity pipeline pieces
# -------------------------------------------------------------------------

def gpu_stereo(smiles: str, pipeline: List[str]) -> str:
    """
    GPU-backed 'stereo' operation.

    Runs the stereo-normalization kernel (e.g. CIP flag assignment /
    cleanup) on the GPU.
    """
    try:
        return gpu_stereo_run(smiles, pipeline)
    except (GPUNotAvailable, GPUStepFailed):
        raise


def gpu_charge(smiles: str, pipeline: List[str]) -> str:
    """
    GPU-backed 'charge' (charge-normalization) operation.
    """
    try:
        return gpu_charge_run(smiles, pipeline)
    except (GPUNotAvailable, GPUStepFailed):
        raise


def gpu_bond_order(smiles: str, pipeline: List[str]) -> str:
    """
    GPU-backed 'bond_order' inference / normalization operation.
    """
    try:
        return gpu_bond_order_run(smiles, pipeline)
    except (GPUNotAvailable, GPUStepFailed):
        raise


def gpu_aromatize(smiles: str, pipeline: List[str]) -> str:
    """
    GPU-backed 'aromatize' operation.

    Wraps `standardize_gpu.gpu_aromaticity`, which runs your GPU
    aromaticity pass and returns a SMILES.
    """
    try:
        return gpu_aromatize_run(smiles, pipeline)
    except (GPUNotAvailable, GPUStepFailed):
        raise


def gpu_mesomerize(smiles: str, pipeline: List[str]) -> str:
    """
    GPU-backed 'mesomerize' operation.

    Wraps the mesomerization / resonance canonicalization kernel
    (`standardize_gpu.gpu_mesomerizer`).
    """
    try:
        return gpu_mesomerize_run(smiles, pipeline)
    except (GPUNotAvailable, GPUStepFailed):
        raise


def gpu_tautomerize(smiles: str, pipeline: List[str]) -> str:
    """
    GPU-backed 'tautomerize' operation.

    This calls the CUDA-backed tautomerization kernel exposed as
    `standardize_gpu.gpu_tautomerize`, then returns a SMILES string
    after tautomer normalization.
    """
    try:
        return gpu_tautomerize_run(smiles, pipeline)
    except (GPUNotAvailable, GPUStepFailed):
        raise


def gpu_final_clean(smiles: str, pipeline: List[str]) -> str:
    """
    GPU-backed 'final_clean' operation.

    Typically used as the last step of the GPU pipeline to do any
    final cleanup / normalization after the main passes.
    """
    try:
        return gpu_final_clean_run(smiles, pipeline)
    except (GPUNotAvailable, GPUStepFailed):
        raise


# -------------------------------------------------------------------------
# Fragment / H / isotope helpers
# -------------------------------------------------------------------------

def gpu_clear_isotopes(smiles: str, pipeline: List[str]) -> str:
    """
    GPU-backed 'clear_isotopes' operation.

    Semantics should match the CPU op_clear_isotopes, but executed via
    the CUDA kernel exported from `standardize_gpu.gpu_clear_isotopes`.
    """
    try:
        return gpu_clear_isotopes_run(smiles, pipeline)
    except (GPUNotAvailable, GPUStepFailed):
        raise


def gpu_clear_stereo(smiles: str, pipeline: List[str]) -> str:
    """
    GPU-backed 'clear_stereo' operation.

    This ultimately calls `standardize_gpu.gpu_clear_stereo_core`,
    which should clear stereochemistry consistently with the CPU path.
    """
    try:
        return gpu_clear_stereo_run(smiles, pipeline)
    except (GPUNotAvailable, GPUStepFailed):
        raise


def gpu_keep_largest_fragment(smiles: str, pipeline: List[str]) -> str:
    """
    GPU-backed 'remove_largest_fragment' / 'keep_largest_fragment' operation.

    Used for ChemAxon-style "keep largest fragment" salt stripping.
    """
    try:
        return gpu_keep_largest_fragment_run(smiles, pipeline)
    except (GPUNotAvailable, GPUStepFailed):
        raise


def gpu_remove_explicit_h(smiles: str, pipeline: List[str]) -> str:
    """
    GPU-backed 'remove_explicit_h' operation.

    Should mirror the semantics of the CPU op_remove_explicit_h (RDKit
    RemoveHs), but performed via the CUDA kernel.
    """
    try:
        return gpu_remove_explicit_h_run(smiles, pipeline)
    except (GPUNotAvailable, GPUStepFailed):
        raise


__all__ = [
    # core pipeline pieces
    "gpu_stereo",
    "gpu_charge",
    "gpu_bond_order",
    "gpu_aromatize",
    "gpu_mesomerize",
    "gpu_tautomerize",
    "gpu_final_clean",
    # helpers
    "gpu_clear_isotopes",
    "gpu_clear_stereo",
    "gpu_keep_largest_fragment",
    "gpu_remove_explicit_h",
]
