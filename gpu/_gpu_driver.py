from __future__ import annotations

from typing import Callable, List, Optional

from ..gpu_exceptions import GPUNotAvailable, GPUStepFailed

try:
    # Compiled pybind11+CUDA extension, built from src/gpu/*.cpp/*.cu
    import standardize_gpu as _lib
except ImportError:  # extension not built / not on PYTHONPATH
    _lib = None


# -------------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------------

def _ensure_lib(op_name: str) -> None:
    """
    Ensure the compiled GPU extension is available. If not, raise
    GPUNotAvailable so the Policy can fall back to CPU cleanly.
    """
    if _lib is None:
        raise GPUNotAvailable(
            f"GPU library 'standardize_gpu' not available for op '{op_name}'"
        )


def _get_symbol(sym_name: str, op_name: str) -> Callable[[str], str]:
    """
    Look up a function in the compiled module, mapping missing symbols
    to GPUNotAvailable so Policy can fall back to CPU.
    """
    _ensure_lib(op_name)
    fn = getattr(_lib, sym_name, None)
    if fn is None:
        raise GPUNotAvailable(
            f"GPU symbol '{sym_name}' missing in 'standardize_gpu' for op '{op_name}'"
        )
    return fn


def _wrap_call(op_name: str, fn: Callable[[str], str], smiles: str, pipeline: List[str]) -> str:
    """
    Uniform error mapping:

      - GPUNotAvailable  → propagate as-is
      - anything else    → GPUStepFailed

    `pipeline` is accepted to match the Policy signature but not used
    by the current kernels.
    """
    try:
        return fn(smiles)
    except GPUNotAvailable:
        # Let Policy treat this as a soft GPU failure.
        raise
    except Exception as e:
        # Anything else is a GPU step failure.
        raise GPUStepFailed(f"GPU op '{op_name}' failed: {e}") from e


# -------------------------------------------------------------------------
# Public per-op "run" functions (smiles, pipeline) -> smiles
# These are what your gpu/core_ops / gpu_ops wrappers should import.
# -------------------------------------------------------------------------

def gpu_stereo_run(smiles: str, pipeline: List[str]) -> str:
    op_name = "stereo"
    fn = _get_symbol("gpu_stereo", op_name)
    return _wrap_call(op_name, fn, smiles, pipeline)


def gpu_charge_run(smiles: str, pipeline: List[str]) -> str:
    op_name = "charge"
    fn = _get_symbol("gpu_charge", op_name)
    return _wrap_call(op_name, fn, smiles, pipeline)


def gpu_bond_order_run(smiles: str, pipeline: List[str]) -> str:
    op_name = "bond_order"
    fn = _get_symbol("gpu_bond_order", op_name)
    return _wrap_call(op_name, fn, smiles, pipeline)


def gpu_aromatize_run(smiles: str, pipeline: List[str]) -> str:
    op_name = "aromatize"
    # C++ side exports gpu_aromaticity
    fn = _get_symbol("gpu_aromaticity", op_name)
    return _wrap_call(op_name, fn, smiles, pipeline)


def gpu_mesomerize_run(smiles: str, pipeline: List[str]) -> str:
    op_name = "mesomerize"
    fn = _get_symbol("gpu_mesomerizer", op_name)
    return _wrap_call(op_name, fn, smiles, pipeline)


def gpu_final_clean_run(smiles: str, pipeline: List[str]) -> str:
    op_name = "final_clean"
    fn = _get_symbol("gpu_final_clean", op_name)
    return _wrap_call(op_name, fn, smiles, pipeline)


def gpu_clear_isotopes_run(smiles: str, pipeline: List[str]) -> str:
    op_name = "clear_isotopes"
    fn = _get_symbol("gpu_clear_isotopes", op_name)
    return _wrap_call(op_name, fn, smiles, pipeline)


def gpu_clear_stereo_run(smiles: str, pipeline: List[str]) -> str:
    """
    Core "clear stereo" op. C++ exports this as `gpu_clear_stereo_core`.
    """
    op_name = "clear_stereo"
    fn = _get_symbol("gpu_clear_stereo_core", op_name)
    return _wrap_call(op_name, fn, smiles, pipeline)


def gpu_keep_largest_fragment_run(smiles: str, pipeline: List[str]) -> str:
    op_name = "keep_largest_fragment"
    fn = _get_symbol("gpu_keep_largest_fragment", op_name)
    return _wrap_call(op_name, fn, smiles, pipeline)


def gpu_remove_explicit_h_run(smiles: str, pipeline: List[str]) -> str:
    op_name = "remove_explicit_h"
    fn = _get_symbol("gpu_remove_explicit_h", op_name)
    return _wrap_call(op_name, fn, smiles, pipeline)


def gpu_tautomerize_run(smiles: str, pipeline: List[str]) -> str:
    op_name = "tautomerize"
    fn = _get_symbol("gpu_tautomerize", op_name)
    return _wrap_call(op_name, fn, smiles, pipeline)
