# rdkit_standardizer/gpu/standardize_gpu.py

"""
GPU-accelerated standardization dispatcher.

This module connects Python to the CUDA kernels exposed by the
pybind11 module `rdkit_standardizer_gpu`:

    gpu_stereo(smiles)       -> smiles
    gpu_charge(smiles)       -> smiles
    gpu_bond_order(smiles)   -> smiles
    gpu_aromaticity(smiles)  -> smiles
    gpu_mesomerizer(smiles)  -> smiles
    gpu_final_clean(smiles)  -> smiles

Internal API here is Mol-based, so we:
    Mol -> SMILES -> C++/CUDA -> SMILES -> Mol
"""

from typing import Callable, Dict, Iterable, List, Optional
from rdkit import Chem

# Try to import the compiled CUDA extension.
# If missing, we degrade to pure RDKit behavior (kernels become no-ops).
try:
    import rdkit_standardizer_gpu as _gpu
except ImportError:
    _gpu = None

# Map logical step name -> function(mol) -> mol
GPU_MODULES: Dict[str, Callable[[Chem.Mol], Chem.Mol]] = {}


def register_gpu_kernel(name: str, fn: Callable[[Chem.Mol], Chem.Mol]) -> None:
    """
    Register a Mol -> Mol GPU kernel under a logical step name.
    Example: register_gpu_kernel("aromaticity", _aromaticity_kernel)
    """
    GPU_MODULES[name] = fn


# -------------------------------------------------------------------------
# Helpers: Mol <-> SMILES wrappers for the C++/CUDA functions
# -------------------------------------------------------------------------

def _mol_to_smiles(mol: Chem.Mol) -> str:
    # Non-canonical here; final canonicalization happens at the end.
    return Chem.MolToSmiles(mol, canonical=False)


def _smiles_to_mol(smi: str) -> Optional[Chem.Mol]:
    return Chem.MolFromSmiles(smi)


def _wrap_gpu_fn(fn_name: str):
    """
    Given the name of a C++ function (e.g. 'gpu_stereo'), return a
    Mol->Mol wrapper that:
        Mol -> SMILES -> _gpu.fn(SMILES) -> SMILES -> Mol

    If the extension or function is missing, this becomes identity.
    """
    if _gpu is None or not hasattr(_gpu, fn_name):
        # No GPU available: identity transform
        def _identity(m: Chem.Mol) -> Chem.Mol:
            return m
        return _identity

    gpu_fn = getattr(_gpu, fn_name)

    def _wrapped(mol: Chem.Mol) -> Chem.Mol:
        if mol is None:
            return None
        smi = _mol_to_smiles(mol)
        out_smi = gpu_fn(smi)  # call into pybind11 / CUDA
        out_mol = _smiles_to_mol(out_smi)
        # If GPU returned nonsense, fall back to original mol
        return out_mol if out_mol is not None else mol

    return _wrapped


# -------------------------------------------------------------------------
# Concrete kernels wired to C++ exports
# -------------------------------------------------------------------------

# stereo.cu
_stereo_kernel       = _wrap_gpu_fn("gpu_stereo")
# charge_normalize.cu
_charge_kernel       = _wrap_gpu_fn("gpu_charge")
# bond_infer.cu
_bond_order_kernel   = _wrap_gpu_fn("gpu_bond_order")
# aromaticity / core_ops.cu
_aromaticity_kernel  = _wrap_gpu_fn("gpu_aromaticity")
# mesomerizer.cu (the one you pasted)
_mesomerizer_kernel  = _wrap_gpu_fn("gpu_mesomerizer")
# optional final cleanup
_final_clean_kernel  = _wrap_gpu_fn("gpu_final_clean")


def _register_default_kernels() -> None:
    """
    Register the CUDA-backed kernels under logical step names.

    These step names must match what your gpu_ops / Policy expect.
    """
    register_gpu_kernel("stereo",       _stereo_kernel)
    register_gpu_kernel("charge",       _charge_kernel)
    register_gpu_kernel("bond_order",   _bond_order_kernel)
    register_gpu_kernel("aromaticity",  _aromaticity_kernel)
    register_gpu_kernel("mesomerizer",  _mesomerizer_kernel)
    register_gpu_kernel("final_clean",  _final_clean_kernel)


_register_default_kernels()


# -------------------------------------------------------------------------
# Pipeline runner and public function
# -------------------------------------------------------------------------

DEFAULT_PIPELINE: List[str] = [
    "stereo",
    "charge",
    "bond_order",
    "aromaticity",
    "mesomerizer",
    "final_clean",
]


def _run_pipeline(
    mol: Chem.Mol,
    steps: Optional[Iterable[str]] = None,
) -> Chem.Mol:
    if mol is None:
        return None

    pipeline = list(steps) if steps is not None else DEFAULT_PIPELINE

    for step in pipeline:
        kernel = GPU_MODULES.get(step)
        if kernel is None:
            continue
        try:
            mol = kernel(mol)
        except Exception as e:
            # Soft-fail: keep going but mark the problem
            print(f"[GPU WARNING] kernel '{step}' failed → skipping: {e}")
    return mol


def gpu_standardize(smiles: str, steps: Optional[Iterable[str]] = None) -> str:
    """
    GPU standardizer entry point.

    - steps is None  → run the full canonicalization pipeline
    - steps is ['stereo'] → run only that step (used by per-op policy).

    Returns canonical SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES for GPU: {smiles}")

    mol = _run_pipeline(mol, steps=steps)
    if mol is None:
        raise ValueError(f"GPU pipeline returned None for SMILES: {smiles}")

    # Final canonical SMILES (RDKit)
    return Chem.MolToSmiles(mol, canonical=True)
