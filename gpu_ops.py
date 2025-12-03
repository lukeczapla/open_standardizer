# molvs_standardizer/gpu_ops.py

from typing import Callable, List
from .gpu_cpu_policy_manager import Policy
from .gpu_cpu_policy_manager import try_gpu_standardize  # if you have this helper
from .gpu.standardize_gpu import gpu_standardize

# If you already created the global manager in __init__.py:
# from . import GPU_CPU_MANAGER
# Otherwise you can create/use one here:
# GPU_CPU_MANAGER: Policy = Policy(...)


def _make_single_step_gpu(step_name: str) -> Callable[[str, List[str]], str]:
    """
    Wrap gpu_standardize() so it looks like a per-op GPU function:

        fn(smiles: str, pipeline: list[str]) -> str

    The 'pipeline' arg is ignored here; just run the single GPU step.
    """
    def _fn(smiles: str, pipeline: List[str]) -> str:
        return gpu_standardize(smiles, steps=[step_name])
    return _fn


def register_gpu_ops(policy: Policy) -> None:
    """
    Populate policy.gpu_ops with per-operation GPU handlers.
    Op names here must match whatever your CPU ops / XML mapping uses.
    """

    # Map high-level op names → GPU step names
    # Adjust keys to match your cpu_ops naming!
    policy.gpu_ops.update({
        # ChemAxon "ClearStereo" equivalent
        "clear_stereo": _make_single_step_gpu("stereo"),

        # Charge normalization
        "charge_normalize": _make_single_step_gpu("charge"),

        # Bond-order inference
        "bond_order_infer": _make_single_step_gpu("bond_order"),

        # Aromaticity perception
        "aromatize": _make_single_step_gpu("aromaticity"),

        # Mesomerize / resonance normalization
        "mesomerize": _make_single_step_gpu("mesomerizer"),
    })

    # Optionally: mark which ops are GPU-friendly / CPU-only
    if hasattr(policy, "gpu_required_ops"):
        policy.gpu_required_ops.update({
            "clear_stereo",
            "aromatize",
            "mesomerize",
            "charge_normalize",
            "bond_order_infer",
        })

    if hasattr(policy, "cpu_only_ops"):
        policy.cpu_only_ops.update({
            # If there are ops you explicitly never want on GPU:
            # "neutralize",
            # "tautomerize",
        })
