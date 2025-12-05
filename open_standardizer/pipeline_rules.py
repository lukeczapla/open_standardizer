from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class CapabilityMatrix:
    """
    Defines which operations are supported by GPU.
    """
    gpu_ops: Dict[str, bool] = field(default_factory=lambda: {
        "clear_stereo": True,
        "remove_fragments": True,
        "remove_explicit_h": True,
        "clear_isotopes": True,
        "neutralize": True,
        "mesomerize": True,
        "tautomerize": True,
        "aromatize": True,
        "charge_norm": True,
        "bond_order": True,
    })


@dataclass
class PipelineRules:
    """
    Defines pipeline ordering and GPU/CPU assignment policy.
    """
    order: List[str] = field(default_factory=lambda: [
        "remove_fragments",
        "clear_stereo",
        "remove_explicit_h",
        "clear_isotopes",
        "neutralize",
        "charge_norm",
        "bond_order",
        "mesomerize",
        "aromatize",
        "tautomerize",
    ])

    capability: CapabilityMatrix = field(default_factory=CapabilityMatrix)

    def classify_ops(self, use_gpu=True):
        """
        Returns:
          gpu_list   : ops to run on GPU
          cpu_list   : ops to run on CPU
          fallback   : GPU-supported ops that have CPU backup
        """
        gpu_list = []
        cpu_list = []
        fallback = []

        for op in self.order:
            gpu_supported = self.capability.gpu_ops.get(op, False)

            if use_gpu and gpu_supported:
                gpu_list.append(op)
                fallback.append(op)
            else:
                cpu_list.append(op)

        return gpu_list, cpu_list, fallback

    def pipeline_for_env(self, use_gpu=True):
        gpu_list, cpu_list, fallback = self.classify_ops(use_gpu)

        return {
            "gpu_first": gpu_list,
            "cpu_first": cpu_list,
            "fallback_ops": fallback,
            "full_order": self.order,
        }
