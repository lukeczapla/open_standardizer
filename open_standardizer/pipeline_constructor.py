from typing import Iterable, List, Optional

from rdkit import Chem

from .gpu_cpu_policy_manager import GPUCPUPolicyManager
from .standardize import (
    Standardizer,
    standardize as standardize_smiles_pipeline,
    DEFAULT_OPS,
)


class PipelineConstructor:
    """
    Thin orchestrator around Standardizer + GPUCPUPolicyManager.

    Responsibilities:
      - Build a policy (GPU/CPU settings)
      - Optionally load a ChemAxon-style XML config
      - Provide simple helpers to run the pipeline on SMILES or RDKit mols
    """

    def __init__(
        self,
        xml_path: Optional[str] = None,
        enable_gpu: bool = True,
        prefer_gpu: bool = True,
    ) -> None:
        # Create the GPU/CPU policy manager used by everything
        self.policy = GPUCPUPolicyManager(
            enable_gpu=enable_gpu,
            prefer_gpu=prefer_gpu,
        )

        # Standardizer optionally uses the XML file
        self.xml_path = xml_path
        self.std = Standardizer(xml_config_path=xml_path, policy=self.policy)

        # If XML present, ops come from XML; otherwise DEFAULT_OPS
        if self.std.actions:
            self.ops: List[str] = [entry["op"] for entry in self.std.actions]
        else:
            self.ops = list(DEFAULT_OPS)

    # ------------------------------------------------------------------
    # Single-molecule APIs
    # ------------------------------------------------------------------

    def run_on_smiles(self, smiles: str) -> Optional[str]:
        """
        Standardize a single SMILES string using the configured pipeline.
        Uses the functional API so per-op GPU/CPU routing is applied.
        """
        return standardize_smiles_pipeline(smiles, ops=self.ops, policy=self.policy)

    def run_on_mol(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """
        Standardize a single RDKit Mol using the Standardizer instance.
        """
        return self.std.standardize(mol)

    # ------------------------------------------------------------------
    # Batch APIs
    # ------------------------------------------------------------------

    def run_on_smiles_batch(self, smiles_list: Iterable[str]) -> List[Optional[str]]:
        """
        Simple batch wrapper: apply `run_on_smiles` to each element.
        """
        out: List[Optional[str]] = []
        for s in smiles_list:
            out.append(self.run_on_smiles(s))
        return out
