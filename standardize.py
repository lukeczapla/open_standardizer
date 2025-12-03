from __future__ import annotations

from typing import List, Optional

from rdkit import Chem

from .xml_loader import load_xml_actions
from .gpu_cpu_policy_manager import (
    GPU_CPU_MANAGER,
    Policy,
    try_gpu_standardize,
)
from .cpu_ops import cpu_execute
from .batch_engine import BatchStandardizer
from .stereo_export import export_curly_block_from_mol
from .enhanced_smiles import enhanced_smiles_to_mol, ChemAxonMeta, parse_chemaxon_enhanced


# ChemAxon-equivalent canonicalization pipeline (XML-matched)
DEFAULT_OPS: List[str] = [
    "clear_stereo",
    "remove_largest_fragment",
    "remove_attached_data",
    "remove_atom_values",
    "remove_explicit_h",
    "clear_isotopes",
    "neutralize",
    "mesomerize",
    "tautomerize",
    "aromatize",
]


class Standardizer:
    """
    Orchestrates standardization:

      - loads ChemAxon-style XML configs
      - applies actions in order
      - tries GPU first per operation
      - falls back to RDKit/MolVS CPU ops

    NOTE:
      This class itself operates on *plain SMILES / Mol*.
      Enhanced SMILES (with '{...}') should go through the
      helper wrappers below (standardize_smiles / standardize).
    """

    def __init__(
        self,
        xml_config_path: Optional[str] = None,
        policy: Optional[Policy] = None,
    ) -> None:
        self.xml_config_path = xml_config_path
        self.policy: Policy = policy or GPU_CPU_MANAGER

        self.actions: List[dict] = []
        if xml_config_path:
            self.actions = load_xml_actions(xml_config_path)

    def set_xml(self, xml_path: str) -> None:
        self.xml_config_path = xml_path
        self.actions = load_xml_actions(xml_path)

    def _apply_op(self, op_name: str, smiles: str) -> Optional[str]:
        """
        Apply a single operation *to plain SMILES*:

          1. Try GPU via policy.try_gpu(op_name)
          2. If that fails or is unavailable → CPU via cpu_execute

        Returns:
            - new SMILES string on success
            - None on fatal failure
        """
        # 1. GPU path
        gpu_fn = self.policy.try_gpu(op_name)
        if gpu_fn is not None:
            gpu_out = try_gpu_standardize(smiles, [op_name], self.policy, gpu_fn)
            if isinstance(gpu_out, str):
                return gpu_out

        # 2. CPU fallback
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        return cpu_execute(op_name, mol)  # expected to return SMILES

    def standardize(self, mol: Optional[Chem.Mol]) -> Optional[Chem.Mol]:
        """
        Run the full XML-driven pipeline on a single RDKit Mol.

        IMPORTANT:
          - This function does *not* know about ChemAxon curly metadata.
          - Use the string-based helpers for "enhanced SMILES in / out".
        """
        if mol is None:
            return None

        smiles = Chem.MolToSmiles(mol)

        if self.actions:
            for entry in self.actions:
                op_name = entry["op"]
                smiles = self._apply_op(op_name, smiles)
                if smiles is None:
                    return None

        return Chem.MolFromSmiles(smiles)

    @staticmethod
    def standardize_many(
        smiles_list: List[str],
        ops: Optional[List[str]] = None,
    ) -> List[Optional[str]]:
        """
        Batch standardization using DEFAULT_OPS and the global GPU/CPU manager.

        INPUT:
          - smiles_list: may be *enhanced SMILES* with ChemAxon-style tails
                        (e.g. 'c1ccccc1[NH2+] {A7=s;A9=r;o1:7,9}')

        BEHAVIOR:
          - Strips curly metadata per input → passes core SMILES into the
            BatchStandardizer.
          - Re-attaches the exact same curly block to each successful output.
        """
        ops = ops or DEFAULT_OPS

        # Parse enhanced → core for the batch
        parsed_list = [parse_chemaxon_enhanced(s) for s in smiles_list]
        core_list = [p.core_smiles for p in parsed_list]

        engine = BatchStandardizer(GPU_CPU_MANAGER)
        core_out_list = engine.run(core_list, ops)

        results: List[Optional[str]] = []
        for parsed, core_out in zip(parsed_list, core_out_list):
            if core_out is None:
                results.append(None)
                continue

            if parsed.meta:
                # Re-attach original curly block EXACTLY as given
                results.append(f"{core_out} {{{parsed.meta.to_raw()}}}")
            else:
                results.append(core_out)

        return results


def standardize_smiles(
    smiles: str,
    xml_path: str,
    preserve_chemaxon_meta: bool = True,
) -> Optional[str]:
    """
    Convenience wrapper: load XML, standardize a single *enhanced* SMILES
    and return SMILES.

    INPUT:
      smiles = "core_smiles {A7=s;A9=r;o1:7,9}"  (ChemAxon-style)
             or just "core_smiles"

    BEHAVIOR:
      - Parses 'smiles' into core + curly metadata.
      - Runs XML-driven pipeline on the *core* using Standardizer.
      - Re-attaches the original curly block if preserve_chemaxon_meta=True.
    """
    parsed = parse_chemaxon_enhanced(smiles)
    core = parsed.core_smiles

    mol = Chem.MolFromSmiles(core)
    if mol is None:
        return None

    std = Standardizer(xml_path)
    out_mol = std.standardize(mol)
    if out_mol is None:
        return None

    core_out = Chem.MolToSmiles(out_mol)

    if preserve_chemaxon_meta and parsed.meta:
        return f"{core_out} {{{parsed.meta.to_raw()}}}"

    return core_out


def standardize(
    smiles: str,
    ops: List[str] = DEFAULT_OPS,
    policy: Policy = GPU_CPU_MANAGER,
    preserve_chemaxon_meta: bool = True,
    regenerate_chemaxon_meta: bool = False,
    index_base: int = 0,
) -> Optional[str]:
    """
    Functional API (no XML), *enhanced SMILES aware*:

      - Takes an enhanced SMILES string like:
          "c1ccccc1[NH2+] {A7=s;A9=r;o1:7,9}"

      - Uses enhanced_smiles_to_mol(...) to:
          * parse curly metadata
          * construct Mol
          * apply ChemAxon StereoGroups + atom assignments

      - Converts Mol to core SMILES for per-op GPU/CPU pipeline
      - At the end:
          * if regenerate_chemaxon_meta=True:
              - rebuilds a curly block from RDKit (CIP + groups)
          * elif preserve_chemaxon_meta=True and original meta exists:
              - reattaches the original curly block verbatim
          * else:
              - returns plain SMILES
    """
    mol, meta = enhanced_smiles_to_mol(smiles, index_base=index_base)
    if mol is None:
        return None

    current = Chem.MolToSmiles(mol)

    for op_name in ops:
        gpu_fn = policy.try_gpu(op_name)
        if gpu_fn is not None:
            gpu_out = try_gpu_standardize(current, [op_name], policy, gpu_fn)
            if isinstance(gpu_out, str):
                current = gpu_out
                continue

        mol = Chem.MolFromSmiles(current)
        if mol is None:
            return None

        current = policy.cpu_ops(op_name, mol)

    if regenerate_chemaxon_meta:
        out_mol = Chem.MolFromSmiles(current)
        if out_mol is None:
            return None
        new_block = export_curly_block_from_mol(
            out_mol, existing_meta=meta, index_base=index_base
        )
        if new_block:
            return f"{current} {{{new_block}}}"
        return current

    if preserve_chemaxon_meta and meta and meta.raw.strip():
        return f"{current} {{{meta.to_raw()}}}"

    return current