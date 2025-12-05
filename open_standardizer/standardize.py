from __future__ import annotations

from typing import List, Optional, Union

from rdkit import Chem

from .xml_loader import load_xml_actions
from .gpu_cpu_policy_manager import (
    GPU_CPU_MANAGER,
    Policy,
    try_gpu_standardize,
)
from .cpu_ops import cpu_execute
from .batch_engine import BatchStandardizer
from .stereo_export import export_curly_block_from_mol, export_enhanced_smiles_from_mol

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


def _apply_ops_to_smiles(
    smiles: str,
    ops: List[str],
    policy: Policy,
) -> Optional[str]:
    """
    Core SMILES → SMILES transformation loop used by both
    `standardize(...)` and `standardize_mol(...)`.

    Takes *plain* SMILES (no curly metadata) and a list of ops.
    """
    current = smiles

    for op_name in ops:
        # 1) GPU attempt
        gpu_fn = policy.try_gpu(op_name)
        if gpu_fn is not None:
            gpu_out = try_gpu_standardize(current, [op_name], policy, gpu_fn)
            if isinstance(gpu_out, str):
                current = gpu_out
                continue

        # 2) CPU fallback
        mol = Chem.MolFromSmiles(current)
        if mol is None:
            return None

        # policy.cpu_ops is expected to return a SMILES string
        current = policy.cpu_ops(op_name, mol)
        if not isinstance(current, str):
            return None

    return current



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


def standardize_mol(
    mol: Optional[Chem.Mol],
    ops: List[str] = DEFAULT_OPS,
    policy: Policy = GPU_CPU_MANAGER,
) -> Optional[Chem.Mol]:
    """
    Functional Mol→Mol API (no XML, no ChemAxon curly metadata):

      - Takes an RDKit Mol.
      - Runs the same op sequence as `standardize(...)` (DEFAULT_OPS by default).
      - Uses GPU first (per op) via `policy.try_gpu`, then falls back to
        `policy.cpu_ops(op_name, mol)` on failure.
      - Returns a *new* Mol (no in-place mutation is guaranteed).

    This is essentially the Mol-level equivalent of:

        standardize(smiles, ops=..., policy=...)

    but without any enhanced SMILES parsing or curly-block handling.
    """
    if mol is None:
        return None

    core_smiles = Chem.MolToSmiles(mol)
    core_out = _apply_ops_to_smiles(core_smiles, ops, policy)
    if core_out is None:
        return None

    return Chem.MolFromSmiles(core_out)


def standardize_mol_xml(
    mol: Optional[Chem.Mol],
    xml_path: str,
    policy: Policy = GPU_CPU_MANAGER,
) -> Optional[Chem.Mol]:
    if mol is None:
        return None
    std = Standardizer(xml_path, policy=policy)
    return std.standardize(mol)


def standardize_molblock(
    molblock: str,
    ops: List[str] = DEFAULT_OPS,
    policy: Policy = GPU_CPU_MANAGER,
    return_molblock: bool = True,
    v3000: bool = False,
) -> Optional[str]:
    """
    Standardize a molfile (MolBlock string) and return either:

      - a new MolBlock (default), or
      - a canonical SMILES.

    Notes:
      - This does *not* handle ChemAxon enhanced SMILES curly metadata;
        it just uses the structure in the molfile.
    """
    if not molblock.strip():
        return None

    mol = Chem.MolFromMolBlock(molblock, sanitize=True)
    if mol is None:
        return None

    out_mol = standardize_mol(mol, ops=ops, policy=policy)
    if out_mol is None:
        return None

    if return_molblock:
        return Chem.MolToMolBlock(out_mol, forceV3000=v3000)

    return Chem.MolToSmiles(out_mol)


def standardize_enhanced_smiles_to_molblock(
    smiles: str,
    ops: List[str] = DEFAULT_OPS,
    policy: Policy = GPU_CPU_MANAGER,
    index_base: int = 0,
    v3000: bool = False,
) -> Optional[str]:
    """
    Enhanced SMILES → standardized molfile.

    - Uses enhanced_smiles_to_mol to apply ChemAxon-style metadata
      to the RDKit Mol (StereoGroups, A/B assignments).
    - Runs the same op list as `standardize(...)`.
    - Returns a MolBlock string (V2000 by default, V3000 if requested).

    NOTE:
      The curly metadata itself isn’t re-encoded into the molfile; the
      info that is mapped to RDKit’s stereochemistry will be reflected
      in the structure (stereo wedges, bond stereo, groups), the rest
      stays in atom/bond props on the Mol object.
    """
    mol, _meta = enhanced_smiles_to_mol(smiles, index_base=index_base)
    if mol is None:
        return None

    out_mol = standardize_mol(mol, ops=ops, policy=policy)
    if out_mol is None:
        return None

    return Chem.MolToMolBlock(out_mol, forceV3000=v3000)


MoleculeLike = Union[str, Chem.Mol]

def standardize_any(
    x: MoleculeLike,
    ops: List[str] = DEFAULT_OPS,
    policy: Policy = GPU_CPU_MANAGER,
    assume_enhanced_smiles: bool = False,
    index_base: int = 0,
    output_format: str = "smiles",  # "smiles", "mol", "molblock"
) -> Optional[Union[str, Chem.Mol]]:
    """
    Convenience front door:

      - If x is Chem.Mol → standardize_mol.
      - If x is a string with 'M  END' or multiple lines → treat as MolBlock.
      - Else:
          - if assume_enhanced_smiles=True or contains '{...}' → enhanced SMILES.
          - otherwise → plain SMILES.

      output_format:
        - "smiles"   → return SMILES
        - "mol"      → return Chem.Mol
        - "molblock" → return molfile string
    """
    if isinstance(x, Chem.Mol):
        out_mol = standardize_mol(x, ops=ops, policy=policy)
        if out_mol is None:
            return None
        if output_format == "mol":
            return out_mol
        if output_format == "molblock":
            return Chem.MolToMolBlock(out_mol)
        return Chem.MolToSmiles(out_mol)

    # x is a string
    s = x.strip()
    if not s:
        return None

    # crude molfile sniff
    if "\n" in s and "M  END" in s:
        mol = Chem.MolFromMolBlock(s, sanitize=True)
        if mol is None:
            return None
        out_mol = standardize_mol(mol, ops=ops, policy=policy)
        if out_mol is None:
            return None
        if output_format == "mol":
            return out_mol
        if output_format == "molblock":
            return Chem.MolToMolBlock(out_mol)
        return Chem.MolToSmiles(out_mol)

    # SMILES / enhanced SMILES
    is_enhanced = assume_enhanced_smiles or ("{" in s and "}" in s)
    if is_enhanced:
        out = standardize(
            s,
            ops=ops,
            policy=policy,
            preserve_chemaxon_meta=True,
            regenerate_chemaxon_meta=False,
            index_base=index_base,
        )
        if out is None:
            return None
        if output_format == "smiles":
            return out
        # parse back to Mol if needed
        mol, _meta = enhanced_smiles_to_mol(out, index_base=index_base)
        if mol is None:
            return None
        if output_format == "mol":
            return mol
        if output_format == "molblock":
            return Chem.MolToMolBlock(mol)
        return out

    # plain SMILES
    core_out = _apply_ops_to_smiles(s, ops, policy)
    if core_out is None:
        return None

    if output_format == "smiles":
        return core_out
    mol = Chem.MolFromSmiles(core_out)
    if mol is None:
        return None
    if output_format == "mol":
        return mol
    if output_format == "molblock":
        return Chem.MolToMolBlock(mol)
    return core_out


def standardize_mol_to_enhanced_smiles(
    mol: Optional[Chem.Mol],
    ops: List[str] = DEFAULT_OPS,
    policy: Policy = GPU_CPU_MANAGER,
    existing_meta: Optional[ChemAxonMeta] = None,
    index_base: int = 0,
    mode: str = "append",  # or "replace"
) -> Optional[str]:
    """
    Mol → standardized Mol → ChemAxon-style enhanced SMILES.

    - Runs the same op list as `standardize_mol`.
    - Then encodes A/B/group tokens from the *result*.
    - If existing_meta is provided:
        * mode="append": keep original non-stereo tokens; add fresh A/B/group.
        * mode="replace": drop old stereo tokens; keep non-stereo; add fresh ones.
    """
    if mol is None:
        return None

    out_mol = standardize_mol(mol, ops=ops, policy=policy)
    if out_mol is None:
        return None

    return export_enhanced_smiles_from_mol(
        out_mol,
        existing_meta=existing_meta,
        index_base=index_base,
        mode=mode,
    )


def standardize_to_enhanced_smiles(
    smiles: str,
    ops: List[str] = DEFAULT_OPS,
    policy: Policy = GPU_CPU_MANAGER,
    index_base: int = 0,
    mode: str = "append",  # how to merge with original ChemAxon meta
) -> Optional[str]:
    """
    Enhanced SMILES (or plain SMILES) → standardized enhanced SMILES.

    - If input has a { ... } block:
        * parse ChemAxon meta via enhanced_smiles_to_mol
        * apply XML-free op list on the core
        * re-encode stereo/meta using export_enhanced_smiles_from_mol,
          optionally merging with original non-stereo tokens.

    - If input is plain SMILES:
        * treat it as no existing_meta and just encode fresh tokens.
    """
    mol, meta = enhanced_smiles_to_mol(smiles, index_base=index_base)
    if mol is None:
        return None

    out_mol = standardize_mol(mol, ops=ops, policy=policy)
    if out_mol is None:
        return None

    # meta contains the original block (if any); pass it if you want merge behavior.
    return export_enhanced_smiles_from_mol(
        out_mol,
        existing_meta=meta,
        index_base=index_base,
        mode=mode,
    )
