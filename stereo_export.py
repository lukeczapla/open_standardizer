from __future__ import annotations

from typing import List, Optional

from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.rdchem import StereoGroupType

from .enhanced_smiles import ChemAxonMeta


def export_cip_assignments(mol: Chem.Mol, index_base: int = 0) -> List[str]:
    rdmolops.AssignStereochemistry(mol, cleanIt=True, force=True)
    tokens: List[str] = []
    for idx, atom in enumerate(mol.GetAtoms()):
        if not atom.HasProp("_CIPCode"):
            continue
        cip = atom.GetProp("_CIPCode").upper()
        if cip not in ("R", "S"):
            continue
        ax_idx = idx + 1 if index_base == 1 else idx
        tokens.append(f"A{ax_idx}={cip.lower()}")
    return tokens


def export_stereogroup_tokens(mol: Chem.Mol, index_base: int = 0) -> List[str]:
    groups = mol.GetStereoGroups()
    if not groups:
        return []

    abs_groups = []
    or_groups = []
    and_groups = []
    for g in groups:
        if g.GetGroupType() == StereoGroupType.STEREO_ABSOLUTE:
            abs_groups.append(g)
        elif g.GetGroupType() == StereoGroupType.STEREO_OR:
            or_groups.append(g)
        elif g.GetGroupType() == StereoGroupType.STEREO_AND:
            and_groups.append(g)

    tokens: List[str] = []

    for g in abs_groups:
        idxs = [a.GetIdx() for a in g.GetAtoms()]
        if index_base == 1:
            idxs = [i + 1 for i in idxs]
        tokens.append("a:" + ",".join(map(str, idxs)))

    for gi, g in enumerate(or_groups, start=1):
        idxs = [a.GetIdx() for a in g.GetAtoms()]
        if index_base == 1:
            idxs = [i + 1 for i in idxs]
        tokens.append(f"o{gi}:" + ",".join(map(str, idxs)))

    for gi, g in enumerate(and_groups, start=1):
        idxs = [a.GetIdx() for a in g.GetAtoms()]
        if index_base == 1:
            idxs = [i + 1 for i in idxs]
        tokens.append(f"&{gi}:" + ",".join(map(str, idxs)))

    return tokens


def _looks_like_assignment(tok: str) -> bool:
    tok = tok.strip()
    return tok.startswith("A") and "=" in tok and tok[1:].split("=", 1)[0].isdigit()


def _looks_like_group(tok: str) -> bool:
    tok = tok.strip()
    if not tok:
        return False
    if tok[0] not in ("a", "o", "&"):
        return False
    return ":" in tok


def export_curly_block_from_mol(
    mol: Chem.Mol,
    existing_meta: Optional[ChemAxonMeta] = None,
    index_base: int = 0,
    mode: str = "append",
) -> str:
    cip_tokens = export_cip_assignments(mol, index_base=index_base)
    group_tokens = export_stereogroup_tokens(mol, index_base=index_base)

    if existing_meta is None:
        merged = cip_tokens + group_tokens
        return ";".join(merged) if merged else ""

    orig_tokens = list(existing_meta.tokens) if existing_meta.tokens else []

    if mode == "replace":
        filtered_orig = []
        for tok in orig_tokens:
            if _looks_like_assignment(tok) or _looks_like_group(tok):
                continue
            filtered_orig.append(tok)
        orig_tokens = filtered_orig

    merged = orig_tokens + cip_tokens + group_tokens
    return ";".join(merged)
