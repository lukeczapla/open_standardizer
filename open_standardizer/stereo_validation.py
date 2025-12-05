# stereo_validation.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.rdchem import BondStereo, ChiralType

from .enhanced_smiles import ChemAxonMeta, enhanced_smiles_to_mol

try:
    from rdkit.Chem import rdCIPLabeler
    HAS_CIP_LABELER = True
except ImportError:
    HAS_CIP_LABELER = False


# -------------------------------------------------------------------
# Core CIP / E,Z comparison dataclasses (unchanged)
# -------------------------------------------------------------------

@dataclass
class AtomCIPComparison:
    atom_index: int              # ChemAxon “A index” (as in A<i>=r/s)
    assignment: str              # ChemAxon code (e.g. "s", "r", "p")
    cip_code: Optional[str]      # RDKit code ("R"/"S") or None
    status: str                  # "match", "mismatch", "missing_cip", "non_cip_code", ...


@dataclass
class BondCIPComparison:
    atom_index1: int             # ChemAxon A index 1 (from B<i,j>=code)
    atom_index2: int             # ChemAxon A index 2
    assignment: str              # ChemAxon code ("e", "z", ...)
    cip_code: Optional[str]      # RDKit code ("E"/"Z") or None
    status: str                  # "match", "mismatch", "missing_cip", "no_bond", ...


# -------------------------------------------------------------------
# NEW: local parity @: / @@: and ring c:/t:/u: comparison dataclasses
# -------------------------------------------------------------------

@dataclass
class AtomParityComparison:
    """
    Represents a local parity token from CX-style notation:

        @:<i1>,<i2>,...   → local ODD parity
        @@:<i1>,<i2>,...  → local EVEN parity

    We don’t try to derive an exact R/S mapping from this; instead we record:

      - whether RDKit thinks the atom is chiral at all
      - whether the index is valid
    """
    atom_index: int              # ChemAxon-style index (pre index_base adjustment)
    parity: str                  # "odd" or "even"
    rd_has_chiral_center: bool   # True if RDKit sees a chiral tag here
    status: str                  # "chiral_center_present", "no_chiral_center", "index_out_of_range"


@dataclass
class RingStereoComparison:
    """
    Represents ring double-bond CIS/TRANS/UNSPEC tokens:

        c:<b1>,<b2>,...   → bond indices with CIS flag
        t:<b1>,<b2>,...   → bond indices with TRANS flag
        u:<b1>,<b2>,...   → bond indices with UNSPEC flag

    We compare them to RDKit BondStereo:

      - BondStereo.STEREOCIS   → "c"
      - BondStereo.STEREOTRANS → "t"
      - no cis/trans           → None
    """
    bond_index: int              # Bond index in ChemAxon’s export (assumed = RDKit idx)
    assignment: str              # "c", "t", or "u"
    rd_stereo: Optional[str]     # "c", "t", or None
    status: str                  # "match", "mismatch", "missing_ring_stereo",
                                 # "expected_unspecified", "index_out_of_range"


@dataclass
class StereoValidationResult:
    # Atom-centric CIP
    atom_matches: List[AtomCIPComparison] = field(default_factory=list)
    atom_mismatches: List[AtomCIPComparison] = field(default_factory=list)
    atom_missing_cip: List[AtomCIPComparison] = field(default_factory=list)
    atom_non_cip_assignments: List[AtomCIPComparison] = field(default_factory=list)

    # Bond-centric E/Z
    bond_matches: List[BondCIPComparison] = field(default_factory=list)
    bond_mismatches: List[BondCIPComparison] = field(default_factory=list)
    bond_missing_cip: List[BondCIPComparison] = field(default_factory=list)
    bond_unknown_assignments: List[BondCIPComparison] = field(default_factory=list)

    # NEW: local parity and ring stereo summaries
    parity_atoms: List[AtomParityComparison] = field(default_factory=list)
    ring_stereo_matches: List[RingStereoComparison] = field(default_factory=list)
    ring_stereo_mismatches: List[RingStereoComparison] = field(default_factory=list)
    ring_stereo_missing: List[RingStereoComparison] = field(default_factory=list)
    ring_stereo_unknown: List[RingStereoComparison] = field(default_factory=list)

    def ok(self) -> bool:
        """
        Keep the original semantics: "ok" means no CIP mismatches.

        Local parity and ring c/t/u mismatches are *reported* but don’t
        change the ok() flag, so existing callers/CLI behavior stays sane.
        """
        return not self.atom_mismatches and not self.bond_mismatches


# -------------------------------------------------------------------
# Helpers: normalize expected codes
# -------------------------------------------------------------------

def _normalize_atom_expected(code: str) -> Optional[str]:
    c = (code or "").strip().upper()
    if c in ("R", "S"):
        return c
    return None  # e.g. "p" → non-CIP code


def _normalize_bond_expected(code: str) -> Optional[str]:
    c = (code or "").strip().upper()
    if c in ("E", "Z"):
        return c
    return None


def _to_rdkit_index(idx: int, index_base: int) -> int:
    """
    Convert ChemAxon indices to RDKit’s 0-based scheme.

    In your data, index_base=0 means "as-is".
    If index_base=1, then A1 → atom 0, A2 → atom 1, etc.
    """
    return idx if index_base == 0 else idx - 1


# -------------------------------------------------------------------
# RDKit stereochemistry assignment
# -------------------------------------------------------------------

def _assign_cip(mol: Chem.Mol) -> None:
    """
    Make sure RDKit has atom + bond stereochemistry assigned, then
    CIP codes (if rdCIPLabeler is available).
    """
    rdmolops.AssignStereochemistry(
        mol, cleanIt=True, force=True, flagPossibleStereoCenters=True
    )
    if HAS_CIP_LABELER:
        rdCIPLabeler.AssignCIPLabels(mol)


def _bond_stereo_to_ez(bond: Chem.Bond) -> Optional[str]:
    st = bond.GetStereo()
    if st == BondStereo.STEREOE:
        return "E"
    if st == BondStereo.STEREOZ:
        return "Z"
    return None


def _bond_stereo_to_ring_ct(bond: Chem.Bond) -> Optional[str]:
    """
    Map RDKit BondStereo to "c"/"t" for ring-like cis/trans info.

    RDKit uses STEREOCIS/STEREOTRANS for some cases; that’s the
    most direct match to CX c:/t: tokens.
    """
    st = bond.GetStereo()
    if st == BondStereo.STEREOCIS:
        return "c"
    if st == BondStereo.STEREOTRANS:
        return "t"
    return None


# -------------------------------------------------------------------
# NEW: parsing CX-like tokens from ChemAxonMeta.tokens
# -------------------------------------------------------------------

def _iter_raw_tokens(meta: ChemAxonMeta) -> List[str]:
    """
    Safely pull the raw token list from ChemAxonMeta.

    We assume something like:
        meta.tokens: List[str]
    exists. If not, this quietly returns an empty list.
    """
    tokens = getattr(meta, "tokens", None)
    if not tokens:
        return []
    return list(tokens)


def _collect_parity_tokens(meta: ChemAxonMeta) -> List[tuple[int, str]]:
    """
    Parse @: / @@: tokens from the raw meta tokens.

    Returns a list of (atom_index, "odd"/"even") in the *ChemAxon* index space.
    """
    out: List[tuple[int, str]] = []
    for tok in _iter_raw_tokens(meta):
        t = tok.strip()
        if t.startswith("@@:"):
            parity = "even"
            idx_part = t[3:]
        elif t.startswith("@:"):
            parity = "odd"
            idx_part = t[2:]
        else:
            continue

        for part in idx_part.split(","):
            part = part.strip()
            if not part:
                continue
            if not part.isdigit():
                continue
            out.append((int(part), parity))

    return out


def _collect_ring_stereo_tokens(meta: ChemAxonMeta) -> List[tuple[int, str]]:
    """
    Parse c:/t:/u: tokens from the raw meta tokens.

    Returns (bond_index, assignment) with assignment ∈ {"c","t","u"}.
    """
    out: List[tuple[int, str]] = []
    for tok in _iter_raw_tokens(meta):
        t = tok.strip()
        if not t or ":" not in t:
            continue

        prefix = t[0]
        if prefix not in ("c", "t", "u"):
            continue

        # "c:1,2,3"
        idx_part = t.split(":", 1)[1]
        for part in idx_part.split(","):
            part = part.strip()
            if not part:
                continue
            if not part.isdigit():
                continue
            out.append((int(part), prefix))

    return out


# -------------------------------------------------------------------
# Core validation on Mol + ChemAxonMeta
# -------------------------------------------------------------------

def validate_cip_assignments_on_mol(
    mol: Chem.Mol,
    meta: ChemAxonMeta,
    index_base: int = 0,
) -> StereoValidationResult:
    res = StereoValidationResult()
    if meta is None:
        return res

    _assign_cip(mol)
    n_atoms = mol.GetNumAtoms()
    n_bonds = mol.GetNumBonds()

    # ---- atoms (A<idx>=code) ----
    for assign in meta.assignments:
        rd_idx = _to_rdkit_index(assign.atom_index, index_base=index_base)
        if rd_idx < 0 or rd_idx >= n_atoms:
            res.atom_non_cip_assignments.append(
                AtomCIPComparison(assign.atom_index, assign.config, None, "index_out_of_range")
            )
            continue

        expected = _normalize_atom_expected(assign.config)
        if expected is None:
            # e.g. "p": ChemAxon code we don't map to R/S
            res.atom_non_cip_assignments.append(
                AtomCIPComparison(assign.atom_index, assign.config, None, "non_cip_code")
            )
            continue

        atom = mol.GetAtomWithIdx(rd_idx)
        if atom.HasProp("_CIPCode"):
            cip = atom.GetProp("_CIPCode").upper()
        else:
            res.atom_missing_cip.append(
                AtomCIPComparison(assign.atom_index, assign.config, None, "missing_cip")
            )
            continue

        if cip == expected:
            res.atom_matches.append(
                AtomCIPComparison(assign.atom_index, assign.config, cip, "match")
            )
        else:
            res.atom_mismatches.append(
                AtomCIPComparison(assign.atom_index, assign.config, cip, "mismatch")
            )

    # ---- bonds (B<i,j>=code) ----
    for bassign in meta.bond_assignments:
        rd_i1 = _to_rdkit_index(bassign.atom_index1, index_base=index_base)
        rd_i2 = _to_rdkit_index(bassign.atom_index2, index_base=index_base)
        if not (0 <= rd_i1 < n_atoms and 0 <= rd_i2 < n_atoms):
            res.bond_unknown_assignments.append(
                BondCIPComparison(
                    bassign.atom_index1,
                    bassign.atom_index2,
                    bassign.config,
                    None,
                    "index_out_of_range",
                )
            )
            continue

        expected = _normalize_bond_expected(bassign.config)
        if expected is None:
            res.bond_unknown_assignments.append(
                BondCIPComparison(
                    bassign.atom_index1,
                    bassign.atom_index2,
                    bassign.config,
                    None,
                    "unknown_expected_code",
                )
            )
            continue

        bond = mol.GetBondBetweenAtoms(rd_i1, rd_i2)
        if bond is None:
            res.bond_unknown_assignments.append(
                BondCIPComparison(
                    bassign.atom_index1,
                    bassign.atom_index2,
                    bassign.config,
                    None,
                    "no_bond",
                )
            )
            continue

        cip = _bond_stereo_to_ez(bond)
        if cip is None:
            res.bond_missing_cip.append(
                BondCIPComparison(
                    bassign.atom_index1,
                    bassign.atom_index2,
                    bassign.config,
                    None,
                    "missing_cip",
                )
            )
            continue

        if cip == expected:
            res.bond_matches.append(
                BondCIPComparison(
                    bassign.atom_index1,
                    bassign.atom_index2,
                    bassign.config,
                    cip,
                    "match",
                )
            )
        else:
            res.bond_mismatches.append(
                BondCIPComparison(
                    bassign.atom_index1,
                    bassign.atom_index2,
                    bassign.config,
                    cip,
                    "mismatch",
                )
            )

    # ---- NEW: local parity @: / @@: ----
    for raw_idx, parity in _collect_parity_tokens(meta):
        rd_idx = _to_rdkit_index(raw_idx, index_base=index_base)
        if rd_idx < 0 or rd_idx >= n_atoms:
            res.parity_atoms.append(
                AtomParityComparison(
                    atom_index=raw_idx,
                    parity=parity,
                    rd_has_chiral_center=False,
                    status="index_out_of_range",
                )
            )
            continue

        atom = mol.GetAtomWithIdx(rd_idx)
        has_chiral = atom.GetChiralTag() != ChiralType.CHI_UNSPECIFIED
        status = "chiral_center_present" if has_chiral else "no_chiral_center"

        res.parity_atoms.append(
            AtomParityComparison(
                atom_index=raw_idx,
                parity=parity,
                rd_has_chiral_center=has_chiral,
                status=status,
            )
        )

    # ---- NEW: ring c:/t:/u: tokens → cis/trans ----
    for bidx, assignment in _collect_ring_stereo_tokens(meta):
        if bidx < 0 or bidx >= n_bonds:
            res.ring_stereo_unknown.append(
                RingStereoComparison(
                    bond_index=bidx,
                    assignment=assignment,
                    rd_stereo=None,
                    status="index_out_of_range",
                )
            )
            continue

        bond = mol.GetBondWithIdx(bidx)
        rd_ct = _bond_stereo_to_ring_ct(bond)  # "c", "t", or None

        if assignment == "u":
            # Unspecified in the data: treat "no cis/trans in RDKit"
            # as consistent, otherwise note that RDKit has more info.
            if rd_ct is None:
                res.ring_stereo_matches.append(
                    RingStereoComparison(
                        bond_index=bidx,
                        assignment=assignment,
                        rd_stereo=None,
                        status="match",
                    )
                )
            else:
                res.ring_stereo_mismatches.append(
                    RingStereoComparison(
                        bond_index=bidx,
                        assignment=assignment,
                        rd_stereo=rd_ct,
                        status="expected_unspecified",
                    )
                )
        else:
            # assignment is "c" or "t"
            if rd_ct is None:
                res.ring_stereo_missing.append(
                    RingStereoComparison(
                        bond_index=bidx,
                        assignment=assignment,
                        rd_stereo=None,
                        status="missing_ring_stereo",
                    )
                )
            elif rd_ct == assignment:
                res.ring_stereo_matches.append(
                    RingStereoComparison(
                        bond_index=bidx,
                        assignment=assignment,
                        rd_stereo=rd_ct,
                        status="match",
                    )
                )
            else:
                res.ring_stereo_mismatches.append(
                    RingStereoComparison(
                        bond_index=bidx,
                        assignment=assignment,
                        rd_stereo=rd_ct,
                        status="mismatch",
                    )
                )

    return res


def validate_cip_assignments_on_smiles(
    smiles: str,
    index_base: int = 0,
) -> Optional[StereoValidationResult]:
    """
    Entry point: enhanced SMILES (with { ... }) → Mol + meta → validation.

    NOTE:
      If you later want to support RDKit-style CXSMILES pipes directly,
      you can:
        - either pre-normalize them into your { ... } scheme
        - or add a parallel enhanced_smiles_to_mol_from_pipe(...) helper.

      This function doesn’t care; it just needs enhanced_smiles_to_mol().
    """
    mol, meta = enhanced_smiles_to_mol(smiles, index_base=index_base)
    if mol is None or meta is None:
        return None
    return validate_cip_assignments_on_mol(mol, meta, index_base=index_base)
