from rdkit import Chem
from rdkit.Chem import rdMolStandardize, AllChem, rdmolops

_STRIPPER_DEFAULT_KEEP_LAST = rdMolStandardize.FragmentRemover()

def _strip_with_remover(remover, mol, allow_empty: bool):
    """
    Helper to apply a FragmentRemover but optionally prevent
    ending up with an empty / zero-atom mol.
    """
    try:
        out = remover(mol)
        if out is None or out.GetNumAtoms() == 0:
            return mol if not allow_empty else mol  # RDKit can't represent empty; keep mol
        return out
    except Exception:
        return mol

# ---------- 1. CLEAR STEREO ----------
def op_clear_stereo(mol):
    Chem.RemoveStereochemistry(mol)
    for atom in mol.GetAtoms():
        atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
    return mol


# ---------- 2. REMOVE FRAGMENT (KEEP LARGEST) ----------
def op_remove_fragment_keeplargest(mol):
    frags = Chem.GetMolFrags(mol, asMols=True, sanitize=False)
    if not frags:
        return mol
    largest = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    Chem.SanitizeMol(largest)
    return largest


# ---------- 3. REMOVE ATTACHED DATA ----------
def op_remove_attached_data(mol):
    for atom in mol.GetAtoms():
        atom.ClearProp("")
        for k in atom.GetPropsAsDict().keys():
            atom.ClearProp(k)
    for bond in mol.GetBonds():
        for k in bond.GetPropsAsDict().keys():
            bond.ClearProp(k)
    return mol


# ---------- 4. REMOVE ATOM VALUES ----------
def op_remove_atom_values(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
        atom.SetFormalCharge(atom.GetFormalCharge())  # normalize
    return mol


# ---------- 5. REMOVE EXPLICIT HYDROGENS ----------
def op_remove_explicit_h(mol):
    return Chem.RemoveHs(mol, updateExplicitCount=True)


# ---------- 6. CLEAR ISOTOPES ----------
def op_clear_isotopes(mol):
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    return mol


# ---------- 7. NEUTRALIZE ----------
def _neutralize(mol):
    patterns = (
        ('[n+;H]', 'n'),
        ('[N+;!H0]', 'N'),
        ('[$([O-]);!$([O-][#7])]', 'O'),
        ('[S-]', 'S'),
        ('[$([N-]);!$([N-][#6]);!$([N-][#7])]', 'N'),
    )
    replaced = False

    for smarts, repl in patterns:
        while True:
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
            if not matches:
                break
            replaced = True
            for atom_idx in matches:
                atom = mol.GetAtomWithIdx(atom_idx[0])
                atom.SetFormalCharge(0)

    if replaced:
        Chem.SanitizeMol(mol)
    return mol


def op_neutralize(mol):
    try:
        return _neutralize(mol)
    except:
        return mol


# ---------- 8. MESOMERIZE (CHEMAXON-LIKE RESONANCE NORMALIZATION) ----------
def op_mesomerize(mol):
    res = rdMolStandardize.ResonanceMolSupplier(mol, Chem.UNCONSTRAINED_RESONANCE)
    # Select lowest-energy canonical resonance form
    if len(res) > 0:
        best = Chem.Mol(res[0])
        Chem.SanitizeMol(best)
        return best
    return mol


# ---------- 9. TAUTOMERIZE ----------
def op_tautomerize(mol):
    enumerator = rdMolStandardize.TautomerEnumerator()
    best = enumerator.Canonicalize(mol)
    return best


# ---------- 10. AROMATIZE ----------
def op_aromatize(mol):
    Chem.SetAromaticity(mol)
    AllChem.SetAromaticity(mol)
    return mol

# ---------- 11. REMOVE FRAGMENT SMALLEST
def op_remove_fragment_smallest(mol):
    """
    Keep all fragments except the one with the *fewest* heavy atoms.
    Simple "remove smallest salt" style behavior.
    """
    frags = Chem.GetMolFrags(mol, asMols=True, sanitize=False)
    if not frags:
        return mol

    # heavy atom counts
    with_counts = [(f, f.GetNumHeavyAtoms()) for f in frags]
    # sort by heavy atom count ascending
    with_counts.sort(key=lambda x: x[1])

    # drop the smallest, keep all others with >0 heavy atoms
    kept = [f for f, cnt in with_counts[1:] if cnt > 0]
    if not kept:
        # if everything is tiny, just keep original
        return mol

    combo = kept[0]
    for f in kept[1:]:
        combo = rdmolops.CombineMols(combo, f)

    Chem.SanitizeMol(combo)
    return combo

# usedefaultsalts="true", dontremovelastcomponent="true"
def op_strip_salts_default_keep_last(mol):
    return _strip_with_remover(_STRIPPER_DEFAULT_KEEP_LAST, mol, allow_empty=False)


# usedefaultsalts="true", dontremovelastcomponent="false"
# We approximate by **allowing all salts to be stripped**, but if RDKit
# returns an empty / broken mol we fall back to original.
def op_strip_salts_default_allow_empty(mol):
    return _strip_with_remover(_STRIPPER_DEFAULT_KEEP_LAST, mol, allow_empty=True)


# usedefaultsalts="false", dontremovelastcomponent="true"
# TODO: wire custom salt SMARTS here. For now we just use the same default
# remover but keep_last semantics.
def op_strip_salts_custom_keep_last(mol):
    # placeholder: behaves like default_keep_last until you add custom patterns
    return _strip_with_remover(_STRIPPER_DEFAULT_KEEP_LAST, mol, allow_empty=False)


# usedefaultsalts="false", dontremovelastcomponent="false"
def op_strip_salts_custom_allow_empty(mol):
    # placeholder: behaves like default_allow_empty until you add custom patterns
    return _strip_with_remover(_STRIPPER_DEFAULT_KEEP_LAST, mol, allow_empty=True)


# --------- MAPPING TO ACTION NAMES ---------
CPU_OPS = {
    "clear_stereo": op_clear_stereo,
    "remove_fragment_keeplargest": op_remove_fragment_keeplargest,
    "remove_largest_fragment": op_remove_fragment_keeplargest,  # alias
    "strip_salts_default_keep_last": op_strip_salts_default_keep_last,
    "strip_salts_default_allow_empty": op_strip_salts_default_allow_empty,
    "strip_salts_custom_keep_last": op_strip_salts_custom_keep_last,
    "strip_salts_custom_allow_empty": op_strip_salts_custom_allow_empty,
    "remove_attached_data": op_remove_attached_data,
    "remove_atom_values": op_remove_atom_values,
    "remove_explicit_h": op_remove_explicit_h,
    "clear_isotopes": op_clear_isotopes,
    "neutralize": op_neutralize,
    "mesomerize": op_mesomerize,
    "tautomerize": op_tautomerize,
    "aromatize": op_aromatize,
    "remove_fragment_smallest": op_remove_fragment_smallest,
}


def cpu_execute(op_name, mol):
    """
    Executes a CPU operation by name.
    Returns a SMILES string.
    """
    fn = CPU_OPS.get(op_name)
    if fn is None:
        # Unknown op → return original SMILES
        return Chem.MolToSmiles(mol)

    new_mol = fn(Chem.Mol(mol))  # operate on a copy
    return Chem.MolToSmiles(new_mol)
