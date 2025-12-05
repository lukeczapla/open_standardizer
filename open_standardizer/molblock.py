from rdkit import Chem
from .enhanced_smiles import enhanced_smiles_to_mol, ChemAxonMeta
from .stereo_export import export_enhanced_smiles_from_mol
from .cx_bridge import mol_to_cxsmiles

def molblock_from_enhanced_smiles(
    smiles: str,
    index_base: int = 0,
) -> Optional[str]:
    """
    Parse enhanced SMILES (your curly variant) into an RDKit Mol
    and return an RDKit molblock string (V2000/V3000 as RDKit chooses).
    """
    mol, _meta = enhanced_smiles_to_mol(smiles, index_base=index_base)
    if mol is None:
        return None
    return Chem.MolToMolBlock(mol)


def enhanced_smiles_from_molblock(
    molblock: str,
    index_base: int = 0,
    existing_meta: Optional[ChemAxonMeta] = None,
) -> Optional[str]:
    """
    Parse a molblock into Mol and then generate an approximate
    ChemAxon-style enhanced SMILES (with CIP/group info) using RDKit.
    """
    mol = Chem.MolFromMolBlock(molblock, sanitize=True)
    if mol is None:
        return None
    return export_enhanced_smiles_from_mol(
        mol, existing_meta=existing_meta, index_base=index_base
    )


def cxsmiles_from_molblock(molblock: str) -> Optional[str]:
    """
    Parse a molblock and emit RDKit CXSMILES (if supported).
    """
    mol = Chem.MolFromMolBlock(molblock, sanitize=True)
    if mol is None:
        return None
    return mol_to_cxsmiles(mol)
