
from rdkit import Chem
from rdkit.Chem import MolStandardize

def standardize_mol(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Basic ChemAxon-like ops
    stab = MolStandardize.Cleanup()
    mol = stab.cleanup(mol)
    te = MolStandardize.TautomerEnumerator()
    mol = te.Canonicalize(mol)
    return Chem.MolToSmiles(mol, canonical=True)
