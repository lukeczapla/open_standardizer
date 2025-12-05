from rdkit import Chem
from .rdkit_fallback_engine.manager import StandardizerFallbackEngine

fallback = StandardizerFallbackEngine()

def standardize_with_fallback(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    out = fallback.gpu_first(mol)
    return Chem.MolToSmiles(out)

