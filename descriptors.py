"""RDKit descriptor calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors


DESCRIPTOR_COLUMNS = [
    "MolWt",
    "ExactMolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "HeavyAtomCount",
    "NumAromaticRings",
    "RingCount",
    "FractionCSP3",
    "NumHeteroatoms",
]


def calculate_descriptors(molecules_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate required descriptors for each molecule with safe failures."""
    out = molecules_df.copy()

    def _calc(mol, fn):
        try:
            if mol is None:
                return np.nan
            return fn(mol)
        except Exception:
            return np.nan

    out["MolWt"] = out["mol"].apply(lambda m: _calc(m, Descriptors.MolWt))
    out["ExactMolWt"] = out["mol"].apply(lambda m: _calc(m, Descriptors.ExactMolWt))
    out["MolLogP"] = out["mol"].apply(lambda m: _calc(m, Descriptors.MolLogP))
    out["TPSA"] = out["mol"].apply(lambda m: _calc(m, rdMolDescriptors.CalcTPSA))
    out["NumHDonors"] = out["mol"].apply(lambda m: _calc(m, Lipinski.NumHDonors))
    out["NumHAcceptors"] = out["mol"].apply(lambda m: _calc(m, Lipinski.NumHAcceptors))
    out["NumRotatableBonds"] = out["mol"].apply(lambda m: _calc(m, Lipinski.NumRotatableBonds))
    out["HeavyAtomCount"] = out["mol"].apply(lambda m: _calc(m, Lipinski.HeavyAtomCount))
    out["NumAromaticRings"] = out["mol"].apply(lambda m: _calc(m, Lipinski.NumAromaticRings))
    out["RingCount"] = out["mol"].apply(lambda m: _calc(m, Lipinski.RingCount))
    out["FractionCSP3"] = out["mol"].apply(lambda m: _calc(m, Lipinski.FractionCSP3))
    out["NumHeteroatoms"] = out["mol"].apply(lambda m: _calc(m, Lipinski.NumHeteroatoms))

    return out
