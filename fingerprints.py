"""Fingerprint generation for molecular similarity analysis."""

from __future__ import annotations

import pandas as pd
from rdkit.Chem import AllChem, RDKFingerprint


ALLOWED_RADII = {2, 3}
ALLOWED_BITS = {1024, 2048, 4096}


def generate_fingerprints(
    molecules_df: pd.DataFrame,
    method: str = "morgan",
    radius: int = 2,
    n_bits: int = 2048,
) -> pd.DataFrame:
    """Generate binary fingerprints with reproducible configuration."""
    if method == "morgan":
        if radius not in ALLOWED_RADII:
            raise ValueError(f"Unsupported radius {radius}. Allowed: {sorted(ALLOWED_RADII)}")
        if n_bits not in ALLOWED_BITS:
            raise ValueError(f"Unsupported n_bits {n_bits}. Allowed: {sorted(ALLOWED_BITS)}")

    out = molecules_df.copy()

    def _fp(mol):
        if mol is None:
            return None
        if method == "morgan":
            return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        if method == "rdk":
            return RDKFingerprint(mol, fpSize=n_bits)
        raise ValueError("method must be 'morgan' or 'rdk'")

    out["fingerprint"] = out["mol"].apply(_fp)
    out["fingerprint_method"] = method
    out["fingerprint_radius"] = radius if method == "morgan" else None
    out["fingerprint_bits"] = n_bits
    return out
