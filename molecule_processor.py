"""Molecule parsing and rendering helpers."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw


def process_molecules(valid_df: pd.DataFrame) -> pd.DataFrame:
    """Convert valid SMILES rows into RDKit molecule objects and metadata."""
    rows = []
    for _, row in valid_df.iterrows():
        mol = None
        processing_error = None
        try:
            mol = Chem.MolFromSmiles(str(row["input_smiles"]))
            if mol is None:
                processing_error = "invalid_structure"
            else:
                AllChem.Compute2DCoords(mol)
        except Exception as exc:  # pragma: no cover
            processing_error = f"processing_exception:{exc.__class__.__name__}"

        rows.append(
            {
                **row.to_dict(),
                "mol": mol,
                "canonical_smiles": Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
                if mol is not None
                else row.get("canonical_smiles"),
                "canonical_isomeric_smiles": Chem.MolToSmiles(
                    mol, canonical=True, isomericSmiles=True
                )
                if mol is not None
                else row.get("canonical_isomeric_smiles"),
                "processing_error": processing_error,
            }
        )

    return pd.DataFrame(rows)


def molecule_image_png(mol: Chem.Mol, size: tuple[int, int] = (250, 250)) -> bytes:
    """Generate PNG bytes for a molecule image."""
    if mol is None:
        return b""
    img = Draw.MolToImage(mol, size=size)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()
