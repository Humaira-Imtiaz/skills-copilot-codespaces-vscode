"""Data loading and SMILES validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from rdkit import Chem

from utils import generate_stable_ids


@dataclass
class ValidationResult:
    """Structured validation output."""

    validated: pd.DataFrame
    valid: pd.DataFrame
    invalid: pd.DataFrame


def load_manual_smiles(smiles_text: str) -> pd.DataFrame:
    """Load one SMILES string per line from text input."""
    rows = [line.strip() for line in smiles_text.splitlines()]
    df = pd.DataFrame({"input_smiles": rows})
    df = df[df["input_smiles"].notna()].reset_index(drop=True)
    df["molecule_id"] = generate_stable_ids(len(df))
    return df[["molecule_id", "input_smiles"]]


def load_csv_data(
    source: str | bytes,
    smiles_column: str = "SMILES",
    id_column: Optional[str] = None,
) -> pd.DataFrame:
    """Load molecules from CSV path or uploaded bytes."""
    df = pd.read_csv(source)
    if smiles_column not in df.columns:
        raise ValueError(f"SMILES column '{smiles_column}' not found.")

    out = pd.DataFrame()
    out["input_smiles"] = df[smiles_column]

    if id_column and id_column in df.columns:
        out["molecule_id"] = df[id_column].astype(str)
    else:
        out["molecule_id"] = generate_stable_ids(len(out))

    return out[["molecule_id", "input_smiles"]]


def validate_smiles(df: pd.DataFrame) -> ValidationResult:
    """Validate SMILES, preserve invalid rows, and detect duplicates."""
    records = []
    for _, row in df.iterrows():
        molecule_id = row.get("molecule_id")
        raw_smiles = row.get("input_smiles")

        error_category = None
        canonical_smiles = None
        canonical_isomeric_smiles = None
        is_valid = False

        if pd.isna(raw_smiles) or not str(raw_smiles).strip():
            error_category = "missing_smiles"
        else:
            try:
                mol = Chem.MolFromSmiles(str(raw_smiles))
                if mol is None:
                    error_category = "invalid_smiles"
                else:
                    is_valid = True
                    canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
                    canonical_isomeric_smiles = Chem.MolToSmiles(
                        mol, canonical=True, isomericSmiles=True
                    )
            except Exception:
                error_category = "parse_exception"

        records.append(
            {
                "molecule_id": molecule_id,
                "input_smiles": raw_smiles,
                "is_valid": is_valid,
                "error_category": error_category,
                "canonical_smiles": canonical_smiles,
                "canonical_isomeric_smiles": canonical_isomeric_smiles,
            }
        )

    validated = pd.DataFrame(records)
    dup_series = validated["canonical_isomeric_smiles"].fillna("__INVALID__")
    dup_count = dup_series.groupby(dup_series).transform("count")
    validated["is_duplicate_structure"] = validated["is_valid"] & (dup_count > 1)

    valid = validated[validated["is_valid"]].copy().reset_index(drop=True)
    invalid = validated[~validated["is_valid"]].copy().reset_index(drop=True)

    return ValidationResult(validated=validated, valid=valid, invalid=invalid)


def prepare_analysis_set(valid_df: pd.DataFrame, merge_duplicates: bool = False) -> pd.DataFrame:
    """Optionally merge duplicates for analysis using canonical isomeric SMILES."""
    if valid_df.empty or not merge_duplicates:
        return valid_df.copy()
    return valid_df.drop_duplicates(subset=["canonical_isomeric_smiles"]).reset_index(drop=True)
