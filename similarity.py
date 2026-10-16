"""Pairwise similarity utilities."""

from __future__ import annotations

import pandas as pd
from rdkit import DataStructs


def build_similarity_matrix(fps_df: pd.DataFrame) -> pd.DataFrame:
    """Build square Tanimoto similarity matrix with molecule IDs as labels."""
    if fps_df.empty:
        return pd.DataFrame()

    ids = fps_df["molecule_id"].tolist()
    fps = fps_df["fingerprint"].tolist()
    matrix = []
    for fp_a in fps:
        row = []
        for fp_b in fps:
            row.append(float(DataStructs.TanimotoSimilarity(fp_a, fp_b)))
        matrix.append(row)

    return pd.DataFrame(matrix, index=ids, columns=ids)


def ranked_similarity_pairs(
    similarity_matrix: pd.DataFrame,
    threshold: float = 0.7,
) -> pd.DataFrame:
    """Create unique non-self ranked pairs above threshold."""
    if similarity_matrix.empty:
        return pd.DataFrame(
            columns=["molecule_a", "molecule_b", "similarity", "above_threshold"]
        )

    ids = list(similarity_matrix.index)
    rows = []
    for i, a in enumerate(ids):
        for j in range(i + 1, len(ids)):
            b = ids[j]
            sim = float(similarity_matrix.iloc[i, j])
            rows.append(
                {
                    "molecule_a": a,
                    "molecule_b": b,
                    "similarity": sim,
                    "above_threshold": sim >= threshold,
                }
            )

    df = pd.DataFrame(rows)
    return df.sort_values("similarity", ascending=False).reset_index(drop=True)


def nearest_neighbors(similarity_matrix: pd.DataFrame, molecule_id: str, top_k: int = 5) -> pd.DataFrame:
    """Return nearest structural neighbors for a selected molecule."""
    if molecule_id not in similarity_matrix.index:
        return pd.DataFrame(columns=["neighbor", "similarity"])

    series = similarity_matrix.loc[molecule_id].drop(index=molecule_id, errors="ignore")
    return (
        series.sort_values(ascending=False)
        .head(top_k)
        .rename_axis("neighbor")
        .reset_index(name="similarity")
    )
