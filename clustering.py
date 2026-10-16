"""Clustering methods for molecular similarity analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from sklearn.cluster import AgglomerativeClustering


def agglomerative_from_similarity(
    similarity_matrix: pd.DataFrame,
    n_clusters: int = 2,
    linkage: str = "average",
) -> pd.DataFrame:
    """Cluster molecules by agglomerative clustering over precomputed distance."""
    if similarity_matrix.empty:
        return pd.DataFrame(columns=["molecule_id", "cluster"])

    ids = similarity_matrix.index.tolist()
    if len(ids) == 1:
        return pd.DataFrame({"molecule_id": ids, "cluster": [0]})

    n_clusters = max(1, min(n_clusters, len(ids)))
    distance_matrix = 1.0 - similarity_matrix.values

    kwargs = {"n_clusters": n_clusters, "linkage": linkage}
    try:
        model = AgglomerativeClustering(metric="precomputed", **kwargs)
    except TypeError:  # pragma: no cover
        model = AgglomerativeClustering(affinity="precomputed", **kwargs)

    labels = model.fit_predict(distance_matrix)
    return pd.DataFrame({"molecule_id": ids, "cluster": labels})


def butina_from_fingerprints(fps_df: pd.DataFrame, cutoff: float = 0.4) -> pd.DataFrame:
    """Cluster fingerprints with RDKit Butina using distance cutoff."""
    if fps_df.empty:
        return pd.DataFrame(columns=["molecule_id", "cluster"])

    ids = fps_df["molecule_id"].tolist()
    fps = fps_df["fingerprint"].tolist()

    if len(fps) == 1:
        return pd.DataFrame({"molecule_id": ids, "cluster": [0]})

    dists = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - x for x in sims])

    clusters = Butina.ClusterData(dists, len(fps), cutoff, isDistData=True)

    assignments = np.full(len(fps), -1, dtype=int)
    for cidx, members in enumerate(clusters):
        for m in members:
            assignments[m] = cidx

    return pd.DataFrame({"molecule_id": ids, "cluster": assignments.tolist()})


def cluster_summary(cluster_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize cluster sizes and singleton clusters."""
    if cluster_df.empty:
        return pd.DataFrame(columns=["cluster", "size", "is_singleton"])
    summary = cluster_df.groupby("cluster").size().reset_index(name="size")
    summary["is_singleton"] = summary["size"] == 1
    return summary.sort_values("cluster").reset_index(drop=True)
