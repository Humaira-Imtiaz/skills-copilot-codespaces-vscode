"""Plotting functions for molecular analytics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


sns.set_theme(style="whitegrid")


def similarity_heatmap(similarity_matrix: pd.DataFrame):
    """Create similarity matrix heatmap figure."""
    fig, ax = plt.subplots(figsize=(7, 5))
    if similarity_matrix.empty:
        ax.set_title("No similarity data available")
        ax.axis("off")
        return fig
    sns.heatmap(similarity_matrix, cmap="viridis", vmin=0, vmax=1, ax=ax)
    ax.set_title("Pairwise Tanimoto Similarity")
    return fig


def cluster_size_bar(summary_df: pd.DataFrame):
    """Create cluster size bar chart."""
    fig, ax = plt.subplots(figsize=(6, 4))
    if summary_df.empty:
        ax.set_title("No cluster assignments available")
        ax.axis("off")
        return fig
    sns.barplot(data=summary_df, x="cluster", y="size", ax=ax)
    ax.set_title("Cluster Size Distribution")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Molecule count")
    return fig


def descriptor_distributions(descriptor_df: pd.DataFrame, descriptor_cols: list[str]):
    """Create descriptor distribution histograms."""
    if not descriptor_cols:
        fig, ax = plt.subplots()
        ax.set_title("No descriptor columns selected")
        ax.axis("off")
        return fig

    n_cols = 3
    n_rows = (len(descriptor_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, col in zip(axes, descriptor_cols):
        sns.histplot(descriptor_df[col].dropna(), bins=20, kde=True, ax=ax)
        ax.set_title(col)
    for ax in axes[len(descriptor_cols) :]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def similarity_distribution(pair_df: pd.DataFrame):
    """Create pairwise similarity histogram."""
    fig, ax = plt.subplots(figsize=(6, 4))
    if pair_df.empty:
        ax.set_title("No pairwise similarities available")
        ax.axis("off")
        return fig
    sns.histplot(pair_df["similarity"], bins=20, kde=True, ax=ax)
    ax.set_xlabel("Tanimoto similarity")
    ax.set_ylabel("Pair count")
    ax.set_title("Pairwise Similarity Distribution")
    return fig


def pca_scatter(descriptor_df: pd.DataFrame, cluster_df: pd.DataFrame, feature_cols: list[str]):
    """Create PCA scatter plot colored by cluster."""
    fig, ax = plt.subplots(figsize=(7, 5))
    if descriptor_df.empty or not feature_cols or len(descriptor_df) < 2:
        ax.set_title("Not enough data for PCA")
        ax.axis("off")
        return fig

    working = descriptor_df[["molecule_id", *feature_cols]].copy()
    imputer = SimpleImputer(strategy="median")
    scaled = StandardScaler().fit_transform(imputer.fit_transform(working[feature_cols]))

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled)

    plot_df = pd.DataFrame(
        {
            "molecule_id": working["molecule_id"],
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
        }
    )
    if not cluster_df.empty:
        plot_df = plot_df.merge(cluster_df, on="molecule_id", how="left")

    hue = "cluster" if "cluster" in plot_df.columns else None
    sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue=hue, ax=ax)
    ax.set_title("PCA Projection of Molecular Descriptors")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
    return fig
