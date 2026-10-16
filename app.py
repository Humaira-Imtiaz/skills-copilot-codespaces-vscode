"""Streamlit app for molecular similarity and clustering."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from clustering import agglomerative_from_similarity, butina_from_fingerprints, cluster_summary
from data_loader import load_csv_data, load_manual_smiles, prepare_analysis_set, validate_smiles
from descriptors import DESCRIPTOR_COLUMNS, calculate_descriptors
from fingerprints import generate_fingerprints
from molecule_processor import molecule_image_png, process_molecules
from similarity import build_similarity_matrix, nearest_neighbors, ranked_similarity_pairs
from utils import dataframe_to_csv_bytes
from visualization import (
    cluster_size_bar,
    descriptor_distributions,
    pca_scatter,
    similarity_distribution,
    similarity_heatmap,
)


def _load_input() -> pd.DataFrame:
    input_mode = st.radio("Input method", ["Manual SMILES", "CSV upload"], horizontal=True)

    if input_mode == "Manual SMILES":
        smiles_text = st.text_area("Enter one SMILES per line", height=180)
        if not smiles_text.strip():
            return pd.DataFrame(columns=["molecule_id", "input_smiles"])
        return load_manual_smiles(smiles_text)

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    smiles_col = st.text_input("SMILES column", value="SMILES")
    id_col = st.text_input("Molecule ID column (optional)", value="")
    if uploaded is None:
        return pd.DataFrame(columns=["molecule_id", "input_smiles"])
    return load_csv_data(uploaded, smiles_column=smiles_col, id_column=id_col or None)


def main() -> None:
    st.set_page_config(page_title="Molecular Similarity & Clustering Tool", layout="wide")
    st.title("Molecular Similarity & Clustering Tool")
    st.write(
        "Compute RDKit-based molecular descriptors, Tanimoto similarity, and clustering from "
        "SMILES inputs. Computational similarity is an analytical metric and does not prove "
        "equivalent chemical behavior."
    )

    with st.sidebar:
        st.header("Fingerprint settings")
        fp_method = st.selectbox("Method", ["morgan", "rdk"])
        radius = st.selectbox("Morgan radius", [2, 3], index=0)
        n_bits = st.selectbox("Fingerprint size (bits)", [1024, 2048, 4096], index=1)
        similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.7, 0.01)
        merge_duplicates = st.checkbox("Merge duplicate structures for analysis", value=False)

        st.header("Clustering")
        algo = st.selectbox("Algorithm", ["Agglomerative", "Butina"])
        n_clusters = st.number_input("Agglomerative cluster count", min_value=1, value=3)
        butina_cutoff = st.slider("Butina distance cutoff", 0.05, 1.0, 0.4, 0.05)

    raw_df = _load_input()
    if raw_df.empty:
        st.info("Enter SMILES or upload CSV to begin analysis.")
        return

    validation = validate_smiles(raw_df)
    st.subheader("Input Validation")
    c1, c2, c3 = st.columns(3)
    c1.metric("Valid molecules", len(validation.valid))
    c2.metric("Invalid molecules", len(validation.invalid))
    c3.metric("Duplicate structures", int(validation.validated["is_duplicate_structure"].sum()))

    st.dataframe(validation.validated, use_container_width=True)
    st.download_button(
        "Download validation report",
        dataframe_to_csv_bytes(validation.validated),
        "validation_report.csv",
        "text/csv",
    )

    if validation.valid.empty:
        st.warning("No valid molecules available for downstream analysis.")
        st.dataframe(validation.invalid, use_container_width=True)
        return

    analysis_df = prepare_analysis_set(validation.valid, merge_duplicates=merge_duplicates)
    processed = process_molecules(analysis_df)
    fp_df = generate_fingerprints(processed, method=fp_method, radius=radius, n_bits=n_bits)
    desc_df = calculate_descriptors(fp_df)

    st.subheader("Molecular Descriptors")
    st.caption("LogP values are calculated estimates (not experimental measurements).")
    desc_view = desc_df[["molecule_id", "canonical_smiles", *DESCRIPTOR_COLUMNS]]
    st.dataframe(desc_view, use_container_width=True)
    st.download_button(
        "Download descriptors",
        dataframe_to_csv_bytes(desc_view),
        "descriptors.csv",
        "text/csv",
    )

    sim_matrix = build_similarity_matrix(fp_df)
    pair_df = ranked_similarity_pairs(sim_matrix, threshold=similarity_threshold)

    st.subheader("Molecular Similarity")
    st.caption("Morgan fingerprints encode local environments; similarity thresholds are user-defined.")
    st.pyplot(similarity_heatmap(sim_matrix), clear_figure=True)
    st.dataframe(sim_matrix, use_container_width=True)
    st.download_button(
        "Download similarity matrix",
        dataframe_to_csv_bytes(sim_matrix.reset_index().rename(columns={"index": "molecule_id"})),
        "similarity_matrix.csv",
        "text/csv",
    )
    st.dataframe(pair_df, use_container_width=True)

    selected = st.selectbox("Select molecule for nearest neighbors", sim_matrix.index.tolist())
    st.dataframe(nearest_neighbors(sim_matrix, selected), use_container_width=True)

    if algo == "Agglomerative":
        if len(sim_matrix) < 2:
            st.warning("Need at least two molecules for agglomerative clustering.")
            cluster_df = pd.DataFrame(columns=["molecule_id", "cluster"])
        else:
            cluster_df = agglomerative_from_similarity(sim_matrix, n_clusters=int(n_clusters))
    else:
        cluster_df = butina_from_fingerprints(fp_df, cutoff=float(butina_cutoff))

    st.subheader("Clustering Analysis")
    cluster_sizes = cluster_summary(cluster_df)
    st.dataframe(cluster_df, use_container_width=True)
    st.dataframe(cluster_sizes, use_container_width=True)
    st.pyplot(cluster_size_bar(cluster_sizes), clear_figure=True)
    st.download_button(
        "Download cluster assignments",
        dataframe_to_csv_bytes(cluster_df),
        "cluster_assignments.csv",
        "text/csv",
    )

    st.subheader("Molecular Visualization")
    shown = desc_df.merge(cluster_df, on="molecule_id", how="left")
    for _, row in shown.iterrows():
        with st.expander(f"{row['molecule_id']} | {row['canonical_smiles']}"):
            st.write(f"Cluster: {row.get('cluster', 'N/A')}")
            st.image(molecule_image_png(row["mol"]))

    st.subheader("Additional Visualizations")
    st.pyplot(descriptor_distributions(desc_df, DESCRIPTOR_COLUMNS[:6]), clear_figure=True)
    st.pyplot(similarity_distribution(pair_df), clear_figure=True)
    st.pyplot(pca_scatter(desc_df, cluster_df, DESCRIPTOR_COLUMNS), clear_figure=True)

    st.subheader("Exports")
    st.download_button(
        "Download valid molecules",
        dataframe_to_csv_bytes(validation.valid),
        "valid_molecules.csv",
        "text/csv",
    )
    st.download_button(
        "Download invalid molecules",
        dataframe_to_csv_bytes(validation.invalid),
        "invalid_molecules.csv",
        "text/csv",
    )
    st.download_button(
        "Download ranked similarity pairs",
        dataframe_to_csv_bytes(pair_df),
        "ranked_similarity_pairs.csv",
        "text/csv",
    )


if __name__ == "__main__":
    main()
