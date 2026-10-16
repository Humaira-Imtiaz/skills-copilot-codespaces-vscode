from clustering import agglomerative_from_similarity, butina_from_fingerprints
from data_loader import load_manual_smiles, prepare_analysis_set, validate_smiles
from fingerprints import generate_fingerprints
from molecule_processor import process_molecules
from similarity import build_similarity_matrix


def test_agglomerative_returns_assignments():
    df = load_manual_smiles("CCO\nCCCO\nCCCCO\nc1ccccc1\n")
    valid = validate_smiles(df).valid
    fps = generate_fingerprints(process_molecules(valid))
    sim = build_similarity_matrix(fps)
    clusters = agglomerative_from_similarity(sim, n_clusters=2)
    assert len(clusters) == len(sim)
    assert "cluster" in clusters.columns


def test_butina_returns_assignments_and_duplicate_merge_option():
    df = load_manual_smiles("CCO\nC(C)O\nc1ccccc1\n")
    validation = validate_smiles(df)
    merged = prepare_analysis_set(validation.valid, merge_duplicates=True)
    assert len(merged) == 2

    fps = generate_fingerprints(process_molecules(validation.valid))
    clusters = butina_from_fingerprints(fps, cutoff=0.4)
    assert len(clusters) == len(validation.valid)
    assert set(clusters.columns) == {"molecule_id", "cluster"}
