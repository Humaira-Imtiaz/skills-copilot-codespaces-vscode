import numpy as np

from data_loader import load_manual_smiles, validate_smiles
from fingerprints import generate_fingerprints
from molecule_processor import process_molecules
from similarity import build_similarity_matrix, ranked_similarity_pairs


def test_fingerprint_size_and_identity_similarity():
    df = load_manual_smiles("CCO\nCCO\n")
    valid = validate_smiles(df).valid
    processed = process_molecules(valid)
    fps = generate_fingerprints(processed, n_bits=2048)
    assert fps.loc[0, "fingerprint"].GetNumBits() == 2048

    sim = build_similarity_matrix(fps)
    assert np.allclose(np.diag(sim.values), 1.0)
    assert sim.iloc[0, 1] == 1.0


def test_similarity_matrix_and_ranked_pairs_requirements():
    df = load_manual_smiles("CCO\nCCCO\nc1ccccc1\n")
    valid = validate_smiles(df).valid
    fps = generate_fingerprints(process_molecules(valid))
    sim = build_similarity_matrix(fps)
    assert np.allclose(sim.values, sim.values.T)
    assert ((sim.values >= 0) & (sim.values <= 1)).all()

    ranked = ranked_similarity_pairs(sim)
    assert all(ranked["molecule_a"] != ranked["molecule_b"])
    seen = set()
    for _, row in ranked.iterrows():
        pair = tuple(sorted([row["molecule_a"], row["molecule_b"]]))
        assert pair not in seen
        seen.add(pair)
