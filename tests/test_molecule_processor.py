from data_loader import load_manual_smiles, validate_smiles
from molecule_processor import process_molecules


def test_valid_and_invalid_smiles_are_separated():
    df = load_manual_smiles("CCO\nINVALID\n")
    result = validate_smiles(df)
    assert len(result.valid) == 1
    assert len(result.invalid) == 1
    assert result.invalid.loc[0, "error_category"] == "invalid_smiles"


def test_canonical_smiles_generation():
    df = load_manual_smiles("C(C)O\n")
    result = validate_smiles(df)
    processed = process_molecules(result.valid)
    assert processed.loc[0, "canonical_smiles"] == "CCO"
