from data_loader import load_manual_smiles, validate_smiles
from descriptors import DESCRIPTOR_COLUMNS, calculate_descriptors
from molecule_processor import process_molecules


def test_descriptor_columns_present():
    df = load_manual_smiles("CCO\nCCCO\n")
    valid = validate_smiles(df).valid
    processed = process_molecules(valid)
    desc = calculate_descriptors(processed)
    for col in DESCRIPTOR_COLUMNS:
        assert col in desc.columns
    assert desc["MolWt"].notna().all()
