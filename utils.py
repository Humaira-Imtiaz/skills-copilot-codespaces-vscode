"""Utility helpers for the molecular similarity application."""

from __future__ import annotations

from io import BytesIO
from typing import Iterable

import pandas as pd


def generate_stable_ids(size: int, prefix: str = "MOL") -> list[str]:
    """Generate stable molecule IDs like MOL_001."""
    return [f"{prefix}_{i:03d}" for i in range(1, size + 1)]


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to CSV bytes for download widgets."""
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()


def required_columns_present(df: pd.DataFrame, columns: Iterable[str]) -> bool:
    """Check whether required columns are present."""
    return all(col in df.columns for col in columns)
