"""CSV fallback loader for normalized futures contract panels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import prepare_contract_data


def load_contract_panel_csv(path: str | Path) -> pd.DataFrame:
    """Load a normalized contract-level CSV and validate required columns."""
    df = pd.read_csv(path)
    return prepare_contract_data(df)
