"""Input-data validation helpers."""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {
    "trade_date",
    "contract",
    "family",
    "futures_close",
    "spot_close",
    "expiry_date",
    "multiplier",
}


def prepare_contract_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize daily contract-level futures data."""
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    out["expiry_date"] = pd.to_datetime(out["expiry_date"])
    out["dte"] = (out["expiry_date"] - out["trade_date"]).dt.days
    out = out[out["dte"] >= 0].sort_values(["trade_date", "family", "expiry_date"])
    return out.reset_index(drop=True)
