"""Basis and carry calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_discount(spot: pd.Series, futures: pd.Series, dte: pd.Series, day_count: int = 365) -> pd.Series:
    """Annualize the simple futures discount relative to spot.

    Positive values mean the futures contract trades below spot (backwardation/discount).
    """
    dte = dte.astype(float)
    result = (spot - futures) / spot * day_count / dte
    return result.where(dte > 0)


def fair_value_futures(spot: pd.Series, funding_rate: pd.Series, dividend_yield: pd.Series, dte: pd.Series, day_count: int = 365) -> pd.Series:
    """Cost-of-carry fair value using continuously compounded rates."""
    tau = dte / day_count
    return spot * np.exp((funding_rate - dividend_yield) * tau)


def add_carry_columns(df: pd.DataFrame, day_count: int = 365) -> pd.DataFrame:
    """Return a copy with basis and annualized carry columns."""
    out = df.copy()
    out["basis_points"] = out["spot_close"] - out["futures_close"]
    out["basis_pct"] = out["basis_points"] / out["spot_close"]
    out["carry_ann"] = annualized_discount(
        out["spot_close"], out["futures_close"], out["dte"], day_count
    )
    return out
