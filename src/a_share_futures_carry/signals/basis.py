"""Basis and carry calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_discount(spot: pd.Series, futures: pd.Series, dte: pd.Series, day_count: int = 365) -> pd.Series:
    """Annualize the simple futures discount relative to spot.

    Positive values mean the futures contract trades below spot (backwardation/discount).
    """
    spot = pd.to_numeric(spot, errors="coerce")
    futures = pd.to_numeric(futures, errors="coerce")
    dte = pd.to_numeric(dte, errors="coerce").astype(float)
    valid = spot.gt(0) & futures.notna() & dte.gt(0) & (day_count > 0)
    result = (spot - futures) / spot * day_count / dte
    return result.where(valid)


def fair_value_futures(spot: pd.Series, funding_rate: pd.Series, dividend_yield: pd.Series, dte: pd.Series, day_count: int = 365) -> pd.Series:
    """Cost-of-carry fair value using continuously compounded rates."""
    tau = pd.to_numeric(dte, errors="coerce") / day_count
    return spot * np.exp((funding_rate - dividend_yield) * tau)


def fair_value_carry(
    spot: pd.Series,
    funding_rate: pd.Series,
    dividend_yield: pd.Series,
    dte: pd.Series,
    day_count: int = 365,
) -> pd.Series:
    """Return the annualized discount implied by the cost-of-carry fair value."""
    fair = fair_value_futures(spot, funding_rate, dividend_yield, dte, day_count)
    return annualized_discount(spot, fair, dte, day_count)


def add_carry_columns(
    df: pd.DataFrame,
    day_count: int = 365,
    *,
    use_fair_value_adjustment: bool = False,
    funding_rate_annual: float = 0.0,
    dividend_yield_annual: float = 0.0,
) -> pd.DataFrame:
    """Return a copy with observed and fair-value-adjusted carry columns.

    When enabled, optional row-level ``funding_rate`` and ``dividend_yield``
    columns take precedence over the scalar defaults.  ``signal_carry`` is the
    column that selectors should use, making the choice explicit in outputs.
    """
    out = df.copy()
    out["basis_points"] = out["spot_close"] - out["futures_close"]
    out["basis_pct"] = out["basis_points"] / out["spot_close"]
    out["carry_ann"] = annualized_discount(
        out["spot_close"], out["futures_close"], out["dte"], day_count
    )
    out["signal_carry"] = out["carry_ann"]

    if use_fair_value_adjustment:
        funding = (
            pd.to_numeric(out["funding_rate"], errors="coerce").fillna(funding_rate_annual)
            if "funding_rate" in out
            else pd.Series(funding_rate_annual, index=out.index, dtype=float)
        )
        dividend = (
            pd.to_numeric(out["dividend_yield"], errors="coerce").fillna(dividend_yield_annual)
            if "dividend_yield" in out
            else pd.Series(dividend_yield_annual, index=out.index, dtype=float)
        )
        out["fair_value_futures"] = fair_value_futures(
            out["spot_close"], funding, dividend, out["dte"], day_count
        )
        out["fair_value_carry"] = fair_value_carry(
            out["spot_close"], funding, dividend, out["dte"], day_count
        )
        out["excess_carry"] = out["carry_ann"] - out["fair_value_carry"]
        out["signal_carry"] = out["excess_carry"]
    return out


def add_cost_adjusted_carry(
    df: pd.DataFrame,
    *,
    carry_column: str = "signal_carry",
    switch_cost_bps: float = 0.0,
    day_count: int = 365,
    dte_column: str = "dte",
    output_column: str = "selection_score",
) -> pd.DataFrame:
    """Add a carry score after charging an estimated contract-switch cost.

    ``switch_cost_bps`` is the estimated round-trip cost of closing the old
    contract and opening the new one.  The cost is annualized over the
    contract's remaining days, which makes the score conservative for
    contracts that would need to be rolled soon.  The calculation is a
    research approximation; actual execution costs still belong in the
    backtest engine.
    """
    if carry_column not in df:
        raise ValueError(f"Missing carry column: {carry_column}")
    if switch_cost_bps < 0:
        raise ValueError("switch_cost_bps must be non-negative")
    if day_count <= 0:
        raise ValueError("day_count must be positive")
    if dte_column not in df:
        raise ValueError(f"Missing DTE column: {dte_column}")

    out = df.copy()
    carry = pd.to_numeric(out[carry_column], errors="coerce")
    dte = pd.to_numeric(out[dte_column], errors="coerce")
    annualized_switch_cost = (switch_cost_bps / 10_000.0) * day_count / dte.where(dte > 0)
    out[output_column] = carry - annualized_switch_cost
    out[output_column] = out[output_column].where(carry.notna() & dte.gt(0))
    out["switch_cost_ann"] = annualized_switch_cost.where(dte.gt(0))
    return out
