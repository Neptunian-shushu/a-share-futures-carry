"""Contract-selection rules for carry strategies."""

from __future__ import annotations

import pandas as pd


def _eligible(
    contracts: pd.DataFrame,
    eligible_families: tuple[str, ...],
    min_dte: int,
    max_dte: int,
) -> pd.DataFrame:
    return contracts[
        contracts["family"].isin(eligible_families)
        & contracts["dte"].between(min_dte, max_dte)
        & contracts["carry_ann"].notna()
    ].copy()


def select_max_carry(
    contracts: pd.DataFrame,
    eligible_families: tuple[str, ...] = ("IC", "IM"),
    min_dte: int = 5,
    max_dte: int = 120,
) -> pd.DataFrame:
    """Select the highest annualized-carry eligible contract each trade date."""
    eligible = _eligible(contracts, eligible_families, min_dte, max_dte)
    if eligible.empty:
        return eligible
    idx = eligible.groupby("trade_date")["carry_ann"].idxmax()
    return eligible.loc[idx].sort_values("trade_date").reset_index(drop=True)


def select_nth_expiry(
    contracts: pd.DataFrame,
    n: int,
    eligible_families: tuple[str, ...] = ("IC",),
    min_dte: int = 1,
    max_dte: int = 180,
) -> pd.DataFrame:
    """Select the nth nearest eligible expiry within each family/date.

    n=1 is the front month; n=2 is the second month. If several families are supplied,
    one row per family/date is returned so callers can backtest each family separately.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    eligible = _eligible(contracts, eligible_families, min_dte, max_dte)
    if eligible.empty:
        return eligible

    eligible = eligible.sort_values(["trade_date", "family", "expiry_date"])
    eligible["expiry_rank"] = eligible.groupby(["trade_date", "family"]).cumcount() + 1
    return eligible[eligible["expiry_rank"] == n].drop(columns="expiry_rank").reset_index(drop=True)


def select_family_max_carry(
    contracts: pd.DataFrame,
    family: str,
    min_dte: int = 5,
    max_dte: int = 120,
) -> pd.DataFrame:
    """Select the best-carry contract within one index-futures family."""
    return select_max_carry(contracts, (family,), min_dte, max_dte)
