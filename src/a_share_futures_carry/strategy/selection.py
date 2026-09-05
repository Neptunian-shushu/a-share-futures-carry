"""Contract-selection rules for carry strategies."""

from __future__ import annotations

import pandas as pd


def select_max_carry(
    contracts: pd.DataFrame,
    eligible_families: tuple[str, ...] = ("IC", "IM"),
    min_dte: int = 5,
    max_dte: int = 120,
) -> pd.DataFrame:
    """Select the highest annualized-carry eligible contract each trade date."""
    eligible = contracts[
        contracts["family"].isin(eligible_families)
        & contracts["dte"].between(min_dte, max_dte)
        & contracts["carry_ann"].notna()
    ].copy()

    if eligible.empty:
        return eligible

    idx = eligible.groupby("trade_date")["carry_ann"].idxmax()
    return eligible.loc[idx].sort_values("trade_date").reset_index(drop=True)
