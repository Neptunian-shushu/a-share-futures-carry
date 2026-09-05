"""Minimal daily futures portfolio simulator."""

from __future__ import annotations

import pandas as pd


def backtest_selected_contracts(
    selected: pd.DataFrame,
    initial_nav: float = 1_000_000.0,
    max_notional_to_nav: float = 1.0,
    collateral_yield_annual: float = 0.015,
    transaction_cost_bps: float = 1.0,
) -> pd.DataFrame:
    """Backtest one selected long contract per date.

    This first-pass engine targets notional exposure as a fraction of NAV and marks
    futures PnL using close-to-close price changes. It intentionally avoids pretending
    that margin equals economic exposure. Production research should model settlement,
    margin cash flows and exact roll execution explicitly.
    """
    df = selected.sort_values("trade_date").copy().reset_index(drop=True)
    if df.empty:
        return df

    nav = initial_nav
    rows = []
    prev_contract = None
    prev_price = None
    prev_multiplier = None
    prev_contracts = 0.0

    for _, row in df.iterrows():
        futures_pnl = 0.0
        trading_cost = 0.0

        if prev_price is not None and row["contract"] == prev_contract:
            futures_pnl = prev_contracts * prev_multiplier * (row["futures_close"] - prev_price)

        nav_before_cost = nav + futures_pnl
        collateral_pnl = nav_before_cost * collateral_yield_annual / 365.0
        nav_before_trade = nav_before_cost + collateral_pnl

        target_notional = nav_before_trade * max_notional_to_nav
        contracts = target_notional / (row["futures_close"] * row["multiplier"])

        if row["contract"] != prev_contract:
            turnover_notional = abs(contracts * row["futures_close"] * row["multiplier"])
            if prev_contract is not None:
                turnover_notional += abs(prev_contracts * prev_price * prev_multiplier)
            trading_cost = turnover_notional * transaction_cost_bps / 10_000.0

        nav = nav_before_trade - trading_cost
        rows.append({
            "trade_date": row["trade_date"],
            "contract": row["contract"],
            "family": row["family"],
            "carry_ann": row.get("carry_ann"),
            "futures_pnl": futures_pnl,
            "collateral_pnl": collateral_pnl,
            "trading_cost": trading_cost,
            "nav": nav,
        })

        prev_contract = row["contract"]
        prev_price = row["futures_close"]
        prev_multiplier = row["multiplier"]
        prev_contracts = contracts

    result = pd.DataFrame(rows)
    result["return"] = result["nav"].pct_change().fillna(0.0)
    return result
