"""Run a minimal synthetic-data carry backtest.

Replace ``make_synthetic_data`` with a real data loader once historical contract data
is connected. Keeping the first version self-contained makes it easy to test the
research pipeline end to end.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a_share_futures_carry.backtest.engine import backtest_selected_contracts
from a_share_futures_carry.data.schema import prepare_contract_data
from a_share_futures_carry.metrics.performance import summarize_backtest
from a_share_futures_carry.signals.basis import add_carry_columns
from a_share_futures_carry.strategy.allocation import add_dynamic_carry_allocation
from a_share_futures_carry.strategy.selection import apply_roll_policy, select_max_carry


def _third_friday(year: int, month: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    days_to_friday = (4 - first.weekday()) % 7
    return first + pd.Timedelta(days=days_to_friday + 14)


def make_synthetic_data() -> pd.DataFrame:
    """Create a deterministic panel with realistic contract lifecycles."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2025-01-02", periods=120)
    expiries = [
        _third_friday(year, month)
        for year in (2024, 2025, 2026)
        for month in range(1, 13)
    ]
    rows = []
    spot = {"IC": 5600.0, "IM": 6100.0}
    multiplier = {"IC": 200, "IM": 200}

    for date in dates:
        for family in ("IC", "IM"):
            spot[family] *= 1 + rng.normal(0.0002, 0.009)
            future_expiries = [expiry for expiry in expiries if expiry > date][:4]
            for rank, expiry in enumerate(future_expiries, start=1):
                discount = 0.004 + rank * 0.003
                price = spot[family] * (1 - discount + rng.normal(0, 0.0008))
                rows.append({
                    "trade_date": date,
                    "contract": f"{family}{expiry:%y%m}",
                    "family": family,
                    "futures_close": price,
                    "settle": price,
                    "spot_close": spot[family],
                    "expiry_date": expiry,
                    "multiplier": multiplier[family],
                    "vol": 10000 - rank * 500,
                    "oi": 50000 - rank * 2500,
                    "funding_rate": 0.02,
                    "dividend_yield": 0.015,
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/strategy.yaml")
    parser.add_argument("--data-output", default=None, help="Optional path for the generated synthetic panel")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data = prepare_contract_data(make_synthetic_data())
    if args.data_output:
        data_output = Path(args.data_output)
        data_output.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(data_output, index=False)
        print(f"Saved synthetic panel to {data_output}")
    carry_cfg = cfg["carry"]
    data = add_carry_columns(
        data,
        carry_cfg["day_count"],
        use_fair_value_adjustment=carry_cfg["use_fair_value_adjustment"],
        funding_rate_annual=carry_cfg["funding_rate_annual"],
        dividend_yield_annual=carry_cfg["dividend_yield_annual"],
    )
    s = cfg["strategy"]
    selected = select_max_carry(
        data,
        tuple(s["eligible_families"]),
        s["min_dte"],
        s["max_dte"],
        s["carry_column"],
        s["min_volume"],
        s["min_open_interest"],
    )
    selected = apply_roll_policy(
        selected,
        data,
        s["roll_before_expiry_days"],
        s["min_dte"],
        s["max_dte"],
        s["carry_column"],
        s["min_volume"],
        s["min_open_interest"],
    )
    allocation_cfg = cfg.get("allocation", {})
    if allocation_cfg.get("enabled", False):
        selected = add_dynamic_carry_allocation(
            selected,
            carry_column=s["carry_column"],
            lookback=allocation_cfg["lookback"],
            min_periods=allocation_cfg["min_periods"],
            method=allocation_cfg["method"],
            entry_threshold=allocation_cfg["entry_threshold"],
            zscore_scale=allocation_cfg["zscore_scale"],
            max_weight=allocation_cfg["max_weight"],
        )

    p = cfg["portfolio"]
    bt = backtest_selected_contracts(
        selected,
        initial_nav=p["initial_nav"],
        max_notional_to_nav=s["max_notional_to_nav"],
        collateral_yield_annual=p["collateral_yield_annual"],
        transaction_cost_bps=p["commission_bps"] + p["slippage_bps"],
        market_data=data,
        signal_lag_sessions=p["signal_lag_sessions"],
        margin_rate=p["margin_rate"],
        margin_buffer=p["margin_buffer"],
        integer_contracts=p["integer_contracts"],
        execution_price_col=p["execution_price_col"],
        mark_price_col=p["mark_price_col"],
    )
    summary = summarize_backtest(bt)
    print(pd.Series(summary).to_string())
    output_dir = Path(cfg.get("report", {}).get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    bt.to_csv(output_dir / "synthetic_equity_curve.csv", index=False)
    print(f"\nSaved equity curve to {output_dir / 'synthetic_equity_curve.csv'}")


if __name__ == "__main__":
    main()
