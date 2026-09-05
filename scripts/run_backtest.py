"""Run a minimal synthetic-data carry backtest.

Replace ``make_synthetic_data`` with a real data loader once historical contract data
is connected. Keeping the first version self-contained makes it easy to test the
research pipeline end to end.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from a_share_futures_carry.backtest.engine import backtest_selected_contracts
from a_share_futures_carry.data.schema import prepare_contract_data
from a_share_futures_carry.metrics.performance import performance_summary
from a_share_futures_carry.signals.basis import add_carry_columns
from a_share_futures_carry.strategy.selection import select_max_carry


def make_synthetic_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2025-01-02", periods=120)
    rows = []
    spot = {"IC": 5600.0, "IM": 6100.0}
    multiplier = {"IC": 200, "IM": 200}

    for date in dates:
        for family in ("IC", "IM"):
            spot[family] *= 1 + rng.normal(0.0002, 0.009)
            for months, discount in ((1, 0.006), (2, 0.010), (3, 0.013)):
                expiry = date + pd.Timedelta(days=30 * months)
                price = spot[family] * (1 - discount + rng.normal(0, 0.0008))
                rows.append({
                    "trade_date": date,
                    "contract": f"{family}-{expiry:%Y%m%d}",
                    "family": family,
                    "futures_close": price,
                    "spot_close": spot[family],
                    "expiry_date": expiry,
                    "multiplier": multiplier[family],
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/strategy.yaml")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data = prepare_contract_data(make_synthetic_data())
    data = add_carry_columns(data, cfg["carry"]["day_count"])
    s = cfg["strategy"]
    selected = select_max_carry(
        data,
        tuple(s["eligible_families"]),
        s["min_dte"],
        s["max_dte"],
    )

    p = cfg["portfolio"]
    bt = backtest_selected_contracts(
        selected,
        initial_nav=p["initial_nav"],
        max_notional_to_nav=s["max_notional_to_nav"],
        collateral_yield_annual=p["collateral_yield_annual"],
        transaction_cost_bps=p["commission_bps"] + p["slippage_bps"],
    )

    print(pd.Series(performance_summary(bt["return"])).to_string())


if __name__ == "__main__":
    main()
