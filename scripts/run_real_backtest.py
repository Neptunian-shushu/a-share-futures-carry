"""Run carry-strategy comparisons on a normalized real-data CSV panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from a_share_futures_carry.backtest.engine import backtest_selected_contracts
from a_share_futures_carry.data.csv_provider import load_contract_panel_csv
from a_share_futures_carry.metrics.performance import performance_summary
from a_share_futures_carry.signals.basis import add_carry_columns
from a_share_futures_carry.strategy.selection import (
    select_family_max_carry,
    select_max_carry,
    select_nth_expiry,
)


def run_one(name: str, selected: pd.DataFrame, cfg: dict) -> dict:
    p = cfg["portfolio"]
    s = cfg["strategy"]
    if selected.empty:
        return {"strategy": name, "n_obs": 0}

    bt = backtest_selected_contracts(
        selected,
        initial_nav=p["initial_nav"],
        max_notional_to_nav=s["max_notional_to_nav"],
        collateral_yield_annual=p["collateral_yield_annual"],
        transaction_cost_bps=p["commission_bps"] + p["slippage_bps"],
    )
    stats = performance_summary(bt["return"])
    return {"strategy": name, "n_obs": len(bt), **stats}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/cffex_panel.csv")
    parser.add_argument("--config", default="configs/strategy.yaml")
    parser.add_argument("--output", default="outputs/strategy_summary.csv")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data = load_contract_panel_csv(args.data)
    data = add_carry_columns(data, cfg["carry"]["day_count"])
    s = cfg["strategy"]

    results: list[dict] = []
    for family in ("IF", "IH", "IC", "IM"):
        family_data = data[data["family"] == family]
        if family_data.empty:
            continue
        results.append(run_one(f"{family}_front", select_nth_expiry(family_data, 1, (family,)), cfg))
        results.append(run_one(f"{family}_second", select_nth_expiry(family_data, 2, (family,)), cfg))
        results.append(
            run_one(
                f"{family}_max_carry",
                select_family_max_carry(family_data, family, s["min_dte"], s["max_dte"]),
                cfg,
            )
        )

    dynamic = select_max_carry(
        data,
        tuple(s["eligible_families"]),
        s["min_dte"],
        s["max_dte"],
    )
    results.append(run_one("dynamic_IC_IM_max_carry", dynamic, cfg))

    summary = pd.DataFrame(results)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved summary to {output}")


if __name__ == "__main__":
    main()
