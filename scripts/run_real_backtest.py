"""Run comparable carry strategies on a normalized historical CSV panel."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a_share_futures_carry.backtest.engine import backtest_selected_contracts
from a_share_futures_carry.data.csv_provider import load_contract_panel_csv
from a_share_futures_carry.reporting.report import generate_research_report, spot_benchmark_returns
from a_share_futures_carry.signals.basis import add_carry_columns
from a_share_futures_carry.strategy.allocation import add_dynamic_carry_allocation
from a_share_futures_carry.strategy.selection import (
    apply_roll_policy,
    select_family_max_carry,
    select_max_carry,
    select_nth_expiry,
)


def _with_roll_policy(selected: pd.DataFrame, data: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    strategy = cfg["strategy"]
    return apply_roll_policy(
        selected,
        data,
        strategy["roll_before_expiry_days"],
        strategy["min_dte"],
        strategy["max_dte"],
        strategy["carry_column"],
    )


def _run_one(name: str, selected: pd.DataFrame, data: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    portfolio = cfg["portfolio"]
    return backtest_selected_contracts(
        selected,
        initial_nav=portfolio["initial_nav"],
        max_notional_to_nav=cfg["strategy"]["max_notional_to_nav"],
        collateral_yield_annual=portfolio["collateral_yield_annual"],
        transaction_cost_bps=portfolio["commission_bps"] + portfolio["slippage_bps"],
        market_data=data,
        signal_lag_sessions=portfolio["signal_lag_sessions"],
        margin_rate=portfolio["margin_rate"],
        margin_buffer=portfolio["margin_buffer"],
        integer_contracts=portfolio["integer_contracts"],
        execution_price_col=portfolio["execution_price_col"],
        mark_price_col=portfolio["mark_price_col"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/cffex_panel.csv")
    parser.add_argument("--config", default="configs/strategy.yaml")
    parser.add_argument("--output", default="outputs/strategy_summary.csv")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data = load_contract_panel_csv(args.data)
    carry_cfg = cfg["carry"]
    data = add_carry_columns(
        data,
        carry_cfg["day_count"],
        use_fair_value_adjustment=carry_cfg["use_fair_value_adjustment"],
        funding_rate_annual=carry_cfg["funding_rate_annual"],
        dividend_yield_annual=carry_cfg["dividend_yield_annual"],
    )
    strategy = cfg["strategy"]
    carry_column = strategy["carry_column"]
    backtests: dict[str, pd.DataFrame] = {}
    benchmarks: dict[str, pd.Series] = {}

    for family in ("IF", "IH", "IC", "IM"):
        family_data = data[data["family"] == family]
        if family_data.empty:
            continue
        benchmark = spot_benchmark_returns(data, family)
        candidates = {
            f"{family}_front": select_nth_expiry(family_data, 1, (family,), carry_column=carry_column),
            f"{family}_second": select_nth_expiry(family_data, 2, (family,), carry_column=carry_column),
            f"{family}_max_carry": select_family_max_carry(
                family_data, family, strategy["min_dte"], strategy["max_dte"], carry_column
            ),
        }
        for name, candidate in candidates.items():
            backtests[name] = _run_one(name, _with_roll_policy(candidate, data, cfg), data, cfg)
            benchmarks[name] = benchmark

    dynamic = select_max_carry(
        data,
        tuple(strategy["eligible_families"]),
        strategy["min_dte"],
        strategy["max_dte"],
        carry_column,
    )
    dynamic = _with_roll_policy(dynamic, data, cfg)
    backtests["dynamic_IC_IM_max_carry"] = _run_one("dynamic_IC_IM_max_carry", dynamic, data, cfg)

    allocation_cfg = cfg.get("allocation", {})
    if allocation_cfg.get("enabled", False) and not dynamic.empty:
        dynamic_allocated = add_dynamic_carry_allocation(
            dynamic,
            carry_column=carry_column,
            lookback=allocation_cfg["lookback"],
            min_periods=allocation_cfg["min_periods"],
            method=allocation_cfg["method"],
            entry_threshold=allocation_cfg["entry_threshold"],
            zscore_scale=allocation_cfg["zscore_scale"],
            max_weight=allocation_cfg["max_weight"],
        )
        backtests["dynamic_IC_IM_carry_allocation"] = _run_one(
            "dynamic_IC_IM_carry_allocation", dynamic_allocated, data, cfg
        )

    configured_dir = cfg.get("report", {}).get("output_dir", "outputs")
    output_path = Path(args.output)
    output_dir = Path(args.output_dir) if args.output_dir else Path(configured_dir)
    summary = generate_research_report(
        backtests,
        output_dir,
        benchmarks=benchmarks,
        periods_per_year=cfg["carry"].get("trading_days_per_year", 252),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved research report to {output_dir}")
    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
