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
from a_share_futures_carry.reporting.report import (
    generate_research_report,
    price_benchmark_backtest,
    price_benchmark_returns,
    spot_benchmark_backtest,
    spot_benchmark_returns,
)
from a_share_futures_carry.signals.basis import add_carry_columns, add_cost_adjusted_carry
from a_share_futures_carry.strategy.allocation import add_dynamic_carry_allocation, add_volatility_target_allocation
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
        strategy.get("roll_score_column", strategy.get("selection_score_column", strategy["carry_column"])),
        strategy["min_volume"],
        strategy["min_open_interest"],
        strategy.get("roll_score_buffer", 0.0),
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
        max_participation_rate=portfolio.get("max_participation_rate"),
        spread_bps_column=portfolio.get("spread_bps_column"),
        default_spread_bps=portfolio.get("default_spread_bps", 0.0),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/cffex_panel.csv")
    parser.add_argument("--config", default="configs/strategy.yaml")
    parser.add_argument("--output", default="outputs/strategy_summary.csv")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Optional ETF/index CSV with a dated price series to add to the report",
    )
    parser.add_argument("--benchmark-name", default="external_price_benchmark")
    parser.add_argument("--benchmark-date-column", default="trade_date")
    parser.add_argument("--benchmark-price-column", default="close")
    parser.add_argument(
        "--benchmark-distribution-column",
        default=None,
        help="Optional per-share cash distribution column for a total-return benchmark",
    )
    parser.add_argument(
        "--carry-mode",
        choices=("config", "observed", "fair"),
        default="config",
        help="Use configured, observed, or fair-value-adjusted carry",
    )
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data = load_contract_panel_csv(args.data)
    carry_cfg = cfg["carry"]
    use_fair_value = carry_cfg["use_fair_value_adjustment"]
    if args.carry_mode == "observed":
        use_fair_value = False
    elif args.carry_mode == "fair":
        use_fair_value = True
    data = add_carry_columns(
        data,
        carry_cfg["day_count"],
        use_fair_value_adjustment=use_fair_value,
        funding_rate_annual=carry_cfg["funding_rate_annual"],
        dividend_yield_annual=carry_cfg["dividend_yield_annual"],
    )
    strategy = cfg["strategy"]
    carry_column = strategy["carry_column"]
    selection_score_column = strategy.get("selection_score_column", carry_column)
    data = add_cost_adjusted_carry(
        data,
        carry_column=carry_column,
        switch_cost_bps=strategy.get("switch_cost_bps", 0.0),
        day_count=carry_cfg["day_count"],
        output_column=selection_score_column,
    )
    backtests: dict[str, pd.DataFrame] = {}
    benchmarks: dict[str, pd.Series] = {}

    for family in ("IF", "IH", "IC", "IM"):
        family_data = data[data["family"] == family]
        if family_data.empty:
            continue
        benchmark = spot_benchmark_returns(data, family)
        backtests[f"{family}_spot_benchmark"] = spot_benchmark_backtest(
            data, family, cfg["portfolio"]["initial_nav"]
        )
        candidates = {
            f"{family}_front": select_nth_expiry(
                family_data,
                1,
                (family,),
                carry_column=carry_column,
                min_volume=strategy["min_volume"],
                min_open_interest=strategy["min_open_interest"],
            ),
            f"{family}_second": select_nth_expiry(
                family_data,
                2,
                (family,),
                carry_column=carry_column,
                min_volume=strategy["min_volume"],
                min_open_interest=strategy["min_open_interest"],
            ),
            f"{family}_max_carry": select_family_max_carry(
                family_data,
                family,
                strategy["min_dte"],
                strategy["max_dte"],
                selection_score_column,
                strategy["min_volume"],
                strategy["min_open_interest"],
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
        selection_score_column,
        strategy["min_volume"],
        strategy["min_open_interest"],
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
        if allocation_cfg.get("vol_target_enabled", False):
            dynamic_vol_targeted = add_volatility_target_allocation(
                dynamic_allocated,
                target_vol_annual=allocation_cfg.get("target_vol_annual", 0.10),
                lookback=allocation_cfg.get("vol_lookback", allocation_cfg["lookback"]),
                min_periods=allocation_cfg.get("vol_min_periods", allocation_cfg["min_periods"]),
                price_column=allocation_cfg.get("vol_price_column", "spot_close"),
                max_weight=allocation_cfg.get("max_weight", 1.0),
                periods_per_year=carry_cfg.get("trading_days_per_year", 252),
            )
            backtests["dynamic_IC_IM_carry_vol_target"] = _run_one(
                "dynamic_IC_IM_carry_vol_target", dynamic_vol_targeted, data, cfg
            )

    if args.benchmark:
        benchmark_prices = pd.read_csv(args.benchmark)
        external_returns = price_benchmark_returns(
            benchmark_prices,
            date_column=args.benchmark_date_column,
            price_column=args.benchmark_price_column,
            distribution_column=args.benchmark_distribution_column,
        )
        backtests[args.benchmark_name] = price_benchmark_backtest(
            benchmark_prices,
            cfg["portfolio"]["initial_nav"],
            date_column=args.benchmark_date_column,
            price_column=args.benchmark_price_column,
            distribution_column=args.benchmark_distribution_column,
        )
        for name in list(backtests):
            if name != args.benchmark_name and not name.endswith("_spot_benchmark"):
                benchmarks[name] = external_returns

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
