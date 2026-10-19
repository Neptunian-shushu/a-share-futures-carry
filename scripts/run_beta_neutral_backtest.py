"""Run a beta-neutral IC/IM carry overlay against the IF front contract."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a_share_futures_carry.backtest.engine import backtest_selected_contracts
from a_share_futures_carry.backtest.neutral import backtest_beta_neutral_pair
from a_share_futures_carry.data.csv_provider import load_contract_panel_csv
from a_share_futures_carry.metrics.performance import summarize_backtest
from a_share_futures_carry.reporting.report import (
    generate_research_report,
    spot_benchmark_returns,
)
from a_share_futures_carry.signals.basis import add_carry_columns, add_cost_adjusted_carry
from a_share_futures_carry.strategy.selection import (
    apply_roll_policy,
    select_front_by_score,
    select_nth_expiry,
)


def _roll(selected: pd.DataFrame, data: pd.DataFrame, cfg: dict, *, front_switch: bool) -> pd.DataFrame:
    strategy = cfg["strategy"]
    front_cfg = cfg.get("front_switch", {})
    return apply_roll_policy(
        selected,
        data,
        front_cfg.get("roll_before_expiry_days", 0) if front_switch else strategy["roll_before_expiry_days"],
        strategy["min_dte"],
        strategy["max_dte"],
        strategy.get("selection_score_column", strategy["carry_column"]),
        strategy["min_volume"],
        strategy["min_open_interest"],
        front_cfg.get("roll_score_buffer", 0.0) if front_switch else strategy.get("roll_score_buffer", 0.0),
        roll_to_nearest_expiry=front_switch,
    )


def _run_single(selected: pd.DataFrame, data: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    p = cfg["portfolio"]
    return backtest_selected_contracts(
        selected,
        initial_nav=p["initial_nav"],
        max_notional_to_nav=cfg["strategy"]["max_notional_to_nav"],
        collateral_yield_annual=p["collateral_yield_annual"],
        transaction_cost_bps=p["commission_bps"] + p["slippage_bps"],
        market_data=data,
        signal_lag_sessions=p["signal_lag_sessions"],
        margin_rate=p["margin_rate"],
        margin_buffer=p["margin_buffer"],
        integer_contracts=p["integer_contracts"],
        execution_price_col=p["execution_price_col"],
        mark_price_col=p["mark_price_col"],
        max_participation_rate=p.get("max_participation_rate"),
        spread_bps_column=p.get("spread_bps_column"),
        default_spread_bps=p.get("default_spread_bps", 0.0),
        require_spread_data=p.get("require_spread_data", False),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/cffex_panel_cffex_public_if_ih_ic_im_long.csv")
    parser.add_argument("--config", default="configs/strategy.yaml")
    parser.add_argument("--output-dir", default="outputs/beta_neutral_report")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    data = load_contract_panel_csv(args.data)
    carry_cfg = cfg["carry"]
    strategy_cfg = cfg["strategy"]
    data = add_carry_columns(
        data,
        carry_cfg["day_count"],
        use_fair_value_adjustment=carry_cfg["use_fair_value_adjustment"],
        funding_rate_annual=carry_cfg["funding_rate_annual"],
        dividend_yield_annual=carry_cfg["dividend_yield_annual"],
    )
    score_column = strategy_cfg.get("selection_score_column", strategy_cfg["carry_column"])
    data = add_cost_adjusted_carry(
        data,
        carry_column=strategy_cfg["carry_column"],
        switch_cost_bps=strategy_cfg.get("switch_cost_bps", 0.0),
        day_count=carry_cfg["day_count"],
        output_column=score_column,
    )

    front_cfg = cfg.get("front_switch", {})
    long_leg = select_front_by_score(
        data,
        tuple(front_cfg.get("eligible_families", ("IC", "IM"))),
        carry_column=score_column,
        min_volume=strategy_cfg["min_volume"],
        min_open_interest=strategy_cfg["min_open_interest"],
    )
    long_leg = _roll(long_leg, data, cfg, front_switch=True)
    hedge_leg = select_nth_expiry(
        data[data["family"].eq("IF")],
        1,
        ("IF",),
        min_dte=strategy_cfg["min_dte"],
        max_dte=strategy_cfg["max_dte"],
        carry_column=score_column,
        min_volume=strategy_cfg["min_volume"],
        min_open_interest=strategy_cfg["min_open_interest"],
    )
    hedge_leg = _roll(hedge_leg, data, cfg, front_switch=False)

    neutral_cfg = cfg.get("neutral_overlay", {})
    p = cfg["portfolio"]
    neutral = backtest_beta_neutral_pair(
        long_leg,
        hedge_leg,
        data,
        initial_nav=p["initial_nav"],
        long_notional_to_nav=neutral_cfg.get("long_notional_to_nav", 0.5),
        gross_notional_to_nav=neutral_cfg.get("gross_notional_to_nav", 1.0),
        collateral_yield_annual=p["collateral_yield_annual"],
        transaction_cost_bps=p["commission_bps"] + p["slippage_bps"],
        signal_lag_sessions=p["signal_lag_sessions"],
        beta_lookback=neutral_cfg.get("beta_lookback", 60),
        beta_min_periods=neutral_cfg.get("beta_min_periods", 20),
        fallback_beta=neutral_cfg.get("fallback_beta", 1.0),
        max_beta=neutral_cfg.get("max_beta", 2.0),
        margin_rate=p["margin_rate"],
        margin_buffer=p["margin_buffer"],
        integer_contracts=p["integer_contracts"],
        execution_price_col=p["execution_price_col"],
        mark_price_col=p["mark_price_col"],
        max_participation_rate=p.get("max_participation_rate"),
        spread_bps_column=p.get("spread_bps_column"),
        default_spread_bps=p.get("default_spread_bps", 0.0),
        require_spread_data=p.get("require_spread_data", False),
    )
    baseline = _run_single(long_leg, data, cfg)
    neutral_name = "dynamic_IC_IM_beta_neutral_IF_hedge"
    baseline_name = "dynamic_IC_IM_front_switch"
    benchmarks = {
        neutral_name: spot_benchmark_returns(data, "IF"),
        baseline_name: spot_benchmark_returns(data, "IF"),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = generate_research_report(
        {baseline_name: baseline, neutral_name: neutral},
        output_dir,
        benchmarks=benchmarks,
    )
    summary.to_csv(output_dir / "strategy_summary.csv", index=False)
    neutral[[
        "trade_date", "long_contract", "hedge_contract", "long_notional", "hedge_notional",
        "gross_notional", "hedge_ratio", "estimated_net_beta_notional", "nav", "return",
        "long_futures_pnl", "hedge_futures_pnl", "basis_pnl", "collateral_pnl", "trading_cost",
    ]].to_csv(output_dir / "neutral_positions.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\n输出目录: {output_dir}")


if __name__ == "__main__":
    main()
