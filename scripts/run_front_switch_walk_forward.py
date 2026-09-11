"""Run strict walk-forward comparison for the IC/IM front switch candidate."""

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
from a_share_futures_carry.data.csv_provider import load_contract_panel_csv
from a_share_futures_carry.metrics.performance import performance_summary, summarize_backtest
from a_share_futures_carry.research.walk_forward import make_walk_forward_windows
from a_share_futures_carry.signals.basis import add_carry_columns, add_cost_adjusted_carry
from a_share_futures_carry.strategy.selection import (
    apply_roll_policy,
    select_front_by_score,
    select_nth_expiry,
)


def _prepare(data: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    carry = cfg["carry"]
    strategy = cfg["strategy"]
    out = add_carry_columns(
        data,
        carry["day_count"],
        use_fair_value_adjustment=carry["use_fair_value_adjustment"],
        funding_rate_annual=carry["funding_rate_annual"],
        dividend_yield_annual=carry["dividend_yield_annual"],
    )
    return add_cost_adjusted_carry(
        out,
        carry_column=strategy["carry_column"],
        switch_cost_bps=strategy.get("switch_cost_bps", 0.0),
        day_count=carry["day_count"],
        output_column=strategy.get("selection_score_column", strategy["carry_column"]),
    )


def _run(selected: pd.DataFrame, data: pd.DataFrame, cfg: dict) -> pd.DataFrame:
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
    )


def _period(selected: pd.DataFrame, data: pd.DataFrame, cfg: dict, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    full = _run(selected, data, cfg)
    dates = pd.to_datetime(full["trade_date"])
    return full[dates.between(start, end)].copy().reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/strategy.yaml")
    parser.add_argument("--output", default="outputs/front_switch_walk_forward_summary.csv")
    parser.add_argument("--train-sessions", type=int, default=252)
    parser.add_argument("--validation-sessions", type=int, default=63)
    parser.add_argument("--test-sessions", type=int, default=63)
    parser.add_argument("--step-sessions", type=int, default=63)
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data = _prepare(load_contract_panel_csv(args.data), cfg)
    strategy = cfg["strategy"]
    front_cfg = cfg["front_switch"]
    front = select_front_by_score(
        data,
        tuple(front_cfg["eligible_families"]),
        strategy.get("selection_score_column", strategy["carry_column"]),
        strategy["min_volume"],
        strategy["min_open_interest"],
    )
    front = apply_roll_policy(
        front,
        data,
        front_cfg["roll_before_expiry_days"],
        strategy["min_dte"],
        strategy["max_dte"],
        strategy.get("selection_score_column", strategy["carry_column"]),
        strategy["min_volume"],
        strategy["min_open_interest"],
        front_cfg["roll_score_buffer"],
        roll_to_nearest_expiry=True,
    )
    ic = select_nth_expiry(
        data[data["family"].eq("IC")],
        1,
        ("IC",),
        carry_column=strategy["carry_column"],
        min_volume=strategy["min_volume"],
        min_open_interest=strategy["min_open_interest"],
    )
    ic = apply_roll_policy(
        ic,
        data,
        strategy["roll_before_expiry_days"],
        strategy["min_dte"],
        strategy["max_dte"],
        strategy.get("selection_score_column", strategy["carry_column"]),
        strategy["min_volume"],
        strategy["min_open_interest"],
        strategy.get("roll_score_buffer", 0.0),
    )
    windows = make_walk_forward_windows(
        front["trade_date"],
        args.train_sessions,
        args.test_sessions,
        args.step_sessions,
        args.validation_sessions,
    )
    rows: list[dict[str, object]] = []
    front_returns: list[float] = []
    ic_returns: list[float] = []
    for window_id, window in enumerate(windows, start=1):
        front_test = _period(front, data, cfg, window.test_start, window.test_end)
        ic_test = _period(ic, data, cfg, window.test_start, window.test_end)
        front_metrics = summarize_backtest(front_test)
        ic_metrics = summarize_backtest(ic_test)
        front_returns.extend(front_test["return"].tolist())
        ic_returns.extend(ic_test["return"].tolist())
        rows.append({
            "window": window_id,
            "train_start": window.train_start,
            "train_end": window.train_end,
            "validation_start": window.validation_start,
            "validation_end": window.validation_end,
            "test_start": window.test_start,
            "test_end": window.test_end,
            "front_switch_return": front_metrics.get("total_return", np.nan),
            "front_switch_cagr": front_metrics.get("cagr", np.nan),
            "front_switch_sharpe": front_metrics.get("sharpe", np.nan),
            "front_switch_max_drawdown": front_metrics.get("max_drawdown", np.nan),
            "ic_front_return": ic_metrics.get("total_return", np.nan),
            "ic_front_cagr": ic_metrics.get("cagr", np.nan),
            "ic_front_sharpe": ic_metrics.get("sharpe", np.nan),
            "ic_front_max_drawdown": ic_metrics.get("max_drawdown", np.nan),
        })

    result = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    front_summary = performance_summary(pd.Series(front_returns))
    ic_summary = performance_summary(pd.Series(ic_returns))
    print(result.to_string(index=False))
    print("\nAggregate test comparison:")
    print(pd.DataFrame({"front_switch": front_summary, "IC_front": ic_summary}).T.to_string())
    print(f"\nSaved front-switch walk-forward summary to {output}")


if __name__ == "__main__":
    main()
