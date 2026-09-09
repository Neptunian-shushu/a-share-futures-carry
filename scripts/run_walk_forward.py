"""Run a small leakage-aware walk-forward threshold selection study."""

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
from a_share_futures_carry.metrics.performance import summarize_backtest
from a_share_futures_carry.research.walk_forward import make_walk_forward_windows
from a_share_futures_carry.signals.basis import add_carry_columns
from a_share_futures_carry.strategy.allocation import add_dynamic_carry_allocation
from a_share_futures_carry.strategy.selection import apply_roll_policy, select_max_carry


def _prepare(data: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    carry = cfg["carry"]
    strategy = cfg["strategy"]
    data = add_carry_columns(
        data,
        carry["day_count"],
        use_fair_value_adjustment=carry["use_fair_value_adjustment"],
        funding_rate_annual=carry["funding_rate_annual"],
        dividend_yield_annual=carry["dividend_yield_annual"],
    )
    selected = select_max_carry(
        data,
        tuple(strategy["eligible_families"]),
        strategy["min_dte"],
        strategy["max_dte"],
        strategy["carry_column"],
        strategy["min_volume"],
        strategy["min_open_interest"],
    )
    selected = apply_roll_policy(
        selected,
        data,
        strategy["roll_before_expiry_days"],
        strategy["min_dte"],
        strategy["max_dte"],
        strategy["carry_column"],
        strategy["min_volume"],
        strategy["min_open_interest"],
    )
    return data, selected


def _run(selected: pd.DataFrame, data: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
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
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/strategy.yaml")
    parser.add_argument("--output", default="outputs/walk_forward_summary.csv")
    parser.add_argument("--train-sessions", type=int, default=60)
    parser.add_argument("--test-sessions", type=int, default=30)
    parser.add_argument("--step-sessions", type=int, default=None)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data, selected = _prepare(load_contract_panel_csv(args.data), cfg)
    windows = make_walk_forward_windows(
        selected["trade_date"], args.train_sessions, args.test_sessions, args.step_sessions
    )
    allocation = cfg.get("allocation", {})
    rows: list[dict[str, object]] = []

    for window_id, window in enumerate(windows, start=1):
        train_mask = selected["trade_date"].between(window.train_start, window.train_end)
        test_mask = selected["trade_date"].between(window.train_end, window.test_end)
        candidate_scores: list[tuple[float, float]] = []
        for threshold in args.thresholds:
            allocated = add_dynamic_carry_allocation(
                selected,
                carry_column=cfg["strategy"]["carry_column"],
                lookback=allocation.get("lookback", 60),
                min_periods=min(allocation.get("min_periods", 20), allocation.get("lookback", 60)),
                method=allocation.get("method", "percentile"),
                entry_threshold=threshold,
                zscore_scale=allocation.get("zscore_scale", 1.0),
                max_weight=allocation.get("max_weight", 1.0),
            )
            train_bt = _run(
                allocated[train_mask],
                data[data["trade_date"].between(window.train_start, window.train_end)],
                cfg,
            )
            score = summarize_backtest(train_bt).get("sharpe", np.nan) if not train_bt.empty else np.nan
            candidate_scores.append((threshold, float(score)))

        finite_scores = [(threshold, score) for threshold, score in candidate_scores if np.isfinite(score)]
        chosen_threshold = max(finite_scores, key=lambda item: item[1])[0] if finite_scores else args.thresholds[0]
        allocated = add_dynamic_carry_allocation(
            selected,
            carry_column=cfg["strategy"]["carry_column"],
            lookback=allocation.get("lookback", 60),
            min_periods=min(allocation.get("min_periods", 20), allocation.get("lookback", 60)),
            method=allocation.get("method", "percentile"),
            entry_threshold=chosen_threshold,
            zscore_scale=allocation.get("zscore_scale", 1.0),
            max_weight=allocation.get("max_weight", 1.0),
        )
        test_bt = _run(
            allocated[test_mask],
            data[data["trade_date"].between(window.train_end, window.test_end)],
            cfg,
        )
        test_summary = summarize_backtest(test_bt) if not test_bt.empty else {}
        rows.append(
            {
                "window": window_id,
                "train_start": window.train_start,
                "train_end": window.train_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
                "chosen_threshold": chosen_threshold,
                "train_best_sharpe": max((score for _, score in candidate_scores if np.isfinite(score)), default=np.nan),
                "test_sharpe": test_summary.get("sharpe", np.nan),
                "test_cagr": test_summary.get("cagr", np.nan),
                "test_total_return": test_summary.get("total_return", np.nan),
                "test_max_drawdown": test_summary.get("max_drawdown", np.nan),
                "test_trading_cost": test_summary.get("total_trading_cost", np.nan),
            }
        )

    result = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"\nSaved walk-forward summary to {output}")


if __name__ == "__main__":
    main()
