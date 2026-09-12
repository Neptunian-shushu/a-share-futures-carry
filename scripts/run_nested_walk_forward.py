"""Nested walk-forward selection for carry score and risk-overlay parameters."""

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
from a_share_futures_carry.signals.basis import add_carry_columns, add_cost_adjusted_carry, add_net_carry_score
from a_share_futures_carry.strategy.risk import add_beta_target_allocation, add_market_regime_filter
from a_share_futures_carry.strategy.selection import apply_roll_policy, select_front_by_score


def _prepare(data: pd.DataFrame, cfg: dict, carry_mode: str) -> pd.DataFrame:
    carry = cfg["carry"]
    strategy = cfg["strategy"]
    use_fair_value = carry["use_fair_value_adjustment"]
    if carry_mode == "observed":
        use_fair_value = False
    elif carry_mode == "fair":
        use_fair_value = True
    out = add_carry_columns(
        data,
        carry["day_count"],
        use_fair_value_adjustment=use_fair_value,
        funding_rate_annual=carry["funding_rate_annual"],
        dividend_yield_annual=carry["dividend_yield_annual"],
    )
    out = add_cost_adjusted_carry(
        out,
        carry_column=strategy["carry_column"],
        switch_cost_bps=strategy.get("switch_cost_bps", 0.0),
        day_count=carry["day_count"],
        output_column=strategy.get("selection_score_column", strategy["carry_column"]),
    )
    net_cfg = cfg.get("net_carry", {})
    if net_cfg.get("enabled", False):
        out = add_net_carry_score(
            out,
            carry_column="carry_ann",
            funding_rate_annual=net_cfg.get("funding_rate_annual", carry["funding_rate_annual"]),
            dividend_yield_annual=net_cfg.get("dividend_yield_annual", carry["dividend_yield_annual"]),
            switch_cost_bps=net_cfg.get("switch_cost_bps", strategy.get("switch_cost_bps", 0.0)),
            day_count=carry["day_count"],
            output_column=net_cfg.get("output_column", "net_carry_score"),
        )
    return out


def _select_front(data: pd.DataFrame, cfg: dict, score_mode: str, switch_buffer: float) -> pd.DataFrame:
    strategy = cfg["strategy"]
    front_cfg = cfg["front_switch"]
    if score_mode == "net":
        score_column = cfg.get("net_carry", {}).get("output_column", "net_carry_score")
    else:
        score_column = strategy.get("selection_score_column", strategy["carry_column"])
    selected = select_front_by_score(
        data,
        tuple(front_cfg.get("eligible_families", strategy["eligible_families"])),
        carry_column=score_column,
        min_volume=strategy["min_volume"],
        min_open_interest=strategy["min_open_interest"],
    )
    return apply_roll_policy(
        selected,
        data,
        roll_before_expiry_days=front_cfg.get("roll_before_expiry_days", 0),
        min_dte=strategy["min_dte"],
        max_dte=strategy["max_dte"],
        score_column=score_column,
        min_volume=strategy["min_volume"],
        min_open_interest=strategy["min_open_interest"],
        min_score_improvement=switch_buffer,
        roll_to_nearest_expiry=True,
    )


def _apply_overlay(selected: pd.DataFrame, cfg: dict, beta_target: float | None, downtrend_weight: float, high_volatility_weight: float) -> pd.DataFrame:
    risk = cfg.get("risk_control", {})
    out = selected
    if beta_target is not None:
        out = add_beta_target_allocation(
            out,
            target_beta=beta_target,
            lookback=risk.get("beta_lookback", 60),
            min_periods=risk.get("beta_min_periods", 20),
            max_weight=risk.get("max_weight", 1.0),
        )
    if downtrend_weight < 1.0 or high_volatility_weight < 1.0:
        out = add_market_regime_filter(
            out,
            momentum_lookback=risk.get("momentum_lookback", 63),
            volatility_lookback=risk.get("volatility_lookback", 20),
            volatility_quantile_lookback=risk.get("volatility_quantile_lookback", 252),
            volatility_quantile=risk.get("volatility_quantile", 0.8),
            downtrend_weight=downtrend_weight,
            high_volatility_weight=high_volatility_weight,
            max_weight=risk.get("max_weight", 1.0),
            periods_per_year=cfg["carry"].get("trading_days_per_year", 252),
        )
    return out


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


def _period(backtest: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if backtest.empty:
        return backtest
    dates = pd.to_datetime(backtest["trade_date"])
    return backtest[dates.between(start, end)].copy().reset_index(drop=True)


def _monthly_positive_rate(backtest: pd.DataFrame) -> float:
    if backtest.empty:
        return np.nan
    monthly = (
        backtest.assign(month=pd.to_datetime(backtest["trade_date"]).dt.to_period("M"))
        .groupby("month")["return"]
        .apply(lambda values: (1.0 + values).prod() - 1.0)
    )
    return float((monthly > 0).mean()) if not monthly.empty else np.nan


def _validation_score(backtest: pd.DataFrame) -> tuple[float, dict[str, float]]:
    metrics = summarize_backtest(backtest) if not backtest.empty else {}
    monthly_positive = _monthly_positive_rate(backtest)
    sharpe = float(metrics.get("sharpe", np.nan))
    drawdown = abs(float(metrics.get("max_drawdown", np.nan)))
    if not np.isfinite(sharpe):
        return -np.inf, {"sharpe": np.nan, "max_drawdown": np.nan, "monthly_positive_rate": monthly_positive}
    # Validation selection favors risk-adjusted returns and stable months,
    # while charging a modest penalty for drawdown.
    score = sharpe + 0.5 * (monthly_positive if np.isfinite(monthly_positive) else 0.0) - 0.5 * drawdown
    return score, {
        "sharpe": sharpe,
        "cagr": float(metrics.get("cagr", np.nan)),
        "max_drawdown": float(metrics.get("max_drawdown", np.nan)),
        "monthly_positive_rate": monthly_positive,
    }


def _candidate_grid() -> list[dict[str, object]]:
    grid: list[dict[str, object]] = []
    for score_mode in ("observed", "net"):
        for switch_buffer in (0.0, 0.005, 0.02, 0.03):
            for beta_target in (None, 0.6, 0.9):
                for downtrend_weight in (1.0, 0.75):
                    for high_volatility_weight in (1.0, 0.75):
                        grid.append(
                            {
                                "score_mode": score_mode,
                                "switch_buffer": switch_buffer,
                                "beta_target": beta_target,
                                "downtrend_weight": downtrend_weight,
                                "high_volatility_weight": high_volatility_weight,
                            }
                        )
    return grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/strategy.yaml")
    parser.add_argument("--output-dir", default="outputs/nested_walk_forward")
    parser.add_argument("--carry-mode", choices=("config", "observed", "fair"), default="config")
    parser.add_argument("--train-sessions", type=int, default=252)
    parser.add_argument("--validation-sessions", type=int, default=63)
    parser.add_argument("--test-sessions", type=int, default=63)
    parser.add_argument("--step-sessions", type=int, default=63)
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    data = _prepare(load_contract_panel_csv(args.data), cfg, args.carry_mode)
    calendar = pd.DatetimeIndex(sorted(pd.to_datetime(data["trade_date"]).unique()))
    windows = make_walk_forward_windows(
        calendar,
        train_sessions=args.train_sessions,
        validation_sessions=args.validation_sessions,
        test_sessions=args.test_sessions,
        step_sessions=args.step_sessions,
    )
    candidates = _candidate_grid()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_rows: list[dict[str, object]] = []
    selected_test_returns: list[pd.DataFrame] = []
    baseline_test_returns: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    # Each candidate is deterministic for the full input panel. Cache these
    # paths once; the rolling windows only slice validation/test dates.
    candidate_paths: list[tuple[dict[str, object], pd.DataFrame]] = []
    for candidate in candidates:
        selected = _select_front(data, cfg, str(candidate["score_mode"]), float(candidate["switch_buffer"]))
        selected = _apply_overlay(
            selected,
            cfg,
            candidate["beta_target"],
            float(candidate["downtrend_weight"]),
            float(candidate["high_volatility_weight"]),
        )
        candidate_paths.append((candidate, _run(selected, data, cfg)))
    baseline_selected = _select_front(data, cfg, "observed", cfg["front_switch"].get("roll_score_buffer", 0.0))
    baseline_full = _run(baseline_selected, data, cfg)

    for window_id, window in enumerate(windows, start=1):
        candidate_results: list[tuple[float, dict[str, object], dict[str, float], pd.DataFrame]] = []
        for candidate, full_backtest in candidate_paths:
            validation = _period(full_backtest, window.validation_start or window.train_start, window.validation_end or window.train_end)
            score, validation_metrics = _validation_score(validation)
            validation_rows.append(
                {
                    "window": window_id,
                    **candidate,
                    "validation_score": score,
                    **validation_metrics,
                }
            )
            candidate_results.append((score, candidate, validation_metrics, full_backtest))

        finite = [item for item in candidate_results if np.isfinite(item[0])]
        chosen_score, chosen, chosen_validation, chosen_full = max(
            finite,
            key=lambda item: item[0],
        ) if finite else candidate_results[0]
        chosen_test = _period(chosen_full, window.test_start, window.test_end)

        baseline_test = _period(baseline_full, window.test_start, window.test_end)
        selected_test_returns.append(chosen_test[["trade_date", "return"]])
        baseline_test_returns.append(baseline_test[["trade_date", "return"]])
        chosen_metrics = summarize_backtest(chosen_test) if not chosen_test.empty else {}
        baseline_metrics = summarize_backtest(baseline_test) if not baseline_test.empty else {}
        summary_rows.append(
            {
                "window": window_id,
                "train_start": window.train_start,
                "train_end": window.train_end,
                "validation_start": window.validation_start,
                "validation_end": window.validation_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
                "chosen_validation_score": chosen_score,
                **{f"chosen_{key}": value for key, value in chosen.items()},
                **{f"validation_{key}": value for key, value in chosen_validation.items()},
                "test_cagr": chosen_metrics.get("cagr", np.nan),
                "test_sharpe": chosen_metrics.get("sharpe", np.nan),
                "test_max_drawdown": chosen_metrics.get("max_drawdown", np.nan),
                "test_monthly_positive_rate": _monthly_positive_rate(chosen_test),
                "baseline_test_cagr": baseline_metrics.get("cagr", np.nan),
                "baseline_test_sharpe": baseline_metrics.get("sharpe", np.nan),
                "baseline_test_max_drawdown": baseline_metrics.get("max_drawdown", np.nan),
                "baseline_test_monthly_positive_rate": _monthly_positive_rate(baseline_test),
            }
        )

    validation_df = pd.DataFrame(validation_rows)
    summary_df = pd.DataFrame(summary_rows)
    validation_df.to_csv(output_dir / "candidate_validation_scores.csv", index=False)
    summary_df.to_csv(output_dir / "nested_walk_forward_summary.csv", index=False)

    selected_returns = pd.concat(selected_test_returns, ignore_index=True) if selected_test_returns else pd.DataFrame()
    baseline_returns = pd.concat(baseline_test_returns, ignore_index=True) if baseline_test_returns else pd.DataFrame()
    aggregate_rows = []
    for name, returns in (("nested_selected", selected_returns), ("front_switch_baseline", baseline_returns)):
        if returns.empty:
            continue
        returns = returns.sort_values("trade_date")
        backtest = pd.DataFrame(
            {
                "trade_date": returns["trade_date"],
                "return": returns["return"],
                "nav": (1.0 + returns["return"]).cumprod(),
                "pnl": returns["return"],
            }
        )
        metrics = summarize_backtest(backtest)
        aggregate_rows.append(
            {
                "strategy": name,
                **metrics,
                "monthly_positive_rate": _monthly_positive_rate(backtest),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(output_dir / "nested_walk_forward_aggregate.csv", index=False)
    print(summary_df.to_string(index=False))
    print("\nAggregate nested test comparison:")
    print(aggregate.to_string(index=False))
    print(f"\nSaved nested walk-forward diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
