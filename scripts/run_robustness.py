"""Run fixed-holdout, regime, bootstrap and implementation-stress diagnostics."""

from __future__ import annotations

import argparse
from copy import deepcopy
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
from a_share_futures_carry.metrics.robustness import block_bootstrap_summary
from a_share_futures_carry.signals.basis import add_carry_columns, add_cost_adjusted_carry
from a_share_futures_carry.strategy.allocation import (
    add_dynamic_carry_allocation,
    add_volatility_target_allocation,
)
from a_share_futures_carry.strategy.selection import (
    apply_roll_policy,
    select_family_max_carry,
    select_max_carry,
    select_nth_expiry,
)


def _prepare(data: pd.DataFrame, cfg: dict, use_fair_value: bool) -> pd.DataFrame:
    carry = cfg["carry"]
    strategy = cfg["strategy"]
    out = add_carry_columns(
        data,
        carry["day_count"],
        use_fair_value_adjustment=use_fair_value,
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


def _roll(selected: pd.DataFrame, data: pd.DataFrame, cfg: dict) -> pd.DataFrame:
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


def _strategies(data: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    strategy = cfg["strategy"]
    carry_column = strategy["carry_column"]
    score_column = strategy.get("selection_score_column", carry_column)
    specs: dict[str, pd.DataFrame] = {}
    for family in ("IF", "IH", "IC", "IM"):
        family_data = data[data["family"] == family]
        if family_data.empty:
            continue
        specs[f"{family}_front"] = select_nth_expiry(
            family_data, 1, (family,), carry_column=carry_column,
            min_volume=strategy["min_volume"], min_open_interest=strategy["min_open_interest"],
        )
        specs[f"{family}_max_carry"] = select_family_max_carry(
            family_data, family, strategy["min_dte"], strategy["max_dte"], score_column,
            strategy["min_volume"], strategy["min_open_interest"],
        )
    dynamic = select_max_carry(
        data, tuple(strategy["eligible_families"]), strategy["min_dte"], strategy["max_dte"],
        score_column, strategy["min_volume"], strategy["min_open_interest"],
    )
    specs["dynamic_IC_IM_max_carry"] = dynamic
    rolled = {name: _roll(candidate, data, cfg) for name, candidate in specs.items()}
    allocation = cfg.get("allocation", {})
    dynamic_selected = rolled["dynamic_IC_IM_max_carry"]
    if not dynamic_selected.empty and allocation.get("enabled", False):
        dynamic_allocated = add_dynamic_carry_allocation(
            dynamic_selected, carry_column=carry_column,
            lookback=allocation["lookback"], min_periods=allocation["min_periods"],
            method=allocation["method"], entry_threshold=allocation["entry_threshold"],
            zscore_scale=allocation["zscore_scale"], max_weight=allocation["max_weight"],
        )
        rolled["dynamic_IC_IM_carry_allocation"] = dynamic_allocated
        if allocation.get("vol_target_enabled", False):
            rolled["dynamic_IC_IM_carry_vol_target"] = add_volatility_target_allocation(
                dynamic_allocated,
                target_vol_annual=allocation.get("target_vol_annual", 0.10),
                lookback=allocation.get("vol_lookback", allocation["lookback"]),
                min_periods=allocation.get("vol_min_periods", allocation["min_periods"]),
                price_column=allocation.get("vol_price_column", "spot_close"),
                max_weight=allocation.get("max_weight", 1.0),
                periods_per_year=cfg["carry"].get("trading_days_per_year", 252),
            )
    return rolled


def _run(selected: pd.DataFrame, data: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    p = cfg["portfolio"]
    return backtest_selected_contracts(
        selected,
        initial_nav=p["initial_nav"], max_notional_to_nav=cfg["strategy"]["max_notional_to_nav"],
        collateral_yield_annual=p["collateral_yield_annual"],
        transaction_cost_bps=p["commission_bps"] + p["slippage_bps"], market_data=data,
        signal_lag_sessions=p["signal_lag_sessions"], margin_rate=p["margin_rate"],
        margin_buffer=p["margin_buffer"], integer_contracts=p["integer_contracts"],
        execution_price_col=p["execution_price_col"], mark_price_col=p["mark_price_col"],
    )


def _fixed_test(backtest: pd.DataFrame, test_start: pd.Timestamp) -> pd.DataFrame:
    if backtest.empty:
        return backtest
    return backtest[pd.to_datetime(backtest["trade_date"]) >= test_start].copy().reset_index(drop=True)


def _regime_labels(data: pd.DataFrame, lookback: int = 63) -> pd.DataFrame:
    spot = data[data["family"] == "IC"][["trade_date", "spot_close"]].drop_duplicates("trade_date")
    spot = spot.sort_values("trade_date").copy()
    spot["spot_momentum"] = spot["spot_close"].pct_change(lookback)
    spot["regime"] = np.where(spot["spot_momentum"] > 0, "up", "down")
    return spot[["trade_date", "regime"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/strategy.yaml")
    parser.add_argument("--output-dir", default="outputs/robustness")
    parser.add_argument("--test-sessions", type=int, default=252)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    raw = load_contract_panel_csv(args.data)
    all_dates = pd.DatetimeIndex(sorted(pd.to_datetime(raw["trade_date"]).unique()))
    if len(all_dates) <= args.test_sessions:
        raise ValueError("test_sessions must be smaller than the available history")
    test_start = all_dates[-args.test_sessions]
    test_end = all_dates[-1]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    regime_rows: list[dict[str, object]] = []
    prepared_by_mode: dict[str, tuple[pd.DataFrame, dict[str, pd.DataFrame]]] = {}
    for mode, fair in (("observed", False), ("fair", True)):
        data = _prepare(raw, cfg, fair)
        strategies = _strategies(data, cfg)
        prepared_by_mode[mode] = (data, strategies)
        regimes = _regime_labels(data)
        for name, selected in strategies.items():
            backtest = _run(selected, data, cfg)
            final = _fixed_test(backtest, test_start)
            metrics = summarize_backtest(final) if not final.empty else {}
            bootstrap = block_bootstrap_summary(
                final["return"], n_bootstrap=args.bootstrap,
                periods_per_year=cfg["carry"].get("trading_days_per_year", 252),
            ) if not final.empty else {}
            row = {
                "carry_mode": mode, "strategy": name,
                "test_start": test_start, "test_end": test_end,
                "test_sessions": args.test_sessions,
                "full_start": all_dates[0], "full_end": all_dates[-1],
            }
            row.update({f"test_{key}": value for key, value in metrics.items()})
            row.update(bootstrap)
            summary_rows.append(row)
            if not final.empty:
                regime_frame = final[["trade_date", "return"]].merge(regimes, on="trade_date", how="left")
                for regime, group in regime_frame.dropna(subset=["regime"]).groupby("regime"):
                    regime_metrics = summarize_backtest(
                        pd.DataFrame({
                            "trade_date": group["trade_date"], "return": group["return"],
                            "nav": (1 + group["return"]).cumprod(), "pnl": group["return"],
                        })
                    )
                    regime_rows.append({
                        "carry_mode": mode, "strategy": name, "regime": regime,
                        "n_obs": int(len(group)), "cagr": regime_metrics.get("cagr", np.nan),
                        "sharpe": regime_metrics.get("sharpe", np.nan),
                        "total_return": regime_metrics.get("total_return", np.nan),
                        "max_drawdown": regime_metrics.get("max_drawdown", np.nan),
                    })

    stress_rows: list[dict[str, object]] = []
    base_data, base_strategies = prepared_by_mode["observed"]
    for strategy_name in (
        "IF_front", "IH_front", "IC_front", "IM_front",
        "dynamic_IC_IM_max_carry", "dynamic_IC_IM_carry_vol_target",
    ):
        selected = base_strategies.get(strategy_name)
        if selected is None:
            continue
        for cost_multiplier in (0.5, 1.0, 2.0):
            for margin_rate in (0.08, 0.12, 0.20):
                variant = deepcopy(cfg)
                variant["portfolio"]["commission_bps"] *= cost_multiplier
                variant["portfolio"]["slippage_bps"] *= cost_multiplier
                variant["portfolio"]["margin_rate"] = margin_rate
                backtest = _fixed_test(_run(selected, base_data, variant), test_start)
                metrics = summarize_backtest(backtest) if not backtest.empty else {}
                stress_rows.append({
                    "strategy": strategy_name, "cost_multiplier": cost_multiplier,
                    "margin_rate": margin_rate, "test_start": test_start,
                    "cagr": metrics.get("cagr", np.nan), "sharpe": metrics.get("sharpe", np.nan),
                    "max_drawdown": metrics.get("max_drawdown", np.nan),
                    "total_trading_cost": metrics.get("total_trading_cost", np.nan),
                    "margin_call_count": metrics.get("margin_call_count", np.nan),
                })

    summary = pd.DataFrame(summary_rows)
    regimes = pd.DataFrame(regime_rows)
    stress = pd.DataFrame(stress_rows)
    summary.to_csv(output_dir / "fixed_holdout_summary.csv", index=False)
    regimes.to_csv(output_dir / "regime_summary.csv", index=False)
    stress.to_csv(output_dir / "implementation_stress.csv", index=False)
    print(summary[["carry_mode", "strategy", "test_cagr", "test_sharpe", "test_max_drawdown", "bootstrap_sharpe_p05", "bootstrap_sharpe_positive_prob"]].to_string(index=False))
    print(f"\nSaved robustness diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
