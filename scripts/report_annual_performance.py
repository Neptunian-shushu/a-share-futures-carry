"""Build annual performance tables from a generated equity-curve report."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


LABELS = {
    "IF_front": "IF近月基线",
    "IH_front": "IH近月基线",
    "dynamic_IC_IM_front_switch": "IC/IM近月动态切换",
    "dynamic_IC_IM_beta_target": "IC/IM Beta目标仓位",
    "IC_front": "IC近月基线",
    "IM_front": "IM近月基线",
    "dynamic_IC_IM_max_carry": "IC/IM动态最大Carry",
    "dynamic_IC_IM_carry_vol_target": "动态Carry+波动率目标",
    "510500_total_return": "510500 ETF总收益",
}

# The long backtest keeps a unified calendar and therefore includes inactive
# rows before a family becomes available.  Exclude those rows from annual
# reporting instead of treating collateral-only wealth as strategy performance.
MIN_START_DATES = {
    "IF_front": pd.Timestamp("2015-01-05"),
    "IH_front": pd.Timestamp("2015-04-16"),
    "IC_front": pd.Timestamp("2015-04-16"),
    "IM_front": pd.Timestamp("2022-07-22"),
    "dynamic_IC_IM_front_switch": pd.Timestamp("2015-04-16"),
    "dynamic_IC_IM_beta_target": pd.Timestamp("2015-04-16"),
    "dynamic_IC_IM_max_carry": pd.Timestamp("2015-04-16"),
    "dynamic_IC_IM_carry_vol_target": pd.Timestamp("2015-04-16"),
    "510500_total_return": pd.Timestamp("2022-07-22"),
}


def _monthly_positive_rate(frame: pd.DataFrame) -> tuple[float, int]:
    monthly = (
        frame.assign(month=frame["trade_date"].dt.to_period("M"))
        .groupby("month")["return"]
        .apply(lambda values: (1.0 + values).prod() - 1.0)
    )
    if monthly.empty:
        return np.nan, 0
    return float((monthly > 0).mean()), int(len(monthly))


def build_annual_performance(
    curves: pd.DataFrame,
    *,
    strategies: list[str] | None = None,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Return one row per strategy and calendar year.

    Annual return is the compounded daily return within that calendar year.
    The first and last years can be partial years; ``period_start`` and
    ``period_end`` make that visible in the output.
    """
    required = {"strategy", "trade_date", "return"}
    missing = required.difference(curves.columns)
    if missing:
        raise ValueError(f"Curves file missing columns: {sorted(missing)}")

    frame = curves[["strategy", "trade_date", "return"]].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame["return"] = pd.to_numeric(frame["return"], errors="coerce")
    frame = frame.dropna(subset=["strategy", "trade_date", "return"])
    if strategies is not None:
        frame = frame[frame["strategy"].isin(strategies)]
    if not frame.empty:
        frame = frame[
            frame.apply(
                lambda row: row["trade_date"] >= MIN_START_DATES.get(
                    row["strategy"], pd.Timestamp.min
                ),
                axis=1,
            )
        ]
    if frame.empty:
        raise ValueError("No requested strategy curves found")

    rows: list[dict[str, object]] = []
    for (strategy, year), group in frame.sort_values("trade_date").groupby(
        ["strategy", frame["trade_date"].dt.year], sort=True
    ):
        returns = group["return"].astype(float)
        total_return = float((1.0 + returns).prod() - 1.0)
        daily_std = float(returns.std(ddof=1)) if len(returns) > 1 else np.nan
        annualized_vol = daily_std * np.sqrt(periods_per_year) if np.isfinite(daily_std) else np.nan
        sharpe = (
            float(returns.mean() / daily_std * np.sqrt(periods_per_year))
            if np.isfinite(daily_std) and daily_std > 0
            else np.nan
        )
        wealth = (1.0 + returns).cumprod()
        max_drawdown = float((wealth / wealth.cummax() - 1.0).min())
        positive_month_rate, month_count = _monthly_positive_rate(group)
        first_date = group["trade_date"].min()
        last_date = group["trade_date"].max()
        rows.append(
            {
                "year": int(year),
                "strategy": strategy,
                "strategy_label": LABELS.get(strategy, strategy),
                "period_start": first_date.date().isoformat(),
                "period_end": last_date.date().isoformat(),
                "n_obs": int(len(group)),
                "total_return": total_return,
                "annualized_vol": annualized_vol,
                "sharpe": sharpe,
                "max_drawdown": max_drawdown,
                "positive_day_ratio": float((returns > 0).mean()),
                "positive_month_ratio": positive_month_rate,
                "month_count": month_count,
                "partial_year": not (
                    first_date.month == 1
                    and last_date.month == 12
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["year", "strategy"]).reset_index(drop=True)


def annual_return_table(details: pd.DataFrame) -> pd.DataFrame:
    """Return a readable year-by-strategy annual-return table."""
    table = details.pivot(index="year", columns="strategy_label", values="total_return")
    return table.sort_index().reset_index()


def _markdown_table(details: pd.DataFrame) -> str:
    returns = annual_return_table(details).copy()
    for column in returns.columns:
        if column != "year":
            returns[column] = returns[column].map(lambda value: "" if pd.isna(value) else f"{value:.2%}")
    lines = [
        "# 分年度策略表现",
        "",
        "年度收益为该自然年内日收益复合结果；首尾年度可能因品种上市日期或样本截止日期而不完整，具体起止日期见下方风险表。",
        "",
        returns.to_markdown(index=False),
    ]
    lines.extend(["", "## 年度风险指标", ""])
    risk = details[[
        "year", "strategy_label", "period_start", "period_end", "annualized_vol",
        "sharpe", "max_drawdown", "positive_month_ratio", "month_count",
    ]].copy()
    for column in ("annualized_vol", "max_drawdown", "positive_month_ratio"):
        risk[column] = risk[column].map(lambda value: "" if pd.isna(value) else f"{value:.2%}")
    risk["sharpe"] = risk["sharpe"].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    risk = risk.rename(columns={
        "year": "年度", "strategy_label": "策略", "period_start": "起始日", "period_end": "结束日",
        "annualized_vol": "年化波动", "sharpe": "Sharpe", "max_drawdown": "最大回撤",
        "positive_month_ratio": "正收益月比例", "month_count": "月份数",
    })
    lines.append(risk.to_markdown(index=False))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--curves",
        default="outputs/full_research_report/equity_curves.csv",
        help="Generated equity_curves.csv from run_real_backtest.py",
    )
    parser.add_argument(
        "--output",
        default="outputs/full_research_report/annual_performance.csv",
    )
    parser.add_argument(
        "--markdown-output",
        default="outputs/full_research_report/annual_performance.md",
    )
    parser.add_argument(
        "--strategies",
        nargs="*",
        default=list(LABELS),
        help="Strategy keys to include; defaults to the core comparison set",
    )
    args = parser.parse_args()

    details = build_annual_performance(pd.read_csv(args.curves), strategies=args.strategies)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(output, index=False)
    markdown_output = Path(args.markdown_output)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(_markdown_table(details), encoding="utf-8")
    print(details.to_string(index=False))
    print(f"\nSaved annual performance CSV to {output}")
    print(f"Saved annual performance Markdown to {markdown_output}")


if __name__ == "__main__":
    main()
