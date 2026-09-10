"""CSV and chart outputs for strategy research runs."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from a_share_futures_carry.metrics.performance import summarize_backtest


def spot_benchmark_returns(panel: pd.DataFrame, family: str) -> pd.Series:
    """Return close-to-close spot-index returns for one futures family."""
    family_data = panel[panel["family"].astype(str).str.upper() == family.upper()]
    if family_data.empty:
        return pd.Series(dtype=float, name=f"{family}_spot_return")
    daily = (
        family_data.sort_values("trade_date")
        .drop_duplicates("trade_date")
        .set_index("trade_date")["spot_close"]
        .astype(float)
    )
    return daily.pct_change().fillna(0.0).rename(f"{family}_spot_return")


def price_benchmark_returns(
    prices: pd.DataFrame,
    *,
    date_column: str = "trade_date",
    price_column: str = "close",
    distribution_column: str | None = None,
) -> pd.Series:
    """Convert a dated ETF/index series into price or total-return returns.

    When ``distribution_column`` is supplied, its cash distribution per share
    is added to the current price before dividing by the prior price. This
    supports a true reinvested-distribution benchmark when the input contains
    ex-date distributions.
    """
    if date_column not in prices or price_column not in prices:
        raise ValueError(f"Benchmark requires {date_column} and {price_column}")
    columns = [date_column, price_column]
    if distribution_column is not None:
        if distribution_column not in prices:
            raise ValueError(f"Benchmark requires {distribution_column}")
        columns.append(distribution_column)
    series = prices[columns].copy()
    series[date_column] = pd.to_datetime(series[date_column])
    series[price_column] = pd.to_numeric(series[price_column], errors="coerce")
    if distribution_column is not None:
        series[distribution_column] = pd.to_numeric(series[distribution_column], errors="coerce").fillna(0.0)
        if (series[distribution_column] < 0).any():
            raise ValueError("Benchmark distributions must be non-negative")
    if series[date_column].duplicated().any():
        raise ValueError("Benchmark contains duplicate dates")
    if (series[price_column] <= 0).any() or series[price_column].isna().any():
        raise ValueError("Benchmark prices must be strictly positive")
    series = series.sort_values(date_column).set_index(date_column)
    if distribution_column is None:
        returns = series[price_column].pct_change()
    else:
        returns = (series[price_column] + series[distribution_column]) / series[price_column].shift(1) - 1.0
    return returns.fillna(0.0).rename("benchmark_return")


def returns_benchmark_backtest(returns: pd.Series, initial_nav: float) -> pd.DataFrame:
    """Build a benchmark-shaped frame from a dated return series."""
    returns = pd.Series(returns, dtype=float).sort_index()
    if returns.empty:
        return pd.DataFrame()
    nav = initial_nav * (1.0 + returns).cumprod()
    return pd.DataFrame(
        {
            "trade_date": returns.index,
            "return": returns.to_numpy(),
            "pnl": nav.diff().fillna(nav - initial_nav).to_numpy(),
            "nav": nav.to_numpy(),
        }
    )


def spot_benchmark_backtest(panel: pd.DataFrame, family: str, initial_nav: float) -> pd.DataFrame:
    """Build a benchmark-shaped frame so spot performance appears in reports."""
    return returns_benchmark_backtest(spot_benchmark_returns(panel, family), initial_nav)


def price_benchmark_backtest(
    prices: pd.DataFrame,
    initial_nav: float,
    *,
    date_column: str = "trade_date",
    price_column: str = "close",
    distribution_column: str | None = None,
) -> pd.DataFrame:
    """Build a benchmark-shaped frame from an ETF or index price series."""
    return returns_benchmark_backtest(
        price_benchmark_returns(
            prices,
            date_column=date_column,
            price_column=price_column,
            distribution_column=distribution_column,
        ),
        initial_nav,
    )


def generate_research_report(
    backtests: dict[str, pd.DataFrame],
    output_dir: str | Path,
    *,
    benchmarks: dict[str, pd.Series] | None = None,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Write summary/equity CSVs and a compact set of research charts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    benchmarks = benchmarks or {}
    summaries: list[dict[str, object]] = []
    curves: list[pd.DataFrame] = []

    for name, backtest in backtests.items():
        if backtest.empty:
            summaries.append({"strategy": name, "n_obs": 0})
            continue
        summary = summarize_backtest(
            backtest,
            benchmark_returns=benchmarks.get(name),
            periods_per_year=periods_per_year,
        )
        summary["strategy"] = name
        summaries.append(summary)
        curve = backtest[["trade_date", "nav", "return"]].copy()
        curve.insert(0, "strategy", name)
        curve["wealth"] = curve["nav"] / curve["nav"].iloc[0]
        curve["drawdown"] = curve["wealth"] / curve["wealth"].cummax() - 1.0
        curves.append(curve)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output / "strategy_summary.csv", index=False)
    if curves:
        combined = pd.concat(curves, ignore_index=True)
        combined.to_csv(output / "equity_curves.csv", index=False)
        monthly = (
            combined.assign(month=pd.to_datetime(combined["trade_date"]).dt.to_period("M"))
            .groupby(["strategy", "month"], as_index=False)["return"]
            .apply(lambda x: (1.0 + x).prod() - 1.0)
            .rename(columns={"return": "monthly_return"})
        )
        monthly.to_csv(output / "monthly_returns.csv", index=False)
        roll_frames = []
        for name, backtest in backtests.items():
            if not backtest.empty and "roll_event" in backtest:
                events = backtest[backtest["roll_event"]].copy()
                if not events.empty:
                    events.insert(0, "strategy", name)
                    roll_frames.append(events)
        if roll_frames:
            pd.concat(roll_frames, ignore_index=True).to_csv(output / "roll_events.csv", index=False)

    try:
        mpl_config = output / ".matplotlib"
        mpl_config.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
        import matplotlib.pyplot as plt
    except ImportError:
        return summary_df

    if not curves:
        return summary_df
    combined = pd.concat(curves, ignore_index=True)
    for column, filename, ylabel in (
        ("wealth", "nav_curves.png", "Normalized NAV"),
        ("drawdown", "drawdowns.png", "Drawdown"),
    ):
        fig, ax = plt.subplots(figsize=(11, 6))
        for name, group in combined.groupby("strategy"):
            ax.plot(group["trade_date"], group[column], label=name)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    summary_plot = summary_df.set_index("strategy")
    summary_plot["cagr"].sort_values().plot.barh(ax=ax)
    ax.set_xlabel("CAGR")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "cagr_comparison.png", dpi=150)
    plt.close(fig)

    pnl_columns = [
        "total_spot_beta_pnl",
        "total_basis_pnl",
        "total_collateral_pnl",
        "total_trading_cost",
    ]
    if all(column in summary_df for column in pnl_columns):
        fig, ax = plt.subplots(figsize=(10, 5))
        summary_plot[pnl_columns].plot.bar(ax=ax)
        ax.set_ylabel("P&L")
        ax.grid(axis="y", alpha=0.25)
        fig.autofmt_xdate(rotation=30)
        fig.tight_layout()
        fig.savefig(output / "pnl_decomposition.png", dpi=150)
        plt.close(fig)
    return summary_df
