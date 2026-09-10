"""Performance, risk and benchmark comparison metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _year_fraction(index: pd.Index, periods_per_year: int) -> float:
    if len(index) >= 2:
        dates = pd.to_datetime(index)
        elapsed_days = (dates[-1] - dates[0]).days
        if elapsed_days > 0:
            return elapsed_days / 365.25
    return len(index) / periods_per_year


def performance_summary(
    returns: pd.Series,
    periods_per_year: int = 252,
    *,
    dates: pd.Series | pd.Index | None = None,
    benchmark_returns: pd.Series | None = None,
) -> dict[str, float]:
    """Calculate return, risk, drawdown and optional benchmark statistics."""
    raw = pd.Series(returns, dtype=float)
    valid = raw.notna() & np.isfinite(raw)
    r = raw[valid]
    if r.empty:
        return {}
    if dates is None:
        index = r.index
    else:
        date_index = pd.Index(pd.to_datetime(dates))
        if len(date_index) != len(raw):
            raise ValueError("dates must have the same length as returns")
        index = date_index[valid.to_numpy()]
    years = _year_fraction(index, periods_per_year)

    total = (1 + r).prod() - 1
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else np.nan
    daily_std = r.std(ddof=1)
    vol = daily_std * np.sqrt(periods_per_year)
    sharpe = r.mean() / daily_std * np.sqrt(periods_per_year) if daily_std > 0 else np.nan
    downside = r.clip(upper=0).pow(2).mean() ** 0.5 * np.sqrt(periods_per_year)
    sortino = r.mean() / (downside / np.sqrt(periods_per_year)) * np.sqrt(periods_per_year) if downside > 0 else np.nan
    wealth = (1 + r).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    max_drawdown = float(drawdown.min())

    summary = {
        "total_return": float(total),
        "cagr": float(cagr),
        "annualized_vol": float(vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(cagr / abs(max_drawdown)) if max_drawdown < 0 else np.nan,
        "max_drawdown": max_drawdown,
        "positive_day_ratio": float((r > 0).mean()),
        "n_obs": float(len(r)),
    }

    if benchmark_returns is not None:
        benchmark = pd.Series(benchmark_returns, dtype=float).reindex(r.index).dropna()
        aligned = pd.concat([r.rename("strategy"), benchmark.rename("benchmark")], axis=1).dropna()
        if len(aligned) >= 2 and aligned["benchmark"].var(ddof=1) > 0:
            covariance = aligned["strategy"].cov(aligned["benchmark"])
            beta = covariance / aligned["benchmark"].var(ddof=1)
            active = aligned["strategy"] - aligned["benchmark"]
            summary["beta"] = float(beta)
            summary["active_total_return"] = float((1 + active).prod() - 1)
            summary["tracking_error"] = float(active.std(ddof=1) * np.sqrt(periods_per_year))
        else:
            summary.update({"beta": np.nan, "active_total_return": np.nan, "tracking_error": np.nan})
    return summary


def drawdown_table(returns: pd.Series, dates: pd.Series | pd.Index | None = None) -> pd.DataFrame:
    """Return drawdown episodes ordered by worst peak-to-trough loss."""
    r = pd.Series(returns, dtype=float).fillna(0.0)
    idx = pd.Index(pd.to_datetime(dates)) if dates is not None else r.index
    if len(idx) != len(r):
        raise ValueError("dates must have the same length as returns")
    wealth = (1 + r).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1
    episodes: list[dict[str, object]] = []
    in_drawdown = False
    start = None
    trough = None
    for i, value in enumerate(dd):
        if value < 0 and not in_drawdown:
            in_drawdown = True
            start = idx[i]
            trough = i
        if in_drawdown and value < dd.iloc[trough]:
            trough = i
        recovered = in_drawdown and value >= 0
        last = i == len(dd) - 1
        if recovered or (last and in_drawdown):
            episodes.append({
                "start": start,
                "trough": idx[trough],
                "end": idx[i] if recovered else pd.NaT,
                "drawdown": float(dd.iloc[trough]),
                "duration_days": int((idx[i] - start).days) if recovered else int((idx[i] - start).days),
            })
            in_drawdown = False
    return pd.DataFrame(episodes).sort_values("drawdown").reset_index(drop=True) if episodes else pd.DataFrame()


def summarize_backtest(
    backtest: pd.DataFrame,
    *,
    benchmark_returns: pd.Series | None = None,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Summarize a backtest and include portfolio-level diagnostics."""
    if backtest.empty:
        return {}
    dated_returns = pd.Series(
        backtest["return"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(backtest["trade_date"])),
    )
    summary = performance_summary(
        dated_returns,
        periods_per_year,
        dates=dated_returns.index,
        benchmark_returns=benchmark_returns,
    )
    for column, name in (
        ("turnover_notional", "total_turnover_notional"),
        ("trading_cost", "total_trading_cost"),
        ("futures_pnl", "total_futures_pnl"),
        ("spot_beta_pnl", "total_spot_beta_pnl"),
        ("basis_pnl", "total_basis_pnl"),
        ("collateral_pnl", "total_collateral_pnl"),
    ):
        if column in backtest:
            summary[name] = float(backtest[column].sum())
    for column, name in (("roll_event", "roll_count"), ("margin_call", "margin_call_count")):
        if column in backtest:
            summary[name] = float(backtest[column].fillna(False).astype(bool).sum())
    if "exposure_to_nav" in backtest:
        summary["average_exposure"] = float(backtest["exposure_to_nav"].mean())
        summary["max_exposure"] = float(backtest["exposure_to_nav"].max())
    summary["initial_nav"] = float(backtest["nav"].iloc[0] - backtest["pnl"].iloc[0]) if "pnl" in backtest else np.nan
    summary["final_nav"] = float(backtest["nav"].iloc[-1])
    return summary
