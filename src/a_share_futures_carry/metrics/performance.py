"""Performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def performance_summary(returns: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
    r = returns.dropna()
    if r.empty:
        return {}

    total = (1 + r).prod() - 1
    years = len(r) / periods_per_year
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else np.nan
    vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year) if r.std(ddof=1) > 0 else np.nan
    wealth = (1 + r).cumprod()
    drawdown = wealth / wealth.cummax() - 1

    return {
        "total_return": float(total),
        "cagr": float(cagr),
        "annualized_vol": float(vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
    }
