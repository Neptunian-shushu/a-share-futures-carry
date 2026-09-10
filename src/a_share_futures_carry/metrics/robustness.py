"""Robustness diagnostics for dependent financial return series."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from a_share_futures_carry.metrics.performance import performance_summary


def block_bootstrap_summary(
    returns: pd.Series,
    *,
    block_size: int = 20,
    n_bootstrap: int = 1000,
    seed: int = 42,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Estimate uncertainty by resampling contiguous return blocks.

    Blocks preserve some short-term dependence and are sampled with a fixed
    seed so reports can be regenerated exactly.  The returned quantiles are
    diagnostics, not a substitute for a genuine out-of-sample test.
    """
    raw = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if raw.empty:
        return {}
    if block_size < 1 or n_bootstrap < 1 or periods_per_year < 1:
        raise ValueError("block_size, n_bootstrap and periods_per_year must be positive")
    values = raw.to_numpy(dtype=float)
    n_obs = len(values)
    max_start = max(n_obs - block_size, 0)
    rng = np.random.default_rng(seed)
    sharpe: list[float] = []
    cagr: list[float] = []
    max_drawdown: list[float] = []
    for _ in range(n_bootstrap):
        pieces = []
        for _ in range(math.ceil(n_obs / block_size)):
            start = int(rng.integers(0, max_start + 1))
            pieces.append(values[start : start + block_size])
        sample = np.concatenate(pieces)[:n_obs]
        summary = performance_summary(pd.Series(sample), periods_per_year)
        sharpe.append(summary.get("sharpe", np.nan))
        cagr.append(summary.get("cagr", np.nan))
        max_drawdown.append(summary.get("max_drawdown", np.nan))

    def quantile(values_: list[float], q: float) -> float:
        valid = np.asarray(values_, dtype=float)
        valid = valid[np.isfinite(valid)]
        return float(np.quantile(valid, q)) if valid.size else np.nan

    sharpe_array = np.asarray(sharpe, dtype=float)
    valid_sharpe = sharpe_array[np.isfinite(sharpe_array)]
    return {
        "bootstrap_n": float(n_bootstrap),
        "bootstrap_seed": float(seed),
        "bootstrap_block_size": float(block_size),
        "bootstrap_sharpe_p05": quantile(sharpe, 0.05),
        "bootstrap_sharpe_p50": quantile(sharpe, 0.50),
        "bootstrap_sharpe_p95": quantile(sharpe, 0.95),
        "bootstrap_sharpe_positive_prob": float((valid_sharpe > 0).mean()) if valid_sharpe.size else np.nan,
        "bootstrap_cagr_p05": quantile(cagr, 0.05),
        "bootstrap_cagr_p50": quantile(cagr, 0.50),
        "bootstrap_cagr_p95": quantile(cagr, 0.95),
        "bootstrap_max_drawdown_p05": quantile(max_drawdown, 0.05),
        "bootstrap_max_drawdown_p50": quantile(max_drawdown, 0.50),
        "bootstrap_max_drawdown_p95": quantile(max_drawdown, 0.95),
    }
