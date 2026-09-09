"""Dynamic carry-based exposure sizing."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_dynamic_carry_allocation(
    selected: pd.DataFrame,
    *,
    carry_column: str = "signal_carry",
    lookback: int = 60,
    min_periods: int = 20,
    method: str = "percentile",
    entry_threshold: float = 0.5,
    zscore_scale: float = 1.0,
    max_weight: float = 1.0,
) -> pd.DataFrame:
    """Add historical carry percentile/z-score and a target exposure weight.

    The rolling window includes only information available on the signal date;
    the backtest execution lag determines when that signal becomes tradable.
    Below-threshold observations receive zero exposure, which creates an
    explicit no-trade regime instead of forcing a position every day.
    """
    if carry_column not in selected:
        raise ValueError(f"Missing carry column: {carry_column}")
    if lookback < 2 or min_periods < 2 or min_periods > lookback:
        raise ValueError("Require 2 <= min_periods <= lookback")
    if method not in {"percentile", "zscore"}:
        raise ValueError("method must be 'percentile' or 'zscore'")
    if not 0 <= entry_threshold < 1:
        raise ValueError("entry_threshold must be in [0, 1)")
    if zscore_scale <= 0 or max_weight < 0:
        raise ValueError("zscore_scale and max_weight must be positive")

    out = selected.sort_values("trade_date").copy().reset_index(drop=True)
    values = pd.to_numeric(out[carry_column], errors="coerce")

    def last_percentile(window: np.ndarray) -> float:
        if not np.isfinite(window[-1]):
            return np.nan
        finite = window[np.isfinite(window)]
        if finite.size == 0:
            return np.nan
        return float((finite <= window[-1]).mean())

    out["carry_percentile"] = values.rolling(
        lookback, min_periods=min_periods
    ).apply(last_percentile, raw=True)
    rolling_mean = values.rolling(lookback, min_periods=min_periods).mean()
    rolling_std = values.rolling(lookback, min_periods=min_periods).std(ddof=0)
    out["carry_zscore"] = (values - rolling_mean) / rolling_std.replace(0, np.nan)

    if method == "percentile":
        score = out["carry_percentile"]
        weight = (score - entry_threshold) / (1.0 - entry_threshold)
    else:
        score = out["carry_zscore"]
        weight = (score - entry_threshold) / zscore_scale
    out["target_weight"] = weight.clip(lower=0, upper=max_weight).fillna(0.0)
    return out
