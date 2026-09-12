"""Leakage-aware risk overlays for selected futures strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_window(lookback: int, min_periods: int) -> None:
    if lookback < 2 or min_periods < 2 or min_periods > lookback:
        raise ValueError("Require 2 <= min_periods <= lookback")


def add_beta_target_allocation(
    selected: pd.DataFrame,
    *,
    target_beta: float = 0.8,
    lookback: int = 60,
    min_periods: int = 20,
    asset_price_column: str = "futures_close",
    benchmark_price_column: str = "spot_close",
    base_weight_column: str = "target_weight",
    max_weight: float = 1.0,
    fallback_beta: float = 1.0,
) -> pd.DataFrame:
    """Scale an existing position toward a rolling beta target.

    The rolling beta is estimated from the selected contract's close return
    against the cash-index return.  The estimate uses only data through the
    signal date; the engine's execution lag controls when that target becomes
    tradable.  Missing or zero-variance windows use ``fallback_beta`` so the
    overlay does not create a warm-up hole in an otherwise valid strategy.
    """
    if target_beta < 0:
        raise ValueError("target_beta must be non-negative")
    if fallback_beta <= 0:
        raise ValueError("fallback_beta must be positive")
    if max_weight < 0:
        raise ValueError("max_weight must be non-negative")
    _validate_window(lookback, min_periods)
    required = {asset_price_column, benchmark_price_column}
    missing = required.difference(selected.columns)
    if missing:
        raise ValueError(f"Missing beta columns: {sorted(missing)}")

    out = selected.sort_values("trade_date").copy().reset_index(drop=True)
    asset = pd.to_numeric(out[asset_price_column], errors="coerce")
    benchmark = pd.to_numeric(out[benchmark_price_column], errors="coerce")
    asset_returns = asset.pct_change()
    benchmark_returns = benchmark.pct_change()
    covariance = asset_returns.rolling(lookback, min_periods=min_periods).cov(benchmark_returns)
    variance = benchmark_returns.rolling(lookback, min_periods=min_periods).var(ddof=0)
    stable_variance = variance.where(variance >= 1e-12)
    realized_beta = covariance / stable_variance
    realized_beta = realized_beta.replace([np.inf, -np.inf], np.nan)
    beta_for_scaling = realized_beta.abs().fillna(float(fallback_beta)).clip(lower=1e-6)
    scaler = (float(target_beta) / beta_for_scaling).clip(lower=0.0, upper=max_weight)
    base_weight = (
        pd.to_numeric(out[base_weight_column], errors="coerce").fillna(0.0)
        if base_weight_column in out
        else pd.Series(1.0, index=out.index)
    )
    out["realized_beta"] = realized_beta
    out["beta_target_scale"] = scaler
    out["target_weight"] = (base_weight * scaler).clip(lower=0.0, upper=max_weight).fillna(0.0)
    return out


def add_market_regime_filter(
    selected: pd.DataFrame,
    *,
    momentum_lookback: int = 63,
    volatility_lookback: int = 20,
    volatility_quantile_lookback: int = 252,
    volatility_quantile: float = 0.8,
    downtrend_weight: float = 0.5,
    high_volatility_weight: float = 0.5,
    benchmark_price_column: str = "spot_close",
    base_weight_column: str = "target_weight",
    max_weight: float = 1.0,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Reduce exposure during a downtrend and/or unusually high volatility.

    A downtrend is a negative ``momentum_lookback`` return.  High volatility is
    the current realized volatility above its trailing quantile.  A missing
    warm-up statistic leaves exposure unchanged rather than forcing a flat
    portfolio.
    """
    _validate_window(momentum_lookback, 2)
    _validate_window(volatility_lookback, 2)
    _validate_window(volatility_quantile_lookback, 2)
    if not 0 < volatility_quantile < 1:
        raise ValueError("volatility_quantile must be in (0, 1)")
    if not 0 <= downtrend_weight <= 1 or not 0 <= high_volatility_weight <= 1:
        raise ValueError("regime weights must be in [0, 1]")
    if max_weight < 0 or periods_per_year <= 0:
        raise ValueError("max_weight must be non-negative and periods_per_year must be positive")
    if benchmark_price_column not in selected:
        raise ValueError(f"Missing regime price column: {benchmark_price_column}")

    out = selected.sort_values("trade_date").copy().reset_index(drop=True)
    benchmark = pd.to_numeric(out[benchmark_price_column], errors="coerce")
    returns = benchmark.pct_change()
    momentum = benchmark.pct_change(momentum_lookback)
    realized_vol = returns.rolling(
        volatility_lookback, min_periods=volatility_lookback
    ).std(ddof=0) * np.sqrt(periods_per_year)
    vol_threshold = realized_vol.rolling(
        volatility_quantile_lookback, min_periods=volatility_quantile_lookback
    ).quantile(volatility_quantile)

    downtrend = momentum.lt(0).fillna(False)
    high_volatility = (realized_vol > vol_threshold).fillna(False)
    multiplier = pd.Series(1.0, index=out.index)
    multiplier = multiplier.where(~downtrend, float(downtrend_weight))
    multiplier = multiplier.where(~high_volatility, multiplier * float(high_volatility_weight))
    regime = pd.Series("normal", index=out.index, dtype="object")
    regime = regime.mask(downtrend & ~high_volatility, "downtrend")
    regime = regime.mask(~downtrend & high_volatility, "high_volatility")
    regime = regime.mask(downtrend & high_volatility, "downtrend_high_volatility")

    base_weight = (
        pd.to_numeric(out[base_weight_column], errors="coerce").fillna(0.0)
        if base_weight_column in out
        else pd.Series(1.0, index=out.index)
    )
    out["market_momentum"] = momentum
    out["realized_market_vol"] = realized_vol
    out["high_volatility_threshold"] = vol_threshold
    out["market_regime"] = regime
    out["regime_weight"] = multiplier
    out["target_weight"] = (base_weight * multiplier).clip(lower=0.0, upper=max_weight).fillna(0.0)
    return out


def add_risk_overlay(
    selected: pd.DataFrame,
    *,
    target_beta: float = 0.8,
    beta_lookback: int = 60,
    beta_min_periods: int = 20,
    momentum_lookback: int = 63,
    volatility_lookback: int = 20,
    volatility_quantile_lookback: int = 252,
    volatility_quantile: float = 0.8,
    downtrend_weight: float = 0.5,
    high_volatility_weight: float = 0.5,
    max_weight: float = 1.0,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Apply beta targeting followed by a market-regime exposure filter."""
    beta_adjusted = add_beta_target_allocation(
        selected,
        target_beta=target_beta,
        lookback=beta_lookback,
        min_periods=beta_min_periods,
        max_weight=max_weight,
    )
    return add_market_regime_filter(
        beta_adjusted,
        momentum_lookback=momentum_lookback,
        volatility_lookback=volatility_lookback,
        volatility_quantile_lookback=volatility_quantile_lookback,
        volatility_quantile=volatility_quantile,
        downtrend_weight=downtrend_weight,
        high_volatility_weight=high_volatility_weight,
        max_weight=max_weight,
        periods_per_year=periods_per_year,
    )
