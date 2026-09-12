import numpy as np
import pandas as pd

from a_share_futures_carry.strategy.risk import (
    add_beta_target_allocation,
    add_market_regime_filter,
    add_risk_overlay,
)


def _selected(periods: int = 80) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-02", periods=periods)
    benchmark_returns = np.where(np.arange(periods) % 2 == 0, 0.01, -0.006)
    futures_returns = np.where(np.arange(periods) % 2 == 0, 0.02, -0.012)
    benchmark = 100.0 * np.cumprod(1.0 + benchmark_returns)
    futures = 200.0 * np.cumprod(1.0 + futures_returns)
    return pd.DataFrame(
        {
            "trade_date": dates,
            "futures_close": futures,
            "spot_close": benchmark,
            "target_weight": 1.0,
        }
    )


def test_beta_target_is_bounded_and_uses_fallback_during_warmup():
    result = add_beta_target_allocation(_selected(), target_beta=0.8, lookback=20, min_periods=5)
    assert result["target_weight"].between(0, 1).all()
    assert result.loc[0, "target_weight"] == 0.8
    assert result["realized_beta"].iloc[-1] > 1.5
    assert result["target_weight"].iloc[-1] < 0.6


def test_regime_filter_reduces_downtrend_and_high_volatility_exposure():
    selected = _selected(100)
    selected["spot_close"] = np.linspace(120.0, 80.0, len(selected))
    selected.loc[70:, "spot_close"] = selected.loc[70:, "spot_close"] + np.where(
        np.arange(len(selected) - 70) % 2 == 0, 4.0, -4.0
    )
    result = add_market_regime_filter(
        selected,
        momentum_lookback=10,
        volatility_lookback=5,
        volatility_quantile_lookback=20,
        volatility_quantile=0.8,
        downtrend_weight=0.5,
        high_volatility_weight=0.5,
    )
    assert (result.loc[20:, "market_regime"] != "normal").any()
    assert (result.loc[20:, "target_weight"] <= 1.0).all()
    assert result["target_weight"].min() <= 0.5


def test_risk_overlay_composes_beta_and_regime_columns():
    result = add_risk_overlay(
        _selected(100),
        target_beta=0.8,
        beta_lookback=20,
        beta_min_periods=5,
        momentum_lookback=10,
        volatility_lookback=5,
        volatility_quantile_lookback=20,
    )
    assert {"realized_beta", "market_regime", "regime_weight", "target_weight"}.issubset(result.columns)
    assert result["target_weight"].between(0, 1).all()
