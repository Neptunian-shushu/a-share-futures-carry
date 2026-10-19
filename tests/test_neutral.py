import pandas as pd

from a_share_futures_carry.backtest.neutral import backtest_beta_neutral_pair


def _market() -> pd.DataFrame:
    dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"])
    rows = []
    for family, contract, multiplier, prices in [
        ("IC", "IC2602", 1.0, [100.0, 101.0, 102.0, 103.0]),
        ("IF", "IF2602", 1.0, [200.0, 202.0, 204.0, 206.0]),
    ]:
        for date, price in zip(dates, prices):
            rows.append(
                {
                    "trade_date": date,
                    "contract": contract,
                    "family": family,
                    "futures_close": price,
                    "settle": price,
                    "spot_close": price,
                    "multiplier": multiplier,
                    "vol": 1_000_000.0,
                }
            )
    return pd.DataFrame(rows)


def test_beta_neutral_pair_offsets_same_beta_move():
    market = _market()
    long = market[market["family"].eq("IC")].copy()
    hedge = market[market["family"].eq("IF")].copy()
    result = backtest_beta_neutral_pair(
        long,
        hedge,
        market,
        initial_nav=100_000.0,
        long_notional_to_nav=0.5,
        gross_notional_to_nav=1.0,
        collateral_yield_annual=0.0,
        transaction_cost_bps=0.0,
        signal_lag_sessions=0,
        beta_lookback=2,
        beta_min_periods=2,
        margin_rate=0.0,
        max_participation_rate=None,
        integer_contracts=False,
    )
    assert result["futures_pnl"].abs().sum() < 1e-8
    assert result["estimated_net_beta_notional"].abs().max() < 1e-8
    assert abs(result["hedge_ratio"].iloc[-1] - 1.0) < 1e-8


def test_pair_respects_gross_notional_cap():
    market = _market()
    result = backtest_beta_neutral_pair(
        market[market["family"].eq("IC")].copy(),
        market[market["family"].eq("IF")].copy(),
        market,
        initial_nav=100_000.0,
        long_notional_to_nav=1.0,
        gross_notional_to_nav=1.0,
        collateral_yield_annual=0.0,
        transaction_cost_bps=0.0,
        signal_lag_sessions=0,
        beta_lookback=2,
        beta_min_periods=2,
        margin_rate=0.0,
        max_participation_rate=None,
    )
    assert (result["gross_notional"] <= result["nav"] + 1e-9).all()
