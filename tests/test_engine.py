import pandas as pd

from a_share_futures_carry.backtest.engine import backtest_selected_contracts


def _panel(prices_by_contract: dict[str, list[float]]) -> pd.DataFrame:
    dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    rows = []
    for contract, prices in prices_by_contract.items():
        for date, price in zip(dates, prices):
            rows.append(
                {
                    "trade_date": date,
                    "contract": contract,
                    "family": "IC",
                    "futures_close": price,
                    "settle": price,
                    "spot_close": 100.0,
                    "expiry_date": pd.Timestamp("2026-02-20"),
                    "multiplier": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_continuous_contract_generates_daily_pnl_and_integer_size():
    market = _panel({"IC2602": [100.0, 110.0, 105.0]})
    selected = market.copy()
    selected["carry_ann"] = 0.05
    selected["signal_carry"] = 0.05

    result = backtest_selected_contracts(
        selected,
        initial_nav=100_000.0,
        max_notional_to_nav=0.1,
        collateral_yield_annual=0.0,
        transaction_cost_bps=0.0,
        market_data=market,
        signal_lag_sessions=0,
        margin_rate=0.0,
        integer_contracts=True,
    )

    assert result["contracts"].map(float.is_integer).all()
    assert result.loc[1, "futures_pnl"] == 1_000.0
    assert result.loc[2, "futures_pnl"] == -455.0
    assert not result["roll_event"].any()


def test_roll_day_marks_old_contract_before_switching():
    market = _panel({"IC2602": [100.0, 110.0, 110.0], "IC2603": [99.0, 98.0, 98.0]})
    selected = market[market["contract"].eq("IC2602")].iloc[[0]].copy()
    selected = pd.concat([selected, market[(market["trade_date"] == pd.Timestamp("2026-01-05")) & market["contract"].eq("IC2603")]])
    selected["carry_ann"] = 0.05
    selected["signal_carry"] = 0.05

    result = backtest_selected_contracts(
        selected,
        initial_nav=100_000.0,
        max_notional_to_nav=0.1,
        collateral_yield_annual=0.0,
        transaction_cost_bps=10.0,
        market_data=market,
        signal_lag_sessions=0,
        margin_rate=0.0,
        integer_contracts=True,
    )

    assert result.loc[1, "futures_pnl"] == 1_000.0
    assert bool(result.loc[1, "roll_event"])
    assert not bool(result.loc[1, "missing_mark"])
    assert result.loc[1, "trading_cost"] > 0


def test_signal_is_executed_on_next_session():
    market = _panel({"IC2602": [100.0, 110.0, 120.0], "IC2603": [200.0, 190.0, 180.0]})
    selected = market[market["contract"].eq("IC2602")].iloc[[0]].copy()
    selected = pd.concat([selected, market[(market["trade_date"] == pd.Timestamp("2026-01-05")) & market["contract"].eq("IC2603")]])
    selected = pd.concat([selected, market[(market["trade_date"] == pd.Timestamp("2026-01-06")) & market["contract"].eq("IC2603")]])
    selected["carry_ann"] = 0.05
    selected["signal_carry"] = 0.05

    result = backtest_selected_contracts(
        selected,
        initial_nav=100_000.0,
        max_notional_to_nav=0.1,
        collateral_yield_annual=0.0,
        transaction_cost_bps=0.0,
        market_data=market,
        signal_lag_sessions=1,
        margin_rate=0.0,
        integer_contracts=True,
    )

    assert result.loc[0, "contract"] is None
    assert result.loc[1, "contract"] == "IC2602"
    assert result.loc[2, "contract"] == "IC2603"
    assert result.loc[1, "signal_date"] == pd.Timestamp("2026-01-02")
