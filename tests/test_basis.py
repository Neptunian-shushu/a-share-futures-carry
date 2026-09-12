import pandas as pd

from a_share_futures_carry.signals.basis import (
    add_carry_columns,
    add_cost_adjusted_carry,
    add_net_carry_score,
    annualized_discount,
)


def test_annualized_discount_positive_for_discounted_future():
    spot = pd.Series([7000.0])
    future = pd.Series([6900.0])
    dte = pd.Series([30])
    carry = annualized_discount(spot, future, dte)
    expected = (100.0 / 7000.0) * (365.0 / 30.0)
    assert abs(carry.iloc[0] - expected) < 1e-12


def test_fair_value_adjustment_creates_excess_carry():
    data = pd.DataFrame(
        {
            "spot_close": [100.0],
            "futures_close": [99.0],
            "dte": [365],
        }
    )
    result = add_carry_columns(
        data,
        use_fair_value_adjustment=True,
        funding_rate_annual=0.02,
        dividend_yield_annual=0.01,
    )
    assert result.loc[0, "fair_value_futures"] > 100.0
    assert result.loc[0, "excess_carry"] > result.loc[0, "carry_ann"]
    assert result.loc[0, "signal_carry"] == result.loc[0, "excess_carry"]


def test_cost_adjusted_carry_penalizes_short_dte_more():
    data = pd.DataFrame({"signal_carry": [0.05, 0.05], "dte": [10, 60]})
    result = add_cost_adjusted_carry(data, switch_cost_bps=2.0)
    assert result.loc[0, "selection_score"] < result.loc[1, "selection_score"]
    assert result.loc[0, "switch_cost_ann"] > result.loc[1, "switch_cost_ann"]


def test_add_net_carry_score_uses_row_level_rates_and_roll_cost():
    data = pd.DataFrame(
        {
            "carry_ann": [0.08],
            "dte": [30],
            "funding_rate": [0.03],
            "dividend_yield": [0.02],
        }
    )
    result = add_net_carry_score(data, switch_cost_bps=6.0, day_count=360)
    expected = 0.08 + 0.02 - 0.03 - (6.0 / 10_000.0) * 360 / 30
    assert abs(result.loc[0, "net_carry_score"] - expected) < 1e-12
