import pandas as pd

from a_share_futures_carry.strategy.allocation import add_dynamic_carry_allocation


def test_percentile_allocation_has_no_trade_warmup_and_bounded_weight():
    selected = pd.DataFrame(
        {
            "trade_date": pd.bdate_range("2026-01-02", periods=6),
            "signal_carry": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        }
    )
    result = add_dynamic_carry_allocation(
        selected,
        lookback=4,
        min_periods=3,
        entry_threshold=0.5,
    )
    assert (result.loc[:1, "target_weight"] == 0).all()
    assert result.loc[5, "target_weight"] > 0
    assert result["target_weight"].between(0, 1).all()
