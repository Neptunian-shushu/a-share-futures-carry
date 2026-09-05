import pandas as pd

from a_share_futures_carry.strategy.selection import select_max_carry, select_nth_expiry


def sample_contracts():
    return pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-05"] * 4),
            "family": ["IC", "IC", "IM", "IM"],
            "contract": ["IC1", "IC2", "IM1", "IM2"],
            "expiry_date": pd.to_datetime(["2026-01-20", "2026-02-20", "2026-01-20", "2026-02-20"]),
            "dte": [15, 46, 15, 46],
            "carry_ann": [0.04, 0.05, 0.06, 0.08],
        }
    )


def test_select_max_carry_across_families():
    selected = select_max_carry(sample_contracts(), ("IC", "IM"), 5, 120)
    assert selected.iloc[0]["contract"] == "IM2"


def test_select_second_expiry_within_family():
    selected = select_nth_expiry(sample_contracts(), 2, ("IC",))
    assert selected.iloc[0]["contract"] == "IC2"
