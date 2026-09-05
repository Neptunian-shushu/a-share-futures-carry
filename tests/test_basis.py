import pandas as pd

from a_share_futures_carry.signals.basis import annualized_discount


def test_annualized_discount_positive_for_discounted_future():
    spot = pd.Series([7000.0])
    future = pd.Series([6900.0])
    dte = pd.Series([30])
    carry = annualized_discount(spot, future, dte)
    expected = (100.0 / 7000.0) * (365.0 / 30.0)
    assert abs(carry.iloc[0] - expected) < 1e-12
