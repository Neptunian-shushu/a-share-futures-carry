import pandas as pd

from a_share_futures_carry.data.reconcile import reconcile_panels
from a_share_futures_carry.reporting.report import price_benchmark_backtest, price_benchmark_returns


def test_reconcile_panels_reports_price_difference():
    left = pd.DataFrame({"trade_date": ["2026-01-02"], "contract": ["IC2602"], "futures_close": [100.0]})
    right = pd.DataFrame({"trade_date": ["2026-01-02"], "contract": ["IC2602"], "futures_close": [100.5]})
    summary, differences = reconcile_panels(left, right)
    assert summary["matched_rows"] == 1
    assert summary["futures_close_max_abs_diff"] == 0.5
    assert differences.loc[0, "futures_close_abs_diff"] == 0.5


def test_price_benchmark_returns_rejects_duplicate_dates():
    prices = pd.DataFrame({"trade_date": ["2026-01-02", "2026-01-02"], "close": [100.0, 101.0]})
    try:
        price_benchmark_returns(prices)
    except ValueError as exc:
        assert "duplicate" in str(exc).lower()
    else:
        raise AssertionError("duplicate benchmark dates should fail")


def test_price_benchmark_backtest_preserves_price_series_dates_and_nav():
    prices = pd.DataFrame(
        {"trade_date": ["2026-01-02", "2026-01-05"], "close": [100.0, 101.0]}
    )
    backtest = price_benchmark_backtest(prices, initial_nav=1_000.0)
    assert backtest["trade_date"].tolist() == [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-05")]
    assert backtest["nav"].iloc[-1] == 1_010.0
