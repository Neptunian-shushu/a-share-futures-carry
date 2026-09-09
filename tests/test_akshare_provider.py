import pandas as pd

from a_share_futures_carry.data.akshare_provider import _normalize_cffex_daily, _normalize_index_history


def test_normalize_akshare_cffex_daily_columns():
    raw = pd.DataFrame(
        {
            "合约代码": ["IC2602", "IO2602"],
            "收盘": [6000, 10],
            "结算价": [5998, 9],
            "成交量": [1000, 20],
            "持仓量": [2000, 40],
        }
    )
    result = _normalize_cffex_daily(raw, "20260102")
    assert result["contract"].tolist() == ["IC2602"]
    assert result.loc[0, "family"] == "IC"
    assert result.loc[0, "multiplier"] == 200.0


def test_normalize_akshare_index_history_columns():
    raw = pd.DataFrame({"日期": ["2026-01-02"], "收盘": [6000]})
    result = _normalize_index_history(raw)
    assert result.loc[0, "trade_date"] == pd.Timestamp("2026-01-02")
    assert result.loc[0, "spot_close"] == 6000.0
