import pandas as pd

from a_share_futures_carry.data.akshare_provider import (
    AkshareProvider,
    _normalize_cffex_daily,
    _normalize_index_history,
)


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


class _FakeAkshare:
    def get_cffex_daily(self, date):
        return pd.DataFrame(
            {
                "合约代码": ["IC2602"],
                "收盘": [6000.0],
                "结算价": [5999.0],
                "成交量": [1000],
                "持仓量": [2000],
            }
        )

    def futures_contract_info_cffex(self, date):
        return pd.DataFrame({"合约代码": ["IC2602"], "最后交易日": ["2026-02-20"]})

    def index_zh_a_hist(self, symbol, period, start_date, end_date):
        return pd.DataFrame({"日期": ["2026-01-02", "2026-01-03"], "收盘": [6100.0, 6110.0]})


def test_akshare_provider_builds_normalized_panel_with_fake_client():
    panel = AkshareProvider(client=_FakeAkshare()).build_contract_panel(
        ["IC"], "20260102", "20260103"
    )
    assert len(panel) == 2
    assert panel["contract"].unique().tolist() == ["IC2602"]
    assert panel["expiry_date"].iloc[0] == pd.Timestamp("2026-02-20")
    assert panel["spot_close"].tolist() == [6100.0, 6110.0]


class _FakeAkshareSinaFallback(_FakeAkshare):
    def index_zh_a_hist(self, symbol, period, start_date, end_date):
        raise RuntimeError("primary index endpoint unavailable")

    def stock_zh_index_daily(self, symbol):
        return pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
                "close": [6090.0, 6100.0, 6110.0],
            }
        )


def test_akshare_provider_falls_back_to_sina_index_history():
    panel = AkshareProvider(client=_FakeAkshareSinaFallback()).build_contract_panel(
        ["IC"], "20260102", "20260103"
    )
    assert panel["spot_close"].tolist() == [6100.0, 6110.0]
