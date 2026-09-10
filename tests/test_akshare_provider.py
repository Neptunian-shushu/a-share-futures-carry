import pandas as pd
from io import BytesIO
import zipfile

from a_share_futures_carry.data.cffex_public_provider import CffexPublicProvider
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


def test_normalize_legacy_cffex_daily_columns():
    raw = pd.DataFrame(
        {
            "合约代码": ["IC2208"],
            "今收盘": [6244.8],
            "今结算": [6233.2],
            "成交量": [76054],
            "持仓量": [96095],
        }
    )
    result = _normalize_cffex_daily(raw, "20220722")
    assert result.loc[0, "futures_close"] == 6244.8
    assert result.loc[0, "settle"] == 6233.2


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


class _FakeAkshareBulk(_FakeAkshare):
    def get_futures_daily(self, start_date, end_date, market):
        assert market == "CFFEX"
        return pd.DataFrame(
            {
                "symbol": ["IC2602", "IO2602"],
                "date": ["2026-01-02", "2026-01-02"],
                "close": [6000.0, 10.0],
                "settle": [5999.0, 9.0],
                "volume": [1000, 20],
                "open_interest": [2000, 40],
            }
        )

    def get_cffex_daily(self, date):
        raise AssertionError("bulk interface should be preferred")


def test_akshare_provider_prefers_bulk_cffex_history():
    panel = AkshareProvider(client=_FakeAkshareBulk()).build_contract_panel(
        ["IC"], "20260102", "20260102"
    )
    assert len(panel) == 1
    assert panel.loc[0, "contract"] == "IC2602"
    assert panel.loc[0, "expiry_date"] == pd.Timestamp("2026-02-20")


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


def test_akshare_provider_ignores_contract_info_endpoint_errors():
    class _BrokenContractInfo(_FakeAkshare):
        def futures_contract_info_cffex(self, date):
            raise RuntimeError("metadata endpoint unavailable")

    provider = AkshareProvider(client=_BrokenContractInfo())
    info = provider.fetch_contract_info("20260102")
    assert info.empty


def test_cffex_public_provider_parses_monthly_zip_without_tushare():
    raw = pd.DataFrame(
        {
            "合约代码": ["IC2501", "IM2501"],
            "收盘": [5600.0, 6100.0],
            "结算价": [5598.0, 6098.0],
            "成交量": [1000, 1200],
            "持仓量": [2000, 2400],
        }
    )
    payload = BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("20250102_1.csv", raw.to_csv(index=False).encode("gb2312"))

    class _FakeIndexProvider:
        def fetch_index_daily(self, family, start_date, end_date):
            return pd.DataFrame(
                {
                    "trade_date": [pd.Timestamp("2025-01-02")],
                    "spot_close": [5650.0 if family == "IC" else 6150.0],
                }
            )

    provider = CffexPublicProvider(index_provider=_FakeIndexProvider())
    provider._month_payload = lambda period: payload.getvalue()
    panel = provider.build_contract_panel(["IC", "IM"], "20250102", "20250102")
    assert panel["contract"].tolist() == ["IC2501", "IM2501"]
    assert panel["multiplier"].tolist() == [200.0, 200.0]
