import importlib.util
from pathlib import Path

import pandas as pd


_MODULE_PATH = Path(__file__).parents[1] / "scripts" / "download_etf_benchmark.py"
_SPEC = importlib.util.spec_from_file_location("download_etf_benchmark", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)

_add_total_return_events = _MODULE._add_total_return_events
_parse_distributions = _MODULE._parse_distributions
_parse_splits = _MODULE._parse_splits


def test_parse_fund_f10_dividend_amount_uses_cash_amount_not_share_count():
    tables = [
        pd.DataFrame(),
        pd.DataFrame(
            {
                "除息日": ["2026-07-15", "2025-01-16"],
                "每10份分红": ["每10份派现金1.4900元", "每10份派现金0.9100元"],
            }
        ),
        pd.DataFrame(),
    ]
    parsed = _parse_distributions(tables)
    assert parsed["distribution"].tolist() == [0.091, 0.149]


def test_parse_fund_f10_split_ratio_and_skip_pre_sample_events():
    tables = [
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(
            {
                "拆分折算日": ["2015-04-14", "2022-08-26"],
                "拆分折算比例": ["1:0.2803", "1:1.1454"],
            }
        ),
    ]
    prices = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2022-07-22", "2022-08-26", "2022-08-29"]),
            "close": [7.187, 7.220, 6.300],
        }
    )
    out = _add_total_return_events(prices, _parse_distributions(tables[:2]), _parse_splits(tables))
    assert out.loc[out["trade_date"].eq("2022-07-22"), "split_factor"].item() == 1.0
    assert out.loc[out["trade_date"].eq("2022-08-29"), "split_factor"].item() == 1.1454
