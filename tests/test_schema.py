import pandas as pd
import pytest

from a_share_futures_carry.data.schema import data_quality_report, prepare_contract_data


def _valid_row():
    return {
        "trade_date": "2026-01-02",
        "contract": "IC2602",
        "family": "ic",
        "futures_close": "6000",
        "spot_close": "6100",
        "expiry_date": "2026-02-20",
        "multiplier": "200",
    }


def test_prepare_contract_data_normalizes_types_and_family():
    result = prepare_contract_data(pd.DataFrame([_valid_row()]))
    assert result.loc[0, "family"] == "IC"
    assert result.loc[0, "futures_close"] == 6000.0
    assert result.loc[0, "dte"] == 49


def test_schema_rejects_duplicate_contract_day():
    data = pd.DataFrame([_valid_row(), _valid_row()])
    report = data_quality_report(data)
    assert not report["ok"]
    assert report["duplicate_trade_date_contract_rows"] == 2
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_contract_data(data)


def test_schema_rejects_non_positive_price():
    row = _valid_row()
    row["futures_close"] = 0
    with pytest.raises(ValueError, match="futures_close"):
        prepare_contract_data(pd.DataFrame([row]))


def test_schema_rejects_non_numeric_optional_field():
    row = _valid_row()
    row["margin_rate"] = "not-a-rate"
    with pytest.raises(ValueError, match="margin_rate"):
        prepare_contract_data(pd.DataFrame([row]))


@pytest.mark.parametrize("column", ["vol", "oi", "spread_bps"])
def test_schema_rejects_negative_execution_or_liquidity_field(column):
    row = _valid_row()
    row[column] = -1
    report = data_quality_report(pd.DataFrame([row]))
    assert not report["ok"]
    with pytest.raises(ValueError, match=column):
        prepare_contract_data(pd.DataFrame([row]))
