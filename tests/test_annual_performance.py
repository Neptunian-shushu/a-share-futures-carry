from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.report_annual_performance import annual_return_table, build_annual_performance


def test_build_annual_performance_compounds_returns_and_marks_partial_year():
    curves = pd.DataFrame(
        {
            "strategy": ["demo"] * 4,
            "trade_date": pd.to_datetime(["2022-07-22", "2022-12-30", "2023-01-03", "2023-12-29"]),
            "return": [0.10, 0.20, -0.10, 0.05],
        }
    )

    result = build_annual_performance(curves, strategies=["demo"])

    assert result.loc[result["year"].eq(2022), "total_return"].iloc[0] == pytest.approx(0.32)
    assert bool(result.loc[result["year"].eq(2022), "partial_year"].iloc[0])
    assert not bool(result.loc[result["year"].eq(2023), "partial_year"].iloc[0])


def test_annual_return_table_pivots_strategy_labels():
    details = pd.DataFrame(
        {
            "year": [2022, 2022],
            "strategy_label": ["A", "B"],
            "total_return": [0.1, -0.1],
        }
    )

    table = annual_return_table(details)

    assert list(table.columns) == ["year", "A", "B"]
    assert table.loc[0, "A"] == 0.1
