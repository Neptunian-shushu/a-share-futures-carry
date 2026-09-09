"""Independent data-source reconciliation helpers."""

from __future__ import annotations

import pandas as pd

KEY_COLUMNS = ["trade_date", "contract"]
COMPARE_COLUMNS = ["futures_close", "settle", "spot_close", "vol", "oi"]


def reconcile_panels(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    tolerance: float = 1e-6,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Compare two normalized panels and return summary plus row-level differences."""
    for name, frame in (("left", left), ("right", right)):
        missing = set(KEY_COLUMNS).difference(frame.columns)
        if missing:
            raise ValueError(f"{name} panel missing key columns: {sorted(missing)}")
    left_keyed = left.copy()
    right_keyed = right.copy()
    for frame in (left_keyed, right_keyed):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["contract"] = frame["contract"].astype(str)
    merged = left_keyed.merge(
        right_keyed,
        on=KEY_COLUMNS,
        how="outer",
        suffixes=("_left", "_right"),
        indicator=True,
    )
    matched = merged[merged["_merge"] == "both"].copy()
    differences = matched[KEY_COLUMNS].copy()
    for column in COMPARE_COLUMNS:
        left_column = f"{column}_left"
        right_column = f"{column}_right"
        if left_column not in matched or right_column not in matched:
            continue
        left_values = pd.to_numeric(matched[left_column], errors="coerce")
        right_values = pd.to_numeric(matched[right_column], errors="coerce")
        differences[f"{column}_abs_diff"] = (left_values - right_values).abs()
    difference_columns = [column for column in differences if column.endswith("_abs_diff")]
    if difference_columns:
        differences["max_abs_diff"] = differences[difference_columns].max(axis=1, skipna=True)
        differences = differences.sort_values("max_abs_diff", ascending=False)

    summary: dict[str, object] = {
        "left_rows": int(len(left)),
        "right_rows": int(len(right)),
        "matched_rows": int((merged["_merge"] == "both").sum()),
        "left_only_rows": int((merged["_merge"] == "left_only").sum()),
        "right_only_rows": int((merged["_merge"] == "right_only").sum()),
        "tolerance": tolerance,
    }
    for column in difference_columns:
        values = differences[column].dropna()
        summary[column.replace("_abs_diff", "_mean_abs_diff")] = float(values.mean()) if not values.empty else None
        summary[column.replace("_abs_diff", "_max_abs_diff")] = float(values.max()) if not values.empty else None
        summary[column.replace("_abs_diff", "_within_tolerance_ratio")] = (
            float((values <= tolerance).mean()) if not values.empty else None
        )
    return summary, differences.reset_index(drop=True)
