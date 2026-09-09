"""Input-data validation and normalization helpers."""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {
    "trade_date",
    "contract",
    "family",
    "futures_close",
    "spot_close",
    "expiry_date",
    "multiplier",
}

NUMERIC_COLUMNS = (
    "futures_close",
    "spot_close",
    "multiplier",
    "settle",
    "vol",
    "oi",
    "funding_rate",
    "dividend_yield",
    "margin_rate",
)


def data_quality_report(df: pd.DataFrame) -> dict[str, object]:
    """Return non-mutating quality diagnostics for a raw contract panel.

    The report is intentionally useful before normalization, so a command-line
    data check can explain all issues instead of stopping at the first one.
    """
    issues: list[str] = []
    if not isinstance(df, pd.DataFrame):
        return {"ok": False, "issues": ["input is not a pandas DataFrame"]}

    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        issues.append(f"missing required columns: {missing}")

    report: dict[str, object] = {
        "ok": False,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_columns": missing,
        "issues": issues,
    }

    if missing:
        return report

    trade_dates = pd.to_datetime(df["trade_date"], errors="coerce")
    expiry_dates = pd.to_datetime(df["expiry_date"], errors="coerce")
    bad_trade_dates = int(trade_dates.isna().sum())
    bad_expiry_dates = int(expiry_dates.isna().sum())
    report["invalid_trade_dates"] = bad_trade_dates
    report["invalid_expiry_dates"] = bad_expiry_dates
    if bad_trade_dates:
        issues.append(f"invalid trade_date values: {bad_trade_dates}")
    if bad_expiry_dates:
        issues.append(f"invalid expiry_date values: {bad_expiry_dates}")

    numeric_invalid: dict[str, int] = {}
    for column in NUMERIC_COLUMNS:
        if column in df:
            raw_values = df[column]
            parsed = pd.to_numeric(raw_values, errors="coerce")
            invalid = int((raw_values.notna() & parsed.isna()).sum())
            if invalid:
                numeric_invalid[column] = invalid
    report["invalid_numeric_values"] = numeric_invalid
    if numeric_invalid:
        issues.append(f"invalid numeric values: {numeric_invalid}")

    duplicate_keys = int(df.duplicated(["trade_date", "contract"], keep=False).sum())
    report["duplicate_trade_date_contract_rows"] = duplicate_keys
    if duplicate_keys:
        issues.append("duplicate (trade_date, contract) rows")

    dte = (expiry_dates - trade_dates).dt.days
    expired = int((dte < 0).sum())
    report["expired_rows"] = expired
    if expired:
        issues.append(f"trade_date after expiry_date: {expired} rows")

    for column in ("futures_close", "spot_close", "multiplier"):
        if column in df:
            values = pd.to_numeric(df[column], errors="coerce")
            invalid = int((values <= 0).sum())
            report[f"non_positive_{column}"] = invalid
            if invalid:
                issues.append(f"non-positive {column} values: {invalid}")
    if "margin_rate" in df:
        values = pd.to_numeric(df["margin_rate"], errors="coerce")
        invalid = int((values < 0).sum())
        report["negative_margin_rate"] = invalid
        if invalid:
            issues.append(f"negative margin_rate values: {invalid}")

    report["date_start"] = trade_dates.min()
    report["date_end"] = trade_dates.max()
    report["families"] = sorted(df["family"].dropna().astype(str).str.upper().unique().tolist())
    report["issues"] = issues
    report["ok"] = not issues
    return report


def prepare_contract_data(df: pd.DataFrame, *, drop_expired: bool = False) -> pd.DataFrame:
    """Validate and normalize daily contract-level futures data.

    ``drop_expired`` is available for exploratory imports, but the default is
    strict so an incorrect expiry mapping cannot silently change a backtest.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out["expiry_date"] = pd.to_datetime(out["expiry_date"], errors="coerce")
    if out[["trade_date", "expiry_date"]].isna().any().any():
        raise ValueError("trade_date and expiry_date must contain valid dates")

    out["family"] = out["family"].astype(str).str.upper().str.strip()
    out["contract"] = out["contract"].astype(str).str.strip()
    for column in NUMERIC_COLUMNS:
        if column in out:
            raw_values = out[column]
            parsed = pd.to_numeric(raw_values, errors="coerce")
            if (raw_values.notna() & parsed.isna()).any():
                raise ValueError(f"{column} contains non-numeric values")
            out[column] = parsed

    if out["family"].isin(["", "NAN", "NONE"]).any():
        raise ValueError("family must not be empty")
    if out["contract"].isin(["", "NAN", "NONE"]).any():
        raise ValueError("contract must not be empty")

    required_numeric = ["futures_close", "spot_close", "multiplier"]
    if out[required_numeric].isna().any().any():
        bad = out[required_numeric].isna().sum()
        raise ValueError(f"Required numeric columns contain nulls: {bad[bad > 0].to_dict()}")
    for column in ("futures_close", "spot_close", "multiplier"):
        if (out[column] <= 0).any():
            raise ValueError(f"{column} must be strictly positive")
    if "margin_rate" in out and ((out["margin_rate"].notna()) & (out["margin_rate"] < 0)).any():
        raise ValueError("margin_rate must be non-negative when provided")
    if out[["trade_date", "contract"]].duplicated().any():
        raise ValueError("Duplicate (trade_date, contract) rows found")

    out["dte"] = (out["expiry_date"] - out["trade_date"]).dt.days
    if (out["dte"] < 0).any() and not drop_expired:
        raise ValueError("trade_date cannot be after expiry_date")
    out = out[out["dte"] >= 0].sort_values(["trade_date", "family", "expiry_date", "contract"])
    return out.reset_index(drop=True)
