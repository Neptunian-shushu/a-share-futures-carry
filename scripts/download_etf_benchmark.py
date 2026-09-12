"""Download a free ETF benchmark snapshot with reinvested distributions.

The daily market history comes from AkShare/Sina. Cash distributions and share
split events come from Eastmoney's public fund F10 page so the resulting CSV can
be used as a total-return benchmark without a Tushare token.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd


EASTMONEY_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
    )
}


def _sina_symbol(symbol: str) -> str:
    code = symbol.lower().removeprefix("sh").removeprefix("sz")
    if code.startswith(("5", "6")):
        return f"sh{code}"
    return f"sz{code}"


def _fund_code(symbol: str) -> str:
    return symbol.lower().removeprefix("sh").removeprefix("sz")


def _fund_f10_tables(symbol: str) -> list[pd.DataFrame]:
    """Read the public F10 dividend/split tables for one fund."""
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - optional provider dependency
        raise RuntimeError("Install the optional akshare dependency first") from exc

    code = _fund_code(symbol)
    url = f"https://fundf10.eastmoney.com/fhsp_{code}.html"
    response = requests.get(url, headers=EASTMONEY_HEADERS, timeout=30)
    response.raise_for_status()
    try:
        return pd.read_html(StringIO(response.text))
    except ValueError as exc:
        raise RuntimeError(f"Could not parse the public fund F10 page for {code}") from exc


def _parse_distributions(tables: list[pd.DataFrame]) -> pd.DataFrame:
    if len(tables) < 2:
        raise RuntimeError("Fund F10 page does not contain a dividend table")
    table = tables[1]
    if table.empty or "除息日" not in table or "每10份分红" not in table:
        return pd.DataFrame(columns=["trade_date", "distribution"])
    out = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(table["除息日"], errors="coerce"),
            "distribution": (
                table["每10份分红"].astype(str)
                .str.extract(r"现金\s*([0-9]+(?:\.[0-9]+)?)", expand=False)
                .pipe(pd.to_numeric, errors="coerce")
                / 10.0
            ),
        }
    )
    return (
        out.dropna(subset=["trade_date", "distribution"])
        .loc[lambda frame: frame["distribution"] >= 0]
        .groupby("trade_date", as_index=False)["distribution"]
        .sum()
    )


def _parse_splits(tables: list[pd.DataFrame]) -> pd.DataFrame:
    if len(tables) < 3:
        return pd.DataFrame(columns=["event_date", "split_factor"])
    table = tables[2]
    required = {"拆分折算日", "拆分折算比例"}
    if table.empty or not required.issubset(table.columns):
        return pd.DataFrame(columns=["event_date", "split_factor"])
    out = pd.DataFrame(
        {
            "event_date": pd.to_datetime(table["拆分折算日"], errors="coerce"),
            "split_factor": (
                table["拆分折算比例"].astype(str)
                .str.extract(r":\s*([0-9]+(?:\.[0-9]+)?)", expand=False)
                .pipe(pd.to_numeric, errors="coerce")
            ),
        }
    )
    return (
        out.dropna(subset=["event_date", "split_factor"])
        .loc[lambda frame: frame["split_factor"] > 0]
        .sort_values("event_date")
        .reset_index(drop=True)
    )


def _add_total_return_events(
    prices: pd.DataFrame,
    distributions: pd.DataFrame,
    splits: pd.DataFrame,
) -> pd.DataFrame:
    """Attach cash distributions and effective split factors to price dates."""
    out = prices.copy()
    out["distribution"] = 0.0
    if not distributions.empty:
        event_distributions = distributions.groupby("trade_date", as_index=True)["distribution"].sum()
        out["distribution"] = out["trade_date"].map(event_distributions).fillna(0.0)

    # The F10 split date is the record/event date. The first later trading
    # session is used as the effective price date (e.g. the 2022-08-26 split is
    # reflected in the 2022-08-29 ETF price after the weekend).
    out["split_factor"] = 1.0
    first_price_date = out["trade_date"].min()
    for _, event in splits.iterrows():
        if event["event_date"] < first_price_date:
            continue
        effective_dates = out.loc[out["trade_date"] > event["event_date"], "trade_date"]
        if effective_dates.empty:
            continue
        effective_date = effective_dates.min()
        out.loc[out["trade_date"].eq(effective_date), "split_factor"] *= float(event["split_factor"])
    return out


def download_etf(symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    try:
        import akshare as ak
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError("Install the optional akshare dependency first") from exc

    raw = ak.fund_etf_hist_sina(symbol=_sina_symbol(symbol))
    if raw.empty:
        raise RuntimeError(f"No ETF history returned for {symbol}")
    missing = {"date", "close"}.difference(raw.columns)
    if missing:
        raise RuntimeError(f"ETF response is missing columns: {sorted(missing)}")
    out = raw.rename(columns={"date": "trade_date"}).copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["trade_date", "close"])
    out = out[out["close"] > 0].sort_values("trade_date")
    if start:
        out = out[out["trade_date"] >= pd.Timestamp(start)]
    if end:
        out = out[out["trade_date"] <= pd.Timestamp(end)]
    if out.empty:
        raise RuntimeError("ETF history is empty after date filtering")
    out["source"] = "akshare.fund_etf_hist_sina"
    tables = _fund_f10_tables(symbol)
    out = _add_total_return_events(
        out,
        _parse_distributions(tables),
        _parse_splits(tables),
    )
    return out.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="510500", help="ETF code, e.g. 510500")
    parser.add_argument("--start", default=None, help="Inclusive date, YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Inclusive date, YYYY-MM-DD")
    parser.add_argument("--output", default="data/raw/etf_510500_sina.csv")
    args = parser.parse_args()

    data = download_etf(args.symbol, args.start, args.end)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False, date_format="%Y-%m-%d")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    metadata = {
        "provider": "akshare.fund_etf_hist_sina",
        "symbol": args.symbol,
        "requested_start": args.start,
        "requested_end": args.end,
        "observed_start": data["trade_date"].min().date().isoformat(),
        "observed_end": data["trade_date"].max().date().isoformat(),
        "rows": int(len(data)),
        "sha256": digest,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "price_semantics": "raw_close plus event columns; use distribution and split_factor to calculate total return",
        "distribution_column": "distribution",
        "distribution_semantics": "cash distribution per share, aligned to the ex-dividend date and reinvested immediately",
        "distribution_source": f"https://fundf10.eastmoney.com/fhsp_{_fund_code(args.symbol)}.html",
        "distribution_events": int((data["distribution"] > 0).sum()),
        "distribution_total_per_share": float(data["distribution"].sum()),
        "split_factor_column": "split_factor",
        "split_semantics": "new shares per old share, aligned to the first trading session after the F10 event date",
    }
    metadata_path = output.with_suffix(output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
