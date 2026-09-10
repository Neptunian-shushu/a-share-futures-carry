"""Download a free ETF benchmark snapshot through AkShare/Sina."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _sina_symbol(symbol: str) -> str:
    code = symbol.lower().removeprefix("sh").removeprefix("sz")
    if code.startswith(("5", "6")):
        return f"sh{code}"
    return f"sz{code}"


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
        "price_semantics": "raw_close; excludes cash distributions unless the input is replaced with a total-return series",
    }
    metadata_path = output.with_suffix(output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
