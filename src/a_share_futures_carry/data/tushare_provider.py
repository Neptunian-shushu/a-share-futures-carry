"""Tushare-backed historical data loader for CFFEX equity index futures.

Environment variable:
    TUSHARE_TOKEN=<your token>

The loader intentionally returns a normalized contract-level table compatible with
``prepare_contract_data``. Tushare access levels can vary by account, so callers
should handle permission errors and can fall back to local CSV files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

INDEX_CODE_MAP = {
    "IF": "000300.SH",   # CSI 300
    "IH": "000016.SH",   # SSE 50
    "IC": "000905.SH",   # CSI 500
    "IM": "000852.SH",   # CSI 1000
}


@dataclass
class TushareProvider:
    token: str | None = None

    def __post_init__(self) -> None:
        token = self.token or os.getenv("TUSHARE_TOKEN")
        if not token:
            raise ValueError("Missing Tushare token. Set TUSHARE_TOKEN or pass token=...")
        try:
            import tushare as ts
        except ImportError as exc:
            raise ImportError("Install the optional dependency with: pip install '.[tushare]'") from exc
        self._pro = ts.pro_api(token)

    def fetch_contract_metadata(self, families: Iterable[str]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for family in families:
            df = self._pro.fut_basic(exchange="CFFEX", fut_type="1", fut_code=family)
            if df is not None and not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        meta = pd.concat(frames, ignore_index=True)
        keep = [c for c in ["ts_code", "symbol", "fut_code", "list_date", "delist_date", "multiplier"] if c in meta]
        return meta[keep].drop_duplicates("ts_code")

    def fetch_index_daily(self, family: str, start_date: str, end_date: str) -> pd.DataFrame:
        index_code = INDEX_CODE_MAP[family]
        df = self._pro.index_daily(ts_code=index_code, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return pd.DataFrame(columns=["trade_date", "spot_close"])
        out = df[["trade_date", "close"]].copy()
        out["trade_date"] = pd.to_datetime(out["trade_date"])
        out = out.rename(columns={"close": "spot_close"})
        return out.sort_values("trade_date").reset_index(drop=True)

    def fetch_futures_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        df = self._pro.fut_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        out["trade_date"] = pd.to_datetime(out["trade_date"])
        return out.sort_values("trade_date").reset_index(drop=True)

    def build_contract_panel(
        self,
        families: Iterable[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Download and normalize all listed contracts intersecting the requested period."""
        families = tuple(families)
        meta = self.fetch_contract_metadata(families)
        if meta.empty:
            return pd.DataFrame()

        meta["list_date"] = pd.to_datetime(meta["list_date"], errors="coerce")
        meta["delist_date"] = pd.to_datetime(meta["delist_date"], errors="coerce")
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        active = meta[(meta["list_date"] <= end) & (meta["delist_date"] >= start)].copy()

        spot_cache = {
            family: self.fetch_index_daily(family, start_date, end_date)
            for family in families
        }
        frames: list[pd.DataFrame] = []

        for row in active.itertuples(index=False):
            family = str(row.fut_code)
            fut = self.fetch_futures_daily(row.ts_code, start_date, end_date)
            if fut.empty:
                continue
            fut = fut.merge(spot_cache[family], on="trade_date", how="inner")
            fut["contract"] = row.ts_code
            fut["family"] = family
            fut["expiry_date"] = row.delist_date
            fut["multiplier"] = float(row.multiplier)
            fut = fut.rename(columns={"close": "futures_close"})
            keep = [
                "trade_date",
                "contract",
                "family",
                "futures_close",
                "spot_close",
                "expiry_date",
                "multiplier",
                "settle",
                "vol",
                "oi",
            ]
            frames.append(fut[[c for c in keep if c in fut.columns]])

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values(["trade_date", "family", "expiry_date"])
