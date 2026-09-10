"""AkShare-backed free-data provider for CFFEX futures panels.

AkShare's current public interfaces expose CFFEX daily data one trading day at
a time and CSI index history through ``index_zh_a_hist``.  The provider keeps
all vendor-specific column names in this module and returns the same normalized
panel as the Tushare provider.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .schema import prepare_contract_data
from .tushare_provider import INDEX_CODE_MAP, MULTIPLIER_MAP


def _first_column(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    return next((name for name in names if name in df.columns), None)


def _contract_expiry(contract: str) -> pd.Timestamp:
    """Fallback to the third Friday when contract metadata is unavailable."""
    match = re.search(r"([A-Z]+)(\d{4})$", str(contract).upper())
    if not match:
        raise ValueError(f"Cannot infer expiry from contract: {contract}")
    year = 2000 + int(match.group(2)[:2])
    month = int(match.group(2)[2:])
    first = pd.Timestamp(year=year, month=month, day=1)
    days_to_friday = (4 - first.weekday()) % 7
    return first + pd.Timedelta(days=days_to_friday + 14)


def _normalize_index_history(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["trade_date", "spot_close"])
    date_column = _first_column(df, ("日期", "date", "trade_date"))
    close_column = _first_column(df, ("收盘", "close", "收盘价"))
    if date_column is None or close_column is None:
        raise ValueError("AkShare index history lacks date or close columns")
    out = df[[date_column, close_column]].copy()
    out.columns = ["trade_date", "spot_close"]
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out["spot_close"] = pd.to_numeric(out["spot_close"], errors="coerce")
    return out.dropna().sort_values("trade_date").reset_index(drop=True)


def _normalize_cffex_daily(df: pd.DataFrame, trade_date: str | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    symbol_column = _first_column(df, ("symbol", "合约代码", "合约"))
    close_column = _first_column(df, ("close", "收盘", "收盘价"))
    if symbol_column is None or close_column is None:
        raise ValueError("AkShare CFFEX data lacks contract or close columns")
    out = pd.DataFrame()
    out["contract"] = df[symbol_column].astype(str).str.strip().str.upper()
    date_column = _first_column(df, ("date", "日期", "trade_date"))
    if date_column is None:
        if trade_date is None:
            raise ValueError("CFFEX data lacks a trade date")
        out["trade_date"] = pd.Timestamp(trade_date)
    else:
        out["trade_date"] = pd.to_datetime(df[date_column], errors="coerce")
    out["futures_close"] = pd.to_numeric(df[close_column], errors="coerce")
    settle_column = _first_column(df, ("settle", "结算", "结算价"))
    volume_column = _first_column(df, ("volume", "成交量", "成交量(手)"))
    oi_column = _first_column(df, ("open_interest", "持仓量", "空盘量"))
    if settle_column:
        out["settle"] = pd.to_numeric(df[settle_column], errors="coerce")
    if volume_column:
        out["vol"] = pd.to_numeric(df[volume_column], errors="coerce")
    if oi_column:
        out["oi"] = pd.to_numeric(df[oi_column], errors="coerce")
    out = out[out["contract"].str.match(r"^(IF|IH|IC|IM)\d{4}$")].copy()
    out["family"] = out["contract"].str.extract(r"^(IF|IH|IC|IM)", expand=False)
    out["multiplier"] = out["family"].map(MULTIPLIER_MAP)
    return out.dropna(subset=["futures_close"]).reset_index(drop=True)


@dataclass
class AkshareProvider:
    """Load CFFEX futures and CSI index history through AkShare."""

    client: object | None = None

    def __post_init__(self) -> None:
        if self.client is not None:
            return
        try:
            import akshare as ak
        except ImportError as exc:
            raise ImportError("Install the optional dependency with: pip install '.[akshare]'") from exc
        self.client = ak

    def fetch_index_daily(self, family: str, start_date: str, end_date: str) -> pd.DataFrame:
        code = INDEX_CODE_MAP[family]
        symbol = code.split(".")[0]
        errors: list[Exception] = []
        if hasattr(self.client, "index_zh_a_hist"):
            try:
                raw = self.client.index_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                )
                normalized = _normalize_index_history(raw)
                if not normalized.empty:
                    return normalized
            except Exception as exc:  # pragma: no cover - vendor/network dependent
                errors.append(exc)
        if hasattr(self.client, "stock_zh_index_daily"):
            try:
                raw = self.client.stock_zh_index_daily(symbol=f"sh{symbol}")
                normalized = _normalize_index_history(raw)
                start = pd.Timestamp(start_date)
                end = pd.Timestamp(end_date)
                return normalized[normalized["trade_date"].between(start, end)].reset_index(drop=True)
            except Exception as exc:  # pragma: no cover - vendor/network dependent
                errors.append(exc)
        if errors:
            raise RuntimeError(f"AkShare index history unavailable for {family}") from errors[-1]
        raise AttributeError("AkShare client lacks an index history interface")

    def fetch_contract_info(self, date: str) -> pd.DataFrame:
        if not hasattr(self.client, "futures_contract_info_cffex"):
            return pd.DataFrame()
        try:
            raw = self.client.futures_contract_info_cffex(date=date)
        except Exception:
            # Contract metadata is an enhancement; the contract code still
            # allows a deterministic third-Friday fallback for the panel.
            return pd.DataFrame()
        if raw is None or raw.empty:
            return pd.DataFrame()
        contract_column = _first_column(raw, ("合约代码", "symbol", "contract"))
        expiry_column = _first_column(raw, ("最后交易日", "expiry_date", "delist_date"))
        if contract_column is None or expiry_column is None:
            return pd.DataFrame()
        out = raw[[contract_column, expiry_column]].copy()
        out.columns = ["contract", "expiry_date"]
        out["contract"] = out["contract"].astype(str).str.strip().str.upper()
        out["expiry_date"] = pd.to_datetime(out["expiry_date"], errors="coerce")
        return out.dropna().drop_duplicates("contract")

    def build_contract_panel(
        self,
        families: Iterable[str],
        start_date: str,
        end_date: str,
        include_contract_info: bool = False,
    ) -> pd.DataFrame:
        families = tuple(str(f).upper() for f in families)
        invalid = set(families).difference(INDEX_CODE_MAP)
        if invalid:
            raise ValueError(f"Unsupported futures families: {sorted(invalid)}")

        daily_frames: list[pd.DataFrame] = []
        expiry_frames: list[pd.DataFrame] = []
        bulk_loaded = False
        if hasattr(self.client, "get_futures_daily"):
            try:
                raw = self.client.get_futures_daily(
                    start_date=start_date,
                    end_date=end_date,
                    market="CFFEX",
                )
                bulk = _normalize_cffex_daily(raw)
                bulk = bulk[bulk["family"].isin(families)].copy()
                if not bulk.empty:
                    daily_frames.append(bulk)
                    bulk_loaded = True
            except Exception:
                # Fall back to the older one-day interface below.
                pass

        if not bulk_loaded:
            dates = pd.date_range(start_date, end_date, freq="D")
            for date in dates:
                date_text = date.strftime("%Y%m%d")
                raw = self.client.get_cffex_daily(date=date_text)
                day = _normalize_cffex_daily(raw, date_text)
                if day.empty:
                    continue
                daily_frames.append(day[day["family"].isin(families)])

        if not daily_frames:
            return pd.DataFrame()

        if include_contract_info:
            info_dates = (
                pd.to_datetime(daily_frames[0]["trade_date"])
                .drop_duplicates()
                .sort_values()
                .dt.strftime("%Y%m%d")
            )
            for date_text in info_dates:
                info = self.fetch_contract_info(date_text)
                if not info.empty:
                    expiry_frames.append(info)

        panel = pd.concat(daily_frames, ignore_index=True)
        if expiry_frames:
            expiry = pd.concat(expiry_frames, ignore_index=True).drop_duplicates("contract")
            panel = panel.merge(expiry, on="contract", how="left")
        else:
            panel["expiry_date"] = panel["contract"].map(_contract_expiry)
        panel["expiry_date"] = panel["expiry_date"].fillna(panel["contract"].map(_contract_expiry))

        spot_frames = []
        for family in families:
            spot = self.fetch_index_daily(family, start_date, end_date)
            spot["family"] = family
            spot_frames.append(spot)
        spot_panel = pd.concat(spot_frames, ignore_index=True)
        panel = panel.merge(spot_panel, on=["trade_date", "family"], how="inner")
        return prepare_contract_data(panel)
