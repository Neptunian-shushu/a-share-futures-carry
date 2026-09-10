"""Direct CFFEX public ZIP provider with no Tushare dependency.

CFFEX publishes monthly ZIP archives containing one CSV per trading day.  This
provider downloads each month once, parses the daily files, and delegates only
the cash-index leg to the existing AkShare/Sina fallback provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO, StringIO
from pathlib import Path
import re
from typing import Iterable
from urllib.request import Request, urlopen
import zipfile

import pandas as pd

from .akshare_provider import _observed_expiry, _normalize_cffex_daily
from .schema import prepare_contract_data
from .tushare_provider import INDEX_CODE_MAP


def _decode_csv(payload: bytes) -> pd.DataFrame:
    for encoding in ("gb2312", "gb18030", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(StringIO(payload.decode(encoding)))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("gb2312", payload, 0, 1, "Unable to decode CFFEX CSV")


@dataclass
class CffexPublicProvider:
    """Download CFFEX futures from official monthly public archives."""

    index_provider: object | None = None
    cache_dir: str | Path | None = None
    timeout: int = 30
    workers: int = 4
    last_errors: list[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
        if self.index_provider is None:
            try:
                from .akshare_provider import AkshareProvider

                self.index_provider = AkshareProvider()
            except ImportError as exc:
                raise ImportError(
                    "Install the optional dependency with: pip install '.[akshare]'"
                ) from exc
        if self.cache_dir is not None:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def month_url(period: pd.Period) -> str:
        month = period.strftime("%Y%m")
        return f"http://www.cffex.com.cn/sj/historysj/{month}/zip/{month}.zip"

    def _month_payload(self, period: pd.Period) -> bytes:
        month = period.strftime("%Y%m")
        cache_path = Path(self.cache_dir) / f"{month}.zip" if self.cache_dir else None
        if cache_path and cache_path.exists():
            return cache_path.read_bytes()
        request = Request(
            self.month_url(period),
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urlopen(request, timeout=self.timeout) as response:
            payload = response.read()
        if cache_path:
            cache_path.write_bytes(payload)
        return payload

    def _parse_month(self, payload: bytes, families: tuple[str, ...]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            for filename in archive.namelist():
                match = re.search(r"(20\d{6})_1\.csv$", Path(filename).name)
                if not match:
                    continue
                raw = _decode_csv(archive.read(filename))
                day = _normalize_cffex_daily(raw, match.group(1))
                if not day.empty:
                    frames.append(day[day["family"].isin(families)])
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def fetch_futures_daily(
        self,
        families: Iterable[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        families = tuple(str(f).upper() for f in families)
        periods = pd.period_range(start_date, end_date, freq="M")
        frames: dict[pd.Period, pd.DataFrame] = {}
        errors: list[str] = []

        def load(period: pd.Period) -> tuple[pd.Period, pd.DataFrame]:
            return period, self._parse_month(self._month_payload(period), families)

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_period = {
                executor.submit(load, period): period for period in periods
            }
            for future in as_completed(future_to_period):
                period = future_to_period[future]
                try:
                    period, month = future.result()
                except Exception as exc:
                    errors.append(f"{period}: {type(exc).__name__}: {exc}")
                    continue
                if not month.empty:
                    frames[period] = month
        self.last_errors = sorted(errors)
        if not frames:
            detail = "; ".join(errors[:3])
            raise RuntimeError(f"No CFFEX public data returned. {detail}")
        panel = pd.concat([frames[period] for period in sorted(frames)], ignore_index=True)
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        return panel[panel["trade_date"].between(start, end)].reset_index(drop=True)

    def build_contract_panel(
        self,
        families: Iterable[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        families = tuple(str(f).upper() for f in families)
        invalid = set(families).difference(INDEX_CODE_MAP)
        if invalid:
            raise ValueError(f"Unsupported futures families: {sorted(invalid)}")
        futures = self.fetch_futures_daily(families, start_date, end_date)
        spots: list[pd.DataFrame] = []
        for family in families:
            spot = self.index_provider.fetch_index_daily(family, start_date, end_date)
            spot["family"] = family
            spots.append(spot)
        spot_panel = pd.concat(spots, ignore_index=True)
        panel = futures.merge(spot_panel, on=["trade_date", "family"], how="inner")
        panel["expiry_date"] = _observed_expiry(panel)
        return prepare_contract_data(panel)
