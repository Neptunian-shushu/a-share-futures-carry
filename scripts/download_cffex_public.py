"""Download a normalized CFFEX panel from official monthly public archives."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a_share_futures_carry.data.cffex_public_provider import CffexPublicProvider


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="+", default=["IC", "IM"])
    parser.add_argument("--start", required=True, help="YYYYMMDD")
    parser.add_argument("--end", required=True, help="YYYYMMDD")
    parser.add_argument("--output", default="data/raw/cffex_panel_public.csv")
    parser.add_argument("--cache-dir", default="data/raw/cffex_monthly_cache")
    args = parser.parse_args()

    panel = CffexPublicProvider(cache_dir=args.cache_dir).build_contract_panel(
        args.families, args.start, args.end
    )
    if panel.empty:
        raise SystemExit("No data returned. Check CFFEX and index endpoint availability.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output, index=False)
    observed_dates = pd.to_datetime(panel["trade_date"])
    metadata = {
        "provider": "cffex_official_monthly_zip+akshare_index_fallback",
        "families": args.families,
        "start_date": args.start,
        "end_date": args.end,
        "observed_start_date": observed_dates.min().strftime("%Y-%m-%d"),
        "observed_end_date": observed_dates.max().strftime("%Y-%m-%d"),
        "rows": len(panel),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_output = output.with_suffix(output.suffix + ".json")
    metadata_output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(panel):,} rows to {output}")
    print(f"Saved metadata to {metadata_output}")


if __name__ == "__main__":
    main()
