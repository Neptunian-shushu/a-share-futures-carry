"""Download a normalized CFFEX panel through the free AkShare provider."""

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

from a_share_futures_carry.data.akshare_provider import AkshareProvider


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="+", default=["IC", "IM"])
    parser.add_argument("--start", required=True, help="YYYYMMDD")
    parser.add_argument("--end", required=True, help="YYYYMMDD")
    parser.add_argument("--output", default="data/raw/cffex_panel_akshare.csv")
    parser.add_argument("--metadata-output", default=None)
    parser.add_argument(
        "--with-contract-info",
        action="store_true",
        help="Query CFFEX expiry metadata for every observed date; slower but more exact",
    )
    args = parser.parse_args()

    panel = AkshareProvider().build_contract_panel(
        args.families,
        args.start,
        args.end,
        include_contract_info=args.with_contract_info,
    )
    if panel.empty:
        raise SystemExit("No data returned. Check AkShare connectivity, dates, and exchange availability.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output, index=False)
    print(f"Saved {len(panel):,} rows to {output}")
    metadata_output = Path(args.metadata_output) if args.metadata_output else output.with_suffix(output.suffix + ".json")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    observed_dates = pd.to_datetime(panel["trade_date"])
    metadata_output.write_text(
        json.dumps(
            {
                "provider": "akshare",
                "families": args.families,
                "start_date": args.start,
                "end_date": args.end,
                "rows": len(panel),
                "observed_start_date": observed_dates.min().strftime("%Y-%m-%d"),
                "observed_end_date": observed_dates.max().strftime("%Y-%m-%d"),
                "sha256": digest,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved metadata to {metadata_output}")


if __name__ == "__main__":
    main()
