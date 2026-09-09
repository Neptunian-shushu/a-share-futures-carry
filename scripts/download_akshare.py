"""Download a normalized CFFEX panel through the free AkShare provider."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a_share_futures_carry.data.akshare_provider import AkshareProvider


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="+", default=["IC", "IM"])
    parser.add_argument("--start", required=True, help="YYYYMMDD")
    parser.add_argument("--end", required=True, help="YYYYMMDD")
    parser.add_argument("--output", default="data/raw/cffex_panel_akshare.csv")
    args = parser.parse_args()

    panel = AkshareProvider().build_contract_panel(args.families, args.start, args.end)
    if panel.empty:
        raise SystemExit("No data returned. Check AkShare connectivity, dates, and exchange availability.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output, index=False)
    print(f"Saved {len(panel):,} rows to {output}")


if __name__ == "__main__":
    main()
