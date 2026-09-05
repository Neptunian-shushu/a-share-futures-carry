"""Download a normalized CFFEX equity-index futures panel from Tushare."""

from __future__ import annotations

import argparse
from pathlib import Path

from a_share_futures_carry.data.tushare_provider import TushareProvider


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="+", default=["IC", "IM"])
    parser.add_argument("--start", required=True, help="YYYYMMDD")
    parser.add_argument("--end", required=True, help="YYYYMMDD")
    parser.add_argument("--output", default="data/raw/cffex_panel.csv")
    args = parser.parse_args()

    provider = TushareProvider()
    panel = provider.build_contract_panel(args.families, args.start, args.end)
    if panel.empty:
        raise SystemExit("No data returned. Check Tushare permissions, dates, and token.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output, index=False)
    print(f"Saved {len(panel):,} rows to {output}")


if __name__ == "__main__":
    main()
