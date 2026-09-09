"""Compare two normalized data-source snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a_share_futures_carry.data.csv_provider import load_contract_panel_csv
from a_share_futures_carry.data.reconcile import reconcile_panels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--differences-output", default=None)
    args = parser.parse_args()

    left = load_contract_panel_csv(args.left)
    right = load_contract_panel_csv(args.right)
    summary, differences = reconcile_panels(left, right, tolerance=args.tolerance)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.differences_output:
        output = Path(args.differences_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        differences.to_csv(output, index=False)
        print(f"Saved row-level differences to {output}")


if __name__ == "__main__":
    main()
