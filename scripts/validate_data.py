"""Validate a normalized futures contract panel before research use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a_share_futures_carry.data.schema import data_quality_report, prepare_contract_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    raw = pd.read_csv(args.data)
    report = data_quality_report(raw)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if not report["ok"]:
        raise SystemExit(1)
    prepare_contract_data(raw)


if __name__ == "__main__":
    main()
