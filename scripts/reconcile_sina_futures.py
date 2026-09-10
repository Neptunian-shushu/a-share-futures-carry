"""Reconcile a normalized CFFEX panel against Sina contract histories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a_share_futures_carry.data.csv_provider import load_contract_panel_csv


def reconcile_contract(panel: pd.DataFrame, contract: str, client: object) -> dict[str, object]:
    raw = client.futures_zh_daily_sina(symbol=contract)
    if raw is None or raw.empty:
        return {"contract": contract, "status": "no_sina_data"}
    sina = raw.rename(
        columns={"date": "trade_date", "close": "sina_close", "volume": "sina_vol", "hold": "sina_oi"}
    )
    required = {"trade_date", "sina_close", "sina_vol", "sina_oi"}
    missing = required.difference(sina.columns)
    if missing:
        return {"contract": contract, "status": "missing_sina_columns", "missing": sorted(missing)}
    sina["trade_date"] = pd.to_datetime(sina["trade_date"], errors="coerce")
    official = panel[panel["contract"] == contract][
        ["trade_date", "futures_close", "vol", "oi", "expiry_date"]
    ].copy()
    official["trade_date"] = pd.to_datetime(official["trade_date"], errors="coerce")
    official["expiry_date"] = pd.to_datetime(official["expiry_date"], errors="coerce")
    merged = official.merge(sina[list(required)], on="trade_date", how="outer", indicator=True)
    fields = {
        "futures_close": "sina_close",
        "vol": "sina_vol",
        "oi": "sina_oi",
    }
    result: dict[str, object] = {
        "contract": contract,
        "status": "ok",
        "official_rows": int(len(official)),
        "sina_rows": int(len(sina)),
        "matched_rows": int((merged["_merge"] == "both").sum()),
        "official_only_rows": int((merged["_merge"] == "left_only").sum()),
        "sina_only_rows": int((merged["_merge"] == "right_only").sum()),
    }
    for left, right in fields.items():
        pair = merged[[left, right]].dropna()
        diff = (pair[left] - pair[right]).abs()
        result[f"{left}_mean_abs_diff"] = float(diff.mean()) if not diff.empty else None
        result[f"{left}_max_abs_diff"] = float(diff.max()) if not diff.empty else None
        result[f"{left}_exact_ratio"] = float((diff <= 1e-6).mean()) if not diff.empty else None
        non_expiry = merged[merged["trade_date"] < merged["expiry_date"]][[left, right]].dropna()
        non_expiry_diff = (non_expiry[left] - non_expiry[right]).abs()
        result[f"{left}_non_expiry_exact_ratio"] = (
            float((non_expiry_diff <= 1e-6).mean()) if not non_expiry_diff.empty else None
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--contracts", nargs="+", default=["IC2402", "IM2501"])
    parser.add_argument("--output", default="outputs/sina_reconciliation.json")
    args = parser.parse_args()

    try:
        import akshare as ak
    except ImportError as exc:
        raise SystemExit("Install AkShare with: pip install -e '.[akshare]'") from exc

    panel = load_contract_panel_csv(args.data)
    available = set(panel["contract"])
    missing = sorted(set(args.contracts).difference(available))
    if missing:
        raise SystemExit(f"Contracts are absent from panel: {missing}")
    results = [reconcile_contract(panel, contract, ak) for contract in args.contracts]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nSaved reconciliation to {output}")


if __name__ == "__main__":
    main()
