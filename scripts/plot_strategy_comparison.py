"""Plot focused cumulative-return comparisons from a generated research report."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LABELS = {
    "dynamic_IC_IM_front_switch": "IC/IM近月动态切换",
    "dynamic_IC_IM_beta_target": "IC/IM Beta目标仓位",
    "IC_front": "IC近月基线",
    "510500_total_return": "510500 ETF（含分红再投资）",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--curves",
        default="outputs/full_research_report/equity_curves.csv",
        help="Generated equity_curves.csv from run_real_backtest.py",
    )
    parser.add_argument(
        "--output",
        default="outputs/full_research_report/strategy_vs_benchmark.png",
    )
    args = parser.parse_args()

    curves = pd.read_csv(args.curves)
    required = {"strategy", "trade_date", "wealth"}
    missing = required.difference(curves.columns)
    if missing:
        raise ValueError(f"Curves file missing columns: {sorted(missing)}")
    wanted = list(LABELS)
    curves = curves[curves["strategy"].isin(wanted)].copy()
    if curves.empty:
        raise ValueError("No requested strategy curves found")
    curves["trade_date"] = pd.to_datetime(curves["trade_date"])
    curves["wealth"] = pd.to_numeric(curves["wealth"], errors="coerce")
    curves = curves.dropna(subset=["trade_date", "wealth"]).sort_values("trade_date")

    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(12, 6.5))
    colors = {
        "dynamic_IC_IM_front_switch": "#c23b22",
        "dynamic_IC_IM_beta_regime": "#9467bd",
        "dynamic_IC_IM_beta_target": "#8c564b",
        "dynamic_IC_IM_net_carry_front_switch": "#ff7f0e",
        "IC_front": "#1f77b4",
        "IM_front": "#2ca02c",
        "510500_total_return": "#7f7f7f",
    }
    for name in wanted:
        group = curves[curves["strategy"].eq(name)]
        if group.empty:
            continue
        first = float(group["wealth"].iloc[0])
        normalized = group["wealth"] / first if first else group["wealth"]
        line_style = "--" if name == "510500_total_return" else "-"
        ax.plot(
            group["trade_date"], normalized,
            label=LABELS[name], color=colors[name], linewidth=2.0 if name == "dynamic_IC_IM_front_switch" else 1.5,
            linestyle=line_style,
        )

    holdout_start = pd.Timestamp("2025-08-27")
    ax.axvline(holdout_start, color="#555555", linewidth=1.0, linestyle=":")
    ax.text(
        holdout_start,
        0.03,
        "固定留出集开始",
        transform=ax.get_xaxis_transform(),
        rotation=90,
        va="bottom",
        ha="right",
        color="#555555",
    )
    ax.set_title("IC/IM策略、Beta风险控制与510500 ETF总收益基准")
    ax.set_ylabel("累计净值（起点=1）")
    ax.set_xlabel("交易日期")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Saved strategy comparison chart to {output}")


if __name__ == "__main__":
    main()
