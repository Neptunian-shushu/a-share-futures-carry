"""Plot focused cumulative-return comparisons from a generated research report."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LABELS = {
    "IF_front": "IF近月基线",
    "IH_front": "IH近月基线",
    "dynamic_IC_IM_front_switch": "IC/IM近月动态切换",
    "dynamic_IC_IM_beta_target": "IC/IM Beta目标仓位",
    "IC_front": "IC近月基线",
    "IM_front": "IM近月基线",
    "dynamic_IC_IM_max_carry": "IC/IM动态最大Carry",
    "dynamic_IC_IM_carry_vol_target": "动态Carry+波动率目标",
    "510500_total_return": "510500 ETF（含分红再投资）",
}

DEFAULT_STRATEGIES = [
    "dynamic_IC_IM_front_switch",
    "dynamic_IC_IM_beta_target",
    "IC_front",
    "510500_total_return",
]

MIN_START_DATES = {
    "IF_front": pd.Timestamp("2015-01-05"),
    "IH_front": pd.Timestamp("2015-04-16"),
    "IC_front": pd.Timestamp("2015-04-16"),
    "IM_front": pd.Timestamp("2022-07-22"),
    "dynamic_IC_IM_front_switch": pd.Timestamp("2015-04-16"),
    "dynamic_IC_IM_beta_target": pd.Timestamp("2015-04-16"),
    "dynamic_IC_IM_max_carry": pd.Timestamp("2015-04-16"),
    "dynamic_IC_IM_carry_vol_target": pd.Timestamp("2015-04-16"),
    "510500_total_return": pd.Timestamp("2022-07-22"),
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
    parser.add_argument(
        "--strategies",
        nargs="*",
        default=DEFAULT_STRATEGIES,
        help="Strategy keys to include; defaults to the focused IC/IM comparison",
    )
    args = parser.parse_args()

    curves = pd.read_csv(args.curves)
    required = {"strategy", "trade_date", "wealth"}
    missing = required.difference(curves.columns)
    if missing:
        raise ValueError(f"Curves file missing columns: {sorted(missing)}")
    wanted = args.strategies
    unknown = sorted(set(wanted).difference(LABELS))
    if unknown:
        raise ValueError(f"Unknown strategy keys: {unknown}")
    curves = curves[curves["strategy"].isin(wanted)].copy()
    if curves.empty:
        raise ValueError("No requested strategy curves found")
    curves["trade_date"] = pd.to_datetime(curves["trade_date"])
    curves["wealth"] = pd.to_numeric(curves["wealth"], errors="coerce")
    curves = curves.dropna(subset=["trade_date", "wealth"]).sort_values("trade_date")
    curves = curves[
        curves.apply(
            lambda row: row["trade_date"] >= MIN_START_DATES.get(
                row["strategy"], pd.Timestamp.min
            ),
            axis=1,
        )
    ]

    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(12, 6.5))
    colors = {
        "IF_front": "#17becf",
        "IH_front": "#bcbd22",
        "dynamic_IC_IM_front_switch": "#c23b22",
        "dynamic_IC_IM_beta_regime": "#9467bd",
        "dynamic_IC_IM_beta_target": "#8c564b",
        "dynamic_IC_IM_net_carry_front_switch": "#ff7f0e",
        "IC_front": "#1f77b4",
        "IM_front": "#2ca02c",
        "dynamic_IC_IM_max_carry": "#9467bd",
        "dynamic_IC_IM_carry_vol_target": "#e377c2",
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
    ax.set_title("策略累计净值与510500 ETF总收益基准")
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
