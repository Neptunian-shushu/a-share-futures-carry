# Fixed-holdout and robustness results

Snapshot: `data/raw/cffex_panel_cffex_public_ic_im.csv`  
History: 2022-07-22 to 2026-09-09  
Final holdout: 252 sessions, 2025-09-09 to 2026-09-09  
Costs: 1 bp per turnover event in the backtest; selection score additionally charges a 2 bp round-trip switch cost annualized over DTE.  
Bootstrap: 300 resamples, contiguous 20-session blocks, seed 42.

## Holdout summary

| Carry mode | Strategy | CAGR | Sharpe | Max drawdown | Bootstrap Sharpe p05 | P(Sharpe > 0) |
|---|---|---:|---:|---:|---:|---:|
| observed | IC front | 17.13% | 0.787 | -16.77% | -0.766 | 77.7% |
| observed | IC max carry | 18.84% | 0.822 | -17.56% | -0.754 | 78.3% |
| observed | IM front | 14.71% | 0.705 | -18.62% | -0.740 | 76.3% |
| observed | dynamic max carry | 12.24% | 0.605 | -18.11% | -0.904 | 74.7% |
| observed | carry allocation | 4.31% | 0.517 | -6.95% | -1.209 | 63.7% |
| observed | carry + volatility target | 1.92% | 0.814 | -2.62% | -1.198 | 70.0% |

The fair-value mode produced the same contract path on this snapshot because the
constant funding/dividend assumptions changed signal levels but not their ranking.
That is a useful diagnostic: the fair-value model is now available, but it should
not be interpreted as a validated dividend forecast until row-level forward inputs
are supplied.

## Interpretation

- The fixed holdout is positive for every always-invested futures variant, but the
  block-bootstrap lower Sharpe quantile is negative for all of them. The result is
  therefore promising but statistically fragile.
- Volatility targeting materially reduced drawdown and raised the point-estimate
  Sharpe, at the cost of much lower CAGR and average exposure.
- The regime split shows that most of the holdout gains came from the up-market
  regime; down-market performance remained negative. This confirms the strategy is
  long equity beta plus basis exposure, not market-neutral arbitrage.
- Stress tests across 0.5x/1x/2x costs and 8%/12%/20% margin rates produced no
  margin calls in the tested snapshot. Cost assumptions still matter most for
  max-carry variants because they switch much more often.

Machine-readable outputs are written to `outputs/real_robustness/`:

- `fixed_holdout_summary.csv`
- `regime_summary.csv`
- `implementation_stress.csv`

## Strict three-way walk-forward

The stricter walk-forward run used 252 training sessions, 63 validation sessions,
63 test sessions, a 63-session step, and thresholds `0.3/0.5/0.7`. It produced 10
out-of-sample windows: 6 had positive returns, mean test return was 1.91%, median
test return was 2.37%, mean test Sharpe was 0.658, and the worst test return was
-5.17%. Each window ran the full history before slicing the evaluation period, so
the portfolio did not reset to zero at the test boundary and the one-session
execution lag remained active.

The machine-readable result is `outputs/real_walk_forward_strict_summary.csv`.
