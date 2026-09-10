# A-Share Index Futures Carry

A research-oriented Python project for studying long-only basis/carry strategies in China's equity index futures market (IF, IH, IC, IM).

## Research question

Can a long investor improve index exposure by replacing cash equity/ETF exposure with discounted stock-index futures and systematically harvesting basis convergence?

The project separates portfolio return into:

`Total return = equity beta + basis/carry + collateral yield - trading/roll costs`

The key point is that this is **not market-neutral arbitrage**. A long futures position retains equity-market beta and can suffer large drawdowns when the underlying index falls.

## Phase 1 strategies

1. ETF / spot buy-and-hold benchmark
2. Front-month futures roll
3. Second-month futures roll
4. Maximum annualized carry contract within each family
5. Dynamic IC/IM maximum-carry selection

## Core definitions

Observed annualized discount:

`carry = (spot - futures) / spot * 365 / DTE`

A later research stage will estimate fair-value basis using expected dividends and funding rates:

`excess carry = observed carry - fair-value carry`

This distinction matters because not all futures discount is alpha.

## Project structure

```text
configs/                  Strategy configuration
scripts/                  Runnable research scripts
src/a_share_futures_carry/
  data/                    Data schemas/loaders/providers
  signals/                 Basis and carry signals
  strategy/                Contract-selection rules
  backtest/                Portfolio simulation
  metrics/                 Performance statistics
tests/                     Unit tests
data/raw/                   Local raw data (gitignored)
data/processed/             Local processed data (gitignored)
outputs/                    Backtest outputs (gitignored)
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,report]'
pytest
python scripts/run_backtest.py --config configs/strategy.yaml
```

The synthetic runner keeps the project executable even when no market-data vendor is connected.
It creates fixed monthly contracts with stable IDs, so the smoke backtest exercises
continuous holding and real roll events. Add `--data-output outputs/synthetic_panel.csv`
to export the generated panel for testing the real-data runner.

## Real historical data with Tushare

Tushare is the first implemented provider. It uses:

- `fut_basic` for CFFEX contract metadata
- `fut_daily` for individual futures contracts
- `index_daily` for the corresponding cash indices

Install the optional dependency and set your token:

```bash
pip install -e '.[tushare]'
export TUSHARE_TOKEN='YOUR_TOKEN'
```

Download a normalized IC/IM contract panel:

```bash
python scripts/download_tushare.py \
  --families IC IM \
  --start 20220722 \
  --end 20260901 \
  --output data/raw/cffex_panel.csv
```

Then compare front-month, second-month, family max-carry and dynamic IC/IM strategies:

```bash
python scripts/run_real_backtest.py \
  --data data/raw/cffex_panel.csv \
  --config configs/strategy.yaml
```

The summary is saved to `outputs/strategy_summary.csv`.

If Tushare permissions are unavailable, any vendor/export can be used through the CSV fallback as long as it contains the normalized columns below.

## Free historical data with AkShare

AkShare is an optional provider for CFFEX daily exchange data and CSI index history.
The adapter prefers the public date-range CFFEX interface, falls back to the older
one-day interface when needed, and filters IF/IH/IC/IM contracts before joining the
corresponding spot index. Install it with:

```bash
pip install -e '.[akshare]'
python scripts/download_akshare.py \
  --families IC IM \
  --start 20220722 \
  --end 20260901 \
  --output data/raw/cffex_panel_akshare.csv
```

The upstream interface is documented in the [AkShare futures documentation](https://github.com/akfamily/akshare/blob/main/docs/data/futures/futures.md).
Because free endpoints can change or rate-limit, save downloaded CSV snapshots and
the command writes a JSON sidecar containing provider, families, requested and observed
date ranges, row count, SHA-256 hash and generation time. Keep that sidecar with the CSV
snapshot.

By default expiry dates are inferred from the contract month, which is fast and
deterministic. Add `--with-contract-info` when exact exchange metadata is required; this
per-day enrichment is slower and may be rate-limited.

## No-Tushare route: official CFFEX monthly archives

For a Tushare-free futures panel, use the direct official CFFEX monthly ZIP archives.
The project downloads one archive per month, parses all daily CSV files inside it, and
uses the existing AkShare/Sina fallback only for the corresponding cash index:

```bash
pip install -e '.[akshare]'
python scripts/download_cffex_public.py \
  --families IC IM \
  --start 20220722 \
  --end 20260909 \
  --output data/raw/cffex_panel_cffex_public.csv
```

Monthly ZIP files are cached locally so an interrupted run can resume without
redownloading completed months. Expiry dates are inferred from contract months and must
still be checked against exchange metadata before production use.

## Normalized data schema

Required columns:

- `trade_date`
- `contract`
- `family` (`IF`, `IH`, `IC`, `IM`)
- `futures_close`
- `spot_close`
- `expiry_date`
- `multiplier`

Useful optional columns include `settle`, `vol`, and `oi`.
Optional research columns include `funding_rate`, `dividend_yield`, and `margin_rate`.

Validate a panel before using it:

```bash
python scripts/validate_data.py --data data/raw/cffex_panel.csv
```

Validation rejects invalid dates, duplicate `(trade_date, contract)` rows, non-positive
prices/multipliers, negative margin rates, and inconsistent expiry dates.

To reconcile two independent snapshots after normalization:

```bash
python scripts/compare_panels.py \
  --left data/raw/cffex_panel.csv \
  --right data/raw/cffex_panel_akshare.csv \
  --differences-output outputs/panel_differences.csv
```

For a field-level independent check, compare selected contracts with Sina's historical
contract series:

```bash
python scripts/reconcile_sina_futures.py \
  --data data/raw/cffex_panel_cffex_public_ic_im.csv \
  --contracts IC2402 IM2501 \
  --output outputs/sina_reconciliation.json
```

The report compares close, volume and open interest separately, including a
non-expiry-day exact-match ratio. Any expiry-day or vendor-specific discrepancy remains
visible in the JSON report for review.

The current reproducible four-family snapshot and its results are summarized in
[`reports/full_data_research_note.md`](reports/full_data_research_note.md).

Recommended research history:

- IF / IH: full available history
- IC: from 2015
- IM: from 2022

## Backtest conventions

The engine treats selector output as a signal series. With the default
`signal_lag_sessions: 1`, a close-based signal from session *t* is executed on the
next available session using `settle` when supplied, otherwise `futures_close`.
The previously held contract is marked from the full contract panel on each session,
including roll days. Exposure is sized in integer contracts, collateral yield accrues
on free cash, and margin usage is tracked separately from economic notional.

The output includes NAV, futures PnL, collateral PnL, trading costs, turnover, margin,
free cash, exposure, roll events, and missing-mark diagnostics. Passing only the
selected rows remains supported for theoretical close-to-close studies, but full-panel
input is required for accurate roll-day marking.

## Risk controls

The default research design caps futures notional exposure at 1.0x NAV. Margin availability is **not** treated as permission to lever the equity beta. The engine applies a configurable margin rate and buffer, integer sizing, liquidity fields and explicit turnover costs. Production implementation should additionally calibrate exchange-specific margin schedules, limit moves and executable bid/ask spreads.

## Research reports

The real-data runner compares front month, second month, family max-carry, dynamic IC/IM
max-carry, and historical carry-percentile allocation. It writes:

- `strategy_summary.csv`: return, risk, beta, costs, PnL decomposition, turnover and margin diagnostics
- `equity_curves.csv` and `monthly_returns.csv`
- `roll_events.csv`
- `nav_curves.png`, `drawdowns.png`, `cagr_comparison.png`, and `pnl_decomposition.png`

For example:

```bash
python scripts/run_real_backtest.py \
  --data data/raw/cffex_panel.csv \
  --config configs/strategy.yaml \
  --output outputs/strategy_summary.csv \
  --output-dir outputs/research_report
```

Each family comparison includes a spot-index buy-and-hold benchmark. An ETF can be
compared by passing a separate dated price CSV; this keeps the cash-index benchmark
separate from dividend-adjusted ETF data while reporting the ETF on the same date axis:

```bash
python scripts/run_real_backtest.py \
  --data data/raw/cffex_panel.csv \
  --config configs/strategy.yaml \
  --benchmark data/raw/etf.csv \
  --benchmark-name CSI500_ETF \
  --benchmark-price-column adj_close
```

The benchmark CSV must contain a unique date column (default `trade_date`) and strictly
positive prices (default `close`).
If it also contains a non-negative per-share cash distribution column, pass
`--benchmark-distribution-column` to calculate reinvested-distribution returns.

Generate a free 510500 benchmark snapshot with:

```bash
python scripts/download_etf_benchmark.py \
  --symbol 510500 \
  --start 20220722 \
  --end 20260909 \
  --output data/raw/etf_510500_sina.csv
```

The Sina endpoint supplies raw close prices. The sidecar records this limitation;
for a true ETF total-return comparison, replace `close` with a reinvested-distribution
or adjusted-total-return series and pass that column through `--benchmark-price-column`.

For parameter selection, run the walk-forward study. Thresholds are selected on each
training window and evaluated on the following test window:

```bash
python scripts/run_walk_forward.py \
  --data data/raw/cffex_panel.csv \
  --train-sessions 252 \
  --validation-sessions 63 \
  --test-sessions 63 \
  --thresholds 0.3 0.5 0.7
```

With validation enabled, each window uses train → validation → test ordering and
keeps the full portfolio path when reporting the test segment. This avoids selecting
parameters on the test period and avoids resetting positions at every test boundary.

Run the stronger fixed-holdout and implementation-robustness study with:

```bash
python scripts/run_robustness.py \
  --data data/raw/cffex_panel_cffex_public_ic_im.csv \
  --config configs/strategy.yaml \
  --test-sessions 252 \
  --bootstrap 500 \
  --output-dir outputs/robustness
```

This keeps the final 252 sessions untouched by parameter selection, compares
observed versus fair-value carry, reports up/down-market regimes, runs a seeded
block bootstrap, and stresses transaction costs and margin rates. It is intended
to expose fragility rather than to manufacture a single best configuration.

## Fair-value carry

Set `carry.use_fair_value_adjustment: true` to calculate theoretical futures value from
funding and dividend yields and select on `excess_carry`. Row-level
`funding_rate`/`dividend_yield` values take precedence over the scalar config defaults.
The default configuration keeps observed carry selection for backward compatibility;
research conclusions should report both versions.

## Current limitations

The engine now models daily settlement-style marking, integer sizing, margin budgets,
expiry roll windows, configurable costs and fair-value signals. Remaining production
work is calibration: verify the exact CFFEX holiday-adjusted last trading dates,
exchange-specific margin schedules, dividend forecasts, bid/ask execution and ETF
total-return data against an independent source.

## Roadmap

- [x] Repository skeleton
- [x] Carry calculations
- [x] Front/second/max-carry selectors
- [x] Minimal futures PnL engine
- [x] Tushare historical-data provider
- [x] CSV vendor fallback
- [x] Real-data strategy comparison runner
- [x] Add AkShare/free-data provider
- [x] Add data-quality validation command
- [x] Build explicit expiry roll policy
- [x] Add dividend and funding fair-value model
- [x] Add integer sizing, daily settlement-style PnL and margin accounting
- [x] Add spot-index buy-and-hold benchmark
- [x] Add dynamic carry percentile/z-score allocation
- [x] Add cost-adjusted selection, rollover hysteresis and volatility targeting
- [x] Add fixed-holdout, regime, bootstrap and margin/cost stress diagnostics
- [x] Extend the reproducible snapshot and diagnostics to IF/IH as well as IC/IM
- [x] Produce research report and charts
- [x] Add regression tests and GitHub Actions CI
- [x] Add free ETF benchmark snapshot tooling and price-semantics metadata

The next research milestone is not another selector: it is a fixed historical data
snapshot, independent-data reconciliation, walk-forward parameter selection, and
out-of-sample performance attribution by beta, basis convergence, collateral yield,
turnover and costs. The repository now provides commands for each of those steps;
the remaining work is to run them against authenticated historical snapshots and
calibrate the resulting assumptions to exchange records.

## Disclaimer

For quantitative research and education only. Futures are leveraged derivatives and can generate losses larger and faster than an unlevered cash-equity position if exposure is not controlled.
