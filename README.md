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
pip install -e '.[dev]'
pytest
python scripts/run_backtest.py --config configs/strategy.yaml
```

The synthetic runner keeps the project executable even when no market-data vendor is connected.

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

Recommended research history:

- IF / IH: full available history
- IC: from 2015
- IM: from 2022

## Risk controls

The default research design caps futures notional exposure at 1.0x NAV. Margin availability is **not** treated as permission to lever the equity beta. Production implementation should additionally model variation margin, margin buffers, liquidity, limit moves, roll execution, commissions and slippage.

## Current limitations

The current backtest engine is deliberately minimal. It marks futures close-to-close and adds collateral yield, but it does not yet model daily settlement cash flows, exchange margin schedules, exact CFFEX expiry rules, roll slippage by liquidity, dividend fair value, or realistic integer contract sizing. These are next-stage research items rather than hidden assumptions.

## Roadmap

- [x] Repository skeleton
- [x] Carry calculations
- [x] Front/second/max-carry selectors
- [x] Minimal futures PnL engine
- [x] Tushare historical-data provider
- [x] CSV vendor fallback
- [x] Real-data strategy comparison runner
- [ ] Add AkShare/free-data provider
- [ ] Build robust CFFEX contract calendar and roll rules
- [ ] Add dividend and funding fair-value model
- [ ] Add integer sizing and variation-margin accounting
- [ ] Add spot/ETF buy-and-hold benchmark
- [ ] Add dynamic carry percentile/z-score allocation
- [ ] Produce research report and charts

## Disclaimer

For quantitative research and education only. Futures are leveraged derivatives and can generate losses larger and faster than an unlevered cash-equity position if exposure is not controlled.
