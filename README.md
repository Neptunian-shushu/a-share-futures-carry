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
4. Maximum annualized carry contract
5. Dynamic IC/IM carry selection

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
  data/                    Data schemas/loaders
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
pip install -e .
pytest
python scripts/run_backtest.py --config configs/strategy.yaml
```

The initial runner uses synthetic data so that the repository is executable before a commercial/official historical data source is connected.

## Data needed for a production backtest

Daily contract-level data should include trade date, contract code, underlying family, futures close/settlement, spot index close, expiry date, volume/open interest, and preferably transaction-cost inputs. Funding/collateral rates and expected index dividends are needed for fair-value/excess-carry analysis.

Recommended research history:

- IF / IH: full available history
- IC: from 2015
- IM: from 2022

## Risk controls

The default research design caps futures notional exposure at 1.0x NAV. Margin availability is **not** treated as permission to lever the equity beta. Production implementation should additionally model variation margin, margin buffers, liquidity, limit moves, roll execution, commissions and slippage.

## Roadmap

- [x] Repository skeleton
- [x] Carry calculations
- [x] Basic max-carry selector
- [x] Minimal futures PnL engine
- [ ] Connect historical CFFEX/index data
- [ ] Build robust contract calendar and roll rules
- [ ] Add dividend and funding fair-value model
- [ ] Compare IC/IM/front/second/max-carry strategies
- [ ] Add dynamic carry percentile/z-score allocation
- [ ] Produce research report and charts

## Disclaimer

For quantitative research and education only. Futures are leveraged derivatives and can generate losses larger and faster than an unlevered cash-equity position if exposure is not controlled.
