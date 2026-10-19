"""Two-leg beta-neutral futures backtest utilities."""

from __future__ import annotations

from math import floor

import numpy as np
import pandas as pd


def _price(row: pd.Series | None, preferred: str, fallback: str = "futures_close") -> float | None:
    if row is None:
        return None
    for column in (preferred, fallback):
        if column in row.index and pd.notna(row[column]):
            value = float(row[column])
            if value > 0:
                return value
    return None


def _lookup(indexed: pd.DataFrame, date: pd.Timestamp, contract: str | None) -> pd.Series | None:
    if contract is None or (date, contract) not in indexed.index:
        return None
    row = indexed.loc[(date, contract)]
    return row.iloc[0] if isinstance(row, pd.DataFrame) else row


def _rolling_beta(
    long_selected: pd.DataFrame,
    hedge_selected: pd.DataFrame,
    *,
    lookback: int,
    min_periods: int,
    fallback_beta: float,
    max_beta: float,
) -> pd.Series:
    """Estimate long-leg beta to the hedge leg using signal-date prices only."""
    long_prices = (
        long_selected.sort_values("trade_date")
        .drop_duplicates("trade_date")
        .set_index("trade_date")["futures_close"]
        .astype(float)
    )
    hedge_prices = (
        hedge_selected.sort_values("trade_date")
        .drop_duplicates("trade_date")
        .set_index("trade_date")["futures_close"]
        .astype(float)
    )
    frame = pd.concat(
        [long_prices.rename("long_price"), hedge_prices.rename("hedge_price")], axis=1
    ).sort_index()
    returns = frame.pct_change()
    covariance = returns["long_price"].rolling(lookback, min_periods=min_periods).cov(
        returns["hedge_price"], ddof=0
    )
    variance = returns["hedge_price"].rolling(lookback, min_periods=min_periods).var(ddof=0)
    beta = covariance / variance.where(variance >= 1e-12)
    beta = beta.replace([np.inf, -np.inf], np.nan).abs()
    return beta.clip(lower=0.0, upper=max_beta).fillna(fallback_beta)


def _apply_participation_cap(
    desired: float,
    previous: float,
    previous_contract: str | None,
    target_contract: str | None,
    target_row: pd.Series | None,
    max_participation_rate: float | None,
    integer_contracts: bool,
) -> tuple[float, bool, bool]:
    """Limit opening or same-contract resizing by the target day's volume."""
    if target_contract is None or target_row is None or max_participation_rate is None:
        return desired, False, False
    volume = target_row.get("vol")
    if pd.isna(volume):
        return desired, False, True
    capacity = max(float(volume), 0.0) * max_participation_rate
    if previous_contract == target_contract:
        clipped = previous + max(-capacity, min(capacity, desired - previous))
    else:
        clipped = min(desired, capacity)
    if integer_contracts:
        clipped = float(floor(clipped))
    clipped = max(clipped, 0.0)
    return clipped, abs(clipped - desired) > 1e-12, False


def _leg_target_contracts(
    target_notional: float,
    price: float | None,
    multiplier: float,
    integer_contracts: bool,
) -> float:
    if price is None or price <= 0 or multiplier <= 0:
        return 0.0
    contracts = target_notional / (price * multiplier)
    return float(floor(contracts)) if integer_contracts else max(contracts, 0.0)


def backtest_beta_neutral_pair(
    long_selected: pd.DataFrame,
    hedge_selected: pd.DataFrame,
    market_data: pd.DataFrame,
    *,
    initial_nav: float = 1_000_000.0,
    long_notional_to_nav: float = 0.5,
    gross_notional_to_nav: float = 1.0,
    collateral_yield_annual: float = 0.015,
    transaction_cost_bps: float = 1.0,
    signal_lag_sessions: int = 1,
    beta_lookback: int = 60,
    beta_min_periods: int = 20,
    fallback_beta: float = 1.0,
    max_beta: float = 2.0,
    margin_rate: float = 0.12,
    margin_buffer: float = 1.20,
    integer_contracts: bool = True,
    execution_price_col: str = "settle",
    mark_price_col: str = "settle",
    max_participation_rate: float | None = None,
    spread_bps_column: str | None = None,
    default_spread_bps: float = 0.0,
    require_spread_data: bool = False,
) -> pd.DataFrame:
    """Backtest a long futures leg against a beta-sized short hedge leg.

    The long leg remains the carry strategy.  The hedge ratio is calculated
    from rolling signal-date futures returns and is lagged by the same number
    of sessions as execution.  Long plus short gross notional is capped before
    integer sizing, so beta-neutralization does not silently add leverage.
    """
    if initial_nav <= 0:
        raise ValueError("initial_nav must be positive")
    if not 0 <= long_notional_to_nav <= gross_notional_to_nav:
        raise ValueError("long_notional_to_nav must be within gross_notional_to_nav")
    if gross_notional_to_nav < 0:
        raise ValueError("gross_notional_to_nav must be non-negative")
    if signal_lag_sessions < 0:
        raise ValueError("signal_lag_sessions must be non-negative")
    if beta_lookback < 2 or beta_min_periods < 2 or beta_min_periods > beta_lookback:
        raise ValueError("Require 2 <= beta_min_periods <= beta_lookback")
    if fallback_beta <= 0 or max_beta <= 0:
        raise ValueError("fallback_beta and max_beta must be positive")
    if margin_rate < 0 or margin_buffer <= 0:
        raise ValueError("margin_rate must be non-negative and margin_buffer must be positive")
    if transaction_cost_bps < 0 or default_spread_bps < 0:
        raise ValueError("transaction costs and spreads must be non-negative")
    if max_participation_rate is not None and not 0 < max_participation_rate <= 1:
        raise ValueError("max_participation_rate must be in (0, 1]")
    if long_selected.empty or hedge_selected.empty or market_data.empty:
        return pd.DataFrame()

    for name, frame in (("long_selected", long_selected), ("hedge_selected", hedge_selected)):
        required = {"trade_date", "contract", "futures_close"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} missing required columns: {sorted(missing)}")
        dates = pd.to_datetime(frame["trade_date"])
        if dates.duplicated().any():
            raise ValueError(f"{name} must contain at most one row per trade_date")
    required_market = {"trade_date", "contract", "futures_close", "multiplier"}
    missing_market = required_market.difference(market_data.columns)
    if missing_market:
        raise ValueError(f"market_data missing required columns: {sorted(missing_market)}")

    market = market_data.copy()
    market["trade_date"] = pd.to_datetime(market["trade_date"])
    if market[["trade_date", "contract"]].duplicated().any():
        raise ValueError("market_data contains duplicate (trade_date, contract) rows")
    market = market.sort_values(["trade_date", "contract"]).reset_index(drop=True)
    market_index = market.set_index(["trade_date", "contract"])
    calendar = pd.DatetimeIndex(sorted(market["trade_date"].unique()))

    long_signal = long_selected.copy()
    hedge_signal = hedge_selected.copy()
    long_signal["trade_date"] = pd.to_datetime(long_signal["trade_date"])
    hedge_signal["trade_date"] = pd.to_datetime(hedge_signal["trade_date"])
    long_index = long_signal.set_index("trade_date").sort_index()
    hedge_index = hedge_signal.set_index("trade_date").sort_index()
    long_contracts = long_index["contract"].reindex(calendar).ffill().shift(signal_lag_sessions)
    hedge_contracts = hedge_index["contract"].reindex(calendar).ffill().shift(signal_lag_sessions)
    raw_long_dates = pd.Series(long_index.index, index=long_index.index).reindex(calendar).ffill()
    raw_hedge_dates = pd.Series(hedge_index.index, index=hedge_index.index).reindex(calendar).ffill()
    long_signal_dates = raw_long_dates.shift(signal_lag_sessions).where(long_contracts.notna())
    hedge_signal_dates = raw_hedge_dates.shift(signal_lag_sessions).where(hedge_contracts.notna())
    beta_signal = _rolling_beta(
        long_signal,
        hedge_signal,
        lookback=beta_lookback,
        min_periods=beta_min_periods,
        fallback_beta=fallback_beta,
        max_beta=max_beta,
    )
    hedge_ratio = beta_signal.reindex(calendar).ffill().shift(signal_lag_sessions).fillna(fallback_beta)

    nav = float(initial_nav)
    previous_date: pd.Timestamp | None = None
    previous_long_contract: str | None = None
    previous_hedge_contract: str | None = None
    previous_long_price: float | None = None
    previous_hedge_price: float | None = None
    previous_long_multiplier = 0.0
    previous_hedge_multiplier = 0.0
    previous_long_contracts = 0.0
    previous_hedge_contracts = 0.0
    previous_long_spot: float | None = None
    previous_hedge_spot: float | None = None
    previous_margin = 0.0
    rows: list[dict[str, object]] = []

    for date in calendar:
        long_contract_value = long_contracts.loc[date]
        hedge_contract_value = hedge_contracts.loc[date]
        long_contract = None if pd.isna(long_contract_value) else str(long_contract_value)
        hedge_contract = None if pd.isna(hedge_contract_value) else str(hedge_contract_value)
        long_target_row = _lookup(market_index, date, long_contract)
        hedge_target_row = _lookup(market_index, date, hedge_contract)
        long_held_row = _lookup(market_index, date, previous_long_contract)
        hedge_held_row = _lookup(market_index, date, previous_hedge_contract)

        long_mark = _price(long_held_row, mark_price_col)
        hedge_mark = _price(hedge_held_row, mark_price_col)
        long_missing_mark = previous_long_contract is not None and long_mark is None
        hedge_missing_mark = previous_hedge_contract is not None and hedge_mark is None
        if long_missing_mark:
            long_mark = previous_long_price
        if hedge_missing_mark:
            hedge_mark = previous_hedge_price

        long_pnl = (
            previous_long_contracts * previous_long_multiplier * (long_mark - previous_long_price)
            if long_mark is not None and previous_long_price is not None
            else 0.0
        )
        hedge_pnl = (
            -previous_hedge_contracts * previous_hedge_multiplier * (hedge_mark - previous_hedge_price)
            if hedge_mark is not None and previous_hedge_price is not None
            else 0.0
        )

        def _spot(row: pd.Series | None) -> float | None:
            if row is None or "spot_close" not in row.index or pd.isna(row["spot_close"]):
                return None
            value = float(row["spot_close"])
            return value if value > 0 else None

        long_spot = _spot(long_held_row) or _spot(long_target_row)
        hedge_spot = _spot(hedge_held_row) or _spot(hedge_target_row)
        long_spot_pnl = (
            previous_long_contracts * previous_long_multiplier * (long_spot - previous_long_spot)
            if long_spot is not None and previous_long_spot is not None
            else 0.0
        )
        hedge_spot_pnl = (
            -previous_hedge_contracts * previous_hedge_multiplier * (hedge_spot - previous_hedge_spot)
            if hedge_spot is not None and previous_hedge_spot is not None
            else 0.0
        )
        long_basis_pnl = long_pnl - long_spot_pnl
        hedge_basis_pnl = hedge_pnl - hedge_spot_pnl

        days = 0 if previous_date is None else max((date - previous_date).days, 1)
        collateral_base = max(nav - previous_margin, 0.0)
        collateral_factor = (1.0 + collateral_yield_annual) ** (days / 365.0) - 1.0
        collateral_pnl = collateral_base * collateral_factor
        nav_before_trade = nav + long_pnl + hedge_pnl + collateral_pnl

        ratio = float(max(hedge_ratio.loc[date], 0.0))
        long_target_price = _price(long_target_row, execution_price_col)
        hedge_target_price = _price(hedge_target_row, execution_price_col)
        long_multiplier = float(long_target_row["multiplier"]) if long_target_row is not None else 0.0
        hedge_multiplier = float(hedge_target_row["multiplier"]) if hedge_target_row is not None else 0.0
        long_target_notional = max(nav_before_trade, 0.0) * long_notional_to_nav
        hedge_target_notional = long_target_notional * ratio
        gross_budget = max(nav_before_trade, 0.0) * gross_notional_to_nav
        if long_target_notional + hedge_target_notional > gross_budget and long_target_notional > 0:
            scale = gross_budget / (long_target_notional + hedge_target_notional)
            long_target_notional *= scale
            hedge_target_notional *= scale

        margin_per_notional = margin_rate * margin_buffer
        if margin_per_notional > 0:
            target_margin = (long_target_notional + hedge_target_notional) * margin_per_notional
            if target_margin > max(nav_before_trade, 0.0):
                scale = max(nav_before_trade, 0.0) / target_margin
                long_target_notional *= scale
                hedge_target_notional *= scale

        desired_long = _leg_target_contracts(long_target_notional, long_target_price, long_multiplier, integer_contracts)
        desired_hedge = _leg_target_contracts(hedge_target_notional, hedge_target_price, hedge_multiplier, integer_contracts)
        new_long, long_liquidity, long_liquidity_missing = _apply_participation_cap(
            desired_long,
            previous_long_contracts,
            previous_long_contract,
            long_contract,
            long_target_row,
            max_participation_rate,
            integer_contracts,
        )
        new_hedge, hedge_liquidity, hedge_liquidity_missing = _apply_participation_cap(
            desired_hedge,
            previous_hedge_contracts,
            previous_hedge_contract,
            hedge_contract,
            hedge_target_row,
            max_participation_rate,
            integer_contracts,
        )
        long_position_contract = long_contract if new_long > 0 else None
        hedge_position_contract = hedge_contract if new_hedge > 0 else None

        def _turnover(
            previous_contract: str | None,
            position_contract: str | None,
            previous_count: float,
            new_count: float,
            previous_price: float | None,
            target_price: float | None,
            previous_multiplier: float,
            target_multiplier: float,
        ) -> float:
            if previous_contract is not None and position_contract == previous_contract:
                return abs(new_count - previous_count) * (previous_price or target_price or 0.0) * previous_multiplier
            held_mark = (
                long_mark
                if previous_contract is not None and previous_contract == previous_long_contract
                else hedge_mark
            )
            close_price = held_mark or previous_price or 0.0
            close_turnover = abs(previous_count * close_price * previous_multiplier)
            open_turnover = abs(new_count * (target_price or 0.0) * target_multiplier)
            return close_turnover + open_turnover

        long_turnover = _turnover(
            previous_long_contract,
            long_position_contract,
            previous_long_contracts,
            new_long,
            previous_long_price,
            long_target_price,
            previous_long_multiplier,
            long_multiplier,
        )
        hedge_turnover = _turnover(
            previous_hedge_contract,
            hedge_position_contract,
            previous_hedge_contracts,
            new_hedge,
            previous_hedge_price,
            hedge_target_price,
            previous_hedge_multiplier,
            hedge_multiplier,
        )

        spread_missing = False
        spread_values = []
        for row in (long_target_row, hedge_target_row, long_held_row, hedge_held_row):
            if row is not None and spread_bps_column and spread_bps_column in row.index and pd.notna(row[spread_bps_column]):
                spread_values.append(max(float(row[spread_bps_column]), 0.0))
        if spread_bps_column and (long_turnover > 0 or hedge_turnover > 0) and not spread_values:
            spread_missing = True
        spread_bps = float(np.mean(spread_values)) if spread_values else default_spread_bps
        if require_spread_data and spread_bps_column and spread_missing:
            raise ValueError(f"Missing required {spread_bps_column} for pair turnover on {date.date()}")
        effective_cost_bps = transaction_cost_bps + spread_bps
        trading_cost = (long_turnover + hedge_turnover) * effective_cost_bps / 10_000.0
        nav = nav_before_trade - trading_cost
        long_notional = abs(new_long * (long_target_price or 0.0) * long_multiplier)
        hedge_notional = abs(new_hedge * (hedge_target_price or 0.0) * hedge_multiplier)
        margin_used = (long_notional + hedge_notional) * margin_per_notional
        free_cash = nav - margin_used

        rows.append(
            {
                "trade_date": date,
                "long_signal_date": long_signal_dates.loc[date],
                "hedge_signal_date": hedge_signal_dates.loc[date],
                "long_contract": long_position_contract,
                "hedge_contract": hedge_position_contract,
                "long_contracts": new_long,
                "hedge_contracts": new_hedge,
                "long_notional": long_notional,
                "hedge_notional": hedge_notional,
                "gross_notional": long_notional + hedge_notional,
                "hedge_ratio": ratio,
                "estimated_net_beta_notional": long_notional * ratio - hedge_notional,
                "long_futures_pnl": long_pnl,
                "hedge_futures_pnl": hedge_pnl,
                "futures_pnl": long_pnl + hedge_pnl,
                "long_spot_beta_pnl": long_spot_pnl,
                "hedge_spot_beta_pnl": hedge_spot_pnl,
                "spot_beta_pnl": long_spot_pnl + hedge_spot_pnl,
                "long_basis_pnl": long_basis_pnl,
                "hedge_basis_pnl": hedge_basis_pnl,
                "basis_pnl": long_basis_pnl + hedge_basis_pnl,
                "collateral_pnl": collateral_pnl,
                "trading_cost": trading_cost,
                "turnover_notional": long_turnover + hedge_turnover,
                "long_turnover_notional": long_turnover,
                "hedge_turnover_notional": hedge_turnover,
                "margin_used": margin_used,
                "free_cash": free_cash,
                "margin_call": free_cash < -1e-9,
                "liquidity_constrained": long_liquidity or hedge_liquidity,
                "liquidity_data_missing": long_liquidity_missing or hedge_liquidity_missing,
                "spread_data_missing": spread_missing,
                "spread_bps": spread_bps,
                "effective_cost_bps": effective_cost_bps,
                "long_roll_event": previous_long_contract is not None and long_position_contract != previous_long_contract,
                "hedge_roll_event": previous_hedge_contract is not None and hedge_position_contract != previous_hedge_contract,
                "roll_event": (
                    (previous_long_contract is not None and long_position_contract != previous_long_contract)
                    or (previous_hedge_contract is not None and hedge_position_contract != previous_hedge_contract)
                ),
                "long_missing_mark": long_missing_mark,
                "hedge_missing_mark": hedge_missing_mark,
                "pnl": long_pnl + hedge_pnl + collateral_pnl - trading_cost,
                "nav": nav,
                "return": 0.0,
            }
        )

        previous_date = date
        previous_long_contract = long_position_contract
        previous_hedge_contract = hedge_position_contract
        previous_long_price = long_target_price if long_position_contract is not None else None
        previous_hedge_price = hedge_target_price if hedge_position_contract is not None else None
        previous_long_multiplier = long_multiplier if long_position_contract is not None else 0.0
        previous_hedge_multiplier = hedge_multiplier if hedge_position_contract is not None else 0.0
        previous_long_contracts = new_long
        previous_hedge_contracts = new_hedge
        previous_long_spot = long_spot if long_position_contract is not None else None
        previous_hedge_spot = hedge_spot if hedge_position_contract is not None else None
        previous_margin = margin_used

    result = pd.DataFrame(rows)
    result["return"] = result["nav"].pct_change().fillna(result["pnl"] / initial_nav)
    result["return"] = result["return"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    result["exposure_to_nav"] = result["gross_notional"] / result["nav"].replace(0, np.nan)
    result["max_exposure"] = result["exposure_to_nav"].cummax()
    return result
