"""Daily futures portfolio simulator with explicit execution and margin rules."""

from __future__ import annotations

from math import floor

import pandas as pd


def _price(row: pd.Series | None, preferred: str, fallback: str = "futures_close") -> float | None:
    """Read a price column, falling back when a vendor field is unavailable."""
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
    if isinstance(row, pd.DataFrame):
        return row.iloc[0]
    return row


def _size_contracts(
    nav: float,
    price: float,
    multiplier: float,
    max_notional_to_nav: float,
    target_weight: float,
    integer_contracts: bool,
    margin_rate: float,
    margin_buffer: float,
) -> tuple[float, bool]:
    """Size exposure and cap it if the configured margin budget is binding."""
    unit_notional = price * multiplier
    target_notional = max(nav, 0.0) * max_notional_to_nav * max(target_weight, 0.0)
    contracts = target_notional / unit_notional
    if integer_contracts:
        contracts = float(floor(contracts))

    margin_constrained = False
    margin_per_contract = unit_notional * margin_rate * margin_buffer
    if margin_per_contract > 0 and contracts * margin_per_contract > max(nav, 0.0):
        contracts = max(nav, 0.0) / margin_per_contract
        if integer_contracts:
            contracts = float(floor(contracts))
        margin_constrained = True
    return max(contracts, 0.0), margin_constrained


def backtest_selected_contracts(
    selected: pd.DataFrame,
    initial_nav: float = 1_000_000.0,
    max_notional_to_nav: float = 1.0,
    collateral_yield_annual: float = 0.015,
    transaction_cost_bps: float = 1.0,
    *,
    market_data: pd.DataFrame | None = None,
    signal_lag_sessions: int = 1,
    margin_rate: float = 0.12,
    margin_buffer: float = 1.20,
    integer_contracts: bool = True,
    execution_price_col: str = "settle",
    mark_price_col: str = "settle",
) -> pd.DataFrame:
    """Backtest one selected contract per signal date.

    ``selected`` is interpreted as a signal series.  When ``market_data`` is
    supplied, the signal is executed after ``signal_lag_sessions`` trading
    sessions using the target contract's execution price, while the previously
    held contract is marked from the full panel on the same day.  This prevents
    a close-based signal from receiving same-close execution and preserves PnL
    on roll days.

    The returned frame contains total NAV, daily futures PnL, collateral PnL,
    turnover, margin usage, exposure and roll diagnostics.  Fractional sizing
    remains available for theoretical studies, but integer sizing is the safe
    default for live-like research.
    """
    if initial_nav <= 0:
        raise ValueError("initial_nav must be positive")
    if max_notional_to_nav < 0:
        raise ValueError("max_notional_to_nav must be non-negative")
    if signal_lag_sessions < 0:
        raise ValueError("signal_lag_sessions must be >= 0")
    if margin_rate < 0 or margin_buffer <= 0:
        raise ValueError("margin_rate must be >= 0 and margin_buffer must be positive")
    if collateral_yield_annual <= -1:
        raise ValueError("collateral_yield_annual must be greater than -100%")
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be non-negative")
    if selected.empty:
        return pd.DataFrame()
    if "trade_date" not in selected or "contract" not in selected:
        raise ValueError("selected must contain trade_date and contract")

    signals = selected.sort_values("trade_date").copy()
    signals["trade_date"] = pd.to_datetime(signals["trade_date"])
    if signals["trade_date"].duplicated().any():
        raise ValueError("selected must contain at most one signal row per trade_date")

    if market_data is None:
        # Backwards-compatible theoretical mode: the selected panel is also
        # the mark panel, so an explicit lag cannot be reconstructed reliably.
        market = signals.copy()
        effective_lag = 0
    else:
        market = market_data.copy()
        effective_lag = signal_lag_sessions
    if market.empty:
        return pd.DataFrame()
    required_market = {"trade_date", "contract", "futures_close", "multiplier"}
    missing = required_market.difference(market.columns)
    if missing:
        raise ValueError(f"market_data missing required columns: {sorted(missing)}")
    market["trade_date"] = pd.to_datetime(market["trade_date"])
    if market[["trade_date", "contract"]].duplicated().any():
        raise ValueError("market_data contains duplicate (trade_date, contract) rows")
    market = market.sort_values(["trade_date", "contract"]).reset_index(drop=True)
    market_index = market.set_index(["trade_date", "contract"])
    calendar = pd.DatetimeIndex(sorted(market["trade_date"].unique()))

    signal_index = signals.set_index("trade_date")
    raw_signal_date = pd.Series(signal_index.index, index=signal_index.index).reindex(calendar).ffill()
    signal_contract = signal_index["contract"].reindex(calendar).ffill().shift(effective_lag)
    signal_date = raw_signal_date.shift(effective_lag).where(signal_contract.notna())
    signal_values: dict[str, pd.Series] = {}
    for column in (
        "carry_ann",
        "signal_carry",
        "selection_score",
        "carry_percentile",
        "carry_zscore",
        "target_weight",
    ):
        if column in signal_index:
            signal_values[column] = signal_index[column].reindex(calendar).ffill().shift(effective_lag)

    nav = float(initial_nav)
    previous_date: pd.Timestamp | None = None
    previous_contract: str | None = None
    previous_price: float | None = None
    previous_multiplier = 0.0
    previous_contracts = 0.0
    previous_margin = 0.0
    previous_spot_price: float | None = None
    rows: list[dict[str, object]] = []

    for date in calendar:
        target_contract_value = signal_contract.loc[date]
        target_contract = None if pd.isna(target_contract_value) else str(target_contract_value)
        target_row = _lookup(market_index, date, target_contract)

        held_row = _lookup(market_index, date, previous_contract)
        missing_mark = False
        if previous_contract is not None and previous_price is not None:
            held_price = _price(held_row, mark_price_col)
            if held_price is None:
                held_price = previous_price
                missing_mark = True
            futures_pnl = previous_contracts * previous_multiplier * (held_price - previous_price)
        else:
            held_price = None
            futures_pnl = 0.0

        spot_price = None
        for spot_row in (held_row, target_row):
            if spot_row is not None and "spot_close" in spot_row.index and pd.notna(spot_row["spot_close"]):
                value = float(spot_row["spot_close"])
                if value > 0:
                    spot_price = value
                    break
        spot_beta_pnl = (
            previous_contracts * previous_multiplier * (spot_price - previous_spot_price)
            if previous_contract is not None
            and previous_spot_price is not None
            and spot_price is not None
            else 0.0
        )
        basis_pnl = futures_pnl - spot_beta_pnl

        days = 0 if previous_date is None else max((date - previous_date).days, 1)
        collateral_base = max(nav - previous_margin, 0.0)
        collateral_factor = (1.0 + collateral_yield_annual) ** (days / 365.0) - 1.0
        collateral_pnl = collateral_base * collateral_factor
        nav_before_trade = nav + futures_pnl + collateral_pnl

        target_price = _price(target_row, execution_price_col)
        target_multiplier = float(target_row["multiplier"]) if target_row is not None else 0.0
        target_margin_rate = margin_rate
        if target_row is not None and "margin_rate" in target_row.index and pd.notna(target_row["margin_rate"]):
            target_margin_rate = float(target_row["margin_rate"])
            if target_margin_rate < 0:
                raise ValueError("market_data margin_rate must be non-negative")
        target_weight = 1.0
        if "target_weight" in signal_values:
            value = signal_values["target_weight"].loc[date]
            if pd.notna(value):
                target_weight = max(float(value), 0.0)

        margin_constrained = False
        if target_price is not None and target_multiplier > 0 and target_contract is not None:
            new_contracts, margin_constrained = _size_contracts(
                nav_before_trade,
                target_price,
                target_multiplier,
                max_notional_to_nav,
                target_weight,
                integer_contracts,
                target_margin_rate,
                margin_buffer,
            )
            target_position_contract = target_contract if new_contracts > 0 else None
        else:
            new_contracts = 0.0
            target_position_contract = None

        same_contract = previous_contract is not None and target_position_contract == previous_contract
        if same_contract:
            turnover_notional = abs(new_contracts - previous_contracts) * (held_price or target_price or 0.0) * previous_multiplier
        else:
            close_turnover = abs(previous_contracts * (held_price or previous_price or 0.0) * previous_multiplier)
            open_turnover = abs(new_contracts * (target_price or 0.0) * target_multiplier)
            turnover_notional = close_turnover + open_turnover

        trading_cost = turnover_notional * transaction_cost_bps / 10_000.0
        nav = nav_before_trade - trading_cost
        margin_used = abs(new_contracts * (target_price or 0.0) * target_multiplier) * target_margin_rate * margin_buffer
        free_cash = nav - margin_used
        margin_call = free_cash < -1e-9
        roll_event = previous_contract is not None and target_position_contract != previous_contract

        output: dict[str, object] = {
            "trade_date": date,
            "signal_date": signal_date.loc[date] if date in signal_date.index else pd.NaT,
            "contract": target_position_contract,
            "family": target_row.get("family") if target_row is not None else None,
            "carry_ann": signal_values["carry_ann"].loc[date] if "carry_ann" in signal_values else None,
            "signal_carry": signal_values["signal_carry"].loc[date] if "signal_carry" in signal_values else None,
            "selection_score": signal_values["selection_score"].loc[date] if "selection_score" in signal_values else None,
            "target_weight": target_weight,
            "margin_rate": target_margin_rate,
            "contracts": new_contracts,
            "notional": abs(new_contracts * (target_price or 0.0) * target_multiplier),
            "exposure_to_nav": abs(new_contracts * (target_price or 0.0) * target_multiplier) / nav if nav else 0.0,
            "futures_pnl": futures_pnl,
            "spot_beta_pnl": spot_beta_pnl,
            "basis_pnl": basis_pnl,
            "collateral_pnl": collateral_pnl,
            "trading_cost": trading_cost,
            "turnover_notional": turnover_notional,
            "margin_used": margin_used,
            "free_cash": free_cash,
            "margin_call": margin_call,
            "margin_constrained": margin_constrained,
            "roll_event": roll_event,
            "missing_mark": missing_mark,
            "nav": nav,
        }
        rows.append(output)

        previous_date = date
        previous_contract = target_position_contract
        previous_price = target_price if target_position_contract is not None else None
        previous_multiplier = target_multiplier if target_position_contract is not None else 0.0
        previous_contracts = new_contracts
        previous_margin = margin_used
        previous_spot_price = spot_price if target_position_contract is not None else None

    result = pd.DataFrame(rows)
    result["pnl"] = result["futures_pnl"] + result["collateral_pnl"] - result["trading_cost"]
    result["return"] = result["nav"].pct_change().fillna(result["pnl"] / initial_nav)
    result["return"] = result["return"].replace([float("inf"), -float("inf")], 0.0).fillna(0.0)
    return result
