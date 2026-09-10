"""Contract-selection rules for carry strategies."""

from __future__ import annotations

import pandas as pd


def _eligible(
    contracts: pd.DataFrame,
    eligible_families: tuple[str, ...],
    min_dte: int,
    max_dte: int,
    carry_column: str = "carry_ann",
    min_volume: float = 0.0,
    min_open_interest: float = 0.0,
) -> pd.DataFrame:
    if carry_column not in contracts:
        raise ValueError(f"Missing carry column: {carry_column}")
    mask = (
        contracts["family"].isin(eligible_families)
        & contracts["dte"].between(min_dte, max_dte)
        & contracts[carry_column].notna()
    )
    for column, threshold in (("vol", min_volume), ("oi", min_open_interest)):
        if threshold > 0:
            if column not in contracts:
                raise ValueError(f"Liquidity filter requires column: {column}")
            mask &= pd.to_numeric(contracts[column], errors="coerce").ge(threshold)
    return contracts[mask].copy()


def select_max_carry(
    contracts: pd.DataFrame,
    eligible_families: tuple[str, ...] = ("IC", "IM"),
    min_dte: int = 5,
    max_dte: int = 120,
    carry_column: str = "carry_ann",
    min_volume: float = 0.0,
    min_open_interest: float = 0.0,
) -> pd.DataFrame:
    """Select the highest annualized-carry eligible contract each trade date."""
    eligible = _eligible(
        contracts, eligible_families, min_dte, max_dte, carry_column, min_volume, min_open_interest
    )
    if eligible.empty:
        return eligible
    idx = eligible.groupby("trade_date")[carry_column].idxmax()
    return eligible.loc[idx].sort_values("trade_date").reset_index(drop=True)


def select_nth_expiry(
    contracts: pd.DataFrame,
    n: int,
    eligible_families: tuple[str, ...] = ("IC",),
    min_dte: int = 1,
    max_dte: int = 180,
    carry_column: str = "carry_ann",
    min_volume: float = 0.0,
    min_open_interest: float = 0.0,
) -> pd.DataFrame:
    """Select the nth nearest eligible expiry within each family/date.

    n=1 is the front month; n=2 is the second month. If several families are supplied,
    one row per family/date is returned so callers can backtest each family separately.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    eligible = _eligible(
        contracts, eligible_families, min_dte, max_dte, carry_column, min_volume, min_open_interest
    )
    if eligible.empty:
        return eligible

    eligible = eligible.sort_values(["trade_date", "family", "expiry_date"])
    eligible["expiry_rank"] = eligible.groupby(["trade_date", "family"]).cumcount() + 1
    return eligible[eligible["expiry_rank"] == n].drop(columns="expiry_rank").reset_index(drop=True)


def select_family_max_carry(
    contracts: pd.DataFrame,
    family: str,
    min_dte: int = 5,
    max_dte: int = 120,
    carry_column: str = "carry_ann",
    min_volume: float = 0.0,
    min_open_interest: float = 0.0,
) -> pd.DataFrame:
    """Select the best-carry contract within one index-futures family."""
    return select_max_carry(
        contracts,
        (family,),
        min_dte,
        max_dte,
        carry_column,
        min_volume,
        min_open_interest,
    )


def apply_roll_policy(
    targets: pd.DataFrame,
    contracts: pd.DataFrame,
    roll_before_expiry_days: int = 3,
    min_dte: int = 1,
    max_dte: int = 180,
    score_column: str = "signal_carry",
    min_volume: float = 0.0,
    min_open_interest: float = 0.0,
    min_score_improvement: float = 0.0,
) -> pd.DataFrame:
    """Apply a deterministic expiry-roll rule to a daily target series.

    A signal may still switch contracts before the roll window.  Once the held
    contract reaches ``roll_before_expiry_days``, it is replaced by the best
    eligible alternative available that day.  This makes expiry handling
    explicit while preserving the original carry-selection behaviour.
    """
    if roll_before_expiry_days < 0:
        raise ValueError("roll_before_expiry_days must be >= 0")
    if min_score_improvement < 0:
        raise ValueError("min_score_improvement must be non-negative")
    if targets.empty:
        return targets.copy()
    if targets["trade_date"].duplicated().any():
        raise ValueError("targets must contain at most one row per trade_date")
    if score_column not in contracts:
        score_column = "carry_ann"
    if score_column not in contracts:
        raise ValueError(f"Missing score column: {score_column}")

    market = contracts.sort_values(["trade_date", "expiry_date", "contract"]).copy()
    allowed_families = set(targets["family"].astype(str).str.upper().unique()) if "family" in targets else None
    target_map = targets.sort_values("trade_date").set_index("trade_date")
    rows: list[dict] = []
    held_contract: str | None = None

    for date, target in target_map.iterrows():
        day = market[market["trade_date"] == date]
        if allowed_families is not None and "family" in day:
            day = day[day["family"].astype(str).str.upper().isin(allowed_families)]
        eligible = day[day["dte"].between(min_dte, max_dte)].copy()
        if min_volume > 0:
            if "vol" not in eligible:
                raise ValueError("Liquidity filter requires column: vol")
            eligible = eligible[pd.to_numeric(eligible["vol"], errors="coerce") >= min_volume]
        if min_open_interest > 0:
            if "oi" not in eligible:
                raise ValueError("Liquidity filter requires column: oi")
            eligible = eligible[pd.to_numeric(eligible["oi"], errors="coerce") >= min_open_interest]
        current = day[day["contract"] == held_contract] if held_contract is not None else day.iloc[0:0]
        force_roll = bool(not current.empty and current.iloc[0]["dte"] <= roll_before_expiry_days)

        chosen = None
        reason = "initial" if held_contract is None else "signal"
        if force_roll:
            alternatives = eligible[eligible["contract"] != held_contract]
            if not alternatives.empty:
                chosen = alternatives.sort_values(
                    [score_column, "expiry_date", "contract"],
                    ascending=[False, True, True],
                ).iloc[0]
                reason = "expiry"
        if chosen is None:
            candidate_contract = target["contract"]
            candidate = day[day["contract"] == candidate_contract]
            candidate_is_eligible = bool(
                not candidate.empty
                and min_dte <= float(candidate.iloc[0]["dte"]) <= max_dte
            )
            if not force_roll and not current.empty and not candidate_is_eligible:
                # Once a roll has moved the position away from a near-expiry
                # contract, do not let the daily front-month signal pull it
                # back into the forbidden DTE region on the next day.
                chosen = current.iloc[0]
                reason = "hold_min_dte"
            elif candidate_is_eligible:
                candidate_row = candidate.iloc[0]
                if not current.empty and str(candidate_contract) != held_contract:
                    current_score = pd.to_numeric(current.iloc[0][score_column], errors="coerce")
                    candidate_score = pd.to_numeric(candidate_row[score_column], errors="coerce")
                    if (
                        pd.notna(current_score)
                        and pd.notna(candidate_score)
                        and candidate_score < current_score + min_score_improvement
                    ):
                        chosen = current.iloc[0]
                        reason = "hysteresis"
                    else:
                        chosen = candidate_row
                else:
                    chosen = candidate_row
            elif not eligible.empty:
                chosen = eligible.sort_values(
                    [score_column, "expiry_date", "contract"],
                    ascending=[False, True, True],
                ).iloc[0]
                reason = "unavailable"

        if chosen is None:
            # No tradable contract is available.  Keep the target row as a
            # close/flat instruction so the engine can liquidate safely.
            output = target.to_dict()
            output["roll_reason"] = "no_eligible_contract"
            output["contract"] = None
            rows.append(output)
            held_contract = None
            continue

        output = chosen.to_dict()
        output["roll_reason"] = reason
        rows.append(output)
        held_contract = str(chosen["contract"])

    return pd.DataFrame(rows).reset_index(drop=True)
