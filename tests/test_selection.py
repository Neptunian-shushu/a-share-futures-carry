import pandas as pd

from a_share_futures_carry.strategy.selection import (
    apply_roll_policy,
    select_front_by_score,
    select_max_carry,
    select_nth_expiry,
)


def sample_contracts():
    return pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-05"] * 4),
            "family": ["IC", "IC", "IM", "IM"],
            "contract": ["IC1", "IC2", "IM1", "IM2"],
            "expiry_date": pd.to_datetime(["2026-01-20", "2026-02-20", "2026-01-20", "2026-02-20"]),
            "dte": [15, 46, 15, 46],
            "carry_ann": [0.04, 0.05, 0.06, 0.08],
        }
    )


def test_select_max_carry_across_families():
    selected = select_max_carry(sample_contracts(), ("IC", "IM"), 5, 120)
    assert selected.iloc[0]["contract"] == "IM2"


def test_select_second_expiry_within_family():
    selected = select_nth_expiry(sample_contracts(), 2, ("IC",))
    assert selected.iloc[0]["contract"] == "IC2"


def test_select_front_by_score_only_compares_front_contracts():
    data = sample_contracts()
    data["selection_score"] = [0.10, 0.90, 0.20, 0.30]
    selected = select_front_by_score(data, ("IC", "IM"), "selection_score")
    assert selected["contract"].tolist() == ["IM1"]


def test_roll_policy_can_keep_expiry_roll_on_nearest_contract():
    data = sample_contracts()
    next_day = data.copy()
    next_day["trade_date"] = pd.Timestamp("2026-01-06")
    next_day["dte"] = next_day["dte"] - 1
    data = pd.concat([data, next_day], ignore_index=True)
    data["signal_carry"] = data["carry_ann"]
    targets = data[data["contract"] == "IC1"].copy()
    rolled = apply_roll_policy(
        targets,
        data,
        roll_before_expiry_days=20,
        min_dte=5,
        max_dte=120,
        roll_to_nearest_expiry=True,
    )
    assert rolled["contract"].tolist() == ["IC1", "IC2"]


def test_roll_policy_does_not_pull_position_back_to_near_expiry_contract():
    data = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-14", "2026-01-15", "2026-01-16"] * 2),
            "family": ["IC"] * 6,
            "contract": ["IC1", "IC1", "IC1", "IC2", "IC2", "IC2"],
            "expiry_date": pd.to_datetime(["2026-01-17"] * 3 + ["2026-02-20"] * 3),
            "dte": [3, 2, 1, 37, 36, 35],
            "carry_ann": [0.1, 1.0, 2.0, 0.2, 0.2, 0.2],
            "signal_carry": [0.1, 1.0, 2.0, 0.2, 0.2, 0.2],
        }
    )
    targets = data[data["contract"] == "IC1"].copy()
    rolled = apply_roll_policy(targets, data, roll_before_expiry_days=3, min_dte=5, max_dte=120)
    assert rolled["contract"].tolist() == ["IC2", "IC2", "IC2"]
    assert rolled.loc[0, "roll_reason"] == "unavailable"


def test_roll_policy_hysteresis_holds_when_new_contract_is_not_better_enough():
    data = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-05"] * 2 + ["2026-01-06"] * 2),
            "family": ["IC"] * 4,
            "contract": ["IC1", "IC2"] * 2,
            "expiry_date": pd.to_datetime(["2026-03-20", "2026-04-17"] * 2),
            "dte": [74, 102, 73, 101],
            "signal_carry": [0.05, 0.0502, 0.05, 0.0502],
        }
    )
    targets = data[data["contract"] == "IC1"].copy()
    targets.loc[targets["trade_date"] == pd.Timestamp("2026-01-06"), "contract"] = "IC2"
    rolled = apply_roll_policy(
        targets,
        data,
        min_dte=5,
        max_dte=180,
        min_score_improvement=0.001,
    )
    assert rolled["contract"].tolist() == ["IC1", "IC1"]
    assert rolled.loc[1, "roll_reason"] == "hysteresis"


def test_roll_policy_keeps_family_when_market_contains_other_families():
    data = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-05"] * 3),
            "family": ["IF", "IF", "IM"],
            "contract": ["IF1", "IF2", "IM1"],
            "expiry_date": pd.to_datetime(["2026-01-06", "2026-02-20", "2026-02-20"]),
            "dte": [1, 46, 46],
            "signal_carry": [0.1, 0.2, 1.0],
        }
    )
    rolled = apply_roll_policy(
        data[data["contract"] == "IF1"], data,
        roll_before_expiry_days=3, min_dte=5, max_dte=120,
    )
    assert rolled.loc[0, "contract"] == "IF2"
    assert rolled.loc[0, "family"] == "IF"
