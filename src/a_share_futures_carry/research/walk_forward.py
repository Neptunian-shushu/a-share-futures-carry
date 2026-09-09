"""Leakage-aware walk-forward window construction."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardWindow:
    """One expanding-or-rolling train/test split expressed by session dates."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_walk_forward_windows(
    dates: pd.Series | pd.Index,
    train_sessions: int = 252,
    test_sessions: int = 63,
    step_sessions: int | None = None,
) -> list[WalkForwardWindow]:
    """Create non-overlapping test windows from an ordered session calendar."""
    if train_sessions < 2 or test_sessions < 1:
        raise ValueError("train_sessions must be >= 2 and test_sessions must be >= 1")
    step = step_sessions or test_sessions
    if step < 1:
        raise ValueError("step_sessions must be >= 1")
    calendar = pd.DatetimeIndex(pd.to_datetime(dates)).drop_duplicates().sort_values()
    windows: list[WalkForwardWindow] = []
    start = 0
    while start + train_sessions + test_sessions <= len(calendar):
        train = calendar[start : start + train_sessions]
        test = calendar[start + train_sessions : start + train_sessions + test_sessions]
        windows.append(
            WalkForwardWindow(
                train_start=train[0],
                train_end=train[-1],
                test_start=test[0],
                test_end=test[-1],
            )
        )
        start += step
    return windows
