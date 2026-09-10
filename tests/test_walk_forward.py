import pandas as pd

from a_share_futures_carry.research.walk_forward import make_walk_forward_windows


def test_walk_forward_windows_are_ordered_and_non_overlapping():
    dates = pd.bdate_range("2026-01-02", periods=10)
    windows = make_walk_forward_windows(dates, train_sessions=4, test_sessions=2, step_sessions=2)
    assert len(windows) == 3
    assert windows[0].train_end < windows[0].test_start
    assert windows[0].test_end < windows[1].test_start


def test_walk_forward_can_include_a_validation_segment():
    dates = pd.bdate_range("2026-01-02", periods=12)
    windows = make_walk_forward_windows(
        dates, train_sessions=4, validation_sessions=2, test_sessions=2, step_sessions=2
    )
    assert len(windows) == 3
    assert windows[0].train_end < windows[0].validation_start
    assert windows[0].validation_end < windows[0].test_start
    assert windows[0].test_end < windows[1].test_start
