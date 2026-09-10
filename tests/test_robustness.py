import pandas as pd

from a_share_futures_carry.metrics.robustness import block_bootstrap_summary


def test_block_bootstrap_is_reproducible():
    returns = pd.Series([0.01, -0.005, 0.002, 0.0] * 10)
    left = block_bootstrap_summary(returns, block_size=4, n_bootstrap=20, seed=7)
    right = block_bootstrap_summary(returns, block_size=4, n_bootstrap=20, seed=7)
    assert left == right
    assert 0 <= left["bootstrap_sharpe_positive_prob"] <= 1
