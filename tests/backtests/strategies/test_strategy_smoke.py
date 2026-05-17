from __future__ import annotations

import pytest

from backtests.strategies.kmp.runner import run_kmp_backtest
from backtests.strategies.kpr.runner import run_kpr_backtest
from backtests.strategies.nulrimok.runner import run_nulrimok_backtest


@pytest.mark.parametrize(
    "runner",
    [run_kmp_backtest, run_kpr_backtest, run_nulrimok_backtest],
)
def test_synthetic_backtest_runs_with_trades(runner):
    result = runner({"capability_level": "synthetic"}, {})
    assert result.metrics["total_trades"] >= 1
    assert result.metrics["same_bar_fill_count"] == 0
    assert result.source_fingerprint


def test_official_kmp_requires_tick_feature_bundle():
    with pytest.raises(ValueError, match="tick"):
        run_kmp_backtest({"capability_level": "official", "available_features": ["ohlcv"]}, {})


@pytest.mark.parametrize(
    "runner,config",
    [
        (run_kmp_backtest, {"capability_level": "official", "available_features": ["tick", "bid_ask", "spread", "vi", "tick_imbalance"]}),
        (run_kpr_backtest, {"capability_level": "official", "available_features": ["investor", "program", "micro_pressure"]}),
        (run_nulrimok_backtest, {"capability_level": "official", "available_features": ["historical_lrs", "dse_artifacts", "30m_ohlcv"]}),
    ],
)
def test_non_synthetic_backtests_require_explicit_replay_bundle(runner, config):
    with pytest.raises(ValueError, match="replay_bundle"):
        runner(config, {})
