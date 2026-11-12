from backtests.strategies.kalcb.calibrated_recovery_phase import research_score


def test_soft_drawdown_preferences_do_not_discard_joint_improvements():
    base=dict(baseline_return=45,combined_return=26,baseline_dd=11.1,combined_dd=12.8,
        baseline_trades=296,combined_trades=299,baseline_active=93,combined_active=93)
    improved={**base,'baseline_return':50,'combined_return':30,'baseline_dd':10,'combined_dd':11}
    assert research_score(improved)>research_score(base)
    assert improved['baseline_dd']>8 and improved['combined_dd']>10
