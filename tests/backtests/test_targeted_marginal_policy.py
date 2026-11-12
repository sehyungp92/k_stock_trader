from copy import deepcopy
import pytest
from backtests.strategies.portfolio_synergy.targeted_policy import marginal_decision


def inputs():
    current={'opportunity_id':'2026-02-06/005930/q1','valid_from':'2026-02-04',
        'baseline_value_bps':3,'combined_value_bps':1,'failure_probability':.0001,'raw_score_support':'nonpositive'}
    smaller={**current,'opportunity_id':'2026-02-06/005930/q0.5','baseline_value_bps':2,'combined_value_bps':2}
    calibration={'valid_from':'2026-02-04','valid_through':'2026-03-06','mode':'support_offset',
        'latest_label_available_at':'2026-02-02T15:35:00+09:00','observed_through':'2026-02-02',
        'cohorts':{f'{s}/q{q}/nonpositive':{'stock_days':20,'sessions':8} for s in ('baseline','combined') for q in ('0.5','1')}}
    return current,smaller,calibration


def test_requires_supported_paired_marginal_values_instead_of_absolute_score():
    current,smaller,cal=inputs()
    assert marginal_decision(current,smaller,cal,'2026-02-06')[0]
    current['combined_value_bps']=3
    assert not marginal_decision(current,smaller,cal,'2026-02-06')[0]


def test_uses_raw_support_even_when_calibration_changes_sign():
    current,smaller,cal=inputs()
    assert marginal_decision(current,smaller,cal,'2026-02-06')[0]
    current['raw_score_support']='positive'
    assert not marginal_decision(current,smaller,cal,'2026-02-06')[0]


@pytest.mark.parametrize('failure',['missing','sparse','unsafe','negative','mixed','future','expired','nan'])
def test_rejects_unsupported_or_noncausal_comparisons(failure):
    current,smaller,cal=inputs()
    if failure=='missing': smaller=None
    if failure=='sparse': cal['cohorts']['baseline/q1/nonpositive']['sessions']=7
    if failure=='unsafe': current['failure_probability']=.001
    if failure=='negative': smaller['combined_value_bps']=0
    if failure=='mixed': smaller['valid_from']='2026-01-07'
    if failure=='future': cal['latest_label_available_at']='2026-02-04T00:00:00+09:00'
    if failure=='expired': cal['valid_through']='2026-02-05'
    if failure=='nan': current['baseline_value_bps']=float('nan')
    if failure in ('mixed','future','expired','nan'):
        with pytest.raises(ValueError): marginal_decision(current,smaller,cal,'2026-02-06')
    else: assert not marginal_decision(current,smaller,cal,'2026-02-06')[0]
