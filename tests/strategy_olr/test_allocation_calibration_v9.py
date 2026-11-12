"""Synthetic causal calibration contracts; never reads production outcomes."""
from datetime import date,timedelta
import importlib.util
from pathlib import Path
import copy
import math
import pytest

path=Path(__file__).resolve().parents[2]/'scripts/olr_allocation_calibration_v9.py'
spec=importlib.util.spec_from_file_location('allocation_calibration_tests',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def sample():
    rows=[]
    for lo,hi in m.BLOCKS:
        for i in range(12):
            day=str(date.fromisoformat(lo)+timedelta(days=i+1))
            for rank in (1,6):
                for scenario in ('baseline','combined'):
                    for budget in (1_000_000,20_000_000):
                        rows.append(dict(day=day,symbol=str(rank),rank=rank,scenario=scenario,budget=budget,
                            status='matured',net_pnl=budget*(.02 if rank==1 else .005),
                            available_for_training_after=day+'T23:59:59+09:00'))
    return rows


def test_weighted_ridge_has_hand_calculated_shrinkage():
    rows=[dict(day=str(i),symbol=str(x),x=x,y=100 if x else -100) for i in range(12) for x in (0.,1.)]
    fitted=m.fit(rows)
    assert fitted['slope']==40 and fitted['intercept']==-20


def test_date_weights_do_not_reward_duplicate_same_date_observations():
    rows=m.paired_records(sample());a=m.fit(rows)
    first=rows[0]['day'];b=m.fit(rows+[dict(r,symbol=r['symbol']+'copy') for r in rows if r['day']==first])
    assert math.isclose(a['slope'],b['slope']) and math.isclose(a['intercept'],b['intercept'])


def test_future_prices_and_labels_cannot_change_earlier_predictions():
    rows=sample();a=m.analyze(rows)
    changed=copy.deepcopy(rows)
    for r in changed:
        if r['day']>='2025-11-01':r['net_pnl']*=100
    b=m.analyze(changed)
    assert a['folds'][0]['model']==b['folds'][0]['model']
    assert [p['model'] for p in a['folds'][0]['predictions']]==[p['model'] for p in b['folds'][0]['predictions']]


def test_unmatured_training_label_is_not_available_at_fit_cutoff():
    rows=sample()
    for r in rows:
        if r['day']=='2025-08-08':r['available_for_training_after']='2025-11-04T23:59:59+09:00'
    fitted=m.analyze(rows)['folds'][0]['model']
    assert fitted['training_dates']==11
    assert all(day!='2025-08-08' for day,_ in fitted['training_keys'])


def test_missing_payoff_is_not_zero_and_blocks_coverage_when_material():
    rows=sample()
    for r in rows:
        if '2025-11-02'<=r['day']<='2025-11-05' and r['budget']==1_000_000:
            r.update(status='unfilled',net_pnl=None,available_for_training_after=None)
    result=m.analyze(rows)
    assert result['folds'][0]['test_dates']==8
    assert not result['folds'][0]['gates']['paired_support']
    assert not result['calibrated_rank_allocation_design_eligible']


def test_one_failed_block_cannot_be_overridden_by_other_block():
    rows=sample()
    for r in rows:
        if r['day'].startswith('2026-01') and r['budget']==20_000_000:r['net_pnl']=-1e6
    result=m.analyze(rows)
    assert not result['folds'][1]['gates']['incremental_net_value_positive']
    assert not result['calibrated_rank_allocation_design_eligible']
    assert not result['promotion'] and not result['portfolio_experiment_eligible']


def test_duplicate_or_incomplete_registered_lanes_fail_closed():
    rows=sample()
    with pytest.raises(ValueError):m.paired_records(rows+[rows[0]])
    with pytest.raises(ValueError):m.paired_records(rows[1:])
